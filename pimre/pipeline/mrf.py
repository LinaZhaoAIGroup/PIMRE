"""MRF band reconstruction pipeline with BSFI offset optimization.

Workflow:
  1. Load preprocessed experimental data and DFT band map
  2. Find HSPs, compute affine transform T (DFT → exp)
  3. BSFI offset search (shared global offset, or per-band)
  4. Final MRF reconstruction with symmetrization
  5. Generate path plots and save results
"""

import json
import math
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from pimre.config import crystallographic_data, load_config
from pimre.dft.reader import load_band_map_any
from pimre.kpath.path import bandpath_map as bpm
from pimre.kpath.path import points2path
from pimre.kpath.symmetry import (
    dft_KM,
    find_hsps_robust,
    select_hsps_by_coverage,
)
from pimre.mrf.evaluation import (
    compute_affine_transform,
    compute_bsfi_2d,
    map_dft_bands,
    path_ridge_score,
)
from pimre.mrf.model import MrfRec
from pimre.mrf.symmetry import sym_band
from pimre.utils.io import loadHDF

BAND_COLORS = np.array([
    "#FF6B6B", "#B39DDB", "#DA70D6", "#FF4D4D", "#8A2BE2",
    "#4ECDC4", "#FFE66D", "#FF8C42", "#95E1D3", "#F38181",
])


def resolve_band_indices(bands_cfg, n_conduction, n_stacked, n_dropped=0):
    """Resolve per-band config entries into stacked-array band indices.

    The stacked band array (see ``load_band_map_h5``) is ordered by
    descending band energy *after* dropping the ``drop_top_bands`` highest
    conduction bands, so a plain ``index`` entry depends on the drop count
    matching the DFT band count.  ``from_vbm`` entries are relative to the
    valence-band maximum instead: ``from_vbm: 0`` is the VBM itself, 1 the
    next band below, and so on — independent of the drop count.

    Parameters
    ----------
    bands_cfg : list of dict
        ``mrf.bands`` config entries, each with ``eta`` and either
        ``index`` (position in the stacked array) or ``from_vbm`` (bands
        below the VBM).
    n_conduction : int
        Number of conduction bands in the band map (before dropping).
    n_stacked : int
        Total number of bands after dropping.
    n_dropped : int
        Number of top bands that were dropped.

    Returns
    -------
    band_idx : list of int
    """
    vbm_pos = n_conduction - n_dropped
    band_idx = []
    for b in bands_cfg:
        if "from_vbm" in b:
            if vbm_pos < 0:
                raise ValueError(
                    f"drop_top_bands={n_dropped} exceeds the number of "
                    f"conduction bands ({n_conduction}); the VBM was dropped "
                    "from the stack, so 'from_vbm' band selection is impossible. "
                    "Reduce drop_top_bands or use absolute 'index' entries.")
            idx = vbm_pos + int(b["from_vbm"])
            if not 0 <= idx < n_stacked:
                raise ValueError(
                    f"Band from_vbm={b['from_vbm']} resolves to stacked index "
                    f"{idx}, outside the band map (0..{n_stacked - 1}).")
        elif "index" in b:
            idx = int(b["index"])
            if not 0 <= idx < n_stacked:
                raise ValueError(
                    f"Band index={idx} outside the band map after dropping "
                    f"(0..{n_stacked - 1}).")
        else:
            raise ValueError(
                f"Band entry {b} needs either 'index' or 'from_vbm'.")
        band_idx.append(idx)
    return band_idx


def select_bands_in_window(E_dft, kx_dft, ky_dft, kx, ky, T_inv, band_indices,
                           E_win, margin=5):
    """Report the fraction of each requested DFT band inside the
    experimental energy window.

    DFT band ordering is physical and must not be changed by alignment;
    this function therefore only reports coverage (diagnostics) and never
    remaps the band indices.

    Parameters
    ----------
    E_dft : 3D array
        DFT band data (nbands, kx_dft, ky_dft).
    kx_dft, ky_dft : 1D array
        DFT momentum axes.
    kx, ky : 1D array
        Experimental momentum axes.
    T_inv : 2×2 array
        Inverse affine transform (exp → DFT).
    band_indices : list of int
        Requested DFT band indices.
    E_win : tuple (emin, emax)
        Experimental energy window.
    margin : int
        Unused; kept for API compatibility.

    Returns
    -------
    coverage : dict {band_index: fraction_in_window}
    """
    coverage = {}
    for i in band_indices:
        band = map_dft_bands(E_dft, kx_dft, ky_dft, kx, ky, T_inv, [i])[0]
        coverage[i] = float(np.mean((band >= E_win[0]) & (band <= E_win[1])))
    return coverage


def _smooth_path_segments(y, max_window=31):
    """Savitzky-Golay smoothing that skips NaN segments (bands outside the
    measured energy window are NaN and must not produce spurious flats)."""
    y = np.asarray(y, dtype=float)
    out = np.full_like(y, np.nan)
    valid = ~np.isnan(y)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return out
    for seg in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        if len(seg) < 3:
            out[seg] = y[seg]
            continue
        window = min(max_window, len(seg) - 2)
        if window < 3:
            window = 3
        if window % 2 == 0:
            window -= 1
        polyorder = min(2, max(1, window // 2 - 1))
        out[seg] = savgol_filter(y[seg], window, polyorder)
    return out


def _segment_npoints(a, b, kx_axis, ky_axis, step):
    """Number of path samples for a segment, from its real momentum length.

    The pixel count of a segment depends on the axis resolution (the few
    ky rows of the direct method compress the K-G segment), so the sample
    density is instead chosen per unit momentum distance.

    Parameters
    ----------
    a, b : tuple (ix, iy)
        Endpoint grid indices.
    kx_axis, ky_axis : 1D array
        Momentum axes.
    step : float
        Momentum length per sample (1/Angstrom).

    Returns
    -------
    n : int
        Number of samples along the segment (>= 2).
    """
    mom = float(np.hypot(kx_axis[b[0]] - kx_axis[a[0]],
                         ky_axis[b[1]] - ky_axis[a[1]]))
    return max(2, int(np.ceil(mom / step)))


def draw_path(recon_bcsm, I_t_data, E_arr, choose, savepath, G, M, M1, K, K1,
              kx_axis=None, ky_axis=None, interp_method="linear",
              sample_step=0.005):
    """Draw and save a band path plot (M-G, K-G, or G-M-K-G).

    The horizontal axis is the accumulated momentum distance along the
    path (real-space Gamma-M / M-K / K-G lengths), not the pixel count,
    so that paths dominated by a low-resolution axis (e.g. the few ky
    rows of the direct method) are not visually compressed.  The path is
    sampled by real momentum length (``sample_step`` Angstrom per
    sample) and extracted with ``interp_method`` interpolation.
    """
    def _seg(a, b):
        if kx_axis is not None and ky_axis is not None:
            return _segment_npoints(a, b, kx_axis, ky_axis, sample_step)
        return max(2, int(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)))

    if choose == "M":
        path_pts = np.asarray([M, G, M1])
        segs = [_seg(M, G), _seg(G, M1)]
    elif choose == "K":
        path_pts = np.asarray([K, G, K1])
        segs = [_seg(K, G), _seg(G, K1)]
    else:
        path_pts = np.asarray([G, M, K, G])
        segs = [_seg(G, M), _seg(M, K), _seg(K, G)]

    row_inds, col_inds, path_inds = points2path(path_pts[:, 0], path_pts[:, 1], npoints=segs)
    pathD = bpm(np.transpose(I_t_data, (1, 2, 0)), pathr=row_inds, pathc=col_inds,
                eaxis=2, interp_method=interp_method)
    prec = bpm(np.moveaxis(recon_bcsm, 0, 2), pathr=row_inds, pathc=col_inds,
               eaxis=2, interp_method=interp_method)

    # Real momentum coordinate along the path (accumulated distance).
    if kx_axis is not None and ky_axis is not None:
        row1d = np.ravel(row_inds)
        col1d = np.ravel(col_inds)
        kx_pts = np.interp(row1d, np.arange(kx_axis.size), kx_axis)
        ky_pts = np.interp(col1d, np.arange(ky_axis.size), ky_axis)
        steps = np.sqrt(np.diff(kx_pts) ** 2 + np.diff(ky_pts) ** 2)
        path_mom = np.concatenate(([0.0], np.cumsum(steps)))
        x_new = np.linspace(0.0, path_mom[-1], path_mom.size)
        # Resample intensity and reconstruction onto the uniform momentum axis.
        fD = interp1d(path_mom, pathD.T, axis=0, bounds_error=False, fill_value=0.0)
        pathD = fD(x_new).T
        fP = interp1d(path_mom, prec, axis=1, bounds_error=False, fill_value=np.nan)
        prec = fP(x_new)
        x_total = path_mom[-1]
        x_ticks = path_mom[np.clip(path_inds, 0, path_mom.size - 1)]
    else:
        x_new = np.arange(pathD.shape[1])
        x_total = x_new[-1]
        x_ticks = path_inds

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pathD, cmap="plasma", extent=[0, x_total, E_arr[0], E_arr[-1]],
              aspect="auto", origin="lower")
    n_bands = min(prec.shape[0], len(BAND_COLORS))
    for ib in range(n_bands):
        ax.plot(x_new, _smooth_path_segments(prec[ib, :]),
                zorder=1, lw=2.3, color=BAND_COLORS[ib])
    ax.set(xlim=(0, x_total), ylim=(E_arr[0], E_arr[-1]))
    ax.set_xticks(x_ticks)
    labels = {
        "M": [r"$\overline{\mathrm{M}}$", r"$\overline{\Gamma}$", r"$\overline{\mathrm{M}}$"],
        "K": [r"$\overline{\mathrm{K}}$", r"$\overline{\Gamma}$", r"$\overline{\mathrm{K}}$"],
        "MK": [r"$\overline{\Gamma}$", r"$\overline{\mathrm{M}}$", r"$\overline{\mathrm{K}}$", r"$\overline{\Gamma}$"],
    }
    ax.set_xticklabels(labels[choose], fontsize=15)
    plt.colorbar(ax.images[0], label="Intensity")
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close()


def run_mrf_pipeline(config_path=None, exp_data=None, band_map=None, output_dir=None):
    """Run the full MRF + BSFI reconstruction pipeline.

    Parameters
    ----------
    config_path : str or None
        Path to config YAML. Defaults to configs/pimre_config.yaml.
    exp_data : str or None
        Path to preprocessed experimental HDF5 (overrides config).
    band_map : str or None
        Path to DFT band map .h5 (overrides config).
    output_dir : str or None
        Output directory (defaults to test/).

    Returns
    -------
    recon : ndarray
        Reconstructed bands (n_bands, nkx, nky).
    final_params : list of dict
        Per-band parameters.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "configs", "pimre_config.yaml")
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "test")
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_config(config_path)
    crystal = crystallographic_data(cfg)
    mrf_cfg = cfg["mrf"]
    bands_cfg = mrf_cfg["bands"]
    bsfi_cfg = mrf_cfg["bsfi"]
    N_BANDS = len(bands_cfg)
    NUM_EPOCHS = mrf_cfg.get("num_epochs", 10)
    BSFI_OFFSET_RANGE = bsfi_cfg["offset_range"]
    BSFI_OFFSET_STEP = bsfi_cfg["offset_step"]
    FINE_TUNE_RANGE = bsfi_cfg.get("fine_tune_range", 0.05)
    W_CORR = bsfi_cfg["weights"]["correlation"]
    W_INT = bsfi_cfg["weights"]["intensity"]
    W_SNR = bsfi_cfg["weights"]["snr"]
    W_RIDGE = bsfi_cfg["weights"].get("ridge", 0.0)
    W_PATH_RIDGE = bsfi_cfg["weights"].get("path_ridge", 0.8)
    RIDGE_SIGMA = bsfi_cfg.get("ridge_sigma", 0.1)
    MAX_SHIFT = mrf_cfg.get("max_shift", 10)
    PATH_INTERP_METHOD = mrf_cfg.get("path_interp_method", "cubic")
    PATH_SAMPLE_STEP = mrf_cfg.get("path_sample_step", 0.005)
    ALIGNMENT = mrf_cfg.get("alignment", "hsp")
    OFFSET_MODE = mrf_cfg.get("offset_mode", "per_band")
    OCCUPIED_ONLY = mrf_cfg.get("occupied_only", True)
    smooth_sigma = mrf_cfg["smooth_sigma"]

    if exp_data is None:
        exp_data = cfg["preprocessing"]["output_path"]
    if band_map is None:
        band_map = os.path.join(output_dir, "band_map.h5")

    # ── STEP 1: Load data ──
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)

    data = loadHDF(exp_data)
    E = data["E"][1:-1]
    kx = data["kx"][1:-1]
    ky = data["ky"][1:-1]
    I = data["V"][1:-1, 1:-1, 1:-1]

    # Exact-shape matching avoids the square-grid ambiguity of
    # shape[0]-based heuristics.  The saved layout is (E, kx, ky).
    if I.shape == (E.size, kx.size, ky.size):
        I_t = I
    elif I.shape == (kx.size, ky.size, E.size):
        I_t = np.transpose(I, (2, 0, 1))
    elif I.shape == (kx.size, E.size, ky.size):
        I_t = np.transpose(I, (1, 0, 2))
    else:
        raise ValueError(
            f"Unexpected data layout {I.shape} for axes E={E.size}, "
            f"kx={kx.size}, ky={ky.size} in {exp_data}")

    mrf = MrfRec(E=E, kx=kx, ky=ky, I=np.transpose(I_t, (1, 2, 0)),
                 eta=mrf_cfg.get("eta", 0.12), max_shift=MAX_SHIFT,
                 device=mrf_cfg.get("device", "auto"))
    mrf.smoothenI(sigma=smooth_sigma)
    print(f"  Exp: E={E.shape}, kx={kx.shape}, ky={ky.shape}, I={I.shape}"
          f", device={mrf.device}")

    E_dft, evb, ecb, kx_dft, ky_dft = load_band_map_any(
        band_map, drop_top_bands=cfg.get("dft", {}).get("drop_top_bands"))
    print(f"  DFT: E_dft={E_dft.shape}, kx_dft={kx_dft.shape}, ky_dft={ky_dft.shape}")

    # ── STEP 2: HSPs, affine transform ──
    print("\n" + "=" * 60)
    print("STEP 2: HSPs, affine transform T, DFT mapping")
    print("=" * 60)

    calib = cfg["calibration"].get("hsps", {})
    result = find_hsps_robust(I_t, kx, ky, crystal, E, calibration=calib)
    hsps = result.hsps
    G = hsps["G"]

    # Pick the K/M pair that actually has data coverage (the default K0/M0
    # directions may fall outside the measured window, e.g. for quadrant
    # symmetrized maps).  Only orientation-compatible (K_i, M_i) pairs are
    # considered so that Gamma-M-K-Gamma is a closed right triangle along
    # the BZ edge, matching the DFT-side construction.
    sel = select_hsps_by_coverage(hsps, I_t, kx, ky)
    k_sel, m_sel = sel["K_index"], sel["M_index"]
    M = hsps[f"M{m_sel}"]
    M1 = hsps[f"M{(m_sel + 3) % 6}"]
    K = hsps[f"K{k_sel}"]
    K1 = hsps[f"K{(k_sel + 3) % 6}"]

    # DFT-side HSPs use their own independent calibration (dft_hsps),
    # applied as rotation/scale around the DFT Gamma point.
    dft_calib = cfg["calibration"].get("dft_hsps", {})
    if dft_calib.get("manual", False):
        KP_dft_raw, MP_dft_raw = dft_KM(
            kx_dft, ky_dft,
            rotation_angle=dft_calib.get("rotation_angle", 0.0),
            scale=dft_calib.get("scale", 1.0))
        print(f"  DFT HSPs calibrated: θ={dft_calib.get('rotation_angle', 0.0):.2f}°,"
              f" scale={dft_calib.get('scale', 1.0):.3f}")
    else:
        KP_dft_raw, MP_dft_raw = dft_KM(kx_dft, ky_dft)
    theta_deg = float(result.rotation_angle)
    k_ang = (theta_deg + k_sel * 60) % 360
    m_ang = (theta_deg - 30 + m_sel * 60) % 360
    print(f"  HSPs: G={G}, K{k_ang}°={K}, M{m_ang}°={M} "
          f"(source={result.source}, coverage={sel['coverage']:.2f})")
    print("  K/M coverage scores: "
          + ", ".join(f"{k}={v:.2f}" for k, v in sel["scores"].items()))

    if ALIGNMENT == "gamma":
        # 1:1 mapping — only valid when both momentum axes are already in
        # consistent absolute units (same lattice calibration on both sides).
        T = np.eye(2)
        T_inv = np.eye(2)
        scale = 1.0
        rotation_deg = 0.0
        print("  Alignment: gamma (identity transform, 1:1 momentum axes)")
    else:
        # HSP alignment: the theory momentum scale differs from the
        # experimental one; T stretches/rotates the DFT grid so that its
        # K and M high-symmetry points land on the experimental ones.
        T, T_inv, scale, rotation_deg = compute_affine_transform(
            kx, ky, G, K, M, kx_dft, ky_dft, KP_dft_raw, MP_dft_raw)
        print("  Alignment: hsp (Gamma-K / Gamma-M vectors matched exactly)")
    print(f"  T = [[{T[0,0]:.6f}, {T[0,1]:.6f}], [{T[1,0]:.6f}, {T[1,1]:.6f}]]")
    print(f"  scale_x={np.linalg.norm(T[:, 0]):.4f}, scale_y={np.linalg.norm(T[:, 1]):.4f}, "
          f"rotation={rotation_deg:.2f}°")

    gx = np.argmin(np.abs(kx_dft))
    gy = np.argmin(np.abs(ky_dft))
    K_vec_exp = np.array([kx[K[0]]-kx[G[0]], ky[K[1]]-ky[G[1]]])
    M_vec_exp = np.array([kx[M[0]]-kx[G[0]], ky[M[1]]-ky[G[1]]])
    K_vec_dft = np.array([kx_dft[KP_dft_raw[0]]-kx_dft[gx], ky_dft[KP_dft_raw[1]]-ky_dft[gy]])
    M_vec_dft = np.array([kx_dft[MP_dft_raw[0]]-kx_dft[gx], ky_dft[MP_dft_raw[1]]-ky_dft[gy]])
    ratio_exp = np.linalg.norm(K_vec_exp) / np.linalg.norm(M_vec_exp)
    ratio_dft = np.linalg.norm(K_vec_dft) / np.linalg.norm(M_vec_dft)
    print(f"  |K|/|M|: exp={ratio_exp:.4f}, DFT={ratio_dft:.4f} (ideal=1.155)")

    E_dft_orig = E_dft.copy()
    n_dropped = cfg.get("dft", {}).get("drop_top_bands") or 0
    band_idx = resolve_band_indices(bands_cfg, len(ecb), len(E_dft),
                                    n_dropped=n_dropped)

    E_win = (E.min(), E.max())
    coverage = select_bands_in_window(E_dft_orig, kx_dft, ky_dft, kx, ky,
                                      T_inv, band_idx, E_win)
    print(f"  Band coverage inside experimental window ({E_win[0]:.2f}..{E_win[1]:.2f} eV):")
    for t, cov in coverage.items():
        print(f"    band {t} (E_dft[{t}]): {100*cov:.1f}%  (band order kept)")
    low = [t for t, cov in coverage.items() if cov < 0.90]
    if low:
        print(f"  Warning: bands {low} have <90% coverage inside the window; "
              "their reconstruction is weakly constrained by the data.")

    print("  Mapping DFT bands via T_inv ...")
    dft_bands = map_dft_bands(E_dft_orig, kx_dft, ky_dft, kx, ky, T_inv, band_idx)
    print(f"  {len(dft_bands)} DFT bands mapped")

    # ── STEP 3: BSFI offset search (shared global offset) ──
    print("\n" + "=" * 60)
    print("STEP 3: BSFI shared-offset search — all bands move together")
    print("=" * 60)

    n_offsets = int(2 * BSFI_OFFSET_RANGE / BSFI_OFFSET_STEP) + 1
    offsets = np.linspace(-BSFI_OFFSET_RANGE, BSFI_OFFSET_RANGE, n_offsets)
    print(f"  offset: {offsets[0]:+.2f}→{offsets[-1]:+.2f} eV ({len(offsets)} steps)")
    print(f"  BSFI weights: corr={W_CORR}, intensity={W_INT}, snr={W_SNR}, "
          f"ridge={W_RIDGE}, path_ridge={W_PATH_RIDGE}")

    # Downsampling stride for the 2D BSFI evaluation; adapt to short
    # momentum axes (e.g. the 36-pixel ky axis of the direct method).
    bsfi_stride = max(1, min(4, I_t.shape[2] // 16))

    def _bsfi(E0):
        return compute_bsfi_2d(E0, I_t, E, w_corr=W_CORR, w_int=W_INT,
                               w_snr=W_SNR, w_ridge=W_RIDGE,
                               ridge_sigma=RIDGE_SIGMA, stride=bsfi_stride)

    # Band-path map along G-M-K-G for path-based ridge evaluation.
    # Sampled by real momentum length and interpolated like the output
    # path plots so the ridge score uses the same representation.
    path_pts = np.asarray([G, M, K, G])
    row_inds, col_inds, path_inds = points2path(
        path_pts[:, 0], path_pts[:, 1],
        npoints=[_segment_npoints(G, M, kx, ky, PATH_SAMPLE_STEP),
                 _segment_npoints(M, K, kx, ky, PATH_SAMPLE_STEP),
                 _segment_npoints(K, G, kx, ky, PATH_SAMPLE_STEP)])
    pathD = bpm(np.moveaxis(I_t, 0, 2), pathr=row_inds, pathc=col_inds,
                eaxis=2, interp_method=PATH_INTERP_METHOD)

    def _path_ridge(E0):
        prec = bpm(np.moveaxis(E0[np.newaxis], 0, 2), pathr=row_inds,
                   pathc=col_inds, eaxis=2,
                   interp_method=PATH_INTERP_METHOD)[0]
        return path_ridge_score(pathD, prec, E, sigma=RIDGE_SIGMA)

    def _offset_score(E0):
        """Combined score for offset selection: weighted path-ridge + 2D BSFI."""
        total_w = W_PATH_RIDGE + 1.0
        return (W_PATH_RIDGE * _path_ridge(E0) + _bsfi(E0)) / total_w

    # Offset search.  The grid scan fills per-band score curves for every
    # band; how the final per-band offsets are chosen depends on the mode:
    #   per_band: each band takes its own score maximum.  For metallic
    #             systems (bands crossing E_F, very different dispersions)
    #             a single shared shift cannot align all bands at once.
    #   shared:   all bands take the global mean-score optimum, which
    #             prevents one band from locking the others into its own
    #             (possibly misidentified) position.
    print(f"  Offset search (mode={OFFSET_MODE}), {len(offsets)} offsets ...")
    scores_shared = np.zeros(len(offsets))
    per_band_scores = np.zeros((N_BANDS, len(offsets)))
    best_shared_score = -np.inf
    best_shared_off = offsets[0]

    def occupied_E0(raw_band, off, min_frac=0.02):
        """Shifted band restricted to measured (ARPES-visible) states.

        With ``occupied_only`` (default): ARPES only measures states at or
        below E_F.  In the additive convention E0 = E_dft + offset, a mapped
        point corresponds to an occupied state only when E0 >= 0
        (non-negative binding energy) AND E_dft <= 0.  Everything else —
        notably the empty-state segments of bands crossing E_F — is masked
        to NaN so that the offset search cannot align empty states to the
        measured intensity.

        Without it (reference-style full-band alignment): only pixels
        without DFT coverage are masked; empty-state segments stay in and
        the band is aligned as a whole.  Returns (masked E0, valid fraction).
        """
        raw = raw_band + off
        if not OCCUPIED_ONLY:
            finite = np.isfinite(raw)
            return np.where(finite, raw, np.nan), float(np.mean(finite))
        occ = np.isfinite(raw) & (raw >= 0) & (raw_band <= 0)
        frac = float(np.mean(occ))
        if frac < min_frac:
            return np.full(raw_band.shape, np.nan), frac
        return np.where(occ, raw, np.nan), frac

    for i_off, off in enumerate(offsets):
        total_bsfi = 0.0
        for ind_band in range(N_BANDS):
            E0, _ = occupied_E0(dft_bands[ind_band], off)
            s = _offset_score(E0)
            per_band_scores[ind_band, i_off] = s
            total_bsfi += s
        scores_shared[i_off] = total_bsfi / N_BANDS
        if scores_shared[i_off] > best_shared_score:
            best_shared_score = scores_shared[i_off]
            best_shared_off = off
    print(f"  Best shared offset = {best_shared_off:+.4f} eV, mean score = {best_shared_score:.4f}")

    if OFFSET_MODE == "per_band":
        best_offsets = np.array([offsets[int(np.argmax(per_band_scores[k]))]
                                 for k in range(N_BANDS)])
        best_score = float(np.mean([per_band_scores[k, int(np.argmax(per_band_scores[k]))]
                                    for k in range(N_BANDS)]))
        print("  Per-band optimal offsets:")
        for k in range(N_BANDS):
            if per_band_scores[k].max() <= 0:
                print(f"    band {k}: NO occupied alignment found (the band's occupied "
                      "part never enters the window at any offset); "
                      "its reconstruction will be empty.")
            else:
                print(f"    band {k}: {best_offsets[k]:+.4f} eV "
                      f"(score={per_band_scores[k, int(np.argmax(per_band_scores[k]))]:.4f})")
    else:
        best_offsets = np.full(N_BANDS, best_shared_off)
        best_score = best_shared_score
        print("  Per-band scores at optimal shared offset:")
        for k in range(N_BANDS):
            print(f"    band {k}: {per_band_scores[k].max():.4f}")

    if OFFSET_MODE == "hierarchical":
        # Stage 2: per-band fine-tune within ±fine_tune_range of the shared
        # optimum (reference behaviour).  Constraining each band to stay near
        # the shared offset prevents band-order crossing from per-band
        # argmax over the full coarse grid.
        fine_step = BSFI_OFFSET_STEP / 2.0
        n_fine = int(2 * FINE_TUNE_RANGE / fine_step) + 1
        fine_offsets = np.linspace(best_shared_off - FINE_TUNE_RANGE,
                                   best_shared_off + FINE_TUNE_RANGE, n_fine)
        print(f"  Stage 2: per-band fine-tune within ±{FINE_TUNE_RANGE:.2f} eV "
              f"of shared best ({n_fine} offsets) ...")
        best_offsets = np.zeros(N_BANDS)
        fine_best = np.zeros(N_BANDS)
        for k in range(N_BANDS):
            fine_scores = np.array([
                _offset_score(occupied_E0(dft_bands[k], off)[0])
                for off in fine_offsets])
            imax = int(np.argmax(fine_scores))
            if fine_scores[imax] <= 0:
                best_offsets[k] = best_shared_off
                print(f"    band {k}: no measured alignment in the fine "
                      "window; keeping shared offset")
                continue
            best_offsets[k] = float(fine_offsets[imax])
            fine_best[k] = fine_scores[imax]
            print(f"    band {k}: {best_offsets[k]:+.4f} eV "
                  f"(Δ={best_offsets[k] - best_shared_off:+.4f}, "
                  f"score={fine_scores[imax]:.4f})")
        if np.any(fine_best > 0):
            best_score = float(np.mean(fine_best[fine_best > 0]))
    print(f"  Mean score = {best_score:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(offsets, scores_shared, "b-", linewidth=2)
    ax1.axvline(best_shared_off, color="red", linestyle="--", label=f"best={best_shared_off:+.4f}")
    ax1.set(xlabel="Energy Offset (eV)", ylabel="Combined BSFI", title="Shared offset search")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    for ind_band in range(N_BANDS):
        ax2.plot(offsets, per_band_scores[ind_band], color=BAND_COLORS[ind_band],
                 linewidth=1.5, alpha=0.8, label=f"B{ind_band}")
    ax2.axvline(best_shared_off, color="gray", linestyle="--", linewidth=1.5, label="shared")
    ax2.set_xlim(-BSFI_OFFSET_RANGE - 0.05, BSFI_OFFSET_RANGE + 0.05)
    ax2.set(xlabel="Energy Offset (eV)", title="Per-band scores (shared offset)")
    ax2.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "bsfi_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ── STEP 4: Final MRF reconstruction ──
    print("\n" + "=" * 60)
    print(f"STEP 4: Final MRF reconstruction ({NUM_EPOCHS} epochs)")
    print("=" * 60)

    recon = -np.ones((N_BANDS, len(mrf.kx), len(mrf.ky)))
    final_params = []

    for ind_band in range(N_BANDS):
        eta = bands_cfg[ind_band]["eta"]
        final_offset = float(best_offsets[ind_band])
        mrf.eta = eta
        raw_band = dft_bands[ind_band]
        # Occupied-state constraint (config mrf.occupied_only, default on):
        # only E0 >= 0 from E_dft <= 0 pixels enter the reconstruction.
        # With occupied_only: false the full band is used, reference-style;
        # either way, pixels without DFT coverage (NaN) get a neutral
        # mid-window start and are masked out after the fit.
        E0_occ, occ_frac = occupied_E0(raw_band, final_offset)
        invalid_init = ~np.isfinite(E0_occ)
        E0_raw = E0_occ
        # Pixels without DFT coverage (NaN) get a neutral mid-window start;
        # they are unconstrained by DFT and are masked out after the fit.
        E0 = np.where(np.isfinite(E0_raw), E0_raw, 0.5 * (E.min() + E.max()))
        EE, EE0 = np.meshgrid(E, E0)
        mrf.indEb = np.argmin(np.abs(EE - EE0), 1).reshape(E0.shape)
        mrf.indE0 = mrf.indEb.copy()
        mrf.delHist()
        mrf.iter_para(num_epoch=NUM_EPOCHS, updateLogP=True, disable_tqdm=True)
        recon[ind_band] = mrf.getEb()

        sym_band(ind_band, recon, mrf.kx, mrf.ky, mrf.lengthKx, mrf.lengthKy)

        # Mask everything that is not a DFT-covered, measured pixel (with
        # occupied_only: true this also drops empty-state segments).
        recon[ind_band][invalid_init] = np.nan
        bsfi_b = _bsfi(recon[ind_band])

        final_params.append({
            "band": ind_band, "dft_band": int(band_idx[ind_band]),
            "T": [float(T[0,0]), float(T[0,1]), float(T[1,0]), float(T[1,1])],
            "offset": float(final_offset), "eta": float(eta), "bsfi_score": float(bsfi_b),
            "occupied_frac": occ_frac,
        })
        print(f"  Band {ind_band}: offset={final_offset:+.4f} eV, eta={eta}, BSFI={bsfi_b:.4f}, "
              f"occupied {100*occ_frac:.1f}% of grid")

    np.save(os.path.join(output_dir, "recon_bands.npy"), recon)

    # ── STEP 5: Path plots ──
    print("\n" + "=" * 60)
    print("STEP 5: Path plots")
    print("=" * 60)
    draw_path(recon[:], I_t, E, "K", os.path.join(output_dir, "path_KG.png"), G, M, M1, K, K1,
              kx_axis=mrf.kx, ky_axis=mrf.ky,
              interp_method=PATH_INTERP_METHOD, sample_step=PATH_SAMPLE_STEP)
    draw_path(recon[:], I_t, E, "M", os.path.join(output_dir, "path_MG.png"), G, M, M1, K, K1,
              kx_axis=mrf.kx, ky_axis=mrf.ky,
              interp_method=PATH_INTERP_METHOD, sample_step=PATH_SAMPLE_STEP)
    draw_path(recon[:], I_t, E, "MK", os.path.join(output_dir, "path_GMKG.png"), G, M, M1, K, K1,
              kx_axis=mrf.kx, ky_axis=mrf.ky,
              interp_method=PATH_INTERP_METHOD, sample_step=PATH_SAMPLE_STEP)
    print("  Saved path plots")

    # ── Save results ──
    np.savez(os.path.join(output_dir, "bsfi_scores.npz"),
             offsets=offsets, scores_shared=scores_shared,
             best_shared_off=best_shared_off, best_offsets=best_offsets)

    save_data = {
        "crystal_data": crystal,
        "alignment": ALIGNMENT,
        "offset_mode": OFFSET_MODE,
        "occupied_only": OCCUPIED_ONLY,
        "fine_tune_range": FINE_TUNE_RANGE,
        "bsfi_weights": {"correlation": W_CORR, "intensity": W_INT,
                         "snr": W_SNR, "ridge": W_RIDGE,
                         "path_ridge": W_PATH_RIDGE},
        "ridge_sigma": RIDGE_SIGMA,
        "max_shift": MAX_SHIFT,
        "device": str(mrf.device),
        "bsfi_offset_range": [-BSFI_OFFSET_RANGE, BSFI_OFFSET_RANGE],
        "bsfi_offset_step": BSFI_OFFSET_STEP,
        "affine_T": [[float(T[0,0]), float(T[0,1])], [float(T[1,0]), float(T[1,1])]],
        "scale": float(scale),
        "rotation_deg": float(rotation_deg),
        "K_M_ratio_exp": float(ratio_exp), "K_M_ratio_dft": float(ratio_dft),
        "bands": final_params,
        "best_offsets": [float(o) for o in best_offsets],
        "combined_bsfi": float(best_score),
        "shared_offset": float(best_shared_off),
        "shared_bsfi": float(best_shared_score),
    }
    with open(os.path.join(output_dir, "final_parameters.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  T = [[{T[0,0]:.6f}, {T[0,1]:.6f}], [{T[1,0]:.6f}, {T[1,1]:.6f}]]")
    print(f"  isotropic scale={scale:.4f}, rotation={rotation_deg:.2f}°")
    print(f"  Stage 1 shared: {best_shared_off:+.4f} eV, BSFI={best_shared_score:.4f}")
    for ib, p in enumerate(final_params):
        print(f"    Band {ib}: offset={p['offset']:.4f} eV, BSFI={p['bsfi_score']:.4f}")
    print(f"  Mean BSFI = {best_score:.4f}")
    for f in sorted(os.listdir(output_dir)):
        sz = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:40s} {sz:>12,d} bytes")

    return recon, final_params
