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
from scipy.signal import savgol_filter

from pimre.config import crystallographic_data, load_config
from pimre.dft.reader import load_band_map_h5
from pimre.kpath.path import bandpath_map as bpm
from pimre.kpath.path import points2path
from pimre.kpath.symmetry import dft_KM, find_hsps_robust
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


def draw_path(recon_bcsm, I_t_data, E_arr, choose, savepath, G, M, M1, K, K1):
    """Draw and save a band path plot (M-G, K-G, or G-M-K-G)."""
    if choose == "M":
        path_pts = np.asarray([M, G, M1])
        n1 = int(math.sqrt((M[0]-G[0])**2 + (M[1]-G[1])**2))
        segs = [n1, n1]
    elif choose == "K":
        path_pts = np.asarray([K, G, K1])
        n1 = int(math.sqrt((K[0]-G[0])**2 + (K[1]-G[1])**2))
        segs = [n1, n1]
    else:
        path_pts = np.asarray([G, M, K, G])
        nGM = int(math.sqrt((M[0]-G[0])**2 + (M[1]-G[1])**2))
        nMK = int(math.sqrt((M[0]-K[0])**2 + (M[1]-K[1])**2))
        nKG = int(math.sqrt((K[0]-G[0])**2 + (K[1]-G[1])**2))
        segs = [nGM, nMK, nKG]

    row_inds, col_inds, path_inds = points2path(path_pts[:, 0], path_pts[:, 1], npoints=segs)
    pathD = bpm(np.transpose(I_t_data, (1, 2, 0)), pathr=row_inds, pathc=col_inds, eaxis=2)
    prec = bpm(np.moveaxis(recon_bcsm, 0, 2), pathr=row_inds, pathc=col_inds, eaxis=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pathD, cmap="plasma", extent=[0, len(row_inds), E_arr[0], E_arr[-1]],
              aspect="auto", origin="lower")
    n_bands = min(prec.shape[0], len(BAND_COLORS))
    for ib in range(n_bands):
        ax.plot(_smooth_path_segments(prec[ib, :]),
                zorder=1, lw=2.3, color=BAND_COLORS[ib])
    ax.set(xlim=(0, len(row_inds)), ylim=(E_arr[0], E_arr[-1]))
    ax.set_xticks(path_inds)
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
    W_CORR = bsfi_cfg["weights"]["correlation"]
    W_INT = bsfi_cfg["weights"]["intensity"]
    W_SNR = bsfi_cfg["weights"]["snr"]
    W_RIDGE = bsfi_cfg["weights"].get("ridge", 0.0)
    W_PATH_RIDGE = bsfi_cfg["weights"].get("path_ridge", 0.8)
    RIDGE_SIGMA = bsfi_cfg.get("ridge_sigma", 0.1)
    MAX_SHIFT = mrf_cfg.get("max_shift", 10)
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

    if I.shape[0] == E.shape[0]:
        I_t = I
    elif I.shape[0] == kx.shape[0]:
        I_t = np.transpose(I, (2, 0, 1))
    else:
        I_t = np.transpose(I, (2, 0, 1))

    mrf = MrfRec(E=E, kx=kx, ky=ky, I=np.transpose(I_t, (1, 2, 0)), eta=0.12,
                 max_shift=MAX_SHIFT)
    mrf.smoothenI(sigma=smooth_sigma)
    print(f"  Exp: E={E.shape}, kx={kx.shape}, ky={ky.shape}, I={I.shape}")

    E_dft, evb, ecb, kx_dft, ky_dft = load_band_map_h5(
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
    M = hsps["M0"]
    M1 = hsps["M3"]
    K = hsps["K0"]
    K1 = hsps["K3"]
    KP_dft_raw, MP_dft_raw = dft_KM(kx_dft, ky_dft)
    print(f"  HSPs: G={G}, M={M}, K={K} (source={result.source})")

    T, T_inv, scale, rotation_deg = compute_affine_transform(
        kx, ky, G, K, M, kx_dft, ky_dft, KP_dft_raw, MP_dft_raw)
    print(f"  T = [[{T[0,0]:.6f}, {T[0,1]:.6f}], [{T[1,0]:.6f}, {T[1,1]:.6f}]]")
    print(f"  isotropic scale={scale:.4f}, rotation={rotation_deg:.2f}°")

    gx = np.argmin(np.abs(kx_dft)); gy = np.argmin(np.abs(ky_dft))
    K_vec_exp = np.array([kx[K[0]]-kx[G[0]], ky[K[1]]-ky[G[1]]])
    M_vec_exp = np.array([kx[M[0]]-kx[G[0]], ky[M[1]]-ky[G[1]]])
    K_vec_dft = np.array([kx_dft[KP_dft_raw[0]]-kx_dft[gx], ky_dft[KP_dft_raw[1]]-ky_dft[gy]])
    M_vec_dft = np.array([kx_dft[MP_dft_raw[0]]-kx_dft[gx], ky_dft[MP_dft_raw[1]]-ky_dft[gy]])
    ratio_exp = np.linalg.norm(K_vec_exp) / np.linalg.norm(M_vec_exp)
    ratio_dft = np.linalg.norm(K_vec_dft) / np.linalg.norm(M_vec_dft)
    print(f"  |K|/|M|: exp={ratio_exp:.4f}, DFT={ratio_dft:.4f} (ideal=1.155)")

    E_dft_orig = E_dft.copy()
    band_idx = [b["index"] for b in bands_cfg]

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

    def _bsfi(E0):
        return compute_bsfi_2d(E0, I_t, E, w_corr=W_CORR, w_int=W_INT,
                               w_snr=W_SNR, w_ridge=W_RIDGE,
                               ridge_sigma=RIDGE_SIGMA)

    # Band-path map along G-M-K-G for path-based ridge evaluation
    nGM = int(math.sqrt((M[0]-G[0])**2 + (M[1]-G[1])**2))
    nMK = int(math.sqrt((M[0]-K[0])**2 + (M[1]-K[1])**2))
    nKG = int(math.sqrt((K[0]-G[0])**2 + (K[1]-G[1])**2))
    path_pts = np.asarray([G, M, K, G])
    row_inds, col_inds, path_inds = points2path(path_pts[:, 0], path_pts[:, 1],
                                                npoints=[nGM, nMK, nKG])
    pathD = bpm(np.moveaxis(I_t, 0, 2), pathr=row_inds, pathc=col_inds, eaxis=2)

    def _path_ridge(E0):
        prec = bpm(np.moveaxis(E0[np.newaxis], 0, 2), pathr=row_inds,
                   pathc=col_inds, eaxis=2)[0]
        return path_ridge_score(pathD, prec, E, sigma=RIDGE_SIGMA)

    def _offset_score(E0):
        """Combined score for offset selection: weighted path-ridge + 2D BSFI."""
        total_w = W_PATH_RIDGE + 1.0
        return (W_PATH_RIDGE * _path_ridge(E0) + _bsfi(E0)) / total_w

    # Shared search: every band is shifted by the same absolute offset and
    # the mean combined score over ALL bands decides the optimum.  This
    # prevents a single band from locking the others into its own (possibly
    # misidentified) position.
    print("  Shared offset search (all bands shifted together) ...")
    scores_shared = np.zeros(len(offsets))
    per_band_scores = np.zeros((N_BANDS, len(offsets)))
    best_shared_score = -np.inf; best_shared_off = offsets[0]
    for i_off, off in enumerate(offsets):
        total_bsfi = 0.0
        for ind_band in range(N_BANDS):
            E0 = np.reshape(dft_bands[ind_band] + off, (kx.shape[0], ky.shape[0]))
            s = _offset_score(E0)
            per_band_scores[ind_band, i_off] = s
            total_bsfi += s
        scores_shared[i_off] = total_bsfi / N_BANDS
        if scores_shared[i_off] > best_shared_score:
            best_shared_score = scores_shared[i_off]; best_shared_off = off
    print(f"  Best shared offset = {best_shared_off:+.4f} eV, mean score = {best_shared_score:.4f}")
    print("  Per-band scores at the shared optimum:")
    for ind_band in range(N_BANDS):
        i0 = int(np.argmin(np.abs(offsets - best_shared_off)))
        print(f"    band {ind_band}: score={per_band_scores[ind_band, i0]:.4f}")

    best_offsets = np.full(N_BANDS, best_shared_off)
    best_bsfi_per_band = np.full(N_BANDS, best_shared_score)

    best_score = best_bsfi_per_band.mean()
    print(f"  Mean score = {best_score:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(offsets, scores_shared, "b-", linewidth=2)
    ax1.axvline(best_shared_off, color="red", linestyle="--", label=f"best={best_shared_off:+.4f}")
    ax1.set(xlabel="Energy Offset (eV)", ylabel="Combined BSFI", title="Shared offset search")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
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
        E0 = np.reshape(dft_bands[ind_band] + final_offset, (kx.shape[0], ky.shape[0]))
        EE, EE0 = np.meshgrid(E, E0)
        mrf.indEb = np.argmin(np.abs(EE - EE0), 1).reshape(E0.shape)
        mrf.indE0 = mrf.indEb.copy()
        mrf.delHist()
        mrf.iter_para(num_epoch=NUM_EPOCHS, updateLogP=True, disable_tqdm=True)
        recon[ind_band] = mrf.getEb()
        bsfi_b = _bsfi(recon[ind_band])

        sym_band(ind_band, recon, mrf.kx, mrf.ky, mrf.lengthKx, mrf.lengthKy)

        # Pixels whose band energy lies outside the measured window have no
        # experimental constraint; mark them NaN instead of clamping to the
        # energy-axis edge (which would produce a spurious flat band).
        outside = (E0 > E.max()) | (E0 < E.min())
        recon[ind_band][outside] = np.nan

        final_params.append({
            "band": ind_band, "dft_band": int(band_idx[ind_band]),
            "T": [float(T[0,0]), float(T[0,1]), float(T[1,0]), float(T[1,1])],
            "offset": float(final_offset), "eta": float(eta), "bsfi_score": float(bsfi_b),
            "outside_window_frac": float(np.mean(outside)),
        })
        print(f"  Band {ind_band}: offset={final_offset:+.4f} eV, eta={eta}, BSFI={bsfi_b:.4f}")

    np.save(os.path.join(output_dir, "recon_bands.npy"), recon)

    # ── STEP 5: Path plots ──
    print("\n" + "=" * 60)
    print("STEP 5: Path plots")
    print("=" * 60)
    draw_path(recon[:], I_t, E, "K", os.path.join(output_dir, "path_KG.png"), G, M, M1, K, K1)
    draw_path(recon[:], I_t, E, "M", os.path.join(output_dir, "path_MG.png"), G, M, M1, K, K1)
    draw_path(recon[:], I_t, E, "MK", os.path.join(output_dir, "path_GMKG.png"), G, M, M1, K, K1)
    print("  Saved path plots")

    # ── Save results ──
    np.savez(os.path.join(output_dir, "bsfi_scores.npz"),
             offsets=offsets, scores_shared=scores_shared,
             best_shared_off=best_shared_off, best_offsets=best_offsets)

    save_data = {
        "crystal_data": crystal,
        "bsfi_weights": {"correlation": W_CORR, "intensity": W_INT,
                         "snr": W_SNR, "ridge": W_RIDGE,
                         "path_ridge": W_PATH_RIDGE},
        "ridge_sigma": RIDGE_SIGMA,
        "max_shift": MAX_SHIFT,
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
