# Quadrant Symmetrization Preprocessing Design

Date: 2026-08-08
Status: Approved

## Goal

Add a new experimental-data preprocessing method alongside the existing
KD-tree based interpolation. After Gamma-point localization and
angle-to-momentum conversion, crop the full 1/4 Brillouin zone
(kx >= 0 and ky >= 0 quadrant around Gamma), reconstruct the full BZ by
simple mirror symmetry, optionally flipping (default on, overlapping
intensities averaged). The resulting regular grid feeds the existing
downstream pipeline (HSP alignment + BSFI + MRF) unchanged.

## Configuration

```yaml
preprocessing:
  method: kdtree        # 'kdtree' (existing) | 'quadrant' (new)
  output_grid: 200
  kd_radius: 0.05       # kdtree-only
  stride: 10            # kdtree-only
  n_rotations: 6        # kdtree-only
  quadrant:
    flip_kx: true       # mirror across ky=0 axis (expand to -kx)
    flip_ky: true       # mirror across kx=0 axis (expand to -ky)
```

Defaults added to `pimre/config.py` DEFAULTS; both boolean flips default
to true. The config generator GUI (`scripts/setup_config.py`) gains a
method selector and the two flip checkboxes.

## Algorithm

For each energy layer (E, kx_angle, ky_angle):

1. Input: layer intensity `bands` and momentum coordinates `KX, KY`
   (scattered points after Angle2Mon; Gamma at momentum origin (0,0)).
2. Crop: select scattered points with `KX >= 0 and KY >= 0` (1/4 BZ).
3. Interpolate the scattered 1/4-BZ pixels onto a regular grid covering
   only that quadrant (griddata; 1/4 of the interpolation work of the
   full-plane KD-tree path).
4. Mirror expansion as pure array operations on the regular grid:
   - `flip_kx`: reverse the kx axis and concatenate (kx < 0 half)
   - `flip_ky`: reverse the ky axis and concatenate (ky < 0 half)
   The symmetry axis row/column is shared (not duplicated); the result is
   exactly symmetric by construction. Disabled flips leave the
   corresponding half/quadrant zero-filled and the array is padded back
   to the full grid shape.

The quadrant path skips the 6-fold rotation used by the kdtree path
(the mirror expansion already covers the full plane).

## Modules

- `pimre/experiment/calibration.py`: new `quadrant_symmetrize(bands, KX,
  KY, flip_kx=True, flip_ky=True, kx_grid=None, ky_grid=None) -> 2D array`.
- `pimre/pipeline/preprocess.py`: `compute_grid` keeps shared logic
  (axes, shifts, Angle2Mon); `preprocess_full` branches on
  `pp["method"]`; quadrant path iterates layers with `stride`
  interpolation fill identical to the kdtree path.
- `pimre/config.py`: DEFAULTS entries.
- `configs/pimre_config.yaml`: add `method` key with example.
- `scripts/setup_config.py`: method dropdown + flip checkboxes wired in
  `_sync_all_from_ui` / `_load_to_ui`.

## Verification

1. Run quadrant preprocessing on the RbTiBi test data; check output
   grid is 4-quadrant symmetric (numerically).
2. Visual check with `test/glm_image_eval.py`.
3. Run the MRF pipeline (`run_mrf_pipeline`) end-to-end with the new
   preprocessed file to confirm downstream compatibility.

## Implementation status: VERIFIED (2026-08-08)

- `test/exp_preprocessed_quadrant.h5` (111 x 200 x 200): all 111 layers
  exactly symmetric across both k axes (max deviation 0); coverage ~49%
  (interp+flip, vs ~14.5% for the earlier binning version).
- `test/quadrant_out/`: MRF pipeline ran end-to-end (10 epochs/band).
  Alignment: isotropic scale 0.9945, rotation -0.44 deg, |K|/|M| 1.1586
  (ideal 1.1547); shared offset +0.60 eV; mean BSFI 0.250.
- glm-4.6v image check of path plots: reconstructed curves follow the
  bright intensity ridges, smooth, no artifacts.
- Cross-check vs kdtree path (`test/kdtree_out/`, same config): T
  scale 0.9903 / rotation -0.65 deg, same shared offset; reconstructed
  bands agree to 0.006-0.011 eV mean difference (corr 0.95-0.99).
- Design change (review): symmetry expansion is now performed as pure
  array operations on the regular grid after interpolating only the 1/4
  quadrant (no scattered-point mirroring + binning), reducing work and
  improving coverage/BSFI.
- Noise diagnosis (user report): kdtree curves look over-smoothed and
  lose band details because KDInterp averages every pixel over a
  radius=0.05 1/Angstrom neighborhood before interpolating. The quadrant
  path originally had no smoothing, keeping noise (path relative noise
  0.059 vs kdtree 0.006). Added configurable `quadrant.smooth_radius`
  (default 0.02, 0 disables): a light neighborhood average on the 1/4
  scattered pixels before interpolation. Result: path noise 0.028,
  curves visibly smoother while retaining more band detail than kdtree
  (confirmed by glm-4.6v comparison).
- Resolution: quadrant output now uses the same grid as the kdtree path
  (auto_grid -> 700x700).
- Gamma-hole fix: cubic interpolation returns NaN at the hull corner
  (Gamma) and hull boundary. Holes are now filled with the nearest pixel
  value only within `quadrant.fill_radius` (default 0.03 1/Angstrom) so
  Gamma and the symmetry axes stay populated while genuinely uncovered
  regions (K points outside the measured window) remain zero and stay
  distinguishable.
- HSP coverage selection: the default K0/M0 directions may fall outside
  the measured window (quadrant maps only cover the kx=0/ky=0
  directions; here K is covered at 90/270 deg and M at 60/120/240/300
  deg). New `select_hsps_by_coverage` scores all 6 K_i and their
  orientation-compatible M_i (30 deg CCW from K, matching the DFT-side
  construction so Gamma-M-K-Gamma is a closed right triangle along the
  BZ edge) and picks the best covered pair. For this data it selects
  K90 deg / M120 deg, giving a physically equivalent T with rotation
  ~60 deg (hexagonal symmetry) and scale 0.988.
- Simplification (user request): new `method: direct` — no 1/4 crop, no
  mirroring, no interpolation. The momentum-space pixel grid at the
  angular resolution (700x36 for this data, axes taken from the
  reference energy layer: kx per column / ky per row) is used as-is,
  and the coverage-based HSP selection finds a closed Gamma-M-K
  triangle inside the measured window. Verified: K90 deg/M120 deg
  selected, T scale 0.970 / rotation 59.6 deg, recon agrees with the
  kdtree path to 0.016-0.022 eV mean difference (corr 0.93-0.98),
  glm-4.6v reports smooth curves along the bright bands with no
  artifacts. Coverage scoring penalizes array-edge candidates
  (truncated windows) so that 1D-axis approximation cannot select
  HSPs outside the data.
- Path plot axis (user request): the horizontal axis of path_GMKG.png
  (and path_KG/path_MG) now uses the real momentum distance along the
  path (accumulated |Gamma-M|, |M-K|, |K-G|), resampling the intensity
  and reconstruction onto a uniform momentum axis, instead of the raw
  pixel count. Without this the few ky rows of the direct method made
  the K-G segment (mostly along ky) only 16 px vs 159 px for the other
  segments. Verified segment lengths 0.487/0.282/0.557 vs theoretical
  0.493/0.285/0.570 (1-2% error from the 1D-axis approximation).
- HSP calibration separation (user report): calibrate_hsps.py wrote the
  same `calibration.hsps` (rotation_angle/scale) for both experimental
  and DFT data, and the pipeline never applied any DFT-side
  calibration. Now:
  - `calibration.hsps` = experimental-data HSP calibration (used by
    find_hsps_robust as before).
  - `calibration.dft_hsps` = independent theory-data calibration,
    applied in `dft_KM(kx, ky, rotation_angle, scale)` (rotation
    around the DFT Gamma point) in the pipeline STEP 2.
  - calibrate_hsps.py reads/writes the mode-specific key
    (`dft_hsps` for --mode dft, `hsps` otherwise).
  - Verified: default config unchanged (identical T/BSFI); enabling
    dft_hsps rotation=60 deg shifts T by -60 deg (hexagonal
    equivalence) with the same mean score (0.3665 vs 0.3641).
- Removed redundant MRF hyperparameters (user request): `k_scale` was
  never consumed (MrfRec.initializeBand is not used by the pipeline)
  and the per-band `offset` was superseded by the shared BSFI offset
  search. Cleaned from config.py DEFAULTS, pimre_config.yaml,
  mrf_hyperparams.yaml and the setup_config GUI band rows (now DFT
  idx + eta only).
- High-precision path extraction (user request): band-path plots and
  the BSFI path-ridge evaluation now
  - sample each segment by real momentum length
    (`mrf.path_sample_step`, default 0.005 1/Angstrom per sample)
    instead of pixel count (the K-G segment of the direct method went
    from 16 to 117 samples), and
  - interpolate with `mrf.path_interp_method` (default 'cubic',
    options linear/cubic/nearest) via scipy.ndimage.map_coordinates
    splines (order 0/1/3). The per-layer 2D interpolation is ~750x
    faster than the 3D RGI cubic (0.098 s vs 74 s for the path map)
    and supports cubic for single-band arrays. Both parameters are
    editable in scripts/setup_config.py (MRF tab). glm-4.6v reports
    finer/sharper intensity ridges, correct segment proportions and
    smooth curves.
  - NaN fix: map_coordinates uses spline prefiltering by default, which
    spreads any NaN over the whole array (reconstructed curves vanished
    because bands outside the measured energy window are NaN). The
    interpolation now uses prefilter=False (local spline kernel), so
    NaN stays localized to the unconstrained regions; curves reappear
    (verified with glm-4.6v: 4 visible curves along the bright bands).
