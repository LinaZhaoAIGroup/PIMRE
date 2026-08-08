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
3. Mirror expansion: for each cropped point (kx, ky) generate copies:
   - `(kx, ky)`          always
   - `(-kx, ky)`         if flip_kx
   - `(kx, -ky)`         if flip_ky
   - `(-kx, -ky)`        if both flips
4. Binning: accumulate all copies onto the regular output grid
   (kx_out x ky_out, taken from `compute_grid`). Intensity of copies
   landing in the same bin is averaged (sum / count). Bins without data
   are filled with 0 (same convention as KDInterp).

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
