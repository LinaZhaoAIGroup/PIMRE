# Configuration Reference

The main configuration file is `configs/pimre_config.yaml`. All fields are
documented below.

## `sample`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | `"unnamed"` | Sample name for reporting |

## `lattice`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `a` | float | `5.8077` | Lattice constant a (Å) |
| `b` | float | `5.8077` | Lattice constant b (Å) |
| `c` | float | `9.1297` | Lattice constant c (Å) |
| `alpha` | float | `90.0` | Lattice angle α (deg) |
| `beta` | float | `90.0` | Lattice angle β (deg) |
| `gamma` | float | `120.0` | Lattice angle γ (deg) |

## `arpes`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | str | `""` | Path to raw ARPES HDF5 file |
| `dataset` | str | `""` | Dataset path within HDF5 (e.g., `"RbTiBi/map"`) |
| `work_function` | float | `16.03` | Work function (eV) |

### `arpes.energy`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start` | float | `1.06` | Starting energy (eV) |
| `delta` | float | `-0.01` | Energy step (eV) |
| `npts` | int | `111` | Number of energy points |
| `flip` | bool | `false` | Quick axis label flip |

### `arpes.kx_angle`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start` | float | `-20.3729` | Starting kx angle (deg) |
| `delta` | float | `0.0448678` | kx angle step (deg) |
| `npts` | int | `700` | Number of kx points |
| `flip` | bool | `false` | Quick axis label flip |

### `arpes.ky_angle`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start` | float | `-7.3251` | Starting ky angle (deg) |
| `delta` | float | `1.0` | ky angle step (deg) |
| `npts` | int | `36` | Number of ky points |
| `flip` | bool | `false` | Quick axis label flip |

## `calibration`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kx_shift` | float | `0.0` | Gamma calibration shift in kx angle (deg) |
| `ky_shift` | float | `0.0` | Gamma calibration shift in ky angle (deg) |
| `kx_grid_shift` | float | `0.0` | Grid shift in kx momentum (Å⁻¹) |
| `ky_grid_shift` | float | `0.0` | Grid shift in ky momentum (Å⁻¹) |

### `calibration.hsps`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `manual` | bool | `false` | Use manual BZ rotation angle |
| `rotation_angle` | float | `0.0` | User-specified BZ rotation (deg): absolute K-ring direction, `K_i` at `rotation_angle + 60i`, `M_i` at `rotation_angle − 30 + 60i` |
| `scale` | float | `1.0` | Momentum scale factor |

`calibration.dft_hsps` uses the same fields for the DFT-side HSPs in
`dft_KM`. Note the hexagonal 30° ambiguity: automatic mirror registration
cannot distinguish the K-ring at 0° from 30°, so if the affine transform
comes out with a spurious ±30° rotation, set `rotation_angle` manually
(60.0 for the reference `band_map.mat` / `HPES_preprocessed_new.h5` pair,
which reproduces the reference `T` exactly).

## `dft`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | str | `""` | Path to DFT raw data directory |
| `csv_file` | str | `"extracted_data.csv"` | DFT CSV file name (resolved against `dft.path`; used when `--dft-csv` is not passed to `run_dft_processing.py`) |
| `fermi_file` | str | `"BAND_GAP"` | Fermi energy / BAND_GAP file name (resolved against `dft.path`; used when neither `--fermi-file` nor `--band-gap-file` is passed) |
| `k_grid` | list | `[20, 20]` | DFT k-point grid size |
| `output_grid` | list | `[101, 101]` | Output interpolation grid size (used by the `griddata` method) |
| `drop_top_bands` | int or null | `null` | Number of highest-energy conduction bands to drop from the stacked band structure. The stack is ordered by descending band energy (highest conduction band first, VBM at position `n_conduction - drop_top_bands`), so plain `mrf.bands.index` entries depend on this number matching the DFT band count. Prefer `from_vbm` entries (see `mrf.bands`) |

The band map passed to the MRF stage (`--band-map`) may be the pipeline's
`.h5` product or a legacy MATLAB `.mat` (`evb`/`ecb`/`kxxsc`/`kyysc`, as in
the reference `band_map.mat`); `drop_top_bands` applies to both. For the
legacy `.mat` the reference recipe is `drop_top_bands: 33` on a file with
37 conduction bands, leaving the four near-E_F conduction bands plus the
valence manifold.

## `preprocessing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_grid` | int | `200` | Output momentum grid size |
| `output_path` | str | `""` | Preprocessed data output path |
| `kd_radius` | float | `0.05` | KD-tree merge radius (Å⁻¹) |
| `n_rotations` | int | `6` | Number of rotational copies (1–6, original orientation included; the 60° step is fixed by the hexagonal symmetry) |
| `stride` | int | `10` | Stride for KD-interpolation |
| `sort_axes` | bool | `false` | Sort axes to increasing, flip data |
| `auto_grid` | bool | `false` | Auto-determine grid size from KX/KY |
| `normalize` | bool | `true` | Divide the raw counts by the global intensity maximum so the data lies in [0, 1] (same convention as the reference HPES preprocessed data, whose maximum is 1). Applied to raw input before rotation/interpolation; the MRF/BSFI metrics are scale-invariant, so this changes bookkeeping, not the reconstruction |
| `workers` | int | `0` | Number of parallel processes for the per-layer KD/quadrant interpolation. `0` = use all CPU cores, `1` = serial. The computed layers (every `stride`-th) are independent, so layer-level parallelism is the effective speedup; the per-layer cubic `griddata` is single-threaded |
| `method` | str | `"kdtree"` | Preprocessing method: `kdtree`, `quadrant`, or `direct` |
| `quadrant.flip_kx` | bool | `true` | Mirror across ky=0 (expand into kx<0) |
| `quadrant.flip_ky` | bool | `true` | Mirror across kx=0 (expand into ky<0) |
| `quadrant.smooth_radius` | float | `0.02` | Neighborhood average radius before interpolation (Å⁻¹) |
| `quadrant.fill_radius` | float | `0.03` | Max distance for nearest-fill of interpolation holes (Å⁻¹) |

## `mrf`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `eta` | float | `0.12` | Default MRF smoothness parameter (overridden per band by `mrf.bands[].eta`) |
| `num_epochs` | int | `10` | MRF iteration epochs |
| `max_shift` | int | `10` | Max energy-grid steps a node may move from its DFT prior |
| `device` | str | `"auto"` | Torch device for the MRF checkerboard update: `auto` = CUDA GPU if available, else CPU; `cpu`/`cuda` force a device. On an RTX 2070 SUPER the GPU runs ~10× faster with bit-identical results (verified on the dataCliu chain) |
| `alignment` | str | `"hsp"` | DFT→experiment momentum mapping: `hsp` = the experimental and theoretical momentum scales differ, so the DFT grid is stretched/rotated by exactly matching the Γ→K and Γ→M vectors on both sides (`T = S_exp @ inv(S_dft)`, as in the reference implementation) — recommended; `gamma` = identity transform, only valid when both axes already share the same absolute momentum calibration |
| `offset_mode` | str | `"per_band"` | Energy-offset selection: `per_band` = each band takes its own BSFI optimum over the full grid; `shared` = all bands take the global mean-score optimum; `hierarchical` = shared coarse search followed by a per-band fine-tune within ±`bsfi.fine_tune_range` of the shared optimum (reference behaviour; prevents band-order crossing) |
| `occupied_only` | bool | `true` | Restrict bands and offset search to occupied states (`E0 >= 0` and `E_dft <= 0`). `false` aligns full bands including empty-state segments (reference behaviour for near-E_F metallic bands) |
| `smooth_sigma` | list | `[0.5, 0.5, 0.1]` | Gaussian smooth sigma (kx, ky, E) |
| `path_interp_method` | str | `"cubic"` | Interpolation for band-path maps (`cubic`, `linear`, `nearest`) |
| `path_sample_step` | float | `0.005` | Path sample density (Å per sample) |

### `mrf.bands`

A list of band configurations, each with:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `from_vbm` | int | — | Bands below the valence-band maximum (`0` = VBM itself, `1` = next band down). Independent of `dft.drop_top_bands`; preferred over `index` |
| `index` | int | — | Absolute position in the stacked band array after `drop_top_bands` (depends on the drop count matching the DFT band count) |
| `eta` | float | `0.05` | MRF smoothness parameter for this band |

### `mrf.bsfi`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offset_range` | float | `1.0` | BSFI offset search range (eV) |
| `offset_step` | float | `0.1` | BSFI offset search step (eV) |
| `fine_tune_range` | float | `0.05` | Per-band fine-tune half-range around the shared optimum in `offset_mode: hierarchical` (eV) |
| `ridge_sigma` | float | `0.1` | Width of the ridge alignment penalty (eV) |

### `mrf.bsfi.weights`

The score is `Σ w_i·metric_i / Σ w_i`; setting a weight to `0` disables that component. All metrics are scaled to [0, 1] (SNR is squashed via `s/(s+1)`), so the weights keep their relative meaning.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `correlation` | float | `0.6` | Weight for dE/dk correlation |
| `intensity` | float | `0.3` | Weight for intensity ratio |
| `snr` | float | `0.1` | Weight for signal-to-noise ratio (squashed to [0, 1)) |
| `ridge` | float | `0.5` | Weight for band-ridge alignment (1 = band on local intensity ridge) |
| `path_ridge` | float | `0.8` | Weight for band-ridge alignment along the Γ-M-K-Γ path (combined with the 2D score) |
