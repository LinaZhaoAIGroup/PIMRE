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
| `rotation_angle` | float | `0.0` | User-specified BZ rotation (deg) |
| `scale` | float | `1.0` | Momentum scale factor |

## `dft`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | str | `""` | Path to DFT raw data directory |
| `csv_file` | str | `"extracted_data.csv"` | DFT CSV file name (resolved against `dft.path`; used when `--dft-csv` is not passed to `run_dft_processing.py`) |
| `fermi_file` | str | `"BAND_GAP"` | Fermi energy / BAND_GAP file name (resolved against `dft.path`; used when neither `--fermi-file` nor `--band-gap-file` is passed) |
| `k_grid` | list | `[20, 20]` | DFT k-point grid size |
| `output_grid` | list | `[101, 101]` | Output interpolation grid size (used by the `griddata` method) |
| `drop_top_bands` | int or null | `null` | Number of highest-energy conduction bands to drop from the stacked band structure. The stack is ordered by descending band energy (highest conduction band first, VBM at position `n_conduction - drop_top_bands`), so plain `mrf.bands.index` entries depend on this number matching the DFT band count. Prefer `from_vbm` entries (see `mrf.bands`) |

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
| `alignment` | str | `"hsp"` | DFT→experiment momentum mapping: `hsp` = the experimental and theoretical momentum scales differ, so the DFT grid is stretched/rotated by exactly matching the Γ→K and Γ→M vectors on both sides (`T = S_exp @ inv(S_dft)`, as in the reference implementation) — recommended; `gamma` = identity transform, only valid when both axes already share the same absolute momentum calibration |
| `offset_mode` | str | `"per_band"` | Energy-offset selection: `per_band` = each band takes its own BSFI optimum (recommended for metallic systems), `shared` = all bands take the global mean-score optimum |
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
| `fine_tune_range` | float | `0.05` | Legacy per-band fine-tune range (kept for GUI compatibility; the pipeline now searches the full `offset_range` per band) |
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
