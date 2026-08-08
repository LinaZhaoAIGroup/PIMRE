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
| `csv_file` | str | `"extracted_data.csv"` | DFT CSV file name |
| `fermi_file` | str | `"BAND_GAP"` | Fermi energy file name |
| `k_grid` | list | `[20, 20]` | DFT k-point grid size |
| `output_grid` | list | `[101, 101]` | Output interpolation grid size |

## `preprocessing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `output_grid` | int | `200` | Output momentum grid size |
| `output_path` | str | `""` | Preprocessed data output path |
| `kd_radius` | float | `0.05` | KD-tree merge radius (Å⁻¹) |
| `n_rotations` | int | `6` | Number of rotational copies |
| `stride` | int | `10` | Stride for KD-interpolation |
| `sort_axes` | bool | `false` | Sort axes to increasing, flip data |
| `sign_correct` | bool | `false` | Apply sign correction to momentum |
| `auto_grid` | bool | `false` | Auto-determine grid size from KX/KY |

## `mrf`

### `mrf.bands`

A list of band configurations, each with:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `index` | int | — | DFT band index (×2 for stacked) |
| `k_scale` | float | — | Momentum scale factor |
| `offset` | float | — | Initial energy offset (eV) |
| `eta` | float | — | MRF smoothness parameter |

### `mrf.bsfi`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offset_range` | float | `1.0` | BSFI offset search range (eV) |
| `offset_step` | float | `0.1` | BSFI offset search step (eV) |
| `fine_tune_range` | float | `0.05` | Legacy per-band fine-tune range (kept for GUI compatibility; the pipeline now searches the full `offset_range` per band) |
| `ridge_sigma` | float | `0.1` | Width of the ridge alignment penalty (eV) |

### `mrf.bsfi.weights`

The score is `Σ w_i·metric_i / Σ w_i`; setting a weight to `0` disables that component.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `correlation` | float | `0.6` | Weight for dE/dk correlation |
| `intensity` | float | `0.3` | Weight for intensity ratio |
| `snr` | float | `0.1` | Weight for signal-to-noise ratio |
| `ridge` | float | `0.5` | Weight for band-ridge alignment (1 = band on local intensity ridge) |

### Other MRF fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `offset_mode` | str | `"hierarchical"` | `"hierarchical"` or `"shared"` |
| `smooth_sigma` | list | `[0.5, 0.5, 0.1]` | Gaussian smooth sigma (kx, ky, E) |