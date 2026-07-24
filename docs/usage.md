# Usage Guide

## CLI Commands

PIMRE provides four scripts in the `scripts/` directory:

### DFT Processing

Process raw DFT calculation data into a band map:

```bash
uv run python scripts/run_dft_processing.py \
    --dft-csv extracted_data.csv \
    --band-gap-file BAND_GAP \
    --output test/band_map.h5
```

Options:
- `--config`: Path to config YAML (default: `configs/defaults.yaml`)
- `--dft-csv`: Path to DFT CSV file (required)
- `--band-gap-file`: Path to BAND_GAP file
- `--fermi-file`: Path to FERMI_ENERGY file
- `--output`: Output file path (default: `band_map.h5`)
- `--output-format`: `h5` or `mat` (default: `h5`)
- `--method`: `grid_cell` or `griddata` (default: `grid_cell`)

### Experimental Preprocessing

Convert raw ARPES data to momentum space with interactive calibration:

```bash
uv run python scripts/preprocess_exp.py
uv run python scripts/preprocess_exp.py --skip-calib  # use saved calibration
uv run python scripts/preprocess_exp.py --calib-only   # only calibrate
```

### HSP Calibration

Interactive BZ high-symmetry point calibration:

```bash
# ARPES data
uv run python scripts/calibrate_hsps.py

# DFT band map
uv run python scripts/calibrate_hsps.py --mode dft --band-map test/band_map.h5
```

### MRF Reconstruction

Run the full MRF + BSFI reconstruction:

```bash
uv run python scripts/run_mrf.py
uv run python scripts/run_mrf.py --exp-data test/exp_preprocessed.h5 --band-map test/band_map.h5
```

## Configuration

All parameters are stored in `configs/pimre_config.yaml`. See the
[Configuration Reference](configuration.md) for the complete schema.

## Python API

### DFT Processing

```python
from pimre.pipeline.dft import run_dft_pipeline

run_dft_pipeline(
    dft_csv="extracted_data.csv",
    band_gap_file="BAND_GAP",
    output="band_map.h5",
)
```

### Experimental Preprocessing

```python
from pimre.pipeline.preprocess import compute_grid, preprocess_full

E_grid, bands, kx_angle, ky_angle, bands_rep, KX_rot, KY_rot, kx_out, ky_out = compute_grid(cfg)
E_Mon = preprocess_full(cfg, E_grid, bands_rep, KX_rot, KY_rot, kx_out, ky_out)
```

### MRF Reconstruction

```python
from pimre.pipeline.mrf import run_mrf_pipeline

recon, params = run_mrf_pipeline(
    config_path="configs/pimre_config.yaml",
    exp_data="test/exp_preprocessed.h5",
    band_map="test/band_map.h5",
)
```

### Low-Level API

```python
from pimre.mrf.model import MrfRec
from pimre.kpath.symmetry import find_hsps_robust
from pimre.mrf.evaluation import compute_bsfi_2d, compute_affine_transform

# Create MRF model
mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=0.12)
mrf.smoothenI(sigma=(0.5, 0.5, 0.1))

# Find HSPs
result = find_hsps_robust(intensity, kx, ky, crystal_data, E_grid)

# Compute affine transform
T, T_inv, sx, sy, rot = compute_affine_transform(
    kx, ky, G, K, M, kx_dft, ky_dft, KP_dft, MP_dft)

# Compute BSFI score
score = compute_bsfi_2d(E0, I_t, E)
```

## Output Files

After running the MRF pipeline, the following files are generated in `test/`:

| File | Description |
|------|-------------|
| `recon_bands.npy` | Reconstructed band energies |
| `bsfi_curve.png` | BSFI vs offset optimization plot |
| `path_GMKG.png` | Γ-M-K-Γ band path |
| `path_KG.png` | K-Γ-K band path |
| `path_MG.png` | M-Γ-M band path |
| `final_parameters.json` | Final band parameters |
| `bsfi_scores.npz` | BSFI search scores |
| `band_map.h5` | DFT band map |
| `exp_preprocessed.h5` | Preprocessed experimental data |