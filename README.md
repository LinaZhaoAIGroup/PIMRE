# PIMRE: Physics-Informed Markov Random Field for ARPES Band Reconstruction

[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/uv-package_manager-blueviolet)](https://github.com/astral-sh/uv)

PIMRE reconstructs electronic band structures from ARPES (Angle-Resolved
Photoemission Spectroscopy) data using Markov Random Field optimization
with DFT prior constraints. The package provides a full pipeline from raw
DFT calculations and experimental data to reconstructed band dispersions.

## Key Features

- **GPU-accelerated MRF**: PyTorch backend with checkerboard parallel updates
- **Automatic BZ registration**: Mirror symmetry-based rotation detection
- **Interactive calibration GUI**: Real-time BZ overlay with rotation/scale sliders
- **BSFI optimization**: Band Structure Fidelity Index for energy offset search
- **Configuration-driven**: All parameters in a single YAML config file
- **Modular design**: Clean API with separable pipeline stages

## Quick Start

```bash
git clone https://github.com/LinaZhaoAIGroup/PIMRE.git
cd PIMRE
uv sync
```

### 1. DFT Processing

```bash
uv run python scripts/run_dft_processing.py \
    --dft-csv extracted_data.csv \
    --band-gap-file BAND_GAP \
    --output test/band_map.h5
```

### 2. Experimental Preprocessing

```bash
uv run python scripts/preprocess_exp.py
```

### 3. HSP Calibration (optional)

```bash
uv run python scripts/calibrate_hsps.py
```

### 4. MRF Reconstruction

```bash
uv run python scripts/run_mrf.py
```

## Pipeline

```
  DFT CSV                    Raw ARPES HDF5
     │                            │
     ▼                            ▼
┌─────────┐              ┌────────────────┐
│ DFT     │              │ Preprocessing  │
│ Process │              │  · Angle calib │
│  · BZ   │              │  · Angle→Mom   │
│  · Grid │              │  · C6 rotation │
└────┬────┘              │  · KD-interp   │
     │                   └───────┬────────┘
     ▼                           ▼
band_map.h5              exp_preprocessed.h5
     │                           │
     └─────────┬─────────────────┘
               ▼
       ┌───────────────┐
       │ HSP Calibration│
       │  · Mirror sym  │
       │  · Manual GUI  │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │ Affine T       │
       │  DFT → Exp     │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │ BSFI Search    │
       │  · Shared off  │
       │  · Per-band    │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │ MRF Recon      │
       │  · PyTorch     │
       │  · Symmetrize  │
       └───────┬───────┘
               ▼
          Band Dispersions
```

## Architecture

```
pimre/
├── dft/           DFT data reading, BZ expansion, grid interpolation
├── experiment/    Angle-to-momentum conversion, KD-tree interpolation
├── kpath/         HSP finding, C6 rotation, mirror symmetry registration
├── mrf/           MRF model (PyTorch), BSFI scoring, evaluation utilities
├── gui/           Interactive calibration widgets (Gamma, BZ)
├── pipeline/      End-to-end pipeline functions (DFT, preprocess, MRF)
└── utils/         HDF5 I/O, image processing, interactive matplotlib tools
```

## Documentation

| Document | Description |
|----------|-------------|
| [Installation](docs/installation.md) | Setup and dependencies |
| [Usage Guide](docs/usage.md) | CLI commands and Python API |
| [Workflow](docs/workflow.md) | Detailed pipeline steps |
| [Configuration](docs/configuration.md) | Full YAML config reference |
| [API Reference](docs/api/overview.md) | Module index |
| [Contributing](docs/contributing.md) | Development guidelines |

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- PyTorch ≥ 2.0 (GPU recommended for MRF optimization)

## Citation

If you use PIMRE in your research, please cite:

```bibtex
@software{pimre2024,
  title       = {PIMRE: Physics-Informed Markov Random Field for ARPES Band Reconstruction},
  author      = {{PIMRE contributors}},
  year        = {2024},
  url         = {https://github.com/LinaZhaoAIGroup/PIMRE}
}
```

## License

PIMRE is licensed under the GNU Lesser General Public License v2.1.
See [LICENSE](LICENSE) for the full text.