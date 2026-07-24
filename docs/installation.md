# Installation

## Requirements

- Python ≥ 3.11
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- PyTorch ≥ 2.0 (GPU recommended for MRF optimization)

## Install with uv

```bash
git clone https://github.com/LinaZhaoAIGroup/PIMRE.git
cd PIMRE
uv sync
```

This creates a virtual environment and installs all dependencies.

## Install with pip

```bash
git clone https://github.com/LinaZhaoAIGroup/PIMRE.git
cd PIMRE
pip install -e .
```

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, scipy | Numerical computation |
| torch | MRF optimization (GPU accelerated) |
| h5py | HDF5 data I/O |
| matplotlib | Visualization and interactive GUIs |
| pyyaml | Configuration parsing |
| tqdm | Progress bars |
| scikit-image | Peak detection (MomentumCorrector) |
| pandas | DFT CSV reading |
| PyQt5 | GUI applications |

## Development

```bash
uv sync --dev
```

This installs additional development dependencies including:

- `ruff` — linting and formatting
- `pytest` — testing

## Verifying Installation

```bash
uv run python -c "from pimre.mrf.model import MrfRec; print('PIMRE installed successfully')"
```