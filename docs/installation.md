# Installation

## Requirements

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) (recommended)

## Install with uv

```bash
cd PIMRE
uv pip install -e .
```

## Install with pip

```bash
cd PIMRE
pip install -e .
```

## Dependencies

| Package | Purpose |
|---------|---------|
| numpy, scipy | Numerical computation |
| torch | MRF optimization (GPU accelerated) |
| h5py | HDF5 data I/O |
| matplotlib | Visualization |
| pyyaml | Configuration parsing |
| tqdm | Progress bars |
| natsort | Natural string sorting (optional) |
| pillow | Image processing (optional) |

## Dev Dependencies

```bash
uv pip install -e ".[dev]"  # Includes ruff for linting
```