# PIMRE: Photoemission Intensity MRF Reconstruction Engine

PIMRE reconstructs electronic band structures from ARPES (Angle-Resolved Photoemission Spectroscopy) data using Markov Random Field (MRF) optimization with DFT prior constraints.

## Quick Start

```bash
# Install
uv pip install -e .

# DFT processing
python scripts/run_dft_processing.py --dft_csv data/band_structure.csv --fermi_file data/FERMI_ENERGY --output data/band_map.mat

# Experimental preprocessing
python scripts/run_exp_preprocessing.py --config configs/experiment.yaml

# MRF reconstruction
python scripts/run_mrf_reconstruction.py --exp_data data/exp_preprocessed.h5 --band_map data/band_map.mat
```

## Architecture

```
pimre/
├── zernike/       # Zernike & Hexike polynomial bases
├── utils/         # IO (HDF5, mat), image processing, interactive tools
├── dft/           # DFT data reading, BZ expansion, interpolation
├── experiment/    # Experimental data loading, angle-to-momentum conversion
├── kpath/         # High-symmetry paths, momentum correction
└── mrf/           # MRF model (PyTorch), symmetrization, evaluation
```

## Key Features

- **PyTorch backend**: GPU-accelerated MRF optimization replaces TensorFlow
- **Configuration-driven**: All parameters in YAML configs under `configs/`
- **Componentized**: Modular design for easy extension and reuse
- **No heavy dependencies**: Removed reliance on `fuller`, `mpes`, `poppy`, `symmetrize`, `ArpesBandRecons`

## See Also

- [Installation](installation.md)
- [Usage Guide](usage.md)
- [API Reference](api/)