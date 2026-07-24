# PIMRE: Physics-Informed Markov Random Field

PIMRE (Physics-Informed Markov Random Field) reconstructs electronic band
structures from ARPES data using Markov Random Field optimization with
DFT prior constraints.

The core idea is to combine two sources of information:

1. **DFT band structure calculations** — provide an initial guess for the
   band energies at each momentum point via an affine transform
2. **ARPES intensity data** — provide a likelihood term through the MRF
   optimization, favoring band positions that align with regions of high
   photoemission intensity

The MRF balances these two terms with a smoothness prior (neighbor
interaction) to produce a physically plausible band structure that is
consistent with both the DFT prediction and the experimental data.

## Algorithm Overview

For each band, the MRF model operates on a 2D momentum grid:

```
E(kx, ky) = argmax [ log I(kx, ky, E) - Σ (E - E_neighbor)² / (2η²) ]
```

where:
- `I(kx, ky, E)` is the ARPES intensity
- `E_neighbor` are the band energies at adjacent k-points
- `η` controls the smoothness of the reconstruction

The optimization uses a checkerboard pattern with PyTorch tensor
operations for efficient parallel updates on GPU.

## Package Structure

```
pimre/
├── config.py          Configuration management, OUTCAR/KPOINTS parsing
├── dft/               DFT data processing
│   └── reader.py      CSV reading, BZ expansion, grid interpolation
├── experiment/        Experimental data preprocessing
│   └── calibration.py Angle-to-momentum, KD-interpolation, rotation
├── kpath/             High-symmetry point finding
│   ├── symmetry.py    Lattice-to-reciprocal, C6 HSP generation
│   ├── registration.py Mirror symmetry BZ registration
│   ├── corrector.py   Momentum corrector for peak detection
│   └── path.py        Band path extraction utilities
├── mrf/               MRF reconstruction
│   ├── model.py       MrfRec model (PyTorch backend)
│   ├── evaluation.py  BSFI scoring, affine transform, band mapping
│   └── symmetry.py    Rotational symmetrization
├── gui/               Interactive calibration widgets
│   └── calibration.py GammaCalibrator, GridCalibrator
├── pipeline/          End-to-end pipeline functions
│   ├── preprocess.py  Experimental preprocessing pipeline
│   ├── mrf.py         MRF + BSFI reconstruction pipeline
│   └── dft.py         DFT processing pipeline
├── utils/             Utilities
│   ├── io.py          HDF5 I/O, band structure loading
│   ├── image.py       Image normalization
│   └── interaction.py Draggable line widgets
└── zernike/           Zernike and Hexike polynomial bases
    └── polynomials.py
```

## Quick Links

- [Installation](installation.md)
- [Usage Guide](usage.md)
- [Workflow](workflow.md)
- [Configuration](configuration.md)
- [API Reference](api/overview.md)
- [Contributing](contributing.md)