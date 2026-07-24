# Changelog

## [0.1.0] — 2024-07-24

### Added

- Initial release of PIMRE (Physics-Informed Markov Random Field)
- DFT data processing pipeline (CSV → band map)
- Experimental ARPES preprocessing (angle calibration, momentum conversion, KD-interpolation)
- C6 rotation-based high-symmetry point generation
- Mirror symmetry BZ registration for automatic rotation detection
- Interactive HSP calibration GUI with overlay controls
- MRF band reconstruction with PyTorch backend (checkerboard parallel updates)
- BSFI (Band Structure Fidelity Index) hierarchical offset optimization
- Affine transform for DFT-to-experimental coordinate mapping
- Band symmetrization (mirror + rotational)
- Band path plotting (Γ-M-K-Γ, K-Γ-K, M-Γ-M)
- PyQt5 configuration GUI (`setup_config.py`)
- YAML-based configuration management
- HDF5 and MATLAB data I/O
- Interactive Gamma point and grid calibration