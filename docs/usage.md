# Usage Guide

## Pipeline Overview

The PIMRE pipeline consists of three stages:

1. **DFT Processing** (`run_dft_processing.py`): Convert DFT band calculations to band maps
2. **Experimental Preprocessing** (`run_exp_preprocessing.py`): Convert raw ARPES data to momentum space
3. **MRF Reconstruction** (`run_mrf_reconstruction.py`): Combine DFT prior with experimental data

## Configuration

All parameters are stored in YAML files under `configs/`:

- `defaults.yaml`: Global defaults (lattice, k-grid, energy parameters)
- `experiment.yaml`: Experiment-specific settings (paths, angles, ranges)
- `mrf_hyperparams.yaml`: MRF reconstruction hyperparameters per band

### Example: experiment.yaml

```yaml
exp_name: "Au"
exp_path: "/path/to/Data/"
wave_name: "Au111"
kx_shift: -6.77
ky_shift: 6.58
kx_dim: 700
ky_dim: 36
```

### Example: mrf_hyperparams.yaml

```yaml
bands:
  - index: 0
    k_scale: 1.24
    offset: 0.75
    eta: 0.0000065
```

## Python API

### DFT Processing

```python
from pimre.dft.reader import read_dft_csv, expand_bz, interpolate_to_grid

cartesian_coords, energy_bands, ebands = read_dft_csv("band_structure.csv", fermi_energy=5.767)
bz_coords, repeated_bands = expand_bz(cartesian_coords, energy_bands)
mapping, kx_grid, ky_grid = interpolate_to_grid(bz_coords, repeated_bands)
```

### Experimental Preprocessing

```python
from pimre.experiment.calibration import load_exp_data, Angle2Mon, SingleLayerConversion

bands = load_exp_data("data.h5", "Au111")
KX, KY = Angle2Mon(E_grid, kx_angle, ky_angle)
E_Mon, kx, ky = SingleLayerConversion(bands, KX, KY, kx_dim=700, ky_dim=36)
```

### MRF Reconstruction

```python
from pimre.mrf.model import MrfRec

mrf = MrfRec(E=E, kx=kx, ky=ky, I=I, eta=0.12)
mrf.smoothenI(sigma=(0.5, 0.5, 0.1))
mrf.initializeBand(kx=kx_dft, ky=ky_dft, Eb=E_dft[0], kScale=1.24, offset=0.75)
mrf.iter_para(num_epoch=10)
reconstructed = mrf.getEb()
```