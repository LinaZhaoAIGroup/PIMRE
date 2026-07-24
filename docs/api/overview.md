# API Reference

## Module Index

### `pimre.config`

Configuration management utilities.

| Function | Description |
|----------|-------------|
| `load_config(path)` | Load YAML config with defaults |
| `save_config(cfg, path)` | Save config to YAML |
| `crystallographic_data(cfg)` | Extract [a,b,c,α,β,γ] |
| `parse_outcar(path)` | Parse VASP OUTCAR for lattice |
| `parse_kpoints(path)` | Parse VASP KPOINTS for k-grid |

### `pimre.dft`

DFT data processing.

| Module | Description |
|--------|-------------|
| `pimre.dft.reader` | CSV reading, BZ expansion, interpolation, I/O |
| `read_dft_csv(path, fermi, nkx, nky)` | Read and reshape DFT CSV |
| `expand_bz(coords, bands)` | C6 rotation + reflection expansion |
| `interpolate_to_grid(coords, bands)` | griddata interpolation to regular grid |
| `save_band_map_h5(path, evb, ecb, kx, ky)` | Save band map as HDF5 |
| `load_band_map_h5(path)` | Load band map from HDF5 |

### `pimre.experiment`

Experimental data preprocessing.

| Module | Description |
|--------|-------------|
| `pimre.experiment.calibration` | Angle-to-momentum, KD-interpolation, rotation |
| `Angle2Mon(E_grid, X_Angle, Y_Angle)` | Angle to momentum conversion |
| `KDInterp(bands, KX, KY, ...)` | KD-tree based interpolation |
| `RotateCoordinates(KX, KY, theta)` | 2D coordinate rotation |
| `save_preprocessed_h5(path, E, kx, ky, V)` | Save preprocessed HDF5 |

### `pimre.kpath`

High-symmetry point finding and BZ registration.

| Module | Description |
|--------|-------------|
| `pimre.kpath.symmetry` | Lattice-to-reciprocal, HSP generation |
| `pimre.kpath.registration` | Mirror symmetry BZ registration |
| `pimre.kpath.path` | Band path extraction |
| `pimre.kpath.corrector` | Momentum corrector for peak detection |
| `lattice_to_reciprocal(a,b,c,α,β,γ)` | Compute K and M coordinates |
| `Get_G_M_K(crystal, kx, ky)` | Find all HSP indices (6K + 6M) |
| `find_hsps_robust(intensity, kx, ky, crystal)` | Robust HSP finding with calibration |
| `register_bz(intensity, kx, ky, crystal, G)` | Mirror symmetry registration |
| `build_hsps_from_registration(crystal, kx, ky, G, theta)` | Build HSPs from rotation |

### `pimre.mrf`

MRF reconstruction and evaluation.

| Module | Description |
|--------|-------------|
| `pimre.mrf.model` | MrfRec model (PyTorch backend) |
| `pimre.mrf.evaluation` | BSFI, affine transform, band mapping |
| `pimre.mrf.symmetry` | Rotational symmetrization |
| `MrfRec(E, kx, ky, I, eta, ...)` | MRF band reconstruction model |
| `compute_bsfi_2d(E0, I_t, E_arr)` | 2D BSFI score computation |
| `compute_affine_transform(...)` | DFT→exp affine transform |
| `map_dft_bands(E_dft, kx, ky, ...)` | Map DFT bands via T_inv |

### `pimre.gui`

Interactive calibration widgets.

| Module | Description |
|--------|-------------|
| `pimre.gui.calibration` | GammaCalibrator, GridCalibrator |

### `pimre.pipeline`

End-to-end pipeline functions.

| Module | Description |
|--------|-------------|
| `pimre.pipeline.dft` | `run_dft_pipeline(...)` |
| `pimre.pipeline.preprocess` | `compute_grid(cfg)`, `preprocess_full(cfg, ...)` |
| `pimre.pipeline.mrf` | `run_mrf_pipeline(...)`, `draw_path(...)` |

### `pimre.utils`

Utilities.

| Module | Description |
|--------|-------------|
| `pimre.utils.io` | `loadHDF`, `saveHDF`, `load_bandstruct` |
| `pimre.utils.image` | `normalize` |
| `pimre.utils.interaction` | `DraggableVLine`, `DraggableHLine` |

## Data Classes

### `HspsResult`

```python
@dataclass
class HspsResult:
    hsps: dict                    # {"G": (gx,gy), "K0": ..., "M0": ...}
    confidence: float             # 0-1 confidence score
    source: str                   # "theory" | "registered" | "manual"
    symmetry_score: float         # Hexagonal symmetry score
    best_layer: int               # Best energy layer index
    rotation_angle: float         # BZ rotation angle (deg)
    scale: float                  # Momentum scale factor
    registration_score: float     # Registration correlation score
    cv_errors: dict               # Cross-validation errors
```

### `MrfRec`

```python
class MrfRec:
    def __init__(self, E, kx, ky, I, E0=None, eta=0.1, ...)
    def smoothenI(self, sigma=(1.0, 1.0, 1.0))
    def initializeBand(self, kx, ky, Eb, offset=0.0, kScale=1.0)
    def iter_para(self, num_epoch=1)       # Parallel (PyTorch)
    def iter_seq(self, num_epoch=1)        # Sequential (numpy)
    def getEb(self)                         # Get band energies
    def getLogP(self)                       # Get log-likelihood
    def saveBand(self, fileName)            # Save to HDF5
    def symmetrizeI(self, mirror=True, rotational=True)
```