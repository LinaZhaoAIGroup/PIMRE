# Workflow

This document describes each step of the PIMRE pipeline in detail.

## Step 1: DFT Processing

**Input**: `extracted_data.csv` (DFT band energies at k-points), `BAND_GAP` (Fermi energy)

**Output**: `band_map.h5` with `axes/kx`, `axes/ky`, `bands/evb`, `bands/ecb`

**Process**:

1. **Read DFT data**: Parse CSV file with k-point coordinates and band energies.
   Subtract Fermi energy to align bands.

2. **Coordinate transform**: Convert reciprocal lattice coordinates to
   Cartesian coordinates using the matrix:

   ```
   [ 1   0.5 ]
   [ 0   √3/2 ]
   ```

3. **BZ expansion**: Apply C6 rotation (6-fold) and reflection to expand
   the irreducible Brillouin zone to the full BZ. This generates 12 copies
   of the original k-points.

4. **Grid interpolation**: Interpolate scattered k-points onto a uniform
   101×101 grid using `scipy.interpolate.griddata` (cubic) or grid-cell
   averaging.

5. **Save**: Split into valence bands (`evb`) and conduction bands (`ecb`),
   save as HDF5 with 1D kx/ky axes.

## Step 2: Experimental Preprocessing

**Input**: Raw ARPES HDF5 data (e.g., `dataCliu.h5`)

**Output**: `exp_preprocessed.h5` with `axes/E`, `axes/kx`, `axes/ky`, `intensity/V`

**Process**:

1. **Build axes**: Construct energy and angle axes from config parameters
   (`energy_start`, `energy_delta`, `kx_angle_start`, etc.).

2. **Angle calibration**: Interactive Gamma point centering. The user
   drags crosshairs to center the Gamma point in angle space. The
   resulting shifts (`kx_shift`, `ky_shift`) are saved to the config.

3. **Angle-to-momentum conversion**: Convert angle coordinates to momentum
   space using the free-electron final state model:

   ```
   K = √(2m·E_kin) / ħ  ·  sin(angle)
   ```

   where `E_kin = work_function - E_binding`.

4. **C6 rotation expansion**: The converted momentum data is replicated 6
   times with 60° rotations to fill the full BZ.

5. **Grid calibration**: Interactive Gamma centering in momentum space.
   The resulting shifts (`kx_grid_shift`, `ky_grid_shift`) are saved.

6. **KD-interpolation**: KD-tree based interpolation with local averaging
   (`radius=0.05`) for each energy layer. Stride-based interpolation
   accelerates processing.

7. **Save**: Energy axis is automatically sorted to increasing order.
   Intensity data is saved as `(E, kx, ky)`.

### Preprocessing Options

| Option | Description |
|--------|-------------|
| `sort_axes` | Sort all axes to increasing order, flipping data accordingly |
| `sign_correct` | Apply notebook-style sign correction to momentum coordinates |
| `auto_grid` | Auto-determine output grid size from KX/KY shape |
| `flip` (per axis) | Quick axis label flip (no data flip) |

## Step 3: HSP Calibration

**Input**: Preprocessed ARPES data, lattice parameters

**Output**: `calibration.hsps` in config file

**Auto Mode** (default in `find_hsps_robust`):

1. Compute theoretical K and M positions from lattice parameters via
   `lattice_to_reciprocal()`.

2. For each angle θ ∈ [0, 60°), reflect the ARPES data across two
   perpendicular lines through Γ (the K-K and M-M directions).

3. The angle with maximum mirror symmetry correlation is the BZ
   rotation angle.

4. Build 6 K points at |Γ-K| distance and 6 M points at |Γ-M| distance
   using C6 rotation around Γ.

**Manual Mode** (`scripts/calibrate_hsps.py`):

- Interactive GUI with a hexagon overlay, circle, and orthogonal cross.
- User adjusts rotation and scale sliders to match the BZ pattern.
- Saves to `calibration.hsps.manual = true` in the config.

## Step 4: MRF Reconstruction

**Input**: `band_map.h5`, `exp_preprocessed.h5`, `pimre_config.yaml`

**Output**: Reconstructed bands, BSFI curves, path plots

**Process**:

1. **Data loading**: Load preprocessed experimental data and DFT band map.
   The experimental intensity is smoothed with a 3D Gaussian filter.

2. **Affine transform**: Compute a 2×2 affine transform T that maps DFT
   coordinates to experimental coordinates by matching Γ→K and Γ→M
   vectors in both spaces. DFT bands are mapped onto the experimental
   grid via `T_inv`.

3. **BSFI offset search**: Hierarchical optimization of the energy offset
   between DFT and experimental bands:

   - **Stage 1**: All bands shifted together, scoring via 2D correlation
     between dE/dk and dI/dk gradients.
   - **Stage 2**: Each band fine-tuned within ±0.05 eV of the shared best.

   BSFI = 0.6 × correlation + 0.3 × intensity_ratio + 0.1 × SNR

4. **MRF optimization**: For each band, the MRF model finds the energy
   at each k-point that maximizes the log-likelihood:

   ```
   log P = log I(kx, ky, E) - Σ (E - E_neighbor)² / (2η²)
   ```

   The optimization uses a checkerboard (white/black) node update pattern
   with PyTorch tensor operations for GPU acceleration.

5. **Symmetrization**: Mirror symmetrization around kx=0 and ky=0 axes.

6. **Output**: Band path plots along Γ-M-K-Γ, K-Γ-K, and M-Γ-M directions.