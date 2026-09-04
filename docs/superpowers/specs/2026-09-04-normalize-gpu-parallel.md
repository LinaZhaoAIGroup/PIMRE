# 2026-09-04 — dataCliu normalization (config-controlled) + GPU/parallel acceleration

Branch `feature/datacliu-normalize-gpu`, merged to `main` after validation.

## Motivation

- The default experimental input `dataCliu.h5` carries raw detector counts
  (min 1, max 7640) while the reference preprocessed product
  `HPES_preprocessed_new.h5` is normalized to [0, 1] (global max = 1).
  Raw counts as input should be normalized by default, switchable in config.
- The pipeline was slow: KD-tree interpolation ≈ 17 s per energy layer
  (≈ 206 s for the 12 computed layers of the dataCliu chain), MRF ≈ 1.0 s
  per epoch on CPU (≈ 9 min for 100 epochs × 5 bands). Rotation expansion
  measured 162 ms total — already negligible, no acceleration needed.

## Changes

### 1. Normalization (`preprocessing.normalize`, default `true`)

In `pipeline/preprocess.py::compute_grid`, raw counts are divided by the
global maximum right after loading (before rotation/interpolation), matching
the HPES convention. All downstream stages (alignment, BSFI, MRF) therefore
run on normalized data. Off switch: `preprocessing.normalize: false`.

Scale-invariance argument (why results don't move): every BSFI metric is a
ratio or correlation, and the MRF node update maximizes
`logI[node, e] - Σ(E_e - E_neighbor)²`, where scaling I by c only adds a
constant `log c` across e — the argmax is unchanged.

### 2. KDInterp vectorization + memory fix (`experiment/calibration.py`)

The per-point Python neighborhood loop (25 200 iterations) is replaced by
chunked query + chunked `bincount` accumulation (`_neighborhood_sums`).
Verified **bit-identical** against the original loop output on a dataCliu
layer (max |Δ| = 0.0). The quadrant smoothing loop is vectorized the same
way.

Memory mattered as much as speed: scipy's `query_ball_point` returns lists
of Python ints (~450 MB for 25 200 points with ~500 neighbors each), and a
global flat-index array plus its fancy-index temporaries held ~1.7 GB live
per worker — a 12-process pool exhausted the ~12 GB of free memory and the
OOM killer produced `BrokenProcessPool`. Chunked query+accumulate caps the
worker footprint at ~0.3 GB (measured peak 702 MB including ~420 MB of
inherited parent data) and is **bit-identical** to the original loop.

### 3. Layer-level process pool (`preprocessing.workers`, default `0` = auto)

The computed layers (every `stride`-th) are independent; with `workers != 1`
`preprocess_full` distributes them over a `ProcessPoolExecutor`. `0`/negative
= all CPU cores (capped at the layer count), `1` = serial, N = N processes.
The per-layer cubic `griddata` (CloughTocher, 2.4 s) and radius query
(4.9 s) are single-threaded, so layer-level parallelism is the effective
lever: **205 s (serial, old code) → 27 s wall** with 12 workers on the
20-core workstation; serial and pool outputs verified identical.

### 4. MRF device (`mrf.device`, default `"auto"`)

`MrfRec` accepts `device="auto"|"cpu"|"cuda"`; the checkerboard `iter_para`
tensors are allocated on it and copied back at the end. Measured on the
dataCliu grid (698×698, 109 energies, RTX 2070 SUPER):
CPU 1039 ms/epoch → GPU 92–98 ms/epoch (≈ 10.7×), with an identical initial
state the GPU and CPU results are **identical** (0 differing nodes over 20
epochs). Rotation stays on CPU/NumPy — it costs 162 ms for all 5 copies.

## Config / GUI

- `pimre/config.py` DEFAULTS: `preprocessing.normalize: true`,
  `preprocessing.workers: 0`, `mrf.device: "auto"`.
- `scripts/setup_config.py`: Normalize checkbox + Workers field in the
  Output (preprocessing) group; Device combo in the MRF tab; save/load sync.
- `docs/configuration.md`: rows for all three keys.
- `final_parameters.json` now records `"device"`.

## Validation

- Vectorized/chunked KDInterp vs original loop: bit-identical on a real
  layer; `workers: 1` (serial) and `workers: 2` pool branches produce
  identical output on synthetic data.
- GPU vs CPU MRF (same init): identical indices after 20 epochs; 1039 →
  92–98 ms/epoch (≈ 10.7×, RTX 2070 SUPER).
- Normalization commutes with the linear interpolation: `new × 7640` vs the
  old raw-count output differs by ≤ 6×10⁻⁹ relative (CloughTocher gradient
  conditioning), far below any physical significance.
- Full dataCliu chain re-run with normalization + `workers: 0` + GPU
  (`test/datacliu_gpu100/`) vs the raw-count CPU baseline
  (`test/datacliu100/`): identical T, identical offsets (0.55/0.55/0.65/
  0.65/0.55 eV), identical BSFI scores to 4 decimals; recon bands identical
  for 3 of 5 bands, the other two differ by ≤ one energy step (max 0.0075
  eV) from float rounding at argmax ties.
- Run time (MRF stage, 100 epochs × 5 bands): ~9 min CPU → ~85 s total
  pipeline incl. BSFI search and plots.
- `scripts/setup_config.py`: offscreen smoke test — device/normalize/
  workers round-trip through save/load; band table now also accepts
  `from_vbm` entries (`vbm:N`), which previously crashed the GUI on load.
