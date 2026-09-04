# 2026-09-03 Bug-fix batch: BSFI SNR scale, band selection, dead config keys

## Context

A full code review surfaced issues in three groups: (1) BSFI scoring scale
inconsistency, (2) fragile DFT band selection semantics, (3) config keys that
are editable in the GUI / present in YAML but never read by the pipeline.

## Decisions

### 1. BSFI SNR squashed to [0, 1)

`compute_bsfi_2d` mixes metrics on different scales: correlation, intensity
ratio and ridge score live in [0, 1] but the SNR (mean/std) is unbounded, so
`w_snr = 0.1` could dominate the weighted sum. SNR is now squashed via
`s / (s + 1)` (monotonic, 0.5 at s=1). BSFI values change scale; the argmax
over the offset grid is preserved for well-separated optima. Weights keep
their documented relative meaning.

### 2. Band selection: `from_vbm` entries

`mrf.bands` entries used absolute positions in the *stacked* band array
(descending energy, after `drop_top_bands`), which silently select the wrong
bands whenever the drop count does not exactly match the DFT conduction-band
count. Entries now accept `from_vbm: k` (0 = VBM, 1 = next band down),
resolved against the conduction-band count independent of the drop count;
plain `index` entries keep working. Out-of-range selections raise instead of
failing silently downstream.

### 3. Dead config keys made effective or removed

- `dft.csv_file`, `dft.fermi_file`: now used as defaults by
  `run_dft_pipeline` / `run_dft_processing.py` when the CLI arguments are
  omitted (resolved against `dft.path`).
- `dft.output_grid`: now passed to `interpolate_to_grid` (griddata method).
- `bz.n_rotations`: now passed to `expand_bz` (was hard-coded 6).
- `mrf.eta`: read by `run_mrf_pipeline` (was hard-coded 0.12); added to
  `DEFAULTS`.
- `dft.drop_top_bands`: added to `DEFAULTS` and documented.
- `mrf.offset_mode`: GUI display removed (pipeline has only the shared mode).
- `sign_correct` (in `pimre_config.yaml`): left untouched per the AGENTS.md
  config caution — it is inert; do not reintroduce it.
- `preprocessing.n_rotations`: validated to 1..6 (60° step is fixed by the
  C6 symmetry; larger values would duplicate rotations).

### 4. Robustness

- `sym_band`: an axis with no strictly-positive value skips the mirror
  instead of crashing on `np.min` of an empty index array.
- `MrfRec.initializeBand`: NaNs from missing DFT coverage initialize at
  mid-window instead of silently indexing energy bin 0.
- `iter_para` logP history now counts each neighbor pair once (same
  convention as `getLogP`), so history entries are comparable to the initial
  value.
- `parse_kpoints` tolerates multi-number point-count lines.
- `calibrate_hsps.py` loads legacy .mat band maps via
  `pimre.dft.reader.load_band_map_any` instead of importing the repo's
  `test/` package (which can be shadowed by the Python stdlib `test`).

## Verification

`uv run ruff check .` passes; an end-to-end run on synthetic data
(DFT CSV + BAND_GAP → band_map.h5 → synthetic ARPES HDF5 → preprocess →
MRF reconstruction with `from_vbm` bands) reproduces the expected band
ordering and completes without warnings.

## Addendum (same day): reconstruction-quality fixes

Visual inspection of the RbTiBi path plots showed reconstructed curves
missing the measured intensity ridges. Root causes established with
diagnostic overlays (test/diag_*.png):

1. **Alignment mode.** The experimental and theoretical momentum scales
   differ; the theory grid must be stretched so that its K and M
   high-symmetry points land on the experimental ones. The reference
   implementation (mrf_bsfi_pipeline.py) solves the 2x2 system exactly,
   `T = S_exp @ inv(S_dft)` (anisotropic scaling + rotation).
   `mrf.alignment: "hsp"` (default) does this; `"gamma"` is an identity
   mapping that is only valid when both axes already share the same
   absolute calibration. (An earlier draft of this addendum dismissed the
   HSP-derived scale of ~1.37 as spurious — that was wrong; the scale is
   the genuine theory-to-experiment stretch and is also consistent with
   the size of the observed hexagonal pattern.)
2. **Shared offset is wrong for metallic systems.** The top "valence"
   bands cross E_F (band full[32] is already above E_F at r = 0.05 Å⁻¹),
   so bands have very different shapes and one shared shift cannot align
   them; the per-band score curves peaked anywhere between −0.35 and
   +0.65 eV. `mrf.offset_mode: "per_band"` (default) lets each band take
   its own score maximum — matching the notebook workflow, which tuned
   per-band offsets of 0.62–0.8 eV by hand. `"shared"` remains available.
3. **Occupied-state constraint.** ARPES measures only states at/below
   E_F. In the additive convention E0 = E_dft + offset, the offset search
   could previously align the EMPTY segments (E_dft > 0) of metallic bands
   to the measured intensity. The search and the reconstruction now
   restrict each band to its occupied part (E0 >= 0 from E_dft <= 0);
   offsets with no occupied overlap score zero with an explicit warning.

Both alignment and offset mode are user-facing and were added to the
setup_config GUI and docs/configuration.md. `configs/pimre_config.yaml`
sets alignment: hsp and offset_mode: per_band.
