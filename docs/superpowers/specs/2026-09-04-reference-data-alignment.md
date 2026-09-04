# 2026-09-04 — Reference-data reproduction: HSP conventions, .mat band maps, full-band alignment

## Goal

Reproduce the pre-refactor reference results (`/home/dengxw/ARPES/mrf_bsfi_pipeline.py`,
outputs in `/home/dengxw/ARPES/tmp_mrf_output/`) from the PIMRE pipeline, using the
same inputs the reference uses: `Data/HPES_preprocessed_new.h5` (experimental,
875×875 regular momentum grid × 110 energies) and `Data/band_map.mat` (legacy DFT
band map, 37 conduction + 121 valence bands on a 101×101 grid). Target: five
near-E_F bands present across the whole Γ-M-K-Γ path, as in
`tmp_mrf_output/final_GMKG_path.png`.

## Root causes found

1. **`dft_KM` component convention bug (PIMRE-only).** The reference indexes
   `KP[0, 1]` (0.2887) against `kx` and `KP[0, 0]` (0.5) against `ky`, placing the
   DFT K point at 60° and M at 30° — the true hexagonal-BZ vertices/midpoints for
   grids whose K vertices sit on the kx axis. PIMRE's port used the swapped order,
   placing "K" at 30° (an M direction — geometrically impossible for a hexagonal
   BZ) and "M" at 60°. Every DFT→exp alignment was therefore rotated by −30°.
   The dataCliu-path manual calibration `dft_hsps.rotation_angle: 30` in the
   tracked config was compensating exactly this bug.

2. **Exp-side M-ring mirror.** `build_hsps_from_registration` placed `M_i` at
   `θ + 30 + 60i` (M counter-clockwise from K), while `Get_G_M_K` and the
   reference pair `M_i` at `θ − 30 + 60i` (clockwise). Combined with (1) the
   K/M pairing was mirror-inconsistent; with (1) fixed it must be K−30° on both
   sides so the affine transform stays reflection-free.

3. **30° registration ambiguity.** Hexagonal mirror axes are 30°-periodic, so
   automatic mirror registration cannot distinguish the two branches (it picked
   −30.6°-equivalent orientation for the HPES map, which is already
   lattice-aligned). Resolution for this data: manual `calibration.hsps` with
   `rotation_angle: 60.0` (K-ring at 60°+60i, (K0, M0) = (60°, 30°)), which
   reproduces the reference `T = [[1.2251, 0.0141], [0.0027, 1.2470]]` exactly
   (verified numerically to 1e-4).

4. **Occupied-state mask.** The reference aligns FULL bands (no E_F masking);
   the earlier PIMRE default masked everything with `E_dft > 0`, which empties
   large parts of the metallic near-E_F bands. Now config-switchable
   (`mrf.occupied_only`, default `true`; `false` = reference behaviour).

5. **Offset search.** The reference uses a hierarchical search (shared coarse
   scan ±1.0 eV step 0.1, then per-band fine-tune ±0.05 eV) to avoid band-order
   crossing. Implemented as `offset_mode: hierarchical` (with
   `bsfi.fine_tune_range`), next to the existing `per_band`/`shared`.

## Changes

- `pimre/kpath/symmetry.py`: `dft_KM` component swap (K→60°, M→30°) with
  explanatory docstring; `select_hsps_by_coverage` docstring updated (M = K − 30°).
- `pimre/kpath/registration.py`: M-ring at `θ − 30 + 60i`.
- `pimre/pipeline/mrf.py`: `load_band_map_any` (legacy `.mat` band maps with
  `drop_top_bands`), `mrf.occupied_only` switch, `offset_mode: hierarchical`
  (stage 2 per-band fine-tune), corrected HSP angle labels
  (`K at θ+60i`, `M at θ−30+60i`).
- `pimre/dft/reader.py`: `load_band_map_mat`/`load_band_map_any` accept
  `drop_top_bands`.
- `scripts/setup_config.py`: offset-mode combo gains `hierarchical`; new
  occupied-only combo; sync/save/load wired.
- `docs/configuration.md`: all of the above.

## Validation

- Reference recipe reproduced numerically: `T` identical to
  `tmp_mrf_output/final_parameters.json` (`[[1.225117, 0.014146],
  [0.002684, 1.247038]]`, rotation 0.13°, scales 1.225/1.247).
- 10-epoch run on `Data/` inputs (`test/config_hpes.yaml`, offsets found
  0.55–0.65 eV vs reference 0.57–0.65 eV) yields five bands across Γ-M-K-Γ
  matching `tmp_mrf_output/final_GMKG_path.png` (side-by-side:
  `test/hpes/cmp_ref_vs_new_GMKG.png`).
- 100-epoch run (notebook `4.mrf.ipynb` equivalent) as final artifact.

## Caveats

- The dataCliu path (`configs/pimre_config.yaml` → raw preprocessing +
  `dataCliu.h5`) was tuned under the old (wrong) conventions; its
  `dft_hsps.rotation_angle: 30` now over-rotates by 30° and the exp-side
  pairing flips mirror. If that path is reused, re-tune the two manual
  calibrations (or set `dft_hsps.rotation_angle: 0`). The superseding
  `Data/` path needs no manual DFT calibration at all.
- Reference batch script snaps the DFT prior without MRF optimization;
  the notebook runs `iter_para(100)`. PIMRE covers both via `num_epochs`.
