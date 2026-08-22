# AGENTS.md

Scientific Python project (Python ≥3.11, uv-managed): reconstructs ARPES band structures via MRF optimization with DFT priors. Single-package repo, no CI.

## Commands

```bash
uv sync --dev                          # setup (.venv, Python 3.11)
uv run ruff check .                    # lint (E,F,W,I; line-length 120; E501 ignored)
uv run python scripts/run_mrf.py       # MRF+BSFI reconstruction (default config)
uv run python scripts/preprocess_exp.py [--skip-calib|--calib-only]
uv run python scripts/calibrate_hsps.py [--mode dft]
uv run python scripts/run_dft_processing.py --dft-csv <csv> --band-gap-file BAND_GAP --output test/band_map.h5
```

There is no test suite to run — see "Testing" below.

## Gotchas

- **`test/` is not a pytest suite.** It's a data/output sandbox (`.h5`, `.npy`, `.png`, JSON outputs land there, all gitignored) plus interactive Qt comparison tools (`compare_h5.py`, `compare_band_map.py`). Despite `docs/contributing.md` saying `uv run pytest`, no tests exist. Verify changes by running the affected pipeline stage and `ruff check`.
- **`pyproject.toml [project.scripts]` (`pimre-dft`, etc.) is broken** — it points to `pimre.cli`, which does not exist. Always invoke via `uv run python scripts/<script>.py`.
- **Config is machine-specific and tracked**: `configs/pimre_config.yaml` contains absolute data paths (`/home/dengxw/ARPES/...`) and locally modified calibration values. Don't "clean up" these changes or commit them casually; other YAMLs in `configs/` are per-sample variants (e.g. `rbtibi`).
- **GUI/calibration steps require a display** (PyQt5 + matplotlib Qt5Agg). On headless environments use `--skip-calib` so saved calibration values from config are reused.
- Scripts do `sys.path.insert(0, repo_root)` and import `pimre.*` directly — no installed package needed, but `uv run` from the repo root is expected.

## Architecture

Pipeline order: DFT CSV + raw ARPES HDF5 → preprocessing (`pipeline/preprocess.py`: angle→momentum, KD-tree or quadrant-symmetrization interpolation) + DFT band map (`pipeline/dft.py`) → HSP calibration (`kpath/symmetry.py`, mirror-registration rotation detection) → affine DFT→exp mapping + BSFI energy-offset search (`mrf/evaluation.py`) → MRF reconstruction (`mrf/model.py`, PyTorch checkerboard updates).

Key modules:
- `pimre/config.py` — single source of truth: `DEFAULTS` dict + `load_config()` deep-merge over user YAML.
- `pimre/pipeline/` — end-to-end functions called by thin CLI wrappers in `scripts/`.
- `pimre/gui/` — PyQt5 calibration widgets used interactively by scripts.

## Conventions

- New config keys: add to `DEFAULTS` in `pimre/config.py` AND document in `docs/configuration.md`; update `scripts/setup_config.py` (config-generator GUI) if user-facing.
- Design decisions live as spec files in `docs/superpowers/specs/YYYY-MM-DD-*.md` before/with implementation.
- Root-level `task_plan.md` / `findings.md` / `progress.md` are stale scratch planning files (gitignored) — ignore them.
