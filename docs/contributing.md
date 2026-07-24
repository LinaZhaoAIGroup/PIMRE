# Contributing

## Development Setup

```bash
git clone https://github.com/LinaZhaoAIGroup/PIMRE.git
cd PIMRE
uv sync --dev
```

## Code Style

PIMRE uses [ruff](https://github.com/astral-sh/ruff) for linting and
formatting.

```bash
uv run ruff check .
```

Configuration is in `pyproject.toml`:

- Line length: 120 characters
- Rules: E, F, W, I (pycodestyle, pyflakes, warnings, isort)

## Project Structure

```
pimre/           Core library
scripts/         CLI entry points
configs/         YAML configuration files
test/            Test data and comparison scripts
docs/            Documentation
```

## Adding New Features

1. Implement core logic in the appropriate `pimre/` subpackage
2. Add a thin CLI wrapper in `scripts/` if needed
3. Update `pimre/config.py` defaults for new config options
4. Update `docs/configuration.md` if new config fields are added
5. Run `uv run ruff check .` before committing

## Running Tests

```bash
uv run pytest
```

## Pull Request Guidelines

- Keep PRs focused on a single feature or fix
- Update documentation for any user-facing changes
- Ensure all imports are sorted and unused imports are removed
- Run the full pipeline to verify changes don't break existing workflows

## License

By contributing, you agree that your contributions will be licensed
under the same LGPL v2.1 license as the project.