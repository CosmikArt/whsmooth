# Contributing to whsmooth

Thanks for your interest in contributing.

## Development setup

```bash
git clone https://github.com/CosmikArt/whsmooth.git
cd whsmooth
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Workflow

1. Open an issue describing the change before starting work on anything substantive.
2. Fork and create a feature branch: `git checkout -b feat/your-feature`.
3. Write tests for new behavior. PRs without tests will not be merged.
4. Run the full check suite locally:
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   pytest -xvs --cov=whsmooth
   ```
5. Commit in atomic units with imperative messages: `feat(api): add weighted 2D smoothing`.
6. Open a PR against `main`. Link the issue.

## Numerical correctness

This library is used for actuarial pricing decisions. Numerical regressions are not acceptable. New algorithms must include:

- A test against a known closed-form result, or
- A test against a reference implementation (R's `MortalitySmooth`, `ungroup`, or hand-derived).

## Code style

- Ruff handles linting and formatting; CI enforces it.
- Type hints required on public API. `py.typed` shipped.
- Docstrings: NumPy style.

## Releasing (maintainers)

1. Bump version in `pyproject.toml` and `CITATION.cff`.
2. Update `CHANGELOG.md`.
3. Tag: `git tag v0.X.Y && git push --tags`.
4. Create GitHub Release --- `publish.yml` ships to PyPI automatically.
