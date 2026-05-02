# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-05-02

### Added

- P&C-first README with severity-by-vehicle-age and 2D age × territory examples.
- `WH1D` and `WH2D` aliases for the 1D/2D classes.
- Sklearn-style API: `WH1D(lam='gcv', order=2).fit(y).fitted_`.
- `LambdaSelector` replaced by `lam=` argument accepting `'gcv'`, `'reml'`, `'aic'`, or float.
- Real numerical test suite (1D, 2D, P&C use cases).
- CI matrix on Python 3.10/3.11/3.12/3.13 across Ubuntu and macOS.
- Trusted-publishing workflow for PyPI releases.
- `CITATION.cff`, `CONTRIBUTING.md`.
- P&C references (Werner & Modlin, Verrall) added.

### Changed

- `pandas` moved from core to optional `[pandas]` extra.
- Parameter `d` → `order` (alias `d` retained with `DeprecationWarning`).
- Fitted attributes renamed to sklearn convention: `fitted_`, `lambda_`, `edf_`.
- Version bumped to 0.1.0; status remains alpha.

### Deprecated

- `WhittakerHenderson1D(...).fit(y, lam=..., d=...)` legacy signature emits
  `DeprecationWarning` redirecting to `WhittakerHenderson1D(lam=..., order=...).fit(y)`.

## [0.0.1] - 2026-04-26

### Added

- Initial project scaffold.
- `WhittakerHenderson1D` class for 1D smoothing with any difference order.
- `WhittakerHenderson2D` class for 2D age x calendar-year grid smoothing.
- `LambdaSelector` with GCV, REML, and AIC criteria.
- `MortalityGraduator` convenience wrapper for mortality table graduation.
- `SeveritySmoother` convenience wrapper for claim severity smoothing.
- Project configuration with `pyproject.toml` (hatchling build).
- MIT license.
