[![tests](https://github.com/CosmikArt/whsmooth/actions/workflows/test.yml/badge.svg)](https://github.com/CosmikArt/whsmooth/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/whsmooth?color=blue)](https://pypi.org/project/whsmooth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

# whsmooth

**Whittaker-Henderson smoothing for P&C ratemaking and actuarial graduation --- 1D, 2D, automatic lambda selection.**

---

## What is whsmooth?

Whittaker-Henderson smoothing is a workhorse technique across actuarial practice: smoothing claim severity by exposure cell, graduating loss development factors, stabilizing rate relativities, and graduating mortality tables. The method has been used since the 1920s; it minimizes a weighted sum of fidelity to the data and roughness of the smoothed curve, controlled by a single parameter lambda.

Despite its centrality, **no dedicated Python implementation exists**. R has `MortalitySmooth` and `ungroup`, both life-actuarial. Python practitioners --- especially in P&C --- are left writing ad-hoc scripts.

**whsmooth** fills this gap with:

- **P&C-first design** --- built for severity, frequency, LDF, and rate relativity smoothing, with mortality graduation as a natural special case.
- **Sparse-matrix solver** --- banded penalty system solved with `scipy.sparse` for O(n) performance.
- **Principled lambda selection** --- REML, GCV, and AIC, so you stop guessing your smoothing parameter.
- **1D and 2D** --- smooth a vector by vehicle age, or an entire age × territory grid in one call.
- **Sklearn-style API** --- `fit(y).fitted_`, no surprises.

## Installation

```bash
pip install whsmooth

# With plotting helpers
pip install "whsmooth[plot]"

# With pandas integration
pip install "whsmooth[pandas]"
```

## Quickstart --- Smoothing claim severity by vehicle age

```python
import numpy as np
from whsmooth import WH1D

# Observed mean severity by vehicle age (0-25), with claim counts as weights
ages = np.arange(0, 26)
severity_obs = np.array([
    8050, 8120, 8210, 8290, 8390, 8520, 8680, 8850, 9050, 9270,
    9510, 9770, 10050, 10350, 10670, 11010, 11380, 11760, 12300, 12800,
    13100, 14200, 13400, 15100, 14000, 16500,
])
counts = np.array([
    5200, 4500, 3900, 3400, 2950, 2550, 2200, 1900, 1640, 1420,
    1230, 1060, 920, 790, 685, 590, 510, 440, 380, 330,
    285, 245, 210, 180, 155, 135,
])

# Fit with GCV-selected lambda; weights = exposure
wh = WH1D(lam='gcv', order=2).fit(severity_obs, weights=counts)

print(f"Selected lambda: {wh.lambda_:.2f}")
print(f"Effective degrees of freedom: {wh.edf_:.2f}")
smoothed = wh.fitted_
```

The high-credibility cells (low ages, large counts) are tracked closely; the sparse tail (older vehicles) is stabilized toward the underlying trend instead of chasing noise.

## Quickstart --- 2D smoothing on age × territory

```python
from whsmooth import WH2D

# Pure premium grid: 10 age bands × 8 territories
pure_premium = ...  # shape (10, 8)
exposure = ...      # shape (10, 8)

wh2 = WH2D(lam=('gcv', 'gcv'), order=(2, 2)).fit(pure_premium, weights=exposure)
pp_smooth = wh2.fitted_
```

## Other use cases

- **LDF smoothing**: stabilize loss development factors before extrapolation.
- **Frequency relativities**: smooth claim frequency by exposure class.
- **Mortality graduation** (life): graduate qx by age --- same algorithm, drop in your `qx_crude` and `exposure`.

```python
# Mortality graduation example
wh = WH1D(lam='reml', order=2).fit(qx_crude, weights=exposure)
qx_grad = wh.fitted_
```

## Features

| Module | Description |
| --- | --- |
| `WhittakerHenderson1D` / `WH1D` | 1D smoothing, any difference order, weighted |
| `WhittakerHenderson2D` / `WH2D` | 2D smoothing on grids (Kronecker penalties) |
| `lam='gcv'` / `'reml'` / `'aic'` | Automatic lambda selection |
| `whsmooth.diagnostics` | `gcv_score`, `reml_score`, `aic_score`, `edf` |
| `whsmooth._penalties` | Difference penalty matrices, composite penalties |

## How it works

The Whittaker-Henderson smoother solves:

```
min_a   sum_i w_i (y_i - a_i)^2  +  lambda * ||D_d a||^2
```

where **D_d** is the d-th order difference matrix and lambda controls the smoothness-fidelity tradeoff. The system has a banded coefficient matrix and is solved via sparse Cholesky factorization --- O(n) instead of O(n^3).

For 2D problems, the penalty extends to Kronecker products of row and column difference matrices, following Currie, Durban & Eilers (2004).

## References

### P&C / general

- Werner, G. & Modlin, C. (2016). *Basic Ratemaking* (5th ed.). Casualty Actuarial Society.
- Verrall, R. J. (1996). "Claims Reserving and Generalised Additive Models." *Insurance: Mathematics and Economics*, 19(1), 31-43.
- Eilers, P. H. C. & Marx, B. D. (1996). "Flexible Smoothing with B-splines and Penalties." *Statistical Science*, 11(2), 89-121.

### Original

- Whittaker, E. T. (1922). "On a New Method of Graduation." *Proc. Edinburgh Math. Soc.*, 41, 63-75.
- Henderson, R. (1924). "A New Method of Graduation." *Trans. Actuarial Society of America*, 25, 29-40.

### Life / mortality

- Currie, I. D., Durban, M. & Eilers, P. H. C. (2004). "Smoothing and Forecasting Mortality Rates." *Statistical Modelling*, 4(4), 279-298.
- Camarda, C. G. (2012). "MortalitySmooth: An R Package for Smoothing Poisson Counts with P-Splines." *J. Statistical Software*, 50(1), 1-24.

## Part of the actuarial Python ecosystem

`whsmooth` is part of a 6-library P&C actuarial stack:

- [`actudist`](https://pypi.org/project/actudist/) --- severity & frequency distributions
- [`burncost`](https://pypi.org/project/burncost/) --- burning cost analysis
- [`actuarcredibility`](https://pypi.org/project/actuarcredibility/) --- credibility methods
- **`whsmooth`** --- smoothing (this library)
- `pyratemaking` --- rating tables (coming)
- `pyinsurancerating` --- rating engine (coming)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). PRs welcome --- open an issue first to discuss substantive changes.

## Citation

If you use whsmooth in academic work, see [CITATION.cff](CITATION.cff) or:

> López, I. (2026). *whsmooth: Whittaker-Henderson smoothing for P&C ratemaking in Python* (v0.1.0). https://github.com/CosmikArt/whsmooth

## Author

**Isaac López**

## License

[MIT](LICENSE)
