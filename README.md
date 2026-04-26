[![PyPI](https://img.shields.io/pypi/v/whsmooth?color=blue)](https://pypi.org/project/whsmooth/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

# whsmooth

**Whittaker-Henderson smoothing for actuarial graduation --- 1D, 2D, with automatic lambda selection.**

---

## What is whsmooth?

Whittaker-Henderson smoothing is the gold standard for graduating mortality tables and smoothing actuarial rate structures. The method has been used by actuaries since the 1920s: it minimizes a weighted sum of fidelity to the data and roughness of the graduated curve, controlled by a single smoothing parameter lambda.

Despite its centrality to actuarial practice, **no dedicated Python implementation exists**. R has `MortalitySmooth` (Camarda, 2012) and `ungroup` (Rizzi et al., 2015), but Python actuaries are left writing ad-hoc scripts or porting R code by hand.

**whsmooth** fills this gap with:

- **Sparse-matrix implementation** --- the Whittaker-Henderson system is banded; we exploit `scipy.sparse` throughout for O(n) solves instead of O(n^3).
- **Proper lambda selection** --- REML (restricted maximum likelihood), GCV (generalized cross-validation), and AIC, so you are not guessing your smoothing parameter.
- **1D and 2D smoothing** --- graduate a mortality vector by age, or smooth an entire age x calendar-year grid in one call.
- **Actuarial convenience wrappers** --- mortality graduation with Poisson likelihood and claim severity smoothing out of the box.

## Installation

```bash
pip install whsmooth
```

From source:

```bash
git clone https://github.com/CosmikArt/whsmooth.git
cd whsmooth
pip install -e .
```

## Quickstart

Smooth a crude mortality table and select lambda via GCV:

```python
import numpy as np
from whsmooth import WhittakerHenderson1D, LambdaSelector

# Crude qx values by age 30-89
ages = np.arange(30, 90)
np.random.seed(42)
qx_true = 0.0005 * np.exp(0.075 * (ages - 30))
qx_crude = qx_true * np.exp(np.random.normal(0, 0.15, size=len(ages)))

# Select lambda via GCV, then smooth
selector = LambdaSelector(y=qx_crude)
lam_opt = selector.select(method="gcv")

wh = WhittakerHenderson1D()
wh.fit(y=qx_crude, lam=lam_opt, d=2)
qx_smooth = wh.fitted_values

# Plot raw vs graduated
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(ages, qx_crude, s=18, alpha=0.6, label="Crude qx")
ax.plot(ages, qx_smooth, color="crimson", lw=2, label="Graduated qx")
ax.plot(ages, qx_true, color="grey", ls="--", lw=1, label="True qx")
ax.set_xlabel("Age")
ax.set_ylabel("qx")
ax.set_title("Mortality graduation with Whittaker-Henderson (GCV)")
ax.legend()
plt.tight_layout()
plt.show()
```

## Features

| Module | Description |
|---|---|
| `smooth1d` | 1D Whittaker-Henderson smoothing with any difference order |
| `smooth2d` | 2D smoothing for age x calendar-year grids (Kronecker-product penalties) |
| `lambda_selection` | Automatic smoothing-parameter selection via REML, GCV, AIC, or manual override |
| `penalties` | Difference penalty matrices of any order, composite penalties for mixed objectives |
| `diagnostics` | Smoothness-vs-fit tradeoff plot, residual analysis, effective degrees of freedom |

## How it works

The Whittaker-Henderson smoother finds the vector **a** that minimizes:

```
S(a; lambda, d) = sum_i w_i (y_i - a_i)^2  +  lambda * ||D_d a||^2
```

where **D_d** is the d-th order difference matrix and lambda controls the smoothness-fidelity tradeoff. The solution is a linear system with a banded coefficient matrix, solved efficiently via sparse Cholesky factorization.

For 2D problems (age x year grids), the penalty extends to Kronecker products of row and column difference matrices, following Currie, Durban & Eilers (2004).

Lambda selection is critical. Rather than forcing the user to guess, whsmooth provides three principled criteria:

- **GCV** (generalized cross-validation) --- minimizes leave-one-out prediction error without explicit cross-validation.
- **REML** (restricted maximum likelihood) --- treats lambda as a variance-component parameter; generally preferred for mortality graduation.
- **AIC** (Akaike information criterion) --- balances fit and complexity via effective degrees of freedom.

## References

- Whittaker, E. T. (1922). "On a New Method of Graduation." *Proceedings of the Edinburgh Mathematical Society*, 41, 63--75.
- Henderson, R. (1924). "A New Method of Graduation." *Transactions of the Actuarial Society of America*, 25, 29--40.
- Eilers, P. H. C. & Marx, B. D. (1996). "Flexible Smoothing with B-splines and Penalties." *Statistical Science*, 11(2), 89--121.
- Currie, I. D., Durban, M. & Eilers, P. H. C. (2004). "Smoothing and Forecasting Mortality Rates." *Statistical Modelling*, 4(4), 279--298.
- Camarda, C. G. (2012). "MortalitySmooth: An R Package for Smoothing Poisson Counts with P-Splines." *Journal of Statistical Software*, 50(1), 1--24.

## Contributing

Contributions are welcome. Please open an issue first to discuss what you would like to change. PRs should include tests and follow the existing code style.

```bash
git clone https://github.com/CosmikArt/whsmooth.git
cd whsmooth
pip install -e ".[dev]"
pytest
```

## Author

**Isaac López**

## License

[MIT](LICENSE)
