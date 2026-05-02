"""Lambda selection via GCV, REML, and AIC.

The three criteria all evaluate the same fitted system and rely on the same
diagnostics (``rss``, ``edf``, ``log|A|``). A scalar minimiser searches over
``log10(lambda)`` to keep the optimiser well-conditioned across the range
``[1e-6, 1e10]``.

For 2D problems with two independent lambdas, a Powell search runs over
``(log10(lam_row), log10(lam_col))``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize, minimize_scalar

from ._core import solve

LambdaMethod = Literal["gcv", "reml", "aic"]
_LOG_LAM_BOUNDS = (-6.0, 10.0)


def _criterion(
    method: LambdaMethod,
    n: int,
    null_dim: int,
    rss: float,
    rss_pen: float,
    edf: float,
    log_det_A: float,
    lam: float,
) -> float:
    eps = 1e-300
    if method == "gcv":
        denom = max(n - edf, 1e-10)
        return n * rss / denom**2
    if method == "aic":
        return n * np.log(max(rss / n, eps)) + 2.0 * edf
    if method == "reml":
        # Wood (2011): V_r ∝ (n-Mp) log(rss_pen) + log|A| - (n-Mp) log(lam)
        m = max(n - null_dim, 1)
        return m * np.log(max(rss_pen, eps)) + log_det_A - m * np.log(max(lam, eps))
    raise ValueError(f"unknown method {method!r}")


def select_lambda_1d(
    y: np.ndarray,
    weights: np.ndarray,
    penalty: sp.csc_matrix,
    null_dim: int,
    method: LambdaMethod,
) -> float:
    """Return the optimal scalar lambda for a 1D smoother."""
    n = y.shape[0]

    def objective(log_lam: float) -> float:
        lam = 10.0**log_lam
        res = solve(y, weights, penalty, lam)
        return _criterion(method, n, null_dim, res.rss, res.rss_penalised, res.edf, res.log_det_A, lam)

    result = minimize_scalar(objective, bounds=_LOG_LAM_BOUNDS, method="bounded", options={"xatol": 1e-3})
    return float(10.0**result.x)


def select_lambda_2d(
    y_vec: np.ndarray,
    weights_vec: np.ndarray,
    build_penalty: Callable[[float, float], sp.csc_matrix],
    null_dim: int,
    method: LambdaMethod,
    fixed: tuple[float | None, float | None] = (None, None),
) -> tuple[float, float]:
    """Return ``(lam_row, lam_col)`` minimising the chosen criterion.

    ``build_penalty(lam_row, lam_col)`` must return the full Kronecker-sum
    penalty matrix scaled by the two lambdas. Either coordinate can be held
    fixed via ``fixed``.
    """
    n = y_vec.shape[0]

    fixed_r, fixed_c = fixed

    def objective(x: np.ndarray) -> float:
        lam_r = fixed_r if fixed_r is not None else 10.0 ** x[0]
        lam_c = fixed_c if fixed_c is not None else 10.0 ** x[1]
        P = build_penalty(lam_r, lam_c)
        # We pass lam=1.0 because P already absorbs lam_r and lam_c.
        res = solve(y_vec, weights_vec, P, lam=1.0)
        # For REML's log(lam) term, use the geometric mean as an effective lam.
        eff_lam = float(np.sqrt(max(lam_r, 1e-300) * max(lam_c, 1e-300)))
        return _criterion(method, n, null_dim, res.rss, res.rss_penalised, res.edf, res.log_det_A, eff_lam)

    x0 = np.array([0.0 if fixed_r is None else 0.0, 0.0 if fixed_c is None else 0.0])
    bounds = [_LOG_LAM_BOUNDS, _LOG_LAM_BOUNDS]
    result = minimize(objective, x0, method="Powell", bounds=bounds, options={"xtol": 1e-3})
    lam_r = fixed_r if fixed_r is not None else float(10.0 ** result.x[0])
    lam_c = fixed_c if fixed_c is not None else float(10.0 ** result.x[1])
    return lam_r, lam_c
