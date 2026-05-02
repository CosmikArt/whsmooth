"""Public diagnostic helpers: scoring functions and EDF for a fixed lambda."""

from __future__ import annotations

import numpy as np

from ._core import solve
from ._lambda import _criterion
from ._penalties import penalty_matrix


def _prep(y: np.ndarray, weights: np.ndarray | None, order: int):
    y_arr = np.asarray(y, dtype=float).ravel()
    n = y_arr.size
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).ravel()
    P = penalty_matrix(n, order)
    return y_arr, w, P, n


def gcv_score(y: np.ndarray, weights: np.ndarray | None, lam: float, order: int = 2) -> float:
    """GCV criterion at the given lambda."""
    y_arr, w, P, n = _prep(y, weights, order)
    res = solve(y_arr, w, P, lam)
    return _criterion("gcv", n, order, res.rss, res.rss_penalised, res.edf, res.log_det_A, lam)


def reml_score(y: np.ndarray, weights: np.ndarray | None, lam: float, order: int = 2) -> float:
    """REML criterion at the given lambda (Wood 2011, up to additive const)."""
    y_arr, w, P, n = _prep(y, weights, order)
    res = solve(y_arr, w, P, lam)
    return _criterion("reml", n, order, res.rss, res.rss_penalised, res.edf, res.log_det_A, lam)


def aic_score(y: np.ndarray, weights: np.ndarray | None, lam: float, order: int = 2) -> float:
    """AIC criterion at the given lambda."""
    y_arr, w, P, n = _prep(y, weights, order)
    res = solve(y_arr, w, P, lam)
    return _criterion("aic", n, order, res.rss, res.rss_penalised, res.edf, res.log_det_A, lam)


def edf(y: np.ndarray, weights: np.ndarray | None, lam: float, order: int = 2) -> float:
    """Effective degrees of freedom at the given lambda."""
    y_arr, w, P, _ = _prep(y, weights, order)
    return float(solve(y_arr, w, P, lam).edf)
