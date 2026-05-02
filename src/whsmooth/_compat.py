"""Deprecated shims for the 0.0.x public API.

These objects are retained so existing code keeps working through 0.1.x,
but emit ``DeprecationWarning`` pointing to the new sklearn-style API.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from ._lambda import select_lambda_1d
from ._penalties import penalty_matrix


class LambdaSelector:
    """Deprecated. Use ``WhittakerHenderson1D(lam='gcv', ...)`` instead."""

    def __init__(
        self,
        y: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        d: int = 2,
    ) -> None:
        warnings.warn(
            "LambdaSelector is deprecated; pass lam='gcv'/'reml'/'aic' "
            "directly to WhittakerHenderson1D, or call "
            "whsmooth.diagnostics.gcv_score / reml_score / aic_score.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._y = np.asarray(y, dtype=float) if y is not None else None
        self._weights = np.asarray(weights, dtype=float) if weights is not None else None
        self._d = int(d)

    def select(self, method: Literal["gcv", "reml", "aic"] = "gcv") -> float:
        if self._y is None:
            raise ValueError("LambdaSelector was constructed without y")
        n = self._y.size
        w = np.ones(n) if self._weights is None else self._weights
        P = penalty_matrix(n, self._d)
        return select_lambda_1d(self._y, w, P, self._d, method)
