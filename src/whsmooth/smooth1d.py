"""1D Whittaker-Henderson smoother."""

from __future__ import annotations

import warnings

import numpy as np

from ._core import solve
from ._lambda import LambdaMethod, select_lambda_1d
from ._penalties import penalty_matrix

LamArg = float | LambdaMethod


class WhittakerHenderson1D:
    """One-dimensional Whittaker-Henderson smoother.

    Minimises ``sum_i w_i (y_i - a_i)^2 + lam * ||D_d a||^2`` where ``D_d``
    is the d-th order difference matrix.

    Parameters
    ----------
    lam : float or {'gcv', 'reml', 'aic'}, default 'gcv'
        Smoothing parameter. Pass a non-negative float for a fixed value, or
        one of the three strings to select it automatically.
    order : int, default 2
        Order of the difference penalty.
    weights : array-like or None
        Default weights applied if ``fit`` is called without ``weights``.

    Attributes
    ----------
    fitted_ : ndarray of shape (n,)
        Smoothed values after ``fit``.
    lambda_ : float
        Lambda actually used (resolved from the criterion if applicable).
    edf_ : float
        Effective degrees of freedom.
    gcv_score_ : float
        Score of the chosen criterion at ``lambda_``.
    residuals_ : ndarray
        ``y - fitted_``.
    """

    def __init__(
        self,
        lam: LamArg = "gcv",
        order: int = 2,
        weights: np.ndarray | None = None,
        *,
        d: int | None = None,
    ) -> None:
        if d is not None:
            warnings.warn(
                "`d` is deprecated; use `order` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            order = d
        self.lam = lam
        self.order = order
        self.weights = weights

        self.fitted_: np.ndarray | None = None
        self.lambda_: float | None = None
        self.edf_: float | None = None
        self.gcv_score_: float | None = None
        self.residuals_: np.ndarray | None = None
        self._criterion: str | None = None

    def fit(
        self,
        y: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        lam: LamArg | None = None,
        d: int | None = None,
    ) -> WhittakerHenderson1D:
        """Fit the smoother to ``y`` with optional ``weights``."""
        if lam is not None or d is not None:
            warnings.warn(
                "Passing `lam` or `d` to fit() is deprecated; pass them to "
                "the constructor as `lam=` and `order=` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if lam is not None:
                self.lam = lam
            if d is not None:
                self.order = d
        y_arr = np.asarray(y, dtype=float).ravel()
        n = y_arr.size
        if weights is None:
            weights = self.weights
        w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).ravel()
        if w.shape != y_arr.shape:
            raise ValueError("weights must have the same shape as y")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

        P = penalty_matrix(n, self.order)

        if isinstance(self.lam, str):
            method = self.lam.lower()
            if method not in {"gcv", "reml", "aic"}:
                raise ValueError(f"unknown lambda criterion {self.lam!r}")
            lam_value = select_lambda_1d(y_arr, w, P, self.order, method)  # type: ignore[arg-type]
            self._criterion = method
        else:
            lam_value = float(self.lam)
            if lam_value < 0:
                raise ValueError("lam must be non-negative")
            self._criterion = "manual"

        res = solve(y_arr, w, P, lam_value)

        from ._lambda import _criterion

        score_method: LambdaMethod = (
            self._criterion if self._criterion in {"gcv", "reml", "aic"} else "gcv"  # type: ignore[assignment]
        )
        score = _criterion(score_method, n, self.order, res, lam_value)

        self.fitted_ = res.fitted
        self.lambda_ = float(lam_value)
        self.edf_ = float(res.edf)
        self.gcv_score_ = float(score)
        self.residuals_ = y_arr - res.fitted
        return self

    @property
    def fitted_values(self) -> np.ndarray | None:
        """Deprecated alias for :attr:`fitted_`."""
        warnings.warn(
            "`fitted_values` is deprecated; use `fitted_` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fitted_

    def predict(self, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Apply the smoother to a new vector of the same length using the fitted lambda."""
        if self.lambda_ is None:
            raise RuntimeError("call fit() before predict()")
        y_arr = np.asarray(y, dtype=float).ravel()
        n = y_arr.size
        w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).ravel()
        P = penalty_matrix(n, self.order)
        res = solve(y_arr, w, P, self.lambda_)
        return res.fitted
