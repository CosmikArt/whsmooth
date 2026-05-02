"""2D Whittaker-Henderson smoother on a regular grid."""

from __future__ import annotations

import warnings

import numpy as np

from ._core import kron_penalty, solve
from ._lambda import LambdaMethod, select_lambda_2d
from ._penalties import penalty_matrix
from .smooth1d import LamArg


def _expand_pair(value: LamArg | tuple[LamArg, LamArg], name: str) -> tuple[LamArg, LamArg]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be a scalar or a 2-tuple")
        return value
    return value, value


def _expand_int_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a 2-tuple of ints")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class WhittakerHenderson2D:
    """Two-dimensional Whittaker-Henderson smoother.

    Smooths a matrix ``Y`` of shape ``(n_row, n_col)`` using a Kronecker-sum
    penalty (Currie, Durban & Eilers, 2004).

    Parameters
    ----------
    lam : float, str, or 2-tuple thereof, default ('gcv', 'gcv')
        Smoothing parameters for the row and column directions. Each element
        accepts the same values as in 1D: a non-negative float, or one of
        ``'gcv'``, ``'reml'``, ``'aic'``. A scalar is broadcast to both axes.
    order : int or (int, int), default (2, 2)
        Difference orders for the row and column penalties.
    weights : array-like or None
        Default weight matrix used when ``fit`` is called without weights.
    """

    def __init__(
        self,
        lam: LamArg | tuple[LamArg, LamArg] = ("gcv", "gcv"),
        order: int | tuple[int, int] = (2, 2),
        weights: np.ndarray | None = None,
        *,
        d: int | tuple[int, int] | None = None,
    ) -> None:
        if d is not None:
            warnings.warn(
                "`d` is deprecated; use `order` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            order = d
        self.lam = _expand_pair(lam, "lam")
        self.order = _expand_int_pair(order, "order")
        self.weights = weights

        self.fitted_: np.ndarray | None = None
        self.lambda_: tuple[float, float] | None = None
        self.edf_: float | None = None
        self.gcv_score_: float | None = None
        self.residuals_: np.ndarray | None = None
        self._criterion: tuple[str, str] | None = None

    def fit(
        self,
        Y: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> WhittakerHenderson2D:
        """Fit the smoother to a 2D matrix ``Y`` with optional weights."""
        Y_arr = np.asarray(Y, dtype=float)
        if Y_arr.ndim != 2:
            raise ValueError("Y must be 2D")
        n_row, n_col = Y_arr.shape

        if weights is None:
            weights = self.weights
        if weights is None:
            W = np.ones_like(Y_arr)
        else:
            W = np.asarray(weights, dtype=float)
            if W.shape != Y_arr.shape:
                raise ValueError("weights must have the same shape as Y")
            if np.any(W < 0):
                raise ValueError("weights must be non-negative")

        order_r, order_c = self.order
        P_row = penalty_matrix(n_row, order_r)
        P_col = penalty_matrix(n_col, order_c)

        y_vec = Y_arr.ravel(order="C")
        w_vec = W.ravel(order="C")

        def build(lr: float, lc: float):
            return kron_penalty(P_row, P_col, n_row, n_col, lr, lc)

        # Resolve lambdas: any string entries trigger optimisation.
        lam_r_in, lam_c_in = self.lam
        any_auto = isinstance(lam_r_in, str) or isinstance(lam_c_in, str)
        null_dim = order_r * order_c

        if any_auto:
            method_r = lam_r_in.lower() if isinstance(lam_r_in, str) else "gcv"
            method_c = lam_c_in.lower() if isinstance(lam_c_in, str) else "gcv"
            for m in (method_r, method_c):
                if m not in {"gcv", "reml", "aic"}:
                    raise ValueError(f"unknown lambda criterion {m!r}")
            method = method_r if isinstance(lam_r_in, str) else method_c  # type: ignore[assignment]
            fixed = (
                None if isinstance(lam_r_in, str) else float(lam_r_in),
                None if isinstance(lam_c_in, str) else float(lam_c_in),
            )
            lam_r, lam_c = select_lambda_2d(y_vec, w_vec, build, null_dim, method, fixed)  # type: ignore[arg-type]
            self._criterion = (method_r, method_c)
        else:
            lam_r = float(lam_r_in)
            lam_c = float(lam_c_in)
            if lam_r < 0 or lam_c < 0:
                raise ValueError("lam must be non-negative")
            self._criterion = ("manual", "manual")

        P_full = build(lam_r, lam_c)
        res = solve(y_vec, w_vec, P_full, lam=1.0)

        self.fitted_ = res.fitted.reshape(n_row, n_col, order="C")
        self.lambda_ = (lam_r, lam_c)
        self.edf_ = float(res.edf)
        # For diagnostics: GCV-style score at the chosen lambdas.
        from ._lambda import _criterion as _crit

        method_for_score: LambdaMethod = (
            self._criterion[0]  # type: ignore[assignment]
            if self._criterion[0] in {"gcv", "reml", "aic"}
            else "gcv"
        )
        eff_lam = float(np.sqrt(max(lam_r, 1e-300) * max(lam_c, 1e-300)))
        self.gcv_score_ = float(
            _crit(
                method_for_score,
                y_vec.size,
                null_dim,
                res.rss,
                res.rss_penalised,
                res.edf,
                res.log_det_A,
                eff_lam,
            )
        )
        self.residuals_ = Y_arr - self.fitted_
        return self
