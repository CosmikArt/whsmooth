"""Difference penalty matrices for Whittaker-Henderson smoothing."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def difference_matrix(n: int, order: int) -> sp.csc_matrix:
    """Build the d-th order difference matrix ``D_d`` of shape ``(n - d, n)``.

    ``D_d @ a`` returns the d-th order forward differences of ``a``.
    """
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    if order >= n:
        raise ValueError(f"order ({order}) must be smaller than n ({n})")
    D = np.eye(n)
    for _ in range(order):
        D = np.diff(D, axis=0)
    return sp.csc_matrix(D)


def penalty_matrix(n: int, order: int) -> sp.csc_matrix:
    """Build the penalty matrix ``D_d.T @ D_d`` of shape ``(n, n)``."""
    D = difference_matrix(n, order)
    return (D.T @ D).tocsc()
