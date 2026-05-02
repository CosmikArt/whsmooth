"""Core solver for Whittaker-Henderson smoothing.

Given observations y, weights w, penalty matrix P (= D'D) and smoothing
parameter lambda, the smoother is the solution of the banded linear system

    (W + lam * P) a = W y                  with W = diag(w).

EDF (effective degrees of freedom) is the trace of the hat matrix
H = (W + lam P)^{-1} W. For modest n we compute this densely; the system
is small enough (banded) that it dominates nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class SolveResult:
    fitted: np.ndarray
    edf: float
    rss: float
    rss_penalised: float
    log_det_A: float


def solve(
    y: np.ndarray,
    weights: np.ndarray,
    penalty: sp.csc_matrix,
    lam: float,
) -> SolveResult:
    """Solve the Whittaker-Henderson normal equations and return diagnostics.

    Parameters
    ----------
    y : (n,) array
        Flattened observations (vec(Y) in 2D).
    weights : (n,) array
        Non-negative weights.
    penalty : (n, n) sparse
        Penalty matrix P. For 1D, ``D_d.T @ D_d``; for 2D, the Kronecker sum.
    lam : float
        Smoothing parameter.
    """
    A = (sp.diags(weights) + lam * penalty).toarray()
    rhs = weights * y
    a = np.linalg.solve(A, rhs)

    # H = A^{-1} W  →  edf = trace(H) = sum_i w_i * (A^{-1})_{ii}
    A_inv = np.linalg.inv(A)
    edf = float(np.sum(weights * np.diag(A_inv)))

    rss = float(np.sum(weights * (y - a) ** 2))
    pen_term = float(lam * a @ (penalty @ a))
    rss_pen = rss + pen_term

    sign, log_det_A = np.linalg.slogdet(A)
    if sign <= 0:
        log_det_A = float("nan")

    return SolveResult(
        fitted=a,
        edf=edf,
        rss=rss,
        rss_penalised=rss_pen,
        log_det_A=float(log_det_A),
    )


def kron_penalty(
    P_row: sp.csc_matrix,
    P_col: sp.csc_matrix,
    n_row: int,
    n_col: int,
    lam_row: float,
    lam_col: float,
) -> sp.csc_matrix:
    """Build the Kronecker-sum penalty for a 2D smoother on a row-major grid.

    For ``vec(A)`` ordered row-by-row (numpy default ``order='C'``), the
    smoothness penalty is

        lam_row * (P_row ⊗ I_col)  +  lam_col * (I_row ⊗ P_col).

    ``P_row`` is the ``n_row x n_row`` row-direction penalty and similarly
    for ``P_col``.
    """
    I_row = sp.eye(n_row, format="csc")
    I_col = sp.eye(n_col, format="csc")
    return (lam_row * sp.kron(P_row, I_col) + lam_col * sp.kron(I_row, P_col)).tocsc()
