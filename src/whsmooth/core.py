"""
Core smoothing classes for whsmooth.

This module implements Whittaker-Henderson smoothing in 1D and 2D,
lambda selection via REML/GCV/AIC, and actuarial convenience wrappers.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp


class WhittakerHenderson1D:
    """One-dimensional Whittaker-Henderson smoother.

    The Whittaker-Henderson smoother minimizes:

        S(a; lam, d) = sum_i w_i (y_i - a_i)^2  +  lam * ||D_d a||^2

    where D_d is the d-th order difference matrix and lam controls the
    smoothness-fidelity tradeoff. The solution reduces to a banded linear
    system solved via sparse Cholesky factorization.

    Parameters
    ----------
    None at construction time. All parameters are passed to ``fit()``.

    Examples
    --------
    >>> import numpy as np
    >>> from whsmooth import WhittakerHenderson1D
    >>> wh = WhittakerHenderson1D()
    >>> y = np.array([0.1, 0.3, 0.2, 0.5, 0.4])
    >>> wh.fit(y, lam=10.0, d=2)
    """

    def __init__(self) -> None:
        self._fitted: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._lam: float | None = None
        self._d: int | None = None

    def fit(
        self,
        y: np.ndarray | pd.Series,
        weights: np.ndarray | None = None,
        lam: float | None = None,
        d: int = 2,
    ) -> WhittakerHenderson1D:
        """Fit the Whittaker-Henderson smoother to data.

        Parameters
        ----------
        y : array-like of shape (n,)
            Observed values to smooth (e.g. crude qx, claim severities).
        weights : array-like of shape (n,) or None
            Non-negative weights. If None, unit weights are used.
            For mortality graduation, exposures are the natural weights.
        lam : float or None
            Smoothing parameter (lambda). Larger values produce smoother
            curves. If None, must be selected via ``LambdaSelector`` first.
        d : int, default 2
            Order of the difference penalty. ``d=2`` penalizes curvature
            (the most common choice); ``d=1`` penalizes slope; ``d=3``
            penalizes changes in curvature.

        Returns
        -------
        self
            The fitted smoother instance.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError(
            "WhittakerHenderson1D.fit() is not yet implemented. "
            "This is the initial scaffold — implementation coming soon."
        )

    def smooth(self) -> np.ndarray:
        """Return the smoothed values after fitting.

        Returns
        -------
        np.ndarray
            The graduated / smoothed values.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("WhittakerHenderson1D.smooth() is not yet implemented.")

    @property
    def fitted_values(self) -> np.ndarray:
        """The fitted (graduated) values after calling ``fit()``.

        Returns
        -------
        np.ndarray
            Graduated values of the same length as the input ``y``.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("WhittakerHenderson1D.fitted_values is not yet implemented.")

    @property
    def residuals(self) -> np.ndarray:
        """Residuals: observed minus fitted values.

        Returns
        -------
        np.ndarray
            ``y - fitted_values``.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("WhittakerHenderson1D.residuals is not yet implemented.")

    @property
    def edf(self) -> float:
        """Effective degrees of freedom of the smoother.

        The EDF is the trace of the smoother (hat) matrix H:

            edf = tr(H) = tr( (W + lam * D_d' D_d)^{-1} W )

        A value close to n means little smoothing; a value close to d
        means heavy smoothing.

        Returns
        -------
        float
            Effective degrees of freedom.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("WhittakerHenderson1D.edf is not yet implemented.")

    def penalty_matrix(self, n: int | None = None, d: int | None = None) -> sp.csc_matrix:
        """Construct the d-th order difference penalty matrix D_d' D_d.

        Parameters
        ----------
        n : int or None
            Size of the vector. If None, uses the length of the fitted data.
        d : int or None
            Difference order. If None, uses the order from ``fit()``.

        Returns
        -------
        scipy.sparse.csc_matrix
            The (n x n) penalty matrix ``D_d.T @ D_d`` in sparse format.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("WhittakerHenderson1D.penalty_matrix() is not yet implemented.")


class WhittakerHenderson2D:
    """Two-dimensional Whittaker-Henderson smoother for age x year grids.

    Extends the 1D smoother to matrices (e.g. mortality rates indexed by
    age and calendar year) using Kronecker-product penalties, following
    Currie, Durban & Eilers (2004).

    The objective becomes:

        S(A; lam_r, lam_c) = ||W^{1/2} (Z - A)||_F^2
            + lam_r * ||D_dr A||_F^2
            + lam_c * ||A D_dc'||_F^2

    where D_dr and D_dc are row and column difference matrices.

    Examples
    --------
    >>> import numpy as np
    >>> from whsmooth import WhittakerHenderson2D
    >>> wh2d = WhittakerHenderson2D()
    >>> Z = np.random.rand(60, 20)  # 60 ages x 20 years
    >>> wh2d.fit(Z, lam_row=100, lam_col=100)
    """

    def __init__(self) -> None:
        self._fitted: np.ndarray | None = None
        self._Z: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._lam_row: float | None = None
        self._lam_col: float | None = None

    def fit(
        self,
        Z: np.ndarray | pd.DataFrame,
        weights: np.ndarray | None = None,
        lam_row: float | None = None,
        lam_col: float | None = None,
        d_row: int = 2,
        d_col: int = 2,
    ) -> WhittakerHenderson2D:
        """Fit the 2D Whittaker-Henderson smoother to a grid.

        Parameters
        ----------
        Z : array-like of shape (n_ages, n_years)
            Observed rate matrix. Rows are ages, columns are calendar years.
        weights : array-like of shape (n_ages, n_years) or None
            Non-negative weight matrix. If None, unit weights are used.
            For mortality, the exposure matrix is the natural choice.
        lam_row : float or None
            Smoothing parameter for the row (age) direction.
        lam_col : float or None
            Smoothing parameter for the column (year) direction.
        d_row : int, default 2
            Difference order for the row penalty.
        d_col : int, default 2
            Difference order for the column penalty.

        Returns
        -------
        self
            The fitted smoother instance.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError(
            "WhittakerHenderson2D.fit() is not yet implemented. "
            "This is the initial scaffold — implementation coming soon."
        )


class LambdaSelector:
    """Automatic selection of the smoothing parameter lambda.

    Supports three principled criteria:

    - **GCV** (generalized cross-validation): minimizes leave-one-out
      prediction error without explicit cross-validation.
    - **REML** (restricted maximum likelihood): treats lambda as a
      variance-component parameter; generally preferred for mortality.
    - **AIC** (Akaike information criterion): balances fit and complexity
      via the effective degrees of freedom.

    Parameters
    ----------
    y : array-like of shape (n,)
        Observed values.
    weights : array-like of shape (n,) or None
        Non-negative weights.
    d : int, default 2
        Difference order for the penalty.

    Examples
    --------
    >>> import numpy as np
    >>> from whsmooth import LambdaSelector
    >>> y = np.random.rand(100)
    >>> sel = LambdaSelector(y=y)
    >>> lam_opt = sel.select(method="gcv")
    """

    def __init__(
        self,
        y: np.ndarray | pd.Series | None = None,
        weights: np.ndarray | None = None,
        d: int = 2,
    ) -> None:
        self._y = np.asarray(y) if y is not None else None
        self._weights = weights
        self._d = d

    def select(
        self,
        method: Literal["gcv", "reml", "aic"] = "gcv",
    ) -> float:
        """Select the optimal lambda using the specified criterion.

        Parameters
        ----------
        method : {"gcv", "reml", "aic"}, default "gcv"
            Selection criterion.

        Returns
        -------
        float
            Optimal smoothing parameter lambda.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError(f"LambdaSelector.select(method={method!r}) is not yet implemented.")

    def gcv(self, lam: float) -> float:
        """Compute the GCV criterion for a given lambda.

        GCV = (1/n) * ||y - a_hat||^2_W / (1 - edf/n)^2

        Parameters
        ----------
        lam : float
            Smoothing parameter to evaluate.

        Returns
        -------
        float
            GCV score (lower is better).

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("LambdaSelector.gcv() is not yet implemented.")

    def reml(self, lam: float) -> float:
        """Compute the REML (restricted maximum likelihood) criterion.

        Parameters
        ----------
        lam : float
            Smoothing parameter to evaluate.

        Returns
        -------
        float
            Negative REML log-likelihood (to be minimized).

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("LambdaSelector.reml() is not yet implemented.")

    def aic(self, lam: float) -> float:
        """Compute the AIC criterion for a given lambda.

        AIC = n * log(RSS/n) + 2 * edf

        Parameters
        ----------
        lam : float
            Smoothing parameter to evaluate.

        Returns
        -------
        float
            AIC score (lower is better).

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("LambdaSelector.aic() is not yet implemented.")

    def plot_criterion(
        self,
        method: Literal["gcv", "reml", "aic"] = "gcv",
        lam_range: tuple[float, float] = (0.01, 1e8),
        n_points: int = 200,
    ) -> None:
        """Plot the selection criterion as a function of lambda.

        Useful for visual inspection of the criterion surface and
        verifying that the optimum is well-defined.

        Parameters
        ----------
        method : {"gcv", "reml", "aic"}, default "gcv"
            Which criterion to plot.
        lam_range : tuple of (float, float)
            Range of lambda values (on log scale).
        n_points : int, default 200
            Number of evaluation points.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError("LambdaSelector.plot_criterion() is not yet implemented.")


class MortalityGraduator:
    """Convenience wrapper for mortality table graduation.

    Combines Whittaker-Henderson smoothing with a Poisson likelihood
    (appropriate for death counts) and automatic lambda selection.

    This is the standard actuarial use case: given crude death rates
    (deaths / exposures) by age, produce a smooth graduated mortality
    table.

    Parameters
    ----------
    deaths : array-like of shape (n,)
        Observed death counts by age.
    exposures : array-like of shape (n,)
        Central exposures by age.
    ages : array-like of shape (n,) or None
        Age labels. If None, uses 0-based integer ages.

    Examples
    --------
    >>> import numpy as np
    >>> from whsmooth import MortalityGraduator
    >>> deaths = np.array([5, 8, 12, 18, 25, 40, 60, 90])
    >>> exposures = np.array([10000] * 8)
    >>> grad = MortalityGraduator(deaths=deaths, exposures=exposures)
    >>> grad.graduate(method="reml")
    """

    def __init__(
        self,
        deaths: np.ndarray | pd.Series | None = None,
        exposures: np.ndarray | pd.Series | None = None,
        ages: np.ndarray | pd.Series | None = None,
    ) -> None:
        self._deaths = np.asarray(deaths) if deaths is not None else None
        self._exposures = np.asarray(exposures) if exposures is not None else None
        self._ages = np.asarray(ages) if ages is not None else None
        self._graduated_qx: np.ndarray | None = None

    def graduate(
        self,
        method: Literal["gcv", "reml", "aic"] = "reml",
        d: int = 2,
        lam: float | None = None,
    ) -> np.ndarray:
        """Graduate the mortality table.

        Uses iteratively reweighted Whittaker-Henderson smoothing with
        Poisson deviance, following Camarda (2012).

        Parameters
        ----------
        method : {"gcv", "reml", "aic"}, default "reml"
            Lambda selection method. Ignored if ``lam`` is provided.
        d : int, default 2
            Difference penalty order.
        lam : float or None
            Manual smoothing parameter. If None, selected automatically.

        Returns
        -------
        np.ndarray
            Graduated qx values (central death rates).

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError(
            "MortalityGraduator.graduate() is not yet implemented. "
            "This is the initial scaffold — implementation coming soon."
        )


class SeveritySmoother:
    """Convenience wrapper for claim severity smoothing.

    Smooths empirical claim severity curves (e.g. average severity by
    development period or by claim size band) using Whittaker-Henderson
    with Gaussian likelihood and automatic lambda selection.

    Parameters
    ----------
    severities : array-like of shape (n,)
        Observed average severities.
    claim_counts : array-like of shape (n,) or None
        Number of claims in each bucket (used as weights).
    labels : array-like of shape (n,) or None
        Labels for each bucket (e.g. development periods, size bands).

    Examples
    --------
    >>> import numpy as np
    >>> from whsmooth import SeveritySmoother
    >>> sev = np.array([1200, 1350, 1280, 1500, 1450, 1600, 1550])
    >>> counts = np.array([200, 180, 150, 120, 90, 60, 40])
    >>> smoother = SeveritySmoother(severities=sev, claim_counts=counts)
    >>> smoother.smooth(method="gcv")
    """

    def __init__(
        self,
        severities: np.ndarray | pd.Series | None = None,
        claim_counts: np.ndarray | pd.Series | None = None,
        labels: np.ndarray | pd.Series | None = None,
    ) -> None:
        self._severities = np.asarray(severities) if severities is not None else None
        self._claim_counts = np.asarray(claim_counts) if claim_counts is not None else None
        self._labels = np.asarray(labels) if labels is not None else None
        self._smoothed: np.ndarray | None = None

    def smooth(
        self,
        method: Literal["gcv", "reml", "aic"] = "gcv",
        d: int = 2,
        lam: float | None = None,
    ) -> np.ndarray:
        """Smooth the claim severity curve.

        Parameters
        ----------
        method : {"gcv", "reml", "aic"}, default "gcv"
            Lambda selection method. Ignored if ``lam`` is provided.
        d : int, default 2
            Difference penalty order.
        lam : float or None
            Manual smoothing parameter. If None, selected automatically.

        Returns
        -------
        np.ndarray
            Smoothed severity values.

        Raises
        ------
        NotImplementedError
            This is a scaffold; implementation is forthcoming.
        """
        raise NotImplementedError(
            "SeveritySmoother.smooth() is not yet implemented. "
            "This is the initial scaffold — implementation coming soon."
        )
