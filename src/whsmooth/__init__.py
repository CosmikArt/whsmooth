"""whsmooth — Whittaker-Henderson smoothing for P&C ratemaking and graduation.

1D and 2D smoothing with sparse banded solvers and automatic lambda selection
via GCV, REML, or AIC.
"""

from __future__ import annotations

from ._compat import LambdaSelector
from .smooth1d import WhittakerHenderson1D
from .smooth2d import WhittakerHenderson2D

WH1D = WhittakerHenderson1D
WH2D = WhittakerHenderson2D

__version__ = "0.1.0"

__all__ = [
    "WhittakerHenderson1D",
    "WhittakerHenderson2D",
    "WH1D",
    "WH2D",
    "LambdaSelector",
    "__version__",
]
