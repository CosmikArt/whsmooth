"""
whsmooth — Whittaker-Henderson smoothing for actuarial graduation.

1D and 2D smoothing with automatic lambda selection via REML, GCV, and AIC.
"""

__version__ = "0.0.1"

from whsmooth.core import (
    LambdaSelector,
    MortalityGraduator,
    SeveritySmoother,
    WhittakerHenderson1D,
    WhittakerHenderson2D,
)

__all__ = [
    "WhittakerHenderson1D",
    "WhittakerHenderson2D",
    "LambdaSelector",
    "MortalityGraduator",
    "SeveritySmoother",
]
