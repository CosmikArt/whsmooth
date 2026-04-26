"""Smoke tests for whsmooth — verify imports and class instantiation."""

import whsmooth
from whsmooth import (
    LambdaSelector,
    MortalityGraduator,
    SeveritySmoother,
    WhittakerHenderson1D,
    WhittakerHenderson2D,
)


def test_version_exists():
    """Package exposes a version string."""
    assert hasattr(whsmooth, "__version__")
    assert isinstance(whsmooth.__version__, str)
    assert whsmooth.__version__ == "0.0.1"


def test_whittaker_henderson_1d_instantiation():
    """WhittakerHenderson1D can be instantiated."""
    wh = WhittakerHenderson1D()
    assert wh is not None


def test_whittaker_henderson_2d_instantiation():
    """WhittakerHenderson2D can be instantiated."""
    wh2d = WhittakerHenderson2D()
    assert wh2d is not None


def test_lambda_selector_instantiation():
    """LambdaSelector can be instantiated with no args."""
    sel = LambdaSelector()
    assert sel is not None


def test_lambda_selector_with_data():
    """LambdaSelector can be instantiated with data."""
    import numpy as np

    y = np.random.rand(50)
    sel = LambdaSelector(y=y, d=3)
    assert sel is not None


def test_mortality_graduator_instantiation():
    """MortalityGraduator can be instantiated."""
    grad = MortalityGraduator()
    assert grad is not None


def test_severity_smoother_instantiation():
    """SeveritySmoother can be instantiated."""
    smoother = SeveritySmoother()
    assert smoother is not None


def test_all_exports():
    """All public names are in __all__."""
    expected = {
        "WhittakerHenderson1D",
        "WhittakerHenderson2D",
        "LambdaSelector",
        "MortalityGraduator",
        "SeveritySmoother",
    }
    assert set(whsmooth.__all__) == expected
