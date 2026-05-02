import numpy as np

from whsmooth import WH2D, WhittakerHenderson2D


def test_alias_identity_2d():
    assert WH2D is WhittakerHenderson2D


def test_2d_constant_surface_unchanged():
    """Una superficie constante debe quedar igual sin importar lambda."""
    Y = np.full((10, 12), 3.14)
    wh = WhittakerHenderson2D(lam=(100.0, 100.0), order=(2, 2)).fit(Y)
    np.testing.assert_allclose(wh.fitted_, Y, atol=1e-8)


def test_2d_shape_preserved():
    rng = np.random.default_rng(0)
    Y = rng.normal(0, 1, (15, 20))
    wh = WhittakerHenderson2D(lam=(1.0, 1.0), order=(2, 2)).fit(Y)
    assert wh.fitted_.shape == Y.shape


def test_2d_kronecker_smoothing_reduces_variance():
    rng = np.random.default_rng(3)
    nx, ny = 20, 25
    xx, yy = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    Y_true = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    Y = Y_true + rng.normal(0, 0.3, Y_true.shape)
    wh = WhittakerHenderson2D(lam=(10.0, 10.0), order=(2, 2)).fit(Y)
    var_raw = np.var(Y - Y_true)
    var_fit = np.var(wh.fitted_ - Y_true)
    assert var_fit < var_raw * 0.5
