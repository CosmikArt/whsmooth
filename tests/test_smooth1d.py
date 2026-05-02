import numpy as np
import pytest

from whsmooth import WH1D, WhittakerHenderson1D


def test_alias_identity():
    """WH1D debe ser literal el mismo objeto que WhittakerHenderson1D."""
    assert WH1D is WhittakerHenderson1D


def test_perfect_signal_recovers():
    """Con lambda muy chico, el smoother recupera la señal cruda."""
    rng = np.random.default_rng(0)
    y = np.linspace(0, 1, 50) + rng.normal(0, 1e-6, 50)
    wh = WhittakerHenderson1D(lam=1e-10, order=2).fit(y)
    np.testing.assert_allclose(wh.fitted_, y, atol=1e-4)


def test_high_lambda_flattens_to_polynomial():
    """Con lambda muy grande y order=2, el resultado debe ser una recta."""
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0, 9.0])
    wh = WhittakerHenderson1D(lam=1e10, order=2).fit(y)
    second_diff = np.diff(wh.fitted_, n=2)
    np.testing.assert_allclose(second_diff, 0, atol=1e-3)


def test_gcv_selects_reasonable_lambda():
    """GCV debe elegir un lambda finito y positivo en un caso bien-condicionado."""
    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, 100)
    y_true = np.sin(x)
    y = y_true + rng.normal(0, 0.2, 100)
    wh = WhittakerHenderson1D(lam="gcv", order=2).fit(y)
    assert 0 < wh.lambda_ < 1e8
    mse_raw = np.mean((y - y_true) ** 2)
    mse_fit = np.mean((wh.fitted_ - y_true) ** 2)
    assert mse_fit < mse_raw * 0.5


def test_weights_zero_ignores_observation():
    """Una observación con weight=0 no debe afectar el fit."""
    y = np.ones(20)
    y[10] = 1000.0
    w = np.ones(20)
    w[10] = 0.0
    wh = WhittakerHenderson1D(lam=1.0, order=2).fit(y, weights=w)
    assert abs(wh.fitted_[10] - 1.0) < 0.1


def test_deprecated_d_parameter_warns():
    """El parámetro `d` debe emitir DeprecationWarning pero seguir funcionando."""
    y = np.ones(20)
    with pytest.warns(DeprecationWarning, match="order"):
        wh = WhittakerHenderson1D(lam=1.0, d=2).fit(y)
    assert wh.fitted_ is not None


def test_edf_bounds():
    """EDF debe estar entre order y n."""
    rng = np.random.default_rng(1)
    n = 100
    y = rng.normal(0, 1, n)
    order = 2
    wh = WhittakerHenderson1D(lam=10.0, order=order).fit(y)
    assert order <= wh.edf_ <= n


def test_three_lambda_methods_agree_on_smooth_data():
    """En data muy suave, GCV/REML/AIC deben elegir lambdas dentro de un orden de magnitud."""
    rng = np.random.default_rng(7)
    x = np.linspace(0, 1, 200)
    y = np.exp(x) + rng.normal(0, 0.05, 200)
    lams = []
    for method in ["gcv", "reml", "aic"]:
        wh = WhittakerHenderson1D(lam=method, order=2).fit(y)
        lams.append(wh.lambda_)
    ratio = max(lams) / min(lams)
    assert ratio < 50
