"""Tests para casos de uso de pricing P&C: severidad por edad-vehículo, LDF smoothing."""

import numpy as np

from whsmooth import WhittakerHenderson1D


def test_severity_by_vehicle_age_smoothing():
    """
    Caso real: severidad media observada por edad de vehículo (0-25 años).
    Hay sparsity en edades altas (pocas observaciones), el smoother debe
    estabilizar la cola sin distorsionar el centro.
    """
    rng = np.random.default_rng(2026)
    ages = np.arange(0, 26)
    true_severity = 8000 + 50 * ages + 5 * ages**2
    counts = np.maximum(1, (5000 * np.exp(-0.15 * ages)).astype(int))
    observed = true_severity + rng.normal(0, 2000 / np.sqrt(counts), len(ages))
    wh = WhittakerHenderson1D(lam="gcv", order=2).fit(observed, weights=counts)
    np.testing.assert_allclose(wh.fitted_[:15], observed[:15], rtol=0.10)
    err_tail = np.abs(wh.fitted_[20:] - true_severity[20:]) / true_severity[20:]
    assert err_tail.mean() < 0.15


def test_ldf_smoothing_monotonic_decay():
    """
    Loss development factors crudos en un triángulo de 10 periodos.
    El smoother debe producir LDFs que decaigan monotónicamente hacia 1.
    """
    rng = np.random.default_rng(11)
    periods = np.arange(1, 11)
    ldf_true = 1.0 + 2.0 * np.exp(-0.5 * periods)
    ldf_obs = ldf_true + rng.normal(0, 0.05, len(periods))
    wh = WhittakerHenderson1D(lam="gcv", order=2).fit(ldf_obs)
    diffs = np.diff(wh.fitted_)
    violations = np.sum(diffs > 0.01)
    assert violations <= 1
