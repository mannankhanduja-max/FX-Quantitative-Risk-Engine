"""
Tests for `fxrisk.strategies.regime`.

The gates are a filter, and a filter's failure mode is not a
crash - it is a plausible-looking improvement that came from
look-ahead. So these mostly pin causality and the arithmetic that
would silently flatter a result if it were wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.strategies import regime as rg


def _idx(n):
    return pd.date_range("2026-01-05", periods=n, freq="15min",
                         tz="America/New_York")


def test_correlation_path_is_causal():
    """
    A shock on the last bar must not move the correlation reported
    FOR that bar. If it does, the gate is reading the bar it is
    deciding about.
    """
    n = 200
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 0.001, n), index=_idx(n))
    b = pd.Series(rng.normal(0, 0.001, n), index=_idx(n))
    base = pd.DataFrame({"A": a, "B": b})

    shocked = base.copy()
    shocked.iloc[-1] = [0.05, -0.05]

    r0 = rg.ewma_correlation_path(base)[("A", "B")]
    r1 = rg.ewma_correlation_path(shocked)[("A", "B")]
    assert r0.iloc[-1] == pytest.approx(r1.iloc[-1], nan_ok=True)


def test_partner_move_is_shifted():
    """The partner's move must be read from before the entry bar."""
    n = 120
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"A": rng.normal(0, 0.001, n), "B": rng.normal(0, 0.001, n)},
        index=_idx(n),
    )
    cfg = rg.GateConfig()
    rho = rg.ewma_correlation_path(df)

    base = rg.alignment("A", df, rho, cfg)
    bumped = df.copy()
    bumped.iloc[-1, bumped.columns.get_loc("B")] = 0.08
    after = rg.alignment("A", bumped, rg.ewma_correlation_path(bumped), cfg)

    assert base["implied"].iloc[-1] == after["implied"].iloc[-1]


def test_alignment_abstains_when_correlation_is_weak():
    """
    Below rho_min the partner says nothing, and the gate must
    record that rather than guess a direction from noise.
    """
    n = 150
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {"A": rng.normal(0, 0.001, n), "B": rng.normal(0, 0.001, n)},
        index=_idx(n),
    )
    rho = rg.ewma_correlation_path(df)
    strict = rg.alignment("A", df, rho, rg.GateConfig(rho_min=0.999))
    assert (strict["implied"] == 0.0).all()


def test_var_rescales_the_t_quantile():
    """
    sigma is a standard deviation, but the Student-t quantile is in
    units of the t SCALE parameter. Skipping the sqrt((nu-2)/nu)
    rescaling inflates VaR - by about 22% at nu = 4, which is where
    these fits usually land.
    """
    sigma = pd.Series([0.01, 0.01], index=_idx(2))
    nu = pd.Series([4.0, 4.0], index=_idx(2))
    v = rg.conditional_var(sigma, nu, conf=0.99)

    from scipy import stats
    raw = -stats.t.ppf(0.01, 4.0) * 0.01
    correct = raw * np.sqrt(2.0 / 4.0)

    assert v.iloc[0] == pytest.approx(correct)
    assert v.iloc[0] < raw
    assert raw / v.iloc[0] == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_var_is_nan_when_the_tail_has_no_variance():
    """nu <= 2 means infinite variance; VaR must refuse, not fabricate."""
    sigma = pd.Series([0.01], index=_idx(1))
    v = rg.conditional_var(sigma, pd.Series([2.0], index=_idx(1)))
    assert np.isnan(v.iloc[0])


def test_gate_thresholds_are_trailing_not_full_sample():
    """
    The subtle look-ahead. A full-sample quantile would let a late
    panic set the threshold deciding whether an early bar was calm
    - which never shows up as an impossible trade, only as a filter
    that was impossibly well calibrated.
    """
    n = 3000
    idx = _idx(n)
    calm = np.full(n, 0.001)
    calm[-200:] = 0.02                      # panic only at the end
    garch = pd.DataFrame({"sigma": calm, "nu": np.full(n, 6.0)}, index=idx)
    bars = pd.DataFrame({"High": 1.0, "Low": 1.0, "Close": 1.0}, index=idx)
    align = pd.DataFrame({"implied": 0.0, "rho": 0.0, "rho_abs": 0.0,
                          "partner": "B"}, index=idx)

    gf = rg.gates(bars, garch, align, rg.GateConfig(quantile_window=1000))
    early = gf["g_regime"].iloc[600:1500]
    assert early.mean() > 0.5, "calm bars judged against a later panic"


def test_config_rejects_nonsense():
    for kw in ({"sigma_low_q": 0.9, "sigma_high_q": 0.1},
               {"var_cap_q": 0.0}, {"var_conf": 0.4},
               {"rho_min": 1.0}, {"partner_lookback": 0},
               {"quantile_window": 10}):
        with pytest.raises(ValueError):
            rg.GateConfig(**kw)
