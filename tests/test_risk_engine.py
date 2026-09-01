"""
Tests for the risk engine.

The point of these is not coverage for its own sake. Each one pins a
property that, if it broke silently, would make the engine's output
wrong in a way nobody would notice by eye:

  - EWMA and GARCH forecasts must not peek at the return they are
    forecasting (the single most common backtest bug)
  - GARCH MLE must recover known parameters from simulated data
  - Kupiec and Christoffersen must reject models that deserve it and
    accept models that do not
  - VaR must be ordered sensibly across confidence levels
  - Stress replay must report partial coverage rather than hide it
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.data.histdata import rolling_vwap, ticks_to_bars
from fxrisk.models.dcc import fit_dcc
from fxrisk.models.ewma import (
    RISKMETRICS_LAMBDA_DAILY,
    correlation_from_covariance,
    effective_observations,
    ewma_covariance_last,
    ewma_variance,
    portfolio_variance_path,
)
from fxrisk.models.garch import (
    fit_garch,
    forecast_variance,
    rolling_garch_forecasts,
    rolling_garch_variance,
)
from fxrisk.risk.backtesting import (
    backtest_var,
    basel_traffic_light,
    christoffersen_independence,
    kupiec_pof,
)
from fxrisk.risk.stress import SCENARIOS, run_scenario, stress_vs_var
from fxrisk.risk.var import (
    component_var,
    garch_var_series,
    historical_var,
    parametric_normal_var,
    portfolio_var_from_covariance,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

def simulate_garch(n=2500, omega=2e-6, alpha=0.08, beta=0.90, seed=0, nu=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n) if nu is None else rng.standard_t(nu, n) / np.sqrt(nu / (nu - 2))
    s2 = np.empty(n)
    e = np.empty(n)
    s2[0] = omega / (1 - alpha - beta)
    e[0] = np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        s2[t] = omega + alpha * e[t - 1] ** 2 + beta * s2[t - 1]
        e[t] = np.sqrt(s2[t]) * z[t]
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(e, index=idx), pd.Series(s2, index=idx)


@pytest.fixture(scope="module")
def garch_series():
    return simulate_garch()


# ---------------------------------------------------------------
# EWMA
# ---------------------------------------------------------------

def test_effective_observations():
    assert effective_observations(0.94) == pytest.approx(16.67, abs=0.01)
    assert effective_observations(0.97) == pytest.approx(33.33, abs=0.01)


def test_ewma_is_not_forward_looking(garch_series):
    """
    Changing the LAST return must not change any earlier forecast,
    and must not change the last forecast either - it is the value
    being forecast.
    """
    r, _ = garch_series
    base = ewma_variance(r)

    bumped = r.copy()
    bumped.iloc[-1] = bumped.iloc[-1] + 0.5  # an enormous shock
    after = ewma_variance(bumped)

    pd.testing.assert_series_equal(base, after)


def test_ewma_recursion_matches_manual():
    r = pd.Series(np.linspace(-0.01, 0.01, 200), index=pd.bdate_range("2020-01-01", periods=200))
    lam, warmup = 0.94, 30
    out = ewma_variance(r, lam=lam, warmup=warmup)

    sigma2 = float(np.var(r.iloc[:warmup], ddof=1))
    assert out.iloc[warmup] == pytest.approx(sigma2)

    expected = lam * sigma2 + (1 - lam) * r.iloc[warmup] ** 2
    assert out.iloc[warmup + 1] == pytest.approx(expected)


def test_ewma_reacts_faster_than_sample_variance():
    """A volatility jump must raise EWMA more than the full-sample sd."""
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.005, 500)
    wild = rng.normal(0, 0.03, 50)
    r = pd.Series(np.concatenate([calm, wild]), index=pd.bdate_range("2020-01-01", periods=550))

    ewma_end = np.sqrt(ewma_variance(r).iloc[-1])
    sample = r.std(ddof=1)
    assert ewma_end > sample


def test_ewma_covariance_is_symmetric_psd():
    rng = np.random.default_rng(11)
    r = pd.DataFrame(rng.normal(0, 0.01, (600, 4)), columns=list("abcd"),
                     index=pd.bdate_range("2020-01-01", periods=600))
    cov = ewma_covariance_last(r)

    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)
    eig = np.linalg.eigvalsh(cov.to_numpy())
    assert eig.min() > -1e-12

    corr = correlation_from_covariance(cov)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)
    assert corr.to_numpy().max() <= 1.0 + 1e-9


def test_portfolio_variance_path_matches_direct_ewma():
    """
    For constant weights the covariance recursion and the univariate
    recursion on portfolio returns must agree.
    """
    rng = np.random.default_rng(5)
    r = pd.DataFrame(rng.normal(0, 0.01, (500, 3)), columns=list("xyz"),
                     index=pd.bdate_range("2020-01-01", periods=500))
    w = np.array([0.5, 0.3, 0.2])

    via_cov = portfolio_variance_path(r, w, warmup=30)
    port = pd.Series(r.to_numpy() @ w, index=r.index)
    via_uni = ewma_variance(port, warmup=30)

    # The seeds differ (matrix vs scalar sample variance uses the same
    # data, so they agree exactly here).
    pd.testing.assert_series_equal(
        via_cov.dropna().rename("v"), via_uni.dropna().rename("v"), rtol=1e-10
    )


# ---------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------

def test_garch_recovers_known_parameters(garch_series):
    r, _ = garch_series
    fit = fit_garch(r, dist="normal", mean="zero")

    assert fit.converged
    assert fit.alpha == pytest.approx(0.08, abs=0.03)
    assert fit.beta == pytest.approx(0.90, abs=0.03)
    assert fit.persistence < 1.0


def test_garch_t_recovers_degrees_of_freedom():
    r, _ = simulate_garch(n=3000, seed=42, nu=5.0)
    fit = fit_garch(r, dist="t", mean="zero")
    assert fit.nu is not None
    assert 3.0 < fit.nu < 9.0


def test_garch_persistence_and_half_life_are_consistent(garch_series):
    r, _ = garch_series
    fit = fit_garch(r, dist="normal", mean="zero")
    implied = 0.5 ** (1 / fit.half_life)
    assert implied == pytest.approx(fit.persistence, rel=1e-6)


def test_garch_forecast_decays_toward_long_run(garch_series):
    r, _ = garch_series
    fit = fit_garch(r, dist="normal", mean="zero")
    f = forecast_variance(fit, horizon=400)
    lr = fit.long_run_variance

    assert abs(f[-1] - lr) < abs(f[0] - lr)
    assert f[-1] == pytest.approx(lr, rel=0.05)


def test_rolling_garch_is_not_forward_looking():
    r, _ = simulate_garch(n=900, seed=9)
    base = rolling_garch_variance(r, window=400, refit_every=100, dist="normal", min_obs=300)

    bumped = r.copy()
    bumped.iloc[-1] += 0.5
    after = rolling_garch_variance(bumped, window=400, refit_every=100, dist="normal", min_obs=300)

    # Every forecast up to and including the last must be unchanged:
    # none of them may use the return they are forecasting.
    pd.testing.assert_series_equal(base, after)


def test_rolling_garch_reports_fit_failures():
    r, _ = simulate_garch(n=800, seed=13)
    fc = rolling_garch_forecasts(r, window=400, refit_every=100, dist="normal", min_obs=300)
    assert "fit_failures" in fc.attrs
    assert fc.attrs["fit_failures"] == 0


def test_rolling_garch_nu_path_is_walk_forward_and_moves():
    """
    nu must be estimated inside the rolling loop, not once on the
    whole sample. Two consequences, both checked: it varies over
    time, and it is unchanged by data after the forecast date.
    """
    r, _ = simulate_garch(n=1200, seed=19, nu=5.0)
    fc = rolling_garch_forecasts(r, window=500, refit_every=60, dist="t", min_obs=400)

    nu = fc["nu"].dropna()
    assert len(nu) > 0
    assert nu.nunique() > 1, "nu is constant - it is being fitted on the full sample"

    # Truncating the sample must not alter forecasts already made.
    cut = 900
    fc_short = rolling_garch_forecasts(
        r.iloc[:cut], window=500, refit_every=60, dist="t", min_obs=400
    )
    pd.testing.assert_frame_equal(
        fc.iloc[:cut][["variance", "nu"]],
        fc_short[["variance", "nu"]],
    )


# ---------------------------------------------------------------
# DCC
# ---------------------------------------------------------------

def test_dcc_estimates_valid_parameters():
    rng = np.random.default_rng(21)
    n = 900
    base = rng.normal(0, 0.01, n)
    a = base + rng.normal(0, 0.006, n)
    b = base + rng.normal(0, 0.006, n)
    df = pd.DataFrame({"A": a, "B": b}, index=pd.bdate_range("2015-01-01", periods=n))

    res = fit_dcc(df, dist="normal", mean="zero")
    assert res.a >= 0
    assert res.b >= 0
    assert res.persistence < 1.0

    rho = res.correlation_series("A", "B")
    assert rho.between(-1, 1).all()
    assert rho.mean() > 0.3  # they genuinely share a common factor


def test_dcc_covariance_matrices_are_psd():
    rng = np.random.default_rng(22)
    n = 700
    df = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), columns=list("PQR"),
                      index=pd.bdate_range("2016-01-01", periods=n))
    res = fit_dcc(df, dist="normal", mean="zero")

    last = res.last_covariance().to_numpy()
    assert np.allclose(last, last.T)
    assert np.linalg.eigvalsh(last).min() > -1e-12


# ---------------------------------------------------------------
# VaR
# ---------------------------------------------------------------

def test_var_increases_with_confidence():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0, 0.01, 1000))
    v95 = historical_var(r, 0.95).var
    v99 = historical_var(r, 0.99).var
    assert v99 > v95


def test_expected_shortfall_exceeds_var():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.standard_t(5, 2000) / 100)
    for c in (0.95, 0.975, 0.99):
        est = historical_var(r, c)
        assert est.expected_shortfall > est.var


def test_parametric_normal_matches_closed_form():
    r = pd.Series(np.random.default_rng(1).normal(0.0002, 0.012, 5000))
    est = parametric_normal_var(r, 0.99)
    expected = -(r.mean() + r.std(ddof=1) * -2.3263478740408408)
    assert est.var == pytest.approx(expected, rel=1e-6)


def test_normal_var_understates_fat_tails():
    """
    The whole argument for Student-t: on genuinely fat-tailed data
    the Gaussian VaR sits below the empirical quantile.
    """
    r = pd.Series(np.random.default_rng(8).standard_t(3, 20000) / 100)
    assert parametric_normal_var(r, 0.99).var < historical_var(r, 0.99).var


def test_garch_var_series_is_not_forward_looking():
    """
    The regression test for the leak the earlier suite missed.

    The old tests perturbed the last return and checked the VARIANCE
    path, which was honest. The quantile was not: nu came from a
    full-sample fit, so future data reached every historical VaR
    through the tail shape rather than through the scale. Checking
    the variance alone could never see it.

    This checks the VaR series end to end. Truncating the sample
    must leave every already-made forecast bit-identical.
    """
    r, _ = simulate_garch(n=1200, seed=23, nu=5.0)

    full = garch_var_series(
        r, confidence=0.99, dist="t", window=500, refit_every=60, min_obs=400
    )
    cut = 900
    short = garch_var_series(
        r.iloc[:cut], confidence=0.99, dist="t", window=500, refit_every=60, min_obs=400
    )

    pd.testing.assert_frame_equal(full.iloc[:cut], short)


def test_garch_var_series_tail_shape_varies_over_time():
    """
    A constant nu column would mean the full-sample leak is back.
    """
    r, _ = simulate_garch(n=1200, seed=27, nu=4.0)
    out = garch_var_series(
        r, confidence=0.99, dist="t", window=500, refit_every=60, min_obs=400
    )
    assert out["nu"].dropna().nunique() > 1


def test_component_var_sums_to_total():
    rng = np.random.default_rng(12)
    r = pd.DataFrame(rng.normal(0, 0.01, (800, 4)), columns=list("abcd"))
    cov = r.cov()
    w = pd.Series([0.4, 0.3, 0.2, 0.1], index=list("abcd"))

    total = portfolio_var_from_covariance(w, cov, 0.99).var
    comp = component_var(w, cov, 0.99)
    assert comp["component_var"].sum() == pytest.approx(total, rel=1e-9)
    assert comp["pct_of_total"].sum() == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------

def test_kupiec_accepts_a_correct_model():
    stat, p = kupiec_pof(n_breaches=10, n_obs=1000, confidence=0.99)
    assert p > 0.05
    assert stat < 3.84


def test_kupiec_rejects_a_badly_calibrated_model():
    stat, p = kupiec_pof(n_breaches=50, n_obs=1000, confidence=0.99)
    assert p < 0.01
    assert stat > 6.63


def test_kupiec_handles_zero_breaches():
    stat, p = kupiec_pof(0, 500, 0.99)
    assert np.isfinite(stat)
    assert 0.0 <= p <= 1.0


def test_christoffersen_rejects_clustered_breaches():
    """Ten breaches in a row is the pathological case."""
    b = np.zeros(1000, dtype=int)
    b[500:520] = 1
    stat, p = christoffersen_independence(b)
    assert p < 0.01


def test_christoffersen_accepts_scattered_breaches():
    rng = np.random.default_rng(17)
    b = (rng.random(2000) < 0.01).astype(int)
    stat, p = christoffersen_independence(b)
    assert p > 0.05


def test_basel_zones():
    assert basel_traffic_light(4, 250)[0] == "green"
    assert basel_traffic_light(6, 250)[0] == "amber"
    assert basel_traffic_light(12, 250)[0] == "red"
    assert basel_traffic_light(4, 250)[1] == 3.00
    assert basel_traffic_light(12, 250)[1] == 4.00


def test_basel_is_not_applied_below_99_percent():
    """
    A well-calibrated 95% model breaches ~12 times per 250 days. The
    Basel thresholds would call that 'red' for working correctly, so
    they must not be applied outside 99%.
    """
    assert basel_traffic_light(12, 250, confidence=0.95)[0] == "n/a"
    assert basel_traffic_light(12, 250, confidence=0.975)[0] == "n/a"
    assert basel_traffic_light(12, 250, confidence=0.99)[0] == "red"


def test_backtest_reports_na_zone_at_95():
    r, s2 = simulate_garch(n=1500, seed=44)
    from scipy import stats as st

    v = pd.Series(-st.norm.ppf(0.05) * np.sqrt(s2), index=r.index)
    res = backtest_var(r, v, confidence=0.95, method="oracle95")
    assert res.basel_zone == "n/a"
    assert np.isnan(res.basel_multiplier)


def test_backtest_end_to_end_on_a_correct_model():
    """
    A VaR built from the true simulated volatility must be well
    calibrated: right number of breaches, green Basel zone.

    Note what this does NOT assert. A single oracle run can still
    trip the independence test - that is a Type I error, and at a 5%
    threshold it happens about 5% of the time by construction.
    asserting `res.passed()` on one fixed seed would be a flaky test
    dressed up as a correctness check. The size of the test is
    checked properly below.
    """
    from scipy import stats as st

    r, s2 = simulate_garch(n=3000, seed=31)
    true_var = pd.Series(-st.norm.ppf(0.01) * np.sqrt(s2), index=r.index)

    res = backtest_var(r, true_var, confidence=0.99, method="oracle")
    assert res.kupiec_pvalue > 0.05
    assert res.basel_zone == "green"
    assert abs(res.breach_rate - 0.01) < 0.006


def test_independence_test_has_nominal_size_under_the_null():
    """
    Across many samples, a correctly specified model must be
    rejected at roughly the nominal rate - not systematically more.
    A test that over-rejects would condemn good models.
    """
    from scipy import stats as st

    rejections = 0
    trials = 20
    for seed in range(trials):
        r, s2 = simulate_garch(n=3000, seed=seed)
        v = pd.Series(-st.norm.ppf(0.05) * np.sqrt(s2), index=r.index)
        res = backtest_var(r, v, confidence=0.95, method="oracle")
        if res.christoffersen_ind_pvalue < 0.05:
            rejections += 1

    # Nominal size is 5%. Allow generous binomial slack on 20 trials.
    assert rejections <= 4, f"over-rejecting the null: {rejections}/{trials}"


def test_christoffersen_has_power_against_a_flat_var():
    """
    A flat VaR ignores volatility clustering, so its breaches arrive
    in bursts. The independence test must catch that far more often
    than it fires on a correctly specified model.

    Power is checked in aggregate, not per seed, because the test is
    genuinely weak when breaches are sparse - see the note in
    backtesting.py. At 99% there are only ~30 breaches in 3000 days
    and power is under 50%; at 95% there are ~150 and it rises well
    above that. That is a property of the test, not a bug, and a
    single-seed assertion would hide it.
    """
    rejections = 0
    trials = 20
    for seed in range(trials):
        r, _ = simulate_garch(n=3000, seed=seed)
        flat = pd.Series(-np.quantile(r, 0.05), index=r.index)
        res = backtest_var(r, flat, confidence=0.95, method="flat")
        if res.christoffersen_ind_pvalue < 0.05:
            rejections += 1

    assert rejections >= trials // 2, f"only {rejections}/{trials} rejections"


def test_flat_var_scores_worse_than_oracle_on_lopez_loss():
    """Magnitude-aware scoring should prefer the correct model."""
    from scipy import stats as st

    r, s2 = simulate_garch(n=3000, seed=31)
    oracle = pd.Series(-st.norm.ppf(0.01) * np.sqrt(s2), index=r.index)
    flat = pd.Series(-np.quantile(r, 0.01), index=r.index)

    lo = backtest_var(r, oracle, 0.99, "oracle").lopez_loss
    lf = backtest_var(r, flat, 0.99, "flat").lopez_loss
    assert lo < lf


def test_backtest_rejects_negative_var_series():
    r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 300))
    bad = pd.Series(np.full(300, -0.02), index=r.index)
    with pytest.raises(ValueError, match="positive loss"):
        backtest_var(r, bad)


# ---------------------------------------------------------------
# Stress testing
# ---------------------------------------------------------------

def _scenario_frame(keys, assets, seed=1):
    """Build a return frame that spans the requested scenarios."""
    rng = np.random.default_rng(seed)
    start = min(SCENARIOS[k].start_ts for k in keys) - pd.Timedelta(days=10)
    end = max(SCENARIOS[k].end_ts for k in keys) + pd.Timedelta(days=10)
    idx = pd.bdate_range(start, end)
    return pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(assets))), index=idx, columns=assets)


def test_scenario_library_dates_are_valid():
    for key, sc in SCENARIOS.items():
        assert sc.start_ts < sc.end_ts, key
        if sc.peak_day:
            peak = pd.Timestamp(sc.peak_day)
            assert sc.start_ts <= peak <= sc.end_ts, key


def test_run_scenario_reports_partial_coverage():
    df = _scenario_frame(["covid_2020"], ["A", "B"])
    w = pd.Series({"A": 0.5, "B": 0.3, "MISSING": 0.2})

    res = run_scenario(df, w, SCENARIOS["covid_2020"])
    assert not res.complete
    assert res.missing_assets == ["MISSING"]
    assert res.coverage == pytest.approx(0.8)


def test_run_scenario_computes_a_real_drawdown():
    idx = pd.bdate_range("2020-02-19", "2020-03-23")
    falling = pd.DataFrame({"A": np.full(len(idx), -0.02)}, index=idx)
    res = run_scenario(falling, pd.Series({"A": 1.0}), SCENARIOS["covid_2020"])

    assert res.total_return < -0.3
    assert res.max_drawdown < -0.3
    assert res.worst_day == pytest.approx(-0.02)


def test_run_scenario_raises_when_window_is_empty():
    df = _scenario_frame(["covid_2020"], ["A"])
    with pytest.raises(ValueError, match="no data in window"):
        run_scenario(df, pd.Series({"A": 1.0}), SCENARIOS["gfc_2008"])


def test_stress_vs_var_flags_breaches():
    idx = pd.bdate_range("2020-02-19", "2020-03-23")
    df = pd.DataFrame({"A": np.full(len(idx), -0.05)}, index=idx)
    results = {"covid_2020": run_scenario(df, pd.Series({"A": 1.0}), SCENARIOS["covid_2020"])}

    table = stress_vs_var(results, one_day_var=0.02)
    assert bool(table.loc["covid_2020", "breaches_var"])
    assert table.loc["covid_2020", "multiple_of_var"] == pytest.approx(2.5, rel=1e-6)


# ---------------------------------------------------------------
# HistData handling
# ---------------------------------------------------------------

def test_ticks_to_bars_produces_tick_counts():
    idx = pd.date_range("2020-01-01 00:00", periods=600, freq="10s", tz="UTC")
    ticks = pd.DataFrame(
        {"bid": np.linspace(1.10, 1.11, 600), "ask": np.linspace(1.1001, 1.1101, 600)},
        index=idx,
    )
    ticks["mid"] = (ticks["bid"] + ticks["ask"]) / 2
    ticks["spread"] = ticks["ask"] - ticks["bid"]

    bars = ticks_to_bars(ticks, "1min")
    assert (bars["tick_count"] == 6).all()
    assert len(bars) == 100


def test_vwap_falls_back_to_twap_and_warns_without_volume():
    idx = pd.date_range("2020-01-01", periods=50, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "high": np.linspace(1.1, 1.2, 50),
            "low": np.linspace(1.0, 1.1, 50),
            "close": np.linspace(1.05, 1.15, 50),
            "tick_count": np.nan,
        },
        index=idx,
    )
    with pytest.warns(RuntimeWarning, match="TWAP, not a VWAP"):
        out = rolling_vwap(bars, window=10)
    assert out.name == "twap"


def test_vwap_weights_by_tick_count():
    """A bar with far more ticks must pull the VWAP toward its price."""
    idx = pd.date_range("2020-01-01", periods=2, freq="1min", tz="UTC")
    bars = pd.DataFrame(
        {
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "tick_count": [1.0, 99.0],
        },
        index=idx,
    )
    out = rolling_vwap(bars, window=2)
    assert out.iloc[-1] == pytest.approx((1.0 * 1 + 2.0 * 99) / 100)
