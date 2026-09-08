"""
Tests for `fxrisk.risk.strategy`.

The interesting ones are at the bottom. Most of this file checks
mechanics - alignment, costs, drawdown arithmetic - but the last
two MEASURE the two claims the module's docstring makes, rather
than asserting them:

  * that leaving flat days in a VaR backtest biases the expected
    breach count upward, and
  * that fitting GARCH to the strategy series produces a different
    (and lower) volatility path than scaling an asset-fitted model.

A docstring that makes a quantitative claim and a test suite that
never checks it is how a plausible-sounding error survives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.risk import strategy as st


# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-04", periods=n)


@pytest.fixture
def garch_returns() -> pd.Series:
    """A return series with genuine volatility clustering."""
    rng = np.random.default_rng(7)
    n = 1500
    omega, alpha, beta = 1e-6, 0.08, 0.90
    sigma2 = np.empty(n)
    r = np.empty(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(n):
        if t:
            sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = rng.standard_normal() * np.sqrt(sigma2[t])
    return pd.Series(r, index=_dates(n), name="asset")


@pytest.fixture
def flat_heavy_signal(garch_returns) -> pd.Series:
    """A signal that sits flat roughly a third of the time."""
    rng = np.random.default_rng(11)
    raw = rng.choice([-1.0, 0.0, 1.0], size=len(garch_returns), p=[0.3, 0.35, 0.35])
    return pd.Series(raw, index=garch_returns.index, name="signal")


# ------------------------------------------------------------
# Construction
# ------------------------------------------------------------

def test_strategy_returns_does_not_shift_again(garch_returns):
    """
    The module must NOT re-shift the signal.

    The shift is the caller's contract. A second shift inside the
    risk module would be invisible and would make every backtest
    one bar too conservative.
    """
    sig = pd.Series(1.0, index=garch_returns.index)
    frame = st.strategy_returns(sig, garch_returns)
    pd.testing.assert_series_equal(
        frame["gross"], garch_returns, check_names=False
    )


def test_costs_reduce_net_and_scale_with_turnover(garch_returns):
    sig = pd.Series(
        np.tile([1.0, -1.0], len(garch_returns) // 2), index=garch_returns.index
    )
    cheap = st.strategy_returns(sig, garch_returns, cost_per_turn=0.0001)
    dear = st.strategy_returns(sig, garch_returns, cost_per_turn=0.0010)

    assert dear["net"].sum() < cheap["net"].sum()
    # Flipping every bar costs 2 units of turnover per flip.
    assert cheap["turnover"].iloc[1:].mean() == pytest.approx(2.0)


def test_negative_cost_rejected(garch_returns):
    sig = pd.Series(1.0, index=garch_returns.index)
    with pytest.raises(ValueError):
        st.strategy_returns(sig, garch_returns, cost_per_turn=-0.001)


def test_exposure_summary_shares_sum_to_one(flat_heavy_signal):
    s = st.exposure_summary(flat_heavy_signal)
    assert s["long_share"] + s["short_share"] + s["flat_share"] == pytest.approx(1.0)
    assert s["time_in_market"] == pytest.approx(1.0 - s["flat_share"])


# ------------------------------------------------------------
# Drawdown
# ------------------------------------------------------------

def test_drawdown_is_compounded_not_summed():
    """-50% then +50% is a 25% loss, not flat."""
    r = pd.Series([-0.5, 0.5], index=_dates(2))
    prof = st.drawdown_profile(r)
    assert prof.max_drawdown == pytest.approx(-0.5)
    assert prof.current_drawdown == pytest.approx(-0.25)


def test_drawdown_duration_is_longest_not_deepest():
    """
    A short sharp fall and a long shallow one: depth and duration
    must come from different periods.
    """
    deep_short = [-0.20, 0.30]                 # deep, 1 day under water
    shallow_long = [-0.01] * 10 + [0.5]        # shallow, 10 days under
    r = pd.Series(deep_short + shallow_long, index=_dates(13))
    prof = st.drawdown_profile(r)

    assert prof.max_drawdown == pytest.approx(-0.20)
    assert prof.max_duration_days == 10


def test_drawdown_rejects_empty():
    with pytest.raises(ValueError):
        st.drawdown_profile(pd.Series([], dtype="float64"))


# ------------------------------------------------------------
# Attribution
# ------------------------------------------------------------

def test_loss_attribution_separates_cost_only_days():
    idx = _dates(3)
    frame = pd.DataFrame(
        {
            "position": [1.0, 1.0, 0.0],
            "gross": [0.001, -0.010, 0.0],   # day 1 gross-positive, day 2 wrong-way
            "cost": [0.002, 0.000, 0.0],     # day 1 loses only because of cost
            "net": [-0.001, -0.010, 0.0],
        },
        index=idx,
    )
    out = st.loss_attribution(frame)
    assert out["losing_days"] == 2
    assert out["cost_only_days"] == 1
    assert out["wrong_way_days"] == 1


def test_loss_attribution_requires_columns():
    with pytest.raises(ValueError):
        st.loss_attribution(pd.DataFrame({"net": [0.1]}))


# ------------------------------------------------------------
# The two claims the docstring makes
# ------------------------------------------------------------

def test_flat_days_bias_the_expected_breach_count(garch_returns, flat_heavy_signal):
    """
    MEASURES the dilution claim.

    Including flat days inflates n, and therefore the expected
    breach count, while the achievable breach count is unchanged -
    a breach is impossible when both the return and the VaR are
    zero. The inflation should track the flat share.
    """
    full, _ = st.compare_var_construction(
        flat_heavy_signal, garch_returns, confidence=0.99, in_market_only=False
    )
    exposed, _ = st.compare_var_construction(
        flat_heavy_signal, garch_returns, confidence=0.99, in_market_only=True
    )

    row = "position_scaled_garch"
    assert row in full.index and row in exposed.index

    # Same breaches, more "expected" - the model is being asked to
    # breach on days where breaching is impossible.
    assert full.loc[row, "breaches"] == exposed.loc[row, "breaches"]
    assert full.loc[row, "expected"] > exposed.loc[row, "expected"]

    flat_share = full.attrs["flat_share"]
    ratio = exposed.loc[row, "expected"] / full.loc[row, "expected"]
    assert ratio == pytest.approx(1.0 - flat_share, abs=0.05)


def test_naive_construction_is_degenerate(garch_returns, flat_heavy_signal):
    """
    MEASURES the contamination claim.

    NOTE ON A CORRECTED CLAIM. This test originally asserted that
    the naive VaR comes out LOWER, on the reasoning that flat days
    look calm and drag the variance down. The unconditional
    standard deviation does fall - but the measured VaR is about
    58% HIGHER, because the atom at zero doubles the excess
    kurtosis, which collapses the fitted degrees of freedom and
    pushes persistence to the IGARCH boundary. The inflated
    variance path dominates the lower unconditional level.

    The test now checks the mechanism, which is what actually makes
    the construction wrong, rather than a direction that happens to
    depend on the flat share.
    """
    from fxrisk.models.garch import fit_garch

    frame = st.strategy_returns(flat_heavy_signal, garch_returns)

    asset_fit = fit_garch(garch_returns, dist="t", mean="zero")
    strat_fit = fit_garch(frame["net"], dist="t", mean="zero")

    # The asset process is stationary; the strategy fit is not.
    assert asset_fit.alpha + asset_fit.beta < 0.995
    assert strat_fit.alpha + strat_fit.beta > asset_fit.alpha + asset_fit.beta
    assert strat_fit.alpha + strat_fit.beta >= 0.999

    # And the tail shape degenerates.
    assert strat_fit.nu < asset_fit.nu / 10

    # The two constructions therefore disagree materially.
    correct = st.strategy_var_series(
        flat_heavy_signal, garch_returns, confidence=0.99
    )
    naive = st.naive_strategy_var_series(frame["net"], confidence=0.99)
    exposed = frame["position"] != 0
    a = correct["var"][exposed.reindex(correct.index, fill_value=False)].dropna()
    b = naive["var"][exposed.reindex(naive.index, fill_value=False)].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 250
    assert b.loc[common].mean() > 1.2 * a.loc[common].mean()


def test_flat_days_carry_zero_var(garch_returns, flat_heavy_signal):
    """A strategy holding nothing cannot lose money."""
    out = st.strategy_var_series(flat_heavy_signal, garch_returns).dropna(
        subset=["asset_var"]
    )
    flat = out["position"] == 0
    assert flat.any()
    assert (out.loc[flat, "var"] == 0).all()
    assert (out.loc[~flat, "var"] > 0).all()
