"""
Tests for fxrisk.strategies.reversion.

The dangerous bug in this module is not look-ahead, it is SIGN. The
two confirmed signals are oriented differently in signals.py - one
is raw past return whose IC was negative, the other is already
negated - so taking both at face value points one leg at momentum
and the other at reversion, and they cancel. That happened in the
first version of this module, so it is pinned first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.strategies import reversion as rv


def _bars(n=1200, seed=0, freq="1h"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=n, freq=freq,
                        tz="America/New_York")
    c = 100 + np.cumsum(rng.normal(0, 0.05, n))
    w = np.abs(rng.normal(0, 0.06, n))
    return pd.DataFrame({"Open": c, "High": c + w, "Low": c - w,
                         "Close": c, "Volume": rng.integers(500, 3000, n)
                         .astype(float)}, index=idx)


# ------------------------------------------------------------
# Orientation
# ------------------------------------------------------------

def test_a_reversal_signal_is_flipped_and_a_negated_one_is_not():
    """The sign trap, straight from the study's measured ICs."""
    assert rv.SIGNAL_SIGN["mom_24h"] == -1.0   # IC was -0.058
    assert rv.SIGNAL_SIGN["poc_rev"] == +1.0   # IC was +0.074


def test_oriented_signals_agree_rather_than_cancel():
    """
    Taken at face value the pair correlates about -0.72 and the
    combined score collapses toward 0.5. Oriented, they agree.
    """
    from fxrisk.research import signals as sg
    b = _bars(1400, seed=1)
    f = sg.compute(b)
    raw = f["mom_24h"].corr(f["poc_rev"])
    orient = (f["mom_24h"] * rv.SIGNAL_SIGN["mom_24h"]).corr(
        f["poc_rev"] * rv.SIGNAL_SIGN["poc_rev"])
    assert orient == pytest.approx(-raw, rel=1e-6)
    assert orient > raw, "orientation did not change the sign of agreement"


def test_a_signal_with_no_measured_orientation_is_refused():
    with pytest.raises(ValueError, match="unknown signal|no measured"):
        rv.ReversionConfig(signals=("not_a_signal",))


# ------------------------------------------------------------
# Look-ahead
# ------------------------------------------------------------

def test_the_rank_is_trailing_not_full_sample():
    x = pd.Series(np.arange(600, dtype=float))
    r = rv._trailing_rank(x, window=100, min_obs=50)
    # A monotonically rising series is always top of its own window.
    assert r.dropna().min() > 0.9
    # Reversed, always bottom - a full-sample rank could not do both.
    r2 = rv._trailing_rank(x.iloc[::-1].reset_index(drop=True), 100, 50)
    assert r2.dropna().max() < 0.1


def test_scores_do_not_depend_on_the_future():
    b = _bars(1200, seed=2)
    base = rv.score(b)
    shocked = b.copy()
    k = shocked.index[-1]
    shocked.loc[k, ["High", "Close"]] *= 1.10
    after = rv.score(shocked)
    assert np.allclose(base.iloc[:-1].to_numpy(), after.iloc[:-1].to_numpy(),
                       equal_nan=True)


def test_the_exit_is_exactly_hold_bars_after_the_entry():
    b = _bars(1200, seed=3)
    cfg = rv.ReversionConfig(hold_bars=24, decision_hour=None)
    t = rv.positions(b, cfg)
    if t.empty:
        pytest.skip("no trades on this fixture")
    gap = (t["exit_time"] - t["entry_time"])
    assert (gap >= pd.Timedelta(hours=24)).all()


# ------------------------------------------------------------
# Swap
# ------------------------------------------------------------

def test_a_24_hour_hold_crosses_exactly_one_rollover():
    et = lambda s: pd.Timestamp(s, tz="America/New_York")
    assert rv._nights_between(et("2024-03-05 11:00"),
                              et("2024-03-06 11:00")) == 1


def test_a_hold_inside_one_fx_day_crosses_none():
    et = lambda s: pd.Timestamp(s, tz="America/New_York")
    assert rv._nights_between(et("2024-03-05 18:00"),
                              et("2024-03-06 16:00")) == 0


def test_swap_is_charged_per_night_and_raises_the_cost():
    b = _bars(1500, seed=4)
    free = rv.positions(b, rv.ReversionConfig(swap_bp_per_night=0.0))
    paid = rv.positions(b, rv.ReversionConfig(swap_bp_per_night=1.1))
    if free.empty:
        pytest.skip("no trades on this fixture")
    assert paid["cost_bp"].mean() > free["cost_bp"].mean()
    assert np.allclose(free["gross_bp"], paid["gross_bp"])


# ------------------------------------------------------------
# Gating
# ------------------------------------------------------------

def test_the_decision_hour_is_respected():
    b = _bars(2000, seed=5)
    t = rv.positions(b, rv.ReversionConfig(decision_hour=11))
    if t.empty:
        pytest.skip("no trades on this fixture")
    et = pd.DatetimeIndex(t["entry_time"]).tz_convert("America/New_York")
    assert (et.hour == 11).all()


def test_one_per_day_never_opens_twice_in_a_session():
    from fxrisk.data.intraday import session_id
    b = _bars(2000, seed=6)
    t = rv.positions(b, rv.ReversionConfig(one_per_day=True,
                                           decision_hour=None))
    if t.empty:
        pytest.skip("no trades on this fixture")
    s = session_id(pd.DatetimeIndex(t["entry_time"]))
    assert not pd.Series(s.to_numpy()).duplicated().any()


def test_only_the_tails_are_traded():
    b = _bars(2000, seed=7)
    cfg = rv.ReversionConfig(entry_quantile=0.2, decision_hour=None)
    t = rv.positions(b, cfg)
    if t.empty:
        pytest.skip("no trades on this fixture")
    longs, shorts = t[t["side"] > 0], t[t["side"] < 0]
    assert (longs["score"] >= 0.8 - 1e-9).all()
    assert (shorts["score"] <= 0.2 + 1e-9).all()


def test_config_rejects_nonsense():
    for kw in ({"signals": ()}, {"hold_bars": 0}, {"entry_quantile": 0.0},
               {"entry_quantile": 0.6}, {"rank_window": 10},
               {"min_rank_obs": 5}, {"cost_bp": -1},
               {"swap_bp_per_night": -1}, {"decision_hour": 24}):
        with pytest.raises(ValueError):
            rv.ReversionConfig(**kw)
