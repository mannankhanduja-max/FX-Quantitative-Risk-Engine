"""
Tests for fxrisk.risk.spread.

The cost model is the one place where being wrong in your own
favour is invisible: nothing crashes, the equity curve just
improves. So these check the direction of every adjustment, not
merely that a number comes back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.risk import spread


def _bars(n=600, freq="5min", vol=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02 00:00", periods=n, freq=freq,
                        tz="America/New_York")
    c = 100 + np.cumsum(rng.normal(0, 0.02, n))
    w = np.abs(rng.normal(0, 0.03, n))
    v = np.full(n, vol) if np.isscalar(vol) else np.asarray(vol, dtype=float)
    return pd.DataFrame({"Open": c, "High": c + w, "Low": c - w,
                         "Close": c, "Volume": v}, index=idx)


# ------------------------------------------------------------
# Corwin-Schultz
# ------------------------------------------------------------

def test_a_wider_quoted_spread_raises_the_estimate():
    """
    The estimator's one job. Widening the bid-ask bounce inflates
    single-bar ranges more than the combined two-bar range, which
    is exactly the asymmetry it identifies on.
    """
    b = _bars(400, seed=1)
    wide = b.copy()
    wide["High"] = b["High"] * 1.001
    wide["Low"] = b["Low"] * 0.999
    assert spread.corwin_schultz(wide).mean() > spread.corwin_schultz(b).mean()


def test_negative_estimates_are_floored_at_zero():
    """A negative spread is noise, not information. Per the paper."""
    s = spread.corwin_schultz(_bars(400, seed=2))
    assert (s.dropna() >= 0).all()


def test_the_first_bar_of_a_session_is_not_priced():
    """Its pair straddles a closure, so the range is a gap."""
    b = _bars(400)
    s = spread.corwin_schultz(b, within_session=True)
    from fxrisk.data.intraday import session_id
    sess = pd.Series(session_id(b.index).to_numpy(), index=b.index)
    first = sess != sess.shift(1)
    assert s[first].isna().all()


def test_estimator_refuses_non_positive_prices():
    b = _bars(50)
    b.loc[b.index[10], "Low"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        spread.corwin_schultz(b)


# ------------------------------------------------------------
# The activity-shaped cost
# ------------------------------------------------------------

def test_thin_bars_cost_more_than_busy_ones():
    """The whole point. Getting this backwards is the Corwin-Schultz
    failure that made the activity proxy necessary in the first
    place, so it is pinned here explicitly."""
    n = 600
    v = np.full(n, 1000.0)
    v[300:360] = 50.0                      # a dead patch
    b = _bars(n, vol=v)
    c = spread.cost_bp_series(b)
    assert c.iloc[320:360].median() > c.iloc[100:200].median()


def test_the_median_bar_still_pays_the_flat_rate():
    """
    Rescaling keeps the LEVEL comparable to every flat-cost result
    in the repository, changing only the shape through the day. If
    this drifts, no cross-run comparison means anything.
    """
    n = 600
    rng = np.random.default_rng(3)
    b = _bars(n, vol=rng.lognormal(6, 1.0, n))
    c = spread.cost_bp_series(b, base_bp=1.0)
    assert c.median() == pytest.approx(1.0, rel=0.05)


def test_a_higher_elasticity_widens_the_spread_of_costs():
    n = 600
    rng = np.random.default_rng(4)
    b = _bars(n, vol=rng.lognormal(6, 1.0, n))
    lo = spread.cost_bp_series(b, power=0.3)
    hi = spread.cost_bp_series(b, power=1.0)
    assert hi.std() > lo.std()


def test_the_cap_binds():
    n = 600
    v = np.full(n, 1000.0)
    v[400] = 1e-6                          # one effectively dead bar
    c = spread.cost_bp_series(_bars(n, vol=v), base_bp=1.0, cap=4.0)
    assert c.max() <= 4.0 + 1e-9


def test_units_do_not_matter_only_the_ratio():
    """Gold and the index report lots, not tick counts."""
    n = 600
    rng = np.random.default_rng(5)
    v = rng.lognormal(6, 0.8, n)
    a = spread.cost_bp_series(_bars(n, vol=v))
    b = spread.cost_bp_series(_bars(n, vol=v / 10_000.0))
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_cost_series_rejects_nonsense():
    b = _bars(100)
    for kw in ({"base_bp": 0}, {"cap": 0.5}, {"power": 0}, {"power": 3}):
        with pytest.raises(ValueError):
            spread.cost_bp_series(b, **kw)


def test_activity_refuses_a_dead_instrument():
    b = _bars(100, vol=0.0)
    with pytest.raises(ValueError, match="no positive volume"):
        spread.activity(b)


# ------------------------------------------------------------
# Blackouts
# ------------------------------------------------------------

def _et(*s):
    return pd.DatetimeIndex(list(s)).tz_localize("America/New_York")


def test_the_rollover_is_blacked_out():
    out = spread.blackout(_et("2024-01-02 17:00", "2024-01-02 16:50",
                              "2024-01-02 12:00"))
    assert bool(out.iloc[0]) and bool(out.iloc[1])
    assert not bool(out.iloc[2])


def test_the_window_opens_before_the_release_not_at_it():
    """Providers widen ahead of the print. Blacking out only the
    minute after it is the mistake this pins."""
    out = spread.blackout(_et("2024-01-02 08:25", "2024-01-02 08:30",
                              "2024-01-02 08:45", "2024-01-02 09:05"))
    assert bool(out.iloc[0])               # five minutes before
    assert bool(out.iloc[1]) and bool(out.iloc[2])
    assert not bool(out.iloc[3])           # well clear afterwards


def test_the_mask_excludes_rather_than_keeps():
    """Named and returned so `entries[~blackout(...)]` reads right.
    An inverted mask would silently trade ONLY the news."""
    assert spread.blackout(_et("2024-01-02 17:00")).iloc[0]
    assert not spread.blackout(_et("2024-01-02 11:30")).iloc[0]


def test_each_component_can_be_switched_off():
    t = _et("2024-01-02 17:00")
    assert not spread.blackout(t, rollover=False, releases=True).iloc[0]
    assert spread.blackout(t, rollover=True, releases=False).iloc[0]


# ------------------------------------------------------------
# Wide-spread gate
# ------------------------------------------------------------

def test_wide_spread_threshold_is_trailing_not_full_sample():
    """A full-sample median is computed from bars the trade could
    not have seen."""
    b = _bars(3000)
    w = spread.wide_spread(b, multiple=3.0)
    assert w.iloc[:200].sum() == 0         # no threshold yet, no firing


def test_wide_spread_rejects_a_multiple_of_one():
    with pytest.raises(ValueError, match="must exceed 1"):
        spread.wide_spread(_bars(300), multiple=1.0)
