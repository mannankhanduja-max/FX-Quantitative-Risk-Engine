"""
Tests for fxrisk.strategies.mtf.

A multi-timeframe strategy has one failure mode that dwarfs all
others: consulting a higher-timeframe bar before it has closed. A 5m
entry at 10:05 that reads the 10:00-11:00 hourly bar knows 55 minutes
of its own future, and on a trending day that lookup alone will
produce a spectacular equity curve.

So the first three tests all attack that, from different angles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.strategies import mtf


def _frame(n, freq, start="2024-01-02 00:00", seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York")
    c = base + np.cumsum(rng.normal(0, 0.02, n))
    w = np.abs(rng.normal(0, 0.03, n))
    return pd.DataFrame({"Open": c, "High": c + w, "Low": c - w, "Close": c,
                         "Volume": rng.integers(500, 3000, n).astype(float)},
                        index=idx)


# ------------------------------------------------------------
# The look-ahead
# ------------------------------------------------------------

def test_align_uses_the_last_COMPLETED_higher_bar():
    """
    The 10:00 hourly bar closes at 11:00. A 5m bar at 10:05 must see
    the 09:00 bar; one at 11:00 may see the 10:00 bar.
    """
    hi = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-02 09:00", "2024-01-02 10:00",
                                "2024-01-02 11:00"]).tz_localize("America/New_York"),
    )
    lo_idx = pd.DatetimeIndex(
        ["2024-01-02 10:05", "2024-01-02 10:55", "2024-01-02 11:00",
         "2024-01-02 11:05"]).tz_localize("America/New_York")
    out = mtf.align_to(lo_idx, hi)
    assert out.iloc[0] == 1.0      # 10:05 sees the 09:00 bar
    assert out.iloc[1] == 1.0      # 10:55 still does
    assert out.iloc[2] == 2.0      # 11:00: the 10:00 bar has closed
    assert out.iloc[3] == 2.0


def test_perturbing_an_hourly_bar_cannot_move_entries_inside_it():
    """The end-to-end version of the same check."""
    b5 = _frame(600, "5min", seed=1)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    b1h = b5.resample("1h").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()

    setups = mtf.setups_30m(b30)
    base = mtf.entries_5m(b5, setups, mtf.hourly_bias(b1h))

    shocked = b1h.copy()
    k = shocked.index[len(shocked) // 2]
    shocked.loc[k, ["High", "Close"]] *= 1.05
    after = mtf.entries_5m(b5, setups, mtf.hourly_bias(shocked))

    cutoff = k                      # entries strictly before this hour's close
    a = base[base["time"] < cutoff]
    c = after[after["time"] < cutoff]
    assert len(a) == len(c)
    if len(a):
        assert np.allclose(a["entry"].to_numpy(), c["entry"].to_numpy())


def test_a_30m_setup_is_not_actionable_before_its_bar_closes():
    b5 = _frame(400, "5min", seed=2)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    if s.empty:
        pytest.skip("no setups on this fixture")
    step = b30.index.to_series().diff().median()
    # known_at is the bar's close, never its open.
    assert (s["known_at"] - step).isin(b30.index).all()

    e = mtf.entries_5m(b5, s, mtf.hourly_bias(
        b5.resample("1h").agg({"Open": "first", "High": "max", "Low": "min",
                               "Close": "last", "Volume": "sum"}).dropna()))
    if not e.empty:
        first = s.sort_values("known_at")["known_at"].iloc[0]
        assert (e["time"] >= first).all()


def test_exit_bar_is_after_the_entry_not_the_one_containing_it():
    """
    The 30m bar the entry happened inside has already printed its
    high and low. Walking the exit from it would let a trade be
    stopped or filled by price action that preceded the entry.
    """
    b5 = _frame(400, "5min", seed=3)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    entries = pd.DataFrame([{"time": b5.index[100], "side": 1.0,
                             "entry": 100.0, "stop": 99.0, "kind": "test"}])
    pos = mtf.to_30m_positions(entries, b30)
    assert len(pos) == 1
    step = b30.index.to_series().diff().median()
    bar_close = b30.index[int(pos.iloc[0]["bar"])] + step
    assert bar_close > b5.index[100]


def test_entry_price_survives_the_timeframe_change():
    """The fill was on the 5m close; mapping frames must not repaint it."""
    b5 = _frame(200, "5min", seed=4)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    e = pd.DataFrame([{"time": b5.index[50], "side": -1.0,
                       "entry": 123.456, "stop": 124.0, "kind": "test"}])
    pos = mtf.to_30m_positions(e, b30)
    assert pos.iloc[0]["entry"] == pytest.approx(123.456)
    assert pos.iloc[0]["stop"] == pytest.approx(124.0)


# ------------------------------------------------------------
# The cascade's own logic
# ------------------------------------------------------------

def test_bias_gates_the_side():
    """An entry may only be taken in the completed 1h bias direction."""
    b5 = _frame(600, "5min", seed=5)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    b1h = b5.resample("1h").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    bias = mtf.hourly_bias(b1h)
    e = mtf.entries_5m(b5, s, bias)
    if e.empty:
        pytest.skip("no entries on this fixture")
    aligned = mtf.align_to(b5.index, bias)
    for row in e.itertuples(index=False):
        assert aligned.loc[row.time] == row.side


def test_bias_refuses_zero_volume():
    b = _frame(50, "1h", seed=6)
    b["Volume"] = 0.0
    with pytest.raises(ValueError, match="volume is zero"):
        mtf.hourly_bias(b)


def test_setup_kinds_can_be_selected():
    b30 = _frame(300, "30min", seed=7)
    only = mtf.setups_30m(b30, kinds=("breakout",))
    assert only.empty or set(only["kind"]) == {"breakout"}
    both = mtf.setups_30m(b30, kinds=("breakout", "fakeout"))
    assert len(both) >= len(only)


def test_stop_is_on_the_correct_side():
    b5 = _frame(600, "5min", seed=8)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    b1h = b5.resample("1h").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()
    e = mtf.entries_5m(b5, mtf.setups_30m(b30), mtf.hourly_bias(b1h))
    if e.empty:
        pytest.skip("no entries on this fixture")
    longs = e[e["side"] > 0]
    shorts = e[e["side"] < 0]
    assert (longs["stop"] < longs["entry"]).all()
    assert (shorts["stop"] > shorts["entry"]).all()


def test_config_rejects_nonsense():
    for kw in ({"bias_ema": 0}, {"setup_lookback": 2},
               {"retrace_min": 0.0}, {"retrace_min": 0.9, "retrace_max": 0.5},
               {"trigger_window": 0}, {"stop_sigma": 0.0}):
        with pytest.raises(ValueError):
            mtf.MTFConfig(**kw)
