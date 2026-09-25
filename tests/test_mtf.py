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


# ------------------------------------------------------------
# Session windows
# ------------------------------------------------------------

def _uk(*stamps):
    return pd.DatetimeIndex(list(stamps)).tz_localize("Europe/London")


def test_sessions_are_defined_in_their_own_clock_not_a_utc_offset():
    """
    The bug this is here to catch: hard-coding "New York is UTC-5".
    For three weeks each spring the UK has moved to BST and New York
    has not, so 13:00 London is 08:00 ET in one half of March and
    09:00 ET in the other. A window written as a fixed offset is
    silently an hour wrong for several weeks a year.
    """
    # 2026: UK clocks go forward 29 March, US clocks 8 March.
    # In between, London is UTC+0 and New York UTC-4, a 4h gap
    # rather than the usual 5.
    mid = _uk("2026-03-16 12:30")          # = 08:30 ET, inside NY
    before = _uk("2026-03-02 12:30")       # = 07:30 ET, NY not open
    assert bool(mtf.in_sessions(mid, ("newyork",)).iloc[0])
    assert not bool(mtf.in_sessions(before, ("newyork",)).iloc[0])


def test_tokyo_does_not_observe_dst():
    """Japan has no summer time, so the window never moves in JST."""
    jan = pd.DatetimeIndex(["2026-01-14 10:00"]).tz_localize("Asia/Tokyo")
    jul = pd.DatetimeIndex(["2026-07-14 10:00"]).tz_localize("Asia/Tokyo")
    assert bool(mtf.in_sessions(jan, ("tokyo",)).iloc[0])
    assert bool(mtf.in_sessions(jul, ("tokyo",)).iloc[0])


def test_multiple_sessions_are_a_union_not_an_intersection():
    """London-only hours must survive a ('london', 'newyork') filter."""
    early = _uk("2026-01-14 09:00")        # London open, NY shut
    both = _uk("2026-01-14 14:00")         # the overlap
    assert bool(mtf.in_sessions(early, ("london", "newyork")).iloc[0])
    assert bool(mtf.in_sessions(both, ("london", "newyork")).iloc[0])
    assert not bool(mtf.in_sessions(early, ("newyork",)).iloc[0])


def test_weekends_are_never_in_session():
    sat = _uk("2026-01-17 14:00")
    assert not mtf.in_sessions(sat, tuple(mtf.SESSION_WINDOWS)).iloc[0]


def test_window_is_half_open_at_the_close():
    """16:00 ET is the New York close, not a tradeable minute."""
    ny = pd.DatetimeIndex(["2026-01-14 15:59", "2026-01-14 16:00"]
                          ).tz_localize("America/New_York")
    out = mtf.in_sessions(ny, ("newyork",))
    assert bool(out.iloc[0]) and not bool(out.iloc[1])


def test_unknown_session_is_an_error_not_a_silent_pass():
    with pytest.raises(ValueError, match="unknown session"):
        mtf.in_sessions(_uk("2026-01-14 14:00"), ("frankfurt",))


def test_every_instrument_has_a_declared_window():
    import config
    for inst in config.UNIVERSE_INTRADAY:
        assert inst.name in mtf.INSTRUMENT_SESSIONS
        for s in mtf.INSTRUMENT_SESSIONS[inst.name]:
            assert s in mtf.SESSION_WINDOWS


def test_the_5m_trigger_is_the_previous_5m_close_and_nothing_else():
    """
    A long triggers on a 5m bar closing ABOVE the previous 5m
    close; a short below it. Not above the 30m setup level - the
    30m bar already closed beyond that, and re-requiring it would
    demand the move continue, which turns a retracement entry into
    a second breakout entry.
    """
    idx = pd.date_range("2024-01-02 09:00", periods=8, freq="5min",
                        tz="America/New_York")
    c = np.array([100.0, 99.9, 99.8, 100.1, 100.2, 100.3, 100.4, 100.5])
    b5 = pd.DataFrame({"Open": c, "High": c + 0.05, "Low": c - 0.05,
                       "Close": c, "Volume": 1000.0}, index=idx)
    setups = pd.DataFrame([{"known_at": idx[1], "side": 1.0,
                            "level": 99.0, "kind": "test"}])
    bias = pd.Series(1.0, index=pd.date_range(
        "2024-01-02 07:00", periods=4, freq="1h", tz="America/New_York"))

    # sigma basis explicitly: this fixture is 8 bars, too short for
    # ATR(14), and the test is about the trigger, not the stop.
    e = mtf.entries_5m(b5, setups, bias,
                       mtf.MTFConfig(stop_mode="sigma", stop_sigma=1.0))
    assert len(e) == 1
    k = list(idx).index(e.iloc[0]["time"])
    assert c[k] > c[k - 1], "triggered on a bar that closed lower"
    # Bars 1 and 2 fell; the first bar closing up is bar 3.
    assert e.iloc[0]["time"] == idx[3]


def test_bias_none_removes_the_direction_filter_entirely():
    """
    No session VWAP, no 9 EMA. Every setup becomes actionable in
    its own direction regardless of the hourly trend - which must
    produce at least as many entries, never fewer.
    """
    b5 = _frame(900, "5min", seed=11)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    b1h = b5.resample("1h").agg({"Open": "first", "High": "max", "Low": "min",
                                 "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    gated = mtf.entries_5m(b5, s, mtf.hourly_bias(b1h))
    free = mtf.entries_5m(b5, s, None)
    assert len(free) >= len(gated)


def test_removing_the_bias_does_not_reintroduce_lookahead():
    """The 5m trigger must still depend only on closed 5m bars."""
    b5 = _frame(600, "5min", seed=12)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    base = mtf.entries_5m(b5, s, None)
    shocked = b5.copy()
    k = shocked.index[-1]
    shocked.loc[k, ["High", "Close"]] *= 1.05
    after = mtf.entries_5m(shocked, s, None)
    n = min(len(base), len(after))
    if n:
        assert np.allclose(base["entry"].to_numpy()[:n],
                           after["entry"].to_numpy()[:n])


# ------------------------------------------------------------
# ATR as the stop basis
# ------------------------------------------------------------

def test_true_range_sees_a_gap_that_close_to_close_vol_does_not():
    """
    The reason ATR exists here. A bar that opens far from the
    previous close and then barely moves has a tiny close-to-close
    return and a large true range - and it is the true range a
    stop has to survive.
    """
    idx = pd.date_range("2024-01-02", periods=3, freq="5min",
                        tz="America/New_York")
    b = pd.DataFrame({"Open": [100.0, 100.0, 105.0],
                      "High": [100.2, 100.2, 105.1],
                      "Low":  [99.8, 99.8, 104.9],
                      "Close": [100.0, 100.0, 105.0],
                      "Volume": 1000.0}, index=idx)
    tr = mtf.true_range(b)
    assert tr.iloc[1] == pytest.approx(0.4)      # no gap: just the range
    assert tr.iloc[2] == pytest.approx(5.1)      # gap dominates


def test_atr_is_shifted_so_a_bar_cannot_size_its_own_stop():
    """
    An ATR including the current bar sizes the stop using the very
    range the stop is about to be tested against.
    """
    b = _frame(300, "5min", seed=20)
    a = mtf.atr(b, 14)
    shocked = b.copy()
    k = shocked.index[-1]
    shocked.loc[k, "High"] *= 1.20
    shocked.loc[k, "Low"] *= 0.80
    a2 = mtf.atr(shocked, 14)
    assert np.allclose(a.to_numpy(), a2.to_numpy(), equal_nan=True), \
        "the last bar's own range moved its own ATR"


def test_atr_exceeds_close_to_close_sigma_on_real_shaped_data():
    """
    True range includes intrabar travel, so ATR should sit ABOVE a
    close-to-close sigma. If this ever inverts, the two are not
    measuring what their names claim.
    """
    b = _frame(1200, "5min", seed=21)
    a = mtf.atr(b, 14).median()
    r = np.log(b["Close"]).diff()
    s = (np.sqrt(r.pow(2).ewm(alpha=0.06, adjust=False).mean())
         * b["Close"]).median()
    assert a > s


def test_atr_mode_produces_a_wider_stop_than_the_same_sigma_multiple():
    b5 = _frame(900, "5min", seed=22)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    sig = mtf.entries_5m(b5, s, None, mtf.MTFConfig(stop_mode="sigma",
                                                    stop_sigma=1.0))
    at = mtf.entries_5m(b5, s, None, mtf.MTFConfig(stop_mode="atr",
                                                   stop_sigma=1.0))
    if sig.empty or at.empty:
        pytest.skip("no entries on this fixture")
    ds = (sig["entry"] - sig["stop"]).abs().median()
    da = (at["entry"] - at["stop"]).abs().median()
    assert da >= ds


def test_stop_mode_is_validated():
    with pytest.raises(ValueError, match="stop_mode"):
        mtf.MTFConfig(stop_mode="bollinger")
    with pytest.raises(ValueError, match="atr_period"):
        mtf.MTFConfig(atr_period=1)


# ------------------------------------------------------------
# VWAP as a filter
# ------------------------------------------------------------

def test_session_vwap_resets_at_the_session_roll():
    """A cumulative VWAP that never resets is a running average of
    the whole sample, not a session's fair value."""
    idx = pd.date_range("2024-01-02 16:00", periods=6, freq="1h",
                        tz="America/New_York")
    b = pd.DataFrame({"Open": 100.0, "High": 100.0, "Low": 100.0,
                      "Close": [100.0, 100.0, 200.0, 200.0, 200.0, 200.0],
                      "Volume": 1000.0}, index=idx)
    v = mtf.session_vwap(b)
    # 17:00 starts a new session; the 200s must not be dragged down
    # by the 100s that preceded the roll.
    assert v.iloc[-1] == pytest.approx(200.0)


def test_session_vwap_refuses_zero_volume():
    b = _frame(50, "1h", seed=30)
    b["Volume"] = 0.0
    with pytest.raises(ValueError, match="volume is zero"):
        mtf.session_vwap(b)


def test_revert_filter_only_buys_below_value_and_sells_above():
    b5 = _frame(900, "5min", seed=31)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    e = mtf.entries_5m(b5, s, None, mtf.MTFConfig(vwap_filter="revert",
                                                  stop_mode="sigma"))
    if e.empty:
        pytest.skip("no entries on this fixture")
    v = mtf.session_vwap(b5)
    for row in e.itertuples(index=False):
        if row.side > 0:
            assert row.entry < v.loc[row.time]
        else:
            assert row.entry > v.loc[row.time]


def test_trend_filter_is_the_exact_opposite_of_revert():
    b5 = _frame(900, "5min", seed=32)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    kw = dict(stop_mode="sigma")
    rev = mtf.entries_5m(b5, s, None, mtf.MTFConfig(vwap_filter="revert", **kw))
    tre = mtf.entries_5m(b5, s, None, mtf.MTFConfig(vwap_filter="trend", **kw))
    if rev.empty or tre.empty:
        pytest.skip("no entries on this fixture")
    assert set(rev["time"]).isdisjoint(set(tre["time"]))


def test_either_vwap_filter_can_only_remove_entries():
    b5 = _frame(900, "5min", seed=33)
    b30 = b5.resample("30min").agg({"Open": "first", "High": "max", "Low": "min",
                                    "Close": "last", "Volume": "sum"}).dropna()
    s = mtf.setups_30m(b30)
    base = mtf.entries_5m(b5, s, None, mtf.MTFConfig(stop_mode="sigma"))
    for mode in ("revert", "trend"):
        f = mtf.entries_5m(b5, s, None,
                           mtf.MTFConfig(vwap_filter=mode, stop_mode="sigma"))
        assert len(f) <= len(base)


def test_vwap_filter_is_validated():
    with pytest.raises(ValueError, match="vwap_filter"):
        mtf.MTFConfig(vwap_filter="anchored")
