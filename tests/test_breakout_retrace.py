"""
Tests for the breakout/retracement strategy and the explicit
bracket walk with a breakeven stop.

These exist because the strategy cannot currently be run on real
data - Yahoo's intraday endpoint is unreachable from CI, and the
60-day window means the cache is never reproducible anyway. So
every mechanical claim the module makes is pinned here on
hand-built bars instead, where the right answer is known by
construction rather than by eyeballing a result.

The breakeven stop gets the most attention, because it is the
piece most likely to be silently wrong in a way that flatters the
backtest: an implementation that arms the stop INSIDE the bar
that triggers it turns losers into scratches for free and is
indistinguishable from a working rule until it is traded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.data import intraday
from fxrisk.risk import barriers as bar
from fxrisk.strategies import breakout_retrace as br


def _bars(o=None, h=None, l=None, c=None, vol=1000, start="2026-01-05 09:30",
          freq="15min"):
    n = len(c)
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": o if o is not None else c,
            "High": h if h is not None else c,
            "Low": l if l is not None else c,
            "Close": c,
            "Volume": [vol] * n if np.isscalar(vol) else vol,
        },
        index=idx,
    )


# ------------------------------------------------------------
# Session boundary and VWAP
# ------------------------------------------------------------

def test_session_rolls_at_six_pm_not_midnight():
    """
    Globex sessions span two calendar dates. Grouping on the
    calendar date would reset the VWAP at midnight, in the middle
    of the evening session - the single worst place to reset it.
    """
    idx = pd.DatetimeIndex(
        ["2026-01-05 17:45", "2026-01-05 18:15", "2026-01-05 23:45",
         "2026-01-06 02:00", "2026-01-06 09:30"]
    ).tz_localize("America/New_York")
    s = intraday.session_id(idx)
    assert s.iloc[0] != s.iloc[1]              # 18:00 starts a new session
    assert s.iloc[1] == s.iloc[2] == s.iloc[3] == s.iloc[4]


def test_session_vwap_resets_and_is_volume_weighted():
    c = [100.0, 100.0, 200.0, 200.0]
    idx = pd.DatetimeIndex(
        ["2026-01-05 09:30", "2026-01-05 09:45",
         "2026-01-05 18:00", "2026-01-05 18:15"]
    ).tz_localize("America/New_York")
    b = pd.DataFrame({"High": c, "Low": c, "Close": c,
                      "Volume": [1, 3, 1, 1]}, index=idx)
    v = br.session_vwap(b)
    assert v.iloc[0] == pytest.approx(100.0)
    assert v.iloc[1] == pytest.approx(100.0)
    # New session: the 200s must not be dragged toward 100.
    assert v.iloc[2] == pytest.approx(200.0)
    assert v.iloc[3] == pytest.approx(200.0)


def test_session_vwap_is_weighted_not_averaged():
    """A TWAP would give 150. A VWAP weighted 1:3 gives 175."""
    c = [100.0, 200.0]
    b = _bars(c=c, vol=[1, 3])
    assert br.session_vwap(b).iloc[1] == pytest.approx(175.0)


def test_session_vwap_refuses_zero_volume():
    """The FX-spot trap. A VWAP on zero volume is a TWAP lying."""
    b = _bars(c=[100.0, 101.0, 102.0], vol=0)
    with pytest.raises(ValueError, match="volume is zero"):
        br.session_vwap(b)


# ------------------------------------------------------------
# Breakout confirmation
# ------------------------------------------------------------

# A dead-flat stretch has zero EWMA volatility, and find_setups
# refuses to size a stop against sigma == 0 - correctly, since
# every stop distance would be the sigma floor of nothing. So the
# quiet part of each fixture oscillates by a tick instead of
# sitting still. The prior-20-bar high is still exactly 100.0.
QUIET = [99.9, 100.0] * 12 + [99.9]      # 25 bars, high 100.0, low 99.9


def test_a_wick_through_the_level_is_not_a_breakout():
    """
    The most important gate in the module. A bar that spikes
    through the level and closes back inside is a stop run, and
    on 15-minute bars that is most of what looks like a breakout.
    """
    c = QUIET + [99.95]                 # closes back inside
    h = QUIET + [108.0]                 # big wick above
    lo = QUIET + [99.9]
    b = _bars(h=h, l=lo, c=c)
    cfg = br.SetupConfig(lookback=20, use_trend_filter=False)
    assert br.find_setups(b, cfg).empty


def test_close_beyond_the_level_arms_a_setup():
    """Flat, then a decisive close up, a pullback, then resumption."""
    c = QUIET + [103.0, 101.2, 103.5]
    h = QUIET + [103.0, 103.0, 103.6]
    lo = QUIET + [99.9, 101.0, 101.2]
    b = _bars(h=h, l=lo, c=c)
    cfg = br.SetupConfig(lookback=20, min_fraction=0.33, max_fraction=1.0,
                         retrace_max_bars=6, use_trend_filter=False)
    s = br.find_setups(b, cfg)
    assert len(s) == 1
    assert s.iloc[0]["side"] == 1.0
    assert s.iloc[0]["entry"] == pytest.approx(103.5)


def test_a_retracement_through_the_level_is_a_failed_breakout():
    """Past max_fraction price is back through the level it broke."""
    c = QUIET + [103.0, 99.0, 103.5]
    h = QUIET + [103.0, 103.0, 103.6]
    lo = QUIET + [99.9, 98.5, 99.0]
    b = _bars(h=h, l=lo, c=c)
    cfg = br.SetupConfig(lookback=20, max_fraction=1.0,
                         retrace_max_bars=6, use_trend_filter=False)
    assert br.find_setups(b, cfg).empty


def test_shorts_are_the_mirror_of_longs():
    c = QUIET + [97.0, 98.8, 96.5]
    h = QUIET + [100.0, 99.0, 98.8]
    lo = QUIET + [97.0, 97.0, 96.4]
    b = _bars(h=h, l=lo, c=c)
    cfg = br.SetupConfig(lookback=20, retrace_max_bars=6,
                         use_trend_filter=False)
    s = br.find_setups(b, cfg)
    assert len(s) == 1
    assert s.iloc[0]["side"] == -1.0


def test_config_rejects_nonsense():
    for kw in ({"lookback": 1}, {"min_fraction": 0.0},
               {"min_fraction": 0.9, "max_fraction": 0.5},
               {"max_fraction": 3.0}, {"stop_min_sigma": 0.0},
               {"retrace_max_bars": 0}):
        with pytest.raises(ValueError):
            br.SetupConfig(**kw)


# ------------------------------------------------------------
# The bracket, and the breakeven stop
# ------------------------------------------------------------

def _entry(bar_i, side, entry, stop):
    return pd.DataFrame([{"bar": bar_i, "side": side,
                          "entry": entry, "stop": stop}])


def test_two_to_one_target_pays_two_r():
    b = _bars(h=[100.0, 100.0, 120.0], l=[100.0, 100.0, 100.0],
              c=[100.0, 100.0, 120.0])
    t = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 95.0), rr=2.0, cost_bp=0.0)
    assert t.iloc[0]["outcome"] == "target"
    assert t.iloc[0]["net_R"] == pytest.approx(2.0)


def test_stop_pays_minus_one_r_regardless_of_rr():
    b = _bars(h=[100.0, 100.0, 100.0], l=[100.0, 100.0, 90.0],
              c=[100.0, 100.0, 94.0])
    t = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 95.0), rr=2.0, cost_bp=0.0)
    assert t.iloc[0]["outcome"] == "stop"
    assert t.iloc[0]["net_R"] == pytest.approx(-1.0)


def test_breakeven_stop_turns_a_loser_into_a_scratch():
    """
    Bar 1 closes above the trigger, arming the stop at entry.
    Bar 2 collapses. Without the rule this is -1R; with it, 0R
    before costs.
    """
    b = _bars(h=[100.0, 102.0, 101.0], l=[100.0, 101.0, 90.0],
              c=[100.0, 102.0, 91.0])
    e = _entry(0, 1.0, 100.0, 95.0)

    control = bar.walk_explicit(b, e, rr=2.0, cost_bp=0.0, breakeven_frac=None)
    assert control.iloc[0]["outcome"] == "stop"
    assert control.iloc[0]["net_R"] == pytest.approx(-1.0)

    armed = bar.walk_explicit(b, e, rr=2.0, cost_bp=0.0, breakeven_frac=0.01)
    assert armed.iloc[0]["outcome"] == "breakeven"
    assert bool(armed.iloc[0]["be_armed"])
    assert armed.iloc[0]["net_R"] == pytest.approx(0.0)


def test_breakeven_stop_also_costs_winners():
    """
    The half everyone forgets. Price arms the stop, retests entry,
    and only then runs to target. The breakeven rule takes a
    scratch where the control takes +2R. If this test ever starts
    passing with both at +2R, the stop is not really armed.
    """
    b = _bars(
        h=[100.0, 102.0, 100.5, 120.0],
        l=[100.0, 101.0, 99.0, 100.0],
        c=[100.0, 102.0, 99.5, 120.0],
    )
    e = _entry(0, 1.0, 100.0, 95.0)

    control = bar.walk_explicit(b, e, rr=2.0, cost_bp=0.0, breakeven_frac=None)
    assert control.iloc[0]["outcome"] == "target"
    assert control.iloc[0]["net_R"] == pytest.approx(2.0)

    armed = bar.walk_explicit(b, e, rr=2.0, cost_bp=0.0, breakeven_frac=0.01)
    assert armed.iloc[0]["outcome"] == "breakeven"
    assert armed.iloc[0]["net_R"] == pytest.approx(0.0)


def test_the_stop_arms_at_the_close_never_inside_the_bar():
    """
    The flattering bug. Bar 1 TOUCHES the trigger intrabar (high
    102) but closes below it, then reverses. An implementation
    that armed on the high would scratch this; the honest one
    takes the full loss.
    """
    b = _bars(h=[100.0, 102.0, 100.0], l=[100.0, 99.5, 90.0],
              c=[100.0, 100.2, 91.0])
    t = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 95.0),
                          rr=2.0, cost_bp=0.0, breakeven_frac=0.01)
    assert not bool(t.iloc[0]["be_armed"])
    assert t.iloc[0]["outcome"] == "stop"
    assert t.iloc[0]["net_R"] == pytest.approx(-1.0)


def test_breakeven_is_mirrored_on_the_short_side():
    b = _bars(h=[100.0, 99.0, 110.0], l=[100.0, 98.0, 99.0],
              c=[100.0, 98.0, 109.0])
    t = bar.walk_explicit(b, _entry(0, -1.0, 100.0, 105.0),
                          rr=2.0, cost_bp=0.0, breakeven_frac=0.01)
    assert bool(t.iloc[0]["be_armed"])
    assert t.iloc[0]["outcome"] == "breakeven"
    assert t.iloc[0]["net_R"] == pytest.approx(0.0)


def test_both_barriers_in_one_bar_still_resolves_against_the_trade():
    b = _bars(h=[100.0, 125.0], l=[100.0, 75.0], c=[100.0, 100.0])
    t = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 95.0), rr=2.0, cost_bp=0.0)
    assert t.iloc[0]["outcome"] == "stop"
    assert bool(t.iloc[0]["ambiguous"])


def test_time_exit_marks_to_the_close():
    b = _bars(h=[100.0] * 6, l=[100.0] * 6, c=[100.0, 100.5, 100.5,
                                               100.5, 100.5, 100.5])
    t = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 95.0), rr=2.0,
                          max_bars=3, cost_bp=0.0)
    assert t.iloc[0]["outcome"] == "time"
    assert t.iloc[0]["bars_held"] == 3


# ------------------------------------------------------------
# Costs and the hurdle
# ------------------------------------------------------------

def test_the_two_to_one_hurdle_is_above_one_third():
    """
    Folklore says 33.3%. It ignores the round trip, which is paid
    whether the trade wins or loses.
    """
    naive = bar.breakeven_win_rate(0.0, rr=2.0)
    real = bar.breakeven_win_rate(0.04, rr=2.0)
    assert naive == pytest.approx(1 / 3)
    assert real > naive
    assert real == pytest.approx(1.04 / 3)


def test_cost_in_r_rises_as_the_stop_tightens():
    """A tight stop is the expensive choice, not the safe one."""
    b = _bars(h=[100.0] * 4, l=[100.0] * 4, c=[100.0] * 4)
    tight = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 99.0),
                              rr=2.0, max_bars=2, cost_bp=1.0)
    wide = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 90.0),
                             rr=2.0, max_bars=2, cost_bp=1.0)
    assert tight.iloc[0]["cost_R"] > wide.iloc[0]["cost_R"]
    assert tight.iloc[0]["cost_R"] == pytest.approx(0.02)


def test_walk_explicit_validates_its_inputs():
    b = _bars(c=[100.0, 101.0])
    e = _entry(0, 1.0, 100.0, 95.0)
    with pytest.raises(ValueError):
        bar.walk_explicit(b, e, rr=0.0)
    with pytest.raises(ValueError):
        bar.walk_explicit(b, e, rr=2.0, breakeven_frac=-0.01)
    with pytest.raises(ValueError):
        bar.walk_explicit(pd.DataFrame({"Close": [1.0, 2.0]}), e)


def test_a_zero_width_stop_is_skipped_not_divided_by():
    b = _bars(c=[100.0, 101.0, 102.0])
    t = bar.walk_explicit(b, _entry(0, 1.0, 100.0, 100.0), rr=2.0)
    assert t.empty


def test_summarise_on_no_trades():
    assert bar.summarise_explicit(pd.DataFrame(), rr=2.0) == {"trades": 0}


# ------------------------------------------------------------
# Config sanity
# ------------------------------------------------------------

def test_every_intraday_instrument_has_a_sane_breakeven_scale():
    """
    The four conversions in config are the most fragile numbers in
    the project - a pip is not a unit that survives four quote
    conventions. They should all land in single-digit basis
    points; anything outside that means an arithmetic slip.
    """
    import config as cfg
    assert len(cfg.UNIVERSE_INTRADAY) == 4
    for inst in cfg.UNIVERSE_INTRADAY:
        assert 1.0 < inst.be_bp < 20.0, inst.name
        assert inst.proxy_for
    names = {i.proxy_for for i in cfg.UNIVERSE_INTRADAY}
    assert names == {"EUR/USD", "XAU/USD", "NAS100", "USD/JPY"}


def test_overlapping_entries_are_dropped_not_stacked():
    """
    A rule that fires on consecutive bars in a trend produces
    entries that overlap. Counting each at one unit of risk levers
    the book without saying so: three overlapping trades is three
    units at risk, not one. The default must drop them.
    """
    n = 12
    b = _bars(h=[100.0 + i for i in range(n)],
              l=[99.0 + i for i in range(n)],
              c=[99.5 + i for i in range(n)])
    e = pd.DataFrame([
        {"bar": 0, "side": 1.0, "entry": 99.5, "stop": 89.5},
        {"bar": 1, "side": 1.0, "entry": 100.5, "stop": 90.5},
        {"bar": 2, "side": 1.0, "entry": 101.5, "stop": 91.5},
    ])

    kept = bar.walk_explicit(b, e, rr=2.0, max_bars=6, cost_bp=0.0)
    assert len(kept) < 3
    assert kept.attrs["dropped_overlapping"] == 3 - len(kept)

    stacked = bar.walk_explicit(b, e, rr=2.0, max_bars=6, cost_bp=0.0,
                                allow_overlap=True)
    assert len(stacked) == 3
    assert stacked.attrs["dropped_overlapping"] == 0


def test_a_later_non_overlapping_entry_still_gets_taken():
    """The guard must not swallow everything after the first trade."""
    n = 14
    c = [100.0, 100.0, 90.0] + [100.0] * (n - 3)
    b = _bars(h=[x + 0.5 for x in c], l=[x - 0.5 for x in c], c=c)
    e = pd.DataFrame([
        {"bar": 0, "side": 1.0, "entry": 100.0, "stop": 95.0},
        {"bar": 8, "side": 1.0, "entry": 100.0, "stop": 95.0},
    ])
    t = bar.walk_explicit(b, e, rr=2.0, max_bars=4, cost_bp=0.0)
    assert len(t) == 2
    assert t.attrs["dropped_overlapping"] == 0
