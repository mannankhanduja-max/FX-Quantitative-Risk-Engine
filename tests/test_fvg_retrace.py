"""
Tests for the FVG-confirmed breakout/retracement strategy.

`fxrisk.strategies.breakout_retrace` is already pinned by
`tests/test_breakout_retrace.py`: the trend filter, the close-based
breakout, the depth band, the confirmation close and the bracket are
identical here, and this file reuses its exact fixture shapes rather
than inventing new ones, so a reader can see directly that nothing but
the one new gate changed. What's new and needs its own tests is that
gate: a retracement only arms if it ALSO touches a fair value gap, and
the gate has to be causal through the whole pipeline, not just inside
`smc.py` in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.strategies import fvg_retrace as fr
from fxrisk.strategies.smc import fair_value_gaps

# Same shape as tests/test_breakout_retrace.py: 25 flat bars (lookback
# 20 needs 21), then a breakout, a pullback and a resumption.
QUIET = [99.9, 100.0] * 12 + [99.9]      # 25 bars, high 100.0, low 99.9


def _bars(h=None, l=None, c=None, vol=1000, start="2026-01-05 09:30", freq="1min"):
    n = len(c)
    idx = pd.date_range(start, periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": c, "High": h if h is not None else c,
            "Low": l if l is not None else c, "Close": c,
            "Volume": [vol] * n if np.isscalar(vol) else vol,
        },
        index=idx,
    )


def _standard_bars():
    """Breakout at bar 25, pullback at bar 26 (low 101.0-high 103.0),
    resumption at bar 27. Identical numbers to
    test_breakout_retrace.test_close_beyond_the_level_arms_a_setup."""
    c = QUIET + [103.0, 101.2, 103.5]
    h = QUIET + [103.0, 103.0, 103.6]
    lo = QUIET + [99.9, 101.0, 101.2]
    return _bars(h=h, l=lo, c=c)


def _zone(known_bar, lo, hi, side=1.0):
    return pd.DataFrame([{"known_bar": known_bar, "side": side, "lo": lo,
                          "hi": hi, "formed_bar": known_bar - 1}])


def _cfg(**kw):
    return fr.FVGRetraceConfig(
        lookback=20, min_fraction=0.33, max_fraction=1.0,
        retrace_max_bars=6, use_trend_filter=False, **kw,
    )


def test_config_rejects_nonsense():
    with pytest.raises(ValueError, match="lookback"):
        fr.FVGRetraceConfig(lookback=1)
    with pytest.raises(ValueError, match="min_fraction"):
        fr.FVGRetraceConfig(min_fraction=0.0)
    with pytest.raises(ValueError, match="stop_min_sigma"):
        fr.FVGRetraceConfig(stop_min_sigma=0.0)


def test_default_smc_config_is_filled_in_by_post_init():
    from fxrisk.strategies.smc import SMCConfig
    assert isinstance(fr.FVGRetraceConfig().smc, SMCConfig)


# ------------------------------------------------------------
# The gate this module adds, isolated with an explicit zone frame -
# the depth band alone is satisfied in every case below (it is the
# same price path `breakout_retrace` enters on); only the zone changes.
# ------------------------------------------------------------

def test_a_retracement_that_misses_every_zone_does_not_arm():
    bars = _standard_bars()
    zones = _zone(known_bar=25, lo=90.0, hi=91.0)  # nowhere near the pullback
    assert fr.find_setups(bars, _cfg(), zone_frame=zones).empty


def test_a_retracement_into_a_known_zone_arms_and_enters():
    bars = _standard_bars()
    # Pullback bar's range is [101.0, 103.0]; this zone overlaps it and
    # is known at bar 25, strictly before the pullback bar (26).
    zones = _zone(known_bar=25, lo=101.0, hi=101.3)
    s = fr.find_setups(bars, _cfg(), zone_frame=zones)
    assert len(s) == 1
    row = s.iloc[0]
    assert row["side"] == 1.0
    assert row["entry"] == pytest.approx(103.5)
    assert row["zone_touch_bar"] == 26


def test_with_no_zone_frame_passed_it_computes_its_own():
    """Default behaviour: compute `smc.fair_value_gaps` internally when
    the caller does not supply one. A plain flat-then-breakout price
    path has no gaps anywhere (every bar overlaps its neighbour), so
    this should come back empty exactly like the explicit-empty case."""
    bars = _standard_bars()
    assert fr.find_setups(bars, _cfg()).empty


def test_removing_the_gap_removes_the_entry_everything_else_equal():
    bars = _standard_bars()
    with_zone = fr.find_setups(
        bars, _cfg(), zone_frame=_zone(known_bar=25, lo=101.0, hi=101.3)
    )
    assert not with_zone.empty
    without_zone = fr.find_setups(
        bars, _cfg(), zone_frame=fair_value_gaps(bars).iloc[0:0]
    )
    assert without_zone.empty


def test_a_zone_known_only_as_of_the_touching_bar_does_not_count():
    """The one-bar-leak guard, exercised end to end: `smc.in_zone`
    refuses `known_bar == bar`, and that refusal has to survive being
    called from inside this module's loop, not just in isolation."""
    bars = _standard_bars()
    same_bar = _zone(known_bar=26, lo=101.0, hi=101.3)  # == the pullback bar
    assert fr.find_setups(bars, _cfg(), zone_frame=same_bar).empty


def test_a_zone_on_the_wrong_side_does_not_count():
    bars = _standard_bars()
    wrong_side = _zone(known_bar=25, lo=101.0, hi=101.3, side=-1.0)
    assert fr.find_setups(bars, _cfg(), zone_frame=wrong_side).empty


def test_shorts_are_the_mirror_of_longs():
    c = QUIET + [97.0, 98.8, 96.5]
    h = QUIET + [100.0, 99.0, 98.8]
    lo = QUIET + [97.0, 97.0, 96.4]
    bars = _bars(h=h, l=lo, c=c)
    # Pullback bar (26) range: [97.0, 99.0].
    zones = _zone(known_bar=25, lo=98.7, hi=99.0, side=-1.0)
    s = fr.find_setups(bars, _cfg(), zone_frame=zones)
    assert len(s) == 1
    assert s.iloc[0]["side"] == -1.0


def test_causality_a_future_price_move_cannot_change_an_already_made_entry():
    bars = _standard_bars()
    zones = _zone(known_bar=25, lo=101.0, hi=101.3)
    baseline = fr.find_setups(bars, _cfg(), zone_frame=zones)
    assert not baseline.empty
    entry_bar = int(baseline.iloc[0]["bar"])

    mutated = bars.copy()
    cols = mutated.columns.get_indexer(["High", "Low", "Close"])
    mutated.iloc[entry_bar + 1:, cols] *= 1.5
    again = fr.find_setups(mutated, _cfg(), zone_frame=zones)
    assert not again.empty
    pd.testing.assert_series_equal(
        baseline.iloc[0][["bar", "side", "entry", "stop"]],
        again.iloc[0][["bar", "side", "entry", "stop"]],
        check_names=False,
    )


def test_describe_on_empty_and_nonempty():
    assert fr.describe(pd.DataFrame()) == "  no setups"
    bars = _standard_bars()
    zones = _zone(known_bar=25, lo=101.0, hi=101.3)
    out = fr.describe(fr.find_setups(bars, _cfg(), zone_frame=zones))
    assert "setups" in out and "1" in out
