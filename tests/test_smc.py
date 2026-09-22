"""
Tests for fxrisk.strategies.smc.

Fair value gaps and order blocks are the easiest setups in this
repository to backtest into imaginary results, because both are
naturally drawn on a chart at the bar they FORMED on rather than
the bar that could first see them. An order block in particular
is identified by a break of structure that happens several bars
AFTER the block candle - index it by the candle and the backtest
knows the future. The first three tests attack exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.strategies import smc


def _bars(o, h, l, c):
    idx = pd.date_range("2024-01-02", periods=len(c), freq="30min",
                        tz="America/New_York")
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": 1000.0}, index=idx)


def _noise(n, seed=0, base=100.0, amp=0.1):
    rng = np.random.default_rng(seed)
    c = base + np.cumsum(rng.normal(0, amp, n))
    o = np.r_[base, c[:-1]]
    return list(o), list(np.maximum(o, c) + amp), list(np.minimum(o, c) - amp), list(c)


# ------------------------------------------------------------
# Causality
# ------------------------------------------------------------

def test_an_fvg_is_known_at_the_third_bar_not_the_middle_one():
    o, h, l, c = _noise(40, seed=1)
    # A clean bullish gap across bars 19, 20, 21.
    h[19], l[19] = 100.2, 99.8
    o[20], c[20], h[20], l[20] = 100.0, 101.5, 101.6, 99.9
    o[21], c[21], h[21], l[21] = 101.5, 101.8, 101.9, 100.9
    f = smc.fair_value_gaps(_bars(o, h, l, c), smc.SMCConfig(min_gap_sigma=0.0))
    hit = f[f["formed_bar"] == 20]
    assert len(hit) == 1
    assert hit.iloc[0]["known_bar"] == 21, "gap usable before its third bar closed"
    assert hit.iloc[0]["side"] == 1.0


def test_an_order_block_is_known_at_the_BREAK_not_at_the_candle():
    """
    The block candle comes first; the break that identifies it
    comes later. `known_bar` must be the break.
    """
    n = 40
    o = [100.0] * n; h = [100.4] * n; l = [99.6] * n; c = [100.0] * n
    o[30], c[30], h[30], l[30] = 100.0, 99.5, 100.1, 99.4   # the down candle
    for k in range(31, 34):                                  # the impulse
        o[k], c[k] = 99.6 + (k - 31) * 0.5, 100.1 + (k - 31) * 0.5
        h[k], l[k] = c[k] + 0.1, o[k] - 0.1
    ob = smc.order_blocks(_bars(o, h, l, c), smc.SMCConfig(ob_lookback=20))
    assert len(ob) >= 1
    row = ob.iloc[0]
    assert row["known_bar"] > row["formed_bar"], "block dated to its own candle"
    assert row["side"] == 1.0


@pytest.mark.parametrize("fn", ["fair_value_gaps", "order_blocks"])
def test_zones_do_not_depend_on_the_future(fn):
    o, h, l, c = _noise(300, seed=2)
    b = _bars(o, h, l, c)
    base = getattr(smc, fn)(b)

    shocked = b.copy()
    k = shocked.index[-1]
    shocked.loc[k, "High"] = 200.0
    shocked.loc[k, "Low"] = 50.0
    after = getattr(smc, fn)(shocked)

    if base.empty and after.empty:
        pytest.skip("no zones on this fixture")
    n = min(len(base), len(after))
    cols = ["known_bar", "side", "lo", "hi"]
    assert base[cols].iloc[:n].equals(after[cols].iloc[:n])


def test_in_zone_refuses_a_zone_confirmed_by_the_same_bar():
    """
    A gap confirmed at bar k's CLOSE cannot justify a decision
    taken from bar k. One bar of leak is still a leak.
    """
    z = pd.DataFrame([{"known_bar": 10, "side": 1.0, "lo": 99.0,
                       "hi": 101.0, "formed_bar": 9, "kind": "fvg"}])
    assert not smc.in_zone(z, 10, 1.0, 99.5, 100.5)
    assert smc.in_zone(z, 11, 1.0, 99.5, 100.5)


# ------------------------------------------------------------
# Gap geometry
# ------------------------------------------------------------

def test_overlapping_bars_are_not_a_gap():
    n = 30
    o, h, l, c = _noise(n, seed=3)
    h[19], l[19] = 100.5, 99.5
    h[21], l[21] = 100.4, 99.6          # fully overlaps bar 19
    f = smc.fair_value_gaps(_bars(o, h, l, c), smc.SMCConfig(min_gap_sigma=0.0))
    assert f[f["formed_bar"] == 20].empty


def test_a_bearish_gap_is_mirrored():
    o, h, l, c = _noise(40, seed=4)
    h[19], l[19] = 100.2, 99.8
    o[20], c[20], h[20], l[20] = 100.0, 98.5, 100.1, 98.4
    o[21], c[21], h[21], l[21] = 98.5, 98.2, 99.0, 98.1
    f = smc.fair_value_gaps(_bars(o, h, l, c), smc.SMCConfig(min_gap_sigma=0.0))
    hit = f[f["formed_bar"] == 20]
    assert len(hit) == 1 and hit.iloc[0]["side"] == -1.0
    assert hit.iloc[0]["lo"] == pytest.approx(99.0)
    assert hit.iloc[0]["hi"] == pytest.approx(99.8)


def test_a_gap_narrower_than_the_floor_is_ignored():
    o, h, l, c = _noise(40, seed=5)
    h[19], l[19] = 100.2, 99.8
    o[20], c[20], h[20], l[20] = 100.0, 100.4, 100.45, 99.9
    o[21], c[21], h[21], l[21] = 100.4, 100.5, 100.6, 100.2001
    tiny = smc.fair_value_gaps(_bars(o, h, l, c),
                               smc.SMCConfig(min_gap_sigma=50.0))
    assert tiny[tiny["formed_bar"] == 20].empty


# ------------------------------------------------------------
# Zones and membership
# ------------------------------------------------------------

def test_a_zone_ages_out():
    z = pd.DataFrame([{"known_bar": 10, "side": 1.0, "lo": 99.0,
                       "hi": 101.0, "formed_bar": 9, "kind": "fvg"}])
    cfg = smc.SMCConfig(zone_max_age=5)
    assert smc.in_zone(z, 14, 1.0, 100.0, 100.5, cfg)
    assert not smc.in_zone(z, 30, 1.0, 100.0, 100.5, cfg)


def test_the_side_must_match():
    z = pd.DataFrame([{"known_bar": 10, "side": 1.0, "lo": 99.0,
                       "hi": 101.0, "formed_bar": 9, "kind": "fvg"}])
    assert smc.in_zone(z, 12, 1.0, 100.0, 100.5)
    assert not smc.in_zone(z, 12, -1.0, 100.0, 100.5)


def test_a_bar_that_misses_the_zone_does_not_count():
    z = pd.DataFrame([{"known_bar": 10, "side": 1.0, "lo": 99.0,
                       "hi": 101.0, "formed_bar": 9, "kind": "fvg"}])
    assert not smc.in_zone(z, 12, 1.0, 101.5, 102.0)
    assert smc.in_zone(z, 12, 1.0, 100.9, 102.0)      # clips the top


def test_zones_can_select_either_type():
    o, h, l, c = _noise(300, seed=6)
    b = _bars(o, h, l, c)
    both = smc.zones(b)
    just_f = smc.zones(b, use_ob=False)
    assert set(just_f["kind"]) <= {"fvg"}
    assert len(both) >= len(just_f)
    assert smc.zones(b, use_fvg=False, use_ob=False).empty


def test_confirmation_can_only_REMOVE_retracements():
    """
    Location is a strictly stronger condition than depth. If
    confirming ever ADDED a setup, the two conditions would not be
    nested and the comparison would be meaningless.
    """
    from fxrisk.strategies import mtf
    rng = np.random.default_rng(7)
    n = 800
    cl = 100 + np.cumsum(rng.normal(0, 0.05, n))
    w = np.abs(rng.normal(0, 0.06, n))
    b = pd.DataFrame({"Open": cl, "High": cl + w, "Low": cl - w,
                      "Close": cl, "Volume": 1000.0},
                     index=pd.date_range("2024-01-02", periods=n, freq="30min",
                                         tz="America/New_York"))
    plain = mtf.setups_30m(b, kinds=("retrace",))
    conf = mtf.setups_30m(b, kinds=("retrace",), zone_frame=smc.zones(b))
    assert len(conf) <= len(plain)


def test_config_rejects_nonsense():
    for kw in ({"min_gap_sigma": -1}, {"ob_lookback": 2},
               {"ob_max_scan": 0}, {"zone_max_age": 0}):
        with pytest.raises(ValueError):
            smc.SMCConfig(**kw)
