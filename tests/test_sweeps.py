"""
Tests for fxrisk.research.barrier_prob and fxrisk.strategies.sweeps.

The benchmark is the more dangerous of the two: if it were wrong in
the strategy's favour it would certify a coin flip as an edge. So
its known closed forms are checked against hand-computable cases and
against simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.research import barrier_prob as bp
from fxrisk.strategies import sweeps as sw


# ------------------------------------------------------------
# The no-information benchmark
# ------------------------------------------------------------

def test_symmetric_barriers_are_a_coin_flip():
    p = bp.touch_probability(100.0, 100.0 * np.exp(0.01), 100.0 * np.exp(-0.01))
    assert float(p) == pytest.approx(0.5)


def test_two_to_one_is_one_third_in_log_space():
    """
    The result the whole study kept running into. A target twice as
    far as the stop is hit first one time in three - before costs,
    before anything.
    """
    b = 0.01
    p = bp.touch_probability(100.0, 100.0 * np.exp(2 * b), 100.0 * np.exp(-b))
    assert float(p) == pytest.approx(1 / 3)


def test_shorts_are_mirrored():
    """Target below entry, stop above: same geometry, same answer."""
    lng = bp.touch_probability(100.0, 100.0 * np.exp(0.02), 100.0 * np.exp(-0.01))
    srt = bp.touch_probability(100.0, 100.0 * np.exp(-0.02), 100.0 * np.exp(0.01))
    assert float(lng) == pytest.approx(float(srt))


def test_matches_simulation():
    """
    The formula against a random walk. If these disagree, the formula
    is wrong and every 'edge' measured against it is an artefact.
    """
    rng = np.random.default_rng(4)
    a, b, s = 0.02, 0.01, 0.002
    hits = 0
    trials = 4000
    for _ in range(trials):
        x = 0.0
        for _ in range(20000):
            x += rng.normal(0, s)
            if x >= a:
                hits += 1
                break
            if x <= -b:
                break
    p = float(bp.touch_probability(1.0, np.exp(a), np.exp(-b)))
    se = np.sqrt(p * (1 - p) / trials)
    assert abs(hits / trials - p) < 4 * se


def test_positive_drift_raises_the_probability():
    up = float(bp.touch_probability(1.0, np.exp(0.02), np.exp(-0.01),
                                    drift=0.5, sigma=0.05))
    flat = float(bp.touch_probability(1.0, np.exp(0.02), np.exp(-0.01)))
    assert up > flat


def test_implied_drift_round_trips():
    a, b, s = 0.02, 0.01, 0.05
    target = float(bp.touch_probability(1.0, np.exp(a), np.exp(-b),
                                        drift=0.4, sigma=s))
    assert bp.implied_drift(target, a, b, s) == pytest.approx(0.4, abs=1e-3)


def test_benchmark_excludes_time_exits():
    t = pd.DataFrame({
        "outcome": ["target", "stop", "time"],
        "entry": [100.0, 100.0, 100.0],
        "target": [102.0, 102.0, 102.0],
        "stop0": [99.0, 99.0, 99.0],
    })
    r = bp.benchmark(t)
    assert r["trades"] == 2
    assert r["time_exits_excluded"] == 1


def test_benchmark_finds_no_edge_in_a_fair_sample():
    """A sample that wins exactly at the model rate must give z ~ 0."""
    n = 900
    out = ["target"] * (n // 3) + ["stop"] * (n - n // 3)
    t = pd.DataFrame({
        "outcome": out,
        "entry": 100.0,
        "target": 100.0 * np.exp(0.02),
        "stop0": 100.0 * np.exp(-0.01),
    })
    r = bp.benchmark(t)
    assert r["model_win"] == pytest.approx(1 / 3, abs=1e-9)
    assert abs(r["z"]) < 0.5


# ------------------------------------------------------------
# Sweeps
# ------------------------------------------------------------

def _bars(h, l, c):
    idx = pd.date_range("2024-01-02", periods=len(c), freq="1h",
                        tz="America/New_York")
    return pd.DataFrame({"High": h, "Low": l, "Close": c,
                         "Open": c, "Volume": 1000.0}, index=idx)


def _quiet(n=25, base=100.0):
    h = [base + 0.05] * n
    l = [base - 0.05] * n
    c = [base + (0.01 if i % 2 else -0.01) for i in range(n)]
    return h, l, c


def test_a_bullish_sweep_pierces_the_low_and_closes_back_above():
    h, l, c = _quiet()
    h += [100.06]; l += [99.50]; c += [100.04]        # long lower wick
    b = _bars(h, l, c)
    d = sw.detect(b, sw.SweepConfig(lookback=20, min_pierce=0.1))
    assert d["sweep"].iloc[-1] == 1.0


def test_a_close_beyond_the_level_is_a_breakout_not_a_sweep():
    """The partition that makes the comparison meaningful."""
    h, l, c = _quiet()
    h += [100.00]; l += [99.50]; c += [99.60]         # closed BELOW the level
    b = _bars(h, l, c)
    d = sw.detect(b, sw.SweepConfig(lookback=20, min_pierce=0.1))
    assert d["sweep"].iloc[-1] == 0.0


def test_a_graze_is_not_a_sweep():
    h, l, c = _quiet()
    h += [100.06]; l += [99.9499]; c += [100.04]      # barely through
    b = _bars(h, l, c)
    d = sw.detect(b, sw.SweepConfig(lookback=20, min_pierce=5.0))
    assert d["sweep"].iloc[-1] == 0.0


def test_a_weak_close_is_not_a_rejection():
    """Pierced the low but closed near it: no reclaim, no sweep."""
    h, l, c = _quiet()
    h += [100.00]; l += [99.50]; c += [99.56]
    b = _bars(h, l, c)
    d = sw.detect(b, sw.SweepConfig(lookback=20, min_pierce=0.1,
                                    close_frac=0.5))
    assert d["sweep"].iloc[-1] == 0.0


def test_bearish_sweep_is_mirrored():
    h, l, c = _quiet()
    h += [100.60]; l += [99.94]; c += [99.96]
    b = _bars(h, l, c)
    d = sw.detect(b, sw.SweepConfig(lookback=20, min_pierce=0.1))
    assert d["sweep"].iloc[-1] == -1.0


def test_detection_is_causal():
    h, l, c = _quiet(40)
    b = _bars(h, l, c)
    base = sw.detect(b)["sweep"]
    shocked = b.copy()
    k = shocked.index[-1]
    shocked.loc[k, "Low"] = 90.0
    after = sw.detect(shocked)["sweep"]
    assert (base.iloc[:-1] == after.iloc[:-1]).all()


def test_entry_side_opposes_the_pierce():
    h, l, c = _quiet()
    h += [100.06]; l += [99.50]; c += [100.04]
    # A trailing bar: an entry needs somewhere to go, so reversal_entries
    # will not open a trade on the final bar of the sample.
    h += [100.05]; l += [99.95]; c += [100.0]
    b = _bars(h, l, c)
    e = sw.reversal_entries(b, sw.SweepConfig(lookback=20, min_pierce=0.1))
    assert len(e) == 1
    assert e.iloc[0]["side"] == 1.0
    assert e.iloc[0]["stop"] < e.iloc[0]["entry"]


def test_recent_sweep_gate_expires():
    h, l, c = _quiet(30)
    h += [100.06]; l += [99.50]; c += [100.04]
    h += [100.05] * 20; l += [99.95] * 20; c += [100.0] * 20
    b = _bars(h, l, c)
    g = sw.recent_sweep(b, within=5, cfg=sw.SweepConfig(lookback=20,
                                                        min_pierce=0.1))
    fired = np.flatnonzero(g.to_numpy() != 0.0)
    assert len(fired) == 6            # the bar itself plus five carried
    assert g.iloc[-1] == 0.0


def test_config_rejects_nonsense():
    for kw in ({"lookback": 1}, {"min_pierce": -1}, {"close_frac": 0.0},
               {"close_frac": 1.5}):
        with pytest.raises(ValueError):
            sw.SweepConfig(**kw)
