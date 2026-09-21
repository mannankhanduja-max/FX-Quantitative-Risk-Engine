"""
Tests for fxrisk.research.signals and fxrisk.research.ic.

The harness exists to say "there is nothing here" with confidence,
so the tests concentrate on the three ways it could instead say
"there is something here" falsely: a feature that peeks at the
future, a forward return misaligned by a bar, and an inference that
treats overlapping returns as independent. A planted-signal test
checks the other direction - that a real effect is found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk.research import ic as icm
from fxrisk.research import signals as sg


def _hourly(n=3000, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02 00:00", periods=n, freq="1h",
                        tz="America/New_York")
    r = rng.normal(0, 0.001, n) if drift is None else drift
    close = 100 * np.exp(np.cumsum(r))
    spread = np.abs(rng.normal(0, 0.0005, n))
    return pd.DataFrame({
        "Open": np.r_[close[0], close[:-1]],
        "High": close * (1 + spread),
        "Low": close * (1 - spread),
        "Close": close,
        "Volume": rng.integers(100, 2000, n).astype(float),
    }, index=idx)


@pytest.mark.parametrize("feature", sg.FEATURES)
def test_every_feature_is_causal(feature):
    """
    Change the LAST bar violently. No feature at any EARLIER bar may
    move. A feature that fails this is reading the future, and every
    IC it produces is fiction.
    """
    bars = _hourly()
    base = sg.compute(bars)

    shocked = bars.copy()
    k = shocked.index[-1]
    shocked.loc[k, ["Close", "High"]] *= 1.10
    shocked.loc[k, "Volume"] *= 50
    after = sg.compute(shocked)

    a = base[feature].iloc[:-1]
    b = after[feature].iloc[:-1]
    both = a.notna() & b.notna()
    assert np.allclose(a[both], b[both]), f"{feature} depends on the future"
    assert (a.isna() == b.isna()).all()


def test_forward_return_is_close_t_to_close_t_plus_h():
    bars = _hourly(200)
    fwd = sg.forward_returns(bars, horizons=(1, 4))
    lc = np.log(bars["Close"])
    assert fwd["fwd_1"].iloc[10] == pytest.approx(lc.iloc[11] - lc.iloc[10])
    assert fwd["fwd_4"].iloc[10] == pytest.approx(lc.iloc[14] - lc.iloc[10])
    assert np.isnan(fwd["fwd_4"].iloc[-1])


def test_forward_return_does_not_include_bar_t():
    """
    The classic off-by-one: if fwd_1 at t contained bar t's own return,
    mom_1h would correlate with it perfectly by construction.
    """
    bars = _hourly(3000, seed=3)
    f = sg.compute(bars)["mom_1h"]
    r = sg.forward_returns(bars, horizons=(1,))["fwd_1"]
    df = pd.concat([f, r], axis=1).dropna()
    assert abs(df.corr(method="spearman").iloc[0, 1]) < 0.1


def test_open_drive_is_nan_before_the_open():
    bars = _hourly(96)
    d = sg.compute(bars)["ny_drive"]
    early = d[(d.index.hour >= 17) | (d.index.hour < 9)]
    assert early.isna().all()
    at_open = d[d.index.hour == 9].dropna()
    assert len(at_open) > 0


def test_a_planted_signal_is_found():
    """
    Build returns that genuinely depend on the previous bar, then check
    the harness sees it. A harness that can never say yes is as useless
    as one that always does.
    """
    rng = np.random.default_rng(7)
    n = 8000
    eps = rng.normal(0, 0.001, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.25 * r[i - 1] + eps[i]           # strong momentum
    bars = _hourly(n, drift=r)
    f = sg.compute(bars)["mom_1h"]
    y = sg.forward_returns(bars, horizons=(1,))["fwd_1"]
    res = icm.summarise(icm.cell_ics(f, y, "X"), "mom_1h", 1)
    assert res.mean_ic > 0.1
    assert res.p < 0.001


def test_noise_rarely_passes():
    """
    On pure noise, the per-test false-positive rate should sit near
    the nominal 5%. Checked across seeds, not one lucky draw.
    """
    hits = 0
    trials = 40
    for s in range(trials):
        bars = _hourly(4000, seed=100 + s)
        f = sg.compute(bars)["mom_1h"]
        y = sg.forward_returns(bars, horizons=(1,))["fwd_1"]
        res = icm.summarise(icm.cell_ics(f, y, "X"), "mom_1h", 1)
        hits += int(np.isfinite(res.p) and res.p < 0.05)
    assert hits <= 6          # 5% of 40 is 2; allow generous slack


def test_blocking_does_not_count_overlapping_returns_as_independent():
    """
    With a 24-bar horizon, the cell count - which is what the t-test
    uses as n - must be the number of months, not the number of bars.
    """
    bars = _hourly(24 * 150)
    f = sg.compute(bars)["mom_24h"]
    y = sg.forward_returns(bars, horizons=(24,))["fwd_24"]
    cells = icm.cell_ics(f, y, "X")
    assert len(cells) <= 6
    assert icm.summarise(cells, "mom_24h", 24).cells == len(cells)


def test_holm_step_down():
    p = [0.001, 0.04, 0.012, 0.9]
    # m = 4: 0.001 <= .0125 reject; 0.012 <= .0167 reject;
    # 0.04 > .025 stop; 0.9 kept.
    assert icm.holm(p, 0.05) == [True, False, True, False]


def test_holm_stops_at_the_first_failure():
    """A later p that would pass a looser bar is still kept."""
    p = [0.02, 0.021, 0.022]
    # 0.02 > 0.05/3 = 0.0167 -> stop immediately, nothing rejected.
    assert icm.holm(p, 0.05) == [False, False, False]


def test_holm_ignores_nans():
    assert icm.holm([0.001, float("nan")], 0.05) == [True, False]
