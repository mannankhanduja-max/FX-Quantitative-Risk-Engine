"""
Tests for `fxrisk.calendar` and `fxrisk.risk.barriers`.

Both modules were written as one-off scripts, used to produce
published numbers, and only then moved into the package. Nothing
had tested them. These pin the properties whose silent failure
would have made those numbers wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxrisk import calendar as cal
from fxrisk.risk import barriers as br


def _bdays(start: str, periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=periods)


# ------------------------------------------------------------
# Calendar
# ------------------------------------------------------------

def test_nfp_is_the_first_friday_present():
    """
    January 2021 began on a Friday, so the 1st is the first Friday
    and the 15th is the third.
    """
    idx = _bdays("2021-01-01", 25)
    fl = cal.flags(idx)
    assert fl.loc["2021-01-01", "nfp"]
    assert not fl.loc["2021-01-08", "nfp"]
    assert fl.loc["2021-01-15", "opex"]


def test_ranks_count_trading_days_not_calendar_days():
    """
    If the true first Friday is missing from the index - a holiday -
    the NEXT Friday present becomes the first ranked one. That is a
    deliberate choice, and it must be stable rather than silently
    shifting the flag onto a date that is not a Friday at all.
    """
    idx = _bdays("2021-01-01", 25).drop(pd.Timestamp("2021-01-01"))
    fl = cal.flags(idx)
    flagged = fl.index[fl["nfp"]]
    assert len(flagged) == 1
    assert flagged[0].dayofweek == 4


def test_month_and_quarter_end():
    idx = _bdays("2021-03-01", 45)
    fl = cal.flags(idx)
    march_end = fl.loc["2021-03-31"]
    assert march_end["month_end"] and march_end["quarter_end"]
    april_end = fl.loc["2021-04-30"]
    assert april_end["month_end"] and not april_end["quarter_end"]


def test_turn_is_the_union_of_first_and_last():
    idx = _bdays("2021-01-01", 60)
    fl = cal.flags(idx)
    assert (fl["turn"] >= fl["month_end"]).all()
    assert fl["turn"].sum() > fl["month_end"].sum()


def test_flags_cover_the_whole_index_with_no_nan():
    idx = _bdays("2020-01-01", 300)
    fl = cal.flags(idx)
    assert list(fl.columns) == list(cal.FLAG_NAMES)
    assert fl.notna().all().all()
    assert fl.dtypes.eq(bool).all()


def test_event_csv_round_trip(tmp_path):
    idx = _bdays("2021-01-01", 40)
    csv = tmp_path / "events.csv"
    csv.write_text("date,event\n2021-01-05,fomc\n2021-01-12,cpi\n2030-01-01,fomc\n")
    out = cal.load_event_csv(csv, idx)
    assert out["fomc"].sum() == 1          # the 2030 row falls outside
    assert out.loc["2021-01-05", "fomc"]
    assert out.loc["2021-01-12", "cpi"]


def test_event_csv_rejects_a_missing_column(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("date,name\n2021-01-05,fomc\n")
    with pytest.raises(ValueError):
        cal.load_event_csv(csv, _bdays("2021-01-01", 10))


# ------------------------------------------------------------
# Barriers
# ------------------------------------------------------------

def _bars(closes, highs=None, lows=None) -> pd.DataFrame:
    n = len(closes)
    idx = _bdays("2021-01-04", n)
    return pd.DataFrame(
        {
            "Close": closes,
            "High": highs if highs is not None else closes,
            "Low": lows if lows is not None else closes,
        },
        index=idx,
    )


def test_target_hit_gives_positive_r():
    bars = _bars([100.0, 100.0, 120.0], highs=[100.0, 100.0, 120.0],
                 lows=[100.0, 100.0, 100.0])
    sig = pd.Series([1.0, 0.0, 0.0], index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    cfg = br.BarrierConfig(stop_k=1.0, rr=1.0, max_days=5, cost_bp=0.0)
    t = br.walk(bars, sig, sigma, cfg)
    assert len(t) == 1
    assert t.iloc[0]["outcome"] == "target"
    assert t.iloc[0]["unit_net_R"] == pytest.approx(1.0)


def test_stop_hit_gives_negative_one_r():
    bars = _bars([100.0, 100.0, 80.0], highs=[100.0, 100.0, 100.0],
                 lows=[100.0, 100.0, 80.0])
    sig = pd.Series([1.0, 0.0, 0.0], index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    cfg = br.BarrierConfig(stop_k=1.0, rr=1.0, max_days=5, cost_bp=0.0)
    t = br.walk(bars, sig, sigma, cfg)
    assert t.iloc[0]["outcome"] == "stop"
    assert t.iloc[0]["unit_net_R"] == pytest.approx(-1.0)


def test_both_barriers_in_one_bar_resolves_against_the_trade():
    """
    The intrabar ambiguity. A bar that spans both barriers must be
    recorded as a stop and flagged, never as a target.
    """
    bars = _bars([100.0, 100.0, 100.0], highs=[100.0, 100.0, 125.0],
                 lows=[100.0, 100.0, 75.0])
    sig = pd.Series([1.0, 0.0, 0.0], index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    cfg = br.BarrierConfig(stop_k=1.0, rr=1.0, max_days=5, cost_bp=0.0)
    t = br.walk(bars, sig, sigma, cfg)
    assert t.iloc[0]["outcome"] == "stop"
    assert bool(t.iloc[0]["ambiguous"])


def test_time_exit_when_neither_barrier_is_touched():
    bars = _bars([100.0] * 8)
    sig = pd.Series([1.0] + [0.0] * 7, index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    cfg = br.BarrierConfig(stop_k=1.0, rr=1.0, max_days=3, cost_bp=0.0)
    t = br.walk(bars, sig, sigma, cfg)
    assert t.iloc[0]["outcome"] == "time"
    assert t.iloc[0]["days_held"] == 3


def test_short_side_is_mirrored():
    """A short into a falling market is a winner, not a loser."""
    bars = _bars([100.0, 100.0, 80.0], highs=[100.0, 100.0, 100.0],
                 lows=[100.0, 100.0, 80.0])
    sig = pd.Series([-1.0, 0.0, 0.0], index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    cfg = br.BarrierConfig(stop_k=1.0, rr=1.0, max_days=5, cost_bp=0.0)
    t = br.walk(bars, sig, sigma, cfg)
    assert t.iloc[0]["outcome"] == "target"
    assert t.iloc[0]["unit_net_R"] == pytest.approx(1.0)


def test_size_scales_r_but_never_changes_the_outcome():
    """
    The property that makes a volatility-targeting variant's win
    rate identical to baseline by construction. If this ever fails,
    every sizing comparison in the repository is wrong.
    """
    bars = _bars([100.0, 100.0, 120.0], highs=[100.0, 100.0, 120.0],
                 lows=[100.0, 100.0, 100.0])
    sig = pd.Series([1.0, 0.0, 0.0], index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    cfg = br.BarrierConfig(stop_k=1.0, rr=1.0, max_days=5, cost_bp=0.0)

    plain = br.walk(bars, sig, sigma, cfg)
    scaled = br.walk(bars, sig, sigma, cfg, size=pd.Series(3.0, index=bars.index))

    assert plain.iloc[0]["outcome"] == scaled.iloc[0]["outcome"]
    assert plain.iloc[0]["unit_net_R"] == pytest.approx(scaled.iloc[0]["unit_net_R"])
    assert scaled.iloc[0]["net_R"] == pytest.approx(3 * plain.iloc[0]["net_R"])


def test_allow_gate_suppresses_entries():
    bars = _bars([100.0] * 8)
    sig = pd.Series([1.0] + [0.0] * 7, index=bars.index)
    sigma = pd.Series(0.10, index=bars.index)
    blocked = br.walk(bars, sig, sigma, allow=pd.Series(False, index=bars.index))
    assert blocked.empty


def test_breakeven_rises_as_the_stop_tightens():
    """
    The counterintuitive one. A tighter stop means a larger cost in
    units of risk, so the hurdle goes UP.
    """
    naive = br.breakeven_win_rate(0.0, rr=1.0)
    tight = br.breakeven_win_rate(0.073, rr=1.0)
    wide = br.breakeven_win_rate(0.025, rr=1.0)

    assert naive == pytest.approx(0.50)
    assert tight > wide > naive
    assert tight == pytest.approx(0.5365, abs=1e-3)


def test_breakeven_falls_as_payoff_improves():
    assert br.breakeven_win_rate(0.0, rr=2.0) == pytest.approx(1 / 3)
    assert br.breakeven_win_rate(0.0, rr=3.0) == pytest.approx(0.25)


def test_config_rejects_nonsense():
    for kwargs in ({"stop_k": 0}, {"rr": -1}, {"max_days": 0}, {"cost_bp": -1}):
        with pytest.raises(ValueError):
            br.BarrierConfig(**kwargs)


def test_walk_requires_ohlc():
    bars = pd.DataFrame({"Close": [1.0, 2.0]}, index=_bdays("2021-01-04", 2))
    with pytest.raises(ValueError):
        br.walk(bars, pd.Series([1.0, 0.0], index=bars.index),
                pd.Series(0.1, index=bars.index))


def test_summarise_on_no_trades():
    assert br.summarise(pd.DataFrame(), rr=1.0) == {"trades": 0}


def test_ewma_sigma_is_causal():
    """A shock must not raise the volatility estimate on its own day."""
    r = pd.Series([0.001] * 50 + [0.5], index=_bdays("2021-01-04", 51))
    sig = br.ewma_sigma(r)
    assert sig.iloc[-1] < 0.05          # the shock has not landed yet
    assert np.isnan(sig.iloc[0])
