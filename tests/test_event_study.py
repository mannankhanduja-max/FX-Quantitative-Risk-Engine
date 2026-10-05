"""
Tests for the forward-return event study.

Four of these exist only to make the measurement impossible to get
wrong in the direction that would manufacture a result: the sign
convention, the horizon offset, the refusal to use a truncated
horizon, and the refusal to match an entry time approximately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fxrisk.research import event_study as es


def bars(closes, start="2024-01-01 00:00", freq="1h", tz="UTC"):
    ix = pd.date_range(start, periods=len(closes), freq=freq, tz=tz)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes,
                         "Close": np.asarray(closes, dtype="float64"),
                         "Volume": 1.0}, index=ix)


# ---------------------------------------------------------------- sign

def test_short_in_a_falling_market_is_a_win():
    b = bars([100.0, 99.0, 98.0, 97.0, 96.0])
    e = pd.DataFrame({"time": [b.index[0]], "side": [-1]})
    out = es.signed_forward(b, e, horizons=(2,))
    assert out["y2"].iloc[0] > 0


def test_long_in_a_falling_market_is_a_loss_of_the_same_size():
    b = bars([100.0, 99.0, 98.0, 97.0, 96.0])
    t0 = b.index[0]
    short = es.signed_forward(b, pd.DataFrame({"time": [t0], "side": [-1]}),
                              horizons=(2,))["y2"].iloc[0]
    long = es.signed_forward(b, pd.DataFrame({"time": [t0], "side": [1]}),
                             horizons=(2,))["y2"].iloc[0]
    assert long == pytest.approx(-short)


def test_value_is_the_log_return_in_basis_points():
    b = bars([100.0, 100.0, 101.0])
    e = pd.DataFrame({"time": [b.index[0]], "side": [1]})
    out = es.signed_forward(b, e, horizons=(2,))
    assert out["y2"].iloc[0] == pytest.approx(np.log(101.0 / 100.0) * 1e4)


# ------------------------------------------------------------- horizon

def test_horizon_h_reads_exactly_h_bars_ahead():
    # +1 per bar, so y_h must scale with h and not with h+-1.
    b = bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    e = pd.DataFrame({"time": [b.index[0]], "side": [1]})
    out = es.signed_forward(b, e, horizons=(1, 2, 3))
    for h in (1, 2, 3):
        want = np.log((100.0 + h) / 100.0) * 1e4
        assert out[f"y{h}"].iloc[0] == pytest.approx(want)


def test_a_horizon_that_runs_off_the_end_is_nan_not_truncated():
    b = bars([100.0, 101.0, 102.0])
    e = pd.DataFrame({"time": [b.index[0]], "side": [1]})
    out = es.signed_forward(b, e, horizons=(2, 5))
    assert np.isfinite(out["y2"].iloc[0])
    assert np.isnan(out["y5"].iloc[0])


def test_bars_beyond_the_horizon_cannot_change_the_answer():
    base = [100.0, 101.0, 102.0, 103.0, 104.0]
    e_at = bars(base).index[0]
    a = es.signed_forward(bars(base), pd.DataFrame({"time": [e_at],
                                                    "side": [1]}),
                          horizons=(2,))["y2"].iloc[0]
    wild = base[:3] + [10_000.0, 0.01]           # only bars 3 and 4 changed
    b2 = es.signed_forward(bars(wild), pd.DataFrame({"time": [e_at],
                                                     "side": [1]}),
                           horizons=(2,))["y2"].iloc[0]
    assert a == pytest.approx(b2)


def test_an_entry_time_off_the_grid_is_refused():
    b = bars([100.0, 101.0, 102.0])
    off = b.index[0] + pd.Timedelta(minutes=17)
    with pytest.raises(ValueError, match="exact bar timestamp"):
        es.signed_forward(b, pd.DataFrame({"time": [off], "side": [1]}),
                          horizons=(1,))


def test_empty_entries_still_return_the_horizon_columns():
    b = bars([100.0, 101.0])
    out = es.signed_forward(b, pd.DataFrame(columns=["time", "side"]),
                            horizons=(1, 4))
    assert list(out.columns)[-2:] == ["y1", "y4"]
    assert out.empty


# ---------------------------------------------------------------- cells

def _events(n_per_month, value, instrument="EUR/USD", month="2024-01"):
    ix = pd.date_range(f"{month}-02", periods=n_per_month, freq="1D", tz="UTC")
    return pd.DataFrame({"instrument": instrument, "time": ix, "y": value})


def test_thin_cells_are_dropped():
    d = pd.concat([_events(5, 1.0, month="2024-01"),
                   _events(2, 99.0, month="2024-02")], ignore_index=True)
    c = es.cell_means(d, "y", min_events=3)
    assert list(c["month"]) == ["2024-01"]


def test_cells_split_by_instrument_and_month():
    d = pd.concat([_events(4, 1.0, "EUR/USD", "2024-01"),
                   _events(4, 2.0, "XAU/USD", "2024-01"),
                   _events(4, 3.0, "EUR/USD", "2024-02")], ignore_index=True)
    c = es.cell_means(d, "y", min_events=3)
    assert len(c) == 3
    assert sorted(c["mean"]) == [1.0, 2.0, 3.0]


def test_months_are_cut_in_utc_not_in_local_time():
    # 2024-01-31 23:30 UTC is 2024-02-01 in Tokyo. It must stay in January.
    ix = pd.DatetimeIndex(["2024-01-31 23:30", "2024-01-31 23:40",
                           "2024-01-31 23:50"]).tz_localize("UTC")
    d = pd.DataFrame({"instrument": "USD/JPY", "time": ix.tz_convert("Asia/Tokyo"),
                      "y": [1.0, 1.0, 1.0]})
    c = es.cell_means(d, "y", min_events=3)
    assert list(c["month"]) == ["2024-01"]


def test_missing_columns_are_an_error_not_an_empty_frame():
    with pytest.raises(KeyError):
        es.cell_means(pd.DataFrame({"time": [], "y": []}), "y")


# --------------------------------------------------------------- pooled

def test_pooled_t_matches_a_one_sample_t_test_on_the_cell_means():
    rng = np.random.default_rng(7)
    vals = rng.normal(0.5, 1.0, 40)
    cells = pd.DataFrame({"instrument": "X", "month": range(40),
                          "n": 5, "mean": vals})
    got = es.pooled(cells)
    want = stats.ttest_1samp(vals, 0.0)
    assert got["t"] == pytest.approx(want.statistic)
    assert got["p"] == pytest.approx(want.pvalue)
    assert got["cells"] == 40
    assert got["events"] == 200


def test_pooled_equal_weights_cells_regardless_of_event_count():
    # One busy cell at +10 must not outvote three quiet cells at 0.
    cells = pd.DataFrame({"instrument": "X", "month": list("abcd"),
                          "n": [1000, 3, 3, 3], "mean": [10.0, 0.0, 0.0, 0.0]})
    assert es.pooled(cells)["mean_bp"] == pytest.approx(2.5)


def test_pooled_refuses_to_report_on_fewer_than_three_cells():
    cells = pd.DataFrame({"instrument": "X", "month": ["a", "b"],
                          "n": [5, 5], "mean": [1.0, 2.0]})
    assert np.isnan(es.pooled(cells)["t"])


def test_share_pos_is_measured_on_the_side_of_the_pooled_mean():
    cells = pd.DataFrame({"instrument": "X", "month": list("abcd"),
                          "n": 5, "mean": [-1.0, -1.0, -1.0, 9.0]})
    out = es.pooled(cells)
    assert out["mean_bp"] == pytest.approx(1.5)
    assert out["share_pos"] == pytest.approx(0.25)
