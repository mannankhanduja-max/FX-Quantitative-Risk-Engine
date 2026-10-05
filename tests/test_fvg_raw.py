"""
Tests for the FVG raw-data fetch helpers and the yearly-cache reader.

No network: `fetch_fvg_raw.fetch_year` is never called here. What's
tested is everything that does not need the network - filenames,
chunk-year arithmetic, the reusable normalise/sanity checks, and the
reader's concatenation, clipping and missing-year error.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import fetch_fvg_raw as fetch
from fxrisk.data import fvg_raw


def test_raw_path_naming():
    assert fetch.raw_path("bid", 2015) == "data/raw/XAUUSD_1m_bid_2015.csv"
    assert fetch.raw_path("ask", 2022, raw_dir="x") == "x/XAUUSD_1m_ask_2022.csv"


def test_year_bounds_span_the_calendar_year():
    start, end = fetch._year_bounds(2019)
    assert start == datetime(2019, 1, 1)
    assert end == datetime(2020, 1, 1)


def test_year_bounds_clip_the_current_year_to_now():
    this_year = datetime.now().year
    start, end = fetch._year_bounds(this_year)
    assert start == datetime(this_year, 1, 1)
    assert end <= datetime.now()


def test_normalise_renames_and_localises_to_utc():
    df = pd.DataFrame(
        {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [10]},
        index=pd.date_range("2024-01-01", periods=1, freq="1min"),
    )
    out = fetch.normalise(df)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out.index.name == "Datetime"
    assert str(out.index.tz) == "UTC"


def test_sanity_flags_high_below_low():
    bad = pd.DataFrame({"Open": [1.0], "High": [0.9], "Low": [1.1],
                        "Close": [1.0], "Volume": [10]})
    assert any("High < Low" in p for p in fetch.sanity("x", bad))


def test_sanity_passes_a_clean_frame():
    ok = pd.DataFrame({"Open": [1.0], "High": [1.2], "Low": [0.9],
                       "Close": [1.0], "Volume": [10]})
    assert fetch.sanity("x", ok) == []


def test_sanity_on_empty_or_none():
    assert fetch.sanity("x", None) == ["x: empty"]
    assert fetch.sanity("x", pd.DataFrame()) == ["x: empty"]


# ------------------------------------------------------------
# The reader
# ------------------------------------------------------------

def _write_year(raw_dir, side, year, start, n=5):
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {"Open": range(n), "High": range(n), "Low": range(n),
         "Close": range(n), "Volume": [10] * n},
        index=idx,
    )
    df.index.name = "Datetime"
    path = fetch.raw_path(side, year, raw_dir)
    df.to_csv(path)
    return df


def test_available_years_reads_the_directory(tmp_path):
    raw_dir = str(tmp_path)
    _write_year(raw_dir, "bid", 2019, "2019-06-01")
    _write_year(raw_dir, "bid", 2021, "2021-06-01")
    assert fvg_raw.available_years("bid", raw_dir) == [2019, 2021]
    assert fvg_raw.available_years("ask", raw_dir) == []


def test_load_side_concatenates_and_clips(tmp_path):
    raw_dir = str(tmp_path)
    _write_year(raw_dir, "bid", 2019, "2019-12-31 22:00")  # spills into 2020
    _write_year(raw_dir, "bid", 2020, "2020-01-01 03:00")
    out = fvg_raw.load_side(
        "bid",
        pd.Timestamp("2019-12-31 23:00", tz="UTC"),
        pd.Timestamp("2020-01-01 04:00", tz="UTC"),
        raw_dir,
    )
    assert out.index.min() >= pd.Timestamp("2019-12-31 23:00", tz="UTC")
    assert out.index.max() < pd.Timestamp("2020-01-01 04:00", tz="UTC")
    assert out.index.is_monotonic_increasing
    assert not out.index.duplicated().any()


def test_load_side_raises_clearly_on_a_missing_year(tmp_path):
    raw_dir = str(tmp_path)
    _write_year(raw_dir, "bid", 2019, "2019-06-01")
    with pytest.raises(FileNotFoundError, match="fetch_fvg_raw.py"):
        fvg_raw.load_side(
            "bid",
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2021-01-01", tz="UTC"),
            raw_dir,
        )


def test_load_bars_is_the_bid_side(tmp_path):
    raw_dir = str(tmp_path)
    written = _write_year(raw_dir, "bid", 2019, "2019-06-01 00:00", n=3)
    out = fvg_raw.load_bars(
        pd.Timestamp("2019-06-01", tz="UTC"),
        pd.Timestamp("2019-06-02", tz="UTC"),
        raw_dir,
    )
    assert len(out) == len(written)


def test_design_and_test_periods_meet_without_overlapping():
    assert fvg_raw.TEST_START == fvg_raw.DESIGN_END
    assert fvg_raw.DESIGN_START < fvg_raw.DESIGN_END
