"""Tests for the optional news-calendar / stress-window subset loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from fxrisk.research import fvg_subsets as sub


def test_missing_files_return_none(tmp_path):
    assert sub.load_news_calendar(str(tmp_path / "nope.csv")) is None
    assert sub.load_stress_windows(str(tmp_path / "nope.csv")) is None


def test_load_news_calendar(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text(
        "timestamp,event,impact\n"
        "2022-01-26T19:00:00Z,FOMC,high\n"
        "2022-02-02T13:30:00Z,NFP,HIGH\n"
        "2022-02-10T13:30:00Z,CPI,medium\n"
    )
    cal = sub.load_news_calendar(str(path))
    assert len(cal) == 3
    assert list(cal["impact"]) == ["high", "high", "medium"]
    assert str(cal["timestamp"].dt.tz) == "UTC"


def test_load_news_calendar_rejects_bad_impact(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text("timestamp,event,impact\n2022-01-01T00:00:00Z,X,extreme\n")
    with pytest.raises(ValueError, match="unknown impact"):
        sub.load_news_calendar(str(path))


def test_load_stress_windows(tmp_path):
    path = tmp_path / "stress.csv"
    path.write_text(
        "name,start,end\n"
        "crash,2020-03-06T00:00:00Z,2020-03-20T00:00:00Z\n"
    )
    w = sub.load_stress_windows(str(path))
    assert len(w) == 1
    assert w.iloc[0]["name"] == "crash"


def test_load_stress_windows_rejects_end_before_start(tmp_path):
    path = tmp_path / "stress.csv"
    path.write_text("name,start,end\nbad,2020-03-20T00:00:00Z,2020-03-06T00:00:00Z\n")
    with pytest.raises(ValueError, match="end must be after start"):
        sub.load_stress_windows(str(path))


def test_in_news_window_flags_only_within_range_and_min_impact():
    cal = pd.DataFrame({
        "timestamp": pd.to_datetime(
            ["2022-01-26T19:00:00Z", "2022-02-10T13:30:00Z"], utc=True
        ),
        "event": ["FOMC", "CPI"],
        "impact": ["high", "medium"],
    })
    entries = pd.DatetimeIndex(pd.to_datetime([
        "2022-01-26T19:10:00Z",   # 10m after FOMC (high) -> flagged
        "2022-01-26T20:00:00Z",   # 60m after FOMC -> outside default 30m window
        "2022-02-10T13:35:00Z",   # near CPI (medium) -> not flagged at min_impact=high
    ], utc=True))
    flags = sub.in_news_window(entries, cal, before_minutes=30, after_minutes=30,
                               min_impact="high")
    assert list(flags) == [True, False, False]


def test_in_news_window_with_no_qualifying_events_is_all_false():
    cal = pd.DataFrame({
        "timestamp": pd.to_datetime(["2022-01-01T00:00:00Z"], utc=True),
        "event": ["x"], "impact": ["low"],
    })
    entries = pd.DatetimeIndex(pd.to_datetime(["2022-01-01T00:05:00Z"], utc=True))
    flags = sub.in_news_window(entries, cal, min_impact="high")
    assert list(flags) == [False]


def test_stress_window_labels_handles_overlap_and_misses():
    windows = pd.DataFrame({
        "name": ["a", "b"],
        "start": pd.to_datetime(["2020-01-01", "2020-01-02"], utc=True),
        "end": pd.to_datetime(["2020-01-05", "2020-01-04"], utc=True),
    })
    entries = pd.DatetimeIndex(pd.to_datetime(
        ["2020-01-03", "2020-01-10"], utc=True
    ))
    labels = sub.stress_window_labels(entries, windows)
    assert labels[0] == ["a", "b"]
    assert labels[1] == []
