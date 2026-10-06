"""
Optional subset breakdowns for the FVG study: news windows and named
stress windows. See `fvg_config/README.md` for the file formats.

Both loaders return `None`, not an empty frame, when the file is
missing - callers branch on that explicitly (`if calendar is not
None:`) so a missing config visibly skips a section of `RESULTS.md`
rather than silently producing a table of zero rows that looks like a
real "no events found" result.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_NEWS_PATH = "fvg_config/news_calendar.csv"
DEFAULT_STRESS_PATH = "fvg_config/stress_windows.csv"
VALID_IMPACTS = ("low", "medium", "high")


def load_news_calendar(path: str = DEFAULT_NEWS_PATH) -> pd.DataFrame | None:
    """Columns: timestamp (UTC), event, impact. None if the file is absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
    bad = set(df["impact"].str.lower()) - set(VALID_IMPACTS)
    if bad:
        raise ValueError(f"{path}: unknown impact value(s) {sorted(bad)}, "
                         f"expected one of {VALID_IMPACTS}")
    df["impact"] = df["impact"].str.lower()
    return df.sort_values("timestamp").reset_index(drop=True)


def load_stress_windows(path: str = DEFAULT_STRESS_PATH) -> pd.DataFrame | None:
    """Columns: name, start, end (UTC). None if the file is absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["start", "end"])
    for col in ("start", "end"):
        if df[col].dt.tz is None:
            df[col] = df[col].dt.tz_localize("UTC")
        else:
            df[col] = df[col].dt.tz_convert("UTC")
    bad = df[df["end"] <= df["start"]]
    if not bad.empty:
        raise ValueError(f"{path}: end must be after start - see row(s) "
                         f"{bad['name'].tolist()}")
    return df.sort_values("start").reset_index(drop=True)


def in_news_window(entry_times: pd.DatetimeIndex, calendar: pd.DataFrame,
                   before_minutes: float = 30.0, after_minutes: float = 30.0,
                   min_impact: str = "high") -> np.ndarray:
    """
    True for every entry that falls within `[event - before, event +
    after]` of a qualifying event.

    `min_impact` keeps only events at or above that level
    (`low < medium < high`), since a 30-minute window around every
    `low`-impact release on the calendar would flag most of the trading
    day on a busy week.
    """
    order = {v: i for i, v in enumerate(VALID_IMPACTS)}
    keep = calendar[calendar["impact"].map(order) >= order[min_impact]]
    if keep.empty:
        return np.zeros(len(entry_times), dtype=bool)

    entries = pd.DatetimeIndex(entry_times)
    before = pd.Timedelta(minutes=before_minutes)
    after = pd.Timedelta(minutes=after_minutes)
    flagged = np.zeros(len(entries), dtype=bool)
    for ts in keep["timestamp"]:
        flagged |= (entries >= ts - before) & (entries <= ts + after)
    return flagged


def stress_window_labels(entry_times: pd.DatetimeIndex,
                         windows: pd.DataFrame) -> list[list[str]]:
    """Per entry, the names of every stress window it falls inside
    (usually zero or one; windows may overlap, so possibly more)."""
    entries = pd.DatetimeIndex(entry_times)
    out: list[list[str]] = [[] for _ in range(len(entries))]
    for row in windows.itertuples(index=False):
        hit = (entries >= row.start) & (entries < row.end)
        for i in np.flatnonzero(hit):
            out[i].append(row.name)
    return out
