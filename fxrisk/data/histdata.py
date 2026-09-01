"""
HistData.com loader: tick and M1 ASCII files.

File formats, per HistData's published specification:

  M1 ASCII   semicolon delimited, no header
             DateTime;Open;High;Low;Close;Volume
             DateTime is "YYYYMMDD HHMMSS", seconds always 00.
             Quotes are BID only.

  Tick ASCII comma delimited, no header
             DateTime,Bid,Ask,Volume
             DateTime is "YYYYMMDD HHMMSSNNN" with milliseconds.

Timestamps are Eastern Time (EST/EDT) for the standard downloads.

THE VOLUME PROBLEM, STATED PLAINLY
-----------------------------------
The Volume column in both formats is zero. Always. Spot FX trades
over the counter with no consolidated tape, so there is no volume
to report - HistData carries the column for format compatibility
and fills it with zeros.

This matters because a volume-weighted average price needs volume.
Two workable substitutes, both implemented here:

  TICK COUNT      The number of quote updates inside a bar. Quote
                  intensity correlates strongly with traded activity
                  in FX, so a tick-count-weighted average price is a
                  genuine VWAP analogue. This is the standard
                  approach and requires tick files.

  TIME WEIGHTED   With only M1 bars, every bar carries equal weight,
                  which makes "VWAP" a simple moving average of
                  typical prices - a TWAP. Honest, but do not call
                  it a VWAP.

`load_m1` therefore refuses to fabricate volume: it returns a
tick_count column of NaN and `vwap` will fall back to TWAP with a
warning unless you supply tick data.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

M1_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
TICK_COLUMNS = ["datetime", "bid", "ask", "volume"]

DEFAULT_SOURCE_TZ = "America/New_York"


@dataclass
class LoadReport:
    """What actually came off disk. Print this before trusting a run."""

    files: int
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicates_dropped: int
    volume_all_zero: bool

    def summary(self) -> str:
        return "\n".join(
            [
                f"  files                 {self.files}",
                f"  rows                  {self.rows:,}",
                f"  first                 {self.first}",
                f"  last                  {self.last}",
                f"  duplicate rows        {self.duplicates_dropped:,}",
                f"  volume column empty   {self.volume_all_zero}",
            ]
        )


def _resolve_paths(path: str) -> list[str]:
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
    elif any(ch in path for ch in "*?["):
        found = sorted(glob.glob(path, recursive=True))
    else:
        found = [path]

    found = [p for p in found if os.path.isfile(p)]
    if not found:
        raise FileNotFoundError(
            f"no CSV files matched '{path}'. Download the ASCII archives from "
            "histdata.com, unzip them, and point this at the folder."
        )
    return found


def load_m1(
    path: str,
    source_tz: str = DEFAULT_SOURCE_TZ,
    target_tz: str = "UTC",
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, LoadReport]:
    """
    Load HistData M1 ASCII bars.

    Parameters
    ----------
    path
        A file, a directory, or a glob. Directories are searched
        recursively, which is what you want after unzipping a year
        of monthly archives.
    source_tz, target_tz
        HistData timestamps are Eastern; everything downstream is
        easier in UTC.

    Returns
    -------
    DataFrame indexed by timestamp with open/high/low/close and a
    tick_count column of NaN (see the module docstring).
    """
    paths = _resolve_paths(path)

    frames = []
    for p in paths:
        frame = pd.read_csv(
            p,
            sep=";",
            header=None,
            names=M1_COLUMNS,
            dtype={"datetime": str},
        )
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"], format="%Y%m%d %H%M%S")

    before = len(raw)
    raw = raw.drop_duplicates(subset="datetime").sort_values("datetime")
    dropped = before - len(raw)

    volume_all_zero = bool((raw["volume"].fillna(0) == 0).all())

    idx = (
        pd.DatetimeIndex(raw["datetime"])
        .tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        .tz_convert(target_tz)
    )

    out = pd.DataFrame(
        {
            "open": raw["open"].to_numpy(),
            "high": raw["high"].to_numpy(),
            "low": raw["low"].to_numpy(),
            "close": raw["close"].to_numpy(),
            "tick_count": np.nan,
        },
        index=idx,
    )
    out = out[out.index.notna()]

    report = LoadReport(
        files=len(paths),
        rows=len(out),
        first=out.index.min() if len(out) else None,
        last=out.index.max() if len(out) else None,
        duplicates_dropped=dropped,
        volume_all_zero=volume_all_zero,
    )

    if return_report:
        return out, report
    return out


def load_ticks(
    path: str,
    source_tz: str = DEFAULT_SOURCE_TZ,
    target_tz: str = "UTC",
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, LoadReport]:
    """
    Load HistData tick ASCII files.

    Returns bid, ask, mid and spread. Tick files are large - roughly
    a million rows per pair per month - so load one pair at a time
    and aggregate to bars before doing anything else.
    """
    paths = _resolve_paths(path)

    frames = []
    for p in paths:
        frame = pd.read_csv(
            p,
            sep=",",
            header=None,
            names=TICK_COLUMNS,
            dtype={"datetime": str},
        )
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    raw["datetime"] = pd.to_datetime(raw["datetime"], format="%Y%m%d %H%M%S%f")

    before = len(raw)
    raw = raw.sort_values("datetime")
    dropped = before - len(raw)

    volume_all_zero = bool((raw["volume"].fillna(0) == 0).all())

    idx = (
        pd.DatetimeIndex(raw["datetime"])
        .tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        .tz_convert(target_tz)
    )

    bid = raw["bid"].to_numpy()
    ask = raw["ask"].to_numpy()

    out = pd.DataFrame(
        {"bid": bid, "ask": ask, "mid": (bid + ask) / 2.0, "spread": ask - bid},
        index=idx,
    )
    out = out[out.index.notna()]

    report = LoadReport(
        files=len(paths),
        rows=len(out),
        first=out.index.min() if len(out) else None,
        last=out.index.max() if len(out) else None,
        duplicates_dropped=dropped,
        volume_all_zero=volume_all_zero,
    )

    if return_report:
        return out, report
    return out


def ticks_to_bars(ticks: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """
    Aggregate ticks into OHLC bars with a real tick_count.

    The tick_count column is the volume proxy that makes a genuine
    VWAP possible. It also carries information of its own: quote
    intensity spikes around data releases and in stressed markets,
    so it is a usable liquidity signal.
    """
    if "mid" not in ticks.columns:
        raise ValueError("expected a mid column - load with load_ticks")

    grouped = ticks["mid"].resample(freq)
    bars = pd.DataFrame(
        {
            "open": grouped.first(),
            "high": grouped.max(),
            "low": grouped.min(),
            "close": grouped.last(),
            "tick_count": grouped.count(),
        }
    )

    if "spread" in ticks.columns:
        bars["mean_spread"] = ticks["spread"].resample(freq).mean()

    return bars.dropna(subset=["close"])


def rolling_vwap(bars: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling VWAP over `window` bars, weighted by tick count.

    Typical price is (high + low + close) / 3, matching the
    convention in the upstream QuantPortfolio code.

    If tick_count is absent or entirely NaN - which is the case for
    M1 files - this degrades to an equal-weighted average and warns.
    That result is a TWAP; the function still returns it because a
    TWAP is a legitimate benchmark, but it must not be reported as a
    VWAP.
    """
    for col in ("high", "low", "close"):
        if col not in bars.columns:
            raise ValueError(f"bars is missing the '{col}' column")

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0

    weights = bars.get("tick_count")
    if weights is None or weights.isna().all() or (weights.fillna(0) <= 0).all():
        warnings.warn(
            "no usable tick_count - falling back to an equal-weighted "
            "average. This is a TWAP, not a VWAP. Use tick files via "
            "load_ticks() + ticks_to_bars() for a true VWAP.",
            RuntimeWarning,
            stacklevel=2,
        )
        return typical.rolling(window, min_periods=window).mean().rename("twap")

    w = weights.fillna(0.0)
    num = (typical * w).rolling(window, min_periods=window).sum()
    den = w.rolling(window, min_periods=window).sum().replace(0, np.nan)
    return (num / den).rename("vwap")


def to_daily(bars: pd.DataFrame, session_close: str = "17:00", tz: str = "America/New_York") -> pd.DataFrame:
    """
    Collapse intraday bars into daily bars on an explicit session
    boundary.

    FX has no exchange close, so the daily boundary is a choice. The
    market convention is 17:00 New York. Making it explicit - rather
    than letting pandas cut at UTC midnight, which lands in the
    middle of the Asian session - is the difference between a daily
    return series that means something and one that does not.
    """
    local = bars.tz_convert(tz)
    offset = pd.Timedelta(hours=int(session_close.split(":")[0]),
                          minutes=int(session_close.split(":")[1]))
    session = (local.index - offset).normalize()

    grouped = local.groupby(session)
    daily = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
        }
    )
    if "tick_count" in local.columns:
        daily["tick_count"] = grouped["tick_count"].sum()

    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily.dropna(subset=["close"])


def daily_returns(daily: pd.DataFrame, kind: str = "log") -> pd.Series:
    """Daily returns from a daily bar frame."""
    close = daily["close"]
    if kind == "log":
        r = np.log(close / close.shift(1))
    elif kind == "simple":
        r = close.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    return r.replace([np.inf, -np.inf], np.nan).dropna().rename("return")
