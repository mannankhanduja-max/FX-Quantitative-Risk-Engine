"""
Reader for the yearly 1-minute XAU/USD bid/ask cache `fetch_fvg_raw.py`
writes into `data/raw/`.

Same split as `fxrisk.data.intraday`: a fetch script downloads, this
module reads, and everything after loading is offline and
reproducible. Kept separate from `intraday.py` rather than folded into
it because this study uses its own directory, its own naming (one file
per calendar year, so a partial download is visibly partial), and its
own fixed design/test date split - conflating the two would make
either module's cache-key logic harder to read for no shared benefit.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

from fetch_fvg_raw import SLUG, raw_path

DEFAULT_RAW_DIR = "data/raw"

# The split this whole study exists to respect. DESIGN is free to
# iterate on; TEST is read only by the one-shot runner, and only once
# PREREGISTRATION.md is signed. Both ends are exclusive of the far
# side: a bar at exactly midnight on DESIGN_END belongs to the test
# period, never both.
DESIGN_START = pd.Timestamp("2015-01-01", tz="UTC")
DESIGN_END = pd.Timestamp("2022-01-01", tz="UTC")
TEST_START = DESIGN_END


def available_years(side: str, raw_dir: str = DEFAULT_RAW_DIR) -> list[int]:
    """Calendar years actually on disk for one side, sorted."""
    pattern = os.path.join(raw_dir, f"{SLUG}_1m_{side}_*.csv")
    years = []
    for path in glob.glob(pattern):
        stem = os.path.splitext(os.path.basename(path))[0]
        tail = stem.rsplit("_", 1)[-1]
        if tail.isdigit():
            years.append(int(tail))
    return sorted(years)


def _missing_data_error(side: str, years: list[int], raw_dir: str) -> FileNotFoundError:
    want = f"{years[0]}-{years[-1]}" if years else "the requested range"
    return FileNotFoundError(
        f"No cached {side} bars for {want} at {raw_dir}/{SLUG}_1m_{side}_*.csv.\n"
        f"Run:  python fetch_fvg_raw.py --start {years[0] if years else 2015} "
        f"--end {years[-1] if years else 2026} --side {side}\n"
        f"(that step needs internet; everything after it does not. "
        f"python fetch_fvg_raw.py --check reports what is already cached.)"
    )


def _as_utc(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")


def load_side(side: str, start: pd.Timestamp, end: pd.Timestamp,
             raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """
    Concatenate every yearly file for one side that overlaps
    `[start, end)`, clipped to that window.

    Raises if ANY year in the requested span has no file on disk - a
    silently shortened window is exactly the kind of thing that should
    be loud, not inferred from how much data happened to load.
    """
    start, end = _as_utc(start), _as_utc(end)
    wanted_years = list(range(start.year, end.year + 1))
    have = set(available_years(side, raw_dir))
    missing = [y for y in wanted_years if y not in have]
    if missing:
        raise _missing_data_error(side, missing, raw_dir)

    frames = []
    for year in wanted_years:
        path = raw_path(side, year, raw_dir)
        df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
        frames.append(df)
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()

    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert("UTC")
    return out.loc[(out.index >= start) & (out.index < end)]


def load_bars(start: pd.Timestamp, end: pd.Timestamp,
             raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """The BID side, as plain OHLCV - what the rule itself trades
    against, same convention as `fxrisk.data.intraday.load_symbol`."""
    return load_side("bid", start, end, raw_dir)


def load_design(raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """2015-01-01 -> 2022-01-01, bid side. Free to load and re-load
    while the rule is being built; nothing about calling this is
    gated."""
    return load_bars(DESIGN_START, DESIGN_END, raw_dir)


def load_test(raw_dir: str = DEFAULT_RAW_DIR) -> pd.DataFrame:
    """2022-01-01 -> now, bid side. Loading the bars is not gated -
    `run_fvg_test.py`'s one-shot marker and the pre-registration check
    are what actually stop the test from being run more than once; see
    that file."""
    return load_bars(TEST_START, pd.Timestamp.now(tz="UTC"), raw_dir)
