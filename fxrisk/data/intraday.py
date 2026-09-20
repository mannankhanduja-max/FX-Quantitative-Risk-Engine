"""
Intraday bars from Dukascopy, via a local CSV cache.

Same split as `yahoo.py`: `fetch_intraday.py` downloads, this
module reads, and everything after the download is offline and
reproducible.

WHY DUKASCOPY AND NOT YAHOO
----------------------------
Three reasons, in order of how much they matter.

1. VOLUME EXISTS. Yahoo reports `Volume = 0` for every FX spot
   symbol, which makes a volume-weighted average price undefined
   — `indicators.rolling_vwap` raises rather than quietly hand
   back a TWAP under a false name. Dukascopy reports TICK VOLUME:
   the number of price updates inside each bar. See below on what
   that is and is not.

2. ALL FOUR INSTRUMENTS EXIST, as the things actually asked for
   rather than as proxies:

       EUR/USD   EUR/USD      spot
       XAU/USD   XAU/USD      spot gold
       NAS100    E_NQ-100     Nasdaq-100 index
       USD/JPY   USD/JPY      spot

   No currency ETFs, no futures standing in for spot, and no
   reciprocal quoting to reason around.

3. HISTORY GOES BACK YEARS, not 60 days. Yahoo serves 15-minute
   bars for a trailing 60-day window and will not serve the
   history behind it, which caps any intraday study at a few
   dozen trades. Dukascopy's archive starts in 2003 for the FX
   majors. That is the difference between a result that can be
   tested out of sample and one that cannot.

WHAT TICK VOLUME IS
--------------------
It is the count of price updates in the bar, not contracts or
notional traded. In spot FX there is no consolidated tape, so
traded volume does not exist at all — any source claiming to give
you spot FX volume is giving you one venue's slice or a tick
count with a better name.

A tick count is a real measure of ACTIVITY, and weighting by it
gives a VWAP that means something precise: the average price
weighted by how busy the market was. That is not the same object
as a share-volume VWAP and should not be described as though it
were. It is also the convention `histdata.py` already uses in
this repository, so the two intraday paths agree.

The count is Dukascopy's own feed, not the whole market. It
tracks broad activity well and will not match another broker's
tick count bar for bar.

THE SESSION BOUNDARY
---------------------
17:00 New York, which is the FX market convention and the same
`SESSION_CLOSE` the daily engine cuts on. A session-anchored
VWAP needs a boundary, and using the calendar date instead would
reset it at midnight — in the middle of the Asian session, the
worst available choice.

The Nasdaq-100 index feed does not keep FX hours, so applying the
FX boundary to it is a deliberate simplification: the VWAP anchor
lands at 17:00 ET, an hour after the US cash close, which is a
defensible place to start a new day for an index and is not the
only defensible one.

BID SIDE ONLY. Bars are fetched on the bid. Using bid OHLC and
then charging a spread as a cost double-counts nothing, but it
does mean the recorded highs are bid highs; an ask-side buy fills
above them. `BREAKOUT_COST_BP` exists to cover that and is set
deliberately wide.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

DEFAULT_CACHE = "data/intraday"

# Bars whose absolute return exceeds this are treated as feed
# artefacts rather than price moves. A 15-minute bar in any of
# these four does not move 4% on its own; a gap left by a feed
# outage across a weekend can look like one.
JUMP_THRESHOLD = 0.04


def cache_path(symbol: str, interval: str = "15m",
               cache_dir: str = DEFAULT_CACHE) -> str:
    """CSV path for one symbol at one interval."""
    safe = (
        symbol.replace("/", "_")
        .replace("=", "_eq_")
        .replace("^", "_c_")
        .replace(".", "_")
    )
    return os.path.join(cache_dir, f"{safe}_{interval}.csv")


def available(interval: str = "15m", cache_dir: str = DEFAULT_CACHE) -> list[str]:
    """Cache stems present for an interval."""
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, f"*_{interval}.csv"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        out.append(stem[: -(len(interval) + 1)])
    return out


def load_symbol(
    symbol: str,
    interval: str = "15m",
    cache_dir: str = DEFAULT_CACHE,
    drop_jumps: bool = True,
) -> pd.DataFrame:
    """
    Load one cached symbol as an intraday OHLCV frame.

    The index is timezone-aware in US/Eastern, because the session
    boundary is an FX-market boundary and naive timestamps make it
    ambiguous across the clock changes inside any long window.
    """
    path = cache_path(symbol, interval, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached intraday data for {symbol} at {path}.\n"
            f"Run:  python fetch_intraday.py\n"
            f"(that step needs internet; everything after it does not)"
        )

    df = pd.read_csv(path, parse_dates=["Datetime"], index_col="Datetime")
    df = df[~df.index.duplicated(keep="last")].sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df.index.name = "Datetime"

    # A bar with no ticks has no price information. Dukascopy
    # emits these across market closures; carrying them forward
    # would put flat bars inside the breakout lookback and make a
    # dead weekend look like a consolidation.
    if "Volume" in df.columns:
        df = df[pd.to_numeric(df["Volume"], errors="coerce").fillna(0) > 0]

    if drop_jumps and len(df) > 1:
        step = df["Close"].pct_change().abs()
        df = df[~(step > JUMP_THRESHOLD).fillna(False)]

    return df


def session_id(index: pd.DatetimeIndex) -> pd.Series:
    """
    Which trading session each bar belongs to.

    The FX day runs 17:00 ET to 17:00 ET, so a session spans two
    calendar dates and the calendar date alone is the wrong
    grouping key — it would reset the VWAP at midnight, in the
    middle of the Asian session. Bars at or after 17:00 are
    assigned to the NEXT calendar day's session.

    This matches `config.SESSION_CLOSE`, so the intraday path and
    the daily engine cut the day at the same instant.
    """
    idx = pd.DatetimeIndex(index)
    day = pd.Series(idx.normalize(), index=idx)
    evening = idx.hour >= 17
    return (day + pd.to_timedelta(evening.astype(int), unit="D")).rename("session")


def coverage(symbol: str, interval: str = "15m",
             cache_dir: str = DEFAULT_CACHE) -> dict:
    """Bars, span, and how much of the tick volume is non-zero."""
    df = load_symbol(symbol, interval, cache_dir)
    vol = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0.0)
    sess = session_id(df.index)
    return {
        "symbol": symbol,
        "bars": len(df),
        "sessions": int(sess.nunique()),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "volume_positive": float((vol > 0).mean()),
        "median_ticks": float(vol.median()),
        "bars_per_session": len(df) / max(sess.nunique(), 1),
    }
