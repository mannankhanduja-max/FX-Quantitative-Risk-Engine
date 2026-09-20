"""
Intraday bars from Yahoo, via a local CSV cache.

Same split as `yahoo.py`: `fetch_intraday.py` downloads, this
module reads, and everything after the download is offline and
reproducible.

WHY FUTURES AND NOT SPOT OR ETFs
---------------------------------
The four instruments asked for were EUR/USD, XAU/USD, NAS100 and
USD/JPY. None of them exists on Yahoo as a tradable intraday
series with volume, so each is represented by its CME future:

    EUR/USD   6E=F    euro FX future
    XAU/USD   GC=F    gold future
    NAS100    NQ=F    E-mini Nasdaq-100 future
    USD/JPY   6J=F    Japanese yen future

This buys the same two things the ETF universe bought for the
daily engine, and one more:

  REAL VOLUME. Yahoo reports Volume = 0 for every FX spot symbol
  (EURUSD=X, JPY=X). A volume-weighted average price computed on
  a zero-volume series is not a VWAP, it is a TWAP with a
  misleading name, and `fxrisk.indicators.rolling_vwap` refuses to
  produce one. CME futures report real contract volume, so the
  session VWAP this strategy is built on is an actual VWAP.

  ONE CLOCK. All four trade on Globex, 18:00-17:00 ET with the
  same daily maintenance break. A session-anchored VWAP needs a
  session boundary, and here every instrument shares one. Mixing
  a 24/5 spot series with an exchange-hours index would anchor
  the four VWAPs at four different moments.

  A REAL ORDER BOOK. Spot FX from a data vendor has no executable
  price behind it. A future does.

6J=F IS INVERTED. It is quoted as USD per JPY, so it is the
reciprocal of USD/JPY: 6J rising means the yen strengthening,
i.e. USD/JPY falling. The strategy is symmetric in direction, so
this changes the interpretation of a trade's sign, not its
outcome. The continuous front-month series (`=F`) also rolls
between contracts, which puts a gap in the price series four
times a year; those roll bars are excluded, see `load_symbol`.

THE 60-DAY WALL
----------------
Yahoo serves 15-minute bars for the trailing 60 days only. That
is roughly 1,500 bars per instrument and, after the strategy's
filters, a few dozen trades each. It is enough to check that the
mechanics work and nowhere near enough to conclude that the rule
has an edge. Every summary this module feeds says so.
"""

from __future__ import annotations

import glob
import os

import pandas as pd

DEFAULT_CACHE = "data/intraday"

# Bars whose absolute return exceeds this are treated as contract
# rolls rather than price moves. A 15-minute bar in any of these
# four instruments does not move 4% on its own; a front-month roll
# in NQ can. Excluding them stops a bookkeeping artefact from
# registering as the largest breakout in the sample.
ROLL_THRESHOLD = 0.04


def cache_path(symbol: str, interval: str = "15m",
               cache_dir: str = DEFAULT_CACHE) -> str:
    """CSV path for one symbol at one interval."""
    safe = symbol.replace("=", "_eq_").replace("^", "_c_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe}_{interval}.csv")


def available(interval: str = "15m", cache_dir: str = DEFAULT_CACHE) -> list[str]:
    """Symbols present in the cache, in their original Yahoo form."""
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, f"*_{interval}.csv"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        stem = stem[: -(len(interval) + 1)]
        out.append(stem.replace("_eq_", "=").replace("_c_", "^"))
    return out


def load_symbol(
    symbol: str,
    interval: str = "15m",
    cache_dir: str = DEFAULT_CACHE,
    drop_rolls: bool = True,
) -> pd.DataFrame:
    """
    Load one cached symbol as an intraday OHLCV frame.

    The index is timezone-aware in US/Eastern, because the session
    boundary this strategy anchors on is an exchange boundary and
    naive timestamps make it ambiguous across the two clock
    changes inside any 60-day window.
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

    if drop_rolls and len(df) > 1:
        step = df["Close"].pct_change().abs()
        df = df[~(step > ROLL_THRESHOLD).fillna(False)]

    return df


def session_id(index: pd.DatetimeIndex) -> pd.Series:
    """
    Which trading session each bar belongs to.

    Globex runs 18:00 ET to 17:00 ET the following day, so a
    session spans two calendar dates and the calendar date alone
    is the wrong grouping key: it would reset the VWAP in the
    middle of the evening session. Bars at or after 18:00 are
    assigned to the NEXT calendar day's session.
    """
    idx = pd.DatetimeIndex(index)
    day = pd.Series(idx.normalize(), index=idx)
    evening = idx.hour >= 18
    return (day + pd.to_timedelta(evening.astype(int), unit="D")).rename("session")


def coverage(symbol: str, interval: str = "15m",
             cache_dir: str = DEFAULT_CACHE) -> dict:
    """Bars, span, and how much of the volume is actually non-zero."""
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
        "bars_per_session": len(df) / max(sess.nunique(), 1),
    }
