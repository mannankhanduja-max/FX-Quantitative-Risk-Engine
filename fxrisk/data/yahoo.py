"""
Yahoo Finance daily bars, via a local CSV cache.

Deliberately split in two:

    fetch_data.py   downloads and writes data/yahoo/<SYMBOL>.csv
    this module     reads those CSVs and needs no network at all

So the download happens once, and every run after it is
reproducible, offline, and identical. It also means a rate limit or
an outage cannot silently change a backtest you already ran.

HOW YAHOO DIFFERS FROM HISTDATA, AND WHY IT MATTERS
----------------------------------------------------
This is not a drop-in replacement for `histdata.py`. Three
differences change what the engine can honestly claim:

1. NO INTRADAY, SO NO REAL VWAP. Yahoo serves daily bars. The
   tick-count-weighted VWAP in histdata.py has no equivalent here -
   there are no ticks to count. Yahoo also reports volume 0 for FX
   pairs, exactly as HistData does, so even a volume-weighted daily
   VWAP is unavailable. Anything VWAP-shaped built on this source
   would be a TWAP.

2. NO CHOSEN SESSION BOUNDARY. histdata.py cuts the day at 17:00
   New York, the FX market convention. Yahoo's daily FX bar is
   already aggregated by Yahoo on its own boundary, which is not
   documented and is not the 17:00 NY convention. The bars are
   consistent with each other, so correlations across Yahoo FX
   symbols are fine; they are NOT directly comparable with a
   HistData-derived series.

3. ADJUSTED VS RAW. `auto_adjust` is applied at fetch time for
   equities and ETFs so returns include distributions. FX pairs
   have no distributions, so the setting is inert for them.

None of this makes Yahoo worse - it makes it different, and the
difference is worth stating rather than discovering.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

DEFAULT_CACHE = "data/yahoo"


def cache_path(symbol: str, cache_dir: str = DEFAULT_CACHE) -> str:
    """CSV path for one symbol. '=' and '^' are not filename-safe."""
    safe = symbol.replace("=", "_eq_").replace("^", "_c_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe}.csv")


def available(cache_dir: str = DEFAULT_CACHE) -> list[str]:
    """Symbols present in the cache, in their original Yahoo form."""
    out = []
    for p in sorted(glob.glob(os.path.join(cache_dir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        out.append(stem.replace("_eq_", "=").replace("_c_", "^"))
    return out


def load_symbol(symbol: str, cache_dir: str = DEFAULT_CACHE) -> pd.DataFrame:
    """Load one cached symbol as a daily OHLCV frame."""
    path = cache_path(symbol, cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached data for {symbol} at {path}.\n"
            f"Run:  python fetch_data.py\n"
            f"(that step needs internet; everything after it does not)"
        )

    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_panel(
    symbols: dict[str, str] | list[str],
    cache_dir: str = DEFAULT_CACHE,
    field: str = "Close",
) -> pd.DataFrame:
    """
    Build an aligned price panel from the cache.

    Parameters
    ----------
    symbols
        Either {display name: yahoo symbol} or a list of symbols.
    field
        Which column to take. "Close" for adjusted closes.

    Missing symbols raise rather than being quietly dropped - a
    panel that is silently short an instrument is worse than one
    that fails.
    """
    if isinstance(symbols, dict):
        pairs = list(symbols.items())
    else:
        pairs = [(s, s) for s in symbols]

    cols = {}
    for name, sym in pairs:
        cols[name] = load_symbol(sym, cache_dir)[field]

    panel = pd.DataFrame(cols)
    return panel.sort_index()


def daily_returns(
    symbols: dict[str, str] | list[str],
    cache_dir: str = DEFAULT_CACHE,
    kind: str = "log",
    min_price: float = 1e-12,
) -> pd.DataFrame:
    """
    Aligned daily return panel from the cache.

    Rows where any instrument is missing are dropped, so the usable
    sample is bounded by the shortest series. `coverage_report`
    shows what that costs before you rely on it.
    """
    prices = load_panel(symbols, cache_dir)
    prices = prices.where(prices > min_price)

    if kind == "log":
        rets = np.log(prices / prices.shift(1))
    elif kind == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'")

    return rets.replace([np.inf, -np.inf], np.nan).dropna()


def coverage_report(
    symbols: dict[str, str] | list[str], cache_dir: str = DEFAULT_CACHE
) -> pd.DataFrame:
    """
    Per-instrument first date, last date and row count, plus what the
    aligned panel costs.

    Print this before trusting a run. If one instrument starts years
    after the others, it is silently deciding the sample.
    """
    if isinstance(symbols, dict):
        pairs = list(symbols.items())
    else:
        pairs = [(s, s) for s in symbols]

    rows = []
    for name, sym in pairs:
        try:
            df = load_symbol(sym, cache_dir)
            rows.append(
                {
                    "instrument": name,
                    "symbol": sym,
                    "first": df.index.min().date(),
                    "last": df.index.max().date(),
                    "rows": len(df),
                }
            )
        except FileNotFoundError:
            rows.append(
                {
                    "instrument": name,
                    "symbol": sym,
                    "first": None,
                    "last": None,
                    "rows": 0,
                }
            )

    out = pd.DataFrame(rows).set_index("instrument")

    try:
        aligned = daily_returns(symbols, cache_dir)
        out.attrs["aligned_rows"] = len(aligned)
        out.attrs["aligned_first"] = aligned.index.min().date()
        out.attrs["aligned_last"] = aligned.index.max().date()
        out.attrs["binding"] = out["first"].astype(str).idxmax()
    except Exception:
        out.attrs["aligned_rows"] = 0

    return out
