"""
Download intraday bars from Dukascopy into a local cache.

    pip install dukascopy-python

    python fetch_intraday.py                    # 2 years of 15m bars
    python fetch_intraday.py --years 5
    python fetch_intraday.py --interval 5m
    python fetch_intraday.py --check            # report cache, download nothing

THIS IS THE ONLY STEP THAT NEEDS INTERNET. It writes one CSV per
instrument into data/intraday/, and `breakout_test.py` reads them
offline.

WHY THIS SOURCE
----------------
`fxrisk/data/intraday.py` has the full argument. The short form:
Dukascopy reports tick volume (so a VWAP is definable at all),
carries all four instruments as the things actually asked for
rather than as proxies, and serves years of history instead of
Yahoo's trailing 60 days.

WHAT YOU ARE GETTING, PRECISELY
--------------------------------
Bid-side OHLC with a tick count per bar, from one broker's feed.
Not a consolidated tape — there is no such thing in spot FX. It
tracks the broad market closely and will not agree bar for bar
with another broker.

BE POLITE TO THE ENDPOINT. This is a free public service with no
authentication and no published rate limit, which means the only
thing stopping abuse is the person running the script. The
default is one instrument at a time with a pause between chunks.
Do not remove it, and do not re-download history you already have
cached — the archive does not change.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from fxrisk.data.intraday import DEFAULT_CACHE, cache_path, coverage  # noqa: E402

# Dukascopy's own interval tokens, keyed by the name used here.
INTERVALS = {
    "1m": "INTERVAL_MIN_1",
    "5m": "INTERVAL_MIN_5",
    "10m": "INTERVAL_MIN_10",
    "15m": "INTERVAL_MIN_15",
    "30m": "INTERVAL_MIN_30",
    "1h": "INTERVAL_HOUR_1",
}

# Fetched in slices rather than one request. A five-year 15m pull
# is ~125k bars, and asking for it in a single call is both more
# likely to fail and less polite than asking six times.
CHUNK_DAYS = 300
PAUSE_SECONDS = 1.5


def fetch_one(dk, instrument: str, interval_attr: str,
              start: datetime, end: datetime, debug: bool = False,
              side: str = "bid"):
    """
    Pull one instrument in chunks and concatenate.

    `side` selects bid or ask. Fetching BOTH and subtracting is
    the only way to get the real spread out of this feed - every
    range- or volume-based estimate of it is a proxy, and
    fxrisk/risk/spread.py documents two that fail at intraday
    frequency. Ask minus bid is measured.

        python fetch_intraday.py --interval 5m --years 2 --side ask

    writes a parallel `_ask` cache alongside the bid one.
    """
    import pandas as pd

    interval = getattr(dk, interval_attr)
    frames, cursor = [], start

    while cursor < end:
        stop = min(cursor + timedelta(days=CHUNK_DAYS), end)
        try:
            offer = dk.OFFER_SIDE_ASK if side == "ask" else dk.OFFER_SIDE_BID
            df = dk.fetch(instrument, interval, offer,
                          cursor, stop, debug=debug)
            if df is not None and len(df):
                frames.append(df)
        except Exception as exc:                          # noqa: BLE001
            print(f"      chunk {cursor:%Y-%m-%d} -> {stop:%Y-%m-%d} "
                  f"failed: {type(exc).__name__}: {exc}")
        cursor = stop
        if cursor < end:
            time.sleep(PAUSE_SECONDS)

    if not frames:
        return None

    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def normalise(df):
    """Dukascopy's lowercase columns -> the repo's OHLCV names."""
    import pandas as pd

    out = df.rename(
        columns={"open": "Open", "high": "High", "low": "Low",
                 "close": "Close", "volume": "Volume"}
    )
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume")
            if c in out.columns]
    out = out[keep].copy()

    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert("UTC")
    out.index.name = "Datetime"
    return out


def sanity(name: str, df) -> list[str]:
    """
    Complaints about a downloaded frame, or an empty list.

    Worth doing before anything is cached. A silently mangled
    column order or a stale frame is much cheaper to catch here
    than after it has become a backtest result.
    """
    problems = []
    if df is None or df.empty:
        return [f"{name}: empty"]

    bad_hl = int((df["High"] < df["Low"]).sum())
    if bad_hl:
        problems.append(f"{name}: High < Low on {bad_hl} bars "
                        f"(column order is wrong)")

    envelope = int(
        ((df["High"] < df[["Open", "Close"]].max(axis=1))
         | (df["Low"] > df[["Open", "Close"]].min(axis=1))).sum()
    )
    if envelope:
        problems.append(f"{name}: {envelope} bars where High/Low do not "
                        f"contain Open/Close")

    if (df["Close"] <= 0).any():
        problems.append(f"{name}: non-positive prices")

    if "Volume" in df and (df["Volume"] < 0).any():
        problems.append(f"{name}: negative tick volume")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="15m", choices=sorted(INTERVALS))
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--side", default="bid", choices=("bid", "ask"),
                    help="ask writes a parallel _ask cache; "
                         "ask minus bid is the real spread")
    args = ap.parse_args()

    universe = config.UNIVERSE_INTRADAY

    if args.check:
        print(f"Cache report: {args.cache_dir}, interval {args.interval}\n")
        for inst in universe:
            try:
                c = coverage(inst.yahoo, args.interval, args.cache_dir)
            except FileNotFoundError:
                print(f"  {inst.name:9s} MISSING")
                continue
            print(f"  {inst.name:9s} {c['bars']:7d} bars  "
                  f"{c['sessions']:5d} sessions  "
                  f"median {c['median_ticks']:6.0f} ticks/bar  "
                  f"{c['start'][:10]} -> {c['end'][:10]}")
        return 0

    try:
        import dukascopy_python as dk
    except ImportError:
        print("dukascopy-python is not installed.\n\n    "
              "pip install dukascopy-python\n")
        return 1

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=int(args.years * 365))

    print(f"Dukascopy, {args.side} side, {args.interval}, "
          f"{start:%Y-%m-%d} -> {end:%Y-%m-%d}")
    print(f"  {len(universe)} instruments, {CHUNK_DAYS}-day chunks, "
          f"{PAUSE_SECONDS:g}s between\n")

    ok, complaints = 0, []
    for inst in universe:
        print(f"  {inst.name:9s} {inst.dukascopy:12s} ...", flush=True)
        raw = fetch_one(dk, inst.dukascopy, INTERVALS[args.interval],
                        start, end, args.debug, side=args.side)
        if raw is None:
            print("      nothing returned")
            continue

        df = normalise(raw)
        found = sanity(inst.name, df)
        if found:
            complaints.extend(found)
            print("      REJECTED: " + "; ".join(found))
            continue

        os.makedirs(args.cache_dir, exist_ok=True)
        tag = args.interval + ("_ask" if args.side == "ask" else "")
        df.to_csv(cache_path(inst.yahoo, tag, args.cache_dir))
        ok += 1
        print(f"      {len(df):7d} bars  {df.index[0]:%Y-%m-%d} -> "
              f"{df.index[-1]:%Y-%m-%d}  "
              f"median {df['Volume'].median():.0f} ticks/bar")

    print(f"\n{ok}/{len(universe)} instruments cached.")
    if complaints:
        print("\nNothing was written for the rejected instruments. These")
        print("checks exist because a mangled frame is far cheaper to catch")
        print("here than after it has become a backtest result.")
        return 1
    if ok < len(universe):
        print("A partial cache will run, but the pooled result then")
        print("describes a different universe than the one configured.")
        return 1

    print("\nNext:  python breakout_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
