"""
Download 15-minute bars for the intraday universe into a local cache.

    python fetch_intraday.py                 # the configured intraday universe
    python fetch_intraday.py --interval 5m
    python fetch_intraday.py --check         # report the cache, download nothing

THIS IS THE ONLY STEP THAT NEEDS INTERNET. It writes one CSV per
symbol into data/intraday/, and `breakout_test.py` reads those
files offline.

YAHOO'S INTRADAY LIMITS, WHICH ARE NOT NEGOTIABLE
--------------------------------------------------
    1m      7 days
    5m     60 days
    15m    60 days
    60m   730 days

A 60-day window at 15 minutes is about 1,500 bars per instrument.
After the breakout filter and the retracement wait, that leaves a
few dozen trades each. Enough to confirm the mechanics are right;
not enough to conclude anything about edge. Re-run this weekly and
the cache will only ever hold a rolling 60 days - Yahoo will not
serve the history behind it, so if you want a longer intraday
sample you have to accumulate it going forward or buy it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from fxrisk.data.intraday import (  # noqa: E402
    DEFAULT_CACHE,
    cache_path,
    coverage,
)

# Yahoo's own ceiling per interval, in days.
MAX_DAYS = {"1m": 7, "2m": 60, "5m": 60, "15m": 60, "30m": 60, "60m": 730}


def fetch(symbol: str, interval: str, days: int, cache_dir: str,
          retries: int = 3):
    """Download one symbol at one interval and write it to the cache."""
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                period=f"{days}d",
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty frame returned")

            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)

            df.index.name = "Datetime"
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(cache_path(symbol, interval, cache_dir))
            return len(df)

        except Exception as exc:                      # noqa: BLE001
            if attempt == retries:
                print(f"  {symbol:8s} FAILED after {retries} tries: {exc}")
                return 0
            time.sleep(2 * attempt)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="15m", choices=sorted(MAX_DAYS))
    ap.add_argument("--days", type=int, default=None,
                    help="default is Yahoo's maximum for the interval")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--check", action="store_true",
                    help="report what is cached and exit")
    args = ap.parse_args()

    universe = config.UNIVERSE_INTRADAY

    if args.check:
        print(f"Cache report: {args.cache_dir}, interval {args.interval}\n")
        for inst in universe:
            try:
                c = coverage(inst.yahoo, args.interval, args.cache_dir)
            except FileNotFoundError:
                print(f"  {inst.name:10s} {inst.yahoo:6s} MISSING")
                continue
            print(f"  {inst.name:10s} {inst.yahoo:6s} "
                  f"{c['bars']:6d} bars  {c['sessions']:3d} sessions  "
                  f"vol>0 {c['volume_positive']:.0%}  "
                  f"{c['start'][:16]} -> {c['end'][:16]}")
        return 0

    days = args.days or MAX_DAYS[args.interval]
    if days > MAX_DAYS[args.interval]:
        print(f"Yahoo serves at most {MAX_DAYS[args.interval]} days at "
              f"{args.interval}; asking for {days} returns an empty frame.")
        return 1

    print(f"Downloading {len(universe)} symbols, {args.interval}, "
          f"{days} days -> {args.cache_dir}\n")

    ok = 0
    for inst in universe:
        n = fetch(inst.yahoo, args.interval, days, args.cache_dir)
        if n:
            ok += 1
            print(f"  {inst.name:10s} {inst.yahoo:6s} {n:6d} bars")

    print(f"\n{ok}/{len(universe)} symbols cached.")
    if ok < len(universe):
        print("A partial cache will run, but the pooled result then "
              "describes a different universe than the one configured.")
        return 1

    print("\nNext:  python breakout_test.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
