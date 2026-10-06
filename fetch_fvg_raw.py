"""
Download 1-minute XAU/USD bars from Dukascopy for the FVG design/test
study, one calendar year at a time, bid and ask both.

    pip install dukascopy-python

    python fetch_fvg_raw.py                       # 2015-01-01 -> today
    python fetch_fvg_raw.py --start 2015 --end 2022
    python fetch_fvg_raw.py --check                # report what's cached, download nothing
    python fetch_fvg_raw.py --year 2019             # just one year, both sides

THIS IS THE ONLY STEP THAT NEEDS INTERNET, and it is not something an
automated session should run: a single year of 1-minute bars is on the
order of half a million rows per side, eleven years is eleven times
that, and the download alone can run for a long time against a free,
unauthenticated, rate-limit-free public endpoint that asks for nothing
in return except not being hammered. Run this yourself, from a machine
with ordinary internet access. `run_fvg_design.py` and
`run_fvg_test.py` read the files this script writes and need no
network at all.

WHY ONE FILE PER CALENDAR YEAR, NOT ONE FILE FOR THE WHOLE WINDOW
--------------------------------------------------------------------
Resumability. If the download is interrupted at year eight of eleven,
re-running this script skips every year whose file already exists and
picks up where it left off - nothing already on disk is re-fetched,
and nothing has to be re-verified by hand. `--force` overrides that for
a specific `--year` if a file needs replacing.

WHY BOTH SIDES
---------------
`fxrisk/risk/spread.py` can only measure the real spread, rather than
assume one, from bid AND ask quoted at the same frequency. Ask minus
bid is the cost this strategy actually pays; every proxy for it
(a fixed basis-point guess, a volatility-scaled guess) is documented
in that module as failing at intraday frequency.

WHAT YOU ARE GETTING, PRECISELY
---------------------------------
One broker's (Dukascopy's) own bid and ask feed for spot gold, not a
consolidated tape - there is no such thing in OTC gold or spot FX. Tick
count substitutes for volume, as everywhere else this repository
touches Dukascopy. Timestamps are stored in UTC, which is GMT for
every practical purpose here (no DST in UTC).

BE POLITE TO THE ENDPOINT. Free, public, unauthenticated, no published
rate limit. The pacing below is deliberately unhurried; do not tighten
it to go faster.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

SYMBOL = "XAU/USD"
SLUG = "XAUUSD"
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_START_YEAR = 2015

# Within one calendar year, 1-minute bars are still ~525k rows - too
# many for one request to the same free endpoint `fetch_intraday.py`
# already treats carefully at coarser intervals. Sub-chunked by days,
# same spirit as that script's CHUNK_DAYS, just smaller because a
# 1-minute bar carries ~15x the rows a 15-minute bar does over the
# same span.
SUBCHUNK_DAYS = 20
PAUSE_SECONDS = 2.0


def _year_bounds(year: int) -> tuple[datetime, datetime]:
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return start, min(end, now)


def raw_path(side: str, year: int, raw_dir: str = DEFAULT_RAW_DIR) -> str:
    return os.path.join(raw_dir, f"{SLUG}_1m_{side}_{year}.csv")


def fetch_year(dk, side: str, year: int, debug: bool = False):
    """One year, one side, sub-chunked and concatenated. None if the
    endpoint returned nothing for the whole year (e.g. before the feed
    existed for this instrument)."""
    import pandas as pd
    from datetime import timedelta

    start, end = _year_bounds(year)
    if start >= end:
        return None

    offer = dk.OFFER_SIDE_ASK if side == "ask" else dk.OFFER_SIDE_BID
    frames, cursor = [], start
    while cursor < end:
        stop = min(cursor + timedelta(days=SUBCHUNK_DAYS), end)
        try:
            df = dk.fetch(SYMBOL, dk.INTERVAL_MIN_1, offer, cursor, stop, debug=debug)
            if df is not None and len(df):
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
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
    """Dukascopy's lowercase columns -> this repo's OHLCV names, UTC index."""
    import pandas as pd

    out = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                             "close": "Close", "volume": "Volume"})
    keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in out.columns]
    out = out[keep].copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    out.index = idx.tz_convert("UTC")
    out.index.name = "Datetime"
    return out


def sanity(name: str, df) -> list[str]:
    """Same checks fetch_intraday.py runs before writing anything to disk."""
    problems = []
    if df is None or df.empty:
        return [f"{name}: empty"]
    bad_hl = int((df["High"] < df["Low"]).sum())
    if bad_hl:
        problems.append(f"{name}: High < Low on {bad_hl} bars (column order is wrong)")
    envelope = int(
        ((df["High"] < df[["Open", "Close"]].max(axis=1))
         | (df["Low"] > df[["Open", "Close"]].min(axis=1))).sum()
    )
    if envelope:
        problems.append(f"{name}: {envelope} bars where High/Low do not contain Open/Close")
    if (df["Close"] <= 0).any():
        problems.append(f"{name}: non-positive prices")
    if "Volume" in df and (df["Volume"] < 0).any():
        problems.append(f"{name}: negative tick volume")
    return problems


def check(raw_dir: str, start_year: int, end_year: int) -> None:
    print(f"Cache report: {raw_dir}\n")
    for year in range(start_year, end_year + 1):
        row = f"  {year}  "
        for side in ("bid", "ask"):
            path = raw_path(side, year, raw_dir)
            if os.path.exists(path):
                import pandas as pd
                n = sum(1 for _ in open(path)) - 1
                size_mb = os.path.getsize(path) / 1e6
                row += f"{side}: {n:>7,} rows ({size_mb:5.1f}MB)   "
            else:
                row += f"{side}: MISSING              "
        print(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=DEFAULT_START_YEAR,
                    help="first calendar year to fetch (default 2015)")
    ap.add_argument("--end", type=int,
                    default=datetime.now(timezone.utc).year,
                    help="last calendar year to fetch (default: this year)")
    ap.add_argument("--year", type=int, default=None,
                    help="fetch just this one year, both sides, and exit")
    ap.add_argument("--side", choices=("bid", "ask", "both"), default="both")
    ap.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    ap.add_argument("--force", action="store_true",
                    help="re-fetch years whose file already exists")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.check:
        check(args.raw_dir, args.start, args.end)
        return 0

    try:
        import dukascopy_python as dk
    except ImportError:
        print("dukascopy-python is not installed.\n\n    pip install dukascopy-python\n")
        return 1

    years = [args.year] if args.year is not None else list(range(args.start, args.end + 1))
    sides = ("bid", "ask") if args.side == "both" else (args.side,)

    os.makedirs(args.raw_dir, exist_ok=True)
    print(f"XAU/USD, 1-minute, {sides}, years {years[0]}-{years[-1]}")
    print(f"  {SUBCHUNK_DAYS}-day sub-chunks, {PAUSE_SECONDS:g}s between requests\n")

    ok, skipped, complaints = 0, 0, []
    for year in years:
        for side in sides:
            path = raw_path(side, year, args.raw_dir)
            if os.path.exists(path) and not args.force:
                print(f"  {year} {side:4s}  already cached, skipping "
                     f"(--force to re-fetch)")
                skipped += 1
                continue

            print(f"  {year} {side:4s}  fetching ...", flush=True)
            raw = fetch_year(dk, side, year, args.debug)
            if raw is None:
                print("      nothing returned (before the feed existed, "
                     "or a genuine gap)")
                continue
            df = normalise(raw)
            found = sanity(f"{year} {side}", df)
            if found:
                complaints.extend(found)
                print("      REJECTED: " + "; ".join(found))
                continue

            df.to_csv(path)
            ok += 1
            print(f"      {len(df):,} bars  {df.index[0]:%Y-%m-%d} -> "
                 f"{df.index[-1]:%Y-%m-%d}  median {df['Volume'].median():.0f} ticks/bar")
            time.sleep(PAUSE_SECONDS)

    print(f"\n{ok} file(s) written, {skipped} already cached.")
    if complaints:
        print("\nNothing was written for the rejected years:")
        for c in complaints:
            print(f"  {c}")
    print("\nCheck progress any time with:  python fetch_fvg_raw.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
