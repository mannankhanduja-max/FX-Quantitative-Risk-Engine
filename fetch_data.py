"""
Download the instrument universe from Yahoo Finance into a local cache.

    python fetch_data.py                    # the configured universe
    python fetch_data.py --start 2003-01-01
    python fetch_data.py --with-gold        # adds spot gold
    python fetch_data.py --check            # report the cache, download nothing

THIS IS THE ONLY STEP THAT NEEDS INTERNET. It writes one CSV per
symbol into data/yahoo/, and every other part of the engine reads
those files. Run it once; re-run it when you want fresher data.

That split is deliberate. A backtest whose inputs can change under
it - because a vendor revised a bar, or rate-limited you halfway
through - is not reproducible. Cached CSVs make every later run
identical and offline.

Universe comes from config.py, so this and the engine and
quant_metrics.py all describe the same book.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from fxrisk.data.yahoo import DEFAULT_CACHE, cache_path, coverage_report  # noqa: E402


def fetch(symbol: str, start: str, end: str | None, cache_dir: str, retries: int = 3):
    """Download one symbol and write it to the cache."""
    import yfinance as yf

    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                symbol,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError("empty frame returned")

            # yfinance returns MultiIndex columns for some versions.
            if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
                df.columns = df.columns.get_level_values(0)

            df.index.name = "Date"
            os.makedirs(cache_dir, exist_ok=True)
            path = cache_path(symbol, cache_dir)
            df.to_csv(path)
            return len(df), df.index.min().date(), df.index.max().date()

        except Exception as exc:  # noqa: BLE001
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"      attempt {attempt} failed ({exc}); retrying in {wait}s")
            time.sleep(wait)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2003-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--with-gold", action="store_true",
                        help="use config.UNIVERSE_FX_GOLD instead of the default")
    parser.add_argument("--check", action="store_true",
                        help="report cache coverage and exit")
    args = parser.parse_args()

    universe = config.UNIVERSE_FX_GOLD if args.with_gold else config.UNIVERSE
    symbols = {i.name: i.yahoo for i in universe}

    if args.check:
        rep = coverage_report(symbols, args.cache_dir)
        print(rep.to_string())
        if rep.attrs.get("aligned_rows"):
            print(f"\naligned panel: {rep.attrs['aligned_rows']} rows, "
                  f"{rep.attrs['aligned_first']} to {rep.attrs['aligned_last']}")
            print(f"binding instrument (latest start): {rep.attrs['binding']}")
        else:
            print("\nno aligned panel - some symbols are missing")
        return 0

    try:
        import yfinance  # noqa: F401
    except ImportError:
        print("yfinance is not installed.\n  pip install -r requirements.txt")
        return 1

    print(f"Downloading {len(symbols)} instruments from {args.start}")
    print(f"Cache: {args.cache_dir}/\n")

    failures = []
    for name, sym in symbols.items():
        print(f"  {name:<12} {sym:<12} ", end="", flush=True)
        try:
            n, first, last = fetch(sym, args.start, args.end, args.cache_dir)
            print(f"{n:>6} rows   {first} to {last}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED: {exc}")
            failures.append((name, sym, str(exc)))

    print()
    if failures:
        print("FAILED SYMBOLS - the panel will not build until these resolve:")
        for name, sym, exc in failures:
            print(f"  {name} ({sym}): {exc}")
        print("\nA wrong Yahoo symbol returns an empty frame rather than an")
        print("error, which is why this fails loudly here instead of")
        print("silently producing a short panel later.")
        return 1

    rep = coverage_report(symbols, args.cache_dir)
    print(rep.to_string())
    if rep.attrs.get("aligned_rows"):
        print(f"\naligned panel: {rep.attrs['aligned_rows']} rows, "
              f"{rep.attrs['aligned_first']} to {rep.attrs['aligned_last']}")
        print(f"binding instrument (latest start): {rep.attrs['binding']}")

    print("\nNext:")
    print("  python run_risk_report.py --source yahoo")
    print("  python docs/make_figures.py --source yahoo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
