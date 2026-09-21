"""
The one-shot test: shallow retracements on data nothing here has seen.

    python prior_period_test.py --verify     # reproduce the seen period first
    python prior_period_test.py              # then the unseen one, once

WHY THIS EXISTS
----------------
After sixty-odd configurations on Sep 2024 - Sep 2026, one pattern
had both a consistent shape and an economic story: on 1-hour bars,
shallow pullbacks (33-50% of the breakout impulse) did better than
medium, and medium better than deep. Before costs, shallow reached
t = +1.77.

That pattern was found by LOOKING at Sep 2024 - Sep 2026, so no
split of that period is a clean test any more - any "held-out" slice
of it was already in view when "shallow" was chosen. The only honest
test is data from outside it. Dukascopy serves history back to 2003,
so the test runs on Sep 2022 - Sep 2024, which no part of this study
has touched.

THE RULE IS FROZEN BELOW AND WAS WRITTEN DOWN BEFORE THE RUN.
Changing anything in FROZEN after seeing the result turns this back
into another in-sample number, and the file's git history would show
it.

WHAT COUNTS AS A PASS
----------------------
Stated in advance, so the result cannot be argued into one afterwards:

  PRIMARY   shallow bucket, pooled across all four instruments, WITH
            the 1bp cost: mean net R > 0.

  PATTERN   the depth ordering replicates: at zero cost, mean R for
            shallow > medium > deep.

A pass on PRIMARY alone is weak evidence - one period, one draw. A
fail on PRIMARY is strong evidence against, because the rule was
chosen as the most flattering thing the seen period offered.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from fxrisk.data.intraday import cache_path, load_symbol  # noqa: E402
from fxrisk.risk import barriers  # noqa: E402
from fxrisk.strategies import breakout_retrace as br  # noqa: E402

# ------------------------------------------------------------------
# FROZEN. Written before the unseen period was loaded.
# ------------------------------------------------------------------
FROZEN = dict(
    interval="1h",
    lookback=5,              # 5 hours
    retrace_max_bars=3,      # 3 hours
    stop_min_sigma=1.0,
    use_trend_filter=False,
    rr=2.0,
    max_bars=10,             # 10 hours
    cost_bp=1.0,             # per side
    breakeven=None,          # removed
)
SHALLOW = (0.33, 0.50)
BUCKETS = {"shallow 33-50%": (0.33, 0.50),
           "medium  50-66%": (0.50, 0.66),
           "deep    66-100%": (0.66, 1.00)}

# The seen period starts here. Everything strictly before it is new.
SEEN_START = pd.Timestamp("2024-09-19", tz="America/New_York")


def hourly(symbol: str) -> pd.DataFrame:
    """Exact 1h bars from the 15m cache, rebuilt so they match its span."""
    src = cache_path(symbol, "15m", "data/intraday")
    df = pd.read_csv(src, index_col="Datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    agg = df.sort_index().resample("1h", label="left", closed="left").agg(
        {"Open": "first", "High": "max", "Low": "min",
         "Close": "last", "Volume": "sum"}).dropna(subset=["Open"])
    agg.index.name = "Datetime"
    agg.to_csv(cache_path(symbol, "1h", "data/intraday"))
    return load_symbol(symbol, "1h", "data/intraday")


def run(bars_by_inst: dict, band: tuple, cost_bp: float) -> pd.DataFrame:
    cfg = br.SetupConfig(
        lookback=FROZEN["lookback"],
        retrace_max_bars=FROZEN["retrace_max_bars"],
        min_fraction=band[0], max_fraction=band[1],
        stop_min_sigma=FROZEN["stop_min_sigma"],
        use_trend_filter=FROZEN["use_trend_filter"],
    )
    acc = []
    for name, bars in bars_by_inst.items():
        s = br.find_setups(bars, cfg)
        if s.empty:
            continue
        t = barriers.walk_explicit(
            bars, s[["bar", "side", "entry", "stop"]],
            rr=FROZEN["rr"], max_bars=FROZEN["max_bars"],
            cost_bp=cost_bp, breakeven_frac=FROZEN["breakeven"])
        if not t.empty:
            t["inst"] = name
            acc.append(t)
    return pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()


def stats(t: pd.DataFrame) -> dict:
    s = barriers.summarise_explicit(t, FROZEN["rr"])
    r = t["net_R"].to_numpy(dtype="float64")
    s["t"] = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 5 else np.nan
    return s


def line(label: str, s: dict) -> str:
    return (f"  {label:22s} n {s['trades']:5d}  win {s['win_rate_barrier']:6.2%}  "
            f"hurdle {s['breakeven_wr']:6.2%}  meanR {s['mean_net_R']:+.4f}  "
            f"t {s['t']:+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the SEEN period only; touches nothing new")
    args = ap.parse_args()

    full = {i.name: hourly(i.yahoo) for i in config.UNIVERSE_INTRADAY}

    if args.verify:
        block = {k: v[v.index >= SEEN_START] for k, v in full.items()}
        label = "SEEN period, re-downloaded (should match the earlier run)"
    else:
        block = {k: v[v.index < SEEN_START] for k, v in full.items()}
        label = "UNSEEN period - the one-shot test"

    print(f"{label}\n")
    for k, v in block.items():
        print(f"  {k:8s} {len(v):6d} bars  {v.index[0].date()} -> {v.index[-1].date()}")
    print()
    print("  Frozen: " + ", ".join(f"{k}={v}" for k, v in FROZEN.items()))
    print(f"          shallow band {SHALLOW}\n")

    primary = stats(run(block, SHALLOW, FROZEN["cost_bp"]))
    print("PRIMARY - shallow, pooled, with costs\n")
    print(line("shallow, 1bp/side", primary))

    print("\nPATTERN - depth ordering at zero cost\n")
    zero = {}
    for name, band in BUCKETS.items():
        zero[name] = stats(run(block, band, 0.0))
        print(line(name, zero[name]))

    print("\nPer instrument, shallow, with costs\n")
    t = run(block, SHALLOW, FROZEN["cost_bp"])
    for name, g in t.groupby("inst"):
        print(line(name, stats(g)))

    if not args.verify:
        m = [zero[k]["mean_net_R"] for k in BUCKETS]
        print("\nVERDICT (criteria fixed in the docstring before the run)\n")
        print(f"  PRIMARY  shallow mean R with costs > 0:  "
              f"{'PASS' if primary['mean_net_R'] > 0 else 'FAIL'}  "
              f"({primary['mean_net_R']:+.4f}, t {primary['t']:+.2f})")
        ordered = m[0] > m[1] > m[2]
        print(f"  PATTERN  shallow > medium > deep at zero cost:  "
              f"{'REPLICATES' if ordered else 'DOES NOT REPLICATE'}  "
              f"({m[0]:+.3f} / {m[1]:+.3f} / {m[2]:+.3f})")
    print("\n  BACKTEST-ONLY. Not a recommendation to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
