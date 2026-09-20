"""
Backtest the breakout/retracement rule on the intraday universe.

    python breakout_test.py
    python breakout_test.py --no-breakeven     # the control run
    python breakout_test.py --no-trend         # drop the VWAP/EMA filter
    python breakout_test.py --sweep            # parameter neighbourhood
    python breakout_test.py --be-pips 20

WHAT TO READ FIRST
-------------------
Not the total R. The win rate against the COST-ADJUSTED hurdle.
At 2:1 the folklore number is 33.3%, but the round trip is paid
whether the trade wins or loses, and in units of risk that is

    cost_R = 2 * cost / stop_distance

so the real hurdle is (1 + cost_R) / (1 + rr). It moves with the
stop distance, which this rule sets from market structure rather
than a constant - so the hurdle is different for every trade and
the figure printed is the average.

AND THEN READ THE SAMPLE SIZE
------------------------------
Yahoo serves 60 days of 15-minute bars. That is about 1,500 bars
per instrument and, after three gates, a few dozen trades. With
40 trades, a true 40% win rate produces an observed rate anywhere
between roughly 25% and 56% one time in twenty. Nothing in this
output can distinguish a working rule from a lucky one. It can,
and does, distinguish a working rule from a BROKEN one, which is
what it is for.

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
from fxrisk.data import intraday  # noqa: E402
from fxrisk.risk import barriers  # noqa: E402
from fxrisk.strategies import breakout_retrace as br  # noqa: E402


def run_one(inst, cfg, rr, cost_bp, max_bars, be_frac, interval, cache_dir):
    """Setups and outcomes for one instrument."""
    bars = intraday.load_symbol(inst.yahoo, interval, cache_dir)
    setups = br.find_setups(bars, cfg)
    if setups.empty:
        return bars, setups, pd.DataFrame()
    trades = barriers.walk_explicit(
        bars,
        setups[["bar", "side", "entry", "stop"]],
        rr=rr,
        max_bars=max_bars,
        cost_bp=cost_bp,
        breakeven_frac=be_frac,
    )
    for col in ("retrace_frac", "wait_bars", "stop_from"):
        if col in setups.columns and len(trades) == len(setups):
            trades[col] = setups[col].to_numpy()
    return bars, setups, trades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--cache-dir", default=intraday.DEFAULT_CACHE)
    ap.add_argument("--rr", type=float, default=config.BREAKOUT_RR)
    ap.add_argument("--cost-bp", type=float, default=config.BREAKOUT_COST_BP)
    ap.add_argument("--max-bars", type=int, default=config.BREAKOUT_MAX_BARS)
    ap.add_argument("--lookback", type=int, default=config.BREAKOUT_LOOKBACK)
    ap.add_argument("--retrace-max-bars", type=int,
                    default=config.RETRACE_MAX_BARS)
    ap.add_argument("--min-fraction", type=float,
                    default=config.RETRACE_MIN_FRACTION)
    ap.add_argument("--be-pips", type=float, default=10.0,
                    help="breakeven trigger, in the instrument's pips")
    ap.add_argument("--no-breakeven", action="store_true")
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cfg = br.SetupConfig(
        lookback=args.lookback,
        retrace_max_bars=args.retrace_max_bars,
        min_fraction=args.min_fraction,
        max_fraction=config.RETRACE_MAX_FRACTION,
        stop_min_sigma=config.BREAKOUT_STOP_MIN_SIGMA,
        use_trend_filter=not args.no_trend,
    )

    print("Breakout -> retracement -> resumption, "
          f"{args.rr:g}:1, {args.cost_bp:g}bp per side, {args.interval} bars")
    print(f"  trend filter    {'VWAP/EMA bias' if cfg.use_trend_filter else 'OFF'}")
    print(f"  breakeven stop  "
          f"{'OFF (control)' if args.no_breakeven else f'{args.be_pips:g} pips'}")
    print()

    rows, pooled = [], []
    for inst in config.UNIVERSE_INTRADAY:
        be_frac = (
            None
            if args.no_breakeven
            else (inst.be_bp / 10_000.0) * (args.be_pips / 10.0)
        )
        try:
            bars, setups, trades = run_one(
                inst, cfg, args.rr, args.cost_bp, args.max_bars,
                be_frac, args.interval, args.cache_dir,
            )
        except FileNotFoundError as exc:
            print(f"  {inst.name:8s} {exc}".splitlines()[0])
            continue
        except ValueError as exc:
            print(f"  {inst.name:8s} unusable: {exc}")
            continue

        s = barriers.summarise_explicit(trades, args.rr)
        s.update(instrument=inst.name, symbol=inst.yahoo,
                 bars=len(bars), setups=len(setups))
        s["dropped"] = int(trades.attrs.get("dropped_overlapping", 0))
        rows.append(s)
        if not trades.empty:
            pooled.append(trades)

    if not rows:
        print("Nothing to report. Run:  python fetch_intraday.py")
        return 1

    out = pd.DataFrame(rows)
    cols = ["instrument", "symbol", "bars", "trades", "win_rate",
            "breakeven_wr", "gap_vs_breakeven", "mean_net_R",
            "total_net_R", "target_hits", "stop_hits", "be_exits",
            "time_exits", "dropped", "mean_bars"]
    have = [c for c in cols if c in out.columns]
    print("PER INSTRUMENT\n")
    print(out[have].round(4).to_string(index=False))

    if not pooled:
        print("\nNo trades. The gates are too tight for this sample.")
        return 0

    allt = pd.concat(pooled, ignore_index=True)
    p = barriers.summarise_explicit(allt, args.rr)

    print("\nPOOLED\n")
    print(f"  trades                {p['trades']}")
    print(f"  win rate              {p['win_rate']:.2%}")
    print(f"  cost-adjusted hurdle  {p['breakeven_wr']:.2%}"
          f"   (folklore says {1 / (1 + args.rr):.2%})")
    print(f"  gap                   {p['gap_vs_breakeven']:+.2%}")
    print(f"  mean net R            {p['mean_net_R']:+.4f}")
    print(f"  total net R           {p['total_net_R']:+.2f}")
    print(f"  target / stop / BE    {p['target_hits']} / {p['stop_hits']}"
          f" / {p['be_exits']}  (time {p['time_exits']})")
    print(f"  mean stop distance    {p['mean_stop_frac']:.3%} of price")
    print(f"  cost in units of risk {p['cost_R']:.4f} R")
    print(f"  intrabar ambiguous    {p['ambiguous_share']:.1%}")
    dropped = int(out["dropped"].sum()) if "dropped" in out.columns else 0
    if dropped:
        print(f"  overlapping, dropped  {dropped}")
        print("  Those entries fired while a trade was already open. Taking")
        print("  them would have levered the book past the stated one unit")
        print("  of risk per trade, so they are not counted.")

    if not args.no_breakeven:
        print(f"\n  breakeven stop armed  {p['be_armed_share']:.1%} of trades")
        print(f"  armed, still reached target  {p['be_armed_then_target']}")
        print("  Run --no-breakeven for the control. The rule is only worth")
        print("  keeping if it removes more from the losing tail than it")
        print("  takes off the winners, and that comparison is the point.")

    # The t-statistic on mean R is what settles it, not the total.
    r = allt["net_R"].to_numpy(dtype="float64")
    if len(r) > 5 and r.std(ddof=1) > 0:
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        print(f"\n  t-stat on mean R      {t:+.2f}  (n={len(r)})")
        print("  With a sample this small, |t| under 2 means the result is")
        print("  indistinguishable from zero - in either direction.")

    if args.sweep:
        print("\nPARAMETER NEIGHBOURHOOD\n")
        print("  A result that survives only at one setting is noise. What")
        print("  matters is whether the sign is stable across the grid.\n")
        grid = []
        for lb in (10, 20, 30):
            for mf in (0.25, 0.33, 0.50):
                c = br.SetupConfig(
                    lookback=lb, retrace_max_bars=args.retrace_max_bars,
                    min_fraction=mf, max_fraction=config.RETRACE_MAX_FRACTION,
                    stop_min_sigma=config.BREAKOUT_STOP_MIN_SIGMA,
                    use_trend_filter=cfg.use_trend_filter,
                )
                acc = []
                for inst in config.UNIVERSE_INTRADAY:
                    bef = (
                        None if args.no_breakeven
                        else (inst.be_bp / 10_000.0) * (args.be_pips / 10.0)
                    )
                    try:
                        _, _, t_ = run_one(inst, c, args.rr, args.cost_bp,
                                           args.max_bars, bef, args.interval,
                                           args.cache_dir)
                    except (FileNotFoundError, ValueError):
                        continue
                    if not t_.empty:
                        acc.append(t_)
                if not acc:
                    continue
                s = barriers.summarise_explicit(
                    pd.concat(acc, ignore_index=True), args.rr)
                grid.append({"lookback": lb, "min_frac": mf,
                             "trades": s["trades"],
                             "win_rate": s["win_rate"],
                             "gap": s["gap_vs_breakeven"],
                             "total_R": s["total_net_R"]})
        if grid:
            print(pd.DataFrame(grid).round(4).to_string(index=False))

    print("\n  BACKTEST-ONLY. Not a recommendation to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
