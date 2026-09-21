"""
Do the risk gates rescue the breakout rule? Held out, once.

    python build_regime.py        # first, caches the GARCH paths
    python gate_test.py
    python gate_test.py --min-trades 40

THE PROTOCOL, AND WHY IT IS NOT OPTIONAL HERE
----------------------------------------------
By the time these gates were written, roughly forty configurations
had been run against the same 433 trades - stop floors, breakeven
on and off, time limits, lookbacks, retracement bands. Anything
further chosen on the full sample is fitted to noise by
construction, and a filter is the easiest way in the world to fit
noise: any partition of a noisy sample has a better half.

So the sample is split once, chronologically, per instrument:

    IN-SAMPLE     first 67%   choose EVERYTHING here
    HELD OUT      last 33%    touched once, rule frozen

Everything means everything. Not only the gate thresholds but the
stop floor too, even though a stop floor of 4 sigma already looked
best on the full sample - that figure has seen the holdout, so it
cannot be used, and the in-sample block has to rediscover it or
not.

WHAT THE GATES ARE
-------------------
  g_regime   GARCH(1,1)-t conditional sigma inside a trailing band
  g_var      conditional Student-t VaR below a trailing quantile
  corr       the most-correlated partner's recent move agrees with
             the trade's direction, once the sign of rho is applied

The first two are near-duplicates - VaR here is a monotone
function of sigma - so switching both on is close to counting one
gate twice. `fxrisk/strategies/regime.py` says so at length. The
correlation gate is the only one carrying information the
instrument's own price does not already contain.

THE COMPARISON THAT MATTERS
----------------------------
Not "is the gated result positive". It is "does the gated result
beat the UNGATED one on the same held-out bars". A filter that
improves the number by refusing three quarters of the trades has
not found anything if the quarter it keeps is no better than the
whole; it has just reduced the sample until the error bars
swallowed the loss. Both are reported side by side, along with
how many trades survived, because a gate that leaves twenty
trades has told you nothing whatever the win rate says.

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
from build_regime import path_for  # noqa: E402
from fxrisk.data import intraday  # noqa: E402
from fxrisk.risk import barriers  # noqa: E402
from fxrisk.strategies import breakout_retrace as br  # noqa: E402

SPLIT = 0.67


def load_all(interval: str, cache_dir: str):
    """Bars and cached gate frames, per instrument."""
    bars, gatef = {}, {}
    for inst in config.UNIVERSE_INTRADAY:
        bars[inst.name] = intraday.load_symbol(inst.yahoo, interval, cache_dir)
        p = path_for(inst.yahoo, interval, cache_dir)
        if not os.path.exists(p):
            raise SystemExit(
                f"No cached regime data at {p}.\nRun:  python build_regime.py"
            )
        g = pd.read_csv(p, index_col="Datetime")
        # The cache spans a DST change, so the offsets in the file
        # are mixed (-04:00 and -05:00) and pandas parses the column
        # as plain objects unless it is told to normalise to UTC
        # first. Skipping utc=True here leaves an object index that
        # silently fails to align with the bars.
        g.index = pd.to_datetime(g.index, utc=True).tz_convert(
            "America/New_York")
        gatef[inst.name] = g
    return bars, gatef


def run_block(bars, gate, block, stop_sigma, use_be, use_corr, rho_min,
              rr, cost_bp, max_bars, be_bp):
    """Trades for one instrument over one contiguous block of bars."""
    b = bars.iloc[block[0]:block[1]]
    if len(b) < 500:
        return pd.DataFrame()

    cfg = br.SetupConfig(
        lookback=config.BREAKOUT_LOOKBACK,
        retrace_max_bars=config.RETRACE_MAX_BARS,
        min_fraction=config.RETRACE_MIN_FRACTION,
        max_fraction=config.RETRACE_MAX_FRACTION,
        stop_min_sigma=stop_sigma,
        use_trend_filter=True,
    )
    setups = br.find_setups(b, cfg)
    if setups.empty:
        return pd.DataFrame()

    g = gate.reindex(b.index)
    passed = g["all_gates"].fillna(False).to_numpy()
    implied = g["implied"].fillna(0.0).to_numpy()
    rho_abs = g["rho_abs"].fillna(0.0).to_numpy()

    keep = []
    for row in setups.itertuples(index=False):
        i = int(row.bar)
        if not passed[i]:
            continue
        if use_corr:
            if rho_abs[i] < rho_min:
                continue
            if implied[i] != np.sign(row.side):
                continue
        keep.append(row._asdict() if hasattr(row, "_asdict") else row)

    if not keep:
        return pd.DataFrame()
    sel = pd.DataFrame(keep)

    return barriers.walk_explicit(
        b, sel[["bar", "side", "entry", "stop"]],
        rr=rr, max_bars=max_bars, cost_bp=cost_bp,
        breakeven_frac=(be_bp / 10_000.0) if use_be else None,
    )


def pooled(bars, gatef, blocks, stop_sigma, use_be, use_corr, rho_min, args):
    """Trades across all four instruments for one parameter set."""
    acc = []
    for inst in config.UNIVERSE_INTRADAY:
        t = run_block(
            bars[inst.name], gatef[inst.name], blocks[inst.name],
            stop_sigma, use_be, use_corr, rho_min,
            args.rr, args.cost_bp, args.max_bars, inst.be_bp,
        )
        if not t.empty:
            acc.append(t)
    return pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()


def ungated(bars, blocks, stop_sigma, use_be, args):
    """The control: same rule, no gates at all."""
    acc = []
    for inst in config.UNIVERSE_INTRADAY:
        b = bars[inst.name].iloc[blocks[inst.name][0]:blocks[inst.name][1]]
        cfg = br.SetupConfig(
            lookback=config.BREAKOUT_LOOKBACK,
            retrace_max_bars=config.RETRACE_MAX_BARS,
            min_fraction=config.RETRACE_MIN_FRACTION,
            max_fraction=config.RETRACE_MAX_FRACTION,
            stop_min_sigma=stop_sigma, use_trend_filter=True,
        )
        s = br.find_setups(b, cfg)
        if s.empty:
            continue
        t = barriers.walk_explicit(
            b, s[["bar", "side", "entry", "stop"]],
            rr=args.rr, max_bars=args.max_bars, cost_bp=args.cost_bp,
            breakeven_frac=(inst.be_bp / 10_000.0) if use_be else None,
        )
        if not t.empty:
            acc.append(t)
    return pd.concat(acc, ignore_index=True) if acc else pd.DataFrame()


# The search space. Kept as data so the figure script and this one
# search exactly the same grid - two copies of a grid drift apart,
# and then the published picture describes a choice this script
# never made.
GRID = [
    (stop_sigma, use_be, use_corr, rho_min)
    for stop_sigma in (1.0, 2.0, 3.0, 4.0)
    for use_be in (True, False)
    for use_corr, rho_min in ((False, 0.0), (True, 0.3), (True, 0.5))
]


def choose_in_sample(bars, gatef, ins, args):
    """
    Search GRID on the in-sample block and return (table, best row).

    `best` maximises in-sample mean R among configurations with at
    least `args.min_trades` trades, or is None if none qualify. The
    held-out block is never passed in, so it cannot influence the
    choice - that is the whole contract.
    """
    rows = []
    for stop_sigma, use_be, use_corr, rho_min in GRID:
        t = pooled(bars, gatef, ins, stop_sigma, use_be,
                   use_corr, rho_min, args)
        if t.empty:
            continue
        s = barriers.summarise_explicit(t, args.rr)
        rows.append({
            "stop_sigma": stop_sigma, "breakeven": use_be,
            "corr": use_corr, "rho_min": rho_min,
            "trades": s["trades"],
            "win_barrier": s["win_rate_barrier"],
            "meanR": s["mean_net_R"],
            "totalR": s["total_net_R"],
        })
    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return gdf, None
    eligible = gdf[gdf["trades"] >= args.min_trades]
    if eligible.empty:
        return gdf, None
    return gdf, eligible.loc[eligible["meanR"].idxmax()]


def report(label: str, trades: pd.DataFrame, rr: float) -> dict:
    if trades.empty:
        print(f"  {label:22s} no trades")
        return {}
    s = barriers.summarise_explicit(trades, rr)
    r = trades["net_R"].to_numpy(dtype="float64")
    t = (r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
         if len(r) > 5 and r.std(ddof=1) > 0 else np.nan)
    print(f"  {label:22s} n {s['trades']:4d}   "
          f"win(barrier) {s['win_rate_barrier']:6.2%}   "
          f"hurdle {s['breakeven_wr']:6.2%}   "
          f"gap {s['gap_vs_breakeven']:+6.2%}   "
          f"meanR {s['mean_net_R']:+.4f}   "
          f"totalR {s['total_net_R']:+8.2f}   t {t:+.2f}")
    s["t_stat"] = t
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--cache-dir", default=intraday.DEFAULT_CACHE)
    ap.add_argument("--rr", type=float, default=config.BREAKOUT_RR)
    ap.add_argument("--cost-bp", type=float, default=config.BREAKOUT_COST_BP)
    ap.add_argument("--max-bars", type=int, default=config.BREAKOUT_MAX_BARS)
    ap.add_argument("--min-trades", type=int, default=30,
                    help="in-sample configs below this are not eligible")
    args = ap.parse_args()

    bars, gatef = load_all(args.interval, args.cache_dir)

    ins = {k: (0, int(len(v) * SPLIT)) for k, v in bars.items()}
    out = {k: (int(len(v) * SPLIT), len(v)) for k, v in bars.items()}

    print("Risk-gated breakout: GARCH regime + VaR cap + correlation")
    print(f"  {args.rr:g}:1, {args.cost_bp:g}bp/side, {args.interval} bars\n")
    for inst in config.UNIVERSE_INTRADAY:
        b = bars[inst.name]
        k = ins[inst.name][1]
        print(f"  {inst.name:8s} in-sample {b.index[0].date()} -> "
              f"{b.index[k - 1].date()}   held out "
              f"{b.index[k].date()} -> {b.index[-1].date()}")

    # ---------- STEP 1: choose everything in-sample ----------
    print("\nSTEP 1  in-sample grid. Every choice is made here.\n")
    gdf, best = choose_in_sample(bars, gatef, ins, args)
    if gdf.empty:
        print("  no configuration produced trades in-sample")
        return 1
    print(gdf.round(4).to_string(index=False))
    if best is None:
        print(f"\n  Nothing cleared {args.min_trades} in-sample trades.")
        return 1
    print(f"\n  FROZEN: stop {best['stop_sigma']:g} sigma, "
          f"breakeven {bool(best['breakeven'])}, "
          f"correlation {bool(best['corr'])} (rho_min {best['rho_min']:g})")
    print(f"  In-sample mean R {best['meanR']:+.4f} on "
          f"{int(best['trades'])} trades.")
    print("  Chosen to MAXIMISE in-sample mean R, i.e. fitted in its own")
    print("  favour across 24 configurations. That is the point - the")
    print("  holdout now has to survive a choice made to flatter it.")

    # ---------- STEP 2: the holdout, touched once ----------
    print("\nSTEP 2  held-out final third, rule frozen\n")

    gated_out = pooled(bars, gatef, out, best["stop_sigma"],
                       bool(best["breakeven"]), bool(best["corr"]),
                       best["rho_min"], args)
    plain_out = ungated(bars, out, best["stop_sigma"],
                        bool(best["breakeven"]), args)

    gated_in = pooled(bars, gatef, ins, best["stop_sigma"],
                      bool(best["breakeven"]), bool(best["corr"]),
                      best["rho_min"], args)
    plain_in = ungated(bars, ins, best["stop_sigma"],
                       bool(best["breakeven"]), args)

    print("  IN-SAMPLE (where the choice was made)")
    report("gated", gated_in, args.rr)
    report("ungated control", plain_in, args.rr)
    print("\n  HELD OUT (touched once)")
    g = report("gated", gated_out, args.rr)
    p = report("ungated control", plain_out, args.rr)

    # ---------- the verdict ----------
    print("\nVERDICT\n")
    if not g or not p:
        print("  Too few held-out trades to say anything.")
        return 0

    kept = g["trades"] / max(p["trades"], 1)
    print(f"  The gates kept {kept:.0%} of the held-out trades "
          f"({g['trades']} of {p['trades']}).")

    delta = g["mean_net_R"] - p["mean_net_R"]
    print(f"  Held-out mean R: gated {g['mean_net_R']:+.4f} vs "
          f"ungated {p['mean_net_R']:+.4f}  ({delta:+.4f})")

    if g["trades"] < 30:
        print("\n  With fewer than 30 held-out trades the comparison cannot")
        print("  resolve anything. This is the filter's real failure mode:")
        print("  not a wrong answer, but a sample too small to have one.")

    if abs(g.get("t_stat", 0) or 0) < 2:
        print("\n  |t| < 2 on the gated held-out mean: indistinguishable from")
        print("  zero. A gate cannot manufacture expectancy that is not")
        print("  present in the population it selects from - it can only")
        print("  concentrate expectancy that was already there.")

    print("\n  BACKTEST-ONLY. Not a recommendation to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
