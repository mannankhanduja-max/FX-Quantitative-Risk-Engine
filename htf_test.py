"""
The same cascade, moved up to 1h / 4h / daily.

WHY THIS IS THE ONE VARIANT WORTH RUNNING
-------------------------------------------
Every negative result in this study reduces to one ratio. Cost in
units of risk is

    cost_R = 2 * spread / stop_distance

and on 5-minute triggers the stop is ~6.8bp while the round trip is
~1.1bp, so 21% of what is risked goes to friction against a 12.8%
edge. Widening the stop within the 5m cascade does not help, because
the reversion being traded lives within about one 5m ATR of the
extreme - the edge decays at the same rate the cost does.

Moving the whole cascade up is different in kind. Volatility scales
roughly with the square root of time, so a 1-hour ATR should be about
sqrt(12) = 3.5x a 5-minute ATR. The spread does not scale at all - it
is the same 1.1bp whether the trade lasts five minutes or five hours.
So cost_R should fall by roughly that same 3.5x for free.

What is NOT free is the edge. If the reversion is a microstructure
effect it may simply not exist at hourly resolution, and the trade
count falls by an order of magnitude, which costs statistical power.
That is the measurement.

The cascade code is timeframe-agnostic - setups_30m, entries_5m and
to_30m_positions care about the ORDER of the frames, not their
duration - so this runs the identical rule with (1h, 4h, daily)
substituted for (5m, 30m, 1h). Nothing about the logic changes,
which is what makes the comparison clean.

BACKTEST-ONLY.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import config
from fxrisk.data import intraday
from fxrisk.research import barrier_prob as bp
from fxrisk.risk import barriers, spread
from fxrisk.strategies import mtf, smc

AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


# (trigger, setup, bias) - the cascade's three frames, low to high
LADDERS = {
    "5m/30m/1h":   ("5min", "30min", "1h"),
    "15m/1h/4h":   ("15min", "1h", "4h"),
    "1h/4h/1d":    ("1h", "4h", "1D"),
    "4h/1d/1W":    ("4h", "1D", "1W"),
}


def run_one(inst, ladder, args):
    b5 = intraday.load_symbol(inst.yahoo, "5m")
    lo, mid, _hi = LADDERS[ladder]
    trig = b5 if lo == "5min" else resample(b5, lo)
    setup_f = resample(b5, mid)

    cfg = mtf.MTFConfig(stop_sigma=args.stop_sigma, stop_mode="atr",
                        atr_period=args.atr_period,
                        setup_lookback=args.lookback,
                        trigger_window=args.trigger_window,
                        max_bars_30m=args.max_bars)

    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg, kinds=("breakout", "fakeout", "retrace"),
                            zone_frame=zf)
    entries = mtf.entries_5m(trig, setups, None, cfg)
    if entries.empty:
        return None

    keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                           mtf.INSTRUMENT_SESSIONS[inst.name])
    entries = entries[keep.to_numpy()].reset_index(drop=True)
    if args.blackout and not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)
    if entries.empty:
        return None

    pos = mtf.to_30m_positions(entries, setup_f)
    if pos.empty:
        return None

    cost = spread.real_cost_bp(setup_f, inst.yahoo, "5m",
                               commission_bp=args.commission_bp).to_numpy()
    t = barriers.walk_explicit(
        setup_f, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=args.rr, max_bars=args.max_bars, cost_bp=cost, breakeven_frac=None)
    if t.empty:
        return None
    t = t.copy()
    t["instrument"] = inst.name
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--stop-sigma", type=float, default=1.0)
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--trigger-window", type=int, default=6)
    ap.add_argument("--max-bars", type=int, default=80)
    ap.add_argument("--commission-bp", type=float, default=0.35)
    ap.add_argument("--blackout", action="store_true", default=True)
    args = ap.parse_args()

    print("THE SAME CASCADE AT FOUR TIMEFRAME LADDERS")
    print(f"  {args.rr:g}:1, ATR({args.atr_period}) x{args.stop_sigma:g}, "
          f"{args.max_bars}-bar limit, measured spread + "
          f"{args.commission_bp:g}bp/side\n")
    print(f"{'ladder':<13}{'n':>6}{'/yr':>6}{'time%':>7}{'stop bp':>9}"
          f"{'win':>8}{'BM':>7}{'z':>7}{'gross':>9}{'cost':>8}{'net':>9}{'t':>7}")

    for name in LADDERS:
        pool = []
        for inst in config.UNIVERSE_INTRADAY:
            t = run_one(inst, name, args)
            if t is not None:
                pool.append(t)
        if not pool:
            print(f"{name:<13}  no trades")
            continue
        a = pd.concat(pool, ignore_index=True).sort_values("entry_time")
        r = a["unit_net_R"]
        b = bp.benchmark(a)
        span = (a["entry_time"].iloc[-1] - a["entry_time"].iloc[0]).days / 365.25
        tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        print(f"{name:<13}{len(a):>6}{len(a)/span:>6.0f}"
              f"{(a['outcome']=='time').mean():>7.1%}"
              f"{a['stop_frac'].median()*1e4:>9.1f}"
              f"{b['actual_win']:>8.2%}{b['model_win']:>7.2%}{b['z']:>+7.2f}"
              f"{a['gross_R'].mean():>+9.4f}{a['cost_R'].mean():>8.4f}"
              f"{r.mean():>+9.4f}{tstat:>+7.2f}")


if __name__ == "__main__":
    main()
