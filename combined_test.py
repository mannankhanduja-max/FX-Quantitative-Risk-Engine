"""
The 1h/4h/1d cascade on every bar available from 2022 onward.

WHAT THIS IS, AND WHAT IT IS NOT
----------------------------------
Both windows in this repository have now been used: Sep 2024 - Sep
2026 built the rule, and Sep 2021 - Aug 2023 tested it once and
failed it. Pooling them gives a bigger sample. It gives NO new
out-of-sample evidence, and a pooled number is not a third opinion -
it is the same two opinions averaged.

What it is good for is an effect-size estimate on the largest sample
available, and a per-year breakdown that shows whether the effect is
present in each year or concentrated in one.

THERE IS A ONE-YEAR HOLE. Sep 2023 - Sep 2024 was never downloaded,
so this is 2022-01 -> 2023-08 plus 2024-09 -> 2026-09. The gap costs
coverage but not correctness: each trade is independent of the gap,
which simply contributes no trades.

    python fetch_intraday.py --interval 5m --years 1 --start 2023-09-01 --tag 2023
    python fetch_intraday.py --interval 5m --years 1 --start 2023-09-01 --tag 2023 --side ask

closes it.

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
CHUNKS = [("5m_2021", "5m_ask_2021"), ("5m", "5m_ask")]


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def run_chunk(inst, iv, ask_iv, args):
    b5 = intraday.load_symbol(inst.yahoo, iv)
    b5 = b5[b5.index >= pd.Timestamp(args.start, tz="America/New_York")]
    if len(b5) < 5000:
        return None
    trig, setup_f = resample(b5, "1h"), resample(b5, "4h")

    cfg = mtf.MTFConfig(stop_sigma=1.0, stop_mode="atr", atr_period=14,
                        setup_lookback=20, trigger_window=6, max_bars_30m=80)
    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg,
                            kinds=("breakout", "fakeout", "retrace"),
                            zone_frame=zf)
    entries = mtf.entries_5m(trig, setups, None, cfg)
    if entries.empty:
        return None
    keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                           mtf.INSTRUMENT_SESSIONS[inst.name])
    entries = entries[keep.to_numpy()].reset_index(drop=True)
    if not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)
    if entries.empty:
        return None
    pos = mtf.to_30m_positions(entries, setup_f)
    if pos.empty:
        return None
    cost = spread.real_cost_bp(setup_f, inst.yahoo, iv,
                               commission_bp=args.commission_bp,
                               ask_interval=ask_iv).to_numpy()
    t = barriers.walk_explicit(
        setup_f, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=args.rr, max_bars=80, cost_bp=cost, breakeven_frac=None)
    if t.empty:
        return None
    t = t.copy()
    t["instrument"] = inst.name
    return t


def row(tag, g, ind=""):
    r = g["unit_net_R"]
    b = bp.benchmark(g)
    ts = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 else np.nan
    print(f"{ind}{tag:<12}{len(g):>6}{b['actual_win']:>8.2%}{b['model_win']:>7.2%}"
          f"{b['z']:>+7.2f}{g['gross_R'].mean():>+9.4f}{g['cost_R'].mean():>8.4f}"
          f"{r.mean():>+9.4f}{ts:>+7.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--commission-bp", type=float, default=0.35)
    args = ap.parse_args()

    parts = []
    for inst in config.UNIVERSE_INTRADAY:
        for iv, ask in CHUNKS:
            t = run_chunk(inst, iv, ask, args)
            if t is not None:
                parts.append(t)
    a = pd.concat(parts, ignore_index=True).sort_values("entry_time")
    a["t"] = pd.to_datetime(a["entry_time"], utc=True, format="mixed")

    print(f"1h/4h/1d cascade, {args.rr:g}:1, ATR(14)x1.0, measured spread "
          f"+ {args.commission_bp:g}bp/side")
    print(f"from {args.start}  ({a['t'].min().date()} -> {a['t'].max().date()}, "
          f"one-year hole Sep 2023 - Sep 2024)\n")
    print(f"{'':<12}{'n':>6}{'win':>8}{'BM':>7}{'z':>7}{'gross':>9}"
          f"{'cost':>8}{'net':>9}{'t':>7}")
    row("POOLED", a)
    print()
    for i, g in a.groupby("instrument"):
        row(i, g, "  ")
    print("\n  by calendar year")
    for y, g in a.groupby(a["t"].dt.year):
        if len(g) < 30:
            continue
        row(str(y), g, "  ")


if __name__ == "__main__":
    main()
