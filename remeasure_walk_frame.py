"""
Re-measure the real rule at the monitoring resolution the null test
says is unbiased.

This is NOT a new test and searches nothing. `null_benchmark_test.py`
showed that walking the exit on a frame coarser than the entries
inflates the measured win rate on data containing nothing: +1.93pp at a
4h walk, -0.78pp at 1h, from unmonitored price movement between the
fill and the first bar that can register a touch. Every headline result
in this study was walked at 4h.

So the same trades are re-scored at 1h, 15m and 5m monitoring. Entries,
stops, targets and the bar limit's wall-clock span are untouched; only
the resolution at which the two barriers are watched changes. The
question is what is left of z = +2.69 once the bias is removed.

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
CHUNKS = [("5m_2021", "5m_ask_2021"), ("5m_2023", "5m_ask_2023"),
          ("5m", "5m_ask")]
WALKS = {"4h": 1, "1h": 4, "15min": 16, "5min": 48}
OUT = "results/remeasure_walk_frame.txt"


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def one(inst, iv, ask_iv, walk_rule, commission_bp):
    try:
        b5 = intraday.load_symbol(inst.yahoo, iv)
    except FileNotFoundError:
        return None
    trig, setup_f = resample(b5, "1h"), resample(b5, "4h")
    if len(setup_f) < 300:
        return None

    cfg = mtf.MTFConfig(stop_sigma=1.0, stop_mode="atr", atr_period=14,
                        setup_lookback=20, trigger_window=6, max_bars_30m=80)
    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg, zone_frame=zf)
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

    walk_f = setup_f if walk_rule == "4h" else resample(b5, walk_rule)
    pos = mtf.to_30m_positions(entries, walk_f)
    if pos.empty:
        return None
    try:
        cost = spread.real_cost_bp(walk_f, inst.yahoo, iv,
                                  commission_bp=commission_bp,
                                  ask_interval=ask_iv).to_numpy()
    except FileNotFoundError:
        return None
    t = barriers.walk_explicit(
        walk_f, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=3.0, max_bars=80 * WALKS[walk_rule], cost_bp=cost,
        breakeven_frac=None)
    if t.empty:
        return None
    t = t.copy()
    t["instrument"] = inst.name
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission-bp", type=float, default=0.35)
    args = ap.parse_args()

    L = ["THE SAME TRADES, RE-SCORED AT FOUR MONITORING RESOLUTIONS",
         "  1h/4h/1d cascade, 3:1, ATR(14) x1.0, measured spread + "
         f"{args.commission_bp:g}bp/side",
         "  entries, stops and targets identical in every row - only the",
         "  resolution the two barriers are watched at changes",
         "  null-test bias on no-information data: 4h +1.93pp, 1h -0.78pp,"
         " 15m -2.41pp, 5m -3.77pp", ""]
    L.append(f"  {'walk':>7}{'n':>7}{'time%':>7}{'win':>8}{'BM':>8}"
             f"{'excess':>9}{'z':>8}{'adj z':>8}{'gross':>9}{'net':>9}{'t':>7}")

    NULLBIAS = {"4h": 0.0193, "1h": -0.0078, "15min": -0.0241,
                "5min": -0.0377}
    for walk in WALKS:
        pool = []
        for inst in config.UNIVERSE_INTRADAY:
            for iv, ask in CHUNKS:
                t = one(inst, iv, ask, walk, args.commission_bp)
                if t is not None:
                    pool.append(t)
        if not pool:
            L.append(f"  {walk:>7}  no trades")
            continue
        a = (pd.concat(pool, ignore_index=True)
             .sort_values("entry_time")
             .drop_duplicates(subset=["instrument", "entry_time"],
                              keep="first"))
        b = bp.benchmark(a)
        r = a["unit_net_R"]
        tstat = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        # z re-centred on what the null actually delivers at this frame
        adj = b["excess"] - NULLBIAS[walk]
        adj_z = b["z"] * (adj / b["excess"]) if b["excess"] else float("nan")
        L.append(f"  {walk:>7}{len(a):>7}{(a['outcome']=='time').mean():>7.1%}"
                 f"{b['actual_win']:>8.2%}{b['model_win']:>8.2%}"
                 f"{b['excess']:>+9.2%}{b['z']:>+8.2f}{adj_z:>+8.2f}"
                 f"{a['gross_R'].mean():>+9.4f}{r.mean():>+9.4f}"
                 f"{tstat:>+7.2f}")

    L += ["", "  'adj z' rescales z by how much of the excess survives after",
          "  subtracting what a driftless random walk produces at the same",
          "  frame. It is an estimate, not a test statistic."]
    text = "\n".join(L) + "\n"
    print(text)
    with open(OUT, "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
