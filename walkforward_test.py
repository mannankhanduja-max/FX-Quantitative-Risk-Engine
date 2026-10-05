"""
Walk-forward the cascade: choose parameters on each training window,
score only on the test window that follows.

    python walkforward_test.py                  # whatever data is on disk
    python walkforward_test.py --train 18 --test 6

THE GRID IS FIXED HERE AND SHOULD NOT BE WIDENED LATER
--------------------------------------------------------
It is built from the parameters this study already varied, before
any walk-forward number existed. Adding to it because a result
disappoints would put the selection bias straight back in - the bias
this whole construction exists to remove.

    reward ratio    2, 3, 4
    stop            ATR(14) x 0.75, 1.0, 1.5
    setups          all three, or the two fades only

18 configurations. Each training window picks one; each test window
scores it and is never consulted for the choice.

The headline is NOT the point. `selection_stability` and the
comparison against a single fixed configuration are, because they
answer whether optimisation is doing anything at all.

BACKTEST-ONLY.
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np
import pandas as pd

import config
from fxrisk.data import intraday
from fxrisk.research import walkforward as wf
from fxrisk.risk import barriers, spread
from fxrisk.strategies import mtf, smc

AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}
CHUNKS = [("5m_2021", "5m_ask_2021"), ("5m_2023", "5m_ask_2023"),
          ("5m", "5m_ask")]

RR = (2.0, 3.0, 4.0)
STOPS = (0.75, 1.0, 1.5)
KINDS = {"all": ("breakout", "fakeout", "retrace"),
         "fades": ("fakeout", "retrace")}
# Must match the f"rr{rr:g}_atr{stop:g}_{kind}" spelling exactly.
# "rr3.0_atr1.0_all" silently matches nothing, and the fixed-config
# comparison then vanishes from the output without complaint.
FIXED = "rr3_atr1_all"              # the documented default


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def chunk_trades(inst, iv, ask_iv, args):
    """Every configuration's trades for one instrument, one data chunk."""
    try:
        b5 = intraday.load_symbol(inst.yahoo, iv)
    except FileNotFoundError:
        return []
    trig, setup_f = resample(b5, "1h"), resample(b5, "4h")
    if len(setup_f) < 300:
        return []
    try:
        cost = spread.real_cost_bp(setup_f, inst.yahoo, iv,
                                   commission_bp=args.commission_bp,
                                   ask_interval=ask_iv).to_numpy()
    except FileNotFoundError:
        return []

    out = []
    for stop, kname in itertools.product(STOPS, KINDS):
        cfg = mtf.MTFConfig(stop_sigma=stop, stop_mode="atr", atr_period=14,
                            setup_lookback=20, trigger_window=6,
                            max_bars_30m=80)
        zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
        setups = mtf.setups_30m(setup_f, cfg, kinds=KINDS[kname], zone_frame=zf)
        entries = mtf.entries_5m(trig, setups, None, cfg)
        if entries.empty:
            continue
        keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                               mtf.INSTRUMENT_SESSIONS[inst.name])
        entries = entries[keep.to_numpy()].reset_index(drop=True)
        if not entries.empty:
            bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                                  index_preopen=(inst.kind == "index"))
            entries = entries[~bad.to_numpy()].reset_index(drop=True)
        if entries.empty:
            continue
        pos = mtf.to_30m_positions(entries, setup_f)
        if pos.empty:
            continue
        ordered = pos.sort_values("bar")[["bar", "side", "entry", "stop"]]
        for rr in RR:
            t = barriers.walk_explicit(setup_f, ordered, rr=rr, max_bars=80,
                                       cost_bp=cost, breakeven_frac=None)
            if t.empty:
                continue
            t = t[["entry_time", "unit_net_R", "outcome"]].copy()
            t["config"] = f"rr{rr:g}_atr{stop:g}_{kname}"
            t["instrument"] = inst.name
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=18)
    ap.add_argument("--test", type=int, default=6)
    ap.add_argument("--commission-bp", type=float, default=0.35)
    ap.add_argument("--min-train", type=int, default=30)
    args = ap.parse_args()

    parts = []
    for inst in config.UNIVERSE_INTRADAY:
        for iv, ask in CHUNKS:
            parts += chunk_trades(inst, iv, ask, args)
    if not parts:
        raise SystemExit("no trades - check the caches")
    all_t = pd.concat(parts, ignore_index=True).sort_values("entry_time")
    all_t["entry_time"] = pd.to_datetime(all_t["entry_time"], utc=True,
                                         format="mixed")
    # Adjacent caches touch at their shared boundary day (5m_2021 ends
    # 2023-08-31, 5m_2023 begins it), so the same setup can be emitted
    # twice. Concatenating without this double-counts those trades and
    # quietly inflates every fold that spans a seam.
    before = len(all_t)
    all_t = all_t.drop_duplicates(subset=["config", "instrument",
                                          "entry_time"], keep="first")
    dropped = before - len(all_t)

    lo, hi = all_t["entry_time"].min(), all_t["entry_time"].max()
    folds = wf.make_folds(lo.normalize(), hi.normalize(),
                          train_months=args.train, test_months=args.test,
                          tz="UTC")

    print("WALK-FORWARD, 1h/4h/1d cascade")
    print(f"  data   {lo.date()} -> {hi.date()}   "
          f"{all_t['config'].nunique()} configurations")
    print(f"  folds  {len(folds)} x ({args.train}m train / {args.test}m test)")
    print(f"  cost   measured spread + {args.commission_bp:g}bp per side")
    if dropped:
        print(f"  seams  {dropped} duplicate trades dropped at cache boundaries")
    print()
    if not folds:
        raise SystemExit("not enough history for even one fold")

    def objective(g):
        return (float("-inf") if len(g) < args.min_train
                else float(g["unit_net_R"].mean()))

    if FIXED not in set(all_t["config"]):
        raise SystemExit(f"fixed config {FIXED!r} is not in the grid: "
                         f"{sorted(set(all_t['config']))[:6]}...")
    res = wf.run(all_t, folds, objective=objective, fixed_config=FIXED)
    s = wf.summarise(res)

    print("  per fold")
    print(f"    {'test window':<26}{'picked':<22}{'n':>5}{'net R':>9}")
    for _, p in res["picks"].iterrows():
        seg = res["oos"][res["oos"]["fold_test_start"] == p["test_start"]]
        m = seg["unit_net_R"].mean() if len(seg) else float("nan")
        win = f"{p['test_start'].date()} -> {p['test_end'].date()}"
        print(f"    {win:<26}"
              f"{p['config']:<22}{int(p['n_test']):>5}{m:>+9.4f}")

    print(f"\n  OUT OF SAMPLE, {s['trades']} trades")
    print(f"    win rate            {s['win_rate']:.2%}")
    print(f"    mean net R          {s['mean_net_R']:+.4f}   t {s['t']:+.2f}")
    print(f"    selection stability {s['selection_stability']:.0%}   "
          f"({s['distinct_configs']} distinct configs chosen)")
    if "fixed_mean_net_R" in s:
        print(f"    fixed {FIXED}: {s['fixed_mean_net_R']:+.4f}")
        print(f"    selection edge      {s['selection_edge']:+.4f}"
              f"   <- does optimising help at all?")


if __name__ == "__main__":
    main()
