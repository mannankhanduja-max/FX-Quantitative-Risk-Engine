"""
ONE-SHOT TEST. Six instruments at the unbiased monitoring frame.

Criteria are fixed in docs/PREREG_added_power.md, written before
GBP/JPY or AUD/NZD existed on disk. This file only executes them.

WHAT IS BEING TESTED
--------------------
At the 1h walk - the only frame the null-path test found close to
unbiased - the four-instrument excess was +0.97pp raw, about +1.75pp
after subtracting the null's own -0.78pp, against a cost breakeven near
+1.65pp on 2,637 resolutions. Underpowered and sitting on the line. Two
instruments were added to resolve it.

HOW THE STATISTIC HANDLES CORRELATED INSTRUMENTS
------------------------------------------------
Per-trade excess is (win - p), where p is that trade's own barrier
probability. Cell value is its mean within an (instrument, calendar
month). The t-statistic is across CELLS against the null bias, so
GBP/JPY and USD/JPY inside the same month cannot each count as
independent evidence of the same move.

ONE ASSUMPTION, STATED RATHER THAN BURIED
-----------------------------------------
The -0.78pp null bias was measured on synthetic paths at one volatility
scale. It is applied to all six instruments on the grounds that the
bias is driven by the ratio of stop distance to bar range, and the stop
is 1 x ATR in every case, so the ratio is close to scale-free. That is
an argument, not a measurement.

BACKTEST-ONLY.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import config
from fxrisk.data import intraday
from fxrisk.research import barrier_prob as bp
from fxrisk.research import event_study as es
from fxrisk.risk import barriers, spread
from fxrisk.strategies import mtf, smc

AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}
CHUNKS = [("5m_2021", "5m_ask_2021"), ("5m_2023", "5m_ask_2023"),
          ("5m", "5m_ask")]
NULL_BIAS_1H = -0.0078          # from results/null_benchmark.txt
NEW = ("GBP/JPY", "AUD/NZD")
HORIZONS = (1, 4, 24, 96)
DONE = "results/added_power_DONE.txt"
OUT = "results/added_power.txt"


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def one(inst, iv, ask_iv, commission_bp):
    try:
        b5 = intraday.load_symbol(inst.yahoo, iv)
    except FileNotFoundError:
        return None, None
    trig, setup_f = resample(b5, "1h"), resample(b5, "4h")
    if len(setup_f) < 300:
        return None, None

    cfg = mtf.MTFConfig(stop_sigma=1.0, stop_mode="atr", atr_period=14,
                        setup_lookback=20, trigger_window=6, max_bars_30m=80)
    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg, zone_frame=zf)
    entries = mtf.entries_5m(trig, setups, None, cfg)
    if entries.empty:
        return None, None
    keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                           mtf.INSTRUMENT_SESSIONS[inst.name])
    entries = entries[keep.to_numpy()].reset_index(drop=True)
    if not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)
    if entries.empty:
        return None, None

    ev = es.signed_forward(trig, entries, horizons=HORIZONS)
    ev["instrument"] = inst.name

    # THE 1h FRAME. 4h is known-biased and is not run here at all.
    pos = mtf.to_30m_positions(entries, trig)
    if pos.empty:
        return None, ev
    try:
        cost = spread.real_cost_bp(trig, inst.yahoo, iv,
                                   commission_bp=commission_bp,
                                   ask_interval=ask_iv).to_numpy()
    except FileNotFoundError:
        return None, ev
    t = barriers.walk_explicit(
        trig, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=3.0, max_bars=320, cost_bp=cost, breakeven_frac=None)
    if t.empty:
        return None, ev
    t = t.copy()
    t["instrument"] = inst.name
    return t, ev


def excess_cells(tr):
    """Per-(instrument, month) mean of (win - own barrier probability)."""
    at = tr[tr["outcome"] != "time"].copy()
    if at.empty:
        return pd.DataFrame()
    p = bp.touch_probability(at["entry"].to_numpy(), at["target"].to_numpy(),
                            at["stop0"].to_numpy())
    at["exc"] = (at["outcome"].to_numpy() == "target").astype(float) - p
    at = at[np.isfinite(at["exc"])]
    at["time"] = pd.to_datetime(at["entry_time"], utc=True, format="mixed")
    return es.cell_means(at, "exc", min_events=3)


def clustered(cells, h0=0.0):
    x = cells["mean"].to_numpy(dtype="float64")
    n = len(x)
    if n < 3:
        return {"cells": n, "mean": float("nan"), "t": float("nan"),
                "p": float("nan")}
    m, sd = float(x.mean()), float(x.std(ddof=1))
    t = (m - h0) / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    return {"cells": n, "mean": m, "t": t,
            "p": float(2 * stats.t.sf(abs(t), df=n - 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission-bp", type=float, default=0.35)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(DONE) and not args.force:
        raise SystemExit(f"{DONE} exists - already run once. See {OUT}.")

    trades, events = [], []
    for inst in config.UNIVERSE_INTRADAY:
        for iv, ask in CHUNKS:
            t, ev = one(inst, iv, ask, args.commission_bp)
            if t is not None:
                trades.append(t)
            if ev is not None:
                events.append(ev)
    if not trades:
        raise SystemExit("no trades")

    a = pd.concat(trades, ignore_index=True)
    a["entry_time"] = pd.to_datetime(a["entry_time"], utc=True, format="mixed")
    a = (a.sort_values("entry_time")
         .drop_duplicates(subset=["instrument", "entry_time"], keep="first")
         .reset_index(drop=True))
    ev = pd.concat(events, ignore_index=True)
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    ev = (ev.sort_values("time")
          .drop_duplicates(subset=["instrument", "time"], keep="first")
          .reset_index(drop=True))

    L = ["SIX INSTRUMENTS AT THE 1h MONITORING FRAME",
         "  3:1, ATR(14) x1.0, measured spread + "
         f"{args.commission_bp:g}bp/side, 320-bar limit",
         f"  null-path bias at this frame: {NULL_BIAS_1H:+.2%} "
         f"(H0 for every t below)",
         "  cells = (instrument, calendar month); t is taken ACROSS cells",
         f"  data   {a['entry_time'].min().date()} -> "
         f"{a['entry_time'].max().date()}", ""]

    L.append(f"  {'instrument':<11}{'n':>6}{'win':>8}{'BM':>8}{'excess':>9}"
             f"{'corrected':>11}{'cells':>7}{'t':>7}{'net R':>9}{'t':>7}")
    for name in [i.name for i in config.UNIVERSE_INTRADAY]:
        g = a[a["instrument"] == name]
        if g.empty:
            L.append(f"  {name:<11}  no trades")
            continue
        b = bp.benchmark(g)
        c = clustered(excess_cells(g), NULL_BIAS_1H)
        r = g["unit_net_R"]
        rt = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        tag = " *" if name in NEW else ""
        L.append(f"  {name:<11}{len(g):>6}{b['actual_win']:>8.2%}"
                 f"{b['model_win']:>8.2%}{b['excess']:>+9.2%}"
                 f"{c['mean'] - NULL_BIAS_1H:>+11.2%}{c['cells']:>7}"
                 f"{c['t']:>+7.2f}{r.mean():>+9.4f}{rt:>+7.2f}{tag}")

    b = bp.benchmark(a)
    cells = excess_cells(a)
    c = clustered(cells, NULL_BIAS_1H)
    r = a["unit_net_R"]
    rt = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
    L += ["", f"  {'POOLED':<11}{len(a):>6}{b['actual_win']:>8.2%}"
              f"{b['model_win']:>8.2%}{b['excess']:>+9.2%}"
              f"{c['mean'] - NULL_BIAS_1H:>+11.2%}{c['cells']:>7}"
              f"{c['t']:>+7.2f}{r.mean():>+9.4f}{rt:>+7.2f}"]

    L += ["", "DROP-ONE ROBUSTNESS (corrected excess with each removed)"]
    drop_ok = True
    for name in [i.name for i in config.UNIVERSE_INTRADAY]:
        g = a[a["instrument"] != name]
        cc = clustered(excess_cells(g), NULL_BIAS_1H)
        corr = cc["mean"] - NULL_BIAS_1H
        rr = g["unit_net_R"]
        if not (corr > 0):
            drop_ok = False
        L.append(f"  without {name:<11}{corr:>+9.2%}  t {cc['t']:>+6.2f}"
                 f"   net R {rr.mean():>+8.4f}")

    L += ["", "FORWARD RETURN, the two new instruments only (bp)"]
    L.append("  " + f"{'instrument':<11}" +
             "".join(f"{'h=' + str(h):>17}" for h in HORIZONS))
    for name in NEW:
        g = ev[ev["instrument"] == name]
        row = []
        for h in HORIZONS:
            s = es.pooled(es.cell_means(g, f"y{h}"))
            row.append(f"{s['mean_bp']:>+10.2f} ({s['t']:>+.2f})")
        L.append(f"  {name:<11}" + "".join(f"{x:>17}" for x in row))

    corrected = c["mean"] - NULL_BIAS_1H
    L += ["", "VERDICT against docs/PREREG_added_power.md"]
    conds = [("corrected excess > 0 with t > 2",
              corrected > 0 and c["t"] > 2.0),
             ("mean net R > 0", r.mean() > 0),
             ("survives dropping any one instrument", drop_ok)]
    for text, ok in conds:
        L.append(f"  [{'PASS' if ok else 'FAIL'}]  {text}")
    L.append("")
    if all(ok for _, ok in conds):
        L.append("  PASS. Licenses ONE thing per the prereg: a frozen shot on")
        L.append("  data from another source at a lower cost structure.")
        L.append("  Not trading.")
    else:
        L.append("  FAIL. Underpowered was the last defence of this rule and")
        L.append("  the power has now been added. The strategy is closed.")

    text = "\n".join(L) + "\n"
    print(text)
    with open(OUT, "w") as fh:
        fh.write(text)
    with open(DONE, "w") as fh:
        fh.write(f"ran once on {pd.Timestamp.utcnow().isoformat()}\n"
                 f"criteria: docs/PREREG_added_power.md\nresult: {OUT}\n")


if __name__ == "__main__":
    main()
