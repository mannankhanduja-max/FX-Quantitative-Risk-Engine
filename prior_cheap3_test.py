"""
ONE-SHOT TEST. The cost-filtered rule on 2016-2021, which it has never seen.

Criteria are fixed in docs/PREREG_prior_period_cheap3.md, and this file is
committed together with it BEFORE the data exists, so neither the bar nor
the arithmetic can move after a number appears.

The whole test in one line: the frozen cascade, monitored at 1h, on the
three instruments with the lowest measured 2*spread/stop, over the five
years immediately before the window everything else here used.

    universe   NAS100, XAU/USD, USD/JPY   (frozen as a LIST, not re-ranked
                                           on the test window)
    window     2016-09-01 -> 2021-08-30   (no overlap with any cache used)
    frame      1h monitoring only          (4h is known-biased)
    in-sample  corrected excess +2.52% (t +2.25), net R +0.0100 (t +0.25)

NET R IS NOT TESTED FOR SIGNIFICANCE. Per-trade sd is 1.77R, so t = 2 on
a +0.0100R mean needs ~124,864 trades, about 320 years. A criterion
nothing can clear is a rigged criterion. Net R is reported as a sign only.

BACKTEST-ONLY.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config
from added_power_test import NULL_BIAS_1H, clustered, excess_cells, resample
from fxrisk.data import intraday
from fxrisk.research import barrier_prob as bp
from fxrisk.research import event_study as es
from fxrisk.risk import barriers, spread
from fxrisk.strategies import mtf, smc

# Frozen. Not re-ranked on the test window - see the prereg.
UNIVERSE = ("NAS100", "XAU/USD", "USD/JPY")
CHUNKS = [("5m_2016", "5m_ask_2016"), ("5m_2018", "5m_ask_2018")]
MIN_YEARS = 3.0
BREAKEVEN_PP = 0.0150          # mean cost_R 0.060 / 0.04 R per pp
MIN_DROP_ONE_PP = 0.0100
HORIZONS = (1, 4, 24, 96)
DONE = "results/prior_cheap3_DONE.txt"
OUT = "results/prior_cheap3.txt"


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


def measured_cost_r(inst, iv, ask_iv):
    """Reported for comparison only. Never used to re-rank the universe."""
    try:
        s = spread.real_spread(inst.yahoo, iv, ask_interval=ask_iv)
        b5 = intraday.load_symbol(inst.yahoo, iv)
    except FileNotFoundError:
        return float("nan")
    half = float(np.median(s)) * 1e4 / 2.0
    trig = resample(b5, "1h")
    stop_bp = float((mtf.atr(trig, 14) / trig["Close"]).median() * 1e4)
    return 2.0 * (half + 0.35) / stop_bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission-bp", type=float, default=0.35)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.exists(DONE) and not args.force:
        raise SystemExit(f"{DONE} exists - this test has already been run "
                         f"once. Result: {OUT}. Re-running a one-shot test "
                         f"until it reads well is the failure the marker "
                         f"exists to prevent.")

    insts = [i for i in config.UNIVERSE_INTRADAY if i.name in UNIVERSE]
    if len(insts) != len(UNIVERSE):
        raise SystemExit(f"universe mismatch: wanted {UNIVERSE}")

    trades, events, costs, excluded = [], [], {}, []
    for inst in insts:
        got = []
        for iv, ask in CHUNKS:
            t, ev = one(inst, iv, ask, args.commission_bp)
            if t is not None:
                got.append(t)
            if ev is not None:
                events.append(ev)
        if not got:
            excluded.append((inst.name, "no data"))
            continue
        g = pd.concat(got, ignore_index=True)
        g["entry_time"] = pd.to_datetime(g["entry_time"], utc=True,
                                         format="mixed")
        span = ((g["entry_time"].max() - g["entry_time"].min()).days / 365.25)
        if span < MIN_YEARS:
            excluded.append((inst.name, f"only {span:.1f}y of history"))
            continue
        trades.append(g)
        costs[inst.name] = measured_cost_r(inst, *CHUNKS[0])

    if not trades:
        raise SystemExit("no instrument cleared the history requirement")

    a = (pd.concat(trades, ignore_index=True).sort_values("entry_time")
         .drop_duplicates(subset=["instrument", "entry_time"], keep="first")
         .reset_index(drop=True))
    ev = pd.concat(events, ignore_index=True)
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    ev = (ev.sort_values("time")
          .drop_duplicates(subset=["instrument", "time"], keep="first")
          .reset_index(drop=True))

    L = ["FROZEN SHOT: cost-filtered rule on 2016-2021, never seen before",
         "  universe frozen as NAS100 + XAU/USD + USD/JPY, 1h monitoring",
         "  3:1, ATR(14) x1.0, measured spread + "
         f"{args.commission_bp:g}bp/side, 320-bar limit",
         f"  H0 for every t is the null-path bias {NULL_BIAS_1H:+.2%}",
         f"  data   {a['entry_time'].min().date()} -> "
         f"{a['entry_time'].max().date()}",
         "  in sample this was corrected +2.52% (t +2.25), net R +0.0100", ""]
    if excluded:
        for name, why in excluded:
            L.append(f"  EXCLUDED  {name}: {why}")
        L.append("")

    L.append("  cost_R re-measured on THIS window (reported, not used to "
             "re-rank):")
    L.append("    " + "  ".join(f"{k} {v:.3f}" for k, v in costs.items()))
    L.append("")
    L.append(f"  {'instrument':<11}{'n':>6}{'win':>8}{'BM':>8}{'excess':>9}"
             f"{'corrected':>11}{'cells':>7}{'t':>7}{'net R':>9}")
    pos_insts = 0
    for name in [i.name for i in insts]:
        g = a[a["instrument"] == name]
        if g.empty:
            continue
        b = bp.benchmark(g)
        c = clustered(excess_cells(g), NULL_BIAS_1H)
        corr = c["mean"] - NULL_BIAS_1H
        if corr > 0:
            pos_insts += 1
        L.append(f"  {name:<11}{len(g):>6}{b['actual_win']:>8.2%}"
                 f"{b['model_win']:>8.2%}{b['excess']:>+9.2%}"
                 f"{corr:>+11.2%}{c['cells']:>7}{c['t']:>+7.2f}"
                 f"{g['unit_net_R'].mean():>+9.4f}")

    b = bp.benchmark(a)
    c = clustered(excess_cells(a), NULL_BIAS_1H)
    corrected = c["mean"] - NULL_BIAS_1H
    net = float(a["unit_net_R"].mean())
    L += ["", f"  {'POOLED':<11}{len(a):>6}{b['actual_win']:>8.2%}"
              f"{b['model_win']:>8.2%}{b['excess']:>+9.2%}"
              f"{corrected:>+11.2%}{c['cells']:>7}{c['t']:>+7.2f}"
              f"{net:>+9.4f}"]

    L += ["", "DROP-ONE (pooled corrected excess with each removed)"]
    drop_min = None
    for name in [i.name for i in insts]:
        g = a[a["instrument"] != name]
        if g.empty:
            continue
        cc = clustered(excess_cells(g), NULL_BIAS_1H)
        v = cc["mean"] - NULL_BIAS_1H
        drop_min = v if drop_min is None else min(drop_min, v)
        L.append(f"  without {name:<11}{v:>+9.2%}  t {cc['t']:>+6.2f}"
                 f"   net R {g['unit_net_R'].mean():>+8.4f}")

    L += ["", "FORWARD RETURN on this window (bp), for continuity"]
    L.append("  " + f"{'instrument':<11}" +
             "".join(f"{'h=' + str(h):>17}" for h in HORIZONS))
    for name in [i.name for i in insts]:
        g = ev[ev["instrument"] == name]
        if g.empty:
            continue
        row = [f"{es.pooled(es.cell_means(g, f'y{h}'))['mean_bp']:>+10.2f}"
               f" ({es.pooled(es.cell_means(g, f'y{h}'))['t']:>+.2f})"
               for h in HORIZONS]
        L.append(f"  {name:<11}" + "".join(f"{x:>17}" for x in row))

    conds = [
        (f"corrected excess > 0 with t > 2.0",
         corrected > 0 and c["t"] > 2.0),
        (f"corrected excess >= {BREAKEVEN_PP:.2%} (cost breakeven)",
         corrected >= BREAKEVEN_PP),
        ("corrected excess positive in >= 2 of 3 instruments",
         pos_insts >= 2),
        (f"drop-one leaves pooled >= {MIN_DROP_ONE_PP:.2%}",
         drop_min is not None and drop_min >= MIN_DROP_ONE_PP),
    ]
    L += ["", "VERDICT against docs/PREREG_prior_period_cheap3.md"]
    for text, ok in conds:
        L.append(f"  [{'PASS' if ok else 'FAIL'}]  {text}")
    L.append(f"  [ sign ]  net R {net:+.4f}  "
             f"({'positive' if net > 0 else 'not positive'}; "
             f"no significance claimed - t = 2 would need ~125,000 trades)")
    L.append("")
    if all(ok for _, ok in conds) and net > 0:
        L += ["  PASS. Per the prereg this licenses ONE conversation: whether",
              "  a cost structure below retail makes a ~+0.01R edge matter.",
              "  It licenses no trading and no capital. The best estimate is",
              "  still indistinguishable from zero at any feasible sample."]
    else:
        L += ["  FAIL. This was the last configuration with an ex-ante",
              "  rationale, on the last unused data, at the only unbiased",
              "  frame. Per the prereg, the line closes here and a FAIL is",
              "  not to be followed by looking for another cut."]

    text = "\n".join(L) + "\n"
    print(text)
    with open(OUT, "w") as fh:
        fh.write(text)
    with open(DONE, "w") as fh:
        fh.write(f"ran once on {pd.Timestamp.utcnow().isoformat()}\n"
                 f"criteria: docs/PREREG_prior_period_cheap3.md\n"
                 f"result:   {OUT}\n")


if __name__ == "__main__":
    main()
