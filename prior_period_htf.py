"""
One-shot test of the HIGHER-TIMEFRAME cascade.

    python prior_period_htf.py --verify    # reproduce the seen period
    python prior_period_htf.py             # the unseen one, ONCE

WHAT IS BEING TESTED, AND WHY IT IS NOT JUST ANOTHER VARIANT
--------------------------------------------------------------
Every negative result in this study reduced to one ratio:

    cost_R = 2 * spread / stop_distance

On 5-minute triggers the stop is 6.8bp and the round trip 1.1bp, so
21% of the risk went to friction against a 12.8% edge. Widening the
stop WITHIN that cascade did not help - the edge decayed at the same
rate the cost did.

Moving the whole cascade up is different in kind, and the reason was
stated before it was measured: volatility scales with the square root
of time, the spread does not scale at all, so the stop grows and
cost_R shrinks for free. The open question was whether the edge
survives the move up.

On the development period it did. The gross edge is flat across a
6.5x change in stop size while cost falls 6x:

    ladder        n     stop bp    win      z     gross   cost     net
    5m/30m/1h   8220      6.8    28.19%  +6.68  +0.1284 0.2086  -0.0802
    15m/1h/4h   3864     11.7    27.20%  +3.16  +0.0904 0.1192  -0.0288
    1h/4h/1d    1051     22.5    28.60%  +2.69  +0.1456 0.0613  +0.0843
    4h/1d/1W     182     44.2    28.57%  +1.09  +0.1429 0.0338  +0.1090

This test asks whether the 1h/4h/1d row holds on data it was not
built on.

AN HONEST WEAKENING OF THIS TEST
----------------------------------
Sep 2021 - Aug 2023 has been used ONCE already, for the 5m 8:1
configuration, which failed. This is therefore a SECOND look at the
same window, and a second look is worth less than a first: the window
is partly spent, and knowing that the 5m rule failed there is itself
information that shaped the decision to come back.

It is still the best test available, because the mechanism being
tested - cost scaling with the stop while the edge does not - is
structural and was written down before any of this was run. But the
result should be read as corroboration or refutation, not as a clean
first-look verdict. A genuinely clean test needs a third window.

THE RULE IS FROZEN BELOW.

WHAT COUNTS AS A PASS
----------------------
  PRIMARY  pooled excess over the no-information benchmark > 0 with
           z > 2.0

  ROBUST   at least 3 of the 4 instruments have positive net R, AND
           both halves of the period have positive net R

The ex-top-N% robustness test used for the 8:1 freeze is deliberately
NOT reused. At a 13% win rate with 8R payoffs, removing the top 1%
removed a handful of freak time exits and was diagnostic. At a 28%
win rate with 3R payoffs, removing the top 5% removes 50 ordinary
winners worth 150R against a total of 88R - it would condemn any
3:1 rule, including a perfectly good one. The wrong instrument for
this geometry.

The seen period FAILS ROBUST: its first half is -0.0033 against
+0.1717 for the second. That is deliberate - a criterion the
development sample already passes cannot separate a real effect from
a lucky one.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config
from fxrisk.data import intraday
from fxrisk.research import barrier_prob as bp
from fxrisk.risk import barriers, spread
from fxrisk.strategies import mtf, smc

FROZEN = {
    "ladder": ("1h", "4h", "1D"),      # trigger / setup / context
    "setups": ("breakout", "fakeout", "retrace"),
    "confirm": "fvg",
    "bias": None,                      # no VWAP, no EMA
    "instruments": ("EUR/USD", "XAU/USD", "NAS100", "USD/JPY"),
    "rr": 3.0,
    "stop_mode": "atr",
    "atr_period": 14,
    "stop_sigma": 1.0,
    "max_bars": 80,
    "setup_lookback": 20,
    "trigger_window": 6,
    "commission_bp": 0.35,
    "sessions": "per-instrument",
    "blackout": True,
}

SEEN = {
    "trades": 1051, "win": 0.2860, "model": 0.2501, "z": 2.69,
    "gross": 0.1456, "cost": 0.0613, "net": 0.0843, "t": 1.51,
    "barrier": 0.0826,
    "first_half": -0.0033, "second_half": 0.1717,
    "instruments_positive": 3,
}

DONE = "results/prior_period_htf_DONE.txt"
AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def run_one(inst, tag):
    iv = "5m" if not tag else f"5m_{tag}"
    ask_iv = "5m_ask" if not tag else f"5m_ask_{tag}"
    b5 = intraday.load_symbol(inst.yahoo, iv)

    lo, mid, _hi = FROZEN["ladder"]
    trig, setup_f = resample(b5, lo), resample(b5, mid)

    cfg = mtf.MTFConfig(stop_sigma=FROZEN["stop_sigma"],
                        stop_mode=FROZEN["stop_mode"],
                        atr_period=FROZEN["atr_period"],
                        setup_lookback=FROZEN["setup_lookback"],
                        trigger_window=FROZEN["trigger_window"],
                        max_bars_30m=FROZEN["max_bars"])

    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg, kinds=FROZEN["setups"], zone_frame=zf)
    entries = mtf.entries_5m(trig, setups, FROZEN["bias"], cfg)
    if entries.empty:
        return None

    keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                           mtf.INSTRUMENT_SESSIONS[inst.name])
    entries = entries[keep.to_numpy()].reset_index(drop=True)
    if FROZEN["blackout"] and not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)
    if entries.empty:
        return None

    pos = mtf.to_30m_positions(entries, setup_f)
    if pos.empty:
        return None

    try:
        cost = spread.real_cost_bp(setup_f, inst.yahoo, iv,
                                   commission_bp=FROZEN["commission_bp"],
                                   ask_interval=ask_iv).to_numpy()
    except FileNotFoundError:
        sys.exit(f"Missing the ask cache for {inst.name} ({ask_iv}).")

    t = barriers.walk_explicit(
        setup_f, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=FROZEN["rr"], max_bars=FROZEN["max_bars"],
        cost_bp=cost, breakeven_frac=None)
    if t.empty:
        return None
    t = t.copy()
    t["instrument"] = inst.name
    return t


def report(trades, label):
    r = trades["unit_net_R"]
    b = bp.benchmark(trades)
    n = len(trades)
    tm = trades["outcome"] == "time"
    h = n // 2
    first, second = r.iloc[:h].mean(), r.iloc[h:].mean()
    per_inst = trades.groupby("instrument")["unit_net_R"].mean()
    n_pos = int((per_inst > 0).sum())

    print(f"\n{label}")
    print(f"  window       {trades['entry_time'].min().date()} -> "
          f"{trades['entry_time'].max().date()}")
    print(f"  trades       {n}   time exits {tm.mean():.1%}")
    print(f"  win rate     {b['actual_win']:.2%}   "
          f"no-information {b['model_win']:.2%}")
    print(f"  excess       {b['excess']:+.2%}   z {b['z']:+.2f}")
    print(f"  gross        {trades['gross_R'].mean():+.4f}   "
          f"cost {trades['cost_R'].mean():.4f}   net {r.mean():+.4f}   "
          f"t {r.mean()/(r.std(ddof=1)/np.sqrt(n)):+.2f}")
    print(f"  barrier-only {r[~tm].mean():+.4f}")
    print(f"  halves       first {first:+.4f}   second {second:+.4f}")
    print("\n  by instrument")
    for i, g in trades.groupby("instrument"):
        gb = bp.benchmark(g)
        print(f"    {i:<9} n={len(g):<5} win {gb['actual_win']:6.2%}  "
              f"z {gb['z']:+5.2f}  net {g['unit_net_R'].mean():+.4f}")

    primary = (b["excess"] > 0) and (b["z"] > 2.0)
    robust = (n_pos >= 3) and (first > 0) and (second > 0)
    print("\n  PRE-REGISTERED CRITERIA")
    print(f"    PRIMARY  excess > 0 and z > 2.0            "
          f"{'PASS' if primary else 'FAIL'}   "
          f"({b['excess']:+.2%}, z {b['z']:+.2f})")
    print(f"    ROBUST   >=3 of 4 instruments and both      "
          f"{'PASS' if robust else 'FAIL'}   "
          f"({n_pos}/4 instruments, halves "
          f"{'+' if first > 0 else '-'}{'+' if second > 0 else '-'})")
    return primary, robust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default="2021")
    args = ap.parse_args()

    print("=" * 66)
    print("HIGHER-TIMEFRAME CASCADE  -  PRE-REGISTERED ONE-SHOT TEST")
    print("=" * 66)
    for k, v in FROZEN.items():
        print(f"  {k:<16} {v}")

    if not args.verify and os.path.exists(DONE):
        sys.exit(f"\nAlready run. Result in {DONE}.\n"
                 "A one-shot test run twice is not a one-shot test.")

    tag = "" if args.verify else args.tag
    label = ("SEEN period, re-run (should match the study)" if args.verify
             else "UNSEEN period - the one-shot test")

    parts = []
    for inst in config.UNIVERSE_INTRADAY:
        if inst.name not in FROZEN["instruments"]:
            continue
        t = run_one(inst, tag)
        if t is not None and not t.empty:
            parts.append(t)
    if not parts:
        sys.exit("No trades. Check the cache tag.")

    trades = pd.concat(parts, ignore_index=True).sort_values("entry_time")
    primary, robust = report(trades, label)

    if args.verify:
        print(f"\n  (study recorded: {SEEN['trades']} trades, "
              f"win {SEEN['win']:.2%}, z {SEEN['z']:+.2f}, "
              f"net {SEEN['net']:+.4f}, halves "
              f"{SEEN['first_half']:+.4f}/{SEEN['second_half']:+.4f})")
        return

    os.makedirs("results", exist_ok=True)
    with open(DONE, "w") as fh:
        fh.write(f"run: {pd.Timestamp.utcnow().isoformat()}\n")
        fh.write(f"window: {trades['entry_time'].min()} -> "
                 f"{trades['entry_time'].max()}\n")
        fh.write(f"trades: {len(trades)}\n")
        fh.write(f"PRIMARY: {'PASS' if primary else 'FAIL'}\n")
        fh.write(f"ROBUST:  {'PASS' if robust else 'FAIL'}\n")
        for k, v in FROZEN.items():
            fh.write(f"frozen.{k}: {v}\n")
    trades.to_csv("results/prior_period_htf_trades.csv", index=False)
    print(f"\n  recorded in {DONE}")


if __name__ == "__main__":
    main()
