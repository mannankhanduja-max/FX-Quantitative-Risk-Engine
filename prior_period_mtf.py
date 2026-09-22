"""
One-shot test of the liquidity-fade cascade on a period it has never seen.

    python fetch_intraday.py --interval 5m --years 2 --start 2021-09-01 --tag 2021
    python fetch_intraday.py --interval 5m --years 2 --start 2021-09-01 --tag 2021 --side ask

    python prior_period_mtf.py --verify    # reproduce the SEEN period first
    python prior_period_mtf.py             # then the unseen one, ONCE

WHY THIS EXISTS
----------------
Roughly 140 configurations have now been measured on Sep 2024 - Sep
2026. The best of them is frozen below. No split of that period is a
clean test any more: every held-out slice of it was already in view
when this configuration was selected, so the only honest test is data
from outside it entirely.

Sep 2021 - Sep 2023 is chosen for a reason stated before the run. It
contains the 2022 Nasdaq bear market (-33%), gold's 2022 drawdown,
and USD/JPY's run to 150 and reversal. The seen period was a broad
uptrend in three of these four instruments, and a separate test
already showed that this rule's apparent long-side advantage was the
trend rather than skill - the long-minus-short gap reversed sign in
down months, correlation +0.45 to the contemporaneous move. So a
falling market is exactly where this rule should be most exposed.

THE RULE IS FROZEN BELOW AND WAS WRITTEN DOWN BEFORE THE DATA WAS
FETCHED. Changing anything in FROZEN after seeing the result turns
this back into another in-sample number, and git history would show
it.

WHAT THE SEEN PERIOD GAVE
--------------------------
    trades                 2501
    win rate             13.24%
    no-information rate  11.11%   (2:1 geometry generalised to 8:1)
    excess               +2.13 points,  z +3.35
    mean net R, spread only          +0.1918
    mean net R, spread + 0.35bp      +0.0533   (t +0.87)
    Sharpe, spread only                +2.22
    quarters positive                    6/9
    mean net R excluding top 1%      -0.0263

The t-statistic on the with-commission mean is +0.87, not the +3.14
that belongs to the spread-only figure. After a realistic commission
this configuration is positive and NOT significant - the same verdict
every other candidate in this study received. What distinguishes it is
only that the sign is right.

WHAT COUNTS AS A PASS
----------------------
Both criteria, stated in advance:

  PRIMARY   excess over the no-information benchmark is POSITIVE
            with z > 2.0, pooled across the three instruments.

  ROBUST    mean net R with 0.35bp commission is POSITIVE after
            EXCLUDING THE TOP 1% OF TRADES BY RETURN.

The second criterion is the point of the exercise. At 8:1 the seen
period's entire profit sits in a handful of trades - removing the top
1% of 2501 trades takes +0.053R to -0.026R. A rule whose edge is that
concentrated can produce a flattering mean from luck alone, so the
test asks for an edge that survives without its best draws. The seen
period FAILS this criterion (-0.0263). It is included deliberately:
if the unseen period passes something the seen period could not, that
is evidence, and if it fails the same way, the concentration is a
property of the rule rather than of one sample.

A pass on PRIMARY alone is weak - one period, one draw. A fail on
PRIMARY is strong evidence against, because this configuration was
chosen as the most flattering thing 140 attempts could find.

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

# ============================================================
# THE FROZEN RULE - do not edit after the unseen run
# ============================================================
FROZEN = {
    "setups": ("fakeout", "retrace"),   # no breakouts: they trade with the move
    "confirm": "fvg",                   # order blocks measured at the coin flip
    "bias": None,                       # no session VWAP, no 9 EMA
    "instruments": ("EUR/USD", "NAS100", "USD/JPY"),   # gold: 1.62bp spread
    "rr": 8.0,
    "stop_sigma": 1.5,
    "max_bars_30m": 60,
    "setup_lookback": 20,
    "trigger_window": 6,
    "commission_bp": 0.35,              # per side, on top of measured spread
    "sessions": "per-instrument",       # mtf.INSTRUMENT_SESSIONS
    "blackout": True,                   # 16:55-18:00 ET rollover
}

SEEN = {
    "trades": 2501, "win": 0.1324, "excess": 0.0213, "z": 3.35,
    "net_spread_only": 0.1918, "net_with_commission": 0.0533,
    "sharpe": 2.22, "ex_top_1pct": -0.0263,
    # t on the with-commission mean is +0.87, NOT the +3.14 that belongs
    # to the zero-commission figure. The distinction matters: after a
    # realistic commission this configuration is positive but not
    # significant, like every other candidate in this study.
    "t_with_commission": 0.87, "t_spread_only": 3.14,
}

CRITERIA = (
    "PRIMARY  excess over the no-information benchmark > 0 with z > 2.0",
    "ROBUST   mean net R (0.35bp commission) > 0 excluding the top 1% of trades",
)

DONE = "results/prior_period_mtf_DONE.txt"
AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def run_one(inst, tag):
    iv = "5m" if not tag else f"5m_{tag}"
    b5 = intraday.load_symbol(inst.yahoo, iv)
    b30, cfg = resample(b5, "30min"), mtf.MTFConfig(
        stop_sigma=FROZEN["stop_sigma"],
        setup_lookback=FROZEN["setup_lookback"],
        trigger_window=FROZEN["trigger_window"],
        max_bars_30m=FROZEN["max_bars_30m"],
    )
    zf = smc.zones(b30, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(b30, cfg, kinds=FROZEN["setups"], zone_frame=zf)
    entries = mtf.entries_5m(b5, setups, FROZEN["bias"], cfg)
    if entries.empty:
        return None

    keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                           mtf.INSTRUMENT_SESSIONS[inst.name])
    entries = entries[keep.to_numpy()].reset_index(drop=True)
    if FROZEN["blackout"] and not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)

    pos = mtf.to_30m_positions(entries, b30)
    if pos.empty:
        return None

    ask_iv = iv.replace("5m", "5m") + "_ask" if not tag else f"5m_ask_{tag}"
    try:
        cost = spread.real_cost_bp(b30, inst.yahoo, iv,
                                   commission_bp=FROZEN["commission_bp"]).to_numpy()
    except FileNotFoundError:
        sys.exit(f"Missing the ask cache for {inst.name} ({ask_iv}).\n"
                 f"Run fetch_intraday.py with --side ask and the same --tag.")

    t = barriers.walk_explicit(
        b30, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=FROZEN["rr"], max_bars=FROZEN["max_bars_30m"],
        cost_bp=cost, breakeven_frac=None)
    t = t.copy()
    t["instrument"] = inst.name
    return t


def report(trades, label):
    r = trades["unit_net_R"]
    b = bp.benchmark(trades)
    n = len(trades)
    cut = max(1, int(round(n * 0.01)))
    ex = r.sort_values(ascending=False).iloc[cut:]

    print(f"\n{label}")
    print(f"  window        {trades['entry_time'].min().date()} -> "
          f"{trades['entry_time'].max().date()}")
    print(f"  trades        {n}")
    print(f"  win rate      {b['actual_win']:.2%}   "
          f"no-information {b['model_win']:.2%}")
    print(f"  excess        {b['excess']:+.2%}   z {b['z']:+.2f}")
    print(f"  mean net R    {r.mean():+.4f}   "
          f"t {r.mean()/(r.std(ddof=1)/np.sqrt(n)):+.2f}")
    print(f"  excluding top {cut} trades ({cut/n:.1%}):  {ex.mean():+.4f}")
    print("\n  by instrument")
    for i, g in trades.groupby("instrument"):
        print(f"    {i:<9} n={len(g):<5} win {(g['unit_net_R']>0).mean():6.2%}"
              f"   net R {g['unit_net_R'].mean():+.4f}")

    primary = (b["excess"] > 0) and (b["z"] > 2.0)
    robust = ex.mean() > 0
    print("\n  PRE-REGISTERED CRITERIA")
    print(f"    PRIMARY  excess > 0 and z > 2.0      "
          f"{'PASS' if primary else 'FAIL'}"
          f"   ({b['excess']:+.2%}, z {b['z']:+.2f})")
    print(f"    ROBUST   net R > 0 ex-top-1%         "
          f"{'PASS' if robust else 'FAIL'}   ({ex.mean():+.4f})")
    return primary, robust


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the SEEN period only; touches nothing new")
    ap.add_argument("--tag", default="2021",
                    help="cache tag of the unseen window")
    args = ap.parse_args()

    print("=" * 66)
    print("LIQUIDITY-FADE CASCADE  -  PRE-REGISTERED ONE-SHOT TEST")
    print("=" * 66)
    for k, v in FROZEN.items():
        print(f"  {k:<16} {v}")
    print("\n  criteria, fixed before the run:")
    for c in CRITERIA:
        print(f"    {c}")

    if not args.verify and os.path.exists(DONE):
        sys.exit(f"\nAlready run. The result is in {DONE}.\n"
                 "A one-shot test run twice is not a one-shot test - if the\n"
                 "rule needs changing, that is a NEW pre-registration on a\n"
                 "NEW period, not a re-run of this one.")

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
              f"net {SEEN['net_with_commission']:+.4f}, "
              f"ex-top-1% {SEEN['ex_top_1pct']:+.4f})")
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
    trades.to_csv("results/prior_period_mtf_trades.csv", index=False)

    print(f"\n  recorded in {DONE}")
    if primary and robust:
        print("  Both criteria met. One period, one draw - but it is the "
              "first\n  thing in this study that has survived a test it could "
              "have failed.")
    else:
        print("  The rule does not replicate. That is the answer, and it was\n"
              "  the more likely one going in: this configuration was the best\n"
              "  of roughly 140 measured on a single sample.")


if __name__ == "__main__":
    main()
