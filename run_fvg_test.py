"""
ONE-SHOT TEST. The FVG-confirmed breakout/retracement rule on
2022-01-01 -> now, which it has never seen.

Criteria are fixed in `PREREGISTRATION.md`, committed before this
script reads a single test-period bar. This script enforces that
order mechanically, not just by convention:

    1. refuses if PREREGISTRATION.md still contains <<SIGN HERE>>
    2. refuses if PREREGISTRATION.md has an uncommitted diff
    3. refuses if results/fvg_test_DONE.txt already exists (this test
       has already run once)

Pass `--force` to bypass gate 3 only, and only if you mean it - that
marker exists to prevent "running it again until it reads well,"
which is the one thing a one-shot test exists to rule out. There is
no flag that bypasses gates 1 or 2; those are about the file, not
this script's mood.

Uses the exact same rule (`fxrisk.strategies.fvg_retrace`) and cost
measurement as `run_fvg_design.py` - same CLI flags, same defaults -
so the configuration run here is read off PREREGISTRATION.md's own
description of what was carried forward, not re-typed from memory.

BACKTEST-ONLY. Not a recommendation to trade.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd

from fxrisk.data import fvg_raw
from fxrisk.risk import barriers
from fxrisk.strategies import fvg_retrace as fr
from fxrisk.strategies.smc import SMCConfig
from run_fvg_design import run_config, section

PREREG = "PREREGISTRATION.md"
SIGN_MARKER = "<<SIGN HERE>>"
DONE = "results/fvg_test_DONE.txt"
OUT = "results/fvg_test.txt"


def _check_preregistration_signed() -> None:
    if not os.path.exists(PREREG):
        raise SystemExit(f"{PREREG} does not exist - write and sign it before "
                         f"running this test. See PREREGISTRATION.md's "
                         f"template for the required sections.")
    with open(PREREG) as fh:
        text = fh.read()
    if SIGN_MARKER in text:
        raise SystemExit(f"{PREREG} still contains {SIGN_MARKER!r} - it has "
                         f"not been signed. Fill in every <<...>> placeholder "
                         f"and replace the sign-here line before running this "
                         f"test.")

    diff = subprocess.run(["git", "diff", "--", PREREG], capture_output=True,
                          text=True)
    status = subprocess.run(["git", "status", "--porcelain", "--", PREREG],
                            capture_output=True, text=True)
    if diff.stdout.strip() or status.stdout.strip():
        raise SystemExit(f"{PREREG} has uncommitted changes - commit it "
                         f"before running this test, so the criteria are "
                         f"frozen in git history before the result exists.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=fvg_raw.DEFAULT_RAW_DIR)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--retrace-max-bars", type=int, default=12)
    ap.add_argument("--min-fraction", type=float, default=0.33)
    ap.add_argument("--max-fraction", type=float, default=1.0)
    ap.add_argument("--stop-min-sigma", type=float, default=1.0)
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--max-bars", type=int, default=240)
    ap.add_argument("--be-bp", type=float, default=None)
    ap.add_argument("--commission-bp", type=float, default=0.0)
    ap.add_argument("--force", action="store_true",
                    help="bypass the already-ran-once marker ONLY")
    args = ap.parse_args()

    _check_preregistration_signed()

    if os.path.exists(DONE) and not args.force:
        raise SystemExit(f"{DONE} exists - this test has already been run "
                         f"once. Result: {OUT}. Re-running a one-shot test "
                         f"until it reads well is the failure the marker "
                         f"exists to prevent.")

    bid = fvg_raw.load_test(args.raw_dir)
    ask = fvg_raw.load_side("ask", fvg_raw.TEST_START, pd.Timestamp.now(tz="UTC"),
                            args.raw_dir)

    cfg = fr.FVGRetraceConfig(
        lookback=args.lookback, retrace_max_bars=args.retrace_max_bars,
        min_fraction=args.min_fraction, max_fraction=args.max_fraction,
        stop_min_sigma=args.stop_min_sigma, use_trend_filter=not args.no_trend,
        smc=SMCConfig(),
    )
    setups, trades = run_config(bid, ask, cfg, args.rr, args.max_bars,
                                args.be_bp, args.commission_bp)

    lines = ["FROZEN SHOT: FVG-confirmed breakout/retracement, 2022 -> now, "
            "never seen before",
            f"  rule: lookback {args.lookback}, retrace "
            f"{args.min_fraction:.0%}-{args.max_fraction:.0%}, stop floor "
            f"{args.stop_min_sigma:g} sigma, trend filter "
            f"{'off' if args.no_trend else 'on'}, {args.rr:g}:1"
            + (f", breakeven {args.be_bp:g}bp" if args.be_bp else ""),
            f"  data  {bid.index[0]} -> {bid.index[-1]} ({len(bid):,} bars)"
            if not bid.empty else "  data  (no test-period bars loaded)",
            f"  setups  {len(setups)}", ""]

    if trades.empty:
        lines.append("  NO TRADES - the rule produced no completed trades in "
                     "the test window. Reported as a null result, per the "
                     "prereg, not retried with different parameters.")
    else:
        s = barriers.summarise_explicit(trades, args.rr)
        lines.append(f"  trades {s['trades']}  win rate (barrier) "
                     f"{s['win_rate_barrier']:.2%}  cost-adj. hurdle "
                     f"{s['breakeven_wr']:.2%}  gap {s['gap_vs_breakeven']:+.2%}"
                     f"  mean net R {s['mean_net_R']:+.4f}  total net R "
                     f"{s['total_net_R']:+.2f}")
        r = trades["net_R"].to_numpy(dtype="float64")
        if len(r) > 5 and r.std(ddof=1) > 0:
            import numpy as np
            t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
            lines.append(f"  t {t:+.2f}")

    lines += ["", section("Test period", trades, args.rr),
             "VERDICT: read this result against the criteria in "
             "PREREGISTRATION.md by hand. This script reports the numbers "
             "the prereg's criteria are defined over; it does not grade "
             "them against the prereg automatically, because the prereg's "
             "own template leaves the exact pass/fail statement to be "
             "written by whoever signs it.",
             "",
             "BACKTEST-ONLY. Nothing here is a recommendation to trade."]

    text = "\n".join(lines) + "\n"
    print(text)

    os.makedirs("results", exist_ok=True)
    trades.to_csv("results/fvg_test_trades.csv", index=False)
    with open(OUT, "w") as fh:
        fh.write(text)
    with open(DONE, "w") as fh:
        fh.write(f"ran once on {pd.Timestamp.utcnow().isoformat()}\n"
                 f"criteria: {PREREG}\n"
                 f"result:   {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
