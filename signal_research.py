"""
Signal-first research: measure information before designing a trade.

    python signal_research.py discover     # 2022-09 -> 2024-09, all 24 tests
    python signal_research.py confirm      # 2024-09 -> 2026-09, survivors only, ONCE

THE PROTOCOL
-------------
Written down, and committed, before either stage was run.

  CANDIDATES   the eight features in fxrisk/research/signals.py,
               at horizons of 1, 4 and 24 hours on 1-hour bars.
               Twenty-four tests. Nothing is added after the fact.

  DISCOVERY    Sep 2022 - Sep 2024. For each test, the mean monthly
               rank IC across (instrument, month) cells, its t across
               cells, and Holm-Bonferroni at 5% over all twenty-four.
               A test SURVIVES only if Holm rejects its null.

  CONFIRMATION Sep 2024 - Sep 2026, survivors only, run ONCE. The
               script writes a marker and refuses a second run. A
               survivor CONFIRMS if its confirmation mean IC has the
               same sign as in discovery and two-sided p < 0.05.

  TRADEABLE    A confirmed signal is worth building a rule around
               only if its quintile edge exceeds the 2bp round trip
               in BOTH blocks. Real-but-too-small is a finding, not a
               strategy.

Both periods are fresh with respect to these features: the earlier
work in this repository tested entry RULES on them, never these
signals. That is a narrower claim than "unseen data" and is stated
as such.

The discovery block's last 24 bars are dropped, so no forward return
measured in discovery reaches into the confirmation period.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from fxrisk.data.intraday import load_symbol  # noqa: E402
from fxrisk.research import ic as icm  # noqa: E402
from fxrisk.research import signals as sg  # noqa: E402

SPLIT = pd.Timestamp("2024-09-19", tz="America/New_York")
ALPHA = 0.05
ROUND_TRIP_BP = 2.0 * config.BREAKOUT_COST_BP

SURVIVORS = "results/signal_survivors.json"
CONFIRM_MARK = "results/signal_confirm_DONE.txt"


def load_block(which: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Features and forward returns per instrument, sliced to one block."""
    hmax = max(sg.HORIZONS)
    out = {}
    for inst in config.UNIVERSE_INTRADAY:
        bars = load_symbol(inst.yahoo, "1h", "data/intraday")
        # Features over the full history: they are causal, so a value
        # at t uses only bars <= t, and early bars in a block may use
        # warm-up from before it without seeing anything after t.
        feats = sg.compute(bars)
        fwd = sg.forward_returns(bars)
        if which == "discover":
            keep = bars.index < SPLIT
            idx = bars.index[keep][:-hmax]       # no fwd return crosses SPLIT
        else:
            idx = bars.index[bars.index >= SPLIT]
        out[inst.name] = (feats.loc[idx], fwd.loc[idx])
    return out


def run(block, tests: list[tuple[str, int]]) -> list[icm.ICResult]:
    results = []
    for feat, h in tests:
        cells = []
        for name, (f, r) in block.items():
            c = icm.cell_ics(f[feat], r[f"fwd_{h}"], name)
            if not c.empty:
                cells.append(c)
        pooled = pd.concat(cells, ignore_index=True) if cells else pd.DataFrame()
        results.append(icm.summarise(pooled, feat, h))
    return results


def table(results, reject=None) -> str:
    lines = [f"  {'feature':11s} {'h':>3s} {'cells':>5s} {'mean IC':>9s} "
             f"{'t':>6s} {'p':>8s} {'hit':>5s} {'edge bp':>8s}"]
    for i, r in enumerate(results):
        flag = ""
        if reject is not None:
            flag = "  SURVIVES" if reject[i] else ""
        lines.append(
            f"  {r.feature:11s} {r.horizon:3d} {r.cells:5d} {r.mean_ic:+9.4f} "
            f"{r.t:+6.2f} {r.p:8.4f} {r.hit:5.0%} {r.edge_bp:+8.2f}{flag}")
    return "\n".join(lines)


def discover() -> int:
    tests = [(f, h) for f in sg.FEATURES for h in sg.HORIZONS]
    block = load_block("discover")
    print("DISCOVERY  Sep 2022 - Sep 2024, 1h bars, all 24 tests\n")
    res = run(block, tests)
    rej = icm.holm([r.p for r in res], ALPHA)
    print(table(res, rej))

    surv = [{"feature": r.feature, "horizon": r.horizon, "mean_ic": r.mean_ic,
             "t": r.t, "edge_bp": r.edge_bp}
            for r, k in zip(res, rej) if k]
    naive = sum(1 for r in res if np.isfinite(r.p) and r.p < ALPHA)
    print(f"\n  Naive p < {ALPHA}: {naive} of {len(res)}  "
          f"(about {len(res) * ALPHA:.1f} expected from noise alone)")
    print(f"  Holm survivors:   {len(surv)}")

    os.makedirs("results", exist_ok=True)
    with open(SURVIVORS, "w") as fh:
        json.dump({"alpha": ALPHA, "survivors": surv}, fh, indent=2)
    print(f"\n  Survivors written to {SURVIVORS}.")
    if surv:
        print("  Next:  python signal_research.py confirm   (runs ONCE)")
    else:
        print("  Nothing to confirm. The confirmation block stays unused.")
    return 0


def confirm(force: bool) -> int:
    if os.path.exists(CONFIRM_MARK) and not force:
        print(f"Confirmation has already been run ({CONFIRM_MARK}).")
        print("It is a one-shot test. Re-running it after seeing the result")
        print("would turn it back into a search. Refusing.")
        return 1
    if not os.path.exists(SURVIVORS):
        print("Run discovery first.")
        return 1

    surv = json.load(open(SURVIVORS))["survivors"]
    if not surv:
        print("Discovery produced no survivors; nothing to confirm.")
        return 0

    tests = [(s["feature"], s["horizon"]) for s in surv]
    block = load_block("confirm")
    print("CONFIRMATION  Sep 2024 - Sep 2026, survivors only, one run\n")
    res = run(block, tests)
    print(table(res))

    print("\nVERDICT  (criteria fixed in the docstring before the run)\n")
    for s, r in zip(surv, res):
        same = np.sign(r.mean_ic) == np.sign(s["mean_ic"])
        ok = bool(same and np.isfinite(r.p) and r.p < ALPHA)
        big = ok and abs(s["edge_bp"]) > ROUND_TRIP_BP and abs(r.edge_bp) > ROUND_TRIP_BP
        verdict = "TRADEABLE" if big else ("CONFIRMED, too small to trade"
                                           if ok else "NOT CONFIRMED")
        print(f"  {r.feature:11s} h={r.horizon:<3d} discovery IC {s['mean_ic']:+.4f} "
              f"-> confirm {r.mean_ic:+.4f} (t {r.t:+.2f})  "
              f"edge {r.edge_bp:+.2f}bp vs {ROUND_TRIP_BP:.1f}bp   {verdict}")

    with open(CONFIRM_MARK, "w") as fh:
        fh.write("Confirmation run completed. Do not re-run.\n")
    print("\n  BACKTEST-ONLY. Not a recommendation to trade.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["discover", "confirm"])
    ap.add_argument("--force", action="store_true",
                    help="re-run confirmation anyway; defeats its purpose")
    args = ap.parse_args()
    return discover() if args.stage == "discover" else confirm(args.force)


if __name__ == "__main__":
    raise SystemExit(main())
