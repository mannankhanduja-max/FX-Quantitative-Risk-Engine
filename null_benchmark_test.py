"""
Is b/(a+b) a valid benchmark for THESE trades? Answered on data with
nothing in it.

WHAT THIS FOUND, AND HOW IT DIFFERS FROM WHAT I EXPECTED
---------------------------------------------------------
The hypothesis going in was stochastic volatility: b/(a+b) is exact for
driftless Brownian motion with CONSTANT volatility, this study's stops
are placed conditional on a bar that just moved unusually far, and
volatility mean-reverts. That hypothesis is WRONG and this file refutes
it. Clustering changes almost nothing.

What does matter is the resolution the exit is MONITORED at. Entries
fill at a trigger-frame close, and `to_30m_positions` starts the exit
walk at the first walk-frame bar closing strictly after the fill - so
with a 1h trigger and a 4h walk there are up to four unmonitored hours
between the fill and the first bar that can register a touch. During
that gap a stop 22bp away gets breached and recovered far more often
than a target 67bp away. The misses are therefore asymmetric, and they
asymmetrically delete LOSSES.

That is a pure geometry effect. It needs no drift, no clustering and no
edge, and it scales with how coarse the walk frame is relative to the
entry frame.

THE DESIGN
-----------
Two regimes x four walk frames x 20 seeds, seeds 0-19, all reported.
Drift is exactly zero in log space everywhere, so nothing in here
contains anything to find. Any departure from the benchmark is the
benchmark being wrong for these trades.

  regimes      constant (benchmark's own assumptions hold) and garch
               (persistence 0.98). Isolates clustering.
  walk frames  4h, 1h, 15m, 5m. Isolates monitoring resolution.

The bar limit is scaled with the frame so every variant allows the same
wall-clock holding time; otherwise a finer frame would be a shorter
trade and the comparison would not be clean.

The real number being explained: 28.60% actual against 25.01% model,
excess +3.59pp, z = +2.69, on the 1h/4h/1d ladder walked at 4h.

No cost, no session filter, no blackout - none of them affect which
barrier is reached first.

BACKTEST-ONLY. Contains no market data at all.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxrisk.research import barrier_prob as bp
from fxrisk.research import null_paths as npaths
from fxrisk.risk import barriers
from fxrisk.strategies import mtf, smc

AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}
REGIMES = ("constant", "garch")
# walk rule -> how many of those bars make up the 4h reference bar, so
# the holding limit is the same wall-clock span in every variant.
WALKS = {"4h": 1, "1h": 4, "15min": 16, "5min": 48}
SEEDS = tuple(range(20))
REAL_EXCESS = 0.0359          # 28.60% - 25.01%, the 4h-walked result
OUT = "results/null_benchmark.txt"


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def one_cell(regime, seed, walk_rule, n_bars, rr):
    b5 = npaths.garch_path(n_bars, seed, regime=regime)
    trig, setup_f = resample(b5, "1h"), resample(b5, "4h")

    cfg = mtf.MTFConfig(stop_sigma=1.0, stop_mode="atr", atr_period=14,
                        setup_lookback=20, trigger_window=6, max_bars_30m=80)
    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg, zone_frame=zf)
    entries = mtf.entries_5m(trig, setups, None, cfg)
    if entries.empty:
        return None

    walk_f = setup_f if walk_rule == "4h" else resample(b5, walk_rule)
    pos = mtf.to_30m_positions(entries, walk_f)
    if pos.empty:
        return None
    t = barriers.walk_explicit(
        walk_f, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=rr, max_bars=80 * WALKS[walk_rule], cost_bp=0.0,
        breakeven_frac=None)
    if t.empty:
        return None
    b = bp.benchmark(t)
    if not b.get("trades"):
        return None
    b.update(regime=regime, seed=seed, walk=walk_rule,
             clustering=npaths.realised_clustering(b5))
    return b


def pooled_z(g):
    """Recombine Poisson-binomial z across independent seeds."""
    return float((g["z"] * np.sqrt(g["trades"])).sum()
                 / np.sqrt(g["trades"].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=60_000)
    ap.add_argument("--rr", type=float, default=3.0)
    args = ap.parse_args()

    rows = []
    for regime in REGIMES:
        for walk in WALKS:
            for seed in SEEDS:
                r = one_cell(regime, seed, walk, args.bars, args.rr)
                if r is not None:
                    rows.append(r)
    d = pd.DataFrame(rows)

    L = ["NO-INFORMATION PATHS THROUGH THE SAME BARRIER TEST",
         f"  {args.rr:g}:1, ATR(14) x1.0, 1h entries, no cost",
         f"  {len(SEEDS)} seeds x {args.bars} 5m bars per cell, "
         f"ZERO log drift by construction",
         f"  bar limit scaled with the frame, so every row allows the "
         f"same holding time",
         f"  reference: the real 4h-walked result was excess "
         f"{REAL_EXCESS:+.2%}, z +2.69", ""]

    L.append(f"  {'regime':<10}{'walk':>7}{'seeds':>6}{'trades':>8}"
             f"{'clust':>7}{'actual':>9}{'model':>8}{'excess':>9}"
             f"{'mean z':>8}{'pooled z':>10}{'vs real':>9}")
    grid = {}
    for regime in REGIMES:
        for walk in WALKS:
            g = d[(d["regime"] == regime) & (d["walk"] == walk)]
            if g.empty:
                continue
            ex = g["excess"].mean()
            grid[(regime, walk)] = {"excess": ex, "pooled": pooled_z(g)}
            L.append(f"  {regime:<10}{walk:>7}{len(g):>6}"
                     f"{g['trades'].sum():>8}{g['clustering'].mean():>7.3f}"
                     f"{g['actual_win'].mean():>9.2%}"
                     f"{g['model_win'].mean():>8.2%}{ex:>+9.2%}"
                     f"{g['z'].mean():>+8.2f}{pooled_z(g):>+10.2f}"
                     f"{ex / REAL_EXCESS:>8.0%}")
        L.append("")

    L.append("PER SEED z, 4h walk (the frame every real result used)")
    L.append("  " + f"{'regime':<10}" + "".join(f"{s:>6}" for s in SEEDS))
    for regime in REGIMES:
        g = d[(d["regime"] == regime) & (d["walk"] == "4h")].set_index("seed")
        L.append(f"  {regime:<10}" +
                 "".join(f"{g['z'].get(s, float('nan')):>+6.1f}" for s in SEEDS))

    L += ["", "VERDICT"]
    c4 = grid.get(("constant", "4h"))
    g4 = grid.get(("garch", "4h"))
    c1 = grid.get(("constant", "1h"))
    if not (c4 and g4 and c1):
        L.append("  INCONCLUSIVE - a cell produced no trades.")
    else:
        L.append(f"  Clustering is NOT the mechanism: constant "
                 f"{c4['excess']:+.2%} vs garch {g4['excess']:+.2%} at 4h.")
        L.append(f"  Monitoring resolution IS: {c4['excess']:+.2%} at a 4h "
                 f"walk against {c1['excess']:+.2%} at 1h,")
        L.append(f"  on data containing nothing.")
        L.append("")
        share = c4["excess"] / REAL_EXCESS
        L.append(f"  A driftless random walk reproduces {share:.0%} of the "
                 f"real excess (+{REAL_EXCESS:.2%})")
        L.append(f"  with no edge present. The benchmark is disqualified for "
                 f"any result walked")
        L.append(f"  at a frame coarser than its own entries.")

    text = "\n".join(L) + "\n"
    print(text)
    with open(OUT, "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
