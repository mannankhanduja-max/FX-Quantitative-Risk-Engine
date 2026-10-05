"""
EXPLORATORY. Not a test, and deliberately writes no DONE marker.

Mannan asked for XAU/USD + NAS100 + USD/JPY after seeing the
six-instrument table, in which those three carried the highest
corrected excess. A number from a subset chosen that way is not
evidence, so reporting it alone would be misleading.

What makes it interpretable is the reference distribution: all 20
three-instrument subsets of the six, ranked. If the requested subset
sits at the top, that IS the demonstration that the choice of subset is
doing the work rather than the rule.

Two selections are then distinguished, because they are not equally
guilty:

  BY OUTCOME   picking the instruments that scored well. Circular. The
               excess it reports means nothing.

  BY COST      picking the instruments with the lowest measured
               2*spread/stop. That is a property of the instrument,
               measurable before any backtest, and a legitimate ex-ante
               tradeability filter. It mechanically improves net R - and
               it does NOT legitimise the win-rate excess, because cost
               does not move barriers. The zero-cost diagnostic earlier
               in this study showed the win rate is identical with
               friction switched off.

If those two selections pick the same three instruments, the net R
figure is admissible and the excess figure still is not.

BACKTEST-ONLY.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config
from added_power_test import (CHUNKS, NULL_BIAS_1H, clustered, excess_cells,
                              one, resample)
from fxrisk.data import intraday
from fxrisk.research import barrier_prob as bp
from fxrisk.risk import spread
from fxrisk.strategies import mtf

REQUESTED = ("XAU/USD", "NAS100", "USD/JPY")
OUT = "results/subset_probe.txt"


def cost_r():
    """Measured 2*spread/stop per instrument. No outcomes involved."""
    out = {}
    for inst in config.UNIVERSE_INTRADAY:
        s = spread.real_spread(inst.yahoo, "5m", ask_interval="5m_ask")
        half = float(np.median(s)) * 1e4 / 2.0
        b5 = intraday.load_symbol(inst.yahoo, "5m")
        trig = resample(b5, "1h")
        stop_bp = float((mtf.atr(trig, 14) / trig["Close"]).median() * 1e4)
        out[inst.name] = 2.0 * (half + 0.35) / stop_bp
    return out


def main():
    parts = []
    for inst in config.UNIVERSE_INTRADAY:
        for iv, ask in CHUNKS:
            t, _ = one(inst, iv, ask, 0.35)
            if t is not None:
                parts.append(t)
    a = pd.concat(parts, ignore_index=True)
    a["entry_time"] = pd.to_datetime(a["entry_time"], utc=True, format="mixed")
    a = (a.sort_values("entry_time")
         .drop_duplicates(subset=["instrument", "entry_time"], keep="first")
         .reset_index(drop=True))

    cr = cost_r()
    cheapest = tuple(sorted(cr, key=cr.get)[:3])
    names = [i.name for i in config.UNIVERSE_INTRADAY]

    rows = []
    for combo in itertools.combinations(names, 3):
        g = a[a["instrument"].isin(combo)]
        if g.empty:
            continue
        c = clustered(excess_cells(g), NULL_BIAS_1H)
        r = g["unit_net_R"]
        rt = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        rows.append({"combo": combo, "n": len(g),
                     "corrected": c["mean"] - NULL_BIAS_1H, "t": c["t"],
                     "net_R": r.mean(), "net_t": rt,
                     "cost_R": np.mean([cr[x] for x in combo])})
    d = pd.DataFrame(rows).sort_values("corrected", ascending=False)
    d = d.reset_index(drop=True)
    d["rank"] = d.index + 1

    L = ["ALL 20 THREE-INSTRUMENT SUBSETS, ranked by corrected excess",
         "  1h monitoring frame, 3:1, measured spread + 0.35bp/side",
         f"  H0 for every t is the null-path bias {NULL_BIAS_1H:+.2%}",
         "  EXPLORATORY. No subset here was chosen before the data was seen.",
         ""]
    L.append("  cost_R per instrument, measured, no outcomes involved:")
    L.append("    " + "  ".join(f"{k} {v:.3f}" for k, v in
                                sorted(cr.items(), key=lambda kv: kv[1])))
    L.append(f"    three cheapest: {', '.join(cheapest)}")
    L.append("")
    L.append(f"  {'#':>3}  {'subset':<34}{'n':>6}{'corrected':>11}{'t':>7}"
             f"{'net R':>9}{'t':>7}{'cost_R':>8}")
    for _, r in d.iterrows():
        mark = ""
        if tuple(r["combo"]) == tuple(REQUESTED):
            mark = "  <- requested"
        if set(r["combo"]) == set(cheapest):
            mark += "  <- cheapest by cost_R"
        L.append(f"  {int(r['rank']):>3}  {' + '.join(r['combo']):<34}"
                 f"{int(r['n']):>6}{r['corrected']:>+11.2%}{r['t']:>+7.2f}"
                 f"{r['net_R']:>+9.4f}{r['net_t']:>+7.2f}"
                 f"{r['cost_R']:>8.3f}{mark}")

    req = d[d["combo"].apply(lambda c: tuple(c) == tuple(REQUESTED))].iloc[0]
    L += ["", "READING IT"]
    L.append(f"  The requested subset ranks {int(req['rank'])} of "
             f"{len(d)} on corrected excess.")
    L.append(f"  {(d['corrected'] > 0).mean():.0%} of all 20 subsets show a "
             f"positive corrected excess;")
    L.append(f"  {(d['t'] > 2.0).mean():.0%} clear t > 2, and "
             f"{(d['net_R'] > 0).mean():.0%} show positive net R.")
    L.append(f"  Spread across subsets: corrected excess "
             f"{d['corrected'].min():+.2%} to {d['corrected'].max():+.2%}, "
             f"net R {d['net_R'].min():+.4f} to {d['net_R'].max():+.4f}.")
    L.append("  That range is what subset choice alone can manufacture from "
             "one rule")
    L.append("  on one dataset. Any single subset must be read against it.")

    if set(REQUESTED) == set(cheapest):
        L += ["", "  The requested three ARE the three cheapest by measured "
                  "cost_R, which is",
              "  an ex-ante property. So the net R figure for this subset is "
              "admissible as",
              "  'the rule on the only instruments cheap enough to carry it'. "
              "The corrected",
              "  excess is NOT, because these are also the three that scored "
              "best, and cost",
              "  does not move barriers - the win rate is identical with "
              "friction off."]
    else:
        L += ["", f"  The requested three are NOT the three cheapest "
                  f"({', '.join(cheapest)}),",
              "  so there is no ex-ante story for this subset. It is "
              "selection on outcome."]

    text = "\n".join(L) + "\n"
    print(text)
    with open(OUT, "w") as fh:
        fh.write(text)


if __name__ == "__main__":
    main()
