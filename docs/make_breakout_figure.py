"""
Regenerate docs/figures/06-win-rate-hurdle.png from the backtest.

    python build_regime.py                  # once, if not already cached
    python docs/make_breakout_figure.py

A separate script from make_figures.py because it needs a different
data source - the Dukascopy intraday cache rather than the Yahoo
daily one - and takes a couple of minutes, so folding it into the
main figure run would make every daily-engine figure wait on it.

Same rule as make_figures.py: every number in the picture comes
from a run of the code, not from a table typed into this file. The
seven rows are recomputed from breakout_test and gate_test, and the
gated configuration is re-chosen on the in-sample block by the same
function gate_test.py uses, so the picture cannot describe a choice
the test never made.

Same rule on annotation too: axes, legend and units only. What the
chart shows is argued in README section 5a, where it can be edited
without regenerating a PNG.
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import config  # noqa: E402
import gate_test  # noqa: E402
import make_figures as mf  # noqa: E402  (house style, colours, _save)
from breakout_test import run_one  # noqa: E402
from fxrisk.data import intraday  # noqa: E402
from fxrisk.risk import barriers  # noqa: E402
from fxrisk.strategies import breakout_retrace as br  # noqa: E402

SHORTFALL = "#b7d3f6"     # sequential blue, step 150 - same hue as S1


def _cfg(stop_sigma: float) -> br.SetupConfig:
    return br.SetupConfig(
        lookback=config.BREAKOUT_LOOKBACK,
        retrace_max_bars=config.RETRACE_MAX_BARS,
        min_fraction=config.RETRACE_MIN_FRACTION,
        max_fraction=config.RETRACE_MAX_FRACTION,
        stop_min_sigma=stop_sigma,
        use_trend_filter=True,
    )


def full_sample_row(label, stop_sigma, use_be, max_bars, cache_dir):
    """One pooled row over the whole sample, via breakout_test."""
    acc = []
    for inst in config.UNIVERSE_INTRADAY:
        be = (inst.be_bp / 10_000.0) if use_be else None
        _, _, t = run_one(inst, _cfg(stop_sigma), config.BREAKOUT_RR,
                          config.BREAKOUT_COST_BP, max_bars, be,
                          "15m", cache_dir)
        if not t.empty:
            acc.append(t)
    s = barriers.summarise_explicit(pd.concat(acc, ignore_index=True),
                                    config.BREAKOUT_RR)
    print(f"  {label:28s} n {s['trades']:4d}  "
          f"win {s['win_rate_barrier']:.2%}  hurdle {s['breakeven_wr']:.2%}")
    return {"group": "Full sample", "label": label, "n": s["trades"],
            "wr": s["win_rate_barrier"], "hz": s["breakeven_wr"]}


def gate_rows(cache_dir, min_trades):
    """In-sample gated, held-out ungated, held-out gated."""
    args = SimpleNamespace(rr=config.BREAKOUT_RR,
                           cost_bp=config.BREAKOUT_COST_BP,
                           max_bars=config.BREAKOUT_MAX_BARS,
                           min_trades=min_trades)
    bars, gatef = gate_test.load_all("15m", cache_dir)
    ins = {k: (0, int(len(v) * gate_test.SPLIT)) for k, v in bars.items()}
    out = {k: (int(len(v) * gate_test.SPLIT), len(v)) for k, v in bars.items()}

    _, best = gate_test.choose_in_sample(bars, gatef, ins, args)
    if best is None:
        raise SystemExit("no in-sample configuration qualified")
    ss, be = best["stop_sigma"], bool(best["breakeven"])
    corr, rho = bool(best["corr"]), best["rho_min"]
    print(f"  frozen in-sample: stop {ss:g} sigma, breakeven {be}, "
          f"correlation {corr} (rho_min {rho:g})")

    rows = []
    for label, trades in (
        ("Gated, in-sample",
         gate_test.pooled(bars, gatef, ins, ss, be, corr, rho, args)),
        ("Ungated control, held out",
         gate_test.ungated(bars, out, ss, be, args)),
        ("Gated, held out",
         gate_test.pooled(bars, gatef, out, ss, be, corr, rho, args)),
    ):
        s = barriers.summarise_explicit(trades, config.BREAKOUT_RR)
        print(f"  {label:28s} n {s['trades']:4d}  "
              f"win {s['win_rate_barrier']:.2%}  hurdle {s['breakeven_wr']:.2%}")
        rows.append({"group": "Risk gates", "label": label, "n": s["trades"],
                     "wr": s["win_rate_barrier"], "hz": s["breakeven_wr"]})
    return rows


def draw(rows: list[dict]) -> plt.Figure:
    mf._style()
    df = pd.DataFrame(rows)

    # Top-to-bottom reading order, with a blank slot between groups.
    labels, wr, hz, ypos = [], [], [], []
    y, last = 0.0, None
    for r in df.itertuples():
        if last is not None and r.group != last:
            y += 0.6
        labels.append(f"{r.label}  (n={r.n})")
        wr.append(r.wr * 100)
        hz.append(r.hz * 100)
        ypos.append(y)
        y += 1.0
        last = r.group

    fig, ax = plt.subplots(figsize=(9, 3.9))
    h = 0.62

    ax.barh(ypos, [b - a for a, b in zip(wr, hz)], left=wr, height=h * 0.45,
            color=SHORTFALL, label="shortfall to hurdle", zorder=2)
    ax.barh(ypos, wr, height=h, color=mf.S1,
            label="win rate, barrier exits", zorder=3)
    for yy, v in zip(ypos, hz):
        ax.plot([v, v], [yy - h / 2 - 0.08, yy + h / 2 + 0.08],
                color=mf.INK, linewidth=2.2, solid_capstyle="butt", zorder=4)
    ax.plot([], [], color=mf.INK, linewidth=2.2,
            label="cost-adjusted breakeven")

    for yy, v in zip(ypos, wr):
        ax.text(v - 0.6, yy, f"{v:.1f}%", ha="right", va="center",
                fontsize=8.5, color=mf.SURFACE, fontweight="bold", zorder=5)

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 50)
    ax.set_xlabel("win rate (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Breakout rule, 2:1: win rate against its own hurdle",
                 loc="left")
    # Below the axis, not inside it: every corner of the plot holds a
    # hurdle tick for some row, so any in-plot legend covers data.
    handles, names = ax.get_legend_handles_labels()
    order = [names.index(n) for n in ("win rate, barrier exits",
                                      "shortfall to hurdle",
                                      "cost-adjusted breakeven")]
    ax.legend([handles[i] for i in order], [names[i] for i in order],
              loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=3,
              fontsize=8)
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=intraday.DEFAULT_CACHE)
    ap.add_argument("--min-trades", type=int, default=30)
    args = ap.parse_args()

    os.chdir(ROOT)
    print("Recomputing every row from the backtest\n")
    rows = [
        full_sample_row("Base rule, 1σ stop", 1.0, True,
                        config.BREAKOUT_MAX_BARS, args.cache_dir),
        full_sample_row("No breakeven stop, 1σ", 1.0, False, 80, args.cache_dir),
        full_sample_row("Stop widened to 2σ", 2.0, False, 80, args.cache_dir),
        full_sample_row("Stop widened to 4σ", 4.0, False, 80, args.cache_dir),
    ]
    rows += gate_rows(args.cache_dir, args.min_trades)

    print()
    mf._save(draw(rows), "06-win-rate-hurdle", "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
