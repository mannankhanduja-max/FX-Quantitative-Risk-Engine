"""
Design-period run of the FVG-confirmed breakout/retracement rule.

    python run_fvg_design.py                 # real data: 2015-01-01 -> 2022-01-01
    python run_fvg_design.py --demo           # synthetic data, no download needed
    python run_fvg_design.py --rr 3 --stop-min-sigma 1.5

THIS IS THE DESIGN-PERIOD RUN. Free to re-run with different
parameters as often as useful - that is what a design period is for.
`run_fvg_test.py` is the one-shot, held-out counterpart, gated behind
`PREREGISTRATION.md`; nothing here writes to that file or its
DONE marker.

Reads `data/raw/XAUUSD_1m_{bid,ask}_{2015..2021}.csv`
(`fetch_fvg_raw.py` writes them; that step needs internet and this run
does not) and writes:

    results/fvg_design.csv         one row per configuration, if --sweep
    results/fvg_design_trades.csv  every trade the headline config took
    docs/figures/fvg_design.png    win rate vs cost-adjusted hurdle
    RESULTS.md                     the human-readable summary

A null result is reported as a null result - there is no step here
that makes a negative number look like anything else.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxrisk.data import fvg_raw  # noqa: E402
from fxrisk.research import fvg_subsets  # noqa: E402
from fxrisk.risk import barriers  # noqa: E402
from fxrisk.strategies import fvg_retrace as fr  # noqa: E402
from fxrisk.strategies.smc import SMCConfig  # noqa: E402

RESULTS_TRADES = "results/fvg_design_trades.csv"
RESULTS_SWEEP = "results/fvg_design.csv"
FIGURE_PATH = "docs/figures/fvg_design.png"
RESULTS_MD = "RESULTS.md"

DEMO_BARS = 60_000      # a bit over 40 trading days of 1-minute bars
DEMO_SEED = 0


def load_demo_bars() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic 1-minute bid/ask gold bars, so the whole pipeline - rule,
    barrier walk, real-cost measurement, figure, RESULTS.md - can be
    exercised and inspected without the real download. A random walk
    has no edge by construction, so a non-null result here would be a
    bug in the harness, not a finding; this run is always labelled
    DEMO_SIMULATED and is never a substitute for the real one.
    """
    rng = np.random.default_rng(DEMO_SEED)
    n = DEMO_BARS
    idx = pd.date_range("2020-01-02", periods=n, freq="1min", tz="UTC")
    ret = rng.normal(0, 0.00025, n)
    for start in range(300, n - 300, 600):   # occasional trend bursts
        ret[start:start + 20] += rng.choice([-1, 1]) * 0.0005
    mid = 1800.0 * np.exp(np.cumsum(ret))
    wick = np.abs(rng.normal(0, 0.00035, n))
    openp = np.roll(mid, 1)
    openp[0] = mid[0]
    high = np.maximum.reduce([mid * (1 + wick), openp, mid])
    low = np.minimum.reduce([mid * (1 - wick), openp, mid])
    vol = rng.integers(20, 300, n).astype(float)
    spread_bp = rng.uniform(1.5, 3.0, n)  # synthetic, gold-ish bp spread

    bid = pd.DataFrame({"Open": openp, "High": high, "Low": low,
                        "Close": mid, "Volume": vol}, index=idx)
    ask_mid = mid * (1 + spread_bp / 10_000.0)
    ask = pd.DataFrame({
        "Open": np.roll(ask_mid, 1), "High": high * (1 + spread_bp / 10_000.0),
        "Low": low * (1 + spread_bp / 10_000.0), "Close": ask_mid, "Volume": vol,
    }, index=idx)
    return bid, ask


def run_config(bid: pd.DataFrame, ask: pd.DataFrame, cfg: fr.FVGRetraceConfig,
               rr: float, max_bars: int, be_bp: float | None,
               commission_bp: float):
    setups = fr.find_setups(bid, cfg)
    if setups.empty:
        return setups, pd.DataFrame()
    cost = fvg_raw.real_cost_bp(bid, ask, bid, commission_bp=commission_bp)
    be_frac = None if be_bp is None else be_bp / 10_000.0
    trades = barriers.walk_explicit(
        bid, setups[["bar", "side", "entry", "stop"]],
        rr=rr, max_bars=max_bars, cost_bp=cost.to_numpy(), breakeven_frac=be_frac,
    )
    for col in ("retrace_frac", "wait_bars", "stop_from", "zone_touch_bar"):
        if col in setups.columns and len(trades) == len(setups):
            trades[col] = setups[col].to_numpy()
    return setups, trades


def section(title: str, trades: pd.DataFrame, rr: float) -> str:
    if trades.empty:
        return f"### {title}\n\nNo trades.\n"
    s = barriers.summarise_explicit(trades, rr)
    lines = [
        f"### {title}", "",
        f"| trades | win rate (barrier) | cost-adj. hurdle | gap | mean net R | total net R | t |",
        f"|---|---|---|---|---|---|---|",
        f"| {s['trades']} | {s['win_rate_barrier']:.2%} | {s['breakeven_wr']:.2%} "
        f"| {s['gap_vs_breakeven']:+.2%} | {s['mean_net_R']:+.4f} | "
        f"{s['total_net_R']:+.2f} | ",
    ]
    r = trades["net_R"].to_numpy(dtype="float64")
    if len(r) > 5 and r.std(ddof=1) > 0:
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        lines[-1] += f"{t:+.2f} |"
    else:
        lines[-1] += "n/a |"
    lines.append("")
    return "\n".join(lines)


def make_figure(headline: dict, subset_summaries: list[tuple[str, dict]], path: str) -> None:
    rows = [("overall", headline)] + subset_summaries
    rows = [(name, s) for name, s in rows if s is not None]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(rows) + 1)))
    names = [r[0] for r in rows]
    win = [r[1]["win_rate_barrier"] for r in rows]
    hurdle = [r[1]["breakeven_wr"] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, win, height=0.5, color="#2a78d6", label="win rate (barrier)")
    ax.scatter(hurdle, y, color="#e34948", zorder=3, label="cost-adjusted hurdle")
    ax.set_yticks(y, names)
    ax.set_xlabel("win rate")
    ax.set_xlim(0, max(0.5, max(win + hurdle) * 1.15))
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("FVG-confirmed breakout/retracement: win rate vs cost hurdle\n"
                 "(design period)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="synthetic data, no download needed")
    ap.add_argument("--raw-dir", default=fvg_raw.DEFAULT_RAW_DIR)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--retrace-max-bars", type=int, default=12)
    ap.add_argument("--min-fraction", type=float, default=0.33)
    ap.add_argument("--max-fraction", type=float, default=1.0)
    ap.add_argument("--stop-min-sigma", type=float, default=1.0)
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--max-bars", type=int, default=240,
                    help="time-stop, in 1-minute bars (240 = 4 hours)")
    ap.add_argument("--be-bp", type=float, default=None,
                    help="breakeven trigger in bp of entry price; omit to disable")
    ap.add_argument("--commission-bp", type=float, default=0.0)
    ap.add_argument("--news-min-impact", default="high", choices=("low", "medium", "high"))
    args = ap.parse_args()

    tag = "DEMO_SIMULATED_" if args.demo else ""
    print(f"FVG-confirmed breakout/retracement, design period "
         f"({'synthetic' if args.demo else 'real'} data)\n")

    if args.demo:
        bid, ask = load_demo_bars()
    else:
        try:
            bid = fvg_raw.load_design(args.raw_dir)
            ask = fvg_raw.load_side("ask", fvg_raw.DESIGN_START, fvg_raw.DESIGN_END,
                                    args.raw_dir)
        except FileNotFoundError as exc:
            print(exc)
            print("\nRun with --demo to exercise the pipeline on synthetic data "
                 "while the real download is in progress.")
            return 1

    print(f"  bars    {len(bid):,}  {bid.index[0]} -> {bid.index[-1]}")

    cfg = fr.FVGRetraceConfig(
        lookback=args.lookback, retrace_max_bars=args.retrace_max_bars,
        min_fraction=args.min_fraction, max_fraction=args.max_fraction,
        stop_min_sigma=args.stop_min_sigma, use_trend_filter=not args.no_trend,
        smc=SMCConfig(),
    )
    setups, trades = run_config(bid, ask, cfg, args.rr, args.max_bars,
                                args.be_bp, args.commission_bp)
    print(f"  setups  {len(setups)}")
    if not trades.empty:
        s = barriers.summarise_explicit(trades, args.rr)
        print(f"  trades  {s['trades']}  win rate (barrier) {s['win_rate_barrier']:.2%}  "
             f"hurdle {s['breakeven_wr']:.2%}  mean net R {s['mean_net_R']:+.4f}")
    os.makedirs("results", exist_ok=True)
    trades.to_csv(RESULTS_TRADES, index=False)

    # ---- optional subsets ----
    calendar = fvg_subsets.load_news_calendar()
    windows = fvg_subsets.load_stress_windows()
    print(f"\n  news_calendar.csv   {'found' if calendar is not None else 'not present (skipped)'}")
    print(f"  stress_windows.csv  {'found' if windows is not None else 'not present (skipped)'}")

    md = [f"# {tag}FVG design-period results\n",
         f"Generated by `run_fvg_design.py`. "
         f"Window: {bid.index[0]} -> {bid.index[-1]} ({len(bid):,} 1-minute bars).\n",
         "Rule: " + ("VWAP/EMA trend filter + " if not args.no_trend else "") +
         f"{args.lookback}-bar breakout, {args.min_fraction:.0%}-{args.max_fraction:.0%} "
         f"retrace into a known fair value gap, {args.rr:g}:1, "
         f"stop floor {args.stop_min_sigma:g} EWMA sigma" +
         (f", breakeven at {args.be_bp:g}bp" if args.be_bp else "") + ".\n",
         "**BACKTEST-ONLY. Not a recommendation to trade.**\n",
         section("Overall", trades, args.rr)]

    headline = barriers.summarise_explicit(trades, args.rr) if not trades.empty else None
    subset_summaries = []

    if calendar is not None and not trades.empty:
        entry_times = pd.to_datetime(trades["entry_time"], utc=True)
        flagged = fvg_subsets.in_news_window(entry_times, calendar,
                                             min_impact=args.news_min_impact)
        near = trades[flagged]
        away = trades[~flagged]
        md.append(section(f"Within 30 minutes of a {args.news_min_impact}-impact event",
                          near, args.rr))
        md.append(section("Away from scheduled news", away, args.rr))
        s_near = barriers.summarise_explicit(near, args.rr) if not near.empty else None
        if s_near:
            subset_summaries.append(("near news", s_near))

    if windows is not None and not trades.empty:
        entry_times = pd.to_datetime(trades["entry_time"], utc=True)
        labels = fvg_subsets.stress_window_labels(entry_times, windows)
        for name in windows["name"]:
            mask = np.array([name in lab for lab in labels])
            sub = trades[mask]
            md.append(section(f"Stress window: {name}", sub, args.rr))
            if not sub.empty:
                subset_summaries.append((name, barriers.summarise_explicit(sub, args.rr)))

    if headline is not None:
        make_figure(headline, subset_summaries, FIGURE_PATH)
        md.append(f"![win rate vs hurdle]({FIGURE_PATH})\n")

    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(md))
    print(f"\nWrote {RESULTS_TRADES}, {FIGURE_PATH if headline else '(no figure, no trades)'}"
         f" and {RESULTS_MD}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
