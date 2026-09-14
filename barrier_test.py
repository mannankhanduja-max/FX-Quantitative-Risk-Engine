"""
1:1 risk-reward on the VWAP/EMA signal, measured.

A bracket with equal stop and target distances is the triple-barrier
method: enter, then exit at whichever of three barriers is hit first
- target, stop, or a time limit.

THE INTRABAR AMBIGUITY, STATED. Daily bars record High and Low but
not their order, so when both barriers fall inside one day's range
there is no way to know which was touched first. Resolving that
optimistically (target first) is the single most common way a
bracket backtest becomes fiction. This resolves it as STOP first,
which is pessimistic but never flattering, and reports how often
the ambiguity arose so the reader can judge how much it matters.

Barrier distance is k * sigma * price, with sigma the EWMA daily
volatility already used elsewhere in this repository, so the stop
adapts to the regime instead of being a fixed percentage that is
too tight in a crisis and too loose in a calm.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from fxrisk import indicators
from fxrisk.data import yahoo
from fxrisk.risk.barriers import ewma_sigma


def barrier_trades(
    bars: pd.DataFrame,
    signal: pd.Series,
    k: float = 1.0,
    rr: float = 1.0,
    max_days: int = 10,
    cost_bp: float = 2.0,
) -> pd.DataFrame:
    """
    Walk the bars, opening a trade whenever the signal flips.

    Exit at target, stop, or after `max_days`. Returns one row per
    closed trade, with the R multiple achieved.
    """
    close = bars["Close"]
    high, low = bars["High"], bars["Low"]
    ret = close.pct_change(fill_method=None)
    sigma = ewma_sigma(ret)

    sig = signal.reindex(bars.index).fillna(0.0)
    flips = sig.diff().fillna(sig).ne(0) & sig.ne(0)

    trades = []
    idx = bars.index
    i = 0
    n = len(idx)

    while i < n - 1:
        if not flips.iloc[i] or not np.isfinite(sigma.iloc[i]):
            i += 1
            continue

        side = float(np.sign(sig.iloc[i]))
        entry = float(close.iloc[i])
        stop_d = k * float(sigma.iloc[i]) * entry
        if stop_d <= 0:
            i += 1
            continue
        targ_d = stop_d * rr

        target = entry + side * targ_d
        stop = entry - side * stop_d

        outcome, exit_px, held, ambiguous = "time", None, 0, False

        for j in range(i + 1, min(i + 1 + max_days, n)):
            held = j - i
            hi, lo = float(high.iloc[j]), float(low.iloc[j])

            hit_t = hi >= target if side > 0 else lo <= target
            hit_s = lo <= stop if side > 0 else hi >= stop

            if hit_t and hit_s:
                # Both inside one bar. Order unknowable - resolve
                # against the trade.
                ambiguous = True
                outcome, exit_px = "stop", stop
                break
            if hit_s:
                outcome, exit_px = "stop", stop
                break
            if hit_t:
                outcome, exit_px = "target", target
                break

        if exit_px is None:
            exit_px = float(close.iloc[min(i + max_days, n - 1)])

        gross_r = side * (exit_px - entry) / stop_d
        cost_r = (2 * cost_bp / 10_000) * entry / stop_d  # round trip
        trades.append({
            "entry_date": idx[i],
            "side": side,
            "entry": entry,
            "exit": exit_px,
            "outcome": outcome,
            "days_held": held,
            "ambiguous": ambiguous,
            "gross_R": gross_r,
            "net_R": gross_r - cost_r,
        })

        i += max(held, 1)

    return pd.DataFrame(trades)


def summarise(df: pd.DataFrame, label: str, rr: float, cost_bp: float) -> dict:
    if df.empty:
        return {"symbol": label, "trades": 0}

    wins = df["net_R"] > 0
    wr = float(wins.mean())

    # Breakeven win rate. The naive form, p = 1/(1+rr), ignores
    # costs and is the version quoted in most trading material - at
    # 1:1 it says you need 50%. But the round trip is paid on every
    # trade regardless of outcome, and expressed in units of risk it
    # is cost_R = 2*cost / stop_distance. Solving
    #
    #   p*rr - (1-p)*1 - cost_R = 0   ->   p = (1 + cost_R)/(1 + rr)
    #
    # A tight stop makes cost_R large, so the hurdle rises as the
    # stop narrows - the opposite of the intuition that a tight stop
    # is conservative.
    cost_r = float(
        ((df["gross_R"] - df["net_R"]).abs()).mean()
    )
    be = (1.0 + cost_r) / (1.0 + rr)

    return {
        "symbol": label,
        "trades": len(df),
        "win_rate": wr,
        "breakeven_wr": be,
        "gap_vs_breakeven": wr - be,
        "cost_R": cost_r,
        "mean_net_R": float(df["net_R"].mean()),
        "total_net_R": float(df["net_R"].sum()),
        "target_hits": int((df["outcome"] == "target").sum()),
        "stop_hits": int((df["outcome"] == "stop").sum()),
        "time_exits": int((df["outcome"] == "time").sum()),
        "ambiguous_bars": int(df["ambiguous"].sum()),
        "ambiguous_share": float(df["ambiguous"].mean()),
        "mean_days": float(df["days_held"].mean()),
    }


def main() -> int:
    rr = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    cost_bp = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0

    print(f"1:{rr:g} risk-reward, stop at {k:g} x EWMA sigma, "
          f"{cost_bp:g}bp per side, 10-day time limit\n")

    rows = []
    for inst in config.UNIVERSE:
        bars = yahoo.load_symbol(inst.yahoo)
        sig = indicators.signal(
            bars, window=config.VWAP_WINDOW_DAILY, span=config.VWAP_EMA_SPAN
        )
        df = barrier_trades(bars, sig, k=k, rr=rr, cost_bp=cost_bp)
        rows.append(summarise(df, inst.yahoo, rr, cost_bp))

    out = pd.DataFrame(rows).set_index("symbol")
    show = out[["trades", "win_rate", "cost_R", "breakeven_wr",
                "gap_vs_breakeven", "mean_net_R", "total_net_R"]]
    print(show.round(4).to_string())
    print()
    print(out[["target_hits", "stop_hits", "time_exits",
               "ambiguous_share", "mean_days"]].round(3).to_string())

    tot_tr = int(out["trades"].sum())
    agg_wr = float((out["win_rate"] * out["trades"]).sum() / tot_tr)
    agg_r = float(out["total_net_R"].sum())
    cost_r = float((out["cost_R"] * out["trades"]).sum() / tot_tr)
    be = (1.0 + cost_r) / (1.0 + rr)
    print(f"\n  ALL: {tot_tr} trades, win rate {agg_wr:.2%}, "
          f"cost {cost_r:.3f}R per trade, breakeven needs {be:.2%}")
    print(f"  gap to breakeven {agg_wr - be:+.2%}   total {agg_r:+.1f}R")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
