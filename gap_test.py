"""
Overnight-gap continuation, with a held-out final third.

THE ONE RULE THIS SCRIPT EXISTS TO ENFORCE
-------------------------------------------
Everything else in this repository was measured on the full sample.
By the time the gap effect surfaced I had run roughly thirty
configurations over the same 4,918 days, so anything further found
in-sample is suspect by construction.

So the sample is split once, chronologically:

    IN-SAMPLE      first 67%   - choose instruments and threshold
    HELD OUT       last 33%    - touched once, with the rule frozen

Both choices that could be tuned are made on the in-sample block
ALONE:

  * which instruments to trade - selected by in-sample rank IC,
    not by the full-sample IC that first drew attention to FXY and
    GLD, because that figure already saw the holdout
  * the gap threshold - chosen as the in-sample percentile that
    maximises in-sample mean return, which is exactly the kind of
    fitting the holdout exists to punish

THE TRADE
---------
    gap(t)      = Open(t) / Close(t-1) - 1
    position(t) = sign(gap) if |gap| > threshold else 0
    return(t)   = position(t) * (Close(t)/Open(t) - 1) - cost

Continuation, not reversion - the measured IC was positive. Entry
at the open, exit at the close, one round trip per trading day.
There are no intraday barriers: daily bars cannot tell you the
order in which the high and low occurred, so a stop inside the
session would be a guess dressed as a rule.

WHAT CAN AND CANNOT BE CONCLUDED
---------------------------------
A positive holdout result here is weak evidence, not proof: one
split, one pass, a few hundred trades. A negative result is
stronger, because the rule was fitted in its own favour and still
failed. Asymmetry of that kind is the normal state of this work
and is worth saying out loud rather than burying.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from fxrisk.data import yahoo

SPLIT = 0.67
COST_BP_DEFAULT = 2.0
IC_MIN = 0.02          # in-sample IC needed to trade an instrument
TRADING_DAYS = 252


def gap_frame(sym: str) -> pd.DataFrame:
    """Overnight gap and the same-day open-to-close move."""
    bars = yahoo.load_symbol(sym)
    out = pd.DataFrame(index=bars.index)
    out["gap"] = bars["Open"] / bars["Close"].shift(1) - 1.0
    out["intraday"] = bars["Close"] / bars["Open"] - 1.0
    return out.dropna()


def split_at(frame: pd.DataFrame, frac: float = SPLIT) -> int:
    return int(len(frame) * frac)


def trade(frame: pd.DataFrame, threshold: float, cost_bp: float,
          long_only: bool = False) -> pd.Series:
    """Daily strategy return. Flat unless |gap| clears the threshold."""
    pos = np.sign(frame["gap"]).where(frame["gap"].abs() > threshold, 0.0)
    if long_only:
        pos = pos.clip(lower=0.0)
    cost = (2 * cost_bp / 10_000) * pos.abs()
    return pos * frame["intraday"] - cost


def stats_for(r: pd.Series) -> dict:
    """Return stats plus the t-statistic, which is what settles it."""
    live = r[r != 0.0]
    n = len(live)
    if n < 20:
        return {"trades": n}
    mean = float(live.mean())
    sd = float(live.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    return {
        "trades": n,
        "win_rate": float((live > 0).mean()),
        "mean_bp": mean * 10_000,
        "total_pct": float((1 + r).prod() - 1) * 100,
        "sharpe": (mean / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
        "t_stat": t,
        "p_value": float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))) if np.isfinite(t) else np.nan,
    }


def main() -> int:
    cost_bp = float(sys.argv[1]) if len(sys.argv) > 1 else COST_BP_DEFAULT
    print(f"Overnight-gap continuation, {cost_bp:g}bp per side\n")

    frames = {i.yahoo: gap_frame(i.yahoo) for i in config.UNIVERSE}
    shortable = {i.yahoo: i.shortable for i in config.UNIVERSE}

    # ---------- STEP 1: instrument selection, in-sample only ----------
    print("STEP 1  in-sample rank IC (gap -> same-day move)\n")
    chosen = []
    for sym, f in frames.items():
        k = split_at(f)
        ins = f.iloc[:k]
        ic, p = stats.spearmanr(ins["gap"], ins["intraday"])
        keep = ic > IC_MIN
        chosen.append(sym) if keep else None
        print(f"  {sym:5s} IC {ic:+.4f}  p {p:.3f}  n {len(ins)}   "
              f"{'SELECTED' if keep else 'rejected'}")

    if not chosen:
        print("\nNo instrument clears the in-sample IC floor. Nothing to test.")
        return 0
    print(f"\n  selected on IN-SAMPLE evidence: {', '.join(chosen)}")
    print("  (FXY and GLD were the full-sample winners; that figure "
          "already saw\n   the holdout, so it is not used here)")

    # ---------- STEP 2: threshold, in-sample only ----------
    print("\nSTEP 2  threshold fitted on the in-sample block\n")
    grid = [0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01]
    best, best_mean = 0.0, -np.inf
    for th in grid:
        rs = []
        for sym in chosen:
            f = frames[sym]
            k = split_at(f)
            rs.append(trade(f.iloc[:k], th, cost_bp,
                            long_only=not shortable[sym]))
        pooled = pd.concat(rs)
        live = pooled[pooled != 0]
        m = float(live.mean()) if len(live) > 50 else -np.inf
        flag = ""
        if m > best_mean:
            best, best_mean, flag = th, m, "  <- best"
        print(f"  threshold {th:.4f}   trades {len(live):5d}   "
              f"mean {m * 10_000:+7.2f} bp{flag}")

    print(f"\n  FROZEN: threshold = {best:.4f}")
    print("  Chosen to maximise in-sample mean return, i.e. fitted in its")
    print("  own favour. That is the point - the holdout now has to survive it.")

    # ---------- STEP 3: the holdout, touched once ----------
    print("\nSTEP 3  held-out final third, rule frozen\n")
    rows = []
    for sym in chosen:
        f = frames[sym]
        k = split_at(f)
        lo = not shortable[sym]
        for label, block in (("in-sample", f.iloc[:k]), ("HELD OUT", f.iloc[k:])):
            r = trade(block, best, cost_bp, long_only=lo)
            s = stats_for(r)
            s.update(symbol=sym, block=label, long_only=lo,
                     start=str(block.index[0].date()),
                     end=str(block.index[-1].date()))
            rows.append(s)

    out = pd.DataFrame(rows)
    cols = ["symbol", "block", "start", "end", "trades", "win_rate",
            "mean_bp", "sharpe", "t_stat", "p_value"]
    print(out[cols].round(4).to_string(index=False))

    # ---------- pooled verdict ----------
    print("\nPOOLED\n")
    for label in ("in-sample", "HELD OUT"):
        rs = []
        for sym in chosen:
            f = frames[sym]
            k = split_at(f)
            block = f.iloc[:k] if label == "in-sample" else f.iloc[k:]
            rs.append(trade(block, best, cost_bp,
                            long_only=not shortable[sym]))
        s = stats_for(pd.concat(rs))
        print(f"  {label:10s} trades {s.get('trades', 0):5d}  "
              f"win {s.get('win_rate', float('nan')):.2%}  "
              f"mean {s.get('mean_bp', float('nan')):+6.2f} bp  "
              f"Sharpe {s.get('sharpe', float('nan')):+.2f}  "
              f"t {s.get('t_stat', float('nan')):+.2f}  "
              f"p {s.get('p_value', float('nan')):.3f}")

    print("\n  A positive holdout here would be weak evidence: one split, one")
    print("  pass, a few hundred trades. A negative one is stronger, because")
    print("  the rule was fitted in its own favour and still failed.")
    print("\n  BACKTEST-ONLY. Not a recommendation to trade.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
