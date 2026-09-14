"""
Triple-barrier exits: target, stop, and a time limit.

The strategy layer in `strategy.py` holds a position until the
signal flips, which means it has no risk-reward at all - no stop,
no target, nothing to measure an R multiple against. This module
supplies the exits.

Enter on a signal flip, then exit at whichever barrier is touched
first: the target at `rr * d`, the stop at `d`, or a time limit.
The distance `d` is `k * sigma * price`, with sigma a conditional
volatility estimate, so the barriers widen in a crisis and narrow
in a calm instead of sitting at a fixed percentage that is wrong in
both.

THE INTRABAR AMBIGUITY
-----------------------
Daily bars record High and Low but not the order they occurred in.
When both barriers fall inside one day's range there is no way to
know which was touched first, and resolving it optimistically -
target first - is the single most common way a bracket backtest
becomes fiction. This module resolves it as STOP first, which is
pessimistic but never flattering, and counts how often the
ambiguity arose so a reader can judge whether it matters. On the
default universe it affects 0-1% of trades.

THE BREAKEVEN WIN RATE IS NOT 1/(1+rr)
---------------------------------------
The form quoted in most trading material ignores costs and says a
1:1 bracket needs 50%. But the round trip is paid on every trade
regardless of outcome, and expressed in units of risk it is

    cost_R = 2 * cost / stop_distance

so the real hurdle is

    p = (1 + cost_R) / (1 + rr)

which has a consequence that runs against instinct: the hurdle
RISES as the stop tightens. On the default universe a 2bp round
trip at a 1-sigma stop costs 0.073R and lifts the 1:1 breakeven to
53.7%; at 3 sigma it costs 0.025R and the hurdle is 51.2%. A tight
stop is the expensive choice, not the conservative one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ewma_sigma(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """
    EWMA daily volatility, shifted one bar so it is causal.

    The same lambda the RiskMetrics estimator in `models/ewma.py`
    uses. Kept here rather than imported so the barrier walk has a
    single, obvious volatility input.
    """
    r = pd.Series(returns, dtype="float64")
    var = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).shift(1)


@dataclass
class BarrierConfig:
    """Barrier geometry and the cost charged on each round trip."""

    stop_k: float = 3.0
    rr: float = 1.0
    max_days: int = 10
    cost_bp: float = 2.0

    def __post_init__(self) -> None:
        if self.stop_k <= 0:
            raise ValueError("stop_k must be positive")
        if self.rr <= 0:
            raise ValueError("rr must be positive")
        if self.max_days < 1:
            raise ValueError("max_days must be at least 1")
        if self.cost_bp < 0:
            raise ValueError("cost_bp must be non-negative")


def walk(
    bars: pd.DataFrame,
    signal: pd.Series,
    sigma: pd.Series,
    cfg: BarrierConfig | None = None,
    allow: pd.Series | None = None,
    size: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Walk the bars and return one row per closed trade.

    Parameters
    ----------
    signal
        Already shifted, per the contract in `fxrisk.indicators`.
        A trade opens where the signal flips to a non-zero value.
    allow
        Optional entry gate - a filter on WHICH trades are taken.
    size
        Optional stake multiplier. It scales the R outcome but
        cannot change which barrier is hit, so any variant that
        only changes `size` has the same win rate by construction.
        A result claiming otherwise has a bug.
    """
    cfg = cfg or BarrierConfig()
    for col in ("High", "Low", "Close"):
        if col not in bars.columns:
            raise ValueError(f"bars is missing the '{col}' column")

    close, high, low = bars["Close"], bars["High"], bars["Low"]
    idx = bars.index
    n = len(idx)

    sig = pd.Series(signal, dtype="float64").reindex(idx).fillna(0.0)
    sg = pd.Series(sigma, dtype="float64").reindex(idx)
    gate = (
        pd.Series(True, index=idx)
        if allow is None
        else pd.Series(allow).reindex(idx).fillna(False).astype(bool)
    )
    stake = (
        pd.Series(1.0, index=idx)
        if size is None
        else pd.Series(size, dtype="float64").reindex(idx).fillna(1.0)
    )

    flips = sig.diff().fillna(sig).ne(0) & sig.ne(0)

    trades = []
    i = 0
    while i < n - 1:
        if (
            not bool(flips.iloc[i])
            or not bool(gate.iloc[i])
            or not np.isfinite(sg.iloc[i])
            or sg.iloc[i] <= 0
        ):
            i += 1
            continue

        side = float(np.sign(sig.iloc[i]))
        entry = float(close.iloc[i])
        stop_d = cfg.stop_k * float(sg.iloc[i]) * entry
        target = entry + side * stop_d * cfg.rr
        stop = entry - side * stop_d

        exit_px, held, ambiguous, outcome = None, 0, False, "time"
        for j in range(i + 1, min(i + 1 + cfg.max_days, n)):
            held = j - i
            hi, lo = float(high.iloc[j]), float(low.iloc[j])
            hit_t = hi >= target if side > 0 else lo <= target
            hit_s = lo <= stop if side > 0 else hi >= stop

            if hit_t and hit_s:
                # Order unknowable inside one bar - resolve against
                # the trade rather than for it.
                ambiguous, outcome, exit_px = True, "stop", stop
                break
            if hit_s:
                outcome, exit_px = "stop", stop
                break
            if hit_t:
                outcome, exit_px = "target", target
                break

        if exit_px is None:
            exit_px = float(close.iloc[min(i + cfg.max_days, n - 1)])

        gross_r = side * (exit_px - entry) / stop_d
        cost_r = (2 * cfg.cost_bp / 10_000) * entry / stop_d
        mult = float(stake.iloc[i]) if np.isfinite(stake.iloc[i]) else 1.0

        trades.append(
            {
                "entry_date": idx[i],
                "side": side,
                "entry": entry,
                "exit": exit_px,
                "outcome": outcome,
                "days_held": held,
                "ambiguous": ambiguous,
                "cost_R": cost_r,
                "unit_net_R": gross_r - cost_r,
                "net_R": (gross_r - cost_r) * mult,
                "stake": mult,
            }
        )
        i += max(held, 1)

    return pd.DataFrame(trades)


def breakeven_win_rate(cost_r: float, rr: float) -> float:
    """
    Win rate needed to break even, INCLUDING costs.

        p * rr - (1 - p) * 1 - cost_R = 0
        p = (1 + cost_R) / (1 + rr)

    Passing cost_r = 0 recovers the naive 1/(1+rr) that most
    trading material quotes.
    """
    if rr <= 0:
        raise ValueError("rr must be positive")
    return (1.0 + cost_r) / (1.0 + rr)


def summarise(trades: pd.DataFrame, rr: float) -> dict:
    """Win rate against the cost-adjusted hurdle, and the R totals."""
    if trades.empty:
        return {"trades": 0}

    cost_r = float(trades["cost_R"].mean())
    wr = float((trades["unit_net_R"] > 0).mean())
    be = breakeven_win_rate(cost_r, rr)

    return {
        "trades": int(len(trades)),
        "win_rate": wr,
        "cost_R": cost_r,
        "breakeven_wr": be,
        "gap_vs_breakeven": wr - be,
        "mean_net_R": float(trades["net_R"].mean()),
        "total_net_R": float(trades["net_R"].sum()),
        "target_hits": int((trades["outcome"] == "target").sum()),
        "stop_hits": int((trades["outcome"] == "stop").sum()),
        "time_exits": int((trades["outcome"] == "time").sum()),
        "ambiguous_share": float(trades["ambiguous"].mean()),
        "mean_days": float(trades["days_held"].mean()),
    }
