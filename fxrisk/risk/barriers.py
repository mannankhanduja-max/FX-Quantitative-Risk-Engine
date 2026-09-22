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


def _is_seq(x) -> bool:
    """A per-bar cost series, as opposed to one flat number."""
    return hasattr(x, "__len__") and not isinstance(x, (str, bytes))


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
        "win_rate_barrier": wr_barrier,
        "cost_R": cost_r,
        "breakeven_wr": be,
        "gap_vs_breakeven": wr_barrier - be,
        "time_exit_share": float((~at_barrier).mean()),
        "mean_net_R": float(trades["net_R"].mean()),
        "total_net_R": float(trades["net_R"].sum()),
        "target_hits": int((trades["outcome"] == "target").sum()),
        "stop_hits": int((trades["outcome"] == "stop").sum()),
        "time_exits": int((trades["outcome"] == "time").sum()),
        "ambiguous_share": float(trades["ambiguous"].mean()),
        "mean_days": float(trades["days_held"].mean()),
    }


# ------------------------------------------------------------
# Explicit brackets, with an optional breakeven stop
#
# `walk` above places the stop at k*sigma from entry, which is
# what a volatility-scaled bracket does. A breakout/retracement
# rule does something different: it puts the stop at a price the
# market has already printed - the retracement extreme - so the
# distance is whatever that structure says it is, and it differs
# trade by trade. `walk_explicit` takes those levels as given.
#
# It also implements the move-to-breakeven rule, which `walk`
# deliberately does not, because on daily bars it cannot be
# honest: a single daily bar can contain the breakeven trigger,
# a run to the target and a collapse back through entry, and no
# amount of care recovers the order they happened in. On 15
# minute bars the ambiguity is smaller but not gone, and it is
# resolved the same way it is everywhere else in this file -
# against the trade.
# ------------------------------------------------------------


def walk_explicit(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    rr: float = 2.0,
    max_bars: int = 40,
    cost_bp: float = 1.0,
    breakeven_frac: float | None = None,
    allow_overlap: bool = False,
) -> pd.DataFrame:
    """
    Walk a list of already-decided entries to their exits.

    Parameters
    ----------
    entries
        One row per setup, with columns `bar` (integer position
        of the entry bar in `bars`), `side` (+1/-1), `entry`
        (fill price) and `stop` (initial stop price). Rows must
        be in ascending `bar` order.
    rr
        Target distance as a multiple of the initial stop
        distance. The target never moves.
    breakeven_frac
        Once price has travelled this fraction of the entry price
        in the trade's favour, the stop moves to entry. None
        disables the rule.
    allow_overlap
        False (the default) drops any entry that would open while
        the previous trade is still running. This is not a detail.
        A rule that fires on consecutive bars during a trend
        produces entries that overlap, and counting each at one
        unit of risk silently levers the book: three overlapping
        trades is three units at risk, not one, so the R total
        describes a position size the stated risk never permitted.
        Dropped entries are counted in the result's attrs.

    WHAT THE BREAKEVEN STOP ACTUALLY DOES
    --------------------------------------
    It is usually sold as risk reduction, and it is - but it is
    not free, and the cost is not where people look for it. It
    converts some losers into scratches, which raises the average
    outcome of the losing tail. It also converts some winners
    into scratches, because a trade that would have gone to
    target now gets stopped at entry on the retest, and those
    were full +rr outcomes. Whether the trade is worth making
    depends on how often price that travels `breakeven_frac`
    comes back through entry before reaching the target, which is
    a property of the instrument, not of the rule.

    The output carries `be_armed` and `outcome == "breakeven"` so
    both sides of that trade can be counted rather than assumed.
    A run with `breakeven_frac=None` is the control.

    ONE HONEST LIMIT. Within the bar that arms the stop, the
    trigger and a subsequent move back to entry are indis-
    tinguishable. This function arms the stop at the CLOSE of the
    triggering bar, never inside it, so a bar that touches the
    trigger and reverses in the same fifteen minutes takes the
    original stop. That is the pessimistic reading and it makes
    the breakeven rule look slightly worse than a tick-level
    simulation would.
    """
    for col in ("High", "Low", "Close"):
        if col not in bars.columns:
            raise ValueError(f"bars is missing the '{col}' column")
    if rr <= 0:
        raise ValueError("rr must be positive")
    if max_bars < 1:
        raise ValueError("max_bars must be at least 1")
    if breakeven_frac is not None and breakeven_frac <= 0:
        raise ValueError("breakeven_frac must be positive or None")

    if entries.empty:
        return pd.DataFrame()

    high = bars["High"].to_numpy(dtype="float64")
    low = bars["Low"].to_numpy(dtype="float64")
    close = bars["Close"].to_numpy(dtype="float64")
    idx = bars.index
    n = len(idx)

    out = []
    dropped = 0
    next_free = -1
    for row in entries.itertuples(index=False):
        i = int(row.bar)
        if not allow_overlap and i < next_free:
            dropped += 1
            continue
        side = float(row.side)
        entry = float(row.entry)
        stop0 = float(row.stop)

        stop_d = abs(entry - stop0)
        if stop_d <= 0 or i >= n - 1:
            continue

        target = entry + side * stop_d * rr
        stop = stop0
        be_armed = False
        be_trigger = (
            entry + side * breakeven_frac * entry
            if breakeven_frac is not None
            else None
        )

        exit_px, held, ambiguous, outcome = None, 0, False, "time"

        for j in range(i + 1, min(i + 1 + max_bars, n)):
            held = j - i
            hi, lo, cl = high[j], low[j], close[j]

            hit_t = hi >= target if side > 0 else lo <= target
            hit_s = lo <= stop if side > 0 else hi >= stop

            if hit_t and hit_s:
                ambiguous = True
                outcome = "breakeven" if be_armed else "stop"
                exit_px = stop
                break
            if hit_s:
                outcome = "breakeven" if be_armed else "stop"
                exit_px = stop
                break
            if hit_t:
                outcome, exit_px = "target", target
                break

            # Arm at the close, never inside the bar. See the
            # docstring - this is the pessimistic reading.
            if be_trigger is not None and not be_armed:
                reached = cl >= be_trigger if side > 0 else cl <= be_trigger
                if reached:
                    be_armed, stop = True, entry

        if exit_px is None:
            exit_px = close[min(i + max_bars, n - 1)]

        next_free = i + max(held, 1)

        gross_r = side * (exit_px - entry) / stop_d
        # cost_bp may be a per-bar series: the spread is not a
        # constant, and a rule that fires on news prints and at
        # the rollover pays more there. See fxrisk/risk/spread.py.
        cbp = float(cost_bp[i]) if _is_seq(cost_bp) else float(cost_bp)
        cost_r = (2 * cbp / 10_000) * entry / stop_d

        out.append(
            {
                "entry_time": idx[i],
                "side": side,
                "entry": entry,
                "stop0": stop0,
                "target": target,
                "exit": exit_px,
                "outcome": outcome,
                "bars_held": held,
                "be_armed": be_armed,
                "ambiguous": ambiguous,
                "stop_frac": stop_d / entry,
                "cost_R": cost_r,
                "gross_R": gross_r,
                "unit_net_R": gross_r - cost_r,
                "net_R": gross_r - cost_r,
            }
        )

    frame = pd.DataFrame(out)
    frame.attrs["dropped_overlapping"] = dropped
    return frame


def summarise_explicit(trades: pd.DataFrame, rr: float) -> dict:
    """
    Win rate against the cost-adjusted hurdle, plus what the
    breakeven stop did to both tails.

    `win_rate` counts net R above zero, so a breakeven exit that
    still paid the round trip counts as a loss - which it is.

    THE HURDLE COMPARISON IS ONLY VALID ON BARRIER EXITS
    -----------------------------------------------------
    `breakeven_wr` is derived from p*rr - (1-p) - cost_R = 0,
    which assumes every trade pays either +rr or -1. A TIME exit
    pays neither: it closes at the market, somewhere in between.

    So the moment time exits are a meaningful share of the
    sample, comparing the overall win rate to the hurdle is
    comparing two different things, and it flatters the strategy
    - a time exit closing at +0.05R counts as a "win" while
    paying a fortieth of what a win is assumed to pay. Widening
    the stop makes this worse, because the target moves further
    away and more trades run out of time.

    That is not hypothetical. On this repository's own data at a
    4-sigma stop floor, 35% of trades were time exits, the win
    rate read 42.1% against a 36.5% hurdle - apparently a healthy
    edge - while mean net R was NEGATIVE at -0.047. The win rate
    was measuring one thing and the hurdle another.

    `win_rate_barrier` restricts to trades that actually
    resolved at a barrier, which is the only population the
    hurdle describes. `time_exit_share` says how much of the
    sample the headline figure is papering over. When that share
    is large, read `mean_net_R` and the t-statistic instead; the
    win rate has stopped being a summary of anything.
    """
    if trades.empty:
        return {"trades": 0}

    cost_r = float(trades["cost_R"].mean())
    wr = float((trades["unit_net_R"] > 0).mean())
    be = breakeven_win_rate(cost_r, rr)
    armed = trades["be_armed"].astype(bool)

    saved = int(((trades["outcome"] == "breakeven")).sum())
    armed_to_target = int((armed & (trades["outcome"] == "target")).sum())

    at_barrier = trades["outcome"] != "time"
    wr_barrier = (
        float((trades.loc[at_barrier, "unit_net_R"] > 0).mean())
        if at_barrier.any()
        else float("nan")
    )

    return {
        "trades": int(len(trades)),
        "win_rate": wr,
        "win_rate_barrier": wr_barrier,
        "cost_R": cost_r,
        "breakeven_wr": be,
        "gap_vs_breakeven": wr_barrier - be,
        "time_exit_share": float((~at_barrier).mean()),
        "mean_net_R": float(trades["net_R"].mean()),
        "total_net_R": float(trades["net_R"].sum()),
        "target_hits": int((trades["outcome"] == "target").sum()),
        "stop_hits": int((trades["outcome"] == "stop").sum()),
        "be_exits": saved,
        "time_exits": int((trades["outcome"] == "time").sum()),
        "be_armed_share": float(armed.mean()),
        "be_armed_then_target": armed_to_target,
        "ambiguous_share": float(trades["ambiguous"].mean()),
        "mean_bars": float(trades["bars_held"].mean()),
        "mean_stop_frac": float(trades["stop_frac"].mean()),
    }
