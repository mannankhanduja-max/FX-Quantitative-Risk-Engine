"""
A daily-horizon reversion rule, built on the two signals that
confirmed out of sample.

WHY THIS EXISTS, AFTER EVERYTHING ELSE FAILED
-----------------------------------------------
The intraday breakout/fade family is finished. Roughly 140
configurations, every timeframe and reward ratio, and the best of
them failed its pre-registered one-shot test outright: z +3.35 in
sample became z -0.31 out of it (see `prior_period_mtf.py`).

What did NOT fail is the signal study. Twelve of twenty-four rank-IC
tests survived Holm correction in discovery and ALL TWELVE confirmed
on a held-out period, most of them stronger. Two of those clear the
2bp round trip at a 24-hour horizon:

    24h reversal           IC -0.058 -> -0.083,   7.3 -> 8.8 bp
    prior-session POC      IC +0.074 -> +0.081,  10.3 ->  8.0 bp

    (quintile spread, top minus bottom, per trade)

That is the ONLY thing in this repository that has survived a test
it could have failed, and it is why this module exists rather than
another variant of the last one.

THE ONE THING THAT IS DIFFERENT, AND IT IS THE POINT
------------------------------------------------------
Frequency. Every rule tested before this one traded 1,600-5,700
times a year and had to pay the spread each time. An 8bp effect
cannot survive a 7bp round trip at that rate - that was the binding
constraint in every single negative result, not the entry logic.

This rule trades at most ONCE PER INSTRUMENT PER DAY and holds for
24 hours. Same spread, roughly a tenth the frequency, so the same
effect has ten times as much room. Whether that is enough is what
the measurement decides; the point is that it is the first version
where the arithmetic is not hopeless before it starts.

WHAT A 24-HOUR HOLD COSTS THAT AN INTRADAY ONE DOES NOT
---------------------------------------------------------
Swap. Holding overnight means paying or receiving the interest rate
differential, and over 2024-26 that was large: roughly 4% annual on
USD/JPY, 1.5% on EUR/USD, and index financing around 4.5%. A single
night at 4% is about 1.1bp of notional - comparable to the entire
spread. It is charged here explicitly, in `swap_bp`, and a rule
that only works with swap set to zero is not a rule.

THE SIGNALS, UNCHANGED FROM THE STUDY
---------------------------------------
Taken from `fxrisk.research.signals` exactly as measured - not
re-tuned, not re-parameterised. Re-optimising them here would throw
away the out-of-sample confirmation that is their entire claim to
attention.

    mom_24h   past 24-hour return, scaled by sigma
    poc_rev   negated distance from the prior session's point of
              control, in sigma units

A SIGN TRAP THAT COSTS A WHOLE RESULT IF MISSED
-------------------------------------------------
These two features are NOT oriented the same way in
`research/signals.py`, and the study reported ICs accordingly:

    mom_24h   IC -0.058   the feature is raw past return, so a
                          NEGATIVE IC is the reversal finding:
                          price that rose tends to fall back.
                          As a TRADING signal it must be FLIPPED.
    poc_rev   IC +0.074   the feature is already negated at source,
                          so a POSITIVE IC means it is right as
                          written. Used as-is.

Taking both at face value points one leg at momentum and the other
at reversion: their rank correlation is -0.72 and they cancel, which
is exactly what the first version of this module did. Oriented by
their measured IC sign the same pair correlates +0.74, because both
are then saying the same thing - price is extended relative to an
anchor, expect it back. `SIGNAL_SIGN` encodes that, and a test
asserts the flip is applied.

BACKTEST-ONLY. Not a recommendation to trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id
from fxrisk.research import signals as sg

# Orientation from the signal study's measured IC sign. A feature
# whose IC was negative is right about the direction and wrong about
# the sign, so trading it means flipping it.
SIGNAL_SIGN: dict[str, float] = {
    "mom_1h": -1.0, "mom_4h": -1.0, "mom_24h": -1.0,
    "vwap_rev": +1.0, "poc_rev": +1.0,
    "flow_proxy": +1.0, "ldn_drive": +1.0, "ny_drive": +1.0,
}


@dataclass
class ReversionConfig:
    """How the confirmed signals become a position."""

    signals: tuple[str, ...] = ("mom_24h", "poc_rev")
    hold_bars: int = 24           # on 1h bars: one day
    entry_quantile: float = 0.20  # trade the top/bottom fifth
    rank_window: int = 500        # trailing bars the rank is taken over
    min_rank_obs: int = 200
    one_per_day: bool = True      # at most one entry per session
    decision_hour: int | None = 11   # ET hour to decide at; None = first
                                     # qualifying bar of the session
    cost_bp: float = 1.0          # per side, on top of nothing
    swap_bp_per_night: float = 0.0

    def __post_init__(self) -> None:
        if not self.signals:
            raise ValueError("need at least one signal")
        for s in self.signals:
            if s not in sg.FEATURES:
                raise ValueError(f"unknown signal {s!r}; known: {sg.FEATURES}")
        if self.hold_bars < 1:
            raise ValueError("hold_bars must be at least 1")
        if not 0 < self.entry_quantile <= 0.5:
            raise ValueError("entry_quantile must be in (0, 0.5]")
        if self.rank_window < 50:
            raise ValueError("rank_window must be at least 50")
        if self.min_rank_obs < 20 or self.min_rank_obs > self.rank_window:
            raise ValueError("need 20 <= min_rank_obs <= rank_window")
        if self.decision_hour is not None and not 0 <= self.decision_hour <= 23:
            raise ValueError("decision_hour must be an hour of the day or None")
        if self.cost_bp < 0 or self.swap_bp_per_night < 0:
            raise ValueError("costs cannot be negative")


def _trailing_rank(x: pd.Series, window: int, min_obs: int) -> pd.Series:
    """
    Each value's percentile within the PRECEDING `window` values.

    Trailing, never full-sample. A full-sample rank would tell the
    bar where it sits among values that had not happened yet, which
    is the quiet way this kind of rule acquires a spectacular
    backtest. The current value is included in its own window -
    that is legitimate, since it is known at the close - but no
    later value ever is.
    """
    return x.rolling(window, min_periods=min_obs).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True)


def score(bars_1h: pd.DataFrame,
          cfg: ReversionConfig | None = None) -> pd.Series:
    """
    Combined signal in [0, 1], where 1 is the strongest long claim.

    The two features are ranked separately and averaged. Averaging
    RANKS rather than raw values is deliberate: `mom_24h` is a log
    return and `poc_rev` is a sigma distance, so averaging them raw
    would weight them by their units rather than by their
    information.
    """
    cfg = cfg or ReversionConfig()
    feats = sg.compute(bars_1h)
    parts = []
    for name in cfg.signals:
        if name not in feats.columns:
            raise ValueError(f"{name!r} missing from computed features")
        oriented = feats[name] * SIGNAL_SIGN[name]
        parts.append(_trailing_rank(oriented, cfg.rank_window,
                                    cfg.min_rank_obs))
    out = pd.concat(parts, axis=1).mean(axis=1)
    return out.rename("score")


def positions(bars_1h: pd.DataFrame,
              cfg: ReversionConfig | None = None) -> pd.DataFrame:
    """
    One row per trade: enter at this bar's close, exit `hold_bars`
    later at that bar's close.

    No stop and no target. That is not an oversight - the signal
    study measured a HORIZON effect, the average return over the
    next 24 hours, and a barrier would be a different bet on a
    different quantity. Adding one here would silently replace the
    thing that was confirmed with something that was not.
    """
    cfg = cfg or ReversionConfig()
    s = score(bars_1h, cfg)
    close = bars_1h["Close"]
    idx = bars_1h.index
    n = len(idx)

    hi, lo = 1.0 - cfg.entry_quantile, cfg.entry_quantile
    sess = pd.Series(session_id(idx).to_numpy(), index=idx)

    # WHICH BAR OF THE DAY THE DECISION IS TAKEN ON
    # ----------------------------------------------
    # Taking the FIRST qualifying bar of a session sounds neutral
    # and is not: sessions open at 17:00 New York, the thinnest and
    # widest-spread hour there is, so "first qualifying"
    # systematically decides in the worst liquidity of the day. A
    # daily-horizon signal has no reason to be read at the open -
    # it is a statement about the next 24 hours, not the next
    # minute - so the decision is taken at one fixed, liquid hour.
    # 11:00 ET is the London/New York overlap.
    et_hour = idx.tz_convert("America/New_York").hour

    rows = []
    last_session = None
    for i in range(n - cfg.hold_bars):
        if cfg.decision_hour is not None and et_hour[i] != cfg.decision_hour:
            continue
        v = s.iloc[i]
        if not np.isfinite(v):
            continue
        side = 1.0 if v >= hi else (-1.0 if v <= lo else 0.0)
        if side == 0.0:
            continue
        if cfg.one_per_day:
            this = sess.iloc[i]
            if this == last_session:
                continue
            last_session = this

        j = i + cfg.hold_bars
        entry, exit_px = float(close.iloc[i]), float(close.iloc[j])
        gross = side * (np.log(exit_px) - np.log(entry)) * 10_000.0   # bp

        nights = _nights_between(idx[i], idx[j])
        cost = 2 * cfg.cost_bp + nights * cfg.swap_bp_per_night

        rows.append({"entry_time": idx[i], "exit_time": idx[j],
                     "side": side, "score": float(v),
                     "entry": entry, "exit": exit_px,
                     "gross_bp": gross, "cost_bp": cost,
                     "net_bp": gross - cost, "nights": nights})

    return pd.DataFrame(rows)


def _fx_day(t) -> int:
    """Index of the FX trading day a timestamp falls in, rolling 17:00 ET."""
    x = pd.Timestamp(t).tz_convert("America/New_York")
    return int((x - pd.Timedelta(hours=17)).floor("D").value // 86_400_000_000_000)


def _nights_between(start, end) -> int:
    """
    Number of 17:00 New York rollovers a hold crosses.

    Swap is charged per roll crossed, so this has to count rolls,
    not elapsed days. A 24-hour hold crosses exactly one unless it
    starts and ends on the same side of 17:00.
    """
    return max(0, _fx_day(end) - _fx_day(start))


def summarise(trades: pd.DataFrame) -> dict:
    """
    Per-trade economics in basis points, with the t-statistic on
    the NET figure - which is the only one that can be traded.
    """
    if trades.empty:
        return {"trades": 0}
    g, n = trades["gross_bp"], trades["net_bp"]
    span_days = (trades["entry_time"].iloc[-1]
                 - trades["entry_time"].iloc[0]).days or 1
    per_year = len(trades) / (span_days / 365.25)
    se = n.std(ddof=1) / np.sqrt(len(n)) if len(n) > 2 else np.nan
    return {
        "trades": int(len(trades)),
        "per_year": float(per_year),
        "win_rate": float((n > 0).mean()),
        "gross_bp": float(g.mean()),
        "cost_bp": float(trades["cost_bp"].mean()),
        "net_bp": float(n.mean()),
        "t": float(n.mean() / se) if se and np.isfinite(se) else float("nan"),
        "sharpe": (float(n.mean() / n.std(ddof=1) * np.sqrt(per_year))
                   if n.std(ddof=1) > 0 else float("nan")),
        "long_share": float((trades["side"] > 0).mean()),
        "mean_nights": float(trades["nights"].mean()),
    }
