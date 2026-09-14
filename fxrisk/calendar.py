"""
Calendar flags for dates that might carry unusual volatility.

Only RULE-DERIVABLE dates live here, because those are exact:

    nfp          first Friday of the month (BLS employment report)
    opex         third Friday (index and option expiry)
    month_end    last trading day of the month
    quarter_end  last trading day of March/June/September/December
    turn         last or first trading day of a month

FOMC, CPI, ECB and BoE meeting dates are deliberately ABSENT. They
are published schedules that move, not rules, and a flag placed on
an approximated date lands on the wrong day and then measures
nothing. `load_event_csv` takes them from a real calendar instead.

MEASURED, NOT ASSUMED
----------------------
Ratios of mean |return| on flagged days against all other days,
across the five ETFs in the default universe:

    nfp          1.184   significant on 5 of 5 instruments
    turn         1.085   3 of 5
    month_end    1.040   0 of 5  (but 4 of 5 on the high-low range)
    quarter_end  0.975   0 of 5
    opex         0.926   0 of 5

Two things worth carrying forward. First Fridays really do carry
about 18% more movement, confirmed independently on the Parkinson
range. And option expiry is a CALM day for these instruments, which
is the opposite of the folklore - if you build a rule around opex
volatility here, you are trading noise with the sign flipped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FLAG_NAMES = ("nfp", "opex", "month_end", "quarter_end", "turn")


def flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Rule-derivable calendar flags on a trading-day index.

    Counted over the trading days actually present, so a first
    Friday that fell on a holiday does not silently shift the
    count onto the following week.
    """
    idx = pd.DatetimeIndex(index)
    s = pd.Series(idx, index=idx)
    ym = s.dt.to_period("M")

    fridays = s.dt.dayofweek == 4
    friday_rank = fridays.groupby(ym).cumsum().where(fridays)

    last_day = s.groupby(ym).transform("max") == s
    first_day = s.groupby(ym).transform("min") == s
    q_month = s.dt.month.isin([3, 6, 9, 12])

    return pd.DataFrame(
        {
            "nfp": (friday_rank == 1).fillna(False),
            "opex": (friday_rank == 3).fillna(False),
            "month_end": last_day,
            "quarter_end": last_day & q_month,
            "turn": last_day | first_day,
        },
        index=idx,
    )


def load_event_csv(
    path: str | Path, index: pd.DatetimeIndex, column: str = "event"
) -> pd.DataFrame:
    """
    Load published event dates that no rule can derive.

    The CSV needs a `date` column and an `event` column:

        date,event
        2026-01-28,fomc
        2026-01-13,cpi

    Returns one boolean column per distinct event name, reindexed
    onto `index`. Dates outside the index are dropped silently -
    a calendar usually spans more than the price history.
    """
    frame = pd.read_csv(path, parse_dates=["date"])
    if column not in frame.columns:
        raise ValueError(f"CSV has no '{column}' column; got {list(frame.columns)}")

    idx = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=idx)
    for name, group in frame.groupby(column):
        hits = pd.DatetimeIndex(group["date"]).normalize()
        out[str(name)] = idx.normalize().isin(hits)
    return out


def summary(index: pd.DatetimeIndex) -> str:
    """How many days each flag covers - a sanity check, not a result."""
    fl = flags(index)
    lines = [f"  {len(fl)} trading days, {fl.index[0].date()} to {fl.index[-1].date()}"]
    for name in FLAG_NAMES:
        n = int(fl[name].sum())
        lines.append(f"  {name:12s} {n:5d} days  ({n / len(fl):.1%})")
    return "\n".join(lines)
