"""
Do the 'high volatility' calendar dates actually show high volatility?

Before conditioning a strategy on event dates, test the premise.
Only rule-derivable dates are used here, because those are exact:

  nfp          first Friday of the month (BLS employment report)
  opex         third Friday (index/option expiry, 'triple witching'
               on quarter-end months)
  month_end    last trading day of the month
  quarter_end  last trading day of March/June/September/December
  turn         last or first trading day of a month

FOMC, CPI, ECB and BoE dates are NOT derivable from a rule - they
are published schedules that move. They are deliberately absent
rather than approximated, because an approximated event date puts
the flag on the wrong day and then measures nothing. The module
that follows this can load them from a CSV.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from fxrisk.data import yahoo
from fxrisk.calendar import flags


def probe(sym: str) -> pd.DataFrame:
    bars = yahoo.load_symbol(sym)
    ret = bars["Close"].pct_change(fill_method=None)
    absret = ret.abs()
    # Parkinson range: a cleaner intraday vol proxy than |close-close|
    rng = np.log(bars["High"] / bars["Low"])

    fl = flags(bars.index)
    rows = []
    for name in fl.columns:
        on = fl[name]
        a, b = absret[on].dropna(), absret[~on].dropna()
        ra, rb = rng[on].dropna(), rng[~on].dropna()
        if len(a) < 30:
            continue
        # Mann-Whitney: no normality assumption on |returns|
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        _, pr = stats.mannwhitneyu(ra, rb, alternative="two-sided")
        rows.append({
            "symbol": sym,
            "event": name,
            "n_event": len(a),
            "mean_abs_ret": a.mean(),
            "mean_abs_other": b.mean(),
            "ratio_ret": a.mean() / b.mean(),
            "p_ret": p,
            "ratio_range": ra.mean() / rb.mean(),
            "p_range": pr,
        })
    return pd.DataFrame(rows)


def main() -> int:
    out = pd.concat([probe(i.yahoo) for i in config.UNIVERSE],
                    ignore_index=True)

    print("Is |return| higher on these dates? ratio > 1 means yes.\n")
    piv = out.pivot(index="event", columns="symbol", values="ratio_ret")
    print(piv.round(3).to_string())

    print("\nMann-Whitney p-values on |return|:\n")
    pp = out.pivot(index="event", columns="symbol", values="p_ret")
    print(pp.round(3).to_string())

    print("\nSame, on the Parkinson high-low range:\n")
    pr = out.pivot(index="event", columns="symbol", values="ratio_range")
    print(pr.round(3).to_string())

    print("\nPooled across instruments:\n")
    agg = out.groupby("event").agg(
        n=("n_event", "sum"),
        mean_ratio_ret=("ratio_ret", "mean"),
        mean_ratio_range=("ratio_range", "mean"),
        significant_ret=("p_ret", lambda s: int((s < 0.05).sum())),
        significant_range=("p_range", lambda s: int((s < 0.05).sum())),
    )
    agg["of_instruments"] = len(config.UNIVERSE)
    print(agg.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
