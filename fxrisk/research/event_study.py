"""
Signed forward return around events, with honest standard errors.

WHY THIS MODULE EXISTS SEPARATELY FROM THE BARRIER CODE
--------------------------------------------------------
`fxrisk.risk.barriers` answers "which of two levels was touched
first". That is a statement about the PATH. This module answers "where
was price h bars later", which is a statement about DRIFT. They are
different questions and they can disagree: a stop placed beyond a
bar's own extreme interacts with volatility clustering, so a barrier
statistic can beat its driftless benchmark while the mean forward
return is exactly zero.

Nothing here knows about stops, targets, reward ratios or bar limits,
which is the point - none of those can be tuned to improve a number
this module produces.

THE STANDARD ERROR IS THE WHOLE DIFFICULTY
-------------------------------------------
A 24-bar forward return sampled at event times overlaps its
neighbours whenever two events fall within 24 bars, and events
cluster - that is what a setup rule does. Treating each event as an
independent observation inflates t by up to sqrt(h).

The fix is blocking, as in `fxrisk.research.ic`: average within each
(instrument, calendar month) cell and take the t-statistic ACROSS
cells. Cells share at most h bars at a month boundary, so they are
close to independent; the overlap inside a cell affects only how
noisy that cell's mean is, not how many independent observations the
test believes it has.

Cells are equal-weighted, not weighted by event count. A month with
40 events and a month with 4 are one observation each about whether
the effect is present. Weighting by count would let a single busy
month - which is to say a single volatility episode - dominate the
test, which is the failure mode this construction exists to avoid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def signed_forward(bars: pd.DataFrame, entries: pd.DataFrame,
                   horizons=(1, 4, 24, 96)) -> pd.DataFrame:
    """
    Signed forward log return in basis points, per event and horizon.

        y_h = side * (log C[k+h] - log C[k])

    `entries` needs `time` and `side`; every `time` must be an exact
    timestamp of `bars.index`, because an approximate match would
    silently shift the whole measurement by a bar. Events with fewer
    than h bars of future left get NaN rather than a truncated
    horizon - a short horizon quietly substituted for a long one is
    the kind of error that reads as a result.
    """
    out = entries.reset_index(drop=True).copy()
    if out.empty:
        for h in horizons:
            out[f"y{h}"] = pd.Series(dtype="float64")
        return out

    idx = bars.index
    when = pd.DatetimeIndex(out["time"])
    k = idx.searchsorted(when, side="left")
    if (k >= len(idx)).any() or not (idx[np.minimum(k, len(idx) - 1)] == when).all():
        raise ValueError("every entry time must be an exact bar timestamp of "
                         "`bars`; got one that is not, which would shift the "
                         "horizon by a bar")

    logc = np.log(bars["Close"].to_numpy(dtype="float64"))
    side = out["side"].to_numpy(dtype="float64")
    for h in horizons:
        j = k + int(h)
        y = np.full(len(k), np.nan)
        ok = j < len(logc)
        y[ok] = logc[j[ok]] - logc[k[ok]]
        out[f"y{h}"] = side * y * 10_000.0
    return out


def cell_means(events: pd.DataFrame, col: str,
               min_events: int = 3) -> pd.DataFrame:
    """
    Mean of `col` per (instrument, calendar month), for cells with at
    least `min_events` usable events.

    Months are cut in UTC before the timezone is dropped, so a cell
    boundary cannot move with the host's clock.
    """
    need = {"instrument", "time", col}
    missing = need - set(events.columns)
    if missing:
        raise KeyError(f"events is missing {sorted(missing)}")
    d = events[["instrument", "time", col]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["instrument", "month", "n", "mean"])
    ix = pd.DatetimeIndex(d["time"])
    ix = ix.tz_convert("UTC").tz_localize(None) if ix.tz is not None else ix
    d = d.assign(month=ix.to_period("M").astype(str))
    g = (d.groupby(["instrument", "month"])[col]
         .agg(n="count", mean="mean").reset_index())
    return g[g["n"] >= min_events].reset_index(drop=True)


def pooled(cells: pd.DataFrame) -> dict:
    """
    One test from the cells: is the mean cell mean non-zero?

    `share_pos` is the fraction of cells on the same side as the
    pooled mean - a mean of +2bp built from 30% positive cells and one
    enormous outlier is not the same finding as one built from 65%.
    """
    if cells.empty or len(cells) < 3:
        return {"cells": len(cells), "events": int(cells["n"].sum()) if len(cells) else 0,
                "mean_bp": float("nan"), "t": float("nan"),
                "p": float("nan"), "share_pos": float("nan")}
    x = cells["mean"].to_numpy(dtype="float64")
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else float("nan")
    return {"cells": n, "events": int(cells["n"].sum()), "mean_bp": mean,
            "t": float(t), "p": p,
            "share_pos": float((x > 0).mean() if mean > 0 else (x < 0).mean())}
