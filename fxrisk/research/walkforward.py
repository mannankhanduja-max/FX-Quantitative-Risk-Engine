"""
Rolling walk-forward: the only framework in which searching many
configurations is not self-defeating.

THE PROBLEM THIS SOLVES
-------------------------
This repository measured roughly 150 configurations on one two-year
window and then froze the best of them for a single out-of-sample
test. That test failed. So did the second one, on a different
geometry. The reason is structural rather than bad luck: with one
development sample and one shot, the sample is exhausted after the
first look, and every subsequent "best configuration" is chosen with
knowledge of what already failed.

Walk-forward inverts the accounting. The timeline is cut into
consecutive folds. For each fold the configuration is chosen using
ONLY the training window, then scored on the test window that
immediately follows and is never used for selection. The reported
result is the concatenation of the test segments alone. Searching a
large grid is then not only allowed but the point - the search
happens inside each training window, and the score is always on data
the search could not see.

WHAT IT CANNOT FIX
-------------------
It does not manufacture data. Five years at this trade frequency is
six or seven non-overlapping test folds, so the out-of-sample sample
is a few hundred trades, not thousands. A weak effect will still be
unresolvable; walk-forward just stops it from LOOKING resolved.

It also does not protect against a grid designed after seeing the
answer. The grid below was fixed from the parameters this study
already varied, and widening it later because the result disappoints
would reintroduce exactly the bias this construction removes.

THE TWO DIAGNOSTICS THAT MATTER MORE THAN THE HEADLINE
--------------------------------------------------------
`selection_stability` - how often the training window picks the same
configuration. A rule whose optimal parameters change every fold is
not a rule, it is a curve fit that has learned the last window.

`vs_fixed` - the same test segments scored with ONE configuration
held constant throughout. If adaptive selection does not beat a
fixed rule out of sample, the selection step is adding nothing and
should be dropped. This comparison is the honest test of whether
optimisation helps at all, and it is usually the one that is left
out.

BACKTEST-ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fold:
    """One train/test pair, with no overlap between them."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __post_init__(self) -> None:
        if not self.train_start < self.train_end <= self.test_start < self.test_end:
            raise ValueError(
                "folds must satisfy train_start < train_end <= test_start "
                "< test_end; anything else leaks the test window into "
                "selection")


def make_folds(start, end, train_months: int = 18, test_months: int = 6,
               step_months: int | None = None, tz=None) -> list[Fold]:
    """
    Consecutive rolling folds across [start, end].

    `step_months` defaults to `test_months`, which makes the test
    segments exactly adjacent and non-overlapping - so concatenating
    them is a real equity curve rather than a double-count.

    `tz` localises the boundaries. Trade timestamps in this
    repository are tz-aware because the session boundary is an
    FX-market boundary, so passing the same zone keeps fold edges and
    session edges talking about the same instant.
    """
    if train_months < 1 or test_months < 1:
        raise ValueError("train_months and test_months must be positive")
    # `step_months or test_months` would turn an explicit 0 into the
    # default, because 0 is falsy - and a zero step is an infinite
    # loop, not a default.
    step = test_months if step_months is None else step_months
    if step < 1:
        raise ValueError("step_months must be positive")

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if tz is not None:
        start = start.tz_localize(tz) if start.tzinfo is None else start.tz_convert(tz)
        end = end.tz_localize(tz) if end.tzinfo is None else end.tz_convert(tz)
    folds, cursor = [], start
    while True:
        tr_end = cursor + pd.DateOffset(months=train_months)
        te_end = tr_end + pd.DateOffset(months=test_months)
        if te_end > end:
            break
        folds.append(Fold(cursor, tr_end, tr_end, te_end))
        cursor = cursor + pd.DateOffset(months=step)
    return folds


def _slice(trades: pd.DataFrame, lo, hi) -> pd.DataFrame:
    """
    Half-open [lo, hi) on entry time, tolerant of timezone.

    Trade timestamps here are tz-aware (the session boundary is an
    FX-market boundary) while fold bounds are naturally written naive.
    Comparing the two raises rather than silently misaligning, which
    is the right behaviour but has to be handled rather than hit.
    """
    t = trades["entry_time"]
    tz = getattr(t.dtype, "tz", None)
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    if tz is not None:
        lo = lo.tz_localize(tz) if lo.tzinfo is None else lo.tz_convert(tz)
        hi = hi.tz_localize(tz) if hi.tzinfo is None else hi.tz_convert(tz)
    elif lo.tzinfo is not None:
        lo, hi = lo.tz_localize(None), hi.tz_localize(None)
    return trades[(t >= lo) & (t < hi)]


def select(trades: pd.DataFrame, fold: Fold, objective) -> object:
    """
    Best configuration on the TRAINING window only.

    `trades` carries a `config` column; the objective scores one
    config's training trades and higher is better. Ties break on the
    lexicographically smallest config key, so the choice is
    deterministic and a reordered grid cannot change the result.
    """
    tr = _slice(trades, fold.train_start, fold.train_end)
    if tr.empty:
        return None
    scores = {}
    for cfg, g in tr.groupby("config"):
        s = objective(g)
        if np.isfinite(s):
            scores[cfg] = s
    if not scores:
        return None
    best = max(scores.values())
    return sorted(k for k, v in scores.items() if v == best)[0]


def mean_net_r(g: pd.DataFrame) -> float:
    """Default objective. Refuses to score a window too thin to mean anything."""
    if len(g) < 30:
        return float("-inf")
    return float(g["unit_net_R"].mean())


def run(trades: pd.DataFrame, folds: list[Fold], objective=mean_net_r,
        fixed_config=None) -> dict:
    """
    Walk the folds and return ONLY out-of-sample test results.

    `trades` must hold every configuration's trades over the whole
    period, with columns `config`, `entry_time` and `unit_net_R`.
    Generating them once and slicing here is equivalent to refitting
    per fold - the rule is deterministic given its parameters - and
    it is far cheaper.
    """
    picked, oos, fixed_oos = [], [], []
    for f in folds:
        cfg = select(trades, f, objective)
        if cfg is None:
            continue
        te = _slice(trades[trades["config"] == cfg], f.test_start, f.test_end)
        if not te.empty:
            seg = te.copy()
            seg["fold_test_start"] = f.test_start
            seg["picked_config"] = cfg
            oos.append(seg)
        picked.append({"test_start": f.test_start, "test_end": f.test_end,
                       "config": cfg, "n_test": len(te)})
        if fixed_config is not None:
            fx = _slice(trades[trades["config"] == fixed_config],
                        f.test_start, f.test_end)
            if not fx.empty:
                fixed_oos.append(fx)

    out = {
        "folds": len(folds),
        "picks": pd.DataFrame(picked),
        "oos": (pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()),
        "fixed_oos": (pd.concat(fixed_oos, ignore_index=True)
                      if fixed_oos else pd.DataFrame()),
    }
    if picked:
        chosen = [p["config"] for p in picked]
        most = max(set(chosen), key=chosen.count)
        out["selection_stability"] = chosen.count(most) / len(chosen)
        out["most_chosen"] = most
        out["distinct_configs"] = len(set(chosen))
    return out


def summarise(res: dict) -> dict:
    """Out-of-sample economics, and whether selection beat a fixed rule."""
    o = res.get("oos", pd.DataFrame())
    if o.empty:
        return {"trades": 0}
    r = o["unit_net_R"]
    d = {
        "trades": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "mean_net_R": float(r.mean()),
        "t": float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
        if len(r) > 2 and r.std(ddof=1) > 0 else float("nan"),
        "selection_stability": res.get("selection_stability", float("nan")),
        "distinct_configs": res.get("distinct_configs", 0),
    }
    f = res.get("fixed_oos", pd.DataFrame())
    if not f.empty:
        d["fixed_mean_net_R"] = float(f["unit_net_R"].mean())
        d["selection_edge"] = d["mean_net_R"] - d["fixed_mean_net_R"]
    return d
