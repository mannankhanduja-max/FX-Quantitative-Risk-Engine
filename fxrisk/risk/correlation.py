"""
Pairwise correlation: static, EWMA and DCC side by side.

The point of this module is one comparison. A sample correlation
matrix reports a single number per pair for the whole history. DCC
reports a path. Putting the sample value next to the DCC minimum and
maximum shows, per pair, how much of the relationship the static
number is averaging away.

That range is not a curiosity. Portfolio variance is
w' H w, so a pair whose correlation runs from 0.2 to 0.8 across the
sample carries risk that a single 0.5 cannot express — and the
0.8 regime is reliably the one that coincides with everything else
going wrong.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..models.dcc import DccResult
from ..models.ewma import (
    RISKMETRICS_LAMBDA_DAILY,
    correlation_from_covariance,
    ewma_covariance_last,
)


def sample_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain Pearson correlation over the whole sample."""
    return returns.replace([np.inf, -np.inf], np.nan).dropna().corr()


def ewma_correlation(
    returns: pd.DataFrame, lam: float = RISKMETRICS_LAMBDA_DAILY
) -> pd.DataFrame:
    """EWMA correlation as at the end of the sample."""
    return correlation_from_covariance(ewma_covariance_last(returns, lam=lam))


def pairwise_table(
    returns: pd.DataFrame,
    dcc: DccResult | None = None,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
) -> pd.DataFrame:
    """
    One row per instrument pair.

    Columns:
      sample        full-sample Pearson correlation
      ewma          EWMA correlation at the end of the sample
      dcc_last      DCC conditional correlation, final date
      dcc_min/max   range of the DCC path
      dcc_range     max - min, i.e. what the static number hides
      drift         ewma - sample; large means the current regime
                    differs from the historical average

    Sorted by `dcc_range` when DCC is supplied, so the pairs whose
    relationship is least stable appear first.
    """
    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[1] < 2:
        raise ValueError("need at least two instruments")

    samp = sample_correlation(df)
    ew = ewma_correlation(df, lam=lam)

    rows = []
    for i, j in combinations(df.columns, 2):
        row = {
            "pair": f"{i} / {j}",
            "sample": float(samp.loc[i, j]),
            "ewma": float(ew.loc[i, j]),
        }
        row["drift"] = row["ewma"] - row["sample"]

        if dcc is not None:
            path = dcc.correlation_series(i, j)
            row.update(
                {
                    "dcc_last": float(path.iloc[-1]),
                    "dcc_mean": float(path.mean()),
                    "dcc_min": float(path.min()),
                    "dcc_max": float(path.max()),
                    "dcc_range": float(path.max() - path.min()),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows).set_index("pair")
    if dcc is not None and "dcc_range" in out.columns:
        out = out.sort_values("dcc_range", ascending=False)
    return out


def dcc_correlation_paths(dcc: DccResult) -> pd.DataFrame:
    """Every pairwise DCC correlation path as one frame."""
    assets = list(dcc.unconditional_correlation.columns)
    data = {}
    for i, j in combinations(assets, 2):
        data[f"{i}/{j}"] = dcc.correlation_series(i, j)
    return pd.DataFrame(data)


def correlation_stress(
    returns: pd.DataFrame, dcc: DccResult, quantile: float = 0.05
) -> pd.DataFrame:
    """
    Correlation conditional on the worst days.

    Splits the sample by portfolio-average return and reports each
    pair's mean DCC correlation on the worst `quantile` of days
    against all other days.

    If the "stressed" column is systematically higher than the calm
    one, diversification is weakest exactly when it is needed - the
    empirical version of the argument for using a conditional
    correlation model at all.
    """
    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    paths = dcc_correlation_paths(dcc)
    common = df.index.intersection(paths.index)
    if len(common) < 50:
        raise ValueError("not enough overlapping dates for a stress split")

    avg = df.loc[common].mean(axis=1)
    cutoff = float(avg.quantile(quantile))
    stressed = avg <= cutoff

    rows = []
    for col in paths.columns:
        s = paths.loc[common, col]
        rows.append(
            {
                "pair": col,
                "calm": float(s[~stressed].mean()),
                "stressed": float(s[stressed].mean()),
                "increase": float(s[stressed].mean() - s[~stressed].mean()),
            }
        )

    out = pd.DataFrame(rows).set_index("pair").sort_values("increase", ascending=False)
    out.attrs["quantile"] = quantile
    out.attrs["n_stressed_days"] = int(stressed.sum())
    return out
