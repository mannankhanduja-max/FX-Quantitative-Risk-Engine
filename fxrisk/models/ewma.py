"""
RiskMetrics-style EWMA variance and covariance.

The sample covariance matrix weights a return from five years ago
exactly as heavily as yesterday's. For risk measurement that is the
wrong assumption: volatility clusters, so recent observations carry
more information about tomorrow than distant ones do.

EWMA replaces the equal weighting with a geometric decay:

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r_{t-1}^2

J.P. Morgan's RiskMetrics Technical Document (1996) sets lambda to
0.94 for daily data and 0.97 for monthly. Those are the defaults
here. Lower lambda reacts faster and is noisier; higher lambda is
smoother and slower to register a regime change.

Note the indexing. The variance used to forecast day t is built from
returns up to t-1 only. Every function in this module returns a
series aligned so that row t is a genuine one-step-ahead forecast,
usable at the close of t-1. Getting this wrong is the most common
way an EWMA VaR backtest ends up looking better than it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# RiskMetrics (1996) daily decay factor.
RISKMETRICS_LAMBDA_DAILY = 0.94
RISKMETRICS_LAMBDA_MONTHLY = 0.97


def _validate_lambda(lam: float) -> None:
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lambda must lie strictly between 0 and 1, got {lam}")


def effective_observations(lam: float) -> float:
    """
    Effective sample size implied by a decay factor.

    The weights (1 - lam) * lam^k sum to one, and the "centre of
    mass" 1 / (1 - lam) is the intuitive read: lambda = 0.94 gives
    roughly 17 days, 0.97 roughly 33.
    """
    _validate_lambda(lam)
    return 1.0 / (1.0 - lam)


def ewma_variance(
    returns: pd.Series,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
    warmup: int = 30,
) -> pd.Series:
    """
    One-step-ahead EWMA variance forecast.

    Parameters
    ----------
    returns
        Periodic (not annualised) returns.
    lam
        Decay factor.
    warmup
        Number of initial observations used to seed the recursion
        with a plain sample variance. Those rows are returned as
        NaN rather than as a forecast built from almost no data.

    Returns
    -------
    pandas.Series
        Variance forecast for period t, conditional on information
        through t-1. NaN for the warmup rows.
    """
    _validate_lambda(lam)

    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan)
    values = r.to_numpy()
    n = len(values)

    if n <= warmup:
        raise ValueError(
            f"need more than warmup={warmup} observations, got {n}"
        )

    seed_window = values[:warmup]
    seed_window = seed_window[~np.isnan(seed_window)]
    if len(seed_window) < 2:
        raise ValueError("warmup window contains fewer than 2 valid returns")

    out = np.full(n, np.nan)
    sigma2 = float(np.var(seed_window, ddof=1))

    for t in range(warmup, n):
        # Forecast for t uses information through t-1 only.
        out[t] = sigma2
        r_prev = values[t]
        if not np.isnan(r_prev):
            sigma2 = lam * sigma2 + (1.0 - lam) * r_prev**2

    return pd.Series(out, index=r.index, name="ewma_variance")


def ewma_volatility(
    returns: pd.Series,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
    warmup: int = 30,
    annualise: bool = False,
    periods_per_year: int = 252,
) -> pd.Series:
    """One-step-ahead EWMA volatility (standard deviation)."""
    var = ewma_variance(returns, lam=lam, warmup=warmup)
    vol = np.sqrt(var)
    if annualise:
        vol = vol * np.sqrt(periods_per_year)
    return vol.rename("ewma_volatility")


def ewma_covariance(
    returns: pd.DataFrame,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
    warmup: int = 30,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Full EWMA covariance matrix path.

    Returns a dict keyed by date. Each value is the covariance
    matrix forecast for that date, conditional on information
    through the previous date.

    Memory grows as n_dates * n_assets^2. For a ten-asset,
    2000-day sample that is a few hundred megabytes at float64,
    which is fine; for a large universe use ``ewma_covariance_last``
    instead.
    """
    _validate_lambda(lam)

    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) <= warmup:
        raise ValueError(
            f"need more than warmup={warmup} complete rows, got {len(df)}"
        )

    values = df.to_numpy()
    cols = list(df.columns)

    cov = np.cov(values[:warmup], rowvar=False, ddof=1)
    cov = np.atleast_2d(cov)

    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for t in range(warmup, len(df)):
        out[df.index[t]] = pd.DataFrame(cov.copy(), index=cols, columns=cols)
        r = values[t].reshape(-1, 1)
        cov = lam * cov + (1.0 - lam) * (r @ r.T)

    return out


def ewma_covariance_last(
    returns: pd.DataFrame,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
    warmup: int = 30,
) -> pd.DataFrame:
    """
    EWMA covariance matrix as at the end of the sample.

    This is the matrix you would use to size risk for the next
    period. Cheaper than building the whole path.
    """
    _validate_lambda(lam)

    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) <= warmup:
        raise ValueError(
            f"need more than warmup={warmup} complete rows, got {len(df)}"
        )

    values = df.to_numpy()
    cov = np.atleast_2d(np.cov(values[:warmup], rowvar=False, ddof=1))

    for t in range(warmup, len(df)):
        r = values[t].reshape(-1, 1)
        cov = lam * cov + (1.0 - lam) * (r @ r.T)

    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix."""
    sd = np.sqrt(np.diag(cov.to_numpy()))
    if np.any(sd <= 0):
        raise ValueError("covariance matrix has a non-positive diagonal entry")
    outer = np.outer(sd, sd)
    corr = cov.to_numpy() / outer
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def portfolio_variance_path(
    returns: pd.DataFrame,
    weights: pd.Series | np.ndarray,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
    warmup: int = 30,
) -> pd.Series:
    """
    EWMA variance of a fixed-weight portfolio, without materialising
    every covariance matrix.

    Equivalent to applying ``ewma_variance`` to the portfolio return
    series when weights are constant, but written through the
    covariance recursion so it stays correct if you later extend it
    to time-varying weights.
    """
    _validate_lambda(lam)

    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    w = np.asarray(weights, dtype="float64").reshape(-1)

    if len(w) != df.shape[1]:
        raise ValueError(
            f"weights has length {len(w)} but returns has {df.shape[1]} columns"
        )
    if len(df) <= warmup:
        raise ValueError(
            f"need more than warmup={warmup} complete rows, got {len(df)}"
        )

    values = df.to_numpy()
    cov = np.atleast_2d(np.cov(values[:warmup], rowvar=False, ddof=1))

    out = np.full(len(df), np.nan)
    for t in range(warmup, len(df)):
        out[t] = float(w @ cov @ w)
        r = values[t].reshape(-1, 1)
        cov = lam * cov + (1.0 - lam) * (r @ r.T)

    return pd.Series(out, index=df.index, name="portfolio_ewma_variance")
