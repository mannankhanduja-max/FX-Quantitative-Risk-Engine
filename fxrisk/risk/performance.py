"""
Risk-adjusted performance: Sharpe, Sortino, drawdown.

Deliberately explicit about two things that most Sharpe
implementations leave implicit and get wrong.

THE RISK-FREE RATE IS AN ANNUAL RATE APPLIED TO PERIODIC RETURNS.
Subtracting an annual 4% from a daily return is a factor-252 error,
and it is a common one. Every function here takes the annual rate
and de-annualises it internally, geometrically:

    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1

ANNUALISING A SHARPE BY sqrt(252) ASSUMES IID RETURNS. Returns are
autocorrelated and volatility clusters, so the scaled figure is
optimistic when returns are positively autocorrelated. The
functions do it because it is the convention and comparability
matters, but `sharpe_ratio` also returns the un-annualised value so
the assumption is visible rather than buried.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceSummary:
    """Standard risk-adjusted performance figures."""

    periods: int
    total_return: float
    annualised_return: float
    annualised_volatility: float
    sharpe: float
    sharpe_per_period: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    risk_free_rate: float

    def summary(self) -> str:
        return "\n".join(
            [
                f"  observations          {self.periods}",
                f"  total return          {self.total_return:+.2%}",
                f"  annualised return     {self.annualised_return:+.2%}",
                f"  annualised volatility {self.annualised_volatility:.2%}",
                f"  Sharpe (annualised)   {self.sharpe:.3f}",
                f"  Sortino               {self.sortino:.3f}",
                f"  max drawdown          {self.max_drawdown:.2%}",
                f"  Calmar                {self.calmar:.3f}",
                f"  hit rate              {self.hit_rate:.2%}",
                f"  risk-free assumed     {self.risk_free_rate:.2%} annual",
            ]
        )


def _clean(returns: pd.Series) -> pd.Series:
    return (
        pd.Series(returns, dtype="float64")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def deannualise(rate: float, periods_per_year: int = TRADING_DAYS) -> float:
    """Convert an annual rate to the equivalent periodic rate."""
    return (1.0 + rate) ** (1.0 / periods_per_year) - 1.0


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
    annualise: bool = True,
) -> float:
    """
    Sharpe ratio.

    `risk_free_rate` is an ANNUAL rate and is de-annualised before
    being subtracted from the periodic returns.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")

    excess = r - deannualise(risk_free_rate, periods_per_year)
    sd = float(excess.std(ddof=1))
    if sd <= 0:
        return float("nan")

    ratio = float(excess.mean()) / sd
    return ratio * np.sqrt(periods_per_year) if annualise else ratio


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """
    Sortino ratio: excess return over downside deviation.

    Downside is measured relative to the minimum acceptable return
    (the de-annualised risk-free rate), not to zero. Using zero is
    common and wrong whenever the risk-free rate is not zero, and it
    makes Sortino non-comparable with the Sharpe printed beside it.
    """
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")

    mar = deannualise(risk_free_rate, periods_per_year)
    excess = r - mar
    downside = excess[excess < 0]
    if len(downside) < 2:
        return float("nan")

    dd = float(np.sqrt((downside**2).mean()))
    if dd <= 0:
        return float("nan")

    return float(excess.mean()) / dd * np.sqrt(periods_per_year)


def rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """
    Rolling annualised Sharpe over `window` periods.

    126 trading days is roughly six months, which is short enough to
    show regime shifts and long enough that the estimate is not pure
    noise. A rolling Sharpe on a 60-day window is mostly noise.
    """
    r = _clean(returns)
    excess = r - deannualise(risk_free_rate, periods_per_year)
    mean = excess.rolling(window, min_periods=window).mean()
    sd = excess.rolling(window, min_periods=window).std(ddof=1)
    out = (mean / sd.replace(0, np.nan)) * np.sqrt(periods_per_year)
    return out.rename(f"rolling_sharpe_{window}")


def annualised_return(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    """Geometric annualised return."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    growth = float((1 + r).prod())
    if growth <= 0:
        return float("nan")
    years = len(r) / periods_per_year
    return growth ** (1 / years) - 1 if years > 0 else float("nan")


def annualised_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS
) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1)) * np.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough decline of the cumulative return path."""
    r = _clean(returns)
    if len(r) == 0:
        return float("nan")
    equity = (1 + r).cumprod()
    return float((equity / equity.cummax() - 1).min())


def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceSummary:
    """Full set of performance figures for one return series."""
    r = _clean(returns)
    ann_ret = annualised_return(r, periods_per_year)
    mdd = max_drawdown(r)

    return PerformanceSummary(
        periods=len(r),
        total_return=float((1 + r).prod() - 1) if len(r) else float("nan"),
        annualised_return=ann_ret,
        annualised_volatility=annualised_volatility(r, periods_per_year),
        sharpe=sharpe_ratio(r, risk_free_rate, periods_per_year),
        sharpe_per_period=sharpe_ratio(
            r, risk_free_rate, periods_per_year, annualise=False
        ),
        sortino=sortino_ratio(r, risk_free_rate, periods_per_year),
        max_drawdown=mdd,
        calmar=(ann_ret / abs(mdd)) if mdd and not np.isnan(mdd) and mdd < 0 else float("nan"),
        hit_rate=float((r > 0).mean()) if len(r) else float("nan"),
        risk_free_rate=risk_free_rate,
    )


def per_asset_performance(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Performance table, one row per asset."""
    rows = {}
    for col in returns.columns:
        s = performance_summary(returns[col], risk_free_rate, periods_per_year)
        rows[col] = {
            "ann_return": s.annualised_return,
            "ann_vol": s.annualised_volatility,
            "sharpe": s.sharpe,
            "sortino": s.sortino,
            "max_drawdown": s.max_drawdown,
            "hit_rate": s.hit_rate,
        }
    return pd.DataFrame(rows).T
