"""
VaR backtesting: Kupiec, Christoffersen, Basel traffic light.

A VaR number nobody has backtested is a decoration. These are the
standard tests a regulator or a risk committee would apply.

  Kupiec (1995) unconditional coverage
      Are there the right NUMBER of breaches? A 99% model should
      breach on 1% of days. Likelihood ratio, chi-squared with 1 df.

  Christoffersen (1998) independence
      Are the breaches SPREAD OUT, or do they cluster? Clustering
      means the model is not adapting to volatility regimes - it
      passes on average while being wrong when it matters. Tests a
      first-order Markov chain for equal transition probabilities,
      chi-squared with 1 df.

  Christoffersen conditional coverage
      Joint test of both. LR_cc = LR_uc + LR_ind, chi-squared with
      2 df.

  Basel traffic light
      The supervisory rule: over 250 trading days at 99%, 0-4
      breaches is green, 5-9 amber, 10+ red, with a capital
      multiplier attached. Blunt, but it is the actual standard.

  Lopez loss
      A magnitude-aware score. Two models can breach equally often
      while one is far wronger when it breaches.

The independence test is the one that separates a serious engine
from a toy. Historical VaR usually passes Kupiec and fails
Christoffersen, because it cannot react to a volatility regime
change; that failure is the entire argument for GARCH.

A LIMITATION WORTH KNOWING BEFORE YOU QUOTE A P-VALUE
------------------------------------------------------
The Christoffersen independence test has low power when breaches
are sparse. At 99% confidence over 3,000 days you expect about 30
breaches, and the transition counts that drive the statistic are
correspondingly thin: in simulation against a deliberately
misspecified flat VaR, the test rejects less than half the time.
At 95% - roughly 150 breaches - power rises above two thirds.

So a passing independence test at 99% is weak evidence, not proof.
Run the backtest at 95% and 97.5% as well; a model that only
survives where the test cannot see is not a model that survives.
The test suite pins this behaviour explicitly rather than papering
over it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestResult:
    """Outcome of a full VaR backtest."""

    method: str
    confidence: float
    n_observations: int
    n_breaches: int
    expected_breaches: float
    breach_rate: float

    kupiec_statistic: float
    kupiec_pvalue: float

    christoffersen_ind_statistic: float
    christoffersen_ind_pvalue: float

    conditional_coverage_statistic: float
    conditional_coverage_pvalue: float

    basel_zone: str
    basel_multiplier: float
    lopez_loss: float
    mean_breach_severity: float
    max_breach_severity: float

    def passed(self, level: float = 0.05) -> bool:
        """True if the model survives both coverage and independence."""
        return (
            self.kupiec_pvalue > level
            and self.christoffersen_ind_pvalue > level
        )

    def summary(self) -> str:
        verdict = "PASS" if self.passed() else "FAIL"
        return "\n".join(
            [
                f"VaR backtest - {self.method} at {self.confidence:.1%}",
                f"  observations            {self.n_observations}",
                f"  breaches                {self.n_breaches} "
                f"(expected {self.expected_breaches:.1f})",
                f"  breach rate             {self.breach_rate:.2%} "
                f"(target {1 - self.confidence:.2%})",
                "",
                f"  Kupiec UC               stat {self.kupiec_statistic:7.3f}   "
                f"p {self.kupiec_pvalue:.4f}",
                f"  Christoffersen IND      stat {self.christoffersen_ind_statistic:7.3f}   "
                f"p {self.christoffersen_ind_pvalue:.4f}",
                f"  Conditional coverage    stat {self.conditional_coverage_statistic:7.3f}   "
                f"p {self.conditional_coverage_pvalue:.4f}",
                "",
                f"  Basel zone              {self.basel_zone.upper()} "
                f"(multiplier {self.basel_multiplier:.2f})",
                f"  Lopez loss              {self.lopez_loss:.6f}",
                f"  mean breach severity    {self.mean_breach_severity:.4%}",
                f"  max breach severity     {self.max_breach_severity:.4%}",
                "",
                f"  VERDICT                 {verdict}",
            ]
        )


def kupiec_pof(n_breaches: int, n_obs: int, confidence: float) -> tuple[float, float]:
    """
    Kupiec proportion-of-failures likelihood ratio test.

    H0: the true breach probability equals 1 - confidence.
    """
    p = 1.0 - confidence
    x, n = n_breaches, n_obs

    if n == 0:
        return float("nan"), float("nan")
    if x == 0:
        lr = -2.0 * n * np.log(1 - p)
        return float(lr), float(1 - stats.chi2.cdf(lr, df=1))
    if x == n:
        lr = -2.0 * n * np.log(p)
        return float(lr), float(1 - stats.chi2.cdf(lr, df=1))

    p_hat = x / n
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
    lr = -2.0 * (ll_null - ll_alt)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """
    Christoffersen independence test.

    H0: a breach today is no more likely after a breach yesterday
    than after a quiet day.
    """
    b = np.asarray(breaches).astype(int)
    if len(b) < 2:
        return float("nan"), float("nan")

    prev, curr = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # Degenerate cases: no clustering evidence available.
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return 0.0, 1.0

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    if pi in (0.0, 1.0) or pi01 in (0.0,) and pi11 in (0.0,):
        return 0.0, 1.0

    def _safe_log(v: float) -> float:
        return np.log(v) if v > 0 else 0.0

    ll_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1 - pi01)
        + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11)
        + n11 * _safe_log(pi11)
    )

    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    return float(lr), float(1 - stats.chi2.cdf(lr, df=1))


def basel_traffic_light(
    n_breaches: int, n_obs: int, confidence: float = 0.99
) -> tuple[str, float]:
    """
    Basel supervisory zones, scaled to a 250-day year.

    Thresholds are defined for 250 observations AT 99% CONFIDENCE.
    The breach count is scaled proportionally for other sample
    lengths so the zone stays interpretable.

    Applying these thresholds at any other confidence level is
    meaningless: a correctly calibrated 95% model breaches on 5% of
    days, which is ~12 breaches per 250 and would be scored "red"
    for behaving exactly as intended. Confidence levels other than
    99% therefore return "n/a" rather than a misleading colour.
    """
    if n_obs == 0:
        return "undefined", float("nan")
    if abs(confidence - 0.99) > 1e-9:
        return "n/a", float("nan")

    scaled = n_breaches * 250.0 / n_obs

    if scaled < 5:
        return "green", 3.00
    if scaled < 6:
        return "amber", 3.40
    if scaled < 7:
        return "amber", 3.50
    if scaled < 8:
        return "amber", 3.65
    if scaled < 9:
        return "amber", 3.75
    if scaled < 10:
        return "amber", 3.85
    return "red", 4.00


def lopez_loss(
    returns: np.ndarray, var: np.ndarray, scale: float = 100.0
) -> float:
    """
    Lopez (1999) magnitude-weighted loss function.

    Adds 1 + (loss - VaR)^2 on a breach and 0 otherwise, so a model
    that breaches badly scores worse than one that breaches
    narrowly.

    THE SCALE MATTERS, AND IT IS EASY TO GET WRONG. On decimal
    returns a typical exceedance is around 0.005, so the quadratic
    term contributes about 2.5e-5 against a constant of 1 - the
    score collapses into a plain breach count and the "magnitude
    aware" property is lost entirely. A model with one extra small
    breach then scores worse than a model with one enormous one.

    `scale` converts to percent before squaring (100.0 by default),
    which puts the exceedance term on the same order as the count
    and makes the metric behave as intended. Pass scale=1.0 to
    reproduce the naive decimal version.

    Only compare Lopez losses computed at the same scale.
    """
    loss = -np.asarray(returns) * scale
    v = np.asarray(var) * scale
    breach = loss > v
    return float(np.sum(np.where(breach, 1.0 + (loss - v) ** 2, 0.0)))


def backtest_var(
    returns: pd.Series,
    var: pd.Series,
    confidence: float = 0.99,
    method: str = "unnamed",
) -> BacktestResult:
    """
    Run the full battery on a realised-return / VaR-forecast pair.

    Parameters
    ----------
    returns
        Realised returns. Row t must be the return that the VaR in
        row t was forecasting - i.e. the VaR series is already
        shifted so it was knowable at the close of t-1.
    var
        Positive VaR forecasts.
    confidence
        Confidence level the VaR was computed at.
    """
    df = pd.DataFrame({"r": returns, "var": var}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(df) < 100:
        raise ValueError(
            f"backtest needs at least 100 aligned observations, got {len(df)}"
        )

    r = df["r"].to_numpy()
    v = df["var"].to_numpy()

    if np.any(v < 0):
        raise ValueError(
            "VaR series contains negative values - this module expects VaR "
            "as a positive loss magnitude"
        )

    loss = -r
    breaches = (loss > v).astype(int)
    n_breach = int(breaches.sum())
    n = len(df)

    kup_stat, kup_p = kupiec_pof(n_breach, n, confidence)
    ind_stat, ind_p = christoffersen_independence(breaches)

    cc_stat = kup_stat + ind_stat if np.isfinite(kup_stat) and np.isfinite(ind_stat) else float("nan")
    cc_p = float(1 - stats.chi2.cdf(cc_stat, df=2)) if np.isfinite(cc_stat) else float("nan")

    zone, mult = basel_traffic_light(n_breach, n, confidence)

    severities = (loss - v)[breaches == 1]

    return BacktestResult(
        method=method,
        confidence=confidence,
        n_observations=n,
        n_breaches=n_breach,
        expected_breaches=n * (1 - confidence),
        breach_rate=n_breach / n,
        kupiec_statistic=kup_stat,
        kupiec_pvalue=kup_p,
        christoffersen_ind_statistic=ind_stat,
        christoffersen_ind_pvalue=ind_p,
        conditional_coverage_statistic=cc_stat,
        conditional_coverage_pvalue=cc_p,
        basel_zone=zone,
        basel_multiplier=mult,
        lopez_loss=lopez_loss(r, v),
        mean_breach_severity=float(severities.mean()) if len(severities) else 0.0,
        max_breach_severity=float(severities.max()) if len(severities) else 0.0,
    )


def compare_models(results: list[BacktestResult]) -> pd.DataFrame:
    """Tabulate several backtests side by side, best-behaved first."""
    rows = []
    for res in results:
        rows.append(
            {
                "method": res.method,
                "breaches": res.n_breaches,
                "expected": round(res.expected_breaches, 1),
                "breach_rate": res.breach_rate,
                "kupiec_p": res.kupiec_pvalue,
                "christoffersen_p": res.christoffersen_ind_pvalue,
                "cc_p": res.conditional_coverage_pvalue,
                "basel_zone": res.basel_zone,
                "lopez_loss": res.lopez_loss,
                "passed": res.passed(),
            }
        )
    df = pd.DataFrame(rows).set_index("method")
    return df.sort_values(["passed", "lopez_loss"], ascending=[False, True])
