"""
Value at Risk and Expected Shortfall.

Five estimators, because no single one is right and the disagreement
between them is itself information:

  historical        Empirical quantile of the last N returns. No
                    distributional assumption; reacts slowly, and a
                    crisis stays in the window until it rolls out,
                    at which point VaR drops discontinuously.

  parametric_normal The textbook mu + z * sigma. Included as a
                    baseline that should FAIL the backtests on real
                    data - daily FX and equity returns are not
                    Gaussian, and demonstrating that is the point.

  ewma              RiskMetrics: normal quantile on an EWMA
                    volatility forecast. Reacts fast, still assumes
                    a Gaussian tail.

  garch_t           GARCH(1,1) volatility forecast with a Student-t
                    quantile. Reacts fast AND has a fat tail. This
                    is the one that should survive.

  cornish_fisher    Normal quantile adjusted for sample skewness and
                    excess kurtosis. Cheap fat-tail correction; the
                    expansion misbehaves at very high confidence, so
                    it is not recommended beyond 99%.

Sign convention: VaR is returned as a POSITIVE number representing
a loss. A 99% one-day VaR of 0.021 means "on 1 day in 100 we expect
to lose more than 2.1%". Expected Shortfall is the mean loss
conditional on breaching that level, also positive.

Expected Shortfall is included because Basel's Fundamental Review of
the Trading Book replaced 99% VaR with 97.5% ES as the capital
measure, on the grounds that VaR says nothing about how bad the tail
gets once you are in it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..models.ewma import RISKMETRICS_LAMBDA_DAILY, ewma_variance
from ..models.garch import rolling_garch_forecasts


@dataclass
class VarEstimate:
    """A VaR/ES pair at one confidence level."""

    var: float
    expected_shortfall: float
    confidence: float
    method: str
    horizon_days: int = 1

    def scale_to_horizon(self, days: int) -> "VarEstimate":
        """
        Square-root-of-time scaling.

        Valid only under iid returns with no drift. Real returns are
        neither, and volatility mean-reverts, so this OVERSTATES risk
        when current volatility is above its long-run level and
        understates it when below. For anything beyond about ten days
        use the GARCH multi-step forecast instead.
        """
        factor = np.sqrt(days / self.horizon_days)
        return VarEstimate(
            var=self.var * factor,
            expected_shortfall=self.expected_shortfall * factor,
            confidence=self.confidence,
            method=f"{self.method}+sqrt_time",
            horizon_days=days,
        )


def _check_confidence(c: float) -> None:
    if not 0.5 < c < 1.0:
        raise ValueError(f"confidence must lie in (0.5, 1.0), got {c}")


def historical_var(
    returns: pd.Series, confidence: float = 0.99
) -> VarEstimate:
    """Empirical quantile VaR and ES."""
    _check_confidence(confidence)
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(r) < 30:
        raise ValueError(f"need at least 30 observations, got {len(r)}")

    q = float(np.quantile(r, 1.0 - confidence))
    tail = r[r <= q]
    es = float(tail.mean()) if len(tail) else q

    return VarEstimate(-q, -es, confidence, "historical")


def parametric_normal_var(
    returns: pd.Series, confidence: float = 0.99
) -> VarEstimate:
    """Gaussian VaR. The baseline that ought to fail."""
    _check_confidence(confidence)
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 30:
        raise ValueError(f"need at least 30 observations, got {len(r)}")

    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)

    var = -(mu + sigma * z)
    es = -(mu - sigma * stats.norm.pdf(z) / alpha)
    return VarEstimate(var, es, confidence, "parametric_normal")


def cornish_fisher_var(
    returns: pd.Series, confidence: float = 0.99
) -> VarEstimate:
    """
    Cornish-Fisher expansion: normal quantile corrected for the third
    and fourth moments.
    """
    _check_confidence(confidence)
    if confidence > 0.99:
        # The expansion is not monotone in the far tail.
        pass
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 60:
        raise ValueError(f"need at least 60 observations, got {len(r)}")

    mu, sigma = float(r.mean()), float(r.std(ddof=1))
    s = float(stats.skew(r))
    k = float(stats.kurtosis(r))  # excess

    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)
    z_cf = (
        z
        + (z**2 - 1) * s / 6
        + (z**3 - 3 * z) * k / 24
        - (2 * z**3 - 5 * z) * s**2 / 36
    )

    var = -(mu + sigma * z_cf)
    # ES has no clean closed form here; integrate the adjusted quantile.
    grid = np.linspace(1e-6, alpha, 500)
    zg = stats.norm.ppf(grid)
    zg_cf = (
        zg
        + (zg**2 - 1) * s / 6
        + (zg**3 - 3 * zg) * k / 24
        - (2 * zg**3 - 5 * zg) * s**2 / 36
    )
    es = -(mu + sigma * float(np.mean(zg_cf)))

    return VarEstimate(var, es, confidence, "cornish_fisher")


def ewma_var_series(
    returns: pd.Series,
    confidence: float = 0.99,
    lam: float = RISKMETRICS_LAMBDA_DAILY,
    warmup: int = 30,
) -> pd.DataFrame:
    """
    One-step-ahead EWMA VaR and ES for every date.

    Row t is the forecast made at the close of t-1 for day t, so it
    lines up directly with the realised return on day t for
    backtesting.
    """
    _check_confidence(confidence)
    var_series = ewma_variance(returns, lam=lam, warmup=warmup)
    sigma = np.sqrt(var_series)

    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)

    var = -(sigma * z)
    es = sigma * stats.norm.pdf(z) / alpha

    return pd.DataFrame({"var": var, "expected_shortfall": es})


def garch_var_series(
    returns: pd.Series,
    confidence: float = 0.99,
    dist: str = "t",
    window: int = 750,
    refit_every: int = 21,
    min_obs: int = 250,
) -> pd.DataFrame:
    """
    Walk-forward GARCH VaR and ES.

    BOTH inputs to the quantile are walk-forward. The variance comes
    from the rolling recursion, and - importantly - so does the
    Student-t degrees of freedom. An earlier version of this
    function estimated nu once on the full sample and applied it to
    every historical date, which leaked future information into the
    tail shape of every past VaR. The variance was honest and the
    quantile was not, which is the more dangerous half to get wrong
    because it is invisible in a plot of the volatility path.

    Between refits nu is held at its last estimated value. Stale,
    but never forward-looking.
    """
    _check_confidence(confidence)

    fc = rolling_garch_forecasts(
        returns, window=window, refit_every=refit_every, dist=dist, min_obs=min_obs
    )
    sigma = np.sqrt(fc["variance"])
    alpha = 1.0 - confidence

    if dist == "t":
        # Vectorised over the nu path. Floor at 2.5 so the variance
        # of the standardised t stays finite.
        nu = fc["nu"].clip(lower=2.5).to_numpy()
        scale = np.sqrt((nu - 2) / nu)
        t_q = stats.t.ppf(alpha, df=nu)
        q = t_q * scale
        es_std = -(
            (nu + t_q**2) / (nu - 1) * stats.t.pdf(t_q, df=nu) / alpha
        ) * scale
    else:
        q = np.full(len(fc), stats.norm.ppf(alpha))
        es_std = np.full(len(fc), -stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)

    out = pd.DataFrame(
        {
            "var": -(sigma.to_numpy() * q),
            "expected_shortfall": -(sigma.to_numpy() * es_std),
            "nu": fc["nu"].to_numpy(),
        },
        index=fc.index,
    )
    out.attrs["fit_failures"] = fc.attrs.get("fit_failures", 0)
    return out


def portfolio_var_from_covariance(
    weights: pd.Series | np.ndarray,
    covariance: pd.DataFrame,
    confidence: float = 0.99,
    dist: str = "normal",
    nu: float | None = None,
) -> VarEstimate:
    """
    Parametric portfolio VaR from a covariance matrix.

    Use this with an EWMA or DCC covariance to get a VaR that
    responds to changing correlation as well as changing volatility.
    """
    _check_confidence(confidence)
    w = np.asarray(weights, dtype="float64").reshape(-1)
    cov = covariance.to_numpy()

    if len(w) != cov.shape[0]:
        raise ValueError(
            f"weights length {len(w)} does not match covariance dimension {cov.shape[0]}"
        )

    variance = float(w @ cov @ w)
    if variance < 0:
        raise ValueError("negative portfolio variance - covariance matrix is not PSD")
    sigma = np.sqrt(variance)

    alpha = 1.0 - confidence
    if dist == "normal":
        q = stats.norm.ppf(alpha)
        es_std = -stats.norm.pdf(q) / alpha
    else:
        if nu is None:
            raise ValueError("Student-t portfolio VaR requires nu")
        scale = np.sqrt((nu - 2) / nu)
        t_q = stats.t.ppf(alpha, df=nu)
        q = t_q * scale
        es_std = -((nu + t_q**2) / (nu - 1) * stats.t.pdf(t_q, df=nu) / alpha) * scale

    return VarEstimate(
        var=-(sigma * q),
        expected_shortfall=-(sigma * es_std),
        confidence=confidence,
        method=f"portfolio_{dist}",
    )


def component_var(
    weights: pd.Series, covariance: pd.DataFrame, confidence: float = 0.99
) -> pd.DataFrame:
    """
    Decompose portfolio VaR into per-asset contributions.

    Marginal VaR is d(VaR)/d(w_i); component VaR is w_i times that,
    and the components sum to total VaR by Euler's theorem. This
    answers "where is the risk actually coming from", which a single
    portfolio number never does.
    """
    _check_confidence(confidence)
    w = weights.to_numpy(dtype="float64")
    cov = covariance.to_numpy()

    sigma = np.sqrt(float(w @ cov @ w))
    if sigma <= 0:
        raise ValueError("portfolio volatility is zero")

    z = abs(stats.norm.ppf(1.0 - confidence))
    marginal = z * (cov @ w) / sigma
    component = w * marginal
    total = component.sum()

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_var": marginal,
            "component_var": component,
            "pct_of_total": component / total if total else np.nan,
        },
        index=weights.index,
    )
