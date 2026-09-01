"""
GARCH(1,1) with Gaussian or Student-t innovations.

    r_t     = mu + eps_t
    eps_t   = sigma_t * z_t,        z_t ~ iid, E[z]=0, Var[z]=1
    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Estimated by maximum likelihood with scipy's SLSQP, subject to
omega > 0, alpha >= 0, beta >= 0 and alpha + beta < 1 (covariance
stationarity).

Implemented directly rather than via the `arch` package so the
likelihood, the constraints and the forecast recursion are all
visible and testable. If you would rather lean on a maintained
library, `arch` is the standard choice and this class is a drop-in
shape match for its `.fit()` / `.forecast()` idiom.

Returns are handled in PERCENT internally. GARCH likelihoods are
badly scaled on raw decimal returns - omega lands around 1e-6 and
the optimiser struggles - so the class rescales on the way in and
back out. This is what `arch` does too, and it matters more than it
looks.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, stats

_SCALE = 100.0  # decimal returns -> percent


@dataclass
class GarchResult:
    """Fitted GARCH(1,1)."""

    mu: float
    omega: float
    alpha: float
    beta: float
    nu: float | None
    dist: str
    loglikelihood: float
    converged: bool
    n_obs: int
    conditional_variance: pd.Series = field(repr=False)
    standardised_residuals: pd.Series = field(repr=False)

    @property
    def persistence(self) -> float:
        """alpha + beta. Approaches 1 as shocks decay more slowly."""
        return self.alpha + self.beta

    @property
    def long_run_variance(self) -> float:
        """Unconditional variance omega / (1 - alpha - beta), in decimals."""
        if self.persistence >= 1.0:
            return float("nan")
        return self.omega / (1.0 - self.persistence)

    @property
    def half_life(self) -> float:
        """Days for a variance shock to decay by half."""
        p = self.persistence
        if not 0.0 < p < 1.0:
            return float("nan")
        return float(np.log(0.5) / np.log(p))

    def summary(self) -> str:
        lines = [
            f"GARCH(1,1) with {self.dist} innovations",
            f"  observations      {self.n_obs}",
            f"  converged         {self.converged}",
            f"  log-likelihood    {self.loglikelihood:,.2f}",
            "",
            f"  mu                {self.mu: .6e}",
            f"  omega             {self.omega: .6e}",
            f"  alpha             {self.alpha: .4f}",
            f"  beta              {self.beta: .4f}",
        ]
        if self.nu is not None:
            lines.append(f"  nu (df)           {self.nu: .3f}")
        lines += [
            "",
            f"  persistence       {self.persistence:.4f}",
            f"  half-life (days)  {self.half_life:.1f}",
            f"  long-run vol      {np.sqrt(self.long_run_variance * 252):.2%} annualised",
        ]
        return "\n".join(lines)


def _garch_recursion(
    eps: np.ndarray, omega: float, alpha: float, beta: float, backcast: float
) -> np.ndarray:
    n = len(eps)
    sigma2 = np.empty(n)
    sigma2[0] = backcast
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def _negative_loglikelihood(
    params: np.ndarray, r: np.ndarray, dist: str, backcast: float
) -> float:
    if dist == "normal":
        mu, omega, alpha, beta = params
        nu = None
    else:
        mu, omega, alpha, beta, nu = params
        if nu <= 2.05:
            return 1e10

    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.99999:
        return 1e10

    eps = r - mu
    sigma2 = _garch_recursion(eps, omega, alpha, beta, backcast)

    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e10

    z = eps / np.sqrt(sigma2)

    if dist == "normal":
        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + z**2)
    else:
        # Student-t standardised to unit variance.
        ll = np.sum(
            stats.t.logpdf(z * np.sqrt(nu / (nu - 2)), df=nu)
            + 0.5 * np.log(nu / (nu - 2))
            - 0.5 * np.log(sigma2)
        )

    return -ll if np.isfinite(ll) else 1e10


def fit_garch(
    returns: pd.Series,
    dist: str = "t",
    mean: str = "constant",
) -> GarchResult:
    """
    Fit GARCH(1,1) by maximum likelihood.

    Parameters
    ----------
    returns
        Periodic decimal returns (0.01 = 1%).
    dist
        "normal" or "t". Student-t is the sensible default for daily
        FX and equity returns - the Gaussian version systematically
        underestimates tail risk, which is precisely what a VaR
        engine must not do.
    mean
        "constant" estimates mu; "zero" fixes it at 0, which is
        common for daily risk work where the drift is not reliably
        estimable over the horizon that matters.
    """
    if dist not in {"normal", "t"}:
        raise ValueError("dist must be 'normal' or 't'")
    if mean not in {"constant", "zero"}:
        raise ValueError("mean must be 'constant' or 'zero'")

    s = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 100:
        raise ValueError(f"GARCH needs at least 100 observations, got {len(s)}")

    r = s.to_numpy() * _SCALE
    backcast = float(np.var(r, ddof=1))

    mu0 = float(np.mean(r)) if mean == "constant" else 0.0
    alpha0, beta0 = 0.08, 0.90
    omega0 = backcast * (1 - alpha0 - beta0)

    if dist == "normal":
        x0 = np.array([mu0, omega0, alpha0, beta0])
        bounds = [(-10, 10), (1e-10, 10 * backcast), (0.0, 0.5), (0.0, 0.999)]
    else:
        x0 = np.array([mu0, omega0, alpha0, beta0, 8.0])
        bounds = [
            (-10, 10),
            (1e-10, 10 * backcast),
            (0.0, 0.5),
            (0.0, 0.999),
            (2.1, 200.0),
        ]

    if mean == "zero":
        bounds[0] = (0.0, 0.0)
        x0[0] = 0.0

    constraints = [
        {"type": "ineq", "fun": lambda p: 0.99999 - (p[2] + p[3])},
    ]

    # SLSQP probes slightly outside the box during line search and
    # scipy warns each time it clips back. That is normal optimiser
    # behaviour, not a numerical problem - the likelihood already
    # returns a large penalty for infeasible parameters, and
    # convergence is checked via res.success below. Suppressed
    # narrowly, around this call only, so genuine warnings from
    # elsewhere in the module still surface.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Values in x were outside bounds",
            category=RuntimeWarning,
        )
        res = optimize.minimize(
            _negative_loglikelihood,
            x0,
            args=(r, dist, backcast),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 2000, "ftol": 1e-10},
        )

    params = res.x
    if dist == "normal":
        mu, omega, alpha, beta = params
        nu = None
    else:
        mu, omega, alpha, beta, nu = params

    eps = r - mu
    sigma2 = _garch_recursion(eps, omega, alpha, beta, backcast)
    z = eps / np.sqrt(sigma2)

    # Undo the percent scaling on the variance parameters.
    return GarchResult(
        mu=float(mu) / _SCALE,
        omega=float(omega) / _SCALE**2,
        alpha=float(alpha),
        beta=float(beta),
        nu=float(nu) if nu is not None else None,
        dist=dist,
        loglikelihood=float(-res.fun),
        converged=bool(res.success),
        n_obs=len(r),
        conditional_variance=pd.Series(
            sigma2 / _SCALE**2, index=s.index, name="conditional_variance"
        ),
        standardised_residuals=pd.Series(z, index=s.index, name="std_resid"),
    )


def forecast_variance(result: GarchResult, horizon: int = 1) -> np.ndarray:
    """
    Multi-step variance forecast from the end of the fitted sample.

    Uses the standard recursion

        E[sigma2_{T+h}] = long_run + (alpha + beta)^(h-1)
                          * (sigma2_{T+1} - long_run)

    so the forecast decays geometrically toward the unconditional
    variance. Returns an array of length `horizon` in decimal
    variance units.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    sigma2_T = float(result.conditional_variance.iloc[-1])
    eps_T = float(
        result.standardised_residuals.iloc[-1] * np.sqrt(sigma2_T)
    )

    sigma2_next = result.omega + result.alpha * eps_T**2 + result.beta * sigma2_T

    lr = result.long_run_variance
    p = result.persistence

    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        if np.isnan(lr):
            out[h - 1] = sigma2_next
        else:
            out[h - 1] = lr + (p ** (h - 1)) * (sigma2_next - lr)
    return out


def rolling_garch_forecasts(
    returns: pd.Series,
    window: int = 750,
    refit_every: int = 21,
    dist: str = "t",
    min_obs: int = 250,
) -> pd.DataFrame:
    """
    Walk-forward one-step-ahead GARCH forecasts.

    This is the series a VaR backtest must be built on. At each date
    t the model has seen returns strictly before t, so the forecast
    is genuinely out of sample.

    Returns a frame with three columns, ALL of them walk-forward:

        variance   one-step-ahead conditional variance
        nu         Student-t degrees of freedom as at that date
        mu         fitted mean

    `nu` is returned as a path rather than a scalar for a specific
    reason. VaR needs both a scale and a tail SHAPE. Estimating the
    variance walk-forward and then applying a degrees-of-freedom
    parameter fitted on the whole sample leaks future information
    into every historical quantile - the 2008 VaR would be computed
    using a tail shape informed by 2020. The leak is modest, because
    nu moves slowly, but it is the exact failure a risk engine is
    supposed to be credible about not having.

    Refitting every day is exact but slow. `refit_every` refits the
    parameters periodically and applies the variance recursion daily
    in between, which is what a risk desk actually does. Between
    refits nu is held at its last estimated value - stale, but never
    forward-looking.

    Parameters
    ----------
    window
        Rolling estimation window in observations.
    refit_every
        Trading days between parameter re-estimations.
    min_obs
        Observations required before the first fit.

    Notes
    -----
    `fit_failures` is recorded in the frame's `.attrs`. A high count
    means the forecasts are being carried on stale parameters and
    the backtest is flattering the model.
    """
    s = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(s)
    if n <= min_obs:
        raise ValueError(f"need more than min_obs={min_obs} observations, got {n}")

    values = s.to_numpy()
    var_out = np.full(n, np.nan)
    nu_out = np.full(n, np.nan)
    mu_out = np.full(n, np.nan)

    params: tuple[float, float, float, float] | None = None
    nu_current: float = np.nan
    sigma2 = float(np.var(values[:min_obs], ddof=1))
    last_fit = -10**9
    fit_failures = 0

    for t in range(min_obs, n):
        if params is None or (t - last_fit) >= refit_every:
            start = max(0, t - window)
            try:
                fitted = fit_garch(s.iloc[start:t], dist=dist)
                params = (fitted.omega, fitted.alpha, fitted.beta, fitted.mu)
                nu_current = fitted.nu if fitted.nu is not None else np.nan
                sigma2 = float(fitted.conditional_variance.iloc[-1])
                eps_last = values[t - 1] - fitted.mu
                sigma2 = (
                    fitted.omega
                    + fitted.alpha * eps_last**2
                    + fitted.beta * sigma2
                )
                last_fit = t
            except Exception:
                # Keep the previous parameters rather than dropping a
                # day out of the backtest. A silently missing forecast
                # inflates the apparent pass rate.
                fit_failures += 1
                if params is None:
                    continue

        omega, alpha, beta, mu = params
        var_out[t] = sigma2
        nu_out[t] = nu_current
        mu_out[t] = mu

        eps = values[t] - mu
        sigma2 = omega + alpha * eps**2 + beta * sigma2

    frame = pd.DataFrame(
        {"variance": var_out, "nu": nu_out, "mu": mu_out}, index=s.index
    )
    frame.attrs["fit_failures"] = fit_failures
    frame.attrs["refits_attempted"] = max(0, (n - min_obs) // refit_every + 1)
    return frame


def rolling_garch_variance(
    returns: pd.Series,
    window: int = 750,
    refit_every: int = 21,
    dist: str = "t",
    min_obs: int = 250,
) -> pd.Series:
    """
    Walk-forward one-step-ahead GARCH variance.

    Thin wrapper over `rolling_garch_forecasts` for callers that only
    need the variance path.
    """
    return rolling_garch_forecasts(
        returns,
        window=window,
        refit_every=refit_every,
        dist=dist,
        min_obs=min_obs,
    )["variance"].rename("garch_variance")
