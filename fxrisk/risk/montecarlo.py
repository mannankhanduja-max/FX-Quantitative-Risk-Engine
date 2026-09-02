"""
Monte Carlo VaR and Expected Shortfall.

Three methods, and the differences between them are the point.

  FILTERED HISTORICAL SIMULATION (the default)
      Standardise the historical returns by the fitted GARCH
      volatility, resample those standardised residuals with
      replacement, and propagate them back through the variance
      recursion. This is Barone-Adesi and Giannopoulos' FHS. It
      keeps the ACTUAL shape of the historical tail - skew,
      kurtosis, the specific bad days - while letting the
      volatility level be current rather than historical.
      Historical simulation gets the shape right and the level
      wrong; parametric gets the level right and the shape wrong.
      FHS gets both.

  PARAMETRIC GARCH-t
      Draw innovations from a standardised Student-t with the
      fitted degrees of freedom. Smooth, and only as good as the
      assumption that a t is the right shape.

  IID BOOTSTRAP
      Resample raw returns with no volatility model at all.
      Included as a baseline that should underperform in a
      volatility regime - it is historical simulation with a
      confidence interval attached.

WHY THIS MATTERS MORE THAN ANOTHER VaR NUMBER
----------------------------------------------
Every other estimator in this package is one-day. Multi-day risk
was available only through sqrt-time scaling, which assumes iid
returns and ignores that volatility mean-reverts - so it
OVERSTATES risk when current volatility is above its long-run
level and understates it below. Monte Carlo propagates the actual
variance recursion forward, so the term structure of risk comes
out of the model instead of an assumption. `compare_to_sqrt_time`
quantifies the gap.

Simulation error is reported, not hidden. A 99% VaR from 20,000
paths rests on ~200 tail observations, and `MonteCarloResult`
carries the bootstrap standard error so nobody quotes the fourth
decimal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..models.garch import GarchResult, fit_garch


@dataclass
class MonteCarloResult:
    """VaR and ES at one horizon, with simulation error."""

    var: float
    expected_shortfall: float
    var_stderr: float
    confidence: float
    horizon_days: int
    n_simulations: int
    method: str
    terminal_returns: np.ndarray = field(repr=False, default=None)

    @property
    def var_ci95(self) -> tuple[float, float]:
        """95% interval for the VaR estimate itself, from simulation error."""
        return (self.var - 1.96 * self.var_stderr, self.var + 1.96 * self.var_stderr)

    def summary(self) -> str:
        lo, hi = self.var_ci95
        return "\n".join(
            [
                f"  {self.method}, {self.horizon_days}-day, "
                f"{self.confidence:.1%}, {self.n_simulations:,} paths",
                f"    VaR                 {self.var:.3%}  "
                f"(±{1.96 * self.var_stderr:.3%} simulation error)",
                f"    95% CI on the VaR   [{lo:.3%}, {hi:.3%}]",
                f"    Expected Shortfall  {self.expected_shortfall:.3%}",
            ]
        )


def _simulate_garch_paths(
    fit: GarchResult,
    innovations: np.ndarray,
    horizon: int,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Propagate the GARCH variance recursion forward along each path.

    `innovations` is an (n_sims, horizon) array of standardised
    shocks - where they come from is what distinguishes FHS from
    the parametric version.
    """
    sigma2_last = float(fit.conditional_variance.iloc[-1])
    z_last = float(fit.standardised_residuals.iloc[-1])
    eps_last = z_last * np.sqrt(sigma2_last)

    # One step ahead of the fitted sample, known at time T.
    sigma2 = np.full(
        n_sims, fit.omega + fit.alpha * eps_last**2 + fit.beta * sigma2_last
    )

    cumulative = np.zeros(n_sims)
    for h in range(horizon):
        eps = innovations[:, h] * np.sqrt(sigma2)
        cumulative += fit.mu + eps
        sigma2 = fit.omega + fit.alpha * eps**2 + fit.beta * sigma2

    return cumulative


def _var_es(
    terminal: np.ndarray, confidence: float, n_boot: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Quantile VaR, ES, and a bootstrap standard error for the VaR."""
    alpha = 1.0 - confidence
    q = float(np.quantile(terminal, alpha))
    tail = terminal[terminal <= q]
    es = float(tail.mean()) if len(tail) else q

    # Simulation error on the quantile, by resampling the paths.
    n = len(terminal)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        boot[b] = np.quantile(terminal[rng.integers(0, n, n)], alpha)
    stderr = float(boot.std(ddof=1))

    return -q, -es, stderr


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
    n_simulations: int = 20000,
    method: str = "fhs",
    dist: str = "t",
    seed: int | None = 42,
    n_boot: int = 200,
) -> MonteCarloResult:
    """
    Monte Carlo VaR and ES.

    Parameters
    ----------
    method
        "fhs"        filtered historical simulation (default)
        "parametric" GARCH-t innovations
        "bootstrap"  iid resampling of raw returns, no vol model
    horizon
        Trading days. The variance recursion is propagated, so the
        term structure is the model's rather than sqrt-time's.
    seed
        Fixed by default. A VaR number that changes every time it
        is computed is not a number anyone can check.
    """
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0.5, 1.0), got {confidence}")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 250:
        raise ValueError(f"need at least 250 observations, got {len(r)}")

    rng = np.random.default_rng(seed)

    if method == "bootstrap":
        draws = rng.choice(r.to_numpy(), size=(n_simulations, horizon), replace=True)
        terminal = draws.sum(axis=1)

    elif method in ("fhs", "parametric"):
        fit = fit_garch(r, dist=dist, mean="zero")

        if method == "fhs":
            z = fit.standardised_residuals.dropna().to_numpy()
            z = z[np.isfinite(z)]
            # Centre and scale so the resampled shocks have the unit
            # variance the recursion assumes.
            z = (z - z.mean()) / z.std(ddof=1)
            innovations = rng.choice(z, size=(n_simulations, horizon), replace=True)
        else:
            nu = max(fit.nu or 8.0, 2.5) if dist == "t" else None
            if nu is None:
                innovations = rng.standard_normal((n_simulations, horizon))
            else:
                innovations = rng.standard_t(nu, (n_simulations, horizon)) * np.sqrt(
                    (nu - 2) / nu
                )

        terminal = _simulate_garch_paths(
            fit, innovations, horizon, n_simulations, rng
        )
    else:
        raise ValueError("method must be 'fhs', 'parametric' or 'bootstrap'")

    var, es, stderr = _var_es(terminal, confidence, n_boot, rng)

    return MonteCarloResult(
        var=var,
        expected_shortfall=es,
        var_stderr=stderr,
        confidence=confidence,
        horizon_days=horizon,
        n_simulations=n_simulations,
        method=method,
        terminal_returns=terminal,
    )


def term_structure(
    returns: pd.Series,
    horizons: list[int],
    confidence: float = 0.99,
    n_simulations: int = 20000,
    method: str = "fhs",
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    VaR and ES across horizons, with the sqrt-time comparison.

    The `sqrt_time` column is the one-day figure scaled by
    sqrt(h); `ratio` is Monte Carlo divided by it. A ratio below 1
    means sqrt-time is OVERSTATING risk, which happens when current
    volatility sits above its long-run level and the model expects
    it to mean-revert over the horizon.
    """
    rows = []
    one_day = None

    for h in sorted(horizons):
        res = monte_carlo_var(
            returns,
            confidence=confidence,
            horizon=h,
            n_simulations=n_simulations,
            method=method,
            seed=seed,
        )
        if h == 1 or one_day is None:
            one_day = res.var

        sqrt_time = one_day * np.sqrt(h)
        rows.append(
            {
                "horizon_days": h,
                "mc_var": res.var,
                "mc_es": res.expected_shortfall,
                "stderr": res.var_stderr,
                "sqrt_time": sqrt_time,
                "ratio": res.var / sqrt_time if sqrt_time else np.nan,
            }
        )

    return pd.DataFrame(rows).set_index("horizon_days")


def compare_methods(
    returns: pd.Series,
    confidence: float = 0.99,
    horizon: int = 1,
    n_simulations: int = 20000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """All three methods side by side at one horizon."""
    rows = []
    for method in ("fhs", "parametric", "bootstrap"):
        try:
            res = monte_carlo_var(
                returns,
                confidence=confidence,
                horizon=horizon,
                n_simulations=n_simulations,
                method=method,
                seed=seed,
            )
            rows.append(
                {
                    "method": method,
                    "var": res.var,
                    "expected_shortfall": res.expected_shortfall,
                    "stderr": res.var_stderr,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append({"method": method, "var": np.nan, "error": str(exc)})

    return pd.DataFrame(rows).set_index("method")
