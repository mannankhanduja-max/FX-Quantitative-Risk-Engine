"""
DCC-GARCH (Engle, 2002): dynamic conditional correlation.

EWMA covariance already lets correlation move, but it forces every
pair to move with the same decay factor and ties the correlation
dynamics to the variance dynamics. DCC separates the two:

  Stage 1  Fit a univariate GARCH(1,1) to each asset and take the
           standardised residuals z_{i,t} = eps_{i,t} / sigma_{i,t}.

  Stage 2  Let Q_t evolve as

               Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}

           and normalise it to a correlation matrix

               R_t = diag(Q_t)^-1/2  Q_t  diag(Q_t)^-1/2

           with Qbar the unconditional correlation of z, and
           a, b >= 0, a + b < 1 estimated by quasi-maximum
           likelihood.

  Then     H_t = D_t R_t D_t,  D_t = diag(sigma_{1,t} ... sigma_{n,t})

Why this matters for a risk engine: correlations rise in crises.
A sample correlation matrix estimated over a calm decade will
understate portfolio risk in exactly the state where the number is
load-bearing. DCC lets the estimate move, and the estimated `a`
tells you how fast.

Caveat worth stating plainly: two-stage DCC is consistent but not
efficient, standard errors from stage 2 ignore stage-1 estimation
error, and the scalar (a, b) specification forces every pair to
share the same correlation dynamics. For a small FX book that is a
reasonable trade; for a large heterogeneous book it is a real
limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from .garch import GarchResult, fit_garch


@dataclass
class DccResult:
    """Fitted DCC-GARCH."""

    a: float
    b: float
    univariate: dict[str, GarchResult] = field(repr=False)
    correlations: dict[pd.Timestamp, pd.DataFrame] = field(repr=False)
    covariances: dict[pd.Timestamp, pd.DataFrame] = field(repr=False)
    unconditional_correlation: pd.DataFrame = field(repr=False)
    loglikelihood: float = 0.0
    converged: bool = True

    @property
    def persistence(self) -> float:
        return self.a + self.b

    def correlation_series(self, asset_i: str, asset_j: str) -> pd.Series:
        """Time series of the conditional correlation between two assets."""
        idx = list(self.correlations.keys())
        vals = [self.correlations[d].loc[asset_i, asset_j] for d in idx]
        return pd.Series(vals, index=pd.DatetimeIndex(idx), name=f"rho_{asset_i}_{asset_j}")

    def last_covariance(self) -> pd.DataFrame:
        return self.covariances[list(self.covariances.keys())[-1]]

    def summary(self) -> str:
        lines = [
            "DCC-GARCH(1,1)",
            f"  assets            {len(self.univariate)}",
            f"  observations      {len(self.correlations)}",
            f"  converged         {self.converged}",
            f"  a                 {self.a:.4f}",
            f"  b                 {self.b:.4f}",
            f"  a + b             {self.persistence:.4f}",
            "",
            "  Univariate persistence:",
        ]
        for name, res in self.univariate.items():
            lines.append(f"    {name:<12} {res.persistence:.4f}")
        return "\n".join(lines)


def _dcc_negative_loglikelihood(
    params: np.ndarray, z: np.ndarray, qbar: np.ndarray
) -> float:
    a, b = params
    if a < 0 or b < 0 or a + b >= 0.9999:
        return 1e10

    n_obs, n_assets = z.shape
    q = qbar.copy()
    ll = 0.0

    for t in range(n_obs):
        d = np.sqrt(np.diag(q))
        if np.any(d <= 0) or not np.all(np.isfinite(d)):
            return 1e10
        r_mat = q / np.outer(d, d)
        np.fill_diagonal(r_mat, 1.0)

        try:
            sign, logdet = np.linalg.slogdet(r_mat)
            if sign <= 0 or not np.isfinite(logdet):
                return 1e10
            r_inv = np.linalg.inv(r_mat)
        except np.linalg.LinAlgError:
            return 1e10

        zt = z[t]
        ll += -0.5 * (logdet + zt @ r_inv @ zt - zt @ zt)

        outer = np.outer(zt, zt)
        q = (1 - a - b) * qbar + a * outer + b * q

    return -ll if np.isfinite(ll) else 1e10


def fit_dcc(
    returns: pd.DataFrame,
    dist: str = "t",
    mean: str = "zero",
) -> DccResult:
    """
    Two-stage DCC-GARCH estimation.

    Parameters
    ----------
    returns
        Decimal returns, one column per asset. Rows with any missing
        value are dropped, so align the panel before calling.
    dist
        Innovation distribution for the stage-1 univariate models.
    mean
        "zero" or "constant" for the stage-1 mean equations.

    Returns
    -------
    DccResult
    """
    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[1] < 2:
        raise ValueError("DCC needs at least two assets")
    if len(df) < 250:
        raise ValueError(f"DCC needs at least 250 aligned observations, got {len(df)}")

    # ---- Stage 1: univariate GARCH per asset -------------------
    univariate: dict[str, GarchResult] = {}
    z_cols = {}
    sigma_cols = {}

    for col in df.columns:
        res = fit_garch(df[col], dist=dist, mean=mean)
        univariate[col] = res
        z_cols[col] = res.standardised_residuals
        sigma_cols[col] = np.sqrt(res.conditional_variance)

    z_df = pd.DataFrame(z_cols).dropna()
    sigma_df = pd.DataFrame(sigma_cols).reindex(z_df.index)

    z = z_df.to_numpy()
    qbar = np.corrcoef(z, rowvar=False)

    # ---- Stage 2: correlation dynamics -------------------------
    res2 = optimize.minimize(
        _dcc_negative_loglikelihood,
        np.array([0.02, 0.95]),
        args=(z, qbar),
        method="SLSQP",
        bounds=[(0.0, 0.5), (0.0, 0.999)],
        constraints=[{"type": "ineq", "fun": lambda p: 0.9999 - (p[0] + p[1])}],
        options={"maxiter": 500, "ftol": 1e-9},
    )

    a, b = float(res2.x[0]), float(res2.x[1])

    # ---- Rebuild the correlation and covariance paths ----------
    cols = list(df.columns)
    q = qbar.copy()
    correlations: dict[pd.Timestamp, pd.DataFrame] = {}
    covariances: dict[pd.Timestamp, pd.DataFrame] = {}

    for t, date in enumerate(z_df.index):
        d = np.sqrt(np.diag(q))
        r_mat = q / np.outer(d, d)
        np.fill_diagonal(r_mat, 1.0)

        correlations[date] = pd.DataFrame(r_mat.copy(), index=cols, columns=cols)

        sig = sigma_df.loc[date].to_numpy()
        h = np.outer(sig, sig) * r_mat
        covariances[date] = pd.DataFrame(h, index=cols, columns=cols)

        zt = z[t]
        q = (1 - a - b) * qbar + a * np.outer(zt, zt) + b * q

    return DccResult(
        a=a,
        b=b,
        univariate=univariate,
        correlations=correlations,
        covariances=covariances,
        unconditional_correlation=pd.DataFrame(qbar, index=cols, columns=cols),
        loglikelihood=float(-res2.fun),
        converged=bool(res2.success),
    )


# ===============================================================
# Walk-forward DCC
# ===============================================================

def rolling_dcc_covariance(
    returns: pd.DataFrame,
    window: int = 750,
    refit_every: int = 126,
    dist: str = "t",
    min_obs: int = 400,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Walk-forward one-step-ahead DCC covariance forecasts.

    `fit_dcc` is an IN-SAMPLE estimator: it uses the whole panel, so
    its correlation path cannot be used for a VaR backtest without
    leaking the future into every historical date. This function is
    the honest version - at each date t the parameters and the
    recursions have only seen returns strictly before t.

    Both layers are advanced daily and refitted periodically:

      univariate  sigma2_{i,t} = omega_i + alpha_i eps^2 + beta_i sigma2
      correlation Q_t = (1-a-b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}

    Refitting is expensive - each fit runs n_assets univariate MLEs
    plus a bivariate-likelihood optimisation over the whole window -
    so `refit_every` defaults to 126 (roughly semi-annual) rather
    than the 21 used for univariate GARCH. Between refits the
    parameters are stale but never forward-looking, which is the
    trade a desk actually makes.

    Returns
    -------
    dict keyed by date -> covariance matrix forecast for that date.
    """
    df = returns.replace([np.inf, -np.inf], np.nan).dropna()
    n, k = df.shape
    if n <= min_obs:
        raise ValueError(f"need more than min_obs={min_obs} rows, got {n}")
    if k < 2:
        raise ValueError("DCC needs at least two assets")

    values = df.to_numpy()
    cols = list(df.columns)

    out: dict[pd.Timestamp, pd.DataFrame] = {}

    params = None          # per-asset (omega, alpha, beta, mu)
    a = b = None
    qbar = None
    sigma2 = None          # current per-asset variance forecast
    q = None               # current Q matrix
    last_fit = -10**9
    fit_failures = 0

    for t in range(min_obs, n):
        if params is None or (t - last_fit) >= refit_every:
            start = max(0, t - window)
            try:
                fitted = fit_dcc(df.iloc[start:t], dist=dist, mean="zero")
                params = [
                    (
                        fitted.univariate[c].omega,
                        fitted.univariate[c].alpha,
                        fitted.univariate[c].beta,
                        fitted.univariate[c].mu,
                    )
                    for c in cols
                ]
                a, b = fitted.a, fitted.b
                qbar = fitted.unconditional_correlation.to_numpy()

                sigma2 = np.array(
                    [
                        float(fitted.univariate[c].conditional_variance.iloc[-1])
                        for c in cols
                    ]
                )
                last_date = list(fitted.correlations.keys())[-1]
                q = fitted.correlations[last_date].to_numpy() * 1.0

                # Advance both recursions to produce the forecast for t.
                r_prev = values[t - 1]
                eps = np.array([r_prev[i] - params[i][3] for i in range(k)])
                z_prev = eps / np.sqrt(sigma2)
                sigma2 = np.array(
                    [
                        params[i][0] + params[i][1] * eps[i] ** 2 + params[i][2] * sigma2[i]
                        for i in range(k)
                    ]
                )
                q = (1 - a - b) * qbar + a * np.outer(z_prev, z_prev) + b * q
                last_fit = t
            except Exception:
                fit_failures += 1
                if params is None:
                    continue

        # ---- forecast for date t, from information through t-1 ----
        d = np.sqrt(np.diag(q))
        r_mat = q / np.outer(d, d)
        np.fill_diagonal(r_mat, 1.0)
        sig = np.sqrt(sigma2)
        h = np.outer(sig, sig) * r_mat
        out[df.index[t]] = pd.DataFrame(h, index=cols, columns=cols)

        # ---- advance one step using the return actually observed at t ----
        r_t = values[t]
        eps = np.array([r_t[i] - params[i][3] for i in range(k)])
        z_t = eps / np.sqrt(sigma2)
        sigma2 = np.array(
            [
                params[i][0] + params[i][1] * eps[i] ** 2 + params[i][2] * sigma2[i]
                for i in range(k)
            ]
        )
        q = (1 - a - b) * qbar + a * np.outer(z_t, z_t) + b * q

    return out
