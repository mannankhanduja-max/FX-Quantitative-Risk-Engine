"""
Regime gates: GARCH volatility, conditional VaR, correlation alignment.

These do not change WHERE the strategy enters. They change WHETHER
it enters, by refusing setups taken under conditions the gate
distrusts. That distinction matters more than it sounds.

WHAT A FILTER CAN AND CANNOT DO
--------------------------------
A filter selects a subsample. It cannot create expectancy that is
not present somewhere inside the population it selects from - it
can only find the part of the population where the expectancy
already lived, and concentrate it.

So a filter is worth adding when there is reason to believe the
edge is CONDITIONAL: present in some states, absent or negative
in others. It is not a repair for a rule with no edge anywhere,
and applied to one it will still produce an improvement, because
any partition of a noisy sample has a better half. That improvement
is the filter fitting noise, and the only defence against
mistaking it for a discovery is to choose the thresholds on one
block of data and measure them on another. `breakout_test.py
--holdout` does exactly that and nothing here should be read
without it.

THE THREE GATES

  VOLATILITY REGIME. GARCH(1,1) with Student-t innovations, fit
  walk-forward, gives a one-step-ahead conditional sigma per bar.
  The gate refuses the extremes: a dead market where a breakout
  has nothing behind it, and a panic where the stop distance is
  meaningless because the next bar can be anywhere.

  CONDITIONAL VaR. The same GARCH sigma turned into a one-bar
  Student-t quantile. This is NOT independent information from
  the volatility gate - VaR here is a monotone function of sigma,
  so a VaR cap and a sigma cap refuse almost the same bars. It is
  kept separate because the threshold is expressed in a unit a
  risk desk actually uses, not because it adds a second view.
  Anyone reading a result where both gates are on should know
  they are counting one gate twice.

  CORRELATION ALIGNMENT. The genuinely different one. For each
  instrument, its most-correlated partner is identified from a
  rolling EWMA correlation, and the trade is taken only when that
  partner's recent move points the same way once the sign of the
  correlation is applied. A long EUR/USD while gold is falling
  hard, with the two positively correlated, is a trade the cross
  section disagrees with.

  This is the only gate carrying information the instrument's own
  price does not already contain, and it is also the one most
  exposed to the Epps effect: correlations measured on short
  intervals are biased toward zero by non-synchronous quoting, so
  `rho_min` is doing more work than it looks like it is.

LOOK-AHEAD
-----------
Every series here is shifted so the value at bar t is computable
from bars strictly before t. The GARCH path is walk-forward with
periodic refits, never a single fit over the whole sample - a
full-sample fit would let the parameters see the crisis they are
being asked to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from fxrisk.models.garch import rolling_garch_forecasts


@dataclass
class GateConfig:
    """Thresholds. All are quantiles of the instrument's own history."""

    # Refuse bars whose conditional sigma sits outside this band,
    # as quantiles of the trailing sigma distribution.
    sigma_low_q: float = 0.10
    sigma_high_q: float = 0.90

    # Refuse bars whose conditional VaR exceeds this quantile.
    var_cap_q: float = 0.90
    var_conf: float = 0.99

    # Correlation alignment.
    rho_min: float = 0.30          # below this the partner says nothing
    partner_lookback: int = 8      # bars of partner move to read

    # Trailing window for the quantile thresholds, in bars.
    quantile_window: int = 2000

    def __post_init__(self) -> None:
        if not 0.0 <= self.sigma_low_q < self.sigma_high_q <= 1.0:
            raise ValueError("need 0 <= sigma_low_q < sigma_high_q <= 1")
        if not 0.0 < self.var_cap_q <= 1.0:
            raise ValueError("var_cap_q must be in (0, 1]")
        if not 0.5 < self.var_conf < 1.0:
            raise ValueError("var_conf must be in (0.5, 1)")
        if not 0.0 <= self.rho_min < 1.0:
            raise ValueError("rho_min must be in [0, 1)")
        if self.partner_lookback < 1:
            raise ValueError("partner_lookback must be at least 1")
        if self.quantile_window < 100:
            raise ValueError("quantile_window is too short to be a quantile")


def garch_path(
    returns: pd.Series,
    window: int = 2000,
    refit_every: int = 1000,
    dist: str = "t",
    min_obs: int = 500,
) -> pd.DataFrame:
    """
    Walk-forward conditional sigma and degrees of freedom.

    `refit_every` is far larger than the daily engine's 21 because
    there are ~49,000 bars here rather than ~4,900 days, and an
    MLE every 21 bars would be 2,300 fits per instrument. The
    variance still recurses every bar; only the PARAMETERS are
    held between refits, which is the standard compromise and is
    what `GARCH_REFIT_EVERY` already encodes at daily frequency.
    """
    out = rolling_garch_forecasts(
        returns, window=window, refit_every=refit_every,
        dist=dist, min_obs=min_obs,
    )
    frame = pd.DataFrame(index=returns.index)
    frame["sigma"] = np.sqrt(out["variance"])
    frame["nu"] = out["nu"] if "nu" in out.columns else np.nan
    return frame


def conditional_var(sigma: pd.Series, nu: pd.Series,
                    conf: float = 0.99) -> pd.Series:
    """
    One-bar Student-t VaR as a positive fraction of price.

    The t quantile is scaled by sqrt((nu-2)/nu) so that sigma
    remains the standard deviation rather than the t scale
    parameter. Skipping that rescaling is a common error and
    inflates VaR by 10-25% at the degrees of freedom these fits
    typically land on.
    """
    nu_f = pd.Series(nu, index=sigma.index).astype("float64")
    nu_f = nu_f.where(nu_f > 2.0, np.nan)
    q = pd.Series(stats.t.ppf(1.0 - conf, nu_f.to_numpy()), index=sigma.index)
    scale = np.sqrt((nu_f - 2.0) / nu_f)
    return (-(q * scale) * sigma).rename("var")


def ewma_correlation_path(
    returns: pd.DataFrame, lam: float = 0.97
) -> dict[tuple[str, str], pd.Series]:
    """
    Rolling EWMA correlation for every pair, as a path not a point.

    `correlation.ewma_correlation` gives the matrix as at the end
    of the sample, which is the wrong object for a per-bar gate.
    This recurses the covariance directly:

        cov_t = lam * cov_{t-1} + (1 - lam) * r_i,t * r_j,t

    A longer lambda than the RiskMetrics 0.94 is used because at
    15 minutes 0.94 has an effective memory of about 16 bars,
    which is four hours - short enough that the "correlation" is
    mostly one or two shocks.

    THE EPPS EFFECT applies and is not cosmetic. Correlations
    estimated on short intervals are biased toward zero by
    non-synchronous quoting, and these four instruments do not
    trade on one clock: the Nasdaq-100 is quiet while Tokyo is
    the only thing open. The numbers here are lower bounds on the
    economic relationship.
    """
    cols = list(returns.columns)
    var = {c: returns[c].pow(2).ewm(alpha=1 - lam, adjust=False).mean()
           for c in cols}

    out = {}
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            i, j = cols[a], cols[b]
            cov = (returns[i] * returns[j]).ewm(alpha=1 - lam,
                                                adjust=False).mean()
            den = np.sqrt(var[i] * var[j]).replace(0.0, np.nan)
            rho = (cov / den).clip(-1.0, 1.0)
            out[(i, j)] = rho.shift(1)          # causal
    return out


def alignment(
    target: str,
    returns: pd.DataFrame,
    rho_paths: dict[tuple[str, str], pd.Series],
    cfg: GateConfig,
) -> pd.DataFrame:
    """
    The direction the cross section implies for `target`, per bar.

    The partner is whichever other instrument currently has the
    largest |rho| with the target - it is not fixed in advance,
    because which pair is informative changes with the regime.

    Returns `implied` (+1/-1/0) and `rho` (the signed correlation
    with the chosen partner). A trade is aligned when its own side
    equals `implied`; `implied == 0` means no partner cleared
    `rho_min` and the gate abstains rather than guesses.
    """
    others = [c for c in returns.columns if c != target]
    idx = returns.index

    rho_mat = pd.DataFrame(index=idx, columns=others, dtype="float64")
    for o in others:
        key = (target, o) if (target, o) in rho_paths else (o, target)
        rho_mat[o] = rho_paths[key]

    # Early bars have no correlation estimate for any partner. A
    # bare idxmax on an all-NA row is deprecated and will raise in
    # a future pandas; more to the point it would name a partner
    # that does not exist. Those rows must resolve to "no partner",
    # which `alignment` reads as abstain.
    absr = rho_mat.abs()
    has_any = absr.notna().any(axis=1)
    best = absr[has_any].idxmax(axis=1).reindex(idx)
    strength = absr.max(axis=1)

    # Partner's move over the recent window, shifted so bar t uses
    # only information from before t.
    moves = {
        o: returns[o].rolling(cfg.partner_lookback).sum().shift(1)
        for o in others
    }
    move_mat = pd.DataFrame(moves, index=idx)

    pos = pd.Series(
        [
            (move_mat.at[t, b] if isinstance(b, str) else np.nan)
            for t, b in best.items()
        ],
        index=idx, dtype="float64",
    )
    signed = pd.Series(
        [
            (rho_mat.at[t, b] if isinstance(b, str) else np.nan)
            for t, b in best.items()
        ],
        index=idx, dtype="float64",
    )

    implied = np.sign(pos) * np.sign(signed)
    implied = implied.where(strength >= cfg.rho_min, 0.0).fillna(0.0)

    return pd.DataFrame({"implied": implied, "rho": signed,
                         "rho_abs": strength, "partner": best}, index=idx)


def gates(
    bars: pd.DataFrame,
    garch: pd.DataFrame,
    align: pd.DataFrame,
    cfg: GateConfig,
) -> pd.DataFrame:
    """
    One boolean column per gate, plus `all_gates`.

    Thresholds are TRAILING quantiles of the instrument's own
    history, never full-sample quantiles. A full-sample quantile
    would let a 2026 panic set the threshold that decides whether
    a 2024 bar was calm, which is look-ahead of the least obvious
    and most damaging kind - it does not show up as an impossible
    trade, only as a filter that was impossibly well calibrated.
    """
    idx = bars.index
    w = cfg.quantile_window

    sigma = garch["sigma"].reindex(idx)
    var = conditional_var(sigma, garch["nu"].reindex(idx), cfg.var_conf)

    lo = sigma.rolling(w, min_periods=w // 4).quantile(cfg.sigma_low_q)
    hi = sigma.rolling(w, min_periods=w // 4).quantile(cfg.sigma_high_q)
    cap = var.rolling(w, min_periods=w // 4).quantile(cfg.var_cap_q)

    out = pd.DataFrame(index=idx)
    out["sigma"] = sigma
    out["var"] = var
    out["g_regime"] = (sigma >= lo) & (sigma <= hi)
    out["g_var"] = var <= cap
    out["implied"] = align["implied"].reindex(idx).fillna(0.0)
    out["rho_abs"] = align["rho_abs"].reindex(idx)
    # Correlation alignment is direction-dependent, so it cannot be
    # collapsed into a single boolean here - `allow_for_side` does
    # that once the side is known.
    out["all_gates"] = out["g_regime"].fillna(False) & out["g_var"].fillna(False)
    return out


def allow_for_side(gate_frame: pd.DataFrame, side: float,
                   use_correlation: bool = True) -> pd.Series:
    """Entry permission for one direction."""
    ok = gate_frame["all_gates"].fillna(False)
    if use_correlation:
        ok = ok & (gate_frame["implied"] == float(np.sign(side)))
    return ok


def describe(gate_frame: pd.DataFrame) -> str:
    """How much each gate refuses, before any outcome is known."""
    n = len(gate_frame)
    if not n:
        return "  no bars"
    lines = [
        f"  bars                  {n}",
        f"  volatility regime     {gate_frame['g_regime'].mean():.1%} pass",
        f"  VaR cap               {gate_frame['g_var'].mean():.1%} pass",
        f"  both                  {gate_frame['all_gates'].mean():.1%} pass",
        f"  correlation says long {(gate_frame['implied'] > 0).mean():.1%}",
        f"  correlation says short{(gate_frame['implied'] < 0).mean():.1%}",
        f"  correlation abstains  {(gate_frame['implied'] == 0).mean():.1%}",
        f"  median |rho| to best  {gate_frame['rho_abs'].median():.3f}",
    ]
    return "\n".join(lines)
