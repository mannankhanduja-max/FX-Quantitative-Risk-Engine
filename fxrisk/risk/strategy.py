"""
Risk on a SIGNAL-DRIVEN return series.

Everything else in `fxrisk/risk/` measures the risk of holding an
asset. This module measures the risk of running a strategy on it,
and the two are not the same object. A strategy return is

    r_strat(t) = position(t) * r_asset(t),   position known at t-1

and that construction breaks two assumptions the standard VaR
battery makes.

THE POINT MASS AT ZERO
-----------------------
When the signal is flat the strategy return is not "small" - it is
exactly zero. A strategy flat on 35% of days has a return
distribution with an atom of probability 0.35 sitting on a single
point, and no continuous density describes that.

What a GARCH-t does when handed such a series is worth stating
precisely, because the intuitive answer is wrong. The obvious
guess is that the zeros look like calm days and drag the variance
DOWN, so the naive VaR comes out too low. Measured, the opposite
happens. On the simulated series in `tests/test_strategy_risk.py`:

                    nu       alpha+beta   excess kurtosis   sd
    asset         200.0        0.981          3.76        0.0072
    strategy        2.4        1.000          7.54        0.0058

The unconditional standard deviation does fall, as expected. But
interleaving zeros with full-size moves doubles the excess
kurtosis, and the fit absorbs that in two places: the degrees of
freedom collapse to the floor where the variance of a Student-t is
barely finite, and persistence is driven to the IGARCH boundary at
alpha + beta = 1.000, where shocks never decay at all. The
variance process becomes non-stationary. That inflated variance
path dominates, and the resulting 99% VaR is about 58% HIGHER than
the correct construction on the days the strategy is exposed.

So the naive construction is not conservative and it is not
lenient - it is degenerate, and which way it errs depends on the
flat share and the asset. That is the reason to avoid it.
Historical simulation survives the atom untouched, an empirical
quantile being indifferent to point masses.

THE CORRECT CONDITIONAL VARIANCE
---------------------------------
Because the position is known at t-1, it is a constant inside the
time-t conditional distribution, so

    Var(r_strat(t) | F(t-1)) = position(t)^2 * Var(r_asset(t) | F(t-1))

The volatility model belongs on the ASSET, and the position scales
its output. With position in {-1, 0, +1} the scale factor is 1 on
days in the market and 0 on days out of it. Fitting the model to
the strategy series instead contaminates the variance recursion
with the flat days - it is the same error as the point mass above,
seen from the other side.

`strategy_var_series` does it correctly. `naive_strategy_var_series`
does it the common wrong way, and is kept so the difference can be
measured rather than asserted; `compare_var_construction` reports
both through the same backtest.

WHAT ACTUALLY KILLS A STRATEGY
-------------------------------
Not a one-day VaR breach. A drawdown that runs long enough to end
the mandate. One-day risk and path risk are different questions,
and `drawdown_profile` answers the second: depth, duration, time
under water, and how much of each is explained by the strategy
being wrong versus being absent.

All of this is backtest-only, and on a strategy series that
warning binds harder than elsewhere: the signal was chosen with
hindsight over this same sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtesting import BacktestResult, backtest_var
from .var import garch_var_series


# ============================================================
# Construction
# ============================================================

def strategy_returns(
    signal: pd.Series,
    asset_returns: pd.Series,
    cost_per_turn: float = 0.0,
) -> pd.DataFrame:
    """
    Combine a signal with asset returns into a strategy return series.

    `signal` must ALREADY be knowable at the close of t-1 - that is
    the contract `fxrisk.indicators.signal` satisfies by shifting.
    This function does not shift again; doing so silently inside a
    risk module would hide a one-bar error rather than prevent one.

    Columns
    -------
    position    the signal, aligned to the return index
    gross       position * asset return
    turnover    |position(t) - position(t-1)|, in units of notional
    cost        turnover * cost_per_turn
    net         gross - cost
    """
    if cost_per_turn < 0:
        raise ValueError("cost_per_turn must be non-negative")

    pos, r = signal.align(asset_returns, join="inner")
    pos = pos.astype("float64").fillna(0.0)
    r = r.astype("float64")

    gross = pos * r
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * float(cost_per_turn)

    return pd.DataFrame(
        {
            "position": pos,
            "gross": gross,
            "turnover": turnover,
            "cost": cost,
            "net": gross - cost,
        }
    )


def exposure_summary(position: pd.Series) -> dict:
    """How much of the time the strategy is actually in the market."""
    pos = pd.Series(position, dtype="float64").dropna()
    n = len(pos)
    if n == 0:
        raise ValueError("empty position series")

    flat = float((pos == 0).mean())
    return {
        "observations": n,
        "long_share": float((pos > 0).mean()),
        "short_share": float((pos < 0).mean()),
        "flat_share": flat,
        "time_in_market": 1.0 - flat,
        "flips": int((pos.diff().fillna(0) != 0).sum()),
        "mean_abs_position": float(pos.abs().mean()),
    }


# ============================================================
# VaR, done two ways
# ============================================================

def strategy_var_series(
    signal: pd.Series,
    asset_returns: pd.Series,
    confidence: float = 0.99,
    dist: str = "t",
    window: int = 750,
    refit_every: int = 21,
) -> pd.DataFrame:
    """
    Position-scaled GARCH VaR - the correct construction.

    The volatility model is fitted to the ASSET return series, where
    the flat days do not exist and the variance recursion sees an
    uninterrupted process. The forecast is then scaled by
    |position(t)|, which is known at t-1.

    On a flat day the VaR is exactly zero. That is not a modelling
    artefact to be smoothed away - a strategy holding nothing cannot
    lose money, and a breach test should treat a zero-VaR day as a
    day on which no loss was possible rather than as a forecast of
    calm.

    Columns: var, expected_shortfall, position, asset_var.
    """
    asset = garch_var_series(
        asset_returns,
        confidence=confidence,
        dist=dist,
        window=window,
        refit_every=refit_every,
    )

    pos = signal.reindex(asset.index).astype("float64").fillna(0.0)
    scale = pos.abs()

    out = pd.DataFrame(
        {
            "var": asset["var"] * scale,
            "expected_shortfall": asset["expected_shortfall"] * scale,
            "position": pos,
            "asset_var": asset["var"],
        }
    )
    out.attrs["fit_failures"] = asset.attrs.get("fit_failures", 0)
    out.attrs["construction"] = "position-scaled asset GARCH"
    return out


def naive_strategy_var_series(
    strategy_return: pd.Series,
    confidence: float = 0.99,
    dist: str = "t",
    window: int = 750,
    refit_every: int = 21,
) -> pd.DataFrame:
    """
    GARCH fitted directly to the strategy series - the common error.

    Kept so the cost of the mistake can be measured rather than
    asserted. The flat days enter the variance recursion as genuine
    observations; the resulting excess kurtosis drives the fitted
    degrees of freedom to their floor and persistence to the IGARCH
    boundary, so the variance process stops being stationary. See
    the module docstring for the measured numbers - the net effect
    on VaR is an OVERSTATEMENT, which is the opposite of what the
    lower unconditional variance suggests.
    """
    out = garch_var_series(
        strategy_return,
        confidence=confidence,
        dist=dist,
        window=window,
        refit_every=refit_every,
    )
    out.attrs["construction"] = "GARCH fitted to strategy returns"
    return out


def _rolling_historical(
    returns: pd.Series, confidence: float, window: int = 500
) -> pd.Series:
    """Trailing empirical quantile, causal (shifted one bar)."""
    alpha = 1.0 - confidence
    return (
        returns.rolling(window, min_periods=window)
        .quantile(alpha)
        .shift(1)
        .mul(-1.0)
        .rename("historical_var")
    )


def compare_var_construction(
    signal: pd.Series,
    asset_returns: pd.Series,
    confidence: float = 0.99,
    cost_per_turn: float = 0.0,
    in_market_only: bool = True,
) -> tuple[pd.DataFrame, list[BacktestResult]]:
    """
    Both constructions, plus historical simulation, through one backtest.

    WHY FLAT DAYS ARE EXCLUDED BY DEFAULT
    --------------------------------------
    Kupiec asks whether the breach rate equals 1 - c over n
    observations. On a flat day the VaR is zero and the return is
    zero, so a breach is not merely unlikely - it is impossible.
    Leaving those days in inflates n, and therefore inflates the
    expected breach count, while the achievable breach count is
    unchanged. A perfectly calibrated model then looks like it is
    systematically under-breaching, and a correct model gets
    rejected for a reason that has nothing to do with the model.
    A strategy flat 30% of the time has its expected breaches
    overstated by roughly 30%.

    So the default backtests the days the strategy was actually
    exposed. The cost is that Christoffersen's independence test
    now reads "consecutive in-market days" rather than consecutive
    calendar days - defensible for a strategy, since a flat stretch
    is not evidence about clustering either way, but it is a change
    of meaning and not a free one. Pass in_market_only=False to see
    the diluted version.
    """
    frame = strategy_returns(signal, asset_returns, cost_per_turn=cost_per_turn)
    realised = frame["net"]

    correct = strategy_var_series(signal, asset_returns, confidence=confidence)
    naive = naive_strategy_var_series(realised, confidence=confidence)
    hist = _rolling_historical(realised, confidence=confidence, window=500)

    exposed = frame["position"] != 0

    def _mask(series: pd.Series) -> pd.Series:
        if not in_market_only:
            return series
        keep = exposed.reindex(series.index, fill_value=False)
        return series[keep]

    realised_bt = _mask(realised)
    candidates = [
        (_mask(correct["var"]), "position_scaled_garch"),
        (_mask(naive["var"]), "naive_garch_on_strategy"),
        (_mask(hist), "historical_on_strategy"),
    ]

    results: list[BacktestResult] = []
    skipped: list[str] = []
    for var_series, name in candidates:
        try:
            results.append(backtest_var(realised_bt, var_series, confidence, name))
        except ValueError as exc:
            # Reported as absent rather than silently dropped.
            skipped.append(f"{name}: {exc}")

    rows = [
        {
            "estimator": res.method,
            "observations": res.n_observations,
            "breaches": res.n_breaches,
            "expected": round(res.expected_breaches, 1),
            "kupiec_p": round(res.kupiec_pvalue, 3),
            "christoffersen_p": round(res.christoffersen_ind_pvalue, 3),
            "verdict": "PASS" if res.passed() else "FAIL",
        }
        for res in results
    ]

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.set_index("estimator")
    out.attrs["in_market_only"] = in_market_only
    out.attrs["flat_share"] = float((~exposed).mean())
    out.attrs["skipped"] = skipped
    return out, results


# ============================================================
# Path risk
# ============================================================

@dataclass
class DrawdownProfile:
    """Depth and duration of the worst stretches."""

    max_drawdown: float
    max_duration_days: int
    current_drawdown: float
    time_under_water: float
    worst_periods: pd.DataFrame

    def summary(self) -> str:
        return "\n".join(
            [
                f"  max drawdown         {self.max_drawdown:.2%}",
                f"  longest drawdown     {self.max_duration_days} trading days",
                f"  time under water     {self.time_under_water:.1%} of the sample",
                f"  currently            {self.current_drawdown:.2%} from the peak",
            ]
        )


def drawdown_profile(returns: pd.Series, top_n: int = 5) -> DrawdownProfile:
    """
    Drawdown depth, duration and time under water.

    Duration matters as much as depth and is reported far less
    often. A 12% drawdown recovered in three weeks and a 12%
    drawdown that takes fourteen months are the same number and
    completely different experiences - only one of them ends a
    mandate.

    Compounded, not summed: a strategy is run on capital, and the
    arithmetic sum of returns overstates recovery after a loss.

    `max_duration_days` is the longest stretch in the sample, not
    the duration of the deepest one - those are different periods
    more often than not.
    """
    r = pd.Series(returns, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        raise ValueError("empty return series")

    equity = (1.0 + r).cumprod()
    # The running peak must include the starting capital. Without
    # the floor at 1.0 the peak on day one is the post-loss equity
    # itself, so a strategy that opens by losing half its capital
    # reports a drawdown of zero.
    peak = equity.cummax().clip(lower=1.0)
    dd = equity / peak - 1.0

    under = dd < 0
    # Label each contiguous stretch below the previous peak.
    group = (~under).cumsum()
    periods = []
    for _, seg in dd[under].groupby(group[under]):
        periods.append(
            {
                "start": seg.index[0],
                "trough": seg.idxmin(),
                "end": seg.index[-1],
                "depth": float(seg.min()),
                "duration_days": int(len(seg)),
            }
        )

    all_periods = pd.DataFrame(periods)
    if all_periods.empty:
        all_periods = pd.DataFrame(
            columns=["start", "trough", "end", "depth", "duration_days"]
        )
        longest = 0
    else:
        longest = int(all_periods["duration_days"].max())

    worst = (
        all_periods.sort_values("depth").head(top_n).reset_index(drop=True)
        if not all_periods.empty
        else all_periods
    )

    return DrawdownProfile(
        max_drawdown=float(dd.min()),
        max_duration_days=longest,
        current_drawdown=float(dd.iloc[-1]),
        time_under_water=float(under.mean()),
        worst_periods=worst,
    )


def loss_attribution(frame: pd.DataFrame) -> dict:
    """
    Split the strategy's losing days into wrong-way and cost-only.

    A day can lose money two ways: the position was on the wrong
    side of the market, or the position barely moved and the
    turnover cost ate the result. They call for different fixes -
    the first is a signal problem, the second a trading-frequency
    problem - and a single P&L number hides which one you have.
    """
    for col in ("gross", "cost", "net", "position"):
        if col not in frame.columns:
            raise ValueError(f"frame is missing the '{col}' column")

    net = frame["net"]
    losing = net < 0
    n_loss = int(losing.sum())
    if n_loss == 0:
        return {"losing_days": 0, "flat_day_share": float((frame["position"] == 0).mean())}

    wrong_way = losing & (frame["gross"] < 0)
    cost_only = losing & (frame["gross"] >= 0)

    return {
        "losing_days": n_loss,
        "wrong_way_days": int(wrong_way.sum()),
        "cost_only_days": int(cost_only.sum()),
        "cost_only_share": float(cost_only.sum() / n_loss),
        "total_cost_drag": float(frame["cost"].sum()),
        "flat_day_share": float((frame["position"] == 0).mean()),
    }
