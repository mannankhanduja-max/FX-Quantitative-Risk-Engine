"""
Stress testing against real historical episodes.

VaR answers "how bad is a normal bad day". It is estimated from a
sample that, by construction, contains few of the events that
actually break books. Stress testing asks a different question:
what would TODAY's portfolio have lost during a specific episode
that really happened?

The scenarios below are dated windows, not invented shocks. Each is
replayed by taking the actual asset returns over that window and
applying current portfolio weights. Where an asset did not exist or
has no data for a window, that is reported rather than silently
zero-filled - a stress result computed over half the book is worse
than no stress result, because it looks complete.

Two families are provided:

  Historical replay   Real returns from a real window. Honest, but
                      limited to what the data covers.

  Hypothetical shock  Instantaneous moves applied to positions, for
                      exposures with no usable history or for
                      "what if this were twice as bad" analysis.

Scenario windows are deliberately generous at the edges: crises
rarely start on the day the newspapers name.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    """A dated historical stress window."""

    key: str
    name: str
    start: str
    end: str
    description: str
    peak_day: str | None = None

    @property
    def start_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.start)

    @property
    def end_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.end)


# ---------------------------------------------------------------
# Scenario library
#
# Dates are the market episode, not the political event. Sources
# for the framing are the standard public record; verify the
# windows against your own data coverage before quoting results.
# ---------------------------------------------------------------

SCENARIOS: dict[str, Scenario] = {
    "gfc_2008": Scenario(
        key="gfc_2008",
        name="Global Financial Crisis (Lehman)",
        start="2008-09-01",
        end="2008-12-31",
        peak_day="2008-09-15",
        description=(
            "Lehman Brothers filed for bankruptcy on 15 September 2008. "
            "Funding markets froze, equity volatility reached record "
            "levels, and cross-asset correlations converged toward one - "
            "the diversification that portfolio optimisers had been "
            "counting on disappeared precisely when it was needed. "
            "Carry currencies sold off violently against the yen."
        ),
    ),
    "gfc_full": Scenario(
        key="gfc_full",
        name="Global Financial Crisis (full drawdown)",
        start="2007-10-09",
        end="2009-03-09",
        description=(
            "Peak to trough of the crisis, from the October 2007 equity "
            "high to the March 2009 low. Use this for drawdown and "
            "time-under-water analysis rather than for shock sizing; the "
            "Lehman window is the sharper test of a one-day VaR model."
        ),
    ),
    "snb_2015": Scenario(
        key="snb_2015",
        name="SNB removes the EUR/CHF floor",
        start="2015-01-12",
        end="2015-01-30",
        peak_day="2015-01-15",
        description=(
            "On 15 January 2015 the Swiss National Bank abandoned its "
            "1.20 EUR/CHF floor without warning. EUR/CHF fell roughly 30% "
            "intraday. The canonical demonstration that a currency peg "
            "produces years of near-zero measured volatility followed by "
            "a move no VaR model calibrated on that history could "
            "anticipate. Essential for any FX risk engine."
        ),
    ),
    "cny_2015": Scenario(
        key="cny_2015",
        name="CNY devaluation and August 2015 selloff",
        start="2015-08-10",
        end="2015-09-30",
        peak_day="2015-08-24",
        description=(
            "The PBoC changed its fixing mechanism on 11 August 2015, "
            "devaluing the yuan. Contagion ran through Asian and "
            "commodity currencies and culminated in the 24 August "
            "'Black Monday' global equity selloff. A useful test of "
            "whether a model registers a regime change originating "
            "outside its own asset class."
        ),
    ),
    "brexit_2016": Scenario(
        key="brexit_2016",
        name="Brexit referendum",
        start="2016-06-20",
        end="2016-07-15",
        peak_day="2016-06-24",
        description=(
            "Sterling fell around 8% against the dollar on 24 June 2016, "
            "its largest single-day move in the modern floating era. An "
            "event-risk test: the date was known in advance, the outcome "
            "was not, and implied volatility was priced accordingly - "
            "which historical VaR had no way of reflecting."
        ),
    ),
    "covid_2020": Scenario(
        key="covid_2020",
        name="COVID-19 crash",
        start="2020-02-19",
        end="2020-03-23",
        peak_day="2020-03-16",
        description=(
            "From the 19 February 2020 equity peak to the 23 March "
            "trough. The fastest major drawdown on record, with a dash "
            "for dollars that broke the usual safe-haven relationships "
            "and a WTI curve that later traded negative. Tests how "
            "quickly a volatility model adapts."
        ),
    ),
    "gilt_2022": Scenario(
        key="gilt_2022",
        name="UK gilt / LDI crisis",
        start="2022-09-21",
        end="2022-10-17",
        peak_day="2022-09-26",
        description=(
            "The 23 September 2022 mini-budget triggered a disorderly "
            "gilt selloff; sterling hit an all-time low against the "
            "dollar on 26 September and the Bank of England intervened. "
            "A liquidity-spiral scenario rather than a pure repricing."
        ),
    ),
}


@dataclass
class ScenarioResult:
    """Result of replaying one scenario."""

    scenario: Scenario
    total_return: float
    worst_day: float
    worst_day_date: pd.Timestamp | None
    volatility_annualised: float
    max_drawdown: float
    n_days: int
    asset_returns: pd.Series = field(repr=False)
    missing_assets: list[str] = field(default_factory=list)
    coverage: float = 1.0

    @property
    def complete(self) -> bool:
        return not self.missing_assets

    def summary(self) -> str:
        flag = "" if self.complete else "  [PARTIAL COVERAGE]"
        lines = [
            f"{self.scenario.name} ({self.scenario.start} to {self.scenario.end}){flag}",
            f"  trading days          {self.n_days}",
            f"  total return          {self.total_return:+.2%}",
            f"  worst single day      {self.worst_day:+.2%}"
            + (f"  on {self.worst_day_date.date()}" if self.worst_day_date is not None else ""),
            f"  max drawdown          {self.max_drawdown:.2%}",
            f"  annualised volatility {self.volatility_annualised:.2%}",
        ]
        if self.missing_assets:
            lines.append(
                f"  NOT COVERED           {', '.join(self.missing_assets)} "
                f"({self.coverage:.0%} of weight included)"
            )
        return "\n".join(lines)


def run_scenario(
    returns: pd.DataFrame,
    weights: pd.Series,
    scenario: Scenario,
    rescale_missing: bool = True,
) -> ScenarioResult:
    """
    Replay one scenario against a fixed-weight portfolio.

    Parameters
    ----------
    returns
        Daily decimal returns, one column per asset, spanning the
        scenario window.
    weights
        Portfolio weights indexed by asset.
    rescale_missing
        If an asset has no data in the window, renormalise the
        remaining weights so the result is a like-for-like return on
        the covered portion. The missing names are always reported.
        Set False to treat missing assets as flat, which understates
        the loss.
    """
    window = returns.loc[scenario.start_ts : scenario.end_ts]

    if window.empty:
        raise ValueError(
            f"no data in window {scenario.start} to {scenario.end}. "
            "Check that your price history covers this scenario."
        )

    available, missing = [], []
    for asset in weights.index:
        if asset in window.columns and window[asset].notna().sum() > 0:
            available.append(asset)
        else:
            missing.append(asset)

    if not available:
        raise ValueError(f"none of the portfolio assets have data in {scenario.key}")

    w = weights.loc[available].astype("float64")
    coverage = float(w.sum() / weights.sum()) if weights.sum() else 0.0
    if rescale_missing and w.sum() > 0:
        w = w / w.sum()

    sub = window[available].fillna(0.0)
    port = sub.to_numpy() @ w.to_numpy()
    port = pd.Series(port, index=sub.index)

    equity = (1 + port).cumprod()
    running_max = equity.cummax()
    drawdown = (equity / running_max - 1).min()

    worst_idx = port.idxmin() if len(port) else None

    return ScenarioResult(
        scenario=scenario,
        total_return=float((1 + port).prod() - 1),
        worst_day=float(port.min()),
        worst_day_date=worst_idx,
        volatility_annualised=float(port.std(ddof=1) * np.sqrt(252)) if len(port) > 1 else float("nan"),
        max_drawdown=float(drawdown),
        n_days=len(port),
        asset_returns=(1 + sub).prod() - 1,
        missing_assets=missing,
        coverage=coverage,
    )


def run_all_scenarios(
    returns: pd.DataFrame,
    weights: pd.Series,
    keys: list[str] | None = None,
    rescale_missing: bool = True,
) -> dict[str, ScenarioResult]:
    """
    Run every scenario the data actually covers.

    Scenarios outside the sample are skipped rather than raising, but
    the skip is visible in the returned keys - compare against
    SCENARIOS to see what was not tested.
    """
    chosen = keys or list(SCENARIOS.keys())
    out: dict[str, ScenarioResult] = {}

    for key in chosen:
        if key not in SCENARIOS:
            raise KeyError(f"unknown scenario '{key}'. Available: {list(SCENARIOS)}")
        try:
            out[key] = run_scenario(
                returns, weights, SCENARIOS[key], rescale_missing=rescale_missing
            )
        except ValueError:
            continue

    return out


def scenario_table(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    """Tabulate scenario results, worst total return first."""
    rows = []
    for key, res in results.items():
        rows.append(
            {
                "scenario": res.scenario.name,
                "start": res.scenario.start,
                "end": res.scenario.end,
                "days": res.n_days,
                "total_return": res.total_return,
                "worst_day": res.worst_day,
                "max_drawdown": res.max_drawdown,
                "ann_vol": res.volatility_annualised,
                "coverage": res.coverage,
                "complete": res.complete,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, index=list(results.keys())).sort_values("total_return")


def stress_vs_var(
    results: dict[str, ScenarioResult], one_day_var: float, confidence: float = 0.99
) -> pd.DataFrame:
    """
    Express each scenario's worst day as a multiple of current VaR.

    This is the number that makes a stress test land with a
    committee. "Our 99% one-day VaR is 1.8%, and the worst day of
    the Lehman window was 4.2% - 2.3 times VaR" says something a
    p-value does not.
    """
    if one_day_var <= 0:
        raise ValueError("one_day_var must be positive")

    rows = []
    for key, res in results.items():
        loss = -res.worst_day
        rows.append(
            {
                "scenario": res.scenario.name,
                "worst_day_loss": loss,
                "var": one_day_var,
                "multiple_of_var": loss / one_day_var,
                "breaches_var": loss > one_day_var,
            }
        )
    df = pd.DataFrame(rows, index=list(results.keys()))
    df.attrs["confidence"] = confidence
    return df.sort_values("multiple_of_var", ascending=False)


def hypothetical_shock(
    weights: pd.Series, shocks: dict[str, float]
) -> float:
    """
    Instantaneous shock applied to positions.

    `shocks` maps asset to a decimal return, e.g.
    {"EURUSD": -0.30} for a repeat of the SNB move. Assets absent
    from the mapping are assumed unchanged, which is an assumption,
    not a finding - correlated assets rarely stay still.
    """
    total = 0.0
    for asset, weight in weights.items():
        total += weight * shocks.get(asset, 0.0)
    return float(total)
