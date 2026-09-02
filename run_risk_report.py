"""
FX risk report.

    python run_risk_report.py --demo
    python run_risk_report.py --data-dir data/histdata

--demo runs the whole pipeline on simulated GARCH data so the
engine is verifiable without downloading anything. Demo output is
NOT a result about any real market and is labelled as such in every
table it writes.

ALL OUTPUT IS BACKTEST-ONLY. See the disclaimer printed at the end
of every run and in README.md.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

import config
from fxrisk.data import yahoo
from fxrisk import indicators
from fxrisk.risk.montecarlo import compare_methods, term_structure
from fxrisk.data.histdata import daily_returns, load_m1, load_ticks, ticks_to_bars, to_daily
from fxrisk.models.dcc import fit_dcc
from fxrisk.models.ewma import (
    correlation_from_covariance,
    ewma_covariance_last,
)
from fxrisk.models.garch import fit_garch
from fxrisk.risk.backtesting import backtest_var, compare_models
from fxrisk.risk.stress import SCENARIOS, run_all_scenarios, scenario_table, stress_vs_var
from fxrisk.risk.correlation import correlation_stress, pairwise_table
from fxrisk.risk.performance import (
    per_asset_performance,
    performance_summary,
    rolling_sharpe,
)
from fxrisk.risk.var import (
    component_var,
    dcc_var_series,
    ewma_var_series,
    garch_var_series,
    portfolio_var_from_covariance,
)

DISCLAIMER = """
================================================================
BACKTEST-ONLY RESULTS

Everything above is computed on historical data. It is a
description of what these models would have reported in the past,
not a prediction and not investment advice.

Specifically:

  - No result here is out-of-sample in the sense that matters. The
    scenario windows, the pairs and the model specifications were
    all chosen with knowledge of what happened.
  - Backtested risk figures ignore execution: no slippage, no
    funding, no bid-offer beyond what the tick data shows, no
    market impact, and no possibility that a position could not be
    exited at the marked price. In every episode stress-tested
    below, that last assumption failed for somebody.
  - Passing a VaR backtest means a model was adequately calibrated
    over one sample. It does not transfer.
  - Past performance does not indicate future results.

Not investment advice. Not a solicitation. No warranty.
================================================================
"""


def _header(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def load_demo_returns(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Simulated GARCH panel with a common factor. Not real data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-01", periods=n)
    names = ["SIM_A", "SIM_B", "SIM_C"]

    factor = np.zeros(n)
    s2f = np.empty(n)
    s2f[0] = 4e-5
    for t in range(1, n):
        s2f[t] = 1e-6 + 0.09 * factor[t - 1] ** 2 + 0.89 * s2f[t - 1]
        factor[t] = np.sqrt(s2f[t]) * rng.standard_t(6) / np.sqrt(6 / 4)

    out = {}
    for i, name in enumerate(names):
        idio = rng.standard_normal(n) * 0.004
        out[name] = 0.7 * factor + idio
    return pd.DataFrame(out, index=idx)


def load_real_returns(data_dir: str, pairs: list[str], granularity: str) -> pd.DataFrame:
    """Load HistData files into a daily return panel."""
    series = {}
    for pair in pairs:
        folder = os.path.join(data_dir, pair)
        if not os.path.isdir(folder):
            print(f"  {pair:<8} SKIPPED - no folder at {folder}")
            continue

        if granularity == "tick":
            ticks, report = load_ticks(folder, return_report=True)
            bars = ticks_to_bars(ticks, "1min")
        else:
            bars, report = load_m1(folder, return_report=True)

        print(f"  {pair}")
        print(report.summary())

        daily = to_daily(bars, config.SESSION_CLOSE, config.SESSION_TZ)
        series[pair] = daily_returns(daily, kind="log")

    if not series:
        raise SystemExit(
            f"No data loaded from '{data_dir}'.\n"
            "Download the ASCII archives from histdata.com, unzip them into\n"
            "one folder per pair, or run with --demo."
        )

    return pd.DataFrame(series).dropna()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run on simulated data")
    parser.add_argument(
        "--source",
        choices=("histdata", "yahoo"),
        default="histdata",
        help="yahoo reads the local cache written by fetch_data.py",
    )
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--skip-dcc", action="store_true", help="DCC is the slow step")
    args = parser.parse_args()

    _header("LOADING DATA")
    if args.demo:
        returns = load_demo_returns()
        print("  SIMULATED DATA - results below describe no real market.")
        print(f"  {returns.shape[1]} series, {len(returns)} observations")
    elif args.source == "yahoo":
        symbols = {i.name: i.yahoo for i in config.UNIVERSE}
        rep = yahoo.coverage_report(symbols)
        print("  Yahoo Finance daily bars, from the local cache.\n")
        print(rep.to_string())
        returns = yahoo.daily_returns(symbols, kind="log")
        yahoo_symbols = symbols
        print(f"\n  aligned panel: {len(returns)} rows")
        if rep.attrs.get("binding"):
            print(f"  sample is bounded by: {rep.attrs['binding']}")
        print("  NOTE: Yahoo daily FX bars use Yahoo's own session boundary,")
        print("  not the 17:00 New York convention, and carry no volume - so")
        print("  no tick-count VWAP is available on this source.")
    else:
        returns = load_real_returns(args.data_dir, config.PAIRS, config.DATA_GRANULARITY)

    print(f"  window                {returns.index.min().date()} to {returns.index.max().date()}")

    assets = list(returns.columns)
    if config.WEIGHTS:
        weights = pd.Series(config.WEIGHTS).reindex(assets).fillna(0.0)
    else:
        weights = pd.Series(1.0 / len(assets), index=assets)

    portfolio = pd.Series(returns.to_numpy() @ weights.to_numpy(), index=returns.index)

    # ---------------------------------------------------------
    _header("COVARIANCE: SAMPLE vs EWMA")

    sample_cov = returns.cov()
    ewma_cov = ewma_covariance_last(returns, lam=config.EWMA_LAMBDA)

    sample_corr = correlation_from_covariance(sample_cov)
    ewma_corr = correlation_from_covariance(ewma_cov)

    print("\nSample correlation (equal weight on every observation):")
    print(sample_corr.round(3).to_string())
    print(f"\nEWMA correlation (lambda={config.EWMA_LAMBDA}, ~{1/(1-config.EWMA_LAMBDA):.0f} day centre of mass):")
    print(ewma_corr.round(3).to_string())

    diff = np.array((ewma_corr - sample_corr).abs().to_numpy(), copy=True)
    np.fill_diagonal(diff, 0.0)
    print(f"\nLargest correlation difference: {diff.max():.3f}")
    print("A large gap means the sample matrix is describing a regime that has ended.")

    # ---------------------------------------------------------
    _header("VOLATILITY MODEL")

    fit = fit_garch(portfolio, dist=config.GARCH_DIST, mean="zero")
    print(fit.summary())

    # ---------------------------------------------------------
    dcc = None
    if not args.skip_dcc and len(assets) >= 2:
        _header("DCC-GARCH")
        try:
            dcc = fit_dcc(returns, dist=config.GARCH_DIST, mean="zero")
            print(dcc.summary())
            i, j = assets[0], assets[1]
            rho = dcc.correlation_series(i, j)
            print(f"\nConditional correlation {i}/{j}:")
            print(f"  mean {rho.mean():.3f}   min {rho.min():.3f}   max {rho.max():.3f}")
            print("  The spread between min and max is the risk a static")
            print("  correlation matrix cannot see.")
        except Exception as exc:  # noqa: BLE001
            print(f"  DCC estimation failed: {exc}")
            dcc = None

    # ---------------------------------------------------------
    _header("PAIRWISE CORRELATION")

    pairs = pairwise_table(returns, dcc=dcc, lam=config.EWMA_LAMBDA)
    print(pairs.round(3).to_string())
    print()
    print("  sample    = one number for the whole history")
    print("  ewma      = current regime")
    print("  dcc_range = how much the single sample number averages away")

    if dcc is not None:
        try:
            stress_corr = correlation_stress(returns, dcc, quantile=0.05)
            print(f"\nCorrelation on the worst 5% of days "
                  f"({stress_corr.attrs['n_stressed_days']} days) vs the rest:")
            print(stress_corr.round(3).to_string())
            print("\n  A positive 'increase' means diversification weakens")
            print("  exactly when it is needed. That is the case for using a")
            print("  conditional correlation model rather than a static one.")
        except ValueError as exc:
            print(f"  correlation stress split unavailable: {exc}")

    # ---------------------------------------------------------
    if args.source == "yahoo" and not args.demo:
        _header("VWAP AND 9-PERIOD EMA")
        print(f"Rolling {config.VWAP_WINDOW_DAILY}-bar VWAP, weighted by real")
        print(f"share volume, with a {config.VWAP_EMA_SPAN}-period EMA on top.\n")
        for name, sym in yahoo_symbols.items():
            try:
                bars = yahoo.load_symbol(sym)
                frame = indicators.vwap_ema(
                    bars,
                    window=config.VWAP_WINDOW_DAILY,
                    span=config.VWAP_EMA_SPAN,
                )
                print(f"{name} ({sym})")
                print(indicators.summary(frame))
                print()
            except ValueError as exc:
                print(f"{name} ({sym}): {exc}\n")

    _header("MONTE CARLO VaR")
    print(f"{config.MC_SIMULATIONS:,} paths, seed fixed.\n")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("Methods at 99%, 1 day:")
            print(compare_methods(
                portfolio, confidence=0.99,
                n_simulations=config.MC_SIMULATIONS).round(5).to_string())

            ts = term_structure(
                portfolio, config.MC_HORIZONS, confidence=0.99,
                n_simulations=config.MC_SIMULATIONS, method="fhs")
            print("\nTerm structure (FHS) vs sqrt-time scaling:")
            print(ts.round(5).to_string())
            print("\n  ratio < 1 means sqrt-time OVERSTATES risk - current")
            print("  volatility is above its long-run level and the model")
            print("  expects it to mean-revert over the horizon.")
    except Exception as exc:  # noqa: BLE001
        print(f"  Monte Carlo unavailable: {exc}")

    _header("RISK-ADJUSTED PERFORMANCE")

    perf = performance_summary(portfolio, risk_free_rate=config.RISK_FREE_RATE)
    print("Equal-weight portfolio:")
    print(perf.summary())

    print("\nPer instrument:")
    print(per_asset_performance(
        returns, risk_free_rate=config.RISK_FREE_RATE).round(4).to_string())

    rs = rolling_sharpe(portfolio, window=config.ROLLING_SHARPE_WINDOW,
                        risk_free_rate=config.RISK_FREE_RATE).dropna()
    if len(rs):
        print(f"\nRolling {config.ROLLING_SHARPE_WINDOW}-day Sharpe: "
              f"min {rs.min():.2f}   mean {rs.mean():.2f}   max {rs.max():.2f}")
        print("  The spread is the point: a single full-sample Sharpe hides it.")

    # ---------------------------------------------------------
    _header("VaR BACKTESTING")

    all_results = []
    for confidence in config.CONFIDENCE_LEVELS:
        print(f"\n--- {confidence:.1%} ---")

        var_frames = {}

        roll = portfolio.rolling(config.HISTORICAL_WINDOW).quantile(1 - confidence).shift(1)
        var_frames["historical"] = -roll

        mu = portfolio.rolling(config.HISTORICAL_WINDOW).mean().shift(1)
        sd = portfolio.rolling(config.HISTORICAL_WINDOW).std().shift(1)
        from scipy import stats as _st

        var_frames["parametric_normal"] = -(mu + sd * _st.norm.ppf(1 - confidence))

        var_frames["ewma_normal"] = ewma_var_series(
            portfolio, confidence=confidence, lam=config.EWMA_LAMBDA
        )["var"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_frames[f"garch_{config.GARCH_DIST}"] = garch_var_series(
                portfolio,
                confidence=confidence,
                dist=config.GARCH_DIST,
                window=config.GARCH_WINDOW,
                refit_every=config.GARCH_REFIT_EVERY,
                min_obs=config.GARCH_MIN_OBS,
            )["var"]

        if not args.skip_dcc and len(assets) >= 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    var_frames["dcc_portfolio"] = dcc_var_series(
                        returns,
                        weights,
                        confidence=confidence,
                        dist=config.GARCH_DIST,
                        nu=fit.nu,
                        window=config.GARCH_WINDOW,
                        refit_every=config.DCC_REFIT_EVERY,
                        min_obs=config.DCC_MIN_OBS,
                    )["var"]
            except Exception as exc:  # noqa: BLE001
                print(f"  dcc_portfolio unavailable: {exc}")

        level_results = []
        for name, series in var_frames.items():
            try:
                res = backtest_var(portfolio, series, confidence, f"{name}@{confidence:.3f}")
                level_results.append(res)
                all_results.append(res)
            except ValueError as exc:
                print(f"  {name}: {exc}")

        if level_results:
            print(compare_models(level_results).round(4).to_string())

    # ---------------------------------------------------------
    _header("CURRENT RISK")

    current = portfolio_var_from_covariance(
        weights, ewma_cov, confidence=0.99, dist="t", nu=max(fit.nu or 8.0, 2.5)
    )
    print(f"  99% 1-day VaR (EWMA cov, Student-t)   {current.var:.3%}")
    print(f"  99% 1-day Expected Shortfall          {current.expected_shortfall:.3%}")
    print(f"  10-day scaled (sqrt-time, see caveat) {current.scale_to_horizon(10).var:.3%}")

    print("\nRisk decomposition:")
    print(component_var(weights, ewma_cov, 0.99).round(4).to_string())

    # ---------------------------------------------------------
    _header("STRESS TESTING")

    results = run_all_scenarios(
        returns,
        weights,
        keys=config.STRESS_SCENARIOS,
        rescale_missing=config.STRESS_RESCALE_MISSING,
    )

    skipped = [k for k in config.STRESS_SCENARIOS if k not in results]
    if skipped:
        print("NOT TESTED - outside the data window:")
        for key in skipped:
            print(f"  {key:<14} {SCENARIOS[key].start} to {SCENARIOS[key].end}")
        print()

    if results:
        for res in results.values():
            print(res.summary())
            print()

        print(scenario_table(results).round(4).to_string())
        print("\nWorst day as a multiple of current 99% VaR:")
        print(stress_vs_var(results, current.var).round(3).to_string())
    else:
        print("No scenario windows are covered by this data.")
        print("Stress testing is the part of this engine that most needs")
        print("real history - a sample that starts after 2020 cannot be")
        print("stressed against 2008.")

    # ---------------------------------------------------------
    if config.SAVE_RESULTS and all_results:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        tag = "DEMO_SIMULATED_" if args.demo else (
            "YAHOO_" if args.source == "yahoo" else "")

        compare_models(all_results).to_csv(
            os.path.join(config.RESULTS_DIR, f"{tag}var_backtests.csv")
        )
        if results:
            scenario_table(results).to_csv(
                os.path.join(config.RESULTS_DIR, f"{tag}stress_scenarios.csv")
            )
        ewma_corr.to_csv(os.path.join(config.RESULTS_DIR, f"{tag}ewma_correlation.csv"))
        print(f"\nResults written to {config.RESULTS_DIR}/")

    print(DISCLAIMER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
