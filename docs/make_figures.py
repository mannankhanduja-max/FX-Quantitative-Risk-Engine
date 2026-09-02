"""
Regenerate the figures in docs/figures/ from live model output.

    python docs/make_figures.py --demo
    python docs/make_figures.py --data-dir data/histdata

Every figure here is drawn from a model run, not hand-authored, so
the pictures cannot drift away from what the code does. Re-run this
after changing a model and the figures follow.

Conceptual diagrams (how the pieces fit together, what the leak was)
live in docs/diagrams/ as hand-drawn SVG. This file produces the
empirical half: what the models actually output.

Figures carry axes, legends and units and nothing else - no
explanatory annotation baked into the image. The claim each one
supports belongs in the README next to it, where it can be edited
without regenerating a PNG.

Backtest-only. Demo output describes simulated data and every file
it writes is prefixed DEMO_.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from fxrisk.models.dcc import fit_dcc  # noqa: E402
from fxrisk.models.ewma import ewma_variance  # noqa: E402
from fxrisk.models.garch import rolling_garch_forecasts  # noqa: E402
from fxrisk.risk.backtesting import backtest_var  # noqa: E402
from fxrisk.risk.performance import rolling_sharpe  # noqa: E402
from fxrisk.risk.var import dcc_var_series, garch_var_series  # noqa: E402

# --- validated categorical slots (see the dataviz reference palette) ---
S1 = "#2a78d6"   # blue
S2 = "#eb6834"   # orange
S3 = "#1baf7a"   # aqua
S4 = "#4a3aa7"   # violet
CRITICAL = "#e34948"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e3e2df"
SURFACE = "#fcfcfb"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def _save(fig, name: str, prefix: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{prefix}{name}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")


def demo_panel(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Simulated GARCH panel with a common factor. Not real data."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-01", periods=n)
    factor = np.zeros(n)
    s2 = np.empty(n)
    s2[0] = 4e-5
    for t in range(1, n):
        s2[t] = 1e-6 + 0.09 * factor[t - 1] ** 2 + 0.89 * s2[t - 1]
        factor[t] = np.sqrt(s2[t]) * rng.standard_t(6) / np.sqrt(6 / 4)
    return pd.DataFrame(
        {
            name: 0.7 * factor + rng.standard_normal(n) * 0.004
            for name in ("SIM_A", "SIM_B", "SIM_C")
        },
        index=idx,
    )


# ---------------------------------------------------------------
# Figures
# ---------------------------------------------------------------

def fig_volatility(portfolio: pd.Series, prefix: str) -> None:
    """Three volatility estimates on one axis, annualised."""
    ann = np.sqrt(252)
    ewma = np.sqrt(ewma_variance(portfolio, lam=config.EWMA_LAMBDA)) * ann
    garch = np.sqrt(
        rolling_garch_forecasts(
            portfolio,
            window=config.GARCH_WINDOW,
            refit_every=config.GARCH_REFIT_EVERY,
            dist=config.GARCH_DIST,
            min_obs=config.GARCH_MIN_OBS,
        )["variance"]
    ) * ann
    rolling_sd = portfolio.rolling(252).std() * ann

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(rolling_sd.index, rolling_sd, color=S3, label="rolling 252-day sd")
    ax.plot(ewma.index, ewma, color=S1, label=f"EWMA λ={config.EWMA_LAMBDA}")
    ax.plot(garch.index, garch, color=S2, label="GARCH(1,1)-t")
    ax.set_title("Annualised volatility forecast")
    ax.set_ylabel("annualised σ")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(loc="upper left", ncols=3)
    _save(fig, "01-volatility", prefix)


def fig_var_breaches(portfolio: pd.Series, var: pd.Series, prefix: str) -> None:
    """Realised returns against the VaR line, breaches marked."""
    df = pd.DataFrame({"r": portfolio, "var": var}).dropna()
    breach = df["r"] < -df["var"]

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(df.index, df["r"], color="#c9c8c4", linewidth=0.7, label="daily return")
    ax.plot(df.index, -df["var"], color=S1, label="99% VaR")
    ax.scatter(
        df.index[breach],
        df["r"][breach],
        s=16,
        color=CRITICAL,
        zorder=3,
        label=f"breach ({int(breach.sum())})",
    )
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.set_title("Realised returns vs 99% VaR (GARCH-t)")
    ax.set_ylabel("daily return")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.legend(loc="lower left", ncols=3)
    _save(fig, "02-var-breaches", prefix)


def fig_correlation(dcc, assets: list[str], prefix: str) -> None:
    """Every pairwise DCC correlation path, with the sample value."""
    from itertools import combinations

    fig, ax = plt.subplots(figsize=(9, 3.4))
    for (i, j), colour in zip(combinations(assets, 2), (S1, S2, S3, S4)):
        path = dcc.correlation_series(i, j)
        ax.plot(path.index, path, color=colour, label=f"{i}/{j}")
        ax.axhline(
            float(dcc.unconditional_correlation.loc[i, j]),
            color=colour,
            linestyle=":",
            linewidth=1.1,
        )
    ax.set_title("DCC conditional correlation (dotted: unconditional)")
    ax.set_ylabel("ρ")
    ax.set_ylim(-0.1, 1.0)
    ax.legend(loc="lower left", ncols=3)
    _save(fig, "03-correlation", prefix)


def fig_rolling_sharpe(portfolio: pd.Series, prefix: str) -> None:
    """Rolling Sharpe against the full-sample value."""
    rs = rolling_sharpe(
        portfolio,
        window=config.ROLLING_SHARPE_WINDOW,
        risk_free_rate=config.RISK_FREE_RATE,
    ).dropna()
    from fxrisk.risk.performance import sharpe_ratio

    full = sharpe_ratio(portfolio, config.RISK_FREE_RATE)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(rs.index, 0, rs, where=rs >= 0, color=S3, alpha=0.30, linewidth=0)
    ax.fill_between(rs.index, 0, rs, where=rs < 0, color=CRITICAL, alpha=0.25, linewidth=0)
    ax.plot(rs.index, rs, color=INK, linewidth=1.2, label=f"rolling {config.ROLLING_SHARPE_WINDOW}-day")
    ax.axhline(full, color=S1, linestyle="--", linewidth=1.4, label=f"full sample {full:.2f}")
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_title(f"Rolling Sharpe (risk-free {config.RISK_FREE_RATE:.1%} annual)")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="upper left", ncols=2)
    _save(fig, "04-rolling-sharpe", prefix)


def fig_backtest_bars(results: list, prefix: str) -> None:
    """Breach rate per method against the target rate."""
    names = [r.method.split("@")[0] for r in results]
    rates = [r.breach_rate for r in results]
    target = 1 - results[0].confidence
    colours = [S3 if r.passed() else CRITICAL for r in results]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    bars = ax.barh(names, rates, color=colours, height=0.6)
    ax.axvline(target, color=INK, linestyle="--", linewidth=1.4)
    ax.annotate(
        f"target {target:.1%}",
        xy=(target, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(4, 4),
        textcoords="offset points",
        color=INK,
        fontsize=8,
    )
    for bar, r in zip(bars, results):
        ax.text(
            bar.get_width() + target * 0.10,
            bar.get_y() + bar.get_height() / 2,
            f"{r.n_breaches} breaches · {'pass' if r.passed() else 'fail'}",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    ax.set_title(f"Breach rate at {results[0].confidence:.1%}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    ax.set_xlim(0, max(rates) * 1.55)
    ax.grid(axis="y", visible=False)
    _save(fig, "05-backtest-rates", prefix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--source", choices=("histdata", "yahoo"), default="histdata")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    args = parser.parse_args()

    _style()

    if args.demo:
        returns = demo_panel()
        prefix = "DEMO_"
        print("Simulated data. Figures describe no real market.")
    elif args.source == "yahoo":
        from fxrisk.data import yahoo

        symbols = {i.name: i.yahoo for i in config.UNIVERSE}
        returns = yahoo.daily_returns(symbols, kind="log")
        prefix = ""
        print(f"Yahoo daily bars: {len(returns)} rows, "
              f"{returns.index.min().date()} to {returns.index.max().date()}")
    else:
        from run_risk_report import load_real_returns

        returns = load_real_returns(
            args.data_dir, config.PAIRS, config.DATA_GRANULARITY
        )
        prefix = ""

    assets = list(returns.columns)
    weights = pd.Series(1.0 / len(assets), index=assets)
    portfolio = pd.Series(returns.to_numpy() @ weights.to_numpy(), index=returns.index)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fig_volatility(portfolio, prefix)

        gv = garch_var_series(
            portfolio,
            confidence=0.99,
            dist=config.GARCH_DIST,
            window=config.GARCH_WINDOW,
            refit_every=config.GARCH_REFIT_EVERY,
            min_obs=config.GARCH_MIN_OBS,
        )
        fig_var_breaches(portfolio, gv["var"], prefix)

        dcc = fit_dcc(returns, dist=config.GARCH_DIST, mean="zero")
        fig_correlation(dcc, assets, prefix)

        fig_rolling_sharpe(portfolio, prefix)

        dv = dcc_var_series(
            returns,
            weights,
            confidence=0.99,
            dist=config.GARCH_DIST,
            window=config.GARCH_WINDOW,
            refit_every=config.DCC_REFIT_EVERY,
            min_obs=config.DCC_MIN_OBS,
        )
        roll = portfolio.rolling(config.HISTORICAL_WINDOW).quantile(0.01).shift(1)
        mu = portfolio.rolling(config.HISTORICAL_WINDOW).mean().shift(1)
        sd = portfolio.rolling(config.HISTORICAL_WINDOW).std().shift(1)

        results = []
        for name, series in (
            ("garch_t", gv["var"]),
            ("dcc_portfolio", dv["var"]),
            ("historical", -roll),
            ("parametric_normal", -(mu + sd * stats.norm.ppf(0.01))),
        ):
            try:
                results.append(backtest_var(portfolio, series, 0.99, name))
            except ValueError:
                pass
        if results:
            fig_backtest_bars(results, prefix)

    print("\nBacktest-only. Not investment advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
