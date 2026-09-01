import os
import sys

import numpy as np
import pandas as pd
import yfinance as yf

# The instrument universe is defined once, in config.py, and shared
# with the fxrisk engine so both halves of this repository describe
# the same book. Previously this file held its own hard-coded list
# (gold futures, EUR/USD, GBP/JPY) while the engine held four dollar
# pairs, and nothing said which was intended.
#
# config.py imports nothing beyond the standard library, so this
# stays runnable with just numpy, pandas and yfinance.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import config

    TICKERS = dict(config.YAHOO_SYMBOLS)
except ImportError:  # pragma: no cover - standalone fallback
    TICKERS = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "USD/CHF": "USDCHF=X",
        "GBP/JPY": "GBPJPY=X",
    }


def fetch_price_data(ticker, period="2y"):
    """Fetches historical daily close data for a given ticker via Yahoo Finance."""
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, period=period)

    if data.empty:
        raise ValueError(
            f"No data returned for {ticker}. Check internet connection or ticker syntax."
        )

    # Flatten MultiIndex columns if present (newer yfinance versions)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Calculate Daily Log Returns cleanly
    data["Returns"] = np.log(data["Close"] / data["Close"].shift(1))
    return data.dropna()


def calculate_rolling_metrics(df, window=126, risk_free_rate=0.0):
    """Calculates a rolling 6-month (126 trading days) annualized Sharpe Ratio

    and a rolling 95% Historical Value-at-Risk (VaR).

    Sharpe Ratio: uses excess return over the annualized risk_free_rate (default 0.0).
    VaR: reported as the 5th percentile of the return distribution (negative = expected loss).
    """
    daily_rf = risk_free_rate / 252

    # 1. Rolling Sharpe Ratio (Annualized: (Mean excess return) / Std * sqrt(252))
    excess_returns = df["Returns"] - daily_rf
    df["Rolling_Sharpe"] = (
        excess_returns.rolling(window).mean()
        / df["Returns"].rolling(window).std()
    ) * np.sqrt(252)

    # 2. Rolling 95% Historical Value-at-Risk (VaR)
    # Negative value = expected worst-case daily loss at 95% confidence
    df["Rolling_VaR_95"] = df["Returns"].rolling(window).quantile(0.05)

    return df


def run_pipeline(asset_name, ticker, risk_free_rate=0.04):
    """Runs the full fetch -> process pipeline for a single asset and prints a snapshot."""
    try:
        price_data = fetch_price_data(ticker=ticker, period="2y")
        processed_data = calculate_rolling_metrics(price_data, risk_free_rate=risk_free_rate)

        print(f"\n--- {asset_name} ({ticker}) Quantitative Pipeline Snapshot ---")
        print(
            processed_data[["Close", "Rolling_Sharpe", "Rolling_VaR_95"]].tail()
        )
        return processed_data

    except Exception as e:
        print(f"\nExecution Pipeline Failed for {asset_name} ({ticker}): {str(e)}")
        return None


if __name__ == "__main__":
    # Run the risk pipeline across Gold Futures and two major FX pairs
    results = {}
    for asset_name, ticker in TICKERS.items():
        results[asset_name] = run_pipeline(asset_name, ticker)
