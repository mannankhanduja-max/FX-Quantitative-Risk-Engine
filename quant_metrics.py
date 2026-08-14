import numpy as np
import pandas as pd
import yfinance as yf


def fetch_fx_data(ticker="GC=F", period="2y"):
    """Fetches historical daily close data for Gold Futures (GC=F) via Yahoo Finance."""
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


if __name__ == "__main__":
    # Using Gold Futures (GC=F) as the pilot asset for the risk engine pipeline
    try:
        fx_data = fetch_fx_data(ticker="GC=F", period="2y")
        processed_data = calculate_rolling_metrics(fx_data, risk_free_rate=0.04)

        # Print snapshot with perfectly matching string indices
        print("\n--- Quantitative Pipeline Snapshot ---")
        print(
            processed_data[["Close", "Rolling_Sharpe", "Rolling_VaR_95"]].tail()
        )

    except Exception as e:
        print(f"\nExecution Pipeline Failed: {str(e)}")
