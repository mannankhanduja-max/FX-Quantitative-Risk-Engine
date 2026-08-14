# FX Quantitative Risk Engine

A lightweight, automated financial engineering pipeline designed to ingest daily time-series price data and compute core risk-adjusted performance metrics across multiple asset classes.

## Purpose

This engine is a compact risk-management pipeline that turns raw daily price data into two continuously-updated signals:

- **Risk-adjusted return** (rolling Sharpe ratio) — how much return is being generated per unit of volatility, recalculated on a rolling 6-month basis so regime shifts are captured rather than smoothed away by a single static figure.
- **Tail risk** (rolling historical VaR) — the empirical worst-case daily loss threshold at 95% confidence, estimated directly from the historical return distribution rather than assuming a normal distribution.

Together these mirror the kind of rolling risk dashboard used on trading desks and by portfolio/risk management teams to monitor an asset's changing risk profile over time. The pipeline runs the same Sharpe/VaR logic across Gold Futures (`GC=F`) and two major FX pairs, EUR/USD (`EURUSD=X`) and GBP/JPY (`GBPJPY=X`), and generalizes to any Yahoo Finance-listed instrument by adding it to the ticker dictionary.

## Features
- **Data Ingestion**: Programmatic multi-year daily tick/close retrieval via Yahoo Finance API wrapper.
- **Multi-Asset Coverage**: Runs the same risk pipeline across Gold Futures, EUR/USD, and GBP/JPY out of the box.
- **Performance Evaluation**: Implements a rolling 6-month (126-day window) annualized Sharpe Ratio (configurable risk-free rate) to track risk-adjusted consistency.
- **Risk Analytics**: Implements a rolling 95% non-parametric Historical Value-at-Risk (VaR) threshold to monitor tail-risk parameters.

## Tech Stack
- **Languages**: Python 3.10+
- **Libraries**: `NumPy`, `pandas`, `yfinance`

## How To Execute
```bash
pip install -r requirements.txt
python quant_metrics.py
```

## Validation

Core logic (rolling Sharpe and VaR calculations) has been verified against synthetic price series for gold, EUR/USD, and GBP/JPY to confirm no NaN leakage after the warm-up window, finite Sharpe values, and correctly-signed VaR output across different price scales. Live execution requires an internet connection to reach the Yahoo Finance API.
