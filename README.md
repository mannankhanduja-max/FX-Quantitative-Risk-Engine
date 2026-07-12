# FX Quantitative Risk Engine

A lightweight, automated financial engineering pipeline designed to ingest daily time-series FX data and compute core risk-adjusted performance metrics.

## Features
- **Data Ingestion**: Programmatic multi-year daily tick/close retrieval via Yahoo Finance API wrapper.
- **Performance Evaluation**: Implements a rolling 6-month (126-day window) annualized Sharpe Ratio to track risk-adjusted consistency.
- **Risk Analytics**: Implements a rolling 95% non-parametric Historical Value-at-Risk (VaR) threshold to monitor tail-risk parameters.

## Tech Stack
- **Languages**: Python 3.10+
- **Libraries**: `NumPy`, `pandas`, `yfinance`

## How To Execute
```bash
pip install numpy pandas yfinance
python quant_metrics.py
