# QuantPortfolio

A Python research project for market data analysis, systematic strategy
testing, portfolio construction and portfolio optimisation.

It combines a rule-based rolling-VWAP mean-reversion strategy with four
portfolio construction methods: equal weighting, Monte Carlo simulation,
maximum Sharpe optimisation and minimum variance optimisation.

The asset universe and every strategy parameter live in `config.py`, so
the model code in `models/` does not need to change when the universe or
the parameters do.

---

## 1. What it actually does

Running `python main.py` executes one linear pipeline:

1. Download daily OHLCV data from Yahoo Finance for every ticker in
   `config.TICKERS`
2. Forward-fill and drop incomplete rows, then save raw and processed
   CSVs under `data/`
3. Compute simple returns, log returns, annualised volatility,
   correlation and annualised covariance
4. Build a rolling VWAP on `config.STRATEGY_TICKER` from the typical
   price `(High + Low + Close) / 3` weighted by volume
5. Measure the percentage deviation of close from VWAP
6. Derive rolling percentile thresholds from the previous
   `THRESHOLD_WINDOW` observations
7. Generate signals: long below the lower percentile, short above the
   upper percentile, flat in between
8. Evaluate signals over 1, 5 and 20-day forward horizons
9. Backtest the strategy daily, net of transaction costs, against a
   buy-and-hold benchmark
10. Build equal-weight, Monte Carlo, maximum-Sharpe and minimum-variance
    portfolios and compare them
11. Write result tables to `results/tables/` and draw the charts

---

## 2. Getting started

Requires Python 3.9 or later.

```bash
git clone <repository-url>
cd Quant_Portoflio
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

`requirements.txt` lists only what the project imports: `numpy`,
`pandas`, `scipy`, `matplotlib` and `yfinance`. Earlier versions of this
file were a full `pip freeze` of the development environment and pulled
in around eighty packages the project never imports.

---

## 3. Project structure

```
Quant_Portoflio/
├── config.py            # All parameters. Start here.
├── config_eu.py         # Alternative CET-clock universe (see §5)
├── main.py              # The pipeline, executed top to bottom
├── requirements.txt
├── README.md
├── models/
│   ├── statistics.py    # Returns, volatility, correlation, covariance
│   ├── strategy.py      # Rolling VWAP, thresholds, signals
│   ├── backtest.py      # Equity curve and performance metrics
│   └── portfolio.py     # Weights, Monte Carlo, SLSQP optimisation
├── utils/
│   └── plotting.py      # Matplotlib charts
├── data/
│   ├── raw/             # Downloaded market data
│   └── processed/       # Cleaned prices and strategy output
└── results/
    ├── tables/          # CSV output
    └── figures/
```

---

## 4. The asset universe

`config.TICKERS` holds ten instruments:

| Ticker | Instrument | Venue |
|---|---|---|
| `AAPL` | Apple | NASDAQ |
| `MSFT` | Microsoft | NASDAQ |
| `AL2SI.PA` | 2CRSi SA | Euronext Growth Paris |
| `PKN.WA` | PKN Orlen | Warsaw |
| `FXE` | Invesco CurrencyShares Euro Trust | NYSE Arca |
| `FXY` | Invesco CurrencyShares Japanese Yen | NYSE Arca |
| `FXB` | Invesco CurrencyShares British Pound | NYSE Arca |
| `GLD` | SPDR Gold Shares | NYSE Arca |
| `USO` | United States Oil Fund | NYSE Arca |
| `UNG` | United States Natural Gas Fund | NYSE Arca |

FX and commodity exposure is taken through exchange-traded funds rather
than through FX spot pairs (`EURUSD=X`) and continuous futures (`CL=F`)
for two reasons.

**Trading hours.** FX quotes run around the clock and futures nearly so,
so their daily "close" is sampled at a different moment from an equity
close. Putting them in one covariance matrix compares prices that were
never observed at the same instant, which biases correlations downward.
Every ETF above closes at 16:00 ET, in line with AAPL and MSFT.

**Volume.** `models/strategy.py` builds VWAP from price and volume, and
discards rows where volume is not positive. Yahoo reports zero volume
for FX spot pairs, so a VWAP on `EURUSD=X` was undefined for every
observation. ETFs report real share volume.

`PKN.WA` and `AL2SI.PA` still close around 17:00-17:30 CET, roughly
11:00-11:30 ET, so they remain non-synchronous with the US names.
`config_eu.py` addresses this.

---

## 5. `config_eu.py` — the CET alternative

A drop-in alternative that puts every asset on one clock: European
equities (`ASML.AS`, `SAP.DE`, `AL2SI.PA`, `PKN.WA`) with commodity ETCs
listed on Borsa Italiana (`SGLD.MI`, `CRUD.MI`, `NGAS.MI`), all closing
at 17:30 CET apart from Warsaw at 17:00.

It carries no FX sleeve. There is no liquid CET-clock equivalent of
`FXE`/`FXY`/`FXB`: UCITS rules make single-currency funds awkward, so
European FX exposure is sold as thinly-traded ETPs whose zero-volume
sessions would reintroduce the exact VWAP problem described above. The
file documents the three available options.

To use it, rename it to `config.py`. The variable names are identical,
so `main.py` and `models/` are unaffected.

---

## 6. Modifying the project

- Universe, dates and every parameter: `config.py`
- Signal logic: `models/strategy.py`
- Performance metrics: `models/backtest.py`
- Optimisers and weighting schemes: `models/portfolio.py`
- Charts: `utils/plotting.py`

---

## 7. Known limitations

These are documented rather than hidden, because they bound what the
output can be used to claim.

**The portfolio section is in-sample.** Expected returns are the
historical mean scaled by 252, fed directly into a maximum-Sharpe
optimiser over the whole sample. Sample means are the noisiest possible
input to a mean-variance optimiser, and the resulting weights describe
what already happened rather than what to hold next. There is no
rebalancing, no walk-forward and no out-of-sample window. `MAX_WEIGHT`
limits concentration but does not fix this.

**Two different performance stories are printed side by side.** The
forward-return analysis reports 1, 5 and 20-day horizons, but the daily
backtest holds each signal for exactly one day. They describe different
strategies.

**The risk-free rate is a single constant.** `RISK_FREE_RATE = 0.04`
applies across the entire sample even though short rates moved from
roughly zero to roughly five per cent over it. Every Sharpe and Sortino
figure in the output is measured against this number.

**Currency exposure is unhedged and unlabelled.** `PKN.WA` is quoted in
złoty and `AL2SI.PA` in euro, so their return series blend the equity
move with a currency move against a USD base.

**Sortino uses zero as the threshold** rather than the minimum
acceptable return, so it is not directly comparable to the Sharpe figure
beside it.

**There are no tests.**

**`main.py` executes at import time.** It is a linear script with no
`main()` function and no `if __name__ == "__main__"` guard, so importing
it runs the entire pipeline, downloads included.

---

## 8. Provenance and licence

Upstream: [carlonimatteoo03/Quant_Portoflio](https://github.com/carlonimatteoo03/Quant_Portoflio),
by Matteo Carloni. `main.py`, `models/` and `utils/` are his work,
unmodified. The changes in this copy are confined to `config.py`,
`config_eu.py`, `requirements.txt`, `.gitignore` and this README.

Licensed under the MIT Licence. `LICENSE` must be kept alongside this
code wherever it is redistributed — the MIT terms require the copyright
notice and permission notice to travel with any copy or substantial
portion of the software. `setup_from_upstream.sh` copies it across
automatically.
