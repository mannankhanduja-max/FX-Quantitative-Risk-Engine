"""
Project configuration.

This is the main control panel of the QuantPortfolio project.
Change parameters here rather than changing the model code.

--------------------------------------------------------------
UNIVERSE NOTE
--------------------------------------------------------------

FX spot pairs (EURUSD=X, JPY=X, GBPJPY=X) and continuous
futures (CL=F, NG=F, GC=F) have been replaced by US-listed,
exchange-traded funds that give the same underlying exposure.

Two reasons:

1. Trading hours.
   FX quotes run ~24/5 and futures run nearly around the
   clock, so their daily "close" is sampled at a different
   moment from an equity close. Mixing them into one
   covariance matrix compares prices that were never
   observed at the same time. Every ETF below trades on
   NYSE Arca and closes at 16:00 ET, in line with AAPL and
   MSFT.

2. Volume.
   models/strategy.py builds VWAP from High/Low/Close and
   Volume, and filters out rows where Volume <= 0. Yahoo
   reports zero volume for FX spot pairs, so a VWAP on
   EURUSD=X was silently undefined for every observation.
   ETFs report real share volume, so VWAP is now meaningful
   across the whole universe.
"""

# ============================================================
# MARKET DATA
# ============================================================

TICKERS = [
    # ----- Equities -----
    "AAPL",       # Apple                        (NASDAQ)
    "MSFT",       # Microsoft                    (NASDAQ)
    "AL2SI.PA",   # 2CRSi SA                     (Euronext Growth Paris)
    "PKN.WA",     # PKN Orlen                    (Warsaw, PLN)

    # ----- FX, via currency ETFs (NYSE Arca) -----
    "FXE",        # Invesco CurrencyShares Euro Trust      ~ EUR/USD
    "FXY",        # Invesco CurrencyShares Japanese Yen    ~ JPY/USD
    "FXB",        # Invesco CurrencyShares British Pound   ~ GBP/USD

    # ----- Commodities, via commodity ETFs (NYSE Arca) -----
    "GLD",        # SPDR Gold Shares                       ~ spot gold
    "USO",        # United States Oil Fund                 ~ WTI crude
    "UNG",        # United States Natural Gas Fund         ~ Henry Hub gas
]

# 2CRSi listed on Euronext Growth in mid-2018, so it has no
# history before then. main.py cleans prices with .dropna(),
# which drops any date where ANY ticker is missing - so the
# usable sample is bounded by the shortest series regardless.
# The start date is set explicitly here rather than leaving
# the truncation to happen silently.
START_DATE = "2018-07-01"
END_DATE = "2026-08-01"

# Asset traded by the VWAP strategy.
# main.py previously fell back to "AAPL" via getattr because
# this was never defined. Now it is explicit.
STRATEGY_TICKER = "AAPL"

# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

PORTFOLIO_METHOD = "equal_weight"

RISK_FREE_RATE = 0.04

MAX_WEIGHT = 0.25

N_SIMULATIONS = 100000

# Used only when PORTFOLIO_METHOD = "manual".
# models/portfolio.py reads this and previously raised
# AttributeError because it did not exist. Weights must be
# keyed by ticker and sum to 1.0.
MANUAL_WEIGHTS = {
    "AAPL": 0.10,
    "MSFT": 0.10,
    "AL2SI.PA": 0.10,
    "PKN.WA": 0.10,
    "FXE": 0.10,
    "FXY": 0.10,
    "FXB": 0.10,
    "GLD": 0.10,
    "USO": 0.10,
    "UNG": 0.10,
}

# ============================================================
# VWAP STRATEGY SETTINGS
# ============================================================

# Number of observations used for rolling VWAP
VWAP_WINDOW = 20

# Historical lookback used to calculate percentile thresholds
THRESHOLD_WINDOW = 252

# Signal thresholds
LOWER_PERCENTILE = 5
UPPER_PERCENTILE = 95

# Holding horizons used for signal testing
HORIZONS = [1, 5, 20]

# ============================================================
# BACKTEST SETTINGS
# ============================================================

INITIAL_CAPITAL = 100000

TRANSACTION_COST = 0.001

# ============================================================
# OUTPUT SETTINGS
# ============================================================

SAVE_RESULTS = True
SHOW_PLOTS = True
