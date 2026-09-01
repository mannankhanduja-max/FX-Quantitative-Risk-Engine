"""
FX risk engine configuration.
"""

# ============================================================
# DATA
# ============================================================

# Folder containing unzipped HistData ASCII archives, one
# subfolder per pair, e.g.
#
#   data/histdata/EURUSD/DAT_ASCII_EURUSD_M1_2015.csv
#   data/histdata/EURUSD/DAT_ASCII_EURUSD_T_201501.csv
#
# Download from https://www.histdata.com/download-free-forex-data/
DATA_DIR = "data/histdata"

# ------------------------------------------------------------
# THE UNIVERSE - single source of truth
#
# Both pipelines in this repository read this list:
#
#   run_risk_report.py  loads the `histdata` id from local files
#   quant_metrics.py    downloads the `yahoo` symbol from Yahoo
#
# They therefore describe the same book. Before this was
# centralised the two disagreed - the engine held four dollar
# pairs while quant_metrics.py held gold, EUR/USD and GBP/JPY -
# and nothing in the code said which was intended.
#
# `available_from` is HistData's own start date for the
# instrument, taken from their download page. It matters because
# main-loop cleaning drops any date where ANY instrument is
# missing, so the shortest history bounds the whole sample.
# ------------------------------------------------------------

class Instrument:
    """One tradable instrument, addressable in both data sources."""

    __slots__ = ("name", "histdata", "yahoo", "available_from", "kind")

    def __init__(self, name, histdata, yahoo, available_from, kind="fx"):
        self.name = name
        self.histdata = histdata
        self.yahoo = yahoo
        self.available_from = available_from
        self.kind = kind

    def __repr__(self):
        return f"Instrument({self.histdata})"


# Default universe: four majors plus one cross. Full HistData
# history back to 2002, which keeps the 2008 stress scenarios
# testable.
UNIVERSE_FX = [
    Instrument("EUR/USD", "EURUSD", "EURUSD=X", "2000-05"),
    Instrument("GBP/USD", "GBPUSD", "GBPUSD=X", "2000-05"),
    Instrument("USD/JPY", "USDJPY", "USDJPY=X", "2000-05"),
    Instrument("USD/CHF", "USDCHF", "USDCHF=X", "2000-05"),
    Instrument("GBP/JPY", "GBPJPY", "GBPJPY=X", "2002-05"),
]

# Same book plus spot gold.
#
# READ THIS BEFORE SWITCHING. HistData's XAU/USD starts in March
# 2009. Because cleaning drops dates where any instrument is
# missing, adding gold truncates the ENTIRE sample to 2009+ and
# the Lehman and full-GFC stress scenarios stop being testable -
# they will be reported as skipped, which is correct behaviour
# and also a real loss. Gold is genuinely useful in an FX book as
# the dollar-stress hedge, so this is a trade, not a mistake in
# either direction.
UNIVERSE_FX_GOLD = UNIVERSE_FX + [
    Instrument("Gold spot", "XAUUSD", "XAUUSD=X", "2009-03", kind="metal"),
]

UNIVERSE = UNIVERSE_FX

# Derived views. Nothing downstream should hard-code a symbol.
PAIRS = [i.histdata for i in UNIVERSE]
YAHOO_SYMBOLS = {i.name: i.yahoo for i in UNIVERSE}
UNIVERSE_STARTS_AFTER = max(i.available_from for i in UNIVERSE)

# "tick" gives a genuine tick-count-weighted VWAP.
# "m1" is far smaller but has no volume, so VWAP degrades to TWAP.
DATA_GRANULARITY = "m1"

# FX has no exchange close. 17:00 New York is the market convention
# and is what the daily return series is cut on.
SESSION_CLOSE = "17:00"
SESSION_TZ = "America/New_York"

# ============================================================
# VOLATILITY MODELS
# ============================================================

# RiskMetrics (1996) daily decay factor.
EWMA_LAMBDA = 0.94

# Innovation distribution for GARCH. "t" is the default because
# Gaussian innovations systematically understate tail risk.
GARCH_DIST = "t"

# Walk-forward settings for the VaR backtest.
GARCH_WINDOW = 750        # ~3 years of estimation data
GARCH_REFIT_EVERY = 21    # refit monthly, recurse daily in between
GARCH_MIN_OBS = 250

# ============================================================
# VaR
# ============================================================

# Backtest at all three. A model that only survives at 99% is
# surviving where the independence test has least power.
CONFIDENCE_LEVELS = [0.95, 0.975, 0.99]

HISTORICAL_WINDOW = 500   # observations in the rolling empirical quantile

# ============================================================
# PORTFOLIO
# ============================================================

# Equal weight across PAIRS unless overridden.
WEIGHTS = None

# ============================================================
# STRESS TESTING
# ============================================================

# Keys from fxrisk.risk.stress.SCENARIOS. Scenarios outside the
# data window are skipped and reported as skipped.
STRESS_SCENARIOS = [
    "gfc_2008",
    "snb_2015",
    "cny_2015",
    "brexit_2016",
    "covid_2020",
    "gilt_2022",
]

# If an asset has no data in a scenario window, renormalise the
# remaining weights rather than treating it as flat. Missing names
# are reported either way.
STRESS_RESCALE_MISSING = True

# ============================================================
# OUTPUT
# ============================================================

RESULTS_DIR = "results"
SAVE_RESULTS = True

# ============================================================
# DCC (walk-forward, used for portfolio VaR)
# ============================================================

# DCC refitting is far more expensive than univariate GARCH: each
# fit runs one MLE per asset plus a likelihood optimisation over the
# whole window. Semi-annual refits keep the backtest tractable while
# staying strictly walk-forward.
DCC_REFIT_EVERY = 126
DCC_MIN_OBS = 400

# ============================================================
# PERFORMANCE
# ============================================================

# Annual risk-free rate used for Sharpe and Sortino. This is an
# ANNUAL figure and is de-annualised internally before being
# subtracted from periodic returns.
#
# A single constant across a long sample is a real simplification:
# short rates went from near zero to around five per cent over
# 2017-2026, and every Sharpe below is measured against this one
# number. Set it to the average over your actual sample window
# rather than today's spot rate.
RISK_FREE_RATE = 0.04

# ~6 months. Short enough to show regime shifts, long enough that
# the estimate is not pure noise.
ROLLING_SHARPE_WINDOW = 126
