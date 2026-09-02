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


# ------------------------------------------------------------
# ETF UNIVERSE - the default
#
# Every instrument is an NYSE Arca ETF, which buys three things
# that FX spot pairs cannot give:
#
#   ONE CLOCK. All six close at 16:00 ET. Daily closes are
#   therefore observed at the same instant, so the covariance
#   matrix compares like with like. FX spot quotes run ~24/5 and
#   their "close" is a vendor convention, which biases
#   correlations toward zero (the Epps effect) before they reach
#   any optimiser.
#
#   REAL VOLUME. Yahoo reports Volume = 0 for every FX spot pair,
#   which makes a volume-weighted average price undefined. ETFs
#   report actual share volume, so VWAP - and the 9-period EMA of
#   it - are computable rather than fabricated.
#
#   HISTORY THROUGH 2008. Inception dates below are all pre-crisis,
#   so the Lehman, SNB, CNY and Brexit stress scenarios stay
#   testable. Starting the sample at 2018 would silently discard
#   four of the six scenarios.
#
# Direction note: FXE/FXB/FXY/FXF are quoted as FOREIGN CURRENCY
# per USD. FXY rising means the yen strengthening, i.e. USD/JPY
# falling. Signs are inverted relative to the USD-base pairs this
# universe replaced. That matters for interpreting a correlation,
# not for measuring risk.
# ------------------------------------------------------------

UNIVERSE_ETF = [
    Instrument("Euro",        "FXE", "FXE", "2005-12", kind="fx_etf"),
    Instrument("Pound",       "FXB", "FXB", "2006-06", kind="fx_etf"),
    Instrument("Yen",         "FXY", "FXY", "2007-02", kind="fx_etf"),
    Instrument("Swiss franc", "FXF", "FXF", "2006-06", kind="fx_etf"),
    Instrument("Gold",        "GLD", "GLD", "2004-11", kind="metal_etf"),
]

# GBP/JPY has no direct ETF. It can be built synthetically as
# FXB / FXY - both are quoted per USD, so the ratio is GBP/JPY -
# and both trade on the same clock with real volume. Enable with
# SYNTHETIC_CROSSES; the caveat is that each leg carries its own
# expense ratio and tracking error, so the level drifts from the
# true cross over years even though daily returns track closely.
SYNTHETIC_CROSSES = {
    # "GBP/JPY": ("FXB", "FXY"),
}

# ---- The FX spot universe, kept for the HistData path ----
#
# Retained because histdata.py reads local tick files, where spot
# pairs are the only thing available and the 17:00 New York cut is
# a deliberate choice rather than a vendor default.
UNIVERSE_FX = [
    Instrument("EUR/USD", "EURUSD", "EURUSD=X", "2000-05"),
    Instrument("GBP/USD", "GBPUSD", "GBPUSD=X", "2000-05"),
    Instrument("USD/JPY", "USDJPY", "USDJPY=X", "2000-05"),
    Instrument("USD/CHF", "USDCHF", "USDCHF=X", "2000-05"),
    Instrument("GBP/JPY", "GBPJPY", "GBPJPY=X", "2002-05"),
]

# Spot FX plus gold. HistData's XAU/USD starts 2009-03, so adding
# it truncates the sample past the 2008 scenarios - a real trade,
# and the reason the ETF universe is preferred: GLD reaches back
# to 2004.
UNIVERSE_FX_GOLD = UNIVERSE_FX + [
    Instrument("Gold spot", "XAUUSD", "XAUUSD=X", "2009-03", kind="metal"),
]

UNIVERSE = UNIVERSE_ETF

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


# ============================================================
# VWAP / EMA INDICATOR
# ============================================================

# Rolling VWAP window in bars. Meaningful only on the ETF
# universe: FX spot reports zero volume, which makes VWAP
# undefined rather than merely noisy.
VWAP_WINDOW_DAILY = 20

# EMA span applied to the VWAP series.
VWAP_EMA_SPAN = 9

# ============================================================
# MONTE CARLO
# ============================================================

MC_SIMULATIONS = 20000

# Horizons in trading days. Monte Carlo is the honest way to get a
# multi-day figure: sqrt-time scaling assumes iid returns and
# ignores that volatility mean-reverts, so it overstates risk when
# current vol is above its long-run level and understates it below.
MC_HORIZONS = [1, 5, 10]
