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

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]

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
