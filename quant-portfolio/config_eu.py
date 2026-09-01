"""
Project configuration - EUROPEAN / CET-CLOCK UNIVERSE.

Drop-in replacement for config.py. Same variable names, so
main.py and models/ need no changes: rename this to config.py,
or import it instead.

--------------------------------------------------------------
WHY THIS VERSION EXISTS
--------------------------------------------------------------

The US-ETF version fixed the FX/futures problem (24-hour
instruments sampled against a 16:00 ET equity close) but left
a US-vs-Europe one: PKN.WA and AL2SI.PA close around 17:00-
17:30 CET, roughly 11:00-11:30 ET, so they were ~4.5 hours
stale against the US closes sitting in the same covariance
matrix. Non-synchronous closes bias correlations downward -
the classic Epps effect - which then feeds straight into the
optimiser.

This universe puts every asset on one clock:

    Euronext Paris / Amsterdam  17:30 CET close
    Xetra                       17:30 CET close
    Borsa Italiana (ETFplus)    17:30 CET close
    Warsaw                      17:00 CET close

Warsaw is still 30 minutes early. That is a residual, not a
solved problem - but it is an order of magnitude better than
4.5 hours.

--------------------------------------------------------------
TWO JUDGEMENT CALLS - REVIEW THESE
--------------------------------------------------------------

1. AAPL and MSFT are gone.
   They cannot be on a CET clock. To keep the shape of the
   universe - two mega-cap tech names plus two smaller
   industrials - they are replaced by ASML and SAP. If you
   would rather keep the US names, use the US-ETF config
   instead and accept the stale-close bias.

2. There is no FX sleeve.
   See the FX section below. This is a real gap, not an
   oversight.
"""

# ============================================================
# MARKET DATA
# ============================================================

TICKERS = [
    # ----- Equities -----
    "ASML.AS",    # ASML Holding        (Euronext Amsterdam, EUR)
    "SAP.DE",     # SAP SE              (Xetra, EUR)
    "AL2SI.PA",   # 2CRSi SA            (Euronext Growth Paris, EUR)
    "PKN.WA",     # PKN Orlen           (Warsaw, PLN)

    # ----- Commodities, via ETCs on Borsa Italiana (EUR) -----
    "SGLD.MI",    # Invesco Physical Gold ETC      ~ spot gold
    "CRUD.MI",    # WisdomTree WTI Crude Oil       ~ WTI crude
    "NGAS.MI",    # WisdomTree Natural Gas         ~ Henry Hub gas
]

# ------------------------------------------------------------
# FX - DELIBERATELY EMPTY
# ------------------------------------------------------------
#
# There is no liquid, CET-clock equivalent of FXE / FXY / FXB.
# UCITS rules make single-currency funds awkward, so European
# FX exposure is sold as ETPs rather than ETFs, and the ones
# that exist are thin: WisdomTree's currency range on the LSE
# (LEUR.L, SEUR.L, SJPP.L, GBUS.L and relatives) often trades
# a few thousand pounds a day, with stale prints and outright
# zero-volume sessions.
#
# That matters specifically here. models/strategy.py drops
# rows where Volume <= 0 before computing VWAP - the exact
# failure mode that made VWAP undefined on the original FX
# spot pairs. Swapping zero-volume FX spot for near-zero-
# volume FX ETPs would reintroduce it wearing a different hat.
#
# They are also mostly SHORT or inverse products (SJPP = short
# JPY long USD, GBUS = long USD short GBP), so the sign of the
# exposure is the opposite of what the name suggests at a
# glance.
#
# Three honest options:
#
#   a) Leave FX out. A 7-asset equity + commodity universe on
#      one clock is defensible. This is the default here.
#
#   b) Keep FX on US ETFs (FXE/FXY/FXB) and accept that the
#      FX sleeve alone is 4.5 hours late. Least-bad if the FX
#      exposure is the point of the exercise.
#
#   c) Model FX separately from the portfolio - run the VWAP
#      strategy on FX spot with a volume-free VWAP variant,
#      and keep the optimiser to the CET names. Cleanest, but
#      needs a code change in models/strategy.py.
#
# ------------------------------------------------------------

# 2CRSi listed on Euronext Growth in mid-2018. main.py cleans
# with .dropna(), which drops any date where ANY ticker is
# missing, so the usable sample is bounded by the shortest
# series regardless of what is set here.
START_DATE = "2018-07-01"
END_DATE = "2026-08-01"

# Asset traded by the VWAP strategy. Needs real, consistent
# volume - ASML is the most liquid name in this universe.
STRATEGY_TICKER = "ASML.AS"

# ============================================================
# PORTFOLIO SETTINGS
# ============================================================

PORTFOLIO_METHOD = "equal_weight"

# NOTE: this portfolio is now EUR-denominated (PKN.WA aside).
# A 4% USD-style risk-free rate is the wrong benchmark for it -
# every Sharpe and Sortino in the output is measured against
# this number. Set it to a EUR risk-free proxy, and prefer
# the average over your actual sample window rather than
# today's spot rate.
RISK_FREE_RATE = 0.04

# 7 assets, so the binding minimum is 1/7 = 14.3%. A 25% cap
# still leaves the optimiser meaningful room.
MAX_WEIGHT = 0.25

N_SIMULATIONS = 100000

# Used only when PORTFOLIO_METHOD = "manual".
# Must be keyed by ticker and sum to 1.0.
MANUAL_WEIGHTS = {
    "ASML.AS": 0.15,
    "SAP.DE": 0.15,
    "AL2SI.PA": 0.10,
    "PKN.WA": 0.15,
    "SGLD.MI": 0.20,
    "CRUD.MI": 0.15,
    "NGAS.MI": 0.10,
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
