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

    __slots__ = ("name", "histdata", "yahoo", "available_from", "kind",
                 "shortable")

    def __init__(self, name, histdata, yahoo, available_from, kind="fx",
                 shortable=True):
        self.name = name
        self.histdata = histdata
        self.yahoo = yahoo
        self.available_from = available_from
        self.kind = kind
        # Can this actually be sold short at a retail broker?
        #
        # Not a modelling nicety. A backtest that holds +/-1 at all
        # times is assuming both directions are available, and for
        # the small currency ETFs they are not - Alpaca rejects the
        # order outright with "asset cannot be sold short", because
        # they are not on the easy-to-borrow list. Roughly 39% of
        # this repository's instrument-days were short positions in
        # those four names, i.e. trades that could never have been
        # opened.
        #
        # Borrow availability varies by broker and over time, so
        # this reflects what was actually rejected on 2026-09-12,
        # not a permanent property of the asset.
        self.shortable = shortable

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
    Instrument("Euro",        "FXE", "FXE", "2005-12", kind="fx_etf", shortable=False),
    Instrument("Pound",       "FXB", "FXB", "2006-06", kind="fx_etf", shortable=False),
    Instrument("Yen",         "FXY", "FXY", "2007-02", kind="fx_etf", shortable=False),
    Instrument("Swiss franc", "FXF", "FXF", "2006-06", kind="fx_etf", shortable=False),
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

# ---- Strategy risk ----
#
# Cost charged per unit of turnover, one-way, as a fraction of
# notional. 2bp is a defensible retail-to-institutional round
# number for a liquid US-listed ETF; it is an ASSUMPTION, not a
# measurement, and the cost-only share of losing days in the report
# is sensitive to it.
COST_PER_TURN = 0.0002

# Walk-forward GARCH refit interval for the strategy section only.
# Coarser than the 21 used elsewhere because this section fits two
# models per instrument and the strategy-series fit converges
# slowly - it pins to the IGARCH boundary, where the optimiser has
# no interior optimum to find. Accuracy for runtime, stated here
# rather than buried in a call site.
STRATEGY_REFIT_EVERY = 63

MC_SIMULATIONS = 20000

# Horizons in trading days. Monte Carlo is the honest way to get a
# multi-day figure: sqrt-time scaling assumes iid returns and
# ignores that volatility mean-reverts, so it overstates risk when
# current vol is above its long-run level and understates it below.
MC_HORIZONS = [1, 5, 10]

# ============================================================
# INTRADAY UNIVERSE - breakout/retracement strategy
#
# Source is Dukascopy, not Yahoo. `fxrisk/data/intraday.py` has
# the argument; the short form is that Yahoo reports Volume = 0
# for FX spot (so no VWAP is definable at all), has no Nasdaq-100
# and no spot gold, and serves a trailing 60 days of 15-minute
# bars with nothing behind them.
#
# These are the instruments themselves, not proxies: spot EUR/USD,
# spot gold, the Nasdaq-100 index, spot USD/JPY. Nothing here is
# quoted backwards and nothing rolls between contracts.
#
# `be_bp` is the breakeven trigger: how far price must travel in
# the trade's favour before the stop moves to entry. The request
# was "10 pips", and a pip is not a unit that survives contact
# with four quote conventions, so each is converted to basis
# points of entry price at a reference level. The arithmetic is
# written out so it can be argued with.
#
#   EUR/USD  10 pips = 0.0010 on 1.08           =  9.3 bp
#   USD/JPY  10 pips = 0.10 yen on 150.00       =  6.7 bp
#   XAU/USD  10 pips = $1.00 on 2400            =  4.2 bp
#            (taking one gold pip as $0.10, the more common retail
#             convention; some brokers say $0.01, which would make
#             this 0.42bp and the trigger essentially never fire)
#   NAS100   10 pips = 10 index points on 20000 =  5.0 bp
#
# Two things this hides. The conversion is level-dependent, so a
# fixed bp figure drifts as price moves - across two years that is
# worth knowing about, and `--be-pips` exists to test sensitivity.
# And the four are NOT the same distance in risk terms: 9.3bp of
# EUR/USD is a much larger move, in that instrument's own
# volatility, than 5.0bp of NAS100. "10 pips" is a round number in
# quote units, not a considered risk threshold.
# ============================================================

class IntradayInstrument(Instrument):
    """An instrument traded on intraday bars, with a pip scale."""

    __slots__ = ("be_bp", "dukascopy")

    def __init__(self, name, dukascopy, be_bp, kind="fx",
                 available_from="2003-01", shortable=True):
        # `yahoo` is inherited as the cache key, not as a Yahoo
        # symbol - the cache reader is source-agnostic and keys on
        # whatever string it is given. Kept as the slug so the
        # filename is stable if the source ever changes again.
        slug = dukascopy.replace("/", "").replace("-", "").replace("_", "")
        super().__init__(name, slug, slug, available_from, kind, shortable)
        # Dukascopy's own instrument identifier.
        self.dukascopy = dukascopy
        # Breakeven trigger in basis points of entry price.
        self.be_bp = be_bp


UNIVERSE_INTRADAY = [
    IntradayInstrument("EUR/USD", "EUR/USD",  be_bp=9.3),
    IntradayInstrument("XAU/USD", "XAU/USD",  be_bp=4.2, kind="metal"),
    IntradayInstrument("NAS100",  "E_NQ-100", be_bp=5.0, kind="index"),
    IntradayInstrument("USD/JPY", "USD/JPY",  be_bp=6.7),
]

# ============================================================
# BREAKOUT / RETRACEMENT STRATEGY
# ============================================================

# Bars of history the breakout level is taken from. 20 bars of
# 15 minutes is five hours - long enough that clearing it means
# something, short enough that a level survives inside one
# session.
BREAKOUT_LOOKBACK = 20

# A breakout must CLOSE beyond the level, not merely touch it.
# Wick-based confirmation turns every liquidity sweep into a
# signal and is the largest source of phantom trades in this
# family of strategy.

# How many bars the retracement may take before the setup is
# abandoned. Too short and normal pullbacks are missed; too long
# and the entry has nothing to do with the breakout any more.
RETRACE_MAX_BARS = 12

# How far back toward the breakout level price must pull before
# the setup is armed, as a fraction of the breakout impulse.
RETRACE_MIN_FRACTION = 0.33

# Past this fraction it is not a pullback, it is a failed
# breakout: beyond 1.0 price is back through the level it broke.
RETRACE_MAX_FRACTION = 1.0

# Reward-to-risk on the bracket. 2.0 puts the target twice as far
# from entry as the stop, so the cost-adjusted breakeven win rate
# is (1 + cost_R) / 3, a little over 33%.
BREAKOUT_RR = 2.0

# Stop distance floor, in EWMA sigmas. The stop normally sits at
# the retracement extreme; this stops it being placed a tick away
# when the pullback is shallow, which would make the cost in
# units of risk enormous.
BREAKOUT_STOP_MIN_SIGMA = 1.0

# Bars a trade may stay open before it is closed at the market.
BREAKOUT_MAX_BARS = 40

# Cost in basis points PER SIDE.
#
# Deliberately wider than a raw Dukascopy spread. Three things it
# has to cover, none of which the backtest models directly:
#
#   Bars are fetched on the BID, so a long entry fills above the
#   recorded price and the spread is not in the bar data at all.
#
#   Spreads are not constant. The average EUR/USD spread is well
#   under a basis point; at 13:30 on a payrolls Friday - which is
#   exactly when breakouts happen - it is several times that, and
#   this strategy is not a random sample of minutes.
#
#   Slippage on a stop is not the spread. A stop is a market order
#   into the move that triggered it.
#
# 1.0bp per side is a guess with a direction: too high rather than
# too low. `--cost-bp` exists so the result's sensitivity to it is
# measurable rather than assumed.
BREAKOUT_COST_BP = 1.0
