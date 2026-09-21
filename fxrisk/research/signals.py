"""
Candidate signals, computed causally, for measurement before trading.

Everything in this repository up to here tested RULES: an entry, a
bracket, a filter, and then asked whether the package made money.
Sixty-odd rules later the answer was uniformly no, and the reason was
always the same - the entry carried no directional information, and
no bracket can create information that is not there.

This module inverts the order. It computes a SIGNAL - one number per
bar that claims to say something about the next move - and nothing
else. `fxrisk.research.ic` then asks the only question that matters
first: is the signal correlated with what happens next, at all? A
bracket is only worth designing around a signal that passes.

THE CONTRACT
-------------
Every feature at bar t is computable at the CLOSE of bar t from data
at or before t. Forward returns start at that close. There is no
shift applied downstream to "fix" alignment - a feature that needed
one would be a bug, and `tests/test_signals.py` perturbs the future
to prove none does.

Sign convention: a positive value is a claim that price goes UP.
Mean-reversion features are therefore negated at source, so that a
positive IC always means "the signal was right" regardless of which
kind of signal it is.

THE CANDIDATES - FIXED BEFORE ANY WAS MEASURED
------------------------------------------------
Chosen to cover the ideas raised during the strategy work, plus the
two baselines any intraday study should report.

  mom_1h, mom_4h, mom_24h
      Past log return over 1, 4 and 24 hours. The baselines. Positive
      IC means momentum; negative means reversal. At short horizons
      in FX, weak reversal is the usual finding.

  vwap_rev
      Minus the distance from the session VWAP, in sigma units. A
      claim that price reverts to where the session's activity has
      been concentrated. The VWAP here is tick-weighted, as in
      `breakout_retrace.session_vwap`.

  flow_proxy
      Bar direction times the log ratio of tick volume to its trailing
      median - a signed activity shock. This is a PROXY for order
      flow and must be described as one. Real order flow needs to
      know whether each trade lifted the offer or hit the bid, and a
      candle with a tick count does not carry that. What it can say
      is "the market moved this way unusually busily".

  poc_rev
      Minus the distance from the PRIOR session's point of control,
      in sigma units. Volume-profile logic: price is drawn back to the
      level where the most business was done. The POC here is crude
      - the close of the busiest hourly bar of the previous session,
      not a true price-at-volume histogram - and is named for what it
      approximates rather than what it is.

  ldn_drive, ny_drive
      The return of the first hour after the London (03:00 ET) and
      New York (09:30 ET, taken as the 09:00 bar) opens, carried
      forward through the rest of that session and NaN before it.
      The market-open-pattern idea: does the opening hour's direction
      persist?

All distances are scaled by a causal EWMA sigma so that one number
means the same thing in a quiet week and a volatile one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxrisk.data.intraday import session_id

FEATURES = (
    "mom_1h", "mom_4h", "mom_24h",
    "vwap_rev", "flow_proxy", "poc_rev",
    "ldn_drive", "ny_drive",
)

HORIZONS = (1, 4, 24)          # in bars; on 1h bars, hours


def _sigma(logret: pd.Series, lam: float = 0.97) -> pd.Series:
    """
    EWMA volatility INCLUDING bar t.

    Unlike the one-step-ahead sigma used for a forecast, a scale for a
    feature observed at t's close may use t's own return - that return
    is known at the moment the feature is computed. Shifting it would
    be harmless but would scale every feature by a stale number.
    """
    return np.sqrt(logret.pow(2).ewm(alpha=1 - lam, adjust=False).mean())


def _session_vwap(bars: pd.DataFrame, sess: pd.Series) -> pd.Series:
    tp = (bars["High"] + bars["Low"] + bars["Close"]) / 3.0
    vol = pd.to_numeric(bars["Volume"], errors="coerce").fillna(0.0)
    num = (tp * vol).groupby(sess).cumsum()
    den = vol.groupby(sess).cumsum().replace(0.0, np.nan)
    return num / den


def _prior_poc(bars: pd.DataFrame, sess: pd.Series) -> pd.Series:
    """
    The prior session's busiest-bar close, known from that session's end.

    Built session by session so that nothing from the current session
    leaks in: a session's POC only becomes available to the NEXT one.
    """
    vol = pd.to_numeric(bars["Volume"], errors="coerce").fillna(0.0)
    frame = pd.DataFrame({"close": bars["Close"], "vol": vol, "sess": sess})
    poc = frame.groupby("sess").apply(
        lambda g: g["close"].iloc[int(np.argmax(g["vol"].to_numpy()))]
        if len(g) else np.nan,
        include_groups=False,
    )
    prior = poc.shift(1)
    return sess.map(prior)


def _open_drive(logret: pd.Series, sess: pd.Series, hour: int) -> pd.Series:
    """
    Return of the bar that STARTS at `hour` ET, carried forward within
    its session from that bar's close onward.

    The drive becomes known at the close of the opening bar, so the
    value on that bar itself is legitimate - it is observed at the
    moment the feature is read. Bars earlier in the session are NaN,
    not zero: before the open there is no drive to speak of, and a
    zero would be an assertion that there was a flat one.
    """
    idx = logret.index
    is_open = pd.Series(idx.hour == hour, index=idx)
    drive = logret.where(is_open)
    return drive.groupby(sess).ffill()


def compute(bars: pd.DataFrame, volume_window: int = 120) -> pd.DataFrame:
    """
    All eight candidate signals on one instrument's bars.

    `bars` must be regular intraday bars with Open/High/Low/Close/
    Volume and a timezone-aware index (see `fxrisk.data.intraday`).
    """
    close = bars["Close"].astype("float64")
    lr = np.log(close).diff()
    sig = _sigma(lr)
    sess = session_id(bars.index)
    sess = pd.Series(sess.to_numpy(), index=bars.index)

    out = pd.DataFrame(index=bars.index)

    out["mom_1h"] = lr
    out["mom_4h"] = lr.rolling(4).sum()
    out["mom_24h"] = lr.rolling(24).sum()

    vwap = _session_vwap(bars, sess)
    out["vwap_rev"] = -(np.log(close) - np.log(vwap)) / sig

    vol = pd.to_numeric(bars["Volume"], errors="coerce").astype("float64")
    med = vol.rolling(volume_window, min_periods=volume_window // 2).median()
    shock = np.log(vol.where(vol > 0) / med.where(med > 0))
    out["flow_proxy"] = np.sign(lr) * shock

    poc = _prior_poc(bars, sess)
    out["poc_rev"] = -(np.log(close) - np.log(poc)) / sig

    out["ldn_drive"] = _open_drive(lr, sess, hour=3)
    out["ny_drive"] = _open_drive(lr, sess, hour=9)

    # Scale-free momentum, so pooled ICs across instruments compare
    # like with like. Rank IC is scale-invariant per instrument
    # anyway; this matters only for the quintile edge in basis points.
    for c in ("mom_1h", "mom_4h", "mom_24h"):
        out[c] = out[c] / sig

    return out.replace([np.inf, -np.inf], np.nan)


def forward_returns(bars: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """
    Log return from the close of bar t to the close of bar t+h.

    Measured in bars, not clock time. Across a weekend the next bar is
    Monday's, so a 1-bar return on Friday's last bar spans the gap -
    which is the return a position held over it would actually earn.
    """
    lc = np.log(bars["Close"].astype("float64"))
    return pd.DataFrame(
        {f"fwd_{h}": lc.shift(-h) - lc for h in horizons}, index=bars.index)
