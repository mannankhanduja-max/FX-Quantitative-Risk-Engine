"""
Conditioned variants of the VWAP/EMA bracket, and their win rates.

Six variants, all on a 1:1 bracket with a 3-sigma stop, all sharing
one pre-computed walk-forward GARCH path so the comparison is
like-for-like:

  baseline      every signal, fixed size
  skip_nfp      no new entry on a first Friday
  only_nfp      entries on first Fridays only
  low_vol_only  enter only when the GARCH forecast sits below its
                own trailing median
  var_cap       skip when the 99% GARCH-t VaR forecast exceeds 1.5x
                its trailing median
  vol_target    every signal, sized to a constant risk budget

A NOTE ON WHAT SIZING CAN AND CANNOT DO. vol_target changes how
much is staked per trade, not which trades are taken, so its WIN
RATE is identical to baseline by construction. Only its R
distribution moves. Any write-up claiming a sizing rule improved
hit rate has a bug.

Every conditioning input is walk-forward: the GARCH variance and
the VaR come from `rolling_garch_forecasts`, which fits on data
strictly before the forecast date, and the medians they are
compared against are expanding rather than full-sample. A filter
built on a full-sample median would be choosing its trades with
hindsight.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/mannan.k/fx-risk-engine")

import config
from fxrisk import indicators
from fxrisk.data import yahoo
from fxrisk.models.garch import rolling_garch_forecasts
from fxrisk.risk.var import garch_var_series

from calendar_probe import flags

STOP_K = 3.0
RR = 1.0
MAX_DAYS = 10
COST_BP = 2.0
REFIT = 126


def ewma_sigma(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    var = returns.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(var).shift(1)


def run_barriers(
    bars: pd.DataFrame,
    signal: pd.Series,
    sigma: pd.Series,
    allow: pd.Series,
    size: pd.Series,
) -> pd.DataFrame:
    """
    Barrier walk. `allow` gates entries; `size` scales the stake.

    Stake scaling multiplies the R outcome but never changes which
    barrier is hit, so win rate is independent of `size`.
    """
    close, high, low = bars["Close"], bars["High"], bars["Low"]
    sig = signal.reindex(bars.index).fillna(0.0)
    flips = sig.diff().fillna(sig).ne(0) & sig.ne(0)

    idx = bars.index
    n = len(idx)
    trades = []
    i = 0

    while i < n - 1:
        if (
            not flips.iloc[i]
            or not bool(allow.iloc[i])
            or not np.isfinite(sigma.iloc[i])
            or sigma.iloc[i] <= 0
        ):
            i += 1
            continue

        side = float(np.sign(sig.iloc[i]))
        entry = float(close.iloc[i])
        stop_d = STOP_K * float(sigma.iloc[i]) * entry
        target = entry + side * stop_d * RR
        stop = entry - side * stop_d

        exit_px, held, ambiguous, outcome = None, 0, False, "time"
        for j in range(i + 1, min(i + 1 + MAX_DAYS, n)):
            held = j - i
            hi, lo = float(high.iloc[j]), float(low.iloc[j])
            hit_t = hi >= target if side > 0 else lo <= target
            hit_s = lo <= stop if side > 0 else hi >= stop
            if hit_t and hit_s:
                ambiguous, outcome, exit_px = True, "stop", stop
                break
            if hit_s:
                outcome, exit_px = "stop", stop
                break
            if hit_t:
                outcome, exit_px = "target", target
                break
        if exit_px is None:
            exit_px = float(close.iloc[min(i + MAX_DAYS, n - 1)])

        stake = float(size.iloc[i]) if np.isfinite(size.iloc[i]) else 1.0
        gross_r = side * (exit_px - entry) / stop_d
        cost_r = (2 * COST_BP / 10_000) * entry / stop_d

        trades.append({
            "date": idx[i],
            "outcome": outcome,
            "days": held,
            "ambiguous": ambiguous,
            "net_R": (gross_r - cost_r) * stake,
            "unit_net_R": gross_r - cost_r,
            "stake": stake,
        })
        i += max(held, 1)

    return pd.DataFrame(trades)


def build(sym: str) -> dict:
    """Everything needed for one instrument, computed once."""
    bars = yahoo.load_symbol(sym)
    ret = bars["Close"].pct_change(fill_method=None).dropna()
    sig = indicators.signal(
        bars, window=config.VWAP_WINDOW_DAILY, span=config.VWAP_EMA_SPAN
    )
    sigma = ewma_sigma(bars["Close"].pct_change(fill_method=None))

    fc = rolling_garch_forecasts(ret, refit_every=REFIT, dist="t")
    gvol = np.sqrt(fc["variance"]).reindex(bars.index)
    var99 = garch_var_series(
        ret, confidence=0.99, refit_every=REFIT
    )["var"].reindex(bars.index)

    fl = flags(bars.index)

    # Expanding medians: no hindsight in the thresholds.
    gvol_med = gvol.expanding(min_periods=250).median()
    var_med = var99.expanding(min_periods=250).median()

    return dict(
        bars=bars, sig=sig, sigma=sigma, gvol=gvol, var99=var99,
        fl=fl, gvol_med=gvol_med, var_med=var_med,
    )


def variants(d: dict) -> dict[str, tuple[pd.Series, pd.Series]]:
    """(allow, size) per variant."""
    idx = d["bars"].index
    yes = pd.Series(True, index=idx)
    one = pd.Series(1.0, index=idx)

    nfp = d["fl"]["nfp"]
    low_vol = (d["gvol"] < d["gvol_med"]).fillna(False)
    var_ok = (d["var99"] <= 1.5 * d["var_med"]).fillna(False)

    # Constant risk budget: stake inversely proportional to forecast
    # volatility, clipped so a quiet regime cannot lever the book up
    # without limit.
    target = d["gvol"].expanding(min_periods=250).median()
    stake = (target / d["gvol"]).clip(0.25, 4.0).fillna(1.0)

    return {
        "baseline": (yes, one),
        "skip_nfp": (~nfp, one),
        "only_nfp": (nfp, one),
        "low_vol_only": (low_vol, one),
        "var_cap": (var_ok, one),
        "vol_target": (yes, stake),
    }


def main() -> int:
    rows = []
    for inst in config.UNIVERSE:
        d = build(inst.yahoo)
        for name, (allow, size) in variants(d).items():
            t = run_barriers(d["bars"], d["sig"], d["sigma"], allow, size)
            if t.empty:
                continue
            wins = t["unit_net_R"] > 0
            cost_r = float((t["unit_net_R"] - t["unit_net_R"]).abs().mean())
            rows.append({
                "variant": name,
                "symbol": inst.yahoo,
                "trades": len(t),
                "win_rate": float(wins.mean()),
                "mean_R": float(t["net_R"].mean()),
                "total_R": float(t["net_R"].sum()),
                "target_share": float((t["outcome"] == "target").mean()),
                "mean_days": float(t["days"].mean()),
            })

    out = pd.DataFrame(rows)

    print("WIN RATE by variant\n")
    print(out.pivot(index="variant", columns="symbol",
                    values="win_rate").round(4).to_string())

    print("\nTOTAL R by variant\n")
    print(out.pivot(index="variant", columns="symbol",
                    values="total_R").round(1).to_string())

    print("\nPOOLED\n")
    agg = out.groupby("variant").apply(
        lambda g: pd.Series({
            "trades": int(g["trades"].sum()),
            "win_rate": float((g["win_rate"] * g["trades"]).sum()
                              / g["trades"].sum()),
            "total_R": float(g["total_R"].sum()),
            "mean_R": float((g["mean_R"] * g["trades"]).sum()
                            / g["trades"].sum()),
            "positive_instruments": int((g["total_R"] > 0).sum()),
        })
    )
    agg = agg.sort_values("total_R", ascending=False)
    print(agg.round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
