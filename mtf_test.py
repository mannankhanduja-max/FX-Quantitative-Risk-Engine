"""
The multi-timeframe cascade, measured.

    1h    bias      9 EMA of the session VWAP
    30m   setup     breakout / fakeout (sweep) / retracement
    5m    trigger   the entry bar
    30m   exit      stop, 2R target, time limit

Everything higher-timeframe is read from the last CLOSED bar; see
fxrisk/strategies/mtf.py and tests/test_mtf.py.

Reported against the Black-Scholes no-information benchmark, which
is the only fair comparison: a 2:1 target on a driftless walk is hit
first one time in three, so 33% is not an edge, it is the coin flip.

BACKTEST-ONLY.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

import config
from fxrisk.data import intraday
from fxrisk.research import barrier_prob as bp
from fxrisk.risk import barriers
from fxrisk.strategies import mtf

AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}


def resample(b5: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def run_one(inst, args, kinds):
    b5 = intraday.load_symbol(inst.yahoo, interval="5m")
    b30 = resample(b5, "30min")
    b1h = resample(b5, "1h")

    cfg = mtf.MTFConfig(stop_sigma=args.stop_sigma,
                        setup_lookback=args.lookback,
                        trigger_window=args.trigger_window,
                        max_bars_30m=args.max_bars)

    setups = mtf.setups_30m(b30, cfg, kinds=kinds)
    bias = mtf.hourly_bias(b1h, cfg)
    entries = mtf.entries_5m(b5, setups, bias, cfg)

    # Session filter, applied at the TRIGGER bar - the moment the
    # order would be sent. Filtering the 30m setup instead would
    # admit a trade whose entry landed an hour outside the window.
    if args.sessions != "all" and not entries.empty:
        names = (mtf.INSTRUMENT_SESSIONS[inst.name]
                 if args.sessions == "per-instrument"
                 else tuple(x.strip() for x in args.sessions.split("+")))
        keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]), names)
        entries = entries[keep.to_numpy()].reset_index(drop=True)

    pos = mtf.to_30m_positions(entries, b30)
    if pos.empty:
        return None, None, b5

    trades = barriers.walk_explicit(
        b30, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=args.rr, max_bars=cfg.max_bars_30m, cost_bp=args.cost_bp,
        breakeven_frac=None,
    )
    # walk_explicit drops overlapping entries, so the kind column
    # must be joined on the entry BAR, never positionally.
    p = pos.sort_values("bar").drop_duplicates("bar")
    lut = pd.Series(p["kind"].to_numpy(), index=b30.index[p["bar"].to_numpy()])
    trades["kind"] = trades["entry_time"].map(lut)
    return trades, pos, b5


def sharpe(r: pd.Series, per_year: float) -> float:
    if len(r) < 3 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(per_year))


def drawdown(r: pd.Series) -> float:
    eq = r.cumsum()
    return float((eq - eq.cummax()).min())


def report(name, trades, rr):
    s = barriers.summarise_explicit(trades, rr)
    b = bp.benchmark(trades)
    n = len(trades)
    r = trades["unit_net_R"]
    t = r.mean() / (r.std(ddof=1) / np.sqrt(n)) if n > 2 and r.std(ddof=1) else np.nan
    # Trades per year, from the actual span.
    span = (trades["entry_time"].iloc[-1] - trades["entry_time"].iloc[0]).days / 365.25
    per_year = n / span if span > 0 else np.nan
    print(f"\n{name}")
    print(f"  trades           {n}   ({per_year:.0f}/yr)"
          f"   dropped overlapping {trades.attrs.get('dropped_overlapping', 0)}")
    print(f"  win rate         {s['win_rate']:.2%}   "
          f"barrier-only {s['win_rate_barrier']:.2%}   "
          f"time exits {s['time_exit_share']:.1%}")
    print(f"  BS benchmark     {b['model_win']:.2%}   "
          f"realised {b['actual_win']:.2%}   z {b['z']:+.2f}"
          f"   (n={b['trades']}, {b['time_exits_excluded']} time exits excluded)")
    print(f"  cost hurdle      {s['breakeven_wr']:.2%}   "
          f"gap {s['gap_vs_breakeven']:+.2%}")
    print(f"  mean net R       {r.mean():+.4f}   t {t:+.2f}   "
          f"Sharpe {sharpe(r, per_year):+.2f}   maxDD {drawdown(r):.1f}R")
    return s, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--stop-sigma", type=float, default=1.0)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--trigger-window", type=int, default=6)
    ap.add_argument("--max-bars", type=int, default=20)
    ap.add_argument("--cost-bp", type=float, default=config.BREAKOUT_COST_BP)
    ap.add_argument("--sessions", default="per-instrument",
                    help="per-instrument | all | london+newyork | ...")
    ap.add_argument("--kinds", default="breakout,fakeout,retrace")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())

    print("MULTI-TIMEFRAME CASCADE  1h bias -> 30m setup -> 5m trigger -> 30m exit")
    print(f"  {args.rr:g}:1, stop floor {args.stop_sigma:g} sigma (5m), "
          f"{args.cost_bp:g}bp/side, setups: {','.join(kinds)}")
    print(f"  sessions: {args.sessions}")

    pool = []
    for inst in config.UNIVERSE_INTRADAY:
        try:
            trades, pos, _ = run_one(inst, args, kinds)
        except FileNotFoundError as e:
            print(f"\n{inst.name}: {e}")
            continue
        if trades is None or trades.empty:
            print(f"\n{inst.name}: no trades")
            continue
        report(inst.name, trades, args.rr)
        trades = trades.copy()
        trades["instrument"] = inst.name
        pool.append(trades)

    if not pool:
        return
    allt = pd.concat(pool, ignore_index=True).sort_values("entry_time")
    allt.attrs["dropped_overlapping"] = sum(
        t.attrs.get("dropped_overlapping", 0) for t in pool)
    print("\n" + "=" * 62)
    report("POOLED", allt, args.rr)

    print("\n  by setup type")
    for k, g in allt.groupby("kind"):
        if len(g) < 10:
            continue
        bb = bp.benchmark(g)
        print(f"    {k:<10} n={len(g):<6} win {(g['unit_net_R'] > 0).mean():.2%}"
              f"   BS {bb['model_win']:.2%}  z {bb['z']:+.2f}"
              f"   meanR {g['unit_net_R'].mean():+.4f}")

    print("\n  by half (split-half stability)")
    mid = len(allt) // 2
    for label, g in (("first", allt.iloc[:mid]), ("second", allt.iloc[mid:])):
        print(f"    {label:<7} n={len(g):<6} win {(g['unit_net_R'] > 0).mean():.2%}"
              f"   meanR {g['unit_net_R'].mean():+.4f}"
              f"   {g['entry_time'].iloc[0].date()} -> {g['entry_time'].iloc[-1].date()}")


if __name__ == "__main__":
    main()
