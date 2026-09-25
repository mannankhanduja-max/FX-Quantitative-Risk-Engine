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
from fxrisk.risk import barriers, spread
from fxrisk.strategies import mtf, smc

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
                        stop_mode=args.stop_mode,
                        vwap_filter=args.vwap_filter,
                        atr_period=args.atr_period,
                        setup_lookback=args.lookback,
                        trigger_window=args.trigger_window,
                        max_bars_30m=args.max_bars)

    zf = None
    if args.confirm != "none":
        zf = smc.zones(b30, use_fvg=args.confirm in ("fvg", "both"),
                       use_ob=args.confirm in ("ob", "both"))
    setups = mtf.setups_30m(b30, cfg, kinds=kinds, zone_frame=zf)
    bias = mtf.hourly_bias(b1h, cfg) if args.bias else None
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

    # Spread blackouts: refuse the rollover and the recurring
    # release slots outright. Clock rules, stated in advance.
    if args.blackout and not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)

    pos = mtf.to_30m_positions(entries, b30)
    if pos.empty:
        return None, None, b5

    # Cost: flat, or shaped by activity so the rollover and the
    # dead hours pay what they should. Same median either way.
    if args.cost == "real":
        cost = spread.real_cost_bp(b30, inst.yahoo, "5m",
                                   commission_bp=args.commission_bp).to_numpy()
    elif args.cost == "activity":
        cost = spread.cost_bp_series(b30, base_bp=args.cost_bp,
                                     power=args.cost_power).to_numpy()
    else:
        cost = args.cost_bp

    trades = barriers.walk_explicit(
        b30, pos.sort_values("bar")[["bar", "side", "entry", "stop"]],
        rr=args.rr, max_bars=cfg.max_bars_30m, cost_bp=cost,
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
    ap.add_argument("--rr", type=float, default=3.0)
    ap.add_argument("--stop-sigma", type=float, default=1.0)
    ap.add_argument("--lookback", type=int, default=20)
    ap.add_argument("--trigger-window", type=int, default=6)
    ap.add_argument("--max-bars", type=int, default=80)
    ap.add_argument("--cost-bp", type=float, default=config.BREAKOUT_COST_BP)
    ap.add_argument("--sessions", default="per-instrument",
                    help="per-instrument | all | london+newyork | ...")
    ap.add_argument("--blackout", action="store_true", default=True)
    ap.add_argument("--no-blackout", dest="blackout", action="store_false")
    ap.add_argument("--cost", default="real",
                    choices=("real", "activity", "flat"),
                    help="real = measured ask-bid; the others are proxies")
    # 0.35bp per side is an ordinary retail raw/ECN commission. The
    # headline figure should be what an account actually pays, not a
    # frictionless number that needs a footnote.
    ap.add_argument("--commission-bp", type=float, default=0.35,
                    help="per side, on top of the measured spread")
    ap.add_argument("--cost-power", type=float, default=0.5)
    # The 1h session-VWAP / 9-EMA direction filter is OFF by default.
    # Measured twice on different geometries: it removes two thirds of
    # the trades and the win rate goes UP without it. --bias restores
    # it for comparison.
    ap.add_argument("--bias", action="store_true", default=False)
    ap.add_argument("--no-bias", dest="bias", action="store_false")
    ap.add_argument("--vwap-filter", default="none",
                    choices=("none","revert","trend"))
    ap.add_argument("--stop-mode", default="atr", choices=("sigma","atr"))
    ap.add_argument("--atr-period", type=int, default=14)
    ap.add_argument("--confirm", default="fvg",
                    choices=("none", "fvg", "ob", "both"),
                    help="retracement must land in a fair value gap "
                         "and/or an order block")
    ap.add_argument("--kinds", default="breakout,fakeout,retrace")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())

    print("MULTI-TIMEFRAME CASCADE  1h bias -> 30m setup -> 5m trigger -> 30m exit")
    print(f"  {args.rr:g}:1, stop floor {args.stop_sigma:g} sigma (5m), "
          f"{args.cost_bp:g}bp/side, setups: {','.join(kinds)}")
    print(f"  stop basis: {args.stop_mode}"
          + (f" ({args.atr_period})" if args.stop_mode=="atr" else "")
          + f"   retrace confirmation: {args.confirm}")
    print(f"  1h bias filter: {'on' if args.bias else 'OFF (no VWAP, no 9 EMA)'}")
    print(f"  sessions: {args.sessions}   blackout: {args.blackout}   "
          f"cost: {args.cost}"
          + (f" +{args.commission_bp:g}bp commission" if args.commission_bp else ""))

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
