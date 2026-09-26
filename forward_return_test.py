"""
ONE-SHOT TEST. Does the cascade predict drift, or was the edge geometry?

Criteria and the asymmetry between the two outcomes are fixed in
docs/PREREG_forward_return.md, written before this ran. Read that
first; this file only executes it.

The barriers are gone. No stop, no target, no reward ratio, no bar
limit - just the signed forward log return from each entry at 1, 4, 24
and 96 trigger bars. Because neither the reward ratio nor the ATR stop
multiple changes which bars fire (the multiple is only a FLOOR on the
stop distance), this single population is the one underlying all 18
configurations the walk-forward searched.

Refuses to run twice. The marker is the point: a one-shot test that
can be re-run until it reads well is not a one-shot test.

BACKTEST-ONLY.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import config
from fxrisk.data import intraday
from fxrisk.research import event_study as es
from fxrisk.research import ic as icmod
from fxrisk.risk import spread
from fxrisk.strategies import mtf, smc

AGG = {"Open": "first", "High": "max", "Low": "min",
       "Close": "last", "Volume": "sum"}
CHUNKS = [("5m_2021", "5m_ask_2021"), ("5m_2023", "5m_ask_2023"),
          ("5m", "5m_ask")]

HORIZONS = (1, 4, 24, 96)          # 1h, 4h, 1 day, 4 days on the 1h trigger
KINDS = ("breakout", "fakeout", "retrace")
DONE = "results/forward_return_DONE.txt"
OUT = "results/forward_return.txt"


def resample(b5, rule):
    out = b5.resample(rule, label="left", closed="left").agg(AGG).dropna()
    return out[out["Volume"] > 0]


def chunk_events(inst, iv, ask_iv, commission_bp):
    """Entries and their forward returns for one instrument, one chunk."""
    try:
        b5 = intraday.load_symbol(inst.yahoo, iv)
    except FileNotFoundError:
        return None
    trig, setup_f = resample(b5, "1h"), resample(b5, "4h")
    if len(setup_f) < 300:
        return None

    cfg = mtf.MTFConfig(stop_sigma=1.0, stop_mode="atr", atr_period=14,
                        setup_lookback=20, trigger_window=6, max_bars_30m=80)
    zf = smc.zones(setup_f, use_fvg=True, use_ob=False)
    setups = mtf.setups_30m(setup_f, cfg, kinds=KINDS, zone_frame=zf)
    entries = mtf.entries_5m(trig, setups, None, cfg)
    if entries.empty:
        return None

    keep = mtf.in_sessions(pd.DatetimeIndex(entries["time"]),
                           mtf.INSTRUMENT_SESSIONS[inst.name])
    entries = entries[keep.to_numpy()].reset_index(drop=True)
    if not entries.empty:
        bad = spread.blackout(pd.DatetimeIndex(entries["time"]),
                              index_preopen=(inst.kind == "index"))
        entries = entries[~bad.to_numpy()].reset_index(drop=True)
    if entries.empty:
        return None

    ev = es.signed_forward(trig, entries, horizons=HORIZONS)
    ev["instrument"] = inst.name

    # Overextension of the trigger bar, in ATR units. Knowable at the
    # entry bar: `atr` is already shifted one bar inside mtf, so the
    # bar cannot size its own denominator.
    a = mtf.atr(trig, cfg.atr_period).reindex(pd.DatetimeIndex(ev["time"]))
    ev["stretch"] = (ev["entry"] - ev["stop"]).abs().to_numpy() / a.to_numpy()

    try:
        per_side = spread.real_cost_bp(trig, inst.yahoo, iv,
                                       commission_bp=commission_bp,
                                       ask_interval=ask_iv)
        ev["round_trip_bp"] = 2.0 * per_side.reindex(
            pd.DatetimeIndex(ev["time"])).to_numpy()
    except FileNotFoundError:
        ev["round_trip_bp"] = np.nan
    return ev


def report(ev, label, lines):
    lines.append(f"\n{label}   {len(ev)} events, "
                 f"{ev['instrument'].nunique()} instruments")
    lines.append(f"  {'h':>4}{'cells':>7}{'n':>7}{'mean bp':>10}{'t':>7}"
                 f"{'p':>9}{'>0':>7}{'cost bp':>9}{'holm':>6}")
    ps, rows = [], []
    for h in HORIZONS:
        cells = es.cell_means(ev, f"y{h}")
        s = es.pooled(cells)
        rows.append((h, s))
        ps.append(s["p"])
    rej = icmod.holm(ps, alpha=0.05)
    cost = float(np.nanmedian(ev["round_trip_bp"])) if len(ev) else float("nan")
    for (h, s), r in zip(rows, rej):
        lines.append(f"  {h:>4}{s['cells']:>7}{s['events']:>7}"
                     f"{s['mean_bp']:>+10.3f}{s['t']:>+7.2f}{s['p']:>9.4f}"
                     f"{s['share_pos']:>7.0%}{cost:>9.2f}"
                     f"{('YES' if r else '-'):>6}")
    return rows, rej, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commission-bp", type=float, default=0.35)
    ap.add_argument("--force", action="store_true",
                    help="ignore the one-shot marker (do not use)")
    args = ap.parse_args()

    if os.path.exists(DONE) and not args.force:
        raise SystemExit(f"{DONE} exists - this test has already been run "
                         f"once. Its result is in {OUT}. Re-running it and "
                         f"keeping the better answer is the exact failure "
                         f"the marker exists to prevent.")

    parts = []
    for inst in config.UNIVERSE_INTRADAY:
        for iv, ask in CHUNKS:
            ev = chunk_events(inst, iv, ask, args.commission_bp)
            if ev is not None:
                parts.append(ev)
    if not parts:
        raise SystemExit("no events - check the caches")

    ev = pd.concat(parts, ignore_index=True)
    ev["time"] = pd.to_datetime(ev["time"], utc=True, format="mixed")
    before = len(ev)
    ev = (ev.sort_values("time")
          .drop_duplicates(subset=["instrument", "time"], keep="first")
          .reset_index(drop=True))

    lines = []
    lines.append("FORWARD RETURN AFTER A SETUP FIRES - no stop, no target")
    lines.append(f"  ladder 1h/4h/1d   ATR(14) x1.0   kinds {', '.join(KINDS)}")
    lines.append(f"  data   {ev['time'].min().date()} -> {ev['time'].max().date()}"
                 f"   {before - len(ev)} seam duplicates dropped")
    lines.append("  cells  (instrument, calendar month), t taken ACROSS cells")
    lines.append(f"  cost   measured round trip + "
                 f"{2 * args.commission_bp:g}bp commission")

    rows, rej, cost = report(ev, "POOLED, all instruments", lines)

    lines.append("\nPER INSTRUMENT, mean bp (t)")
    insts = sorted(ev["instrument"].unique())
    lines.append("  " + f"{'instrument':<12}" +
                 "".join(f"{'h=' + str(h):>18}" for h in HORIZONS))
    signs = {h: 0 for h in HORIZONS}
    for name in insts:
        g = ev[ev["instrument"] == name]
        cell = []
        for h in HORIZONS:
            s = es.pooled(es.cell_means(g, f"y{h}"))
            if np.isfinite(s["mean_bp"]) and s["mean_bp"] > 0:
                signs[h] += 1
            cell.append(f"{s['mean_bp']:>+11.3f} ({s['t']:>+.2f})")
        lines.append(f"  {name:<12}" + "".join(f"{c:>18}" for c in cell))
    lines.append("  " + f"{'positive':<12}" +
                 "".join(f"{str(signs[h]) + '/' + str(len(insts)):>18}"
                         for h in HORIZONS))

    lines.append("\nBY SETUP KIND (descriptive - not part of the criteria)")
    lines.append("  " + f"{'kind':<12}" +
                 "".join(f"{'h=' + str(h):>18}" for h in HORIZONS))
    for k in KINDS:
        g = ev[ev["kind"] == k]
        cell = [f"{es.pooled(es.cell_means(g, f'y{h}'))['mean_bp']:>+11.3f}"
                f" ({es.pooled(es.cell_means(g, f'y{h}'))['t']:>+.2f})"
                for h in HORIZONS]
        lines.append(f"  {k:<12}" + "".join(f"{c:>18}" for c in cell))

    lines.append("\nSECONDARY, rank IC of stretch = |entry-stop| / ATR")
    lines.append(f"  {'h':>4}{'cells':>7}{'mean IC':>10}{'t':>7}{'edge bp':>10}")
    for h in HORIZONS:
        cs = []
        for name in insts:
            g = ev[ev["instrument"] == name].dropna(subset=["stretch", f"y{h}"])
            if g.empty:
                continue
            gi = g.set_index(pd.DatetimeIndex(g["time"]))
            # y is already in basis points and `cell_ics` scales its
            # quintile edge by 1e4, so it is handed back in raw return
            # units. Skipping this reports the edge 10,000x too large.
            cs.append(icmod.cell_ics(gi["stretch"], gi[f"y{h}"] / 1e4, name,
                                     min_obs=10))
        cells = pd.concat(cs, ignore_index=True) if cs else pd.DataFrame()
        r = icmod.summarise(cells, "stretch", h)
        lines.append(f"  {h:>4}{r.cells:>7}{r.mean_ic:>+10.4f}{r.t:>+7.2f}"
                     f"{r.edge_bp:>+10.3f}")

    # ---- the verdict, applied mechanically to the pre-registered rule
    passed = []
    for (h, s), r in zip(rows, rej):
        if r and s["mean_bp"] > 0 and s["t"] > 2.0 and signs[h] >= 3:
            passed.append((h, s))
    lines.append("\nVERDICT against docs/PREREG_forward_return.md")
    if not passed:
        lines.append("  FAIL. No horizon shows positive drift surviving Holm")
        lines.append("  with t > 2 and 3 of 4 instruments agreeing.")
        lines.append("  The barrier edge was not drift. It was geometry.")
    else:
        for h, s in passed:
            econ = "ECONOMIC PASS" if s["mean_bp"] > cost else "PASS, below cost"
            lines.append(f"  {econ} at h={h}: {s['mean_bp']:+.3f}bp "
                         f"vs {cost:.2f}bp round trip, t {s['t']:+.2f}")
        lines.append("  Read the asymmetry section of the prereg before")
        lines.append("  treating this as more than a reason for one more test.")

    text = "\n".join(lines) + "\n"
    print(text)
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write(text)
    with open(DONE, "w") as fh:
        fh.write(f"ran once on {pd.Timestamp.utcnow().isoformat()}\n"
                 f"criteria: docs/PREREG_forward_return.md\n"
                 f"result:   {OUT}\n")


if __name__ == "__main__":
    main()
