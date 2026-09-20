"""
Precompute the GARCH and correlation paths the regime gates need.

    python build_regime.py
    python build_regime.py --refit-every 500     # finer, much slower

Walk-forward GARCH over ~49,000 bars is the expensive step in this
repository - an MLE per refit, per instrument - so it is done once
and cached to data/intraday/regime_<SYMBOL>_15m.csv. Everything
downstream reads the cache.

This is the same split `fetch_intraday.py` uses for the same
reason: a slow, non-deterministic step should happen once, and
every run after it should be fast and identical.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from fxrisk.data import intraday  # noqa: E402
from fxrisk.strategies import regime  # noqa: E402

CACHE = "data/intraday"


def path_for(symbol: str, interval: str, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"regime_{symbol}_{interval}.csv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--window", type=int, default=2000)
    ap.add_argument("--refit-every", type=int, default=1000)
    ap.add_argument("--lam", type=float, default=0.97)
    args = ap.parse_args()

    universe = config.UNIVERSE_INTRADAY

    print("Loading bars and aligning returns on one index\n")
    bars, rets = {}, {}
    for inst in universe:
        b = intraday.load_symbol(inst.yahoo, args.interval, args.cache_dir)
        bars[inst.name] = b
        rets[inst.name] = np.log(b["Close"]).diff()

    # Correlation needs the instruments on a common index. An outer
    # join with a forward fill would invent quotes for a market that
    # was closed; an inner join keeps only bars where every
    # instrument actually traded, which is the honest choice and
    # costs coverage rather than correctness.
    panel = pd.DataFrame(rets).dropna(how="any")
    print(f"  common index: {len(panel)} bars, "
          f"{panel.index[0]} -> {panel.index[-1]}")
    for inst in universe:
        own = rets[inst.name].dropna()
        print(f"    {inst.name:8s} {len(own):6d} own bars, "
              f"{len(panel) / max(len(own), 1):.0%} survive the join")

    print(f"\nEWMA correlation paths, lambda {args.lam}")
    rho = regime.ewma_correlation_path(panel, lam=args.lam)
    for (i, j), s in rho.items():
        print(f"  {i:8s} {j:8s} median rho {s.median():+.3f}  "
              f"|rho|>0.3 on {float((s.abs() > 0.3).mean()):.0%} of bars")

    cfg = regime.GateConfig()
    os.makedirs(args.cache_dir, exist_ok=True)

    for inst in universe:
        t0 = time.time()
        print(f"\n{inst.name}: GARCH(1,1)-t walk-forward "
              f"(window {args.window}, refit every {args.refit_every})",
              flush=True)

        r = rets[inst.name].dropna()
        g = regime.garch_path(r, window=args.window,
                              refit_every=args.refit_every)
        al = regime.alignment(inst.name, panel, rho, cfg)
        gf = regime.gates(bars[inst.name], g, al, cfg)

        gf.to_csv(path_for(inst.yahoo, args.interval, args.cache_dir))
        print(f"  {time.time() - t0:.0f}s")
        print(regime.describe(gf))

    print("\nNext:  python breakout_test.py --gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
