"""
Alpaca paper-trading harness.

    export ALPACA_API_KEY_ID=...
    export ALPACA_API_SECRET_KEY=...

    python paper_trade.py                 # dry run: show intended orders
    python paper_trade.py --submit         # actually place them (paper only)
    python paper_trade.py --reconcile      # compare intent against fills

CREDENTIALS COME FROM THE ENVIRONMENT AND NOWHERE ELSE. There is no
key argument, no config entry and no default. A key committed to a
repository is a key that has to be rotated, and a key passed on the
command line lands in your shell history.

WHY THE SIGNAL DOES NOT COME FROM ALPACA
-----------------------------------------
Alpaca's free data tier serves IEX only - one exchange, a low
single-digit share of US consolidated volume. Every backtest number
in this repository rests on consolidated bars, and this strategy is
VOLUME-WEIGHTED, so computing its VWAP from IEX prints would be
measuring a small non-random slice of the tape. That is the same
error `fxrisk/indicators.py` refuses for FX spot, only harder to
notice because the numbers still look plausible.

So signals are computed from the local consolidated cache
(`data/yahoo/`, the same source the backtest used) and Alpaca is
used for execution only. If you upgrade to a SIP subscription,
`--feed sip` makes the data path consolidated too; the free tier
will reject it, which is the honest failure rather than a silent
downgrade to IEX.

WHAT THIS MEASURES THAT A BACKTEST CANNOT
------------------------------------------
One number: slippage. The backtest assumes 2bp per side, which is
an assumption, not a measurement. Every order here records the
decision price, the submitted price and the fill, so after a few
weeks `--reconcile` reports what execution actually cost. If it
comes in above 2bp, the backtest's cost model was optimistic and
the strategy results are worse than reported.

PAPER ONLY. The live endpoint is refused unless BOTH the
--i-understand-this-is-live flag and ALPACA_ALLOW_LIVE=yes are set,
and even then this script places no order it was not explicitly
asked to. Nothing here is a recommendation to trade.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config
from fxrisk import indicators
from fxrisk.data import yahoo

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

LOG_PATH = Path("results/paper_fills.jsonl")
TIMEOUT = 20


# ============================================================
# Credentials and session
# ============================================================

def credentials() -> dict[str, str]:
    """
    Read the key pair from the environment.

    Raises with an instruction rather than returning None, so a
    missing key can never be mistaken for an empty account.
    """
    key = os.environ.get("ALPACA_API_KEY_ID", "").strip()
    secret = os.environ.get("ALPACA_API_SECRET_KEY", "").strip()

    if not key or not secret:
        raise SystemExit(
            "Alpaca credentials are not set.\n\n"
            "  export ALPACA_API_KEY_ID=...\n"
            "  export ALPACA_API_SECRET_KEY=...\n\n"
            "Generate them at app.alpaca.markets under API Keys. Do not put "
            "them in a file inside this repository."
        )

    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def base_url(live: bool, allow_live: bool) -> str:
    """
    Paper unless BOTH the flag and the environment variable say live.

    Two independent switches, because one is too easy to leave on by
    accident in a shell script.
    """
    if not live:
        return PAPER_URL

    if os.environ.get("ALPACA_ALLOW_LIVE", "").lower() != "yes":
        raise SystemExit(
            "--i-understand-this-is-live was passed but ALPACA_ALLOW_LIVE is "
            "not set to 'yes'. Refusing to touch the live endpoint.\n\n"
            "This strategy's measured information coefficient is "
            "statistically zero (see README section 3). There is no edge here "
            "to trade with real money."
        )

    print("!! LIVE ENDPOINT. Real orders. Ctrl-C now if this is a mistake.",
          file=sys.stderr)
    return LIVE_URL


def get(url: str, headers: dict, **params) -> dict:
    r = requests.get(url, headers=headers, params=params or None, timeout=TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} from {url}: {r.text[:300]}")
    return r.json()


# ============================================================
# Signal - from the consolidated cache, not from Alpaca
# ============================================================

@dataclass
class Intent:
    """One instrument's target position and why."""

    symbol: str
    target: float          # -1, 0 or +1
    current: float
    delta: float
    decision_close: float  # the close the signal was computed from
    asof: str

    def as_order(self) -> dict | None:
        """Market order to move current -> target, or None if flat."""
        if abs(self.delta) < 1e-9:
            return None
        return {
            "symbol": self.symbol,
            "qty": str(abs(round(self.delta))),
            "side": "buy" if self.delta > 0 else "sell",
            "type": "market",
            "time_in_force": "day",
        }


def signals(unit_size: int) -> dict[str, tuple[float, float, str]]:
    """
    Target position per symbol, from the local consolidated cache.

    `indicators.signal` is already shifted one bar, so the value
    returned is knowable at the previous close. This function does
    not shift again.
    """
    out = {}
    for inst in config.UNIVERSE:
        bars = yahoo.load_symbol(inst.yahoo)
        sig = indicators.signal(
            bars,
            window=config.VWAP_WINDOW_DAILY,
            span=config.VWAP_EMA_SPAN,
        ).dropna()
        if sig.empty:
            continue
        last = float(sig.iloc[-1])

        # A short in a name the broker will not lend is not a
        # position, it is a rejected order. Clamp to flat rather
        # than submit something certain to fail - and note that the
        # backtest did NOT clamp, so its short leg in these names
        # was never executable. See config.Instrument.shortable.
        if last < 0 and not inst.shortable:
            last = 0.0

        out[inst.yahoo] = (
            last * unit_size,
            float(bars["Close"].loc[sig.index[-1]]),
            str(sig.index[-1].date()),
        )
    return out


def positions(url: str, headers: dict) -> dict[str, float]:
    held = get(f"{url}/v2/positions", headers)
    return {p["symbol"]: float(p["qty"]) for p in held}


def build_intents(url: str, headers: dict, unit_size: int) -> list[Intent]:
    held = positions(url, headers)
    targets = signals(unit_size)

    intents = []
    for sym, (target, close, asof) in sorted(targets.items()):
        current = held.get(sym, 0.0)
        intents.append(
            Intent(
                symbol=sym,
                target=target,
                current=current,
                delta=target - current,
                decision_close=close,
                asof=asof,
            )
        )
    return intents


# ============================================================
# Submission and the fill log
# ============================================================

def log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def submit(url: str, headers: dict, intents: list[Intent]) -> None:
    """Place the orders and record intent alongside the broker's reply."""
    for intent in intents:
        order = intent.as_order()
        if order is None:
            continue

        r = requests.post(
            f"{url}/v2/orders", headers=headers, json=order, timeout=TIMEOUT
        )
        ok = r.status_code < 400
        body = r.json() if ok else {"error": r.text[:300]}

        log({
            "event": "submit",
            "intent": asdict(intent),
            "order": order,
            "accepted": ok,
            "order_id": body.get("id"),
            "broker": body,
        })
        print(f"  {intent.symbol:5s} {order['side']:4s} {order['qty']:>5s}  "
              f"{'accepted' if ok else 'REJECTED'}")


def reconcile(url: str, headers: dict) -> None:
    """
    What execution actually cost, against the 2bp the backtest assumes.

    Slippage is measured from the decision close - the price the
    signal was computed on - to the fill. That is the honest
    reference: it includes the overnight gap between deciding and
    trading, which a backtest triggering on the close silently
    assumes away.
    """
    if not LOG_PATH.exists():
        raise SystemExit(f"no fill log at {LOG_PATH} - nothing submitted yet")

    rows = []
    for line in LOG_PATH.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("event") != "submit" or not rec.get("order_id"):
            continue

        order = get(f"{url}/v2/orders/{rec['order_id']}", headers)
        filled = order.get("filled_avg_price")
        if not filled:
            continue

        decision = rec["intent"]["decision_close"]
        fill = float(filled)
        side = 1 if rec["order"]["side"] == "buy" else -1
        # Positive slippage = worse than the decision price.
        slip_bp = side * (fill / decision - 1.0) * 10_000

        rows.append({
            "symbol": rec["intent"]["symbol"],
            "side": rec["order"]["side"],
            "decision": decision,
            "fill": fill,
            "slippage_bp": slip_bp,
        })

    if not rows:
        print("no filled orders yet")
        return

    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    mean = df["slippage_bp"].mean()
    print(f"\n  orders filled        {len(df)}")
    print(f"  mean slippage        {mean:+.2f} bp")
    print(f"  worst                {df['slippage_bp'].max():+.2f} bp")
    print(f"  backtest assumption  {config.COST_PER_TURN * 10_000:.2f} bp per side")

    if mean > config.COST_PER_TURN * 10_000:
        print("\n  Measured slippage EXCEEDS the backtest assumption, so the")
        print("  reported strategy results are optimistic. Raise")
        print("  config.COST_PER_TURN and re-run the backtest before drawing")
        print("  any conclusion from it.")


# ============================================================

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submit", action="store_true",
                   help="actually place orders (default is a dry run)")
    p.add_argument("--reconcile", action="store_true",
                   help="report measured slippage from the fill log")
    p.add_argument("--unit-size", type=int, default=10,
                   help="shares per unit of signal (default 10)")
    p.add_argument("--i-understand-this-is-live", action="store_true",
                   help="target the live endpoint; also needs ALPACA_ALLOW_LIVE=yes")
    args = p.parse_args()

    headers = credentials()
    url = base_url(args.i_understand_this_is_live, allow_live=True)

    acct = get(f"{url}/v2/account", headers)
    print(f"account {acct['account_number']}  status {acct['status']}  "
          f"equity {float(acct['equity']):,.2f}")
    print(f"endpoint {url}"
          f"{'   (PAPER)' if url == PAPER_URL else '   (LIVE)'}\n")

    if args.reconcile:
        reconcile(url, headers)
        return 0

    intents = build_intents(url, headers, args.unit_size)

    blocked = [i.yahoo for i in config.UNIVERSE if not i.shortable]
    if blocked:
        print(f"Not shortable at this broker, long-or-flat only: "
              f"{', '.join(blocked)}")
    print("Signals from the local consolidated cache "
          f"(as of {intents[0].asof if intents else 'n/a'}):")
    frame = pd.DataFrame([asdict(i) for i in intents])
    print(frame[["symbol", "decision_close", "current", "target", "delta"]]
          .to_string(index=False))

    actionable = [i for i in intents if i.as_order()]
    if not actionable:
        print("\nNothing to do - every position already matches its target.")
        return 0

    if not args.submit:
        print(f"\nDry run. {len(actionable)} order(s) would be placed. "
              "Re-run with --submit to place them.")
        return 0

    print(f"\nPlacing {len(actionable)} order(s):")
    submit(url, headers, actionable)
    print(f"\nLogged to {LOG_PATH}. Run --reconcile once they fill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
