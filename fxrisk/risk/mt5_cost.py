"""
MT5 account cost profiles, in the units brokers actually quote.

WHY THIS IS NOT JUST A NUMBER
-------------------------------
Everything in this repository is costed from Dukascopy's measured
ask-minus-bid. That is an ECN aggregate and is NOT what an MT5
account pays. Brokers quote in native units - pips, index points,
cents of gold - and charge in one of two shapes:

    RAW / ECN     a tight spread PLUS commission per side
    STANDARD      no commission, the spread marked up instead

Converting a native quote to basis points needs the instrument's
price level, so the conversion is done here against the ACTUAL
median close of the data being tested rather than a remembered
price. Gold at 3,733 makes a 12-cent spread 0.32bp; the same 12
cents at 1,800 would be 0.67bp. Hard-coding bp would silently
bake in whatever the price was the day the number was written.

THE VALUES BELOW ARE PUBLISHED TYPICALS, NOT A MEASUREMENT
------------------------------------------------------------
They are representative of advertised MT5 pricing and are good
enough to answer "does the shape of MT5 cost change the verdict".
They are NOT this account's spreads, and no conclusion here should
be stated as though they were. The real numbers come from
`tools/mt5/ExportSpread.mq5`, which dumps the terminal's own
per-bar spread; once those CSVs exist they replace this file.

Two things these typicals CANNOT capture, both of which make real
trading worse than anything computed here:

  1  Advertised spreads are averages over liquid hours. The
     rollover and the seconds around a release are multiples of
     them, and this rule set trades through both.
  2  Slippage. The spread is what you are quoted, not what you get
     on a market order into a thin book.

GOLD IS THE INSTRUMENT THIS CHANGES MOST
------------------------------------------
Dukascopy's measured gold spread is 1.62bp - about 60 cents at
3,733. Advertised MT5 raw gold is 10-20 cents, a quarter of that.
Gold was excluded from the rules on the strength of the Dukascopy
number, so if the MT5 figure is right for a given account that
exclusion may be wrong. That is a reason to MEASURE the account,
not a reason to assume the tighter number.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Quote:
    """A spread in the unit the broker quotes it in."""

    native: float          # pips for FX, index points, dollars for gold
    unit_price: float      # price move of one unit (0.0001 for a EUR/USD pip)

    def bp(self, price_level: float) -> float:
        """Basis points of the mid, at a given price level."""
        return self.native * self.unit_price / price_level * 10_000.0


# One "unit" per instrument, in price terms.
UNIT = {
    "EUR/USD": 0.0001,     # pip
    "USD/JPY": 0.01,       # pip
    "XAU/USD": 1.0,        # dollar  (a "10 cent" spread is native=0.10)
    "NAS100": 1.0,         # index point
}


@dataclass(frozen=True)
class Profile:
    """An account type: spread per instrument, plus commission."""

    name: str
    spread_native: dict          # instrument -> native spread
    commission_bp_per_side: float = 0.0
    note: str = ""

    def spread_bp(self, instrument: str, price_level: float) -> float:
        """Half-spread is charged per side by the barrier walker."""
        q = Quote(self.spread_native[instrument], UNIT[instrument])
        return q.bp(price_level)

    def cost_bp_per_side(self, instrument: str, price_level: float) -> float:
        return (self.spread_bp(instrument, price_level) / 2.0
                + self.commission_bp_per_side)


# ------------------------------------------------------------------
# Published typicals. Replace with a measured export when available.
# ------------------------------------------------------------------

RAW = Profile(
    name="MT5 raw / ECN",
    spread_native={"EUR/USD": 0.10, "USD/JPY": 0.20,
                   "XAU/USD": 0.12, "NAS100": 1.0},
    commission_bp_per_side=0.35,      # ~$3.5 per lot per side on FX
    note="tight spread plus commission; the commission dominates on FX",
)

STANDARD = Profile(
    name="MT5 standard",
    spread_native={"EUR/USD": 1.10, "USD/JPY": 1.20,
                   "XAU/USD": 0.30, "NAS100": 2.0},
    commission_bp_per_side=0.0,
    note="no commission; the markup is in the spread",
)

WIDE = Profile(
    name="MT5 standard, wide",
    spread_native={"EUR/USD": 1.60, "USD/JPY": 1.80,
                   "XAU/USD": 0.50, "NAS100": 3.0},
    commission_bp_per_side=0.0,
    note="a less competitive book, or the same book outside peak hours",
)

PROFILES = {"raw": RAW, "standard": STANDARD, "wide": WIDE}
