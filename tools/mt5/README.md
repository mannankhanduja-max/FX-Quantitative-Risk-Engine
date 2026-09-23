# MT5 spread export

`ExportSpread.mq5` writes one CSV per symbol containing OHLC, tick
volume and **the broker's own spread at each bar**, so the backtest can
be costed at what an actual MT5 account charges.

## Why it matters

Everything in this repository is currently costed from Dukascopy's
measured ask-minus-bid — an ECN aggregate, median **0.34bp per side**
on EUR/USD. An MT5 account charges one of two shapes, and neither is
that number:

| Account type | Spread | Commission |
|---|---|---|
| Raw / ECN | close to the ECN aggregate | added, typically ~0.35bp per side |
| Standard | marked up instead | none |

The difference decides results outright. Breakeven total cost per
side, measured on the current rules:

| Configuration | Breakeven cost, bp/side |
|---|---|
| 2:1, fades, no gold | 0.15 |
| 8:1, fades, no gold | 0.38 |
| 8:1, all setups, 4 instruments | 0.42 |
| 8:1, all setups, **NAS100 only** | **0.89** |

A raw MT5 account on the FX majors lands near 0.5–0.65bp per side once
commission is included, which is above every line in that table except
the last one. NAS100 is the only cell with real headroom — so it is the
one worth measuring properly rather than assuming.

## Running it

1. MetaEditor → open `ExportSpread.mq5` → **Compile** (F7)
2. In MT5, open a chart of the symbol
3. **Scroll the chart back past your start date** (press Home a few
   times). MT5 can only export bars it has already downloaded, and a
   short export is the easy mistake here — the script prints how many
   bars it actually got.
4. Drag **ExportSpread** from the Navigator onto the chart
5. Set the date range and timeframe, press OK
6. The CSV appears in `MQL5/Files/` — *File → Open Data Folder* finds it

Do this for each symbol you trade. Drop the CSVs into
`data/mt5/` and they can be wired into `fxrisk/risk/spread.py`
alongside the Dukascopy path.

## The one thing that can make the export useless

Some brokers do not record a per-bar spread on history, and write zero
instead. The script warns when more than 5% of bars report zero. If
that happens, this route is closed and the alternative is to record
live spread forward from now — which means waiting, not backtesting.
