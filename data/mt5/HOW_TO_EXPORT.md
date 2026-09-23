# Getting your broker's real spread out of MT5

Two routes. **Route A is better** — smaller files, two years of
history, and it gives exactly the number the backtest needs.

Save everything into **this folder**
(`~/fx-risk-engine/data/mt5/`). It is connected to the Claude
session, so anything dropped here can be picked up and used
immediately.

---

## Route A — the script (recommended)

Gives per-bar spread for the full two years, ~10MB per symbol.

1. **MT5 → File → Open Data Folder.** Finder opens.
2. Go into `MQL5` → `Scripts`.
3. Copy `~/fx-risk-engine/tools/mt5/ExportSpread.mq5` into that
   folder.
4. Back in MT5: **Tools → MetaQuotes Language Editor** (F4).
   Find `ExportSpread` in the Navigator, open it, press **F7** to
   compile. It should say `0 errors`.
5. In MT5, open a chart of the symbol (**XAUUSD** first).
6. **Scroll the chart all the way back past September 2024** —
   press `Home` a few times, or hold the left arrow. *MT5 can only
   export bars it has already downloaded.* This is the step people
   skip, and it silently produces a short file.
7. In the Navigator panel, expand **Scripts**, drag
   **ExportSpread** onto the chart.
8. In the Inputs box set the range (defaults are already
   2024.09.22 → 2026.09.22, M5). Press OK.
9. The Experts tab at the bottom prints how many bars it got and
   the median spread. **Check that number looks sane** before
   moving on.
10. The CSV is in the data folder under `MQL5/Files/`. Copy it
    into `~/fx-risk-engine/data/mt5/`.

Repeat for **NAS100** (or whatever your broker calls it — US100,
NAS100.cash, USTEC…). Those two are where the answer hinges; the
FX majors are already clearly negative at any MT5 pricing.

---

## Route B — tick export (no compiling)

Simpler but the files are large and brokers usually keep far less
tick history than bar history.

1. **MT5 → View → Symbols** (Ctrl+U)
2. Find the symbol in the tree, select it
3. Click the **Ticks** tab
4. Set the date range, press **Request**
5. Press **Export Ticks**, save into
   `~/fx-risk-engine/data/mt5/`

Ticks carry bid and ask directly, so spread is exact rather than
sampled at the bar close.

---

## What can go wrong

**Zero spread on every bar.** Some brokers do not record spread
on history. The script warns if more than 5% of bars come back
zero. If that happens both routes are closed and the only honest
option is recording spread forward from today.

**A short file.** You did not scroll back far enough (Route A) or
the broker does not keep that much tick history (Route B). The
script prints its actual first and last bar — check them.

**A different symbol name.** Brokers rename things. Export
whatever your terminal actually calls gold and the Nasdaq; the
loader keys on the filename, and I will map it.

---

## Then

Say the word and the CSVs get wired into `fxrisk/risk/spread.py`
alongside the Dukascopy path, and every result is recomputed at
your account's real cost.
