# TradingView

[`breakout_retrace.pine`](breakout_retrace.pine) is a Pine Script v6
indicator that reproduces the §5a breakout/retracement rule
(`fxrisk/strategies/breakout_retrace.py` and the exit walk in
`fxrisk/risk/barriers.py`) on a live chart, and plots a **BUY** or
**SELL** label the instant a trade is confirmed — i.e. on gate 4, the
first bar that closes back beyond the breakout bar's own extreme after
a qualifying retracement.

**BACKTEST-ONLY. Not a recommendation to trade.** The whole reason
this script exists is to let the same rule the README reports on be
inspected on a chart; it does not change what that backtest found.
See the file's header comment and [README.md §5a](../README.md#5a-intraday-breakout--retracement-21-breakeven-stop)
for the numbers: at a 1-sigma stop the rule's barrier win rate was
23.0% against a 42.2% cost-adjusted hurdle (t = -5.02), and the
held-out risk-gated run did worse than not gating at all.

## What it plots

| Marker | Fires when |
|---|---|
| `BUY` / `SELL` (large label) | Gate 4: the retracement confirms and the trade is taken. This is the signal, and it is also what the alerts fire on. |
| `TP` / `SL` / `BE` / `T/O` (small label) | The open trade exits at target, stop, a breakeven scratch, or the time-stop. Toggle with **Mark exits**. |
| `BO` / `armed` (tiny label, off by default) | Gate 2 (a breakout closed) and gate 3 (the retracement reached the arming band). Toggle with **Mark the breakout bar and the armed bar too** — useful for seeing *why* a signal did or didn't fire. |
| Dashed / dotted lines | Entry, stop and target while a trade is open. Toggle with **Draw stop / target / entry**. |
| Blue / orange lines | Session VWAP and its 9-period EMA — the gate-1 trend filter. |

Two alerts are defined (`alertcondition`) so a TradingView alert can be
set on a confirmed BUY or SELL without polling the chart.

## Using it

1. Pine Editor → New indicator → paste `breakout_retrace.pine` → Add to chart.
2. Pick a chart that reports real or tick volume — the rule's VWAP is
   undefined without it, the same condition `session_vwap()` raises on
   in Python. The script shows a red warning label if the feed's
   volume is zero.
3. Set **Breakeven trigger (bp of entry price)** to the instrument's
   figure from `config.py`: EUR/USD 9.3, XAU/USD 4.2, NAS100 5.0,
   USD/JPY 6.7. It defaults to EUR/USD's.
4. The rule was tested on 15-minute bars; it will run on any timeframe
   you apply it to, but only the 15-minute result is the one the
   README reports on.

## What was ported, and what was simplified for a live chart

Every gate (trend filter, close-based breakout, retracement band,
confirmation close, stop floor, breakeven arming, pessimistic same-bar
tie-break) is the same arithmetic as the Python, parameter for
parameter. The one difference: `find_setups()` is a vectorised scan
that can examine a new breakout on any bar, even while a previous
retracement is still being watched, and only drops overlapping
*entries* afterwards in `walk_explicit()`. A chart indicator runs
forward one bar at a time, so this script instead keeps at most one
setup active — watching, or in a trade — and looks for the next
breakout only once the current one has resolved. The script's header
comment goes into this, and the one related one-bar edge case, in
full.

## Not run against a Pine compiler

There is no TradingView API or Pine compiler available in the
environment this was written in, so this file has been reviewed
carefully line by line — parameter names, built-in signatures, paren
and indentation balance — but not actually compiled. Paste it into the
Pine Editor first; it will report line and column for anything missed.
