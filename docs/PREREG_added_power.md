# Pre-registration: two more crosses, to see whether the residual is real

Written **before GBP/JPY or AUD/NZD data existed on disk**. Criteria are
fixed here. Runner refuses to re-run once
`results/added_power_DONE.txt` exists.

## Why this test exists at all

The honest state of the study before this test:

- Every headline number was measured against `b/(a+b)` with the exit
  walked at 4h while entries fired at 1h. The null-path test showed that
  frame inflates the win rate by **+1.93pp on data containing nothing**.
- Re-measured at 1h, the frame where the null is closest to flat
  (−0.78pp), the real excess is **+0.97pp, z +1.15** — positive, not
  significant. Net R is −0.0244.
- Bias-corrected, the excess is roughly **+1.75pp**. The cost breakeven
  at 3:1 is about **+1.65pp** (1pp of excess ≈ 0.04R; cost ≈ 0.066R).

So the corrected estimate and the breakeven are the same number to
within noise, on 2,637 barrier resolutions. That is not evidence of no
edge. It is an underpowered measurement sitting exactly on the line,
and the only fix is more independent observations.

## What is being added, and the honest accounting of what it buys

| | why | independence |
|---|---|---|
| GBP/JPY | ~2x EUR/USD volatility on a spread that does not scale, so `2*spread/stop` falls | **Poor.** Shares the JPY leg with USD/JPY and the GBP complex correlates with EUR/USD. Adds trades faster than it adds information. |
| AUD/NZD | the only pair here with **no USD leg** | **Good.** The one instrument whose errors are close to independent of the rest. Also the one most likely to be killed by its own spread. |

Neither was chosen because anything about it looked promising. Nothing
about either has been measured. They were chosen for volatility scale
and for independence, which are properties of the instruments, not of
any result.

**Effective sample, not trade count, is the criterion.** Six correlated
instruments are not six times one instrument. The pooled statistic is
therefore taken across `(instrument, calendar month)` cells with the
t-statistic across cells, as in the forward-return test, so that
correlated instruments inside the same month cannot each count as
independent evidence.

## The test

Frame: **1h monitoring only.** The 4h frame is known-biased and is
reported for continuity, never for inference. Entries, stops, targets,
sessions, blackouts and the 3:1 ratio are unchanged from the frozen
rule. No parameters are searched. Nothing is optimised.

Sample: 2021-09 to 2026-09, all six instruments, bid and ask.

Primary statistic: excess win rate over `b/(a+b)` at the 1h walk,
**minus the null-path bias at the same frame (−0.78pp)**, pooled by
cells. Secondary: mean net R per trade after measured spread + 0.35bp
per side, and signed forward return at 1, 4, 24 and 96 bars for the two
new instruments, since drift was absent in the first four.

## Criteria

**PASS** — bias-corrected excess is positive with cell-clustered
t > 2.0, **and** mean net R over the six instruments is positive, **and**
the result does not depend on a single instrument (dropping any one
instrument leaves the corrected excess positive).

**FAIL** — anything else. In particular a positive excess that does not
clear cost, or one that collapses when GBP/JPY is dropped, is a FAIL.

## Stated in advance, because this is the part that can be abused

The prior test pre-registration recorded that a PASS on already-used
data is weak. **The two new instruments are genuinely unused**, so a
PASS here is worth more than the last one would have been — but the
*rule* was still designed on the original four, and its session
windows, lookback and trigger were chosen there. So:

- **FAIL closes the strategy for good.** Underpowered was the last
  remaining defence, and this test removes it by adding the power.
- **PASS licenses one thing:** a fresh frozen shot on data from a
  different source, at a lower cost structure. Not trading.
- **A PASS carried by GBP/JPY alone is a FAIL**, because GBP/JPY is the
  instrument most correlated with what came before, and "the new
  instrument that resembles the old ones agreed with them" is the
  result selection would produce.

Predicted before running, for the record: AUD/NZD fails on cost — its
spread is wide relative to its volatility, so `2*spread/stop` should be
the worst in the universe despite the wider nominal stop. If AUD/NZD
comes out as the strongest instrument, that is a reason to suspect the
measurement, not to celebrate.

BACKTEST-ONLY. Nothing here is a recommendation to trade.
