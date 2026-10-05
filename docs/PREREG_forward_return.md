# Pre-registration: does the cascade predict drift, or was the edge geometry?

Written **before the test was run**. Criteria are fixed here and are
not to be edited afterwards. The runner refuses to re-run once
`results/forward_return_DONE.txt` exists.

## The question

Every positive number this study has produced is a **barrier**
statistic: the share of trades that touched a target before a stop,
compared to the driftless benchmark `b/(a+b)`. Beating that benchmark
is supposed to imply favourable drift. Two things can fake it:

1. **Time exits.** The benchmark describes barrier resolutions only.
   A population where many trades exit on the bar limit is not the
   population the benchmark describes. This already bit this study
   once: the 8:1 result was a time-exit artefact, with negative
   barrier populations and time exits averaging +2.8 to +3.6R.
2. **Path geometry.** An ATR-sized stop placed beyond a trigger bar's
   own extreme interacts with volatility clustering. Which barrier is
   touched first is a statement about the *path*, and a path effect
   can exist with zero drift.

So the gross edge that has stayed stubbornly positive across a 6.5x
change in stop width (+0.128, +0.090, +0.146, +0.143) may not be
drift at all. This test removes the barriers and asks the only
question underneath them:

> **After a setup fires, does price actually move in the setup's
> direction on average?**

No stop, no target, no reward ratio, no bar limit. Because the entry
population is identical for every reward ratio, this one test speaks
to the whole `rr` grid at once.

## What is measured

Population: the frozen rule's entries at the `1h / 4h / 1d` ladder —
all three setup kinds, no 1h bias filter, session windows per
instrument, rollover and index pre-open blackouts on, ATR(14) x 1.0.
The same population the walk-forward scored.

Statistic: signed forward log return from the entry price,

    y_h = side * ( log(close[k+h]) - log(close[k]) )

in basis points, at horizons **h = 1, 4, 24, 96 trigger bars**
(1 hour, 4 hours, 1 day, 4 days). The horizons are the cascade's own
frames — one trigger bar, one setup bar, one bias bar, four bias bars
— chosen so that no result can have influenced them.

Inference: overlapping forward returns sampled at event times are not
independent. Cell means are taken per `(instrument, calendar month)`
and the t-statistic is taken **across cells**, the same blocking
`fxrisk/research/ic.py` already uses. Cells with fewer than 3 events
are dropped. Holm-Bonferroni across the family of four horizons.

Secondary: rank IC of `stretch = |entry - stop| / ATR`, the trigger
bar's overextension, against `y_h`. Reversion predicts a larger
stretch produces a larger bounce. This is reported but is not the
criterion.

## Criteria

**FAIL — the edge was geometry, and this line of work closes.**
Pooled mean `y_h` is not distinguishable from zero at every horizon
(no horizon survives Holm at alpha = 0.05 with the pooled sign
positive).

**PASS — a real drift effect exists that the geometry was not
capturing.** At one or more horizons, pooled mean `y_h` > 0 with
clustered t > 2.0 surviving Holm, **and** the sign is positive in at
least 3 of the 4 instruments at that horizon.

**ECONOMIC PASS** — a PASS whose mean `y_h` also exceeds the measured
round trip (2 x half-spread + 2 x 0.35bp commission) at that horizon.

## What each outcome is worth, stated in advance

This matters more than the criteria, because the two outcomes are
**not symmetric** and it would be easy to over-read a PASS.

There is **no virgin data left**. 2021-2026 has been used for
development, for two frozen shots and for a seven-fold walk-forward.
So:

- A **FAIL is conclusive even in-sample.** It is a negative claim
  about the same data that produced the positive barrier statistic:
  if drift is absent here, the barrier edge cannot have been drift,
  and no amount of further parameter work can recover it. Nothing
  about selection bias rescues a null.
- A **PASS is weak.** The setups were chosen on this data, so a
  positive mean is exactly what selection produces. A PASS would
  license one thing only: a new hypothesis about *cost*, tested on
  data not yet collected. It would not license trading anything.

Recorded so that a PASS cannot later be read as more than it is.

BACKTEST-ONLY. Nothing here is a recommendation to trade.
