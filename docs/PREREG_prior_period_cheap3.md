# Pre-registration: the cost-filtered rule on history it has never seen

Written **before any pre-2021 5m data existed on disk**, and committed
with the runner that executes it, so neither the criteria nor the code
can be adjusted after seeing a number. The runner refuses to re-run once
`results/prior_cheap3_DONE.txt` exists.

## The hypothesis, stated as narrowly as it deserves

Exactly one configuration survives this study with an ex-ante story:

> The frozen cascade, monitored at **1h** (the only frame the null-path
> test found unbiased), on the **three instruments with the lowest
> measured `2*spread/stop`** — NAS100 0.042, XAU/USD 0.067, USD/JPY
> 0.072.

The cost filter is legitimate because `cost_R` is built from spread and
ATR, with no outcome in it. Nothing else about this configuration is.

**The universe is frozen as the literal list `NAS100, XAU/USD,
USD/JPY`.** `cost_R` will be re-measured on the test window and reported,
but the list will NOT be re-ranked on it. Re-ranking inside the test
window is selection inside the test.

In-sample (2021-09 → 2026-09, n = 1951): corrected excess **+2.52%**
(t +2.25), net R **+0.0100** (t +0.25).

## Window

**2016-09-01 → 2021-08-30.** Five years, ending exactly where the
existing caches begin, so there is no overlap with anything this study
has touched. Instruments with fewer than **3 years** of usable history in
that window are excluded, and the exclusion is printed in the headline
rather than quietly changing the universe.

Everything else is the frozen rule: 1h/4h/1d ladder, 3:1, ATR(14) x 1.0,
all three setup kinds, no 1h bias filter, session windows, rollover and
index pre-open blackouts, measured spread + 0.35bp per side. No
parameter is searched. No grid exists.

## THE POWER ARITHMETIC, BEFORE THE CRITERIA

This is the most important section, and it constrains what may honestly
be asked of the test.

Net R has a per-trade standard deviation of **1.77R** (from se 0.0400 at
n = 1951). For a mean of +0.0100R to reach t = 2:

    n = (2 x 1.77 / 0.0100)^2 = 124,864 trades ~= 320 years

**So a significance test on net R is unachievable and will not be used
as a criterion.** Writing one in would be a rigged test — a bar nothing
could ever clear, which then gets quietly relaxed when it is missed.
Recorded here so it cannot be.

The excess is measurable: se **1.12pp** at n = 1951. That is what the
test rests on.

Cost breakeven for these three: mean `cost_R` 0.060, and 1pp of excess is
worth ~0.04R, so the rule needs about **1.50pp** of corrected excess to
pay for itself.

## Criteria

**PRIMARY — the excess replicates and clears its own cost.**
Corrected excess (raw excess minus the −0.78pp null-path bias at the 1h
frame) must be:
  1. positive with cell-clustered **t > 2.0**, and
  2. a point estimate of **≥ 1.50pp** — the cost breakeven.

**SECONDARY — direction only, no significance claimed.**
Net R point estimate **> 0**. Reported as a sign, never as a test, for
the reason above.

**ROBUSTNESS.**
  3. Corrected excess positive in **at least 2 of 3** instruments.
  4. Dropping any one instrument leaves the pooled corrected excess
     **≥ 1.00pp**.

**PASS** requires 1, 2, 3 and 4. **FAIL** is anything else.

## What each outcome is worth, fixed in advance

**A PASS does not mean this is tradeable, and must not be read that
way.** The best in-sample estimate is +0.0100R per trade. At ~390 trades
a year that is roughly +3.9R annually before any slippage beyond the
measured spread, on a quantity whose confidence interval spans −0.07 to
+0.09R. A PASS would mean *the reversion is real and the cost filter is
the right way to isolate it*, and it would license exactly one
conversation: whether a lower cost structure than retail makes it
matter. It licenses no trading and no capital.

**A FAIL closes the line permanently.** This is the last configuration
with an ex-ante rationale, tested on the last unused data, at the only
unbiased frame. There is no further cut, instrument, frame or ratio
behind it, and a FAIL should not be followed by looking for one.

**Anything in between is a FAIL.** In particular: a positive excess
below 1.50pp is a FAIL, because it does not pay. A positive excess
carried by one instrument is a FAIL. A PASS on the excess with negative
net R is a FAIL on criterion 2's companion and will be reported as a
FAIL.

Predicted before running, for the record: I expect a FAIL. The
in-sample +2.52% ranked 2nd of 20 subsets and 15% of those subsets clear
t > 2 with no edge present, so the honest prior is that most of the
+2.52% is subset selection and the replication lands nearer +1pp. If it
comes back above +2.5pp I will say so and the prediction was wrong.

BACKTEST-ONLY. Nothing here is a recommendation to trade.
