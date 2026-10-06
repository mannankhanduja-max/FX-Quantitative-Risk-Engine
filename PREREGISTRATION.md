# Pre-registration: FVG-confirmed breakout/retracement, 2022 -> now

Written and committed **before `run_fvg_test.py` reads a single bar of
2022+ data.** `run_fvg_test.py` refuses to run until this file no
longer contains `<<SIGN HERE>>` and is committed with no uncommitted
diff, and writes `results/fvg_test_DONE.txt` so it can only run once.
Nothing below may be edited after the test result exists — that is
the entire point of a pre-registration.

**This file is deliberately left as a template.** Fill in every
`<<...>>` placeholder yourself, in a text editor, reading your own
design-period result in `RESULTS.md` first. Nothing here should be
written by an automated session, for the same reason the news
calendar and stress windows in `fvg_config/` aren't: criteria set
after seeing the number they're meant to judge aren't criteria.

## The hypothesis, stated as narrowly as it deserves

<<State the one configuration you are carrying into the test, exactly
as it was run in `make run`. Name the rule (lookback, retrace
fraction, stop floor, trend filter on/off, R:R, breakeven, commission
assumption) and say why this is the configuration you are committing
to, not the best of several you tried. If you swept parameters during
the design period, say so, and say explicitly that the test below
will NOT re-sweep or cherry-pick on 2022+ data.>>

Design-period result (2015-01-01 -> 2022-01-01), for the record:

<<Paste the Overall row from your real (non-demo) `RESULTS.md`: trade
count, win rate, cost-adjusted hurdle, mean net R, t-stat.>>

## How 2022+ was already seen

<<This repository's earlier intraday work (§5a / `breakout_test.py`,
and anything else in this repo that already touched 2022 or later)
looked at some of this window before this study existed. Say plainly
what was already seen, by which script, and why you believe this
test is not simply re-running a configuration that was shaped, even
indirectly, by having looked at the answer. If you believe the prior
work and this rule are independent enough that this isn't a concern,
say why. If you can't make that case cleanly, say so — an honest "this
test is partially compromised and here's how" is worth more than a
pre-registration that quietly ignores the problem.>>

## Window

**2022-01-01 -> now.** Held out from every design-period run in this
study (`fxrisk.data.fvg_raw.TEST_START == DESIGN_END`). No parameter
is to be touched after this date is read.

## THE POWER ARITHMETIC, BEFORE THE CRITERIA

<<Before writing a pass/fail bar, work out whether it's achievable.
From your design-period trades: per-trade standard deviation of net
R, and the sample size the 2022+ window is likely to produce at this
setup's trade frequency. State what mean net R a test of that size
could actually distinguish from zero at t > 2, the way
`docs/PREREG_prior_period_cheap3.md` does. If a significance test on
some quantity is infeasible at the sample size you'll get, say so now
and say what you'll test instead (a point estimate against a
cost-breakeven hurdle, win rate vs. the barrier hurdle, direction
only, etc.) — never write in a bar you already know can't be cleared
and can't be honestly reported against.>>

## Success criteria

**PRIMARY.** <<State it as a single falsifiable statement: the
quantity, the threshold, and the test (t-stat, point estimate vs.
hurdle, etc.). Example shape: "mean net R > 0 with cell-clustered
t > 2.0" or "win rate clears the cost-adjusted hurdle with a positive
point estimate and the gap is not carried entirely by one subset."
Use the arithmetic above to make sure this is achievable.>>

**SECONDARY.** <<Anything reported as a sign only, not a test —
mirror the net-R-as-sign convention in the cheap3 prereg if your
primary can't carry a full significance claim either.>>

**ROBUSTNESS.** <<E.g.: result holds (or at least isn't reversed) in
the news-window subset and the stress-window subset, if those config
files exist; result isn't carried by a single calendar year; dropping
the largest cluster of trades doesn't flip the sign.>>

**PASS** requires <<list which numbered criteria, in full>>.
**FAIL** is anything else — define this precisely enough that there
is no reading under which a disappointing number becomes "almost a
pass."

## Optional primary variant

<<Optional. If you want to register a second, pre-specified variant
(different trend-filter setting, different R:R, with/without
breakeven) to be reported ALONGSIDE the primary rather than selected
from afterward, name it exactly here, with its own expected
flags/CLI arguments to `run_fvg_test.py`. If you don't want a second
variant, write "None — the primary configuration above is the only
configuration this test will report" and delete this placeholder
text.>>

## What each outcome is worth, fixed in advance

<<State plainly, before the number exists, what a PASS licenses and
what a FAIL means. Follow the house convention: a PASS on a backtest
licenses a conversation, never trading or capital on its own; a FAIL
on the last pre-registered configuration closes the line and should
not be followed by searching for a different cut of the same data.>>

Predicted before running, for the record: <<write your own prediction
and the reasoning behind it, the way the cheap3 prereg predicts a
FAIL before running. If you're wrong, say so when the result comes
back — that's the point of writing it down first.>>

BACKTEST-ONLY. Nothing here is a recommendation to trade.

---

<<SIGN HERE>> — replace this line with your name/handle and today's
date once every section above is filled in, then commit this file
before running `make run-test`.
