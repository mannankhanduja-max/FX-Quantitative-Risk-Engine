# The liquidity-fade cascade — current specification

Independent research project. **Not connected to, and not merged
into, any other repository.** Sole remote is
`mannankhanduja-max/FX-Quantitative-Risk-Engine`.

Reproduce the headline result with:

```bash
python mtf_test.py                       # the defaults below
python mtf_test.py --bias                # 1h VWAP/EMA filter, for comparison
python mtf_test.py --vwap-filter revert  # VWAP as a reversion gate
python mtf_test.py --commission-bp 0     # spread only
```

## The rule

| Timeframe | Job | Detail |
|---|---|---|
| 30 min | Setup | Breakout, liquidity sweep (fakeout), or retracement |
| 5 min | Trigger | First close beyond the previous 5m close, in the setup's direction |
| 30 min | Exit | Stop, 3R target, 80-bar limit |

- **Stop** — ATR(14) × 1.0 on 5m bars, floored at the trigger bar's
  own extreme. ATR rather than close-to-close sigma because a stop is
  hit by the intrabar extreme; ATR runs 1.46× the sigma here.
- **Reward** — 3:1, with an 80-bar limit so time exits stay at 0.1%.
  These are one decision, not two: a target too far for the limit
  turns the result into an artefact of where the limit fell.
- **Retracement confirmation** — the pullback must land in a fair
  value gap. Location, not just depth.
- **Direction** — both sides. No trend filter.
- **Sessions** — USD/JPY Tokyo + New York; EUR/USD and XAU/USD London
  + New York; NAS100 New York only. Cash hours in each centre's own
  clock.
- **Blackout** — no entries 16:55–18:00 ET. Measured spread at the
  rollover is 9.3× its daily median on USD/JPY.
- **Cost** — measured Dukascopy ask-minus-bid, plus 0.35bp per side
  commission.

## Result, 22 Sep 2024 – 22 Sep 2026

```
POOLED          8220 trades (4118/yr)      time exits 0.1%
win rate        28.19%   vs 25.00% no-information benchmark
z               +6.68
gross           +0.1284 R     cost 0.2086 R     net -0.0802 R
t               -4.03         Sharpe -2.85      maxDD -695 R
```

| | n | win | z | net R |
|---|---|---|---|---|
| EUR/USD | 2213 | 27.28% | +2.47 | −0.1424 |
| USD/JPY | 2519 | 28.23% | +3.74 | −0.1119 |
| XAU/USD | 1885 | 28.33% | +3.34 | −0.0666 |
| NAS100 | 1603 | 29.23% | +3.91 | **+0.0560** |

| setup | n | win | z | net R |
|---|---|---|---|---|
| breakout | 4960 | 27.88% | +4.58 | −0.0980 |
| fakeout | 2249 | 28.37% | +3.52 | −0.0721 |
| retrace + FVG | 1011 | 29.97% | +3.65 | −0.0110 |

## What this says

**The directional effect is real.** 28.19% against a 25.00%
barrier-touch benchmark at z +6.68, on 8,220 trades, with time exits
at 0.1% so the comparison is valid. Every instrument and every setup
beats the benchmark independently.

**It does not pay.** The edge is worth +0.128R per trade and the round
trip costs 0.209R. Breakeven needs cost at 62% of current. NAS100 is
the single positive cell, at +0.056R, because its spread is small
relative to how far it moves.

**And it did not replicate.** An earlier configuration of this same
family was frozen and run once on Sep 2021 – Aug 2023: z +3.35 became
z −0.31, with the realised win rate *below* the benchmark. See
`prior_period_mtf.py` and `results/prior_period_mtf_DONE.txt`. Every
number above is in-sample and should be read with that attached.

## What was tested and removed

| Component | Verdict |
|---|---|
| 9 EMA of session VWAP (trend) | Removed — cut ⅔ of trades, win rate rose without it |
| VWAP as reversion gate | Removed — cut ⅔ of trades, halved the gross edge |
| Order blocks | Removed — 33.33% against a 33.33% benchmark, z −0.00 |
| Breakeven stop at +10 pips | Removed — must save >rr losers per winner killed |
| GARCH / VaR / correlation gates | Removed — fitted noise; held-out worse than ungated |
| News-time blackout | Removed — measured spread at 08:30 ET is 0.8–1.0× median |
| 8:1 reward ratio | Removed — a time-exit artefact; barrier population lost |
| Fair value gaps | **Kept** — fewer trades, better gross and barrier mean |
| ATR stop basis | **Kept** — correctness; reproduces sigma×1.5 |
| Session windows | **Kept** — liquidity claim, statable in advance |

## Reproducibility

265 tests, including look-ahead tests that perturb future bars and
assert no earlier decision moves. Data is Dukascopy 5m bid and ask;
`fetch_intraday.py` downloads, everything after is offline.

**Backtest research only. Not investment advice. This strategy lost
money after realistic costs in every configuration tested, and failed
its pre-registered out-of-sample test.**
