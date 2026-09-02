# FX Quantitative Risk Engine

Volatility modelling, Value at Risk with formal backtesting, Monte
Carlo simulation and stress testing against dated historical
episodes — run on **4,918 trading days of real market data**,
February 2007 to September 2026.

> **Backtest-only.** Every number this repository produces is
> computed on historical data. None of it is a prediction, a
> recommendation, or investment advice. See
> [§10 Backtest-only results](#10-backtest-only-results) before
> quoting anything from here.

---

## 1. What this is

Most published portfolio projects estimate risk from a sample
covariance matrix and stop. That has two problems, and this
repository is organised around fixing them.

**A sample covariance matrix has no memory of when things
happened.** It weights a return from five years ago exactly as
heavily as yesterday's. Volatility clusters and correlations move,
so the equal-weighted estimate describes an average of regimes
rather than the one you are in.

**Nobody checks whether the risk number was right.** A VaR
estimate is a falsifiable claim: at 99%, losses should exceed it on
1% of days, and those breaches should be scattered rather than
bunched. That is testable, and untested VaR is decoration.

So:

| Instead of | This uses |
|---|---|
| Sample covariance | EWMA (RiskMetrics, λ = 0.94) and DCC-GARCH |
| Constant volatility | GARCH(1,1) with Student-t innovations |
| Correlation ignored in VaR | DCC covariance contracted with weights |
| Unverified VaR | Kupiec, Christoffersen, Basel traffic light, Lopez loss |
| No tail analysis | Expected Shortfall alongside every VaR |
| Daily bars, no volume | ETF bars with real volume: 20-day VWAP, 9-period EMA |
| sqrt-time for multi-day risk | Monte Carlo through the variance recursion |
| "What if markets fall 10%" | Dated replays: Lehman, SNB, CNY, Brexit, COVID, LDI |

---

## 2. Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# One download. Everything after it is offline and reproducible.
python fetch_data.py --start 2006-01-01

python run_risk_report.py --source yahoo
python docs/make_figures.py --source yahoo

pytest tests/ -q
```

`fetch_data.py` is the only step that touches the network. It writes
one CSV per instrument into `data/yahoo/`, and every later run reads
that cache — so a vendor revision or a rate limit cannot silently
change a backtest you already ran.

`python run_risk_report.py --demo` runs the whole pipeline on
simulated data if you want to check the machinery without
downloading anything. Its output is labelled `DEMO_SIMULATED_`.

**The figures in this README are real.** 4,918 trading days,
February 2007 to September 2026.


### What it finds

At 99% over the full sample, the estimators separate cleanly:

| method | breaches (exp. 46) | Kupiec p | Christoffersen p | |
|---|---|---|---|---|
| `dcc_portfolio` | 49 | 0.57 | 0.56 | **pass** |
| `garch_t` | 49 | 0.74 | 0.54 | **pass** |
| `historical` | 51 | 0.31 | 0.003 | fail |
| `parametric_normal` | 60 | 0.02 | 0.001 | fail |
| `ewma_normal` | 76 | 0.0003 | 0.49 | fail |

`ewma_normal` is the instructive one. It **passes** the independence
test and fails coverage badly — 76 breaches against 48.9 expected. It
gets the volatility right and the tail shape wrong, which is exactly
what a Gaussian quantile on fat-tailed returns should do. `historical`
does the reverse: roughly the right count, clustered in the wrong
places.

Monte Carlo at 99%, 20,000 paths: 1-day VaR **1.41%**, 10-day
**4.33%**. Square-root-of-time scaling would say 4.46% — a ratio of
**0.971**, so the iid assumption overstates ten-day risk here.

---

<figure>
<img src="docs/diagrams/01-pipeline.svg" alt="End-to-end pipeline: HistData tick files become daily returns cut at the New York close, feed three volatility models, and drive VaR, backtests, correlation and stress replay." width="100%">
<figcaption><sub>End-to-end pipeline: HistData tick files become daily returns cut at the New York close, feed three volatility models, and drive VaR, backtests, correlation and stress replay.</sub></figcaption>
</figure>

---

## 3. What the models do

### EWMA — `fxrisk/models/ewma.py`

    sigma2_t = lambda * sigma2_{t-1} + (1 - lambda) * r_{t-1}^2

λ = 0.94 is the RiskMetrics daily default, a ~17-day centre of mass.
Provides variance, full covariance paths, correlation, and a
portfolio variance path that avoids materialising every matrix.

<figure>
<img src="docs/figures/01-volatility.png" alt="Three volatility estimates on the equal-weight ETF portfolio, 2007-2026. The rolling 252-day standard deviation steps when a shock enters and leaves the window; EWMA and GARCH respond on the day." width="100%">
<figcaption><sub>Three volatility estimates on the equal-weight ETF portfolio, 2007-2026. The rolling 252-day standard deviation steps when a shock enters and leaves the window; EWMA and GARCH respond on the day.</sub></figcaption>
</figure>


### GARCH(1,1) — `fxrisk/models/garch.py`

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Maximum likelihood via SLSQP, Gaussian or Student-t innovations,
with stationarity enforced as a constraint. Written directly rather
than wrapping the `arch` package so the likelihood, constraints and
forecast recursion are visible and testable — the test suite
recovers known parameters from simulated data.

Student-t is the default. Gaussian GARCH systematically
understates tail risk, which is the one thing a VaR engine must not
do.

`rolling_garch_variance` produces walk-forward one-step-ahead
forecasts: at each date the model has only seen returns strictly
before it. Refits periodically and recurses daily in between, which
is what a desk actually does.

<figure>
<img src="docs/diagrams/02-walk-forward.svg" alt="The forecast for day t may only use returns before t. The leak that was fixed: the Student-t degrees of freedom were fitted on the whole sample, so the scale was honest and the tail shape was not." width="100%">
<figcaption><sub>The forecast for day t may only use returns before t. The leak that was fixed: the Student-t degrees of freedom were fitted on the whole sample, so the scale was honest and the tail shape was not.</sub></figcaption>
</figure>


### DCC-GARCH — `fxrisk/models/dcc.py`

Engle's two-stage estimator. Univariate GARCH per asset, then

    Q_t = (1 - a - b) Qbar + a z_{t-1} z_{t-1}' + b Q_{t-1}

normalised to a correlation matrix. This separates correlation
dynamics from volatility dynamics, which EWMA cannot do — EWMA
forces both to share one decay factor.

Correlations rise in crises. A static matrix estimated over a calm
decade understates portfolio risk in exactly the state where the
number carries weight.

`rolling_dcc_covariance` is the walk-forward version, and it is the
one the VaR path uses. `fit_dcc` sees the whole panel, so its
correlation path cannot be used for a backtest without leaking the
future into every historical date. Refits are semi-annual by
default rather than monthly — each fit runs one MLE per asset plus
a likelihood optimisation over the window, so daily refitting is
not tractable.

Stated limits: two-stage DCC is consistent but not efficient,
stage-2 standard errors ignore stage-1 estimation error, and the
scalar (a, b) form makes every pair share the same correlation
dynamics.

### VaR and ES — `fxrisk/risk/var.py`

Six estimators, because the disagreement between them is
informative: `historical`, `parametric_normal`, `ewma`, `garch_t`,
`cornish_fisher`, and `dcc_portfolio`. Plus portfolio VaR from any
covariance matrix and a component-VaR decomposition that answers
where the risk actually sits.

`dcc_var_series` is what makes DCC load-bearing rather than
decorative. The other estimators model the volatility of the
*portfolio return series*, so a change in correlation between
constituents only reaches the risk number after it has already
shown up in realised portfolio volatility. This one forecasts the
covariance matrix asset by asset and contracts it with the weights,

    sigma2_p,t = w' H_t w

so a correlation regime shift moves VaR on the day the model
detects it rather than after the portfolio has lived through it.

<figure>
<img src="docs/diagrams/03-dcc-to-var.svg" alt="Collapsing to a portfolio return series destroys the correlation information before it reaches the risk number. Contracting the DCC covariance with the weights preserves it." width="100%">
<figcaption><sub>Collapsing to a portfolio return series destroys the correlation information before it reaches the risk number. Contracting the DCC covariance with the weights preserves it.</sub></figcaption>
</figure>


Expected Shortfall accompanies every VaR — Basel's FRTB replaced
99% VaR with 97.5% ES as the capital measure, because VaR says
nothing about how bad the tail gets once you are in it.

The Gaussian estimator is included specifically as a baseline that
should fail. Demonstrating that is the point.

<figure>
<img src="docs/figures/02-var-breaches.png" alt="Realised daily returns against the 99% GARCH-t VaR line over 4,918 trading days. The line widens through 2008, the January 2015 SNB break, March 2020 and 2022, and the 49 breaches are scattered rather than bunched." width="100%">
<figcaption><sub>Realised daily returns against the 99% GARCH-t VaR line over 4,918 trading days. The line widens through 2008, the January 2015 SNB break, March 2020 and 2022, and the 49 breaches are scattered rather than bunched.</sub></figcaption>
</figure>


### Backtesting — `fxrisk/risk/backtesting.py`

- **Kupiec (1995)** unconditional coverage — right *number* of breaches?
- **Christoffersen (1998)** independence — are breaches *scattered* or clustered?
- **Conditional coverage** — joint test
- **Basel traffic light** — the supervisory rule, 99% only
- **Lopez loss** — magnitude-aware score

The independence test is what separates a serious engine from a
toy. Historical VaR typically passes Kupiec and fails
Christoffersen: it has the right average and is wrong at the moments
that matter. That failure is the entire argument for GARCH.

<figure>
<img src="docs/diagrams/04-backtests.svg" alt="Two models, ten breaches each in 250 days. Kupiec cannot tell them apart; Christoffersen fails the one whose breaches arrive in a single volatile fortnight." width="100%">
<figcaption><sub>Two models, ten breaches each in 250 days. Kupiec cannot tell them apart; Christoffersen fails the one whose breaches arrive in a single volatile fortnight.</sub></figcaption>
</figure>


Two properties documented in the code because they change how you
read the output:

*The independence test is weak when breaches are sparse.* At 99%
over 3,000 days you get ~30 breaches, and simulation shows the test
rejects a deliberately misspecified flat VaR less than half the
time. At 95% (~150 breaches) power rises above two thirds. A pass at
99% alone is weak evidence — the runner backtests at 95%, 97.5% and
99% for this reason.

*Basel zones apply only at 99%.* A correct 95% model breaches ~12
times per 250 days and would be scored "red" for working properly.
Other confidence levels return `n/a` rather than a misleading
colour.

<figure>
<img src="docs/figures/05-backtest-rates.png" alt="Breach rate per estimator at 99% against the 1% target, on real data. The two conditional fat-tailed models land near the target; the historical and Gaussian baselines do not." width="100%">
<figcaption><sub>Breach rate per estimator at 99% against the 1% target, on real data. The two conditional fat-tailed models land near the target; the historical and Gaussian baselines do not.</sub></figcaption>
</figure>


### Pairwise correlation — `fxrisk/risk/correlation.py`

`pairwise_table` puts three estimates of every instrument pair side
by side: the full-sample Pearson correlation, the EWMA value at the
end of the sample, and the DCC path's last / mean / min / max. The
column that matters is `dcc_range` — max minus min — because that is
precisely what the single sample number averages away.

<figure>
<img src="docs/figures/03-correlation.png" alt="All ten pairwise DCC correlations against their unconditional levels (dotted). Euro/Swiss franc sits near 0.72 and collapses toward zero on the day the SNB removed the floor." width="100%">
<figcaption><sub>All ten pairwise DCC correlations against their unconditional levels (dotted). Euro/Swiss franc sits near 0.72 and collapses toward zero on the day the SNB removed the floor.</sub></figcaption>
</figure>

`correlation_stress` splits the sample on portfolio-average return
and reports each pair's mean conditional correlation on the worst
5% of days against all others. A positive `increase` means
diversification weakens exactly when it is needed, which is the
empirical case for using a conditional correlation model at all.

<figure>
<img src="docs/diagrams/05-correlation-regimes.svg" alt="One sample correlation of 0.548 stands in for a DCC path running 0.223 to 0.810 — and every pair tightens further on the worst 5% of days." width="100%">
<figcaption><sub>One sample correlation of 0.548 stands in for a DCC path running 0.223 to 0.810 — and every pair tightens further on the worst 5% of days.</sub></figcaption>
</figure>


### Risk-adjusted performance — `fxrisk/risk/performance.py`

Sharpe, Sortino, Calmar, drawdown and rolling Sharpe, plus a
per-instrument table. Two things it is deliberately explicit about:

*The risk-free rate is annual and is de-annualised before use.*
Subtracting an annual 4% from a daily return is a factor-252 error
and a common one. Every function here takes the annual rate and
converts it geometrically.

*Sortino measures downside against the minimum acceptable return,*
not against zero. Using zero is common and wrong whenever the
risk-free rate is non-zero, and it makes Sortino non-comparable
with the Sharpe printed beside it.

`sharpe_ratio` also returns the un-annualised value, so the
sqrt-time assumption behind annualisation stays visible rather than
buried.

<figure>
<img src="docs/figures/04-rolling-sharpe.png" alt="Rolling 126-day Sharpe against the full-sample value, real ETF data." width="100%">
<figcaption><sub>Rolling 126-day Sharpe against the full-sample value, real ETF data.</sub></figcaption>
</figure>


<figure>
<img src="docs/diagrams/07-sharpe.svg" alt="An annual risk-free rate subtracted from daily returns is a factor-252 error; and a zero downside threshold misses returns that are positive but below the risk-free rate." width="100%">
<figcaption><sub>An annual risk-free rate subtracted from daily returns is a factor-252 error; and a zero downside threshold misses returns that are positive but below the risk-free rate.</sub></figcaption>
</figure>


### Stress testing — `fxrisk/risk/stress.py`

Dated windows replayed against current weights:

| Key | Episode | Window |
|---|---|---|
| `gfc_2008` | Lehman | 2008-09-01 → 2008-12-31 |
| `gfc_full` | Full GFC drawdown | 2007-10-09 → 2009-03-09 |
| `snb_2015` | SNB abandons the EUR/CHF floor | 2015-01-12 → 2015-01-30 |
| `cny_2015` | CNY devaluation, August selloff | 2015-08-10 → 2015-09-30 |
| `brexit_2016` | Referendum | 2016-06-20 → 2016-07-15 |
| `covid_2020` | COVID crash | 2020-02-19 → 2020-03-23 |
| `gilt_2022` | UK gilt / LDI crisis | 2022-09-21 → 2022-10-17 |

`snb_2015` is the one that matters most for FX. The Swiss National
Bank abandoned its 1.20 floor without warning on 15 January 2015 and
EUR/CHF fell ~30% intraday. Years of near-zero measured volatility
followed by a move no VaR model calibrated on that history could
have anticipated — the cleanest available demonstration of what
these models cannot do.

Scenarios outside your data window are **skipped and reported as
skipped**. Assets missing from a window are reported with a coverage
percentage rather than silently zero-filled: a stress result
computed over half the book is worse than none, because it looks
complete.

`stress_vs_var` expresses each scenario's worst day as a multiple of
current VaR — the framing that lands in a risk meeting.

---

## 4. Real data

**The default path is Yahoo Finance.** `python fetch_data.py`
downloads daily bars for the five ETFs in §5 and writes them to
`data/yahoo/`. That is the only step that touches the network; every
report afterwards reads the cache, so results are reproducible and
the figures in this README were generated from it.

### The HistData path

Retained for spot FX, and the reason `fxrisk/data/histdata.py`
exists. It is exercised only when `config.UNIVERSE` is switched to
one of the spot books.

[HistData.com](https://www.histdata.com/download-free-forex-data/)
provides free tick and 1-minute FX history. Download the ASCII
archives, unzip one folder per pair:

```
data/histdata/EURUSD/DAT_ASCII_EURUSD_M1_2015.csv
data/histdata/EURUSD/DAT_ASCII_EURUSD_T_201501.csv
```

then

```bash
python run_risk_report.py --data-dir data/histdata
```

### The volume problem, stated plainly

**The Volume column in every HistData file is zero.** Spot FX trades
over the counter with no consolidated tape, so there is nothing to
report; the column exists for format compatibility.

This matters because a volume-weighted average price needs volume.
Two substitutes, both implemented:

- **Tick count** — quote updates per bar. Quote intensity tracks
  activity closely in FX, so a tick-count-weighted average price is
  a genuine VWAP analogue. Requires tick files. This is the
  supported path.
- **Time weighted** — with only M1 bars every bar weighs the same,
  which makes the result a TWAP.

`rolling_vwap` will not fabricate volume. Given M1 bars it returns a
TWAP, names the series `twap`, and emits a `RuntimeWarning`. A TWAP
is a legitimate benchmark; calling it a VWAP is not.

<figure>
<img src="docs/diagrams/06-vwap-volume.svg" alt="Every HistData Volume field is zero, so a volume filter silently discards every row. Tick counts restore a real VWAP; M1 bars fall back to a TWAP that is labelled as one." width="100%">
<figcaption><sub>Every HistData Volume field is zero, so a volume filter silently discards every row. Tick counts restore a real VWAP; M1 bars fall back to a TWAP that is labelled as one.</sub></figcaption>
</figure>


### The session boundary

FX has no exchange close, so the daily boundary is a choice.
`to_daily` cuts at 17:00 New York — the market convention — rather
than letting the default land at UTC midnight, in the middle of the
Asian session. This is the difference between a daily return series
that means something and one that does not.

---

## 5. The instrument universe

Defined once, in `config.py`, and read by both pipelines — so the
two halves of this repository describe the same book.

| Instrument | Symbol | Exposure | History from |
|---|---|---|---|
| Euro | `FXE` | EUR/USD | 2005-12 |
| Pound | `FXB` | GBP/USD | 2006-06 |
| Yen | `FXY` | JPY/USD | 2007-02 |
| Swiss franc | `FXF` | CHF/USD | 2006-06 |
| Gold | `GLD` | XAU/USD | 2004-11 |

These are exchange-traded currency and metal funds, not spot FX.
Both pipelines read `config.UNIVERSE`; neither holds its own list,
and a test fails if they drift apart. `fetch_data.py` is the only
step that touches the network — everything downstream reads the
local cache, so a report is reproducible offline.

### Why ETFs rather than spot

**One clock.** Spot FX trades 24×5 and the daily "close" is
whatever cut the vendor chose; equities and ETFs close together at
16:00 New York. Non-synchronous closes bias measured correlations
toward zero — the Epps effect — and a correlation model is the one
thing this repository is built around. On a single exchange clock
the covariance is measured on returns that actually overlap.

**Real volume.** Yahoo reports `Volume = 0` on FX spot, so a VWAP
computed from it is either a division by zero or a TWAP wearing a
VWAP's name. `fxrisk/indicators.py` raises rather than fabricate a
weight. ETFs report actual share volume, so the 20-day VWAP and its
9-period EMA in §3 are genuine.

**Gold without losing 2008.** `GLD` lists in November 2004, so gold
sits in the default book. The binding constraint is `FXY` at
2007-02, which is where the usable sample starts: Lehman, the SNB
floor, the 2015 CNY devaluation, Brexit, March 2020 and the 2022
LDI episode are all inside it. The spot path had the opposite
trade — `XAUUSD` starts 2009-03 and adding it truncated the sample
past the 2008 scenarios entirely.

**What you give up.** An ETF is not the underlying. Each carries an
expense ratio and tracking error, so multi-year *levels* drift from
the cross even though daily *returns* track closely; and the bars
stop at the equity close, so an overnight FX gap lands inside the
next day's return rather than its own.

### Two things worth knowing about this book

**These are not five independent risks.** `FXE`, `FXB`, `FXY` and
`FXF` are all dollar crosses, and `FXE`/`FXF` have historically been
close to mirror images. DCC reports that as high conditional
correlation, which is correct — but an equal-weight portfolio across
these five is less diversified than "five instruments" suggests. The
pairwise table in §3 is the place to look before assuming otherwise.

**GBP/JPY has no direct ETF.** It can be built synthetically as
`FXB / FXY` — both are quoted per USD, so the ratio is the cross,
and both trade on the same clock with real volume. It is off by
default (`config.SYNTHETIC_CROSSES`) because the two expense ratios
compound into the level.

### Switching universes

The spot books are retained for the HistData path, where local tick
files are the only source and the 17:00 New York cut is a deliberate
choice rather than a vendor default:

```python
# config.py
UNIVERSE = UNIVERSE_ETF       # default: one clock, real volume, from 2007-02
# UNIVERSE = UNIVERSE_FX      # spot FX via HistData, back to 2002
# UNIVERSE = UNIVERSE_FX_GOLD # spot plus gold, sample starts 2009-03
```

Nothing downstream hard-codes a symbol, so changing that one line
changes the whole report — including which stress scenarios are in
range.

---

## 6. `quant_metrics.py` — the original pipeline

The repository began as a compact rolling-metrics pipeline, and that
script is still here and still runs standalone:

```bash
python quant_metrics.py
```

It pulls daily closes from Yahoo Finance for whatever
`config.YAHOO_SYMBOLS` resolves to — the five ETFs, by default —
then computes a rolling 6-month (126-day) annualised Sharpe ratio
and a rolling 95% non-parametric historical VaR. Change the universe
in `config.py` and this script follows. It falls back to its own
hard-coded ticker list only if `config` cannot be imported, so it
still runs as a standalone file.

It has not been folded into `fxrisk/` because the two answer
different questions and the comparison is useful. `quant_metrics.py`
gives a fast rolling read on any Yahoo-listed instrument with three
dependencies and no estimation step. The engine below models the
volatility process explicitly, tests whether its risk numbers were
actually right, and works from local tick data rather than a vendor
API.

Where they overlap, the engine is the stricter of the two: rolling
historical VaR is exactly the estimator that passes Kupiec and fails
Christoffersen in §3, because a rolling empirical quantile cannot
react to a volatility regime change. That is not a criticism of the
original script — it is the finding that motivated the rest of this
repository.

---

## 7. Layout

```
fx-risk-engine/
├── quant_metrics.py             # original rolling Sharpe/VaR pipeline
├── config.py                    # every parameter
├── run_risk_report.py           # the pipeline
├── fxrisk/
│   ├── data/histdata.py         # tick + M1 loaders, VWAP, sessions
│   ├── models/
│   │   ├── ewma.py              # RiskMetrics EWMA
│   │   ├── garch.py             # GARCH(1,1), normal or Student-t
│   │   └── dcc.py               # DCC-GARCH
│   └── risk/
│       ├── var.py               # VaR + Expected Shortfall
│       ├── performance.py       # Sharpe, Sortino, drawdown
│       ├── correlation.py       # pairwise static vs EWMA vs DCC
│       ├── backtesting.py       # Kupiec, Christoffersen, Basel
│       └── stress.py            # dated historical scenarios
├── docs/
│   ├── diagrams/                # hand-drawn SVG: how the pieces fit
│   ├── figures/                 # generated from model output
│   └── make_figures.py          # regenerates docs/figures/
├── tests/test_risk_engine.py    # 56 tests
└── quant-portfolio/             # vendored, see §7
```

---

## 8. What the tests actually check

Not coverage for its own sake. Each test pins a property whose
silent failure would make output wrong in a way nobody would catch
by eye:

- **No look-ahead.** Perturbing the final return must leave every
  earlier forecast identical — for both EWMA and rolling GARCH.
  This is the most common backtest bug and the easiest to miss.
- **GARCH MLE recovers known parameters** from simulated series,
  including the Student-t degrees of freedom.
- **The independence test has nominal size**, checked across seeds
  rather than one lucky sample.
- **It has real power** against a flat VaR — and the aggregate
  framing documents that the power is limited.
- **Component VaR sums to total VaR** (Euler's theorem).
- **Stress replay reports partial coverage** instead of hiding it.
- **VWAP refuses to pass off a TWAP as a VWAP.**

Bugs found by writing these tests rather than by reading the code:
the Basel zone being applied below 99%; Lopez loss collapsing into
a plain breach count on decimal returns (the quadratic term is
~1e-4 against a constant of 1); and two test assertions that were
themselves statistically unsound.

### The leak the first version of these tests missed

`garch_var_series` originally estimated the Student-t degrees of
freedom **once on the full sample** and applied that single `nu` to
every historical quantile. The variance path was properly
walk-forward; the tail shape was not. Future data reached every
past VaR through the shape parameter rather than through the scale.

The original look-ahead tests could not see it. They perturbed the
last return and compared the *variance* series, which was honest.
Nothing checked the VaR series end to end.

Both are now estimated inside the rolling loop, and
`test_garch_var_series_is_not_forward_looking` compares the full
VaR frame against one computed on a truncated sample — the check
that would have caught it.

**Honest note on the size of the effect.** On simulated data with a
true `nu` of 4.5, the walk-forward estimate ranges 3.25 to 7.67
against a full-sample 4.28, and the resulting VaR differs by 1.1%
on average and 4.6% at most. The backtest verdicts were identical
before and after. So the bug was real but immaterial to any
conclusion drawn here. It is fixed because a reviewer who spots a
look-ahead leak discounts everything else in the repository, and
because the same class of error is not always this benign — not
because the numbers were wrong.

---

## 9. `quant-portfolio/`

A vendored copy of
[carlonimatteoo03/Quant_Portoflio](https://github.com/carlonimatteoo03/Quant_Portoflio)
by Matteo Carloni, MIT licensed — a VWAP strategy and mean-variance
portfolio construction. Kept separate because its portfolio
optimisation is in-sample by construction; see its own README.

---

## 10. Backtest-only results

**Everything this repository produces is a backtest.** It describes
what these models would have reported about the past. It is not a
prediction and not investment advice.

- **Nothing here is out-of-sample in the way that matters.** The
  scenario windows, the currency pairs and the model specifications
  were all chosen knowing what happened. Choosing to stress-test
  January 2015 is only possible because the SNB already moved.
- **Execution is ignored entirely.** No slippage, no funding, no
  market impact, no bid-offer beyond what tick data shows, and an
  implicit assumption that any position could be exited at the
  marked price. In every episode stress-tested here, that last
  assumption failed for somebody — that was largely what made them
  crises.
- **Passing a VaR backtest means adequate calibration over one
  sample.** It does not transfer to a different period, a different
  book, or a different market.
- **Model risk is not quantified.** GARCH and DCC are assumptions.
  Their parameters carry estimation error the reported figures do
  not show, and stage-2 DCC standard errors ignore stage-1 error
  entirely.
- **The stress library is not exhaustive.** It contains episodes
  that happened. The event that matters next is, by construction,
  not in it.
- **Past performance does not indicate future results.**

Not investment advice. Not a solicitation to trade. No warranty of
any kind.

---

## 11. References

- J.P. Morgan / Reuters, *RiskMetrics Technical Document*, 4th ed., 1996
- Engle, R. (2002), "Dynamic Conditional Correlation", *JBES* 20(3)
- Kupiec, P. (1995), "Techniques for Verifying the Accuracy of Risk Measurement Models", *Journal of Derivatives* 3(2)
- Christoffersen, P. (1998), "Evaluating Interval Forecasts", *International Economic Review* 39(4)
- Lopez, J. (1999), "Methods for Evaluating Value-at-Risk Estimates", *FRBSF Economic Review*
- Basel Committee, *Supervisory Framework for the Use of Backtesting*, 1996; *Minimum Capital Requirements for Market Risk* (FRTB), 2019

## 12. Licence

MIT.
