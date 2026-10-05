# Fetch 2016–2021 for the three cheapest instruments — run these yourself

Four commands. Each covers all three instruments; `--only` keeps it from
re-pulling anything already cached. Run them one at a time.

```
python3 fetch_intraday.py --only NAS100,XAU/USD,USD/JPY --interval 5m --side bid --start 2016-09-01 --years 2 --tag 2016
python3 fetch_intraday.py --only NAS100,XAU/USD,USD/JPY --interval 5m --side ask --start 2016-09-01 --years 2 --tag 2016

python3 fetch_intraday.py --only NAS100,XAU/USD,USD/JPY --interval 5m --side bid --start 2018-08-31 --years 3 --tag 2018
python3 fetch_intraday.py --only NAS100,XAU/USD,USD/JPY --interval 5m --side ask --start 2018-08-31 --years 3 --tag 2018
```

Windows are chosen to end exactly where the existing caches begin
(2021-08-30), so **nothing overlaps data this study has already used.**
That is the entire point — an overlap would silently turn a frozen shot
back into an in-sample fit.

| tag | window | file |
|---|---|---|
| `2016` | 2016-09-01 → 2018-08-31 | `*_5m_2016.csv` + `*_5m_ask_2016.csv` |
| `2018` | 2018-08-31 → 2021-08-30 | `*_5m_2018.csv` + `*_5m_ask_2018.csv` |

You want 12 files: three instruments x two windows x bid and ask.

## Check it landed

```
ls -la data/intraday/*_5m_2016.csv data/intraday/*_5m_2018.csv
ls -la data/intraday/*_5m_ask_2016.csv data/intraday/*_5m_ask_2018.csv
```

## The one thing that might not work

**NAS100 may not go back to 2016.** Dukascopy's index CFD history is
shorter than its FX history, and I can't check from here. If it returns
little or nothing, don't work around it — the runner will exclude any
instrument with under 3 years and print that exclusion in its headline,
which is the pre-registered behaviour. Tell me what you see.

Gold and USD/JPY should both be fine.

## Then

Say "done" and I'll run `prior_cheap3_test.py` once. It is already
committed, along with the criteria in
`docs/PREREG_prior_period_cheap3.md`, so nothing about the analysis can
change now that would not be visible in git history.

Read the prereg before the result if you want the honest frame. The short
version: a significance test on net R is impossible here — 320 years of
trades — so the test rests on the excess clearing its 1.50pp cost
breakeven, and even a PASS means "real but indistinguishable from zero at
any feasible sample", which licenses a conversation about cost structure
and nothing else. I've predicted a FAIL on the record.
