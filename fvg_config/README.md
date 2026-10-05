# Optional subset configs

Both files here are **optional**. Neither exists in this repository -
`run_fvg_design.py` and `run_fvg_test.py` check for them at the start
and print one line saying whether each was found; if a file is
missing, its subset tables are simply left out of `RESULTS.md` rather
than failing the run. Add either (or both) to see the rule's results
broken out around news and named stress episodes, on top of the
headline numbers.

Neither file should be invented or filled with placeholder rows by an
automated session - a news calendar or a stress-window list is a claim
about real events, and a wrong or fabricated one would quietly bias
the subset it defines. Source real data for `news_calendar.csv` (a
broker's or data vendor's economic calendar export, filtered to the
instrument's relevant currencies and commodities desk) or leave it
out.

## `news_calendar.csv`

One row per scheduled news event, in UTC.

```csv
timestamp,event,impact
2022-01-26T19:00:00Z,FOMC rate decision,high
2022-02-02T13:30:00Z,US nonfarm payrolls,high
2022-02-10T13:30:00Z,US CPI,medium
```

| column | meaning |
|---|---|
| `timestamp` | UTC, ISO 8601, when the event is/was scheduled to print |
| `event` | free text, for the table's row labels |
| `impact` | `low` / `medium` / `high` - used to filter which events count |

`fxrisk.research.fvg_subsets.load_news_calendar` reads this and
`in_news_window` flags every trade whose entry falls within a
configurable window (default 30 minutes) either side of a `high`-impact
event, so `RESULTS.md` can show the rule's numbers with and without
those windows. This is a different question from the daily engine's
dated stress replays below `config.STRESS_SCENARIOS`: those are whole
historical episodes: this is "does this intraday rule behave
differently in the minutes around a scheduled print," which needs
calendar data at minute resolution, not a handful of named weeks.

## `stress_windows.csv`

One row per named window to break results out by, in UTC.

```csv
name,start,end
gold_2020_covid_crash,2020-03-06T00:00:00Z,2020-03-20T00:00:00Z
gold_2022_rate_shock,2022-03-01T00:00:00Z,2022-03-31T00:00:00Z
```

| column | meaning |
|---|---|
| `name` | label for the row in `RESULTS.md` |
| `start` / `end` | UTC, ISO 8601, half-open `[start, end)` |

These can be anything worth isolating that isn't already one of
`config.STRESS_SCENARIOS` - a period specific to gold rather than the
cross-asset dated episodes the daily engine replays, or a window
inside the design period worth a closer look. Windows may overlap; a
trade can appear in more than one row.

## Loading them yourself

```python
from fxrisk.research import fvg_subsets

calendar = fvg_subsets.load_news_calendar()   # None if the file is absent
windows = fvg_subsets.load_stress_windows()   # None if the file is absent
```
