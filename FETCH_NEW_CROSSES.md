# Fetch GBP/JPY and AUD/NZD — run these yourself

Dukascopy answers `403 Forbidden` through the proxy in both of my
environments, so this is the one step I can't do. Six commands, run from
the repo root. `--only` is new: without it the script re-downloads five
years for the four instruments already cached, which is both slow and
rude to a free endpoint.

Roughly 10–20 minutes each; they pause between chunks on purpose. Run
them one at a time, not in parallel.

```
python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --side bid --start 2021-08-31 --years 2 --tag 2021
python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --side ask --start 2021-08-31 --years 2 --tag 2021

python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --side bid --start 2023-08-31 --years 1 --tag 2023
python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --side ask --start 2023-08-31 --years 1 --tag 2023

python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --side bid --years 2
python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --side ask --years 2
```

The three windows match the chunk layout the other four already use, so
the runners pick them up with no change:

| tag | window | matches |
|---|---|---|
| `2021` | 2021-08-31 → 2023-08-31 | `*_5m_2021.csv` |
| `2023` | 2023-08-31 → 2024-08-30 | `*_5m_2023.csv` |
| none | trailing 2 years | `*_5m.csv` |

## Check it landed

```
python3 fetch_intraday.py --only GBP/JPY,AUD/NZD --interval 5m --check
ls -la data/intraday/GBPJPY_5m* data/intraday/AUDNZD_5m*
```

You want 12 files: two instruments x three windows x bid and ask. Say
"done" and I'll run the pre-registered test in
`docs/PREREG_added_power.md`.

## If something goes wrong

- `dukascopy-python is not installed` → `pip3 install dukascopy-python`
- `REJECTED: ...` on one instrument → the sanity check caught something
  odd in the feed. Send me the message; don't work around it.
- A window returns nothing → tell me which one. A short chunk is fine,
  a missing year is not, and I'd rather know than silently pool it.
