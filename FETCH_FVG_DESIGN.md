# Fetch XAU/USD 1-minute bid+ask, 2015 -> now — run these yourself

The sandbox this repository is developed in cannot reach
`freeserv.dukascopy.com` (outbound network policy blocks it - a
confirmed 403, not a transient failure), so this step has to run on
your own machine. Everything after it is offline and reproducible.

```bash
pip install dukascopy-python

python fetch_fvg_raw.py --side both
# equivalent to: --start 2015 --end <this year> --side both
```

That's the whole thing — one command, resumable. It writes one file
per year per side into `data/raw/`:

```
data/raw/XAUUSD_1m_bid_2015.csv
data/raw/XAUUSD_1m_ask_2015.csv
data/raw/XAUUSD_1m_bid_2016.csv
...
```

It fetches each year in 20-day sub-chunks with a pause between
requests, and skips any year+side file that's already on disk, so if
it's interrupted or rate-limited, just run the same command again.

## Check what's landed

```bash
python fetch_fvg_raw.py --check
```

Prints which year+side files exist and which are still missing,
without touching the network.

## If you only want the design period for now

`run_fvg_design.py` only reads 2015-01-01 -> 2022-01-01, so you can
stop the fetch there and come back for 2022+ later, right before you
run the held-out test:

```bash
python fetch_fvg_raw.py --start 2015 --end 2021 --side both
```

The held-out years (2022 -> now) only need to be fetched once you've
actually signed `PREREGISTRATION.md` — there's no reason to have that
data sitting on disk, tempting a peek, before the criteria are frozen.

## One thing that might not be exactly 1-minute-complete at the very
start of 2015

Dukascopy's free tick/candle history can be thinner in the very
earliest part of a long window. If a given year comes back short,
`fetch_fvg_raw.py --check` will show it; it isn't a bug in the
fetcher and isn't worth working around — a thin first year will just
mean fewer design-period trades near 2015, which the design-period
`RESULTS.md` will show honestly.

## Then

```bash
make run        # or: python run_fvg_design.py
```

Read `RESULTS.md`. If it's a null result, that's the result — the
design period doesn't exist to be pushed on until something clears a
hurdle.
