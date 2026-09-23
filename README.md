# nasdaq-research

[![tests](https://github.com/nithinnnv/nasdaq-research/actions/workflows/ci.yml/badge.svg)](https://github.com/nithinnnv/nasdaq-research/actions/workflows/ci.yml)

Directional equity research over Nasdaq-listed common stock, on daily bars from
a local ThetaData Terminal. Swing horizon (days to weeks).

## Prerequisite

The ThetaData Terminal must be running before anything here works — it holds the
credentials and serves the REST API on `:25503`:

```bash
~/ThetaTerminal/start.sh
```

## Subscription reality (probed against the live terminal, 2026-09-06)

Stock and options entitlements are **separate tiers**. Current state: options
on VALUE, stock on **FREE**.

| Stock data | Tier needed | Available now |
|---|---|---|
| EOD daily bars, 2023-06-01 → today | FREE | ✅ |
| EOD daily bars, 2021-01-01 → 2023-05-31 | VALUE | ❌ |
| EOD daily bars, 2016-01-01 → 2020-12-31 | STANDARD | ❌ |
| EOD daily bars, 2012-06-01 → 2015-12-31 | PRO | ❌ |
| 1-minute intraday OHLC / quotes | VALUE | ❌ |
| Trades / trade-quote | STANDARD | ❌ |
| **Splits endpoint** | VALUE | ❌ (worked around — see below) |

Two hard API constraints that shape everything: **30 requests/minute** and a
**365-day maximum window per request**. `data/throttle.py` handles the first;
`StockEODClient.fetch` chunks transparently for the second.

## The two data hazards

**1. Prices are unadjusted.** ThetaData serves raw prices at every tier. NVDA
closes 1208.88 on 2024-06-07 and opens 120.45 on 2024-06-10 — a 10:1 split that
reads as a −90% day. The Splits endpoint that would resolve this needs VALUE, so
`data/splits.py` recovers split events from the series itself, requiring a price
ratio near a known split ratio **and** a corroborating step in median volume.
Validated at **21/21** against seven confirmed Nasdaq splits (NVDA, AVGO, MSTR,
SMCI, LRCX, PANW, NFLX) and fourteen non-splitting controls.

**2. Bad prints survive into the aggregates.** NVDA's 2024-06-10 bar reports a
high of 195.95 on a session that traded 117–123. Anything reading `high`/`low`
needs outlier filtering; `close` and `open` were clean across every symbol
checked.

## What no ThetaData tier provides

- **Fundamentals** — no earnings, revenue, margins, share count, valuation.
- **Dividends** — Splits only, even on VALUE. Total-return work needs another source.
- **Listing metadata** — no exchange, sector, or security type. Comes from
  Nasdaq Trader's public listing file instead (`data/universe.py`).

## Universe

`build_universe()` intersects Nasdaq's daily listing file with ThetaData's
symbol coverage:

```
nasdaq listing file          5,592
drop test issues             5,584
drop ETFs                    4,326
drop delinquent filers       3,961
round lot filter             3,881
common stock only            3,018   ← units/warrants/rights/preferreds removed
intersect ThetaData          3,018   ← full coverage, nothing lost
```

`UniverseFilter(exclude_spacs=True)` drops a further 162 pre-deal SPACs.

## Backfill

```bash
python scripts/backfill.py --phase probe          # 1 req/symbol, ~1.9h
python scripts/backfill.py --phase full           # remaining windows, liquid names only
```

Measured throughput: **~27 symbols/min**. Full history for the whole universe is
4 requests × 3,018 symbols ≈ 7.5 hours, which is why `probe` exists — one recent
window per symbol is enough to screen for liquidity, and full history is then
worth pulling only for the few hundred names that survive. The per-symbol
Parquet cache makes any run resumable.

## Known limitations

- **No bear market in range.** 2023-06 → today is a single broadly rising
  regime. Signals validated here are hypothesis-generating, not validated.
  Upgrading to VALUE extends the window to 2021-01 and adds the 2022 drawdown;
  the code clamps automatically, so it is a re-run, not a rewrite.
- **Survivorship bias.** The listing file is a snapshot of *live* listings.
  Anything delisted mid-window is absent, biasing any historical study upward.
- **NBBO timing.** The `bid`/`ask` on each row are stamped ~17:15 ET, after the
  close, so they do not correspond to the `close` price. Use them for spread
  estimation, not for fill modelling.

## Layout

```
data/throttle.py     rate limiting (tier-aware)
data/theta_stock.py  EOD client: chunking, tier errors, retry policy
data/splits.py       split detection + back-adjustment
data/universe.py     Nasdaq listing + ThetaData intersection
scripts/backfill.py  resumable two-phase backfill
tests/               65 tests: synthetic fixtures, no network, no subscription
```

```bash
python -m pytest tests/ -q
```

The suite builds its own data -- `tests/conftest.py` generates a synthetic
`cache/bars/` containing the hazards worth testing against (a 10:1 split, a
high contaminated by a bad print, an IPO too short to resolve the long
windows). Nothing here needs the Terminal, a subscription, or a network, so
the same command runs in CI on every push.
