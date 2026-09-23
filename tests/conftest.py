"""Shared synthetic fixtures.

Everything the suite needs is generated here rather than committed as data.
The cached Parquet bars this project actually runs on come from a paid
ThetaData subscription, so they can neither be redistributed nor regenerated
in CI -- but the *shape* of that cache is simple enough to reproduce exactly,
and a generated fixture has the advantage of being able to contain the cases
that matter (a split, a bad print, a short-history IPO) on demand.

`bars_cache` mirrors the real `cache/bars/` layout: one Parquet file per
symbol, named for the symbol, with the columns `StockEODClient.fetch`
returns.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Columns and dtypes as they come back from the EOD endpoint.
BAR_COLUMNS = ["symbol", "date", "open", "high", "low", "close",
               "volume", "bid", "ask", "count"]


def make_bars(symbol: str, n_days: int = 80, start_price: float = 100.0,
              start: date = date(2024, 1, 2), seed: int = 0,
              split_at: int | None = None, split_ratio: float = 10.0,
              bad_print_at: int | None = None,
              zero_at: tuple[int, ...] = (),
              partial_zero_at: tuple[int, ...] = (),
              zero_open_at: tuple[int, ...] = (),
              base_volume: int = 1_000_000) -> pd.DataFrame:
    """One symbol's daily bars, with optional split and bad-print hazards.

    `split_at` divides the price by `split_ratio` and multiplies volume by it
    from that index on -- the unadjusted discontinuity the real feed serves.
    `bad_print_at` inflates that session's `high` without touching open/close,
    which is the exact shape of NVDA's 2024-06-10 bar.

    Three shapes of zero bar, all present in the live cache:
    `zero_at` is fully zeroed -- the feed's marker for a session the symbol
    did not trade. `partial_zero_at` is a real open against a zeroed
    high/low/close. `zero_open_at` is the reverse: a real close with the open
    missing, which is the only one of the three that leaves a usable bar.
    """
    rng = np.random.default_rng(seed)
    rows, price, volume, day = [], start_price, base_volume, start

    for i in range(n_days):
        if split_at is not None and i == split_at:
            price /= split_ratio
            volume = int(volume * split_ratio)

        o = price
        price *= 1 + rng.normal(0.0005, 0.012)
        c = price
        hi, lo = max(o, c) * 1.004, min(o, c) * 0.996
        if bad_print_at is not None and i == bad_print_at:
            hi *= 1.6  # aggregate contaminated by a single bad print

        if i in zero_at:
            rows.append({"symbol": symbol, "date": day, "open": 0.0, "high": 0.0,
                         "low": 0.0, "close": 0.0, "volume": 0, "bid": 0.0,
                         "ask": 0.0, "count": 0})
            day += timedelta(days=1)
            continue
        if i in zero_open_at:
            rows.append({"symbol": symbol, "date": day, "open": 0.0,
                         "high": round(hi, 4), "low": round(lo, 4),
                         "close": round(c, 4), "volume": int(volume),
                         "bid": round(c * 0.9995, 4), "ask": round(c * 1.0005, 4),
                         "count": 300})
            day += timedelta(days=1)
            continue
        if i in partial_zero_at:
            rows.append({"symbol": symbol, "date": day, "open": round(o, 4),
                         "high": 0.0, "low": 0.0, "close": 0.0, "volume": 6,
                         "bid": 0.0, "ask": 0.0, "count": 1})
            day += timedelta(days=1)
            continue

        rows.append({
            "symbol": symbol, "date": day,
            "open": round(o, 4), "high": round(hi, 4),
            "low": round(lo, 4), "close": round(c, 4),
            "volume": int(rng.uniform(0.8, 1.2) * volume),
            "bid": round(c * 0.9995, 4), "ask": round(c * 1.0005, 4),
            "count": int(rng.uniform(200, 900)),
        })
        day += timedelta(days=1)

    return pd.DataFrame(rows)[BAR_COLUMNS]


# (symbol, kwargs) -- one clean name, one splitter, one with a bad print,
# and one IPO too short for the long-window features to resolve.
FIXTURE_SYMBOLS = {
    "AAAA": dict(seed=1, start_price=90.0),
    "BBBB": dict(seed=2, start_price=1200.0, split_at=40, split_ratio=10.0),
    "CCCC": dict(seed=3, start_price=45.0, bad_print_at=25),
    "DDDD": dict(seed=4, start_price=30.0, n_days=12),
    "EEEE": dict(seed=5, start_price=60.0, zero_at=(20, 21, 22),
                 partial_zero_at=(50,), zero_open_at=(60,)),
}


@pytest.fixture
def bars_cache(tmp_path: Path) -> Path:
    """A `cache/bars/` directory populated with the fixture symbols."""
    cache = tmp_path / "bars"
    cache.mkdir()
    for symbol, kwargs in FIXTURE_SYMBOLS.items():
        make_bars(symbol, **kwargs).to_parquet(cache / f"{symbol}.parquet", index=False)
    return cache


@pytest.fixture
def patched_cache(bars_cache, monkeypatch) -> Path:
    """Point `research.dataset` at the fixture cache for the test's duration."""
    from research import dataset
    monkeypatch.setattr(dataset, "CACHE", bars_cache)
    return bars_cache
