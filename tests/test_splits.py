"""Split detection tests.

These run against synthetic series, not the live Terminal, so they stay green
without a subscription or a network. The ratios and the shape of the volume
step are taken from the seven confirmed Nasdaq splits the detector was tuned
on (NVDA, AVGO, MSTR, SMCI, LRCX, PANW, NFLX).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.splits import detect_splits, adjust_for_splits  # noqa: E402


def series(n=60, start_price=1000.0, split_at=None, ratio=10.0,
           daily_drift=0.0, base_volume=1_000_000, split_day_gap=1.0):
    """Flat-ish price series with an optional split at index `split_at`."""
    rows, price, volume = [], start_price, base_volume
    day = date(2024, 1, 1)
    for i in range(n):
        if split_at is not None and i == split_at:
            price = price / ratio * split_day_gap
            volume = int(volume * ratio)
        o = price
        c = price * (1 + daily_drift)
        rows.append({"symbol": "TEST", "date": day, "open": o, "high": max(o, c) * 1.01,
                     "low": min(o, c) * 0.99, "close": c, "volume": volume,
                     "bid": c, "ask": c * 1.001, "count": 1000})
        price = c
        day += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("ratio", [2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
def test_detects_forward_splits(ratio):
    ev = detect_splits(series(split_at=30, ratio=ratio), "TEST")
    assert len(ev) == 1
    assert ev[0].ratio == pytest.approx(ratio)


@pytest.mark.parametrize("ratio", [0.5, 0.1, 0.05])
def test_detects_reverse_splits(ratio):
    ev = detect_splits(series(split_at=30, ratio=ratio, start_price=5.0), "TEST")
    assert len(ev) == 1
    assert ev[0].ratio == pytest.approx(ratio)


def test_clean_series_has_no_splits():
    assert detect_splits(series(daily_drift=0.002), "TEST") == []


def test_crash_is_not_mistaken_for_a_split():
    """A -50% overnight move with no change in share count is the single most
    dangerous false positive: it has exactly the price ratio of a 2:1 split.
    Only the absence of a volume step separates them."""
    df = series(split_at=30, ratio=2.0)
    # Undo the share-count step, keeping the price halving.
    df.loc[30:, "volume"] = 1_000_000
    assert detect_splits(df, "TEST") == []


def test_large_ratio_tolerates_a_concurrent_gap():
    """MSTR's 10:1 landed 4.3% off nominal because the stock also gapped that
    night. Large ratios must survive that; the alternative is missing them."""
    ev = detect_splits(series(split_at=30, ratio=10.0, split_day_gap=1.045), "TEST")
    assert len(ev) == 1 and ev[0].ratio == pytest.approx(10.0)


def test_small_ratio_rejects_a_concurrent_gap():
    """The same slack must NOT be extended to 2:1, where a real move can
    counterfeit the ratio."""
    assert detect_splits(series(split_at=30, ratio=2.0, split_day_gap=1.045), "TEST") == []


def test_adjustment_removes_the_discontinuity():
    df = series(split_at=30, ratio=10.0)
    adj = adjust_for_splits(df, detect_splits(df, "TEST"))
    returns = adj["close"].pct_change().abs()
    assert returns.max() < 0.05, "back-adjusted series still has a split-sized jump"


def test_adjustment_preserves_the_current_segment():
    """Vendor convention: history is restated, today's prices are untouched."""
    df = series(split_at=30, ratio=10.0)
    adj = adjust_for_splits(df, detect_splits(df, "TEST"))
    pd.testing.assert_series_equal(df["close"][30:], adj["close"][30:])


def test_empty_and_short_inputs_are_safe():
    assert detect_splits(pd.DataFrame(), "TEST") == []
    assert detect_splits(series(n=1), "TEST") == []
