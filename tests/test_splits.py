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


# --- the price floor applies after the event, not before --------------------
#
# Gating on the prior close excluded reverse splits as a class: Nasdaq requires
# a $1 minimum bid, so a company reverse-splits precisely because it is trading
# below a dollar. A pre-event floor is therefore not a noise filter for this
# population, it is a blanket exclusion of it. Across the live cache the fix
# recovered 99 reverse splits and lost nothing -- forward detections were
# unchanged at 62, and every recovered event had volume fall by the ratio.

@pytest.mark.parametrize("ratio,start", [(0.1, 0.40), (0.05, 0.25), (1 / 30, 0.40)])
def test_detects_reverse_splits_from_below_a_dollar(ratio, start):
    """The real shape: a sub-$1 stock reverse-splitting up to a few dollars."""
    ev = detect_splits(series(split_at=30, ratio=ratio, start_price=start), "TEST")
    assert len(ev) == 1
    assert ev[0].ratio == pytest.approx(ratio)


def test_penny_stock_noise_below_the_floor_is_still_rejected():
    """The floor's original job: a sub-$1 move that lands sub-$1 is not a split.

    A 2x price step on a $0.30 stock has the ratio of a 1:2 reverse, but it
    ends at $0.60 -- no real reverse split leaves the stock under a dollar,
    which is the whole reason the company did it.
    """
    assert detect_splits(series(split_at=30, ratio=0.5, start_price=0.30), "TEST") == []


def test_reverse_split_still_needs_the_volume_step():
    """Recovering reverse splits must not weaken the corroboration rule.

    A sub-$1 stock that genuinely rallies 10x on news has the price ratio of a
    1:10 reverse. Only the share count separates them, and news does not cut it.
    """
    df = series(split_at=30, ratio=0.1, start_price=0.40)
    df.loc[30:, "volume"] = 1_000_000  # share count never changed
    assert detect_splits(df, "TEST") == []
