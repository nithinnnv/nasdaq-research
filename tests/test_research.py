"""Point-in-time and evaluation tests.

The centrepiece is `test_no_lookahead_in_features`. Lookahead is the failure
mode that makes a backtest look brilliant and lose money, and it does not
announce itself -- a feature that peeks forward still produces plausible
numbers. The test states the property directly: truncating the panel after
date t must not change any feature value at date t. If a feature used future
data, deleting that data would move it.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.features import FEATURES, add_features, add_forward_returns  # noqa: E402
from research.evaluate import (  # noqa: E402
    cross_sectional_ic, decile_returns, ic_summary,
    sample_non_overlapping, walk_forward_split,
)


def panel(n_symbols=30, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_symbols):
        price, day = 100.0 * (1 + rng.random()), date(2023, 1, 2)
        for _ in range(n_days):
            r = rng.normal(0.0004, 0.02)
            o = price
            price *= 1 + r
            rows.append({"symbol": f"S{s:02d}", "date": day, "open": o,
                         "high": max(o, price) * 1.005, "low": min(o, price) * 0.995,
                         "close": price, "volume": int(rng.uniform(1e5, 5e6)),
                         "bid": price, "ask": price * 1.001, "count": 500})
            day += timedelta(days=1)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_no_lookahead_in_features():
    """Features at date t must not move when everything after t is deleted."""
    full = add_features(panel())
    cutoff = np.sort(full["date"].unique())[300]
    truncated = add_features(panel()[lambda d: d["date"] <= cutoff])

    a = full[full["date"] == cutoff].set_index("symbol")[FEATURES].sort_index()
    b = truncated[truncated["date"] == cutoff].set_index("symbol")[FEATURES].sort_index()
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9)


def test_forward_returns_enter_at_next_open():
    """A signal from today's close cannot be filled at today's price."""
    df = add_forward_returns(add_features(panel(n_symbols=3, n_days=60)), horizons=(5,))
    one = df[df["symbol"] == "S00"].sort_values("date").reset_index(drop=True)
    i = 10
    assert one["entry_price"][i] == pytest.approx(one["open"][i + 1])
    expected = one["open"][i + 6] / one["open"][i + 1] - 1
    assert one["fwd_5d"][i] == pytest.approx(expected)


def test_forward_returns_are_nan_at_the_tail():
    """The last h rows have no future to measure; they must not silently be 0."""
    df = add_forward_returns(add_features(panel(n_symbols=2, n_days=40)), horizons=(5,))
    one = df[df["symbol"] == "S00"].sort_values("date")
    assert one["fwd_5d"].tail(5).isna().all()


def test_momentum_skips_the_reversal_window():
    """mom_21d must ignore the last 5 sessions, so a spike inside that window
    cannot move it."""
    base = panel(n_symbols=2, n_days=120)
    spiked = base.copy()
    last_dates = np.sort(spiked["date"].unique())[-3:]
    mask = spiked["date"].isin(last_dates)
    spiked.loc[mask, ["open", "close", "high", "low"]] *= 1.5

    t = np.sort(base["date"].unique())[-1]
    a = add_features(base).set_index(["date", "symbol"]).loc[t, "mom_21d"]
    b = add_features(spiked).set_index(["date", "symbol"]).loc[t, "mom_21d"]
    pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-9)


def test_random_features_have_no_significant_ic():
    """On random walks every IC should be indistinguishable from zero. A
    harness that finds signal in noise is broken."""
    df = add_forward_returns(add_features(panel(n_symbols=40, n_days=300, seed=7)))
    for feat in ("mom_63d", "reversal_5d", "px_to_ma50"):
        s = ic_summary(df, feat, 10)
        if s is not None:
            assert abs(s.mean_ic) < 0.10, f"{feat} found structure in noise: IC={s.mean_ic:.3f}"


def test_newey_west_tstat_is_not_larger_than_naive_under_overlap():
    """The whole point of the correction: overlapping windows must not be
    allowed to look more significant than independent ones."""
    df = add_forward_returns(add_features(panel(n_symbols=40, n_days=400, seed=3)))
    s = ic_summary(df, "mom_63d", 20)
    assert s is not None
    assert abs(s.t_stat_nw) <= abs(s.t_stat_naive) * 1.15


def test_walk_forward_split_is_chronological_and_disjoint():
    df = add_forward_returns(add_features(panel(n_symbols=10, n_days=200)))
    parts = walk_forward_split(df)
    assert parts["train"]["date"].max() < parts["validation"]["date"].min()
    assert parts["validation"]["date"].max() < parts["test"]["date"].min()


def test_non_overlapping_sample_spaces_dates_by_horizon():
    df = add_forward_returns(add_features(panel(n_symbols=5, n_days=100)))
    sampled = sample_non_overlapping(df, 10)
    dates = np.sort(sampled["date"].unique())
    gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
    assert (gaps >= 10).all()


def test_decile_buckets_are_ordered_and_populated():
    df = add_forward_returns(add_features(panel(n_symbols=60, n_days=250)))
    d = decile_returns(df, "mom_63d", 10)
    assert len(d) == 10
    assert d["n"].min() > 0


def test_ic_requires_breadth():
    """Too few names on a date is noise, not a measurement."""
    df = add_forward_returns(add_features(panel(n_symbols=5, n_days=200)))
    assert cross_sectional_ic(df, "mom_63d", "fwd_10d", min_names=20).empty
