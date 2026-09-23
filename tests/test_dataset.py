"""Cache-loading tests for `research.dataset`.

This is the seam between the Parquet cache and everything downstream, and it
is the one module that cannot be exercised without data on disk -- which is
why it went untested until the synthetic cache in conftest existed.

The property that matters here is the one `dataset`'s own docstring claims:
back-adjustment repairs the split discontinuity in the *return* series while
leaving `raw_close` alone, so price-level filters stay point-in-time honest.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.dataset import load_panel, load_symbol  # noqa: E402
from research.features import FEATURES, add_features, add_forward_returns  # noqa: E402
from research.evaluate import ic_summary  # noqa: E402


def test_missing_symbol_is_empty_not_an_error(patched_cache):
    """A universe scan hits uncached names constantly; that must not raise."""
    assert load_symbol("NOSUCH").empty


def test_clean_symbol_loads_sorted_with_raw_close(patched_cache):
    df = load_symbol("AAAA")
    assert not df.empty
    assert df["date"].is_monotonic_increasing
    assert (df["raw_close"] == df["close"]).all()  # no split, nothing to adjust


def test_unadjusted_split_is_a_90_percent_day(patched_cache):
    """The hazard itself: without adjustment BBBB's 10:1 reads as -90%."""
    raw = load_symbol("BBBB", adjust=False)
    assert raw["close"].pct_change().min() < -0.85


def test_adjustment_removes_the_split_discontinuity(patched_cache):
    adj = load_symbol("BBBB", adjust=True)
    assert adj["split_events"].iloc[0] == 1
    # Every remaining daily move is ordinary; the -90% artefact is gone.
    assert adj["close"].pct_change().abs().max() < 0.10


def test_adjustment_leaves_raw_close_untouched(patched_cache):
    """`raw_close` is what price-level filters read, so it must stay raw."""
    adj = load_symbol("BBBB", adjust=True)
    raw = load_symbol("BBBB", adjust=False)
    pd.testing.assert_series_equal(adj["raw_close"], raw["close"], check_names=False)
    assert (adj["close"] != adj["raw_close"]).any()  # pre-split rows were restated


def test_panel_is_one_row_per_symbol_date(patched_cache):
    panel = load_panel()
    assert not panel.duplicated(subset=["symbol", "date"]).any()
    assert set(panel["symbol"]) == {"AAAA", "BBBB", "CCCC", "DDDD"}
    assert panel["date"].is_monotonic_increasing
    assert pd.api.types.is_datetime64_any_dtype(panel["date"])


def test_panel_subsets_and_skips_uncached_names(patched_cache):
    panel = load_panel(["AAAA", "NOSUCH"])
    assert set(panel["symbol"]) == {"AAAA"}


def test_panel_of_nothing_is_empty(patched_cache):
    assert load_panel(["NOSUCH", "NEITHER"]).empty


def test_pipeline_runs_end_to_end_from_cache(patched_cache):
    """cache -> panel -> features -> forward returns -> IC, with no network.

    Breadth is far below `min_names` here (4 symbols), so this asserts the
    wiring holds, not that any number is meaningful.
    """
    df = add_forward_returns(add_features(load_panel()), horizons=(5,))
    assert set(FEATURES).issubset(df.columns)
    assert "fwd_5d" in df.columns

    summary = ic_summary(df, "mom_21d", horizon=5, min_names=3)
    assert summary.n_dates > 0
    assert -1.0 <= summary.mean_ic <= 1.0
