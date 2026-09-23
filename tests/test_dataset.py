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

import numpy as np
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
    assert set(panel["symbol"]) == {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE"}
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


# --- zero bars -------------------------------------------------------------
#
# The feed returns a fully-zeroed row for a session a symbol did not trade.
# Left in place they are not merely missing data: a pct_change across a zero
# prints -1 on the way in and inf on the way out, and inf survives every rank,
# rolling window and mean downstream. This was found in the live cache -- 3,767
# such rows across 175 of the first 600 cached symbols.

def test_bars_without_a_usable_close_are_dropped(patched_cache):
    """Three fully-zeroed sessions and one with a real open but no close."""
    raw = pd.read_parquet(patched_cache / "EEEE.parquet")
    assert (raw["close"] == 0).sum() == 4

    df = load_symbol("EEEE")
    assert (df["close"] > 0).all()
    assert len(df) == len(raw) - 4


def test_a_missing_open_voids_the_field_not_the_bar(patched_cache):
    """The close is real, so close-to-close survives; only the legs are lost."""
    raw = pd.read_parquet(patched_cache / "EEEE.parquet")
    kept = raw.iloc[60]
    assert kept["open"] == 0 and kept["close"] > 0

    df = load_symbol("EEEE")
    row = df[df["date"] == kept["date"]]
    assert len(row) == 1                      # the bar is kept
    assert pd.isna(row["open"].iloc[0])       # but its open is not a price
    assert row["close"].iloc[0] > 0


def test_zero_bars_do_not_produce_infinities(patched_cache):
    """The failure this guards against: one inf poisons every rank downstream."""
    from research.features import FEATURES, add_features
    df = add_features(load_panel())
    for col in ["ret_1d", "ret_overnight", "ret_intraday", *FEATURES]:
        assert np.isfinite(df[col].dropna()).all(), f"{col} carries a non-finite value"


def test_zero_bars_are_not_mistaken_for_splits(patched_cache):
    """A zero bar makes a price ratio of inf; the detector must never see it."""
    assert "split_events" not in load_symbol("EEEE").columns
