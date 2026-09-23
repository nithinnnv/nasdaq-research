"""Overnight / intraday decomposition tests.

The decomposition is worth testing mainly because it is arithmetic with an
exact identity behind it: the two legs of a session multiply back to the
session. That identity holds over any number of sessions -- every day is
(overnight x intraday), and multiplication commutes, so compounding all the
overnight legs and all the intraday legs separately and multiplying the two
results reproduces close-to-close exactly. Any indexing slip in the rolling
or shifting breaks it, which makes it a sharper check than comparing against
a hand-computed number.

The last test is the one that states the point of the exercise: a name whose
two legs pull against each other looks inert close-to-close and is not.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.features import (  # noqa: E402
    FEATURES_CLOSE, FEATURES_COMPONENT, SKIP_DAYS,
    add_component_forward_returns, add_features,
)


def panel_from_legs(overnight, intraday, symbol="TEST", start_price=100.0):
    """Build bars from prescribed per-session legs, so both are known exactly."""
    rows, close, day = [], start_price, date(2024, 1, 1)
    for on, intra in zip(overnight, intraday):
        opn = close * (1 + on)
        close = opn * (1 + intra)
        rows.append({"symbol": symbol, "date": pd.Timestamp(day), "open": opn,
                     "high": max(opn, close), "low": min(opn, close),
                     "close": close, "volume": 1_000_000})
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def drifting(n=300, on=0.003, intra=-0.0025, symbol="TEST", seed=0, noise=0.0):
    """A name that rises overnight and gives it back during the session."""
    rng = np.random.default_rng(seed)
    o = np.full(n, on) + rng.normal(0, noise, n)
    i = np.full(n, intra) + rng.normal(0, noise, n)
    return panel_from_legs(o, i, symbol=symbol)


# --- the identity ----------------------------------------------------------

def test_legs_multiply_back_to_the_session():
    df = add_features(drifting(n=60, seed=1, noise=0.01))
    rebuilt = (1 + df["ret_overnight"]) * (1 + df["ret_intraday"]) - 1
    pd.testing.assert_series_equal(
        rebuilt.iloc[1:], df["ret_1d"].iloc[1:], check_names=False)


def test_first_session_has_no_overnight_leg():
    """There is no prior close to gap from, so it must be NaN, not zero."""
    df = add_features(drifting(n=30))
    assert pd.isna(df["ret_overnight"].iloc[0])
    assert not pd.isna(df["ret_intraday"].iloc[0])


def test_overnight_gap_still_aliases_the_leg():
    """gap_vol_21d and anything else downstream reads the original name."""
    df = add_features(drifting(n=60, noise=0.01))
    pd.testing.assert_series_equal(
        df["overnight_gap"], df["ret_overnight"], check_names=False)


# --- compounding -----------------------------------------------------------

def test_component_momentum_compounds_the_right_window():
    """Constant legs make the expected value exact: (1+on)^63 - 1."""
    df = add_features(drifting(n=200, on=0.003, intra=-0.0025))
    expected_on = (1.003 ** 63) - 1
    expected_id = (0.9975 ** 63) - 1
    assert df["mom_on_63d"].iloc[-1] == pytest.approx(expected_on, rel=1e-9)
    assert df["mom_id_63d"].iloc[-1] == pytest.approx(expected_id, rel=1e-9)


def test_component_momentum_skips_the_reversal_window():
    """A recent burst in the skipped window must not reach mom_on_21d."""
    n = 120
    on = np.full(n, 0.001)
    on[-SKIP_DAYS:] = 0.25  # violent, and entirely inside the skip
    df = add_features(panel_from_legs(on, np.zeros(n)))
    assert df["mom_on_21d"].iloc[-1] == pytest.approx((1.001 ** 21) - 1, rel=1e-9)


def test_component_reversal_does_not_skip():
    """reversal_5d_on is the skipped window itself, so it must see the burst."""
    n = 120
    on = np.full(n, 0.001)
    on[-SKIP_DAYS:] = 0.25
    df = add_features(panel_from_legs(on, np.zeros(n)))
    assert df["reversal_5d_on"].iloc[-1] == pytest.approx((1.25 ** 5) - 1, rel=1e-9)


def test_missed_split_does_not_seed_infinities():
    """A -100% leg is a split the detector missed; log1p would return -inf."""
    n = 120
    on = np.full(n, 0.001)
    on[60] = -0.999999
    df = add_features(panel_from_legs(on, np.zeros(n)))
    assert np.isfinite(df["mom_on_63d"].dropna()).all()


# --- forward legs ----------------------------------------------------------

def test_forward_legs_multiply_back_to_close_to_close():
    """The identity again, forward: the two legs reconstruct close-to-close."""
    df = add_component_forward_returns(
        add_features(drifting(n=200, seed=3, noise=0.01)), horizons=(5,))
    rebuilt = (1 + df["fwd_on_5d"]) * (1 + df["fwd_id_5d"]) - 1
    actual = df["close"].shift(-5) / df["close"] - 1
    valid = rebuilt.notna() & actual.notna()
    assert valid.sum() > 150
    np.testing.assert_allclose(rebuilt[valid], actual[valid], rtol=1e-9)


def test_forward_legs_look_only_forward():
    """fwd_on_5d at t compounds sessions t+1..t+5 and nothing else."""
    df = add_component_forward_returns(
        add_features(drifting(n=80, seed=4, noise=0.01)), horizons=(5,))
    t = 40
    manual = np.prod(1 + df["ret_overnight"].iloc[t + 1: t + 6].to_numpy()) - 1
    assert df["fwd_on_5d"].iloc[t] == pytest.approx(manual, rel=1e-9)


def test_forward_legs_are_nan_without_a_full_window():
    df = add_component_forward_returns(
        add_features(drifting(n=60)), horizons=(5,))
    assert df["fwd_on_5d"].tail(5).isna().all()
    assert df["fwd_id_5d"].tail(5).isna().all()


def test_forward_legs_do_not_bleed_across_symbols():
    """Concatenated symbols must not compound one name's tail into the next."""
    a, b = drifting(n=60, symbol="AAA", seed=5), drifting(n=60, symbol="BBB", seed=6)
    df = add_component_forward_returns(
        add_features(pd.concat([a, b], ignore_index=True)), horizons=(5,))
    for sym in ("AAA", "BBB"):
        assert df[df["symbol"] == sym]["fwd_on_5d"].tail(5).isna().all()


def test_component_forward_returns_demand_add_features_first():
    raw = drifting(n=30)
    with pytest.raises(ValueError, match="add_features"):
        add_component_forward_returns(raw)


# --- the point of the exercise --------------------------------------------

def test_opposing_legs_are_invisible_close_to_close():
    """A name drifting +0.3% overnight and -0.25% intraday looks flat.

    This is the whole argument for the decomposition: close-to-close reports
    a rounding error while each leg is carrying a large, steady, opposite
    signal that a close-only feature set cannot see.
    """
    df = add_features(drifting(n=200, on=0.003, intra=-0.0025))
    last = df.iloc[-1]

    assert abs(last["mom_63d"]) < 0.05          # close-to-close: nearly inert
    assert last["mom_on_63d"] > 0.15            # overnight: strongly positive
    assert last["mom_id_63d"] < -0.13           # intraday: strongly negative
    assert last["tug_63d"] > 0.3                # and the spread is large


def test_feature_lists_are_disjoint_and_complete():
    df = add_features(drifting(n=300))
    assert not set(FEATURES_CLOSE) & set(FEATURES_COMPONENT)
    assert set(FEATURES_CLOSE + FEATURES_COMPONENT).issubset(df.columns)
