"""Cross-sectional features for a swing (days-to-weeks) horizon.

Every feature at date t is computed from data at or before t. That is enforced
structurally -- each one is a rolling or shifted operation within a symbol,
never a whole-sample statistic -- because the usual way a backtest lies to you
is a feature that quietly peeked forward.

The momentum family skips the most recent 5 sessions. Short-horizon momentum
inverts (last week's winners tend to give it back), so leaving the gap in
separates the two effects instead of letting them cancel inside one number.

The same cancellation argument applies a second time, along a different axis.
Lou, Polk & Skouras (JFE 2019) decompose each session into its overnight leg
(prior close -> open) and its intraday leg (open -> close) and find the two
carry continuation of *opposite* sign, because different clienteles dominate
the open and the rest of the session. A close-to-close signal sums the two and
is therefore partly them netting out. Every momentum window here is computed
on each leg separately as well as on the whole session.

That mirroring is deliberate: the component features are the existing signal
set re-cut, not a new family of guesses, so the comparison is like-for-like.
It still triples the number of things being tested, which is a multiple-
comparisons problem -- judge them against each other, not against zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKIP_DAYS = 5  # short-term reversal window excluded from momentum


def _by_symbol(panel: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return panel.groupby("symbol", group_keys=False, observed=True)


def _compound(df: pd.DataFrame, col: str, window: int, skip: int) -> pd.Series:
    """Compound `col` over `window` sessions ending `skip` sessions ago.

    Summing logs rather than chaining products: a 63-session cumulative
    product over a panel this size is where floating-point drift and NaN
    propagation both show up. The clip only ever binds on a -100% leg, which
    means a split the detector missed -- log1p would hand back -inf and seed
    it through every rank downstream.
    """
    log_leg = np.log1p(df[col].clip(lower=-0.99))
    ending = _by_symbol(df.assign(_l=log_leg))["_l"].shift(skip)
    summed = _by_symbol(df.assign(_e=ending))["_e"].transform(
        lambda s: s.rolling(window, min_periods=max(2, window // 2)).sum()
    )
    return np.expm1(summed)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the feature set. Input is the long panel from dataset.load_panel."""
    df = panel.sort_values(["symbol", "date"]).copy()
    g = _by_symbol(df)

    df["ret_1d"] = g["close"].pct_change()
    df["dollar_volume"] = df["close"] * df["volume"]

    # --- the two legs of a session ---
    # Both are point-in-time at t: the overnight leg closed at this morning's
    # open and the intraday leg at this afternoon's close.
    df["ret_overnight"] = df["open"] / g["close"].shift(1) - 1
    df["ret_intraday"] = df["close"] / df["open"] - 1

    # --- momentum, skipping the reversal window ---
    for window in (21, 63, 126, 252):
        past = g["close"].shift(SKIP_DAYS)
        df[f"mom_{window}d"] = past / _by_symbol(df.assign(_p=past))["_p"].shift(window) - 1

    # --- the same windows, cut by leg ---
    for window in (21, 63, 126, 252):
        df[f"mom_on_{window}d"] = _compound(df, "ret_overnight", window, SKIP_DAYS)
        df[f"mom_id_{window}d"] = _compound(df, "ret_intraday", window, SKIP_DAYS)

    # Firm-level tug of war: how much of this name's drift arrived overnight
    # rather than during the session. The paper's own spread is a time-series
    # timing signal on a strategy; this is the cross-sectional analogue and
    # should be judged on its own evidence, not on theirs.
    df["tug_63d"] = df["mom_on_63d"] - df["mom_id_63d"]

    # --- short-term reversal (the effect momentum deliberately excludes) ---
    df["reversal_5d"] = g["close"].pct_change(SKIP_DAYS)
    df["reversal_5d_on"] = _compound(df, "ret_overnight", SKIP_DAYS, skip=0)
    df["reversal_5d_id"] = _compound(df, "ret_intraday", SKIP_DAYS, skip=0)

    # --- realised volatility ---
    for window in (21, 63):
        df[f"vol_{window}d"] = g["ret_1d"].transform(
            lambda s, w=window: s.rolling(w, min_periods=w // 2).std() * np.sqrt(252)
        )

    # Risk-adjusted momentum: the same signal per unit of noise carrying it.
    df["mom_63d_riskadj"] = df["mom_63d"] / df["vol_63d"].replace(0, np.nan)

    # --- trend / position in range ---
    df["ma_50"] = g["close"].transform(lambda s: s.rolling(50, min_periods=25).mean())
    df["ma_200"] = g["close"].transform(lambda s: s.rolling(200, min_periods=100).mean())
    df["px_to_ma50"] = df["close"] / df["ma_50"] - 1
    df["px_to_ma200"] = df["close"] / df["ma_200"] - 1
    high_252 = g["close"].transform(lambda s: s.rolling(252, min_periods=126).max())
    df["pct_off_52w_high"] = df["close"] / high_252 - 1

    # --- liquidity / attention ---
    df["dv_21d"] = g["dollar_volume"].transform(lambda s: s.rolling(21, min_periods=10).median())
    dv_63 = g["dollar_volume"].transform(lambda s: s.rolling(63, min_periods=30).median())
    # Volume surprise: recent turnover against its own longer norm, in logs so
    # a 2x and a 0.5x are symmetric. Both sides are guarded -- a halted or
    # untraded stretch puts a genuine zero on either, and log(0) would seed
    # -inf through every downstream rank.
    df["volume_surprise"] = np.log(
        df["dv_21d"].where(df["dv_21d"] > 0) / dv_63.where(dv_63 > 0)
    )

    # --- gap behaviour ---
    # Same quantity as the overnight leg; kept under its original name so
    # gap_vol_21d and anything downstream still reads.
    df["overnight_gap"] = df["ret_overnight"]
    df["gap_vol_21d"] = g["overnight_gap"].transform(
        lambda s: s.rolling(21, min_periods=10).std()
    )
    return df


# Close-to-close signals: the original set.
FEATURES_CLOSE = [
    "mom_21d", "mom_63d", "mom_126d", "mom_252d", "mom_63d_riskadj",
    "reversal_5d", "vol_21d", "vol_63d",
    "px_to_ma50", "px_to_ma200", "pct_off_52w_high",
    "volume_surprise", "gap_vol_21d",
]

# The same signals cut into their overnight and intraday legs.
FEATURES_COMPONENT = [
    "mom_on_21d", "mom_on_63d", "mom_on_126d", "mom_on_252d",
    "mom_id_21d", "mom_id_63d", "mom_id_126d", "mom_id_252d",
    "reversal_5d_on", "reversal_5d_id", "tug_63d",
]

FEATURES = FEATURES_CLOSE + FEATURES_COMPONENT


def add_forward_returns(df: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """Forward returns, entered at the NEXT open after the signal date.

    Signals are computed from a session's close, so the earliest realistic
    entry is the following open. Measuring from the signal-date close instead
    would book a move that was already unavailable -- the single most common
    way a daily-bar backtest overstates itself.
    """
    out = df.sort_values(["symbol", "date"]).copy()
    g = _by_symbol(out)
    entry = g["open"].shift(-1)
    out["entry_price"] = entry
    for h in horizons:
        exit_price = _by_symbol(out.assign(_o=entry))["_o"].shift(-h)
        out[f"fwd_{h}d"] = exit_price / entry - 1
    return out


def add_component_forward_returns(df: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    """Forward returns of each leg separately: `fwd_on_{h}d`, `fwd_id_{h}d`.

    `add_forward_returns` measures open-to-open, which is what a tradeable
    close-to-close signal earns. It cannot express the paper's actual claim,
    which is that overnight continuation predicts *overnight* returns while
    intraday continuation predicts intraday ones, with opposite signs. Testing
    that needs the legs kept apart on the forward side too.

    These are accounting quantities, not tradeable ones. Capturing `fwd_on_5d`
    means buying each close and selling each open for five sessions, which is
    ten auction crossings a week -- and the bid/ask in this data is stamped
    ~17:15, so nothing here can price that. Read them as evidence about where
    the drift lives, then let `fwd_{h}d` decide whether it is reachable.
    """
    out = df.sort_values(["symbol", "date"]).copy()
    missing = {"ret_overnight", "ret_intraday"} - set(out.columns)
    if missing:
        raise ValueError(f"call add_features first; missing {sorted(missing)}")

    for tag, col in (("on", "ret_overnight"), ("id", "ret_intraday")):
        log_leg = np.log1p(out[col].clip(lower=-0.99))
        # Cumulative sums let any horizon be differenced out in one pass, but
        # a single NaN would poison every later value, so the legs are filled
        # and counted separately -- a window is only valid if it holds `h`
        # real observations.
        cum = _by_symbol(out.assign(_c=log_leg.fillna(0.0)))["_c"].cumsum()
        seen = _by_symbol(out.assign(_n=log_leg.notna().astype("int64")))["_n"].cumsum()

        for h in horizons:
            ahead = _by_symbol(out.assign(_a=cum))["_a"].shift(-h) - cum
            counted = _by_symbol(out.assign(_k=seen))["_k"].shift(-h) - seen
            out[f"fwd_{tag}_{h}d"] = np.expm1(ahead).where(counted == h)

    return out
