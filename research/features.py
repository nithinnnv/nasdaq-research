"""Cross-sectional features for a swing (days-to-weeks) horizon.

Every feature at date t is computed from data at or before t. That is enforced
structurally -- each one is a rolling or shifted operation within a symbol,
never a whole-sample statistic -- because the usual way a backtest lies to you
is a feature that quietly peeked forward.

The momentum family skips the most recent 5 sessions. Short-horizon momentum
inverts (last week's winners tend to give it back), so leaving the gap in
separates the two effects instead of letting them cancel inside one number.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKIP_DAYS = 5  # short-term reversal window excluded from momentum


def _by_symbol(panel: pd.DataFrame) -> pd.core.groupby.DataFrameGroupBy:
    return panel.groupby("symbol", group_keys=False, observed=True)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach the feature set. Input is the long panel from dataset.load_panel."""
    df = panel.sort_values(["symbol", "date"]).copy()
    g = _by_symbol(df)

    df["ret_1d"] = g["close"].pct_change()
    df["dollar_volume"] = df["close"] * df["volume"]

    # --- momentum, skipping the reversal window ---
    for window in (21, 63, 126, 252):
        past = g["close"].shift(SKIP_DAYS)
        df[f"mom_{window}d"] = past / _by_symbol(df.assign(_p=past))["_p"].shift(window) - 1

    # --- short-term reversal (the effect momentum deliberately excludes) ---
    df["reversal_5d"] = g["close"].pct_change(SKIP_DAYS)

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
    df["overnight_gap"] = df["open"] / g["close"].shift(1) - 1
    df["gap_vol_21d"] = g["overnight_gap"].transform(
        lambda s: s.rolling(21, min_periods=10).std()
    )
    return df


FEATURES = [
    "mom_21d", "mom_63d", "mom_126d", "mom_252d", "mom_63d_riskadj",
    "reversal_5d", "vol_21d", "vol_63d",
    "px_to_ma50", "px_to_ma200", "pct_off_52w_high",
    "volume_surprise", "gap_vol_21d",
]


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
