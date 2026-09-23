"""Load cached bars into a split-adjusted panel.

One frame, indexed (date, symbol), which is the shape every cross-sectional
step downstream wants: rank a feature across names on a date, form deciles,
measure what happened next.

On split adjustment and lookahead: back-adjusting uses the knowledge that a
split occurred, which you did not have beforehand. That is harmless for the
*return* series -- adjustment only repairs the one discontinuous day, and every
other return is unchanged -- but it does restate historical price *levels*. So
returns and anything derived from them are point-in-time honest; an absolute
price filter ("was it above $5 back then") is not, and is applied on raw
prices before adjustment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from data.splits import adjust_for_splits, detect_splits

CACHE = Path(__file__).resolve().parent.parent / "cache" / "bars"


def load_symbol(symbol: str, adjust: bool = True) -> pd.DataFrame:
    path = CACHE / f"{symbol.replace('/', '_').replace('.', '_')}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if df.empty:
        return df
    df = df.sort_values("date").reset_index(drop=True)
    # Raw close is kept so price-level filters can stay point-in-time.
    df["raw_close"] = df["close"]
    if adjust:
        events = detect_splits(df, symbol)
        if events:
            df = adjust_for_splits(df, events)
            df["split_events"] = len(events)
    return df


def load_panel(symbols: Optional[Iterable[str]] = None, adjust: bool = True) -> pd.DataFrame:
    """Long panel of every cached symbol: one row per (date, symbol)."""
    if symbols is None:
        symbols = [p.stem for p in sorted(CACHE.glob("*.parquet"))]
    frames = [d for s in symbols if not (d := load_symbol(s, adjust)).empty]
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)
