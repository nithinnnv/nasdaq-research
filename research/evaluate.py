"""Cross-sectional evaluation: does a feature rank next period's winners?

The unit of analysis is a date, not a stock. On each date every name is ranked
by the feature and that ranking is compared with realised forward returns
(Spearman, so the result depends on order rather than on outliers). The series
of daily rank correlations -- the information coefficient -- is the signal.

Two statistical traps this module refuses to walk into:

1. **Overlapping windows.** A 20-day forward return sampled daily reuses 19 of
   20 days with its neighbour, so consecutive ICs are mechanically correlated.
   Treating them as independent inflates the t-statistic by roughly sqrt(h).
   `ic_summary` therefore reports a Newey-West corrected t-stat alongside the
   naive one, and `sample_non_overlapping` is available when you want
   genuinely independent observations.

2. **Breadth.** A rank correlation over 8 names is noise. Dates with fewer
   than `min_names` are dropped rather than averaged in.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

MIN_NAMES = 20


def _spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation, computed directly rather than via pandas.

    pandas routes method="spearman" through scipy, which is a heavy dependency
    for an identity: Spearman is Pearson applied to ranks. Average ranks for
    ties, which is the standard convention and matters here because feature
    values tie constantly at bucket edges.
    """
    if len(a) < 3:
        return float("nan")
    ra, rb = a.rank(method="average"), b.rank(method="average")
    if ra.std(ddof=0) == 0 or rb.std(ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class ICSummary:
    feature: str
    horizon: int
    n_dates: int
    mean_ic: float
    std_ic: float
    ic_ir: float          # mean / std -- the signal's consistency
    t_stat_naive: float
    t_stat_nw: float      # Newey-West, corrected for the overlap
    hit_rate: float       # share of dates with IC > 0

    def as_row(self) -> dict:
        return asdict(self)


def cross_sectional_ic(df: pd.DataFrame, feature: str, fwd_col: str,
                       min_names: int = MIN_NAMES) -> pd.Series:
    """Spearman rank IC per date."""
    sub = df[["date", feature, fwd_col]].dropna()
    if sub.empty:
        return pd.Series(dtype=float)
    counts = sub.groupby("date")["date"].transform("size")
    sub = sub[counts >= min_names]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.groupby("date").apply(
        lambda g: _spearman(g[feature], g[fwd_col]),
        include_groups=False,
    ).dropna()


def _newey_west_tstat(x: pd.Series, lags: int) -> float:
    """t-stat for the mean of an autocorrelated series (Newey-West/Bartlett)."""
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 3:
        return float("nan")
    demeaned = x - x.mean()
    var = (demeaned @ demeaned) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cov = (demeaned[lag:] @ demeaned[:-lag]) / n
        var += 2.0 * weight * cov
    if var <= 0:
        return float("nan")
    return float(x.mean() / np.sqrt(var / n))


def ic_summary(df: pd.DataFrame, feature: str, horizon: int,
               min_names: int = MIN_NAMES) -> Optional[ICSummary]:
    ic = cross_sectional_ic(df, feature, f"fwd_{horizon}d", min_names)
    if len(ic) < 20:
        return None
    mean, std = ic.mean(), ic.std(ddof=1)
    return ICSummary(
        feature=feature, horizon=horizon, n_dates=len(ic),
        mean_ic=float(mean), std_ic=float(std),
        ic_ir=float(mean / std) if std else float("nan"),
        t_stat_naive=float(mean / std * np.sqrt(len(ic))) if std else float("nan"),
        # Overlap runs for `horizon` days, so that is the lag structure to correct.
        t_stat_nw=_newey_west_tstat(ic, lags=horizon),
        hit_rate=float((ic > 0).mean()),
    )


def decile_returns(df: pd.DataFrame, feature: str, horizon: int,
                   n_buckets: int = 10, min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Mean forward return by feature bucket, ranked within each date."""
    fwd = f"fwd_{horizon}d"
    sub = df[["date", "symbol", feature, fwd]].dropna()
    counts = sub.groupby("date")["date"].transform("size")
    sub = sub[counts >= max(min_names, n_buckets * 2)]
    if sub.empty:
        return pd.DataFrame()
    sub = sub.copy()
    sub["bucket"] = sub.groupby("date")[feature].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False, duplicates="drop")
    )
    out = sub.groupby("bucket").agg(
        mean_fwd=(fwd, "mean"), median_fwd=(fwd, "median"), n=(fwd, "size")
    ).reset_index()
    out["mean_fwd_bps"] = out["mean_fwd"] * 10_000
    return out


def long_short_spread(df: pd.DataFrame, feature: str, horizon: int,
                      n_buckets: int = 10) -> float:
    d = decile_returns(df, feature, horizon, n_buckets)
    if d.empty or len(d) < n_buckets:
        return float("nan")
    return float(d["mean_fwd"].iloc[-1] - d["mean_fwd"].iloc[0])


def sample_non_overlapping(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep every h-th date so forward windows do not share days."""
    dates = np.sort(df["date"].unique())
    return df[df["date"].isin(dates[::horizon])]


def walk_forward_split(df: pd.DataFrame, train=0.5, validation=0.25) -> dict[str, pd.DataFrame]:
    """Chronological split. Never random: shuffling dates leaks the future into
    the past through overlapping windows and shared market regimes."""
    dates = np.sort(df["date"].unique())
    i, j = int(len(dates) * train), int(len(dates) * (train + validation))
    return {
        "train": df[df["date"].isin(dates[:i])],
        "validation": df[df["date"].isin(dates[i:j])],
        "test": df[df["date"].isin(dates[j:])],
    }
