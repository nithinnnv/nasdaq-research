"""Split detection and back-adjustment for unadjusted ThetaData stock bars.

ThetaData serves raw prices and exposes a Splits endpoint only from the VALUE
tier up, so on FREE the split dates have to be recovered from the series
itself. Getting this wrong is not a rounding error: NVDA's 10:1 reads as a
-90% day, which would dominate any momentum or drawdown screen it lands in.

Detection uses two signals that must agree:

  price:  prev_close / open ~= k        (a k:1 split divides the price by k)
  volume: median(vol after) / median(vol before) ~= k   (share count x k)

Volume is compared as a median over a window either side, never day-over-day.
The session a split takes effect often carries its own volume spike, and a
single-day ratio picks that up instead of the permanent change in share
count -- which is what actually distinguishes a split.

Requiring both is what separates a split from a crash. A stock that genuinely
halves overnight also has a price ratio of 2.0, but its volume does not
reliably double -- and a stock whose volume happens to double rarely also
prints an exact 2.0 price ratio. Demanding the two agree, on a ratio close to
a simple fraction, is a much narrower target than either test alone.

Known ratios are enumerated rather than accepting any rational: real splits
cluster on a short list, and allowing arbitrary p/q turns every noisy
low-volume gap into a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List

import pandas as pd

# Forward splits as k (price divided by k), reverses as fractions.
# Ratios below ~1.5 are deliberately excluded. 5:4 and 6:5 splits are
# vanishingly rare, and they sit exactly where ordinary earnings gaps live --
# admitting them turned every 15-20% gap (MSTR 2024-03-19, SMCI 2024-08-05)
# into a false split. Missing a hypothetical 6:5 costs far less than
# corrupting real gaps.
KNOWN_RATIOS = [
    1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0,
    0.5, 1 / 3, 0.25, 0.2, 0.1, 1 / 15, 0.05,  # reverse splits
    1 / 30, 1 / 50, 1 / 100,
]

# Price tolerance scales with the ratio, because the prior that a move *is* a
# split does too. Nothing legitimately gaps 10x overnight, so a large ratio can
# be matched loosely and still be near-certain (MSTR's 10:1 landed 4.3% off
# because the stock also gapped +4% that night). A 2:1 candidate is a different
# story -- a real -50% crash produces exactly 2.0 -- so small ratios must match
# almost exactly before volume is even consulted.
PRICE_TOL_LARGE = 0.055   # |ratio| >= 3
PRICE_TOL_SMALL = 0.025   # everything else
LARGE_RATIO = 3.0


def _price_tol(ratio: float) -> float:
    k = ratio if ratio >= 1 else 1 / ratio
    return PRICE_TOL_LARGE if k >= LARGE_RATIO else PRICE_TOL_SMALL
# Volume is tested against the two competing hypotheses rather than as a
# tolerance band around the nominal ratio. A symmetric band fails on small
# ratios: +/-62% around 2.0 spans [0.76, 3.24], which admits volume_ratio=1.0 --
# no share-count change whatsoever -- and so lets a -50% crash through as a 2:1
# split. Requiring the observed step to fall past the geometric midpoint of 1
# and `ratio` asks the right question: is this nearer "shares multiplied" than
# "nothing happened"? Confirmed splits clear it with room (10:1 events ran
# 5.9x-8.6x against a 3.16x bar; PANW's 2:1 ran 1.84x against 1.41x).
VOLUME_SANITY_MULTIPLE = 3.0  # guards against unrelated volume explosions
VOLUME_WINDOW = 10  # sessions each side of the event
MIN_PRICE = 1.0    # sub-$1 tick noise produces meaningless ratios
MIN_VOLUME = 10_000


@dataclass(frozen=True)
class SplitEvent:
    symbol: str
    effective: date   # first session trading at the new price
    ratio: float      # k, where prices before this date divide by k
    price_ratio: float
    volume_ratio: float

    @property
    def label(self) -> str:
        return f"{self.ratio:g}:1" if self.ratio >= 1 else f"1:{1 / self.ratio:g}"


def _volume_supports_split(volume_ratio: float, ratio: float) -> bool:
    """Does the volume step favour a split over 'no share-count change'?

    Threshold is the geometric midpoint of 1.0 (no split) and `ratio` (split),
    which is where the two hypotheses are equally likely on a log scale.
    """
    if volume_ratio <= 0:
        return False
    midpoint = ratio ** 0.5
    if ratio > 1:
        return midpoint < volume_ratio < ratio * VOLUME_SANITY_MULTIPLE
    return ratio / VOLUME_SANITY_MULTIPLE < volume_ratio < midpoint


def _nearest_known(value: float) -> tuple[float, float]:
    best = min(KNOWN_RATIOS, key=lambda k: abs(value - k) / k)
    return best, abs(value - best) / best


def detect_splits(df: pd.DataFrame, symbol: str | None = None) -> List[SplitEvent]:
    """Find split events in one symbol's daily bars (as returned by StockEODClient)."""
    if df.empty or len(df) < 2:
        return []

    symbol = symbol or (df["symbol"].iloc[0] if "symbol" in df else "?")
    d = df.sort_values("date").reset_index(drop=True)
    prev_close = d["close"].shift(1)
    vol = d["volume"]

    events: List[SplitEvent] = []
    for i in range(1, len(d)):
        pc, po = prev_close[i], d["open"][i]
        if not (pc > MIN_PRICE and po > MIN_PRICE):
            continue
        # Medians either side, excluding the event day itself.
        before = vol[max(0, i - VOLUME_WINDOW):i]
        after = vol[i + 1:i + 1 + VOLUME_WINDOW]
        if len(before) < 3 or len(after) < 3:
            continue
        v_before, v_after = before.median(), after.median()
        if v_before < MIN_VOLUME or v_after < MIN_VOLUME:
            continue

        price_ratio = pc / po
        ratio, price_err = _nearest_known(price_ratio)
        if price_err > _price_tol(ratio):
            continue

        # Share count moves by the same factor, permanently.
        volume_ratio = v_after / v_before
        if not _volume_supports_split(volume_ratio, ratio):
            continue

        events.append(
            SplitEvent(symbol, d["date"][i], ratio, round(price_ratio, 4), round(volume_ratio, 3))
        )
    return events


def adjust_for_splits(df: pd.DataFrame, events: List[SplitEvent]) -> pd.DataFrame:
    """Back-adjust prices and volumes so the series is continuous.

    Convention matches every vendor's 'adjusted close': the most recent segment
    keeps its real prices and history is restated, so today's number is the one
    you could actually trade at.
    """
    if df.empty or not events:
        return df.copy()

    out = df.sort_values("date").reset_index(drop=True).copy()
    for ev in sorted(events, key=lambda e: e.effective, reverse=True):
        mask = out["date"] < ev.effective
        for col in ("open", "high", "low", "close", "bid", "ask"):
            if col in out:
                out.loc[mask, col] = out.loc[mask, col] / ev.ratio
        if "volume" in out:
            out.loc[mask, "volume"] = (out.loc[mask, "volume"] * ev.ratio).round().astype("int64")
    return out
