"""Daily stock bars from a local ThetaData Terminal (v3 API).

The Terminal authenticates itself, so no credentials live here -- it must be
running (`~/ThetaTerminal/start.sh`) and serving on :25503.

Endpoint used:
  GET /v3/stock/history/eod  -- daily OHLCV + closing NBBO

Tier notes (verified against the live terminal, matches ThetaData's published
tiers). The FREE stock tier serves *only* this endpoint and only from
2023-06-01; intraday OHLC/quote need VALUE, trades need STANDARD, and history
before 2021-01-01 needs STANDARD or PRO. Requests outside entitlement return
403 with a message naming the required tier, which is surfaced as
SubscriptionError rather than being retried -- retrying an entitlement failure
just burns the rate limit.

Prices are UNADJUSTED. A split shows as a raw discontinuity in this series
(NVDA closes 1208.88 on 2024-06-07 and opens 120.45 on 2024-06-10). Nothing in
this module corrects for that; see data/splits.py.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import requests

from data.throttle import RateLimiter

DEFAULT_BASE_URL = "http://127.0.0.1:25503"

# Earliest date each tier may request for stock EOD. Probed directly against
# the terminal rather than taken on faith from the docs.
TIER_HISTORY_START = {
    "FREE": date(2023, 6, 1),
    "VALUE": date(2021, 1, 1),
    "STANDARD": date(2016, 1, 1),
    "PRO": date(2012, 6, 1),
}

NO_DATA_STATUS = 472  # ThetaData's "request valid, nothing to return"

# The endpoint rejects any window wider than 365 days with a 400. `fetch`
# splits longer requests transparently, which is why a multi-year backfill
# costs ceil(years) requests per symbol rather than one.
MAX_WINDOW_DAYS = 365


class ThetaError(RuntimeError):
    pass


class SubscriptionError(ThetaError):
    """Raised on 403. Carries the tier the endpoint actually needs so callers
    can report 'upgrade to X' instead of a bare HTTP error."""

    def __init__(self, message: str, required_tier: Optional[str] = None):
        super().__init__(message)
        self.required_tier = required_tier


@dataclass
class StockEODClient:
    base_url: str = DEFAULT_BASE_URL
    tier: str = "FREE"
    timeout: int = 60
    retries: int = 3
    limiter: Optional[RateLimiter] = None

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.limiter = self.limiter or RateLimiter()
        self._session = requests.Session()

    @property
    def history_start(self) -> date:
        return TIER_HISTORY_START[self.tier]

    def _fetch_window(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Daily bars for one symbol, inclusive of both endpoints.

        Returns an empty frame (not an error) when the symbol simply has no
        data in the window -- delisted tickers and recent IPOs are normal in a
        universe scan and must not abort it.
        """
        # Clamping instead of erroring keeps a universe-wide backfill from
        # dying on its first symbol when the caller asks for more history than
        # the subscription covers.
        start = max(start, self.history_start)
        if start > end:
            return _empty_frame()

        params = {
            "symbol": symbol,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
        }
        url = f"{self.base_url}/v3/stock/history/eod"

        last_exc: Optional[Exception] = None
        for attempt in range(self.retries):
            self.limiter.acquire()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                continue

            if resp.status_code == 200:
                return _parse_eod_csv(resp.text, symbol)
            if resp.status_code == NO_DATA_STATUS:
                return _empty_frame()
            if resp.status_code == 403:
                raise SubscriptionError(resp.text.strip()[:300], _required_tier(resp.text))
            if resp.status_code == 404:
                return _empty_frame()
            if resp.status_code == 400:
                raise ThetaError(f"{symbol}: HTTP 400: {resp.text.strip()[:200]}")
            # 5xx and anything else unexpected: worth another attempt.
            last_exc = ThetaError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        raise ThetaError(f"{symbol}: failed after {self.retries} attempts") from last_exc


def _required_tier(body: str) -> Optional[str]:
    for tier in ("PROFESSIONAL", "PRO", "STANDARD", "VALUE"):
        if tier in body.upper():
            return tier
    return None


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume", "bid", "ask", "count"]
    ).astype({"open": "float64", "close": "float64", "volume": "int64"})


def _parse_eod_csv(text: str, symbol: str) -> pd.DataFrame:
    if not text.strip():
        return _empty_frame()
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return _empty_frame()

    # `created` is the row's stamp (~17:15 ET on the session); the calendar date
    # is what every downstream join keys on, so normalise it here once.
    df["date"] = pd.to_datetime(df["created"]).dt.date
    keep = ["date", "open", "high", "low", "close", "volume", "bid", "ask", "count"]
    out = df[[c for c in keep if c in df.columns]].copy()
    out.insert(0, "symbol", symbol)
    return out.sort_values("date").reset_index(drop=True)


def _windows(start: date, end: date, span: int = MAX_WINDOW_DAYS):
    """Split [start, end] into consecutive windows of at most `span` days."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=span - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def _fetch_chunked(self, symbol: str, start: date, end: date) -> pd.DataFrame:
    """Daily bars over an arbitrary span, transparently chunked to the API cap."""
    start = max(start, self.history_start)
    if start > end:
        return _empty_frame()

    frames = [self._fetch_window(symbol, a, b) for a, b in _windows(start, end)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_frame()
    out = pd.concat(frames, ignore_index=True)
    # Windows are disjoint by construction, but a vendor-side boundary
    # duplicate would silently double-count a day in any return series.
    return (out.drop_duplicates(subset=["date"], keep="last")
               .sort_values("date").reset_index(drop=True))


StockEODClient.fetch = _fetch_chunked
