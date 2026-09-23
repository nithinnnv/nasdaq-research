"""EOD client tests, driven by a stub session instead of the Terminal.

Everything worth testing in this module is a *policy* decision -- which HTTP
statuses are fatal, which are ordinary, which are worth another attempt, and
how a request too wide for the API gets split. None of that needs a live
terminal, and none of it was covered before: the client was the one place
where a wrong call silently costs either data or an hour of rate limit.

The two policies most worth pinning down:

  - A 403 is an entitlement answer, not a transient failure. Retrying it
    burns three slots of a 30/minute budget to be told the same thing.
  - A 472 ("valid request, nothing to return") is *normal*. Recent IPOs and
    delisted tickers produce it constantly in a universe scan, and treating
    it as an error would abort the scan on its first thin name.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.throttle import RateLimiter  # noqa: E402
from data.theta_stock import (  # noqa: E402
    MAX_WINDOW_DAYS, StockEODClient, SubscriptionError, ThetaError, _windows,
)


class StubResponse:
    def __init__(self, status_code: int, text: str = ""):
        self.status_code = status_code
        self.text = text


class StubSession:
    """Stands in for requests.Session, returning a scripted reply per call."""

    def __init__(self, *responses, raise_first: int = 0):
        self._responses = list(responses)
        self._raise_first = raise_first
        self.calls: list[dict] = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}})
        if self._raise_first > 0:
            self._raise_first -= 1
            raise requests.ConnectionError("terminal not reachable")
        if len(self._responses) == 1:
            return self._responses[0]
        return self._responses.pop(0)


def csv_for(days: list[date]) -> str:
    """The subset of EOD columns `_parse_eod_csv` actually reads."""
    header = "created,open,high,low,close,volume,bid,ask,count"
    rows = [f"{d.isoformat()}T17:15:00,10.0,10.5,9.5,10.2,1000000,10.19,10.21,500"
            for d in days]
    return "\n".join([header, *rows]) + "\n"


def client(*responses, tier="FREE", retries=3, raise_first=0) -> StockEODClient:
    c = StockEODClient(tier=tier, retries=retries,
                       limiter=RateLimiter(per_minute=1_000_000))
    c._session = StubSession(*responses, raise_first=raise_first)
    return c


# --- status handling -------------------------------------------------------

def test_200_parses_into_bars():
    days = [date(2024, 1, 2), date(2024, 1, 3)]
    c = client(StubResponse(200, csv_for(days)))
    df = c.fetch("AAPL", days[0], days[-1])

    assert list(df["date"]) == days
    assert (df["symbol"] == "AAPL").all()
    assert df["close"].tolist() == [10.2, 10.2]


def test_472_no_data_is_empty_not_an_error():
    c = client(StubResponse(472, "no data"))
    assert c.fetch("THIN", date(2024, 1, 2), date(2024, 1, 3)).empty


def test_404_unknown_symbol_is_empty_not_an_error():
    c = client(StubResponse(404, "not found"))
    assert c.fetch("GONE", date(2024, 1, 2), date(2024, 1, 3)).empty


def test_403_raises_subscription_error_naming_the_tier():
    c = client(StubResponse(403, "Your subscription does not include VALUE data"))
    with pytest.raises(SubscriptionError) as exc:
        c.fetch("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert exc.value.required_tier == "VALUE"


def test_403_is_not_retried():
    """An entitlement failure is an answer; retrying it just spends the budget."""
    c = client(StubResponse(403, "requires STANDARD"), retries=3)
    with pytest.raises(SubscriptionError):
        c.fetch("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert len(c._session.calls) == 1


def test_400_is_not_retried():
    c = client(StubResponse(400, "bad window"), retries=3)
    with pytest.raises(ThetaError):
        c.fetch("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert len(c._session.calls) == 1


def test_5xx_is_retried_then_gives_up():
    c = client(StubResponse(503, "unavailable"), retries=3)
    with pytest.raises(ThetaError):
        c.fetch("AAPL", date(2024, 1, 2), date(2024, 1, 3))
    assert len(c._session.calls) == 3


def test_connection_error_recovers_on_a_later_attempt():
    days = [date(2024, 1, 2)]
    c = client(StubResponse(200, csv_for(days)), retries=3, raise_first=2)
    assert len(c.fetch("AAPL", days[0], days[0])) == 1
    assert len(c._session.calls) == 3


# --- windowing -------------------------------------------------------------

def test_windows_never_exceed_the_api_cap_and_tile_the_range():
    start, end = date(2021, 1, 1), date(2024, 6, 15)
    windows = list(_windows(start, end))

    assert windows[0][0] == start and windows[-1][1] == end
    assert all((b - a).days + 1 <= MAX_WINDOW_DAYS for a, b in windows)
    # Consecutive and gapless: each window resumes the day after the last.
    assert all(nxt[0] == prev[1] + timedelta(days=1)
               for prev, nxt in zip(windows, windows[1:]))


def test_multi_year_request_is_chunked_transparently():
    start, end = date(2023, 6, 1), date(2025, 6, 1)
    c = client(StubResponse(200, csv_for([date(2024, 1, 2)])))
    c.fetch("AAPL", start, end)

    assert len(c._session.calls) == 3  # 731 days / 365
    sent = [call["params"] for call in c._session.calls]
    assert sent[0]["start_date"] == "20230601"
    assert sent[-1]["end_date"] == "20250601"


def test_boundary_duplicate_is_collapsed():
    """A date returned by two windows must not double-count in a return series."""
    shared = date(2024, 1, 2)
    c = client(StubResponse(200, csv_for([shared])))  # every window returns it
    df = c.fetch("AAPL", date(2023, 6, 1), date(2025, 6, 1))
    assert len(df) == 1


# --- tier clamping ---------------------------------------------------------

def test_request_below_tier_history_is_clamped_not_rejected():
    """FREE starts 2023-06-01; asking for 2020 trims rather than 403s."""
    c = client(StubResponse(200, csv_for([date(2024, 1, 2)])))
    c.fetch("AAPL", date(2020, 1, 1), date(2024, 1, 2))
    assert c._session.calls[0]["params"]["start_date"] == "20230601"


def test_window_entirely_below_tier_history_costs_no_request():
    c = client(StubResponse(200, ""))
    df = c.fetch("AAPL", date(2019, 1, 1), date(2020, 1, 1))
    assert df.empty
    assert c._session.calls == []


def test_history_start_tracks_the_tier():
    assert StockEODClient(tier="FREE").history_start == date(2023, 6, 1)
    assert StockEODClient(tier="PRO").history_start == date(2012, 6, 1)
