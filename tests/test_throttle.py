"""Rate limiter tests, on a fake clock.

The limiter is the difference between a backfill that finishes and one that
dies partway, but every interesting case involves waiting out a 60-second
window -- so the clock is injected rather than endured. `FakeClock.sleep`
advances time instead of spending it, which makes the sliding-window
behaviour assertable in microseconds.
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import throttle  # noqa: E402
from data.throttle import RATE_LIMITS, RateLimiter  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(throttle, "time", fake)
    return fake


def test_capacity_holds_back_from_the_published_cap():
    """90% of the cap, because our window and the server's are not aligned."""
    assert RateLimiter(per_minute=30).capacity == 27
    assert RateLimiter(per_minute=60).capacity == 54


def test_capacity_never_drops_below_one():
    """A safety factor that floors to zero would deadlock the backfill."""
    assert RateLimiter(per_minute=1).capacity == 1


def test_tier_limits_are_ordered_by_tier():
    tiers = ["FREE", "VALUE", "STANDARD", "PRO"]
    limits = [RATE_LIMITS[t] for t in tiers]
    assert limits == sorted(limits)


def test_calls_under_capacity_never_wait(clock):
    limiter = RateLimiter(per_minute=30)
    for _ in range(limiter.capacity):
        limiter.acquire()
    assert clock.sleeps == []


def test_exceeding_capacity_waits_out_the_window(clock):
    limiter = RateLimiter(per_minute=30)
    for _ in range(limiter.capacity):
        limiter.acquire()

    limiter.acquire()  # one over
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] == pytest.approx(60.0, abs=0.1)


def test_window_slides_so_old_calls_stop_counting(clock):
    limiter = RateLimiter(per_minute=30)
    for _ in range(limiter.capacity):
        limiter.acquire()

    clock.t += 61.0  # the whole first burst ages out
    for _ in range(limiter.capacity):
        limiter.acquire()
    assert clock.sleeps == []


def test_concurrent_acquires_do_not_lose_updates():
    """The limiter is shared across threads on VALUE+; the lock has to hold."""
    limiter = RateLimiter(per_minute=1000)
    threads = [threading.Thread(target=lambda: [limiter.acquire() for _ in range(10)])
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(limiter._calls) == 80
