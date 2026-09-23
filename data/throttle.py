"""Client-side rate limiting for the ThetaData Terminal.

The FREE stock tier is capped at 30 requests/minute and allows 1 concurrent
request; exceeding either gets requests rejected rather than queued. A full
universe backfill is ~5,600 symbols, so the limiter is what makes the
difference between a 3-hour run that completes and one that dies partway.

Upgrading tiers raises both caps (VALUE: 2 concurrent, STANDARD: 4, PRO: 8) --
change RATE_LIMITS rather than hunting for hardcoded sleeps.
"""
import threading
import time
from collections import deque

# requests/minute by subscription tier, per ThetaData's published limits.
RATE_LIMITS = {"FREE": 30, "VALUE": 60, "STANDARD": 120, "PRO": 240}


class RateLimiter:
    """Sliding-window limiter. Thread-safe so a future concurrent fetcher
    (VALUE+ allows >1 in flight) can share one instance."""

    def __init__(self, per_minute: int = RATE_LIMITS["FREE"], safety: float = 0.9):
        # Run at 90% of the published cap: the window the server measures is not
        # perfectly aligned with ours, and a burst that lands on a window edge
        # reads as over-limit even when our own count says otherwise.
        self.capacity = max(1, int(per_minute * safety))
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] >= 60.0:
                    self._calls.popleft()
                if len(self._calls) < self.capacity:
                    self._calls.append(now)
                    return
                wait = 60.0 - (now - self._calls[0]) + 0.01
            time.sleep(wait)
