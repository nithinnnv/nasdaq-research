"""Resumable universe backfill.

Cost is dominated by the FREE tier's 30 req/min cap and the endpoint's 365-day
window limit: a full 2023-06 -> today pull is 4 requests per symbol, so ~3,000
symbols is ~12,000 requests, ~7.5 hours. That is long enough that the run must
survive interruption, hence the per-symbol Parquet cache and the skip-if-present
check -- re-running resumes rather than restarts.

Two phases, because most of the universe is not worth full history:

  --phase probe    one request per symbol, most recent window only (~1.9h).
                   Enough to measure liquidity and drop what is untradeable.
  --phase full     remaining windows, for symbols that passed the liquidity
                   screen. Typically a few hundred names, so ~1h rather than 7.

Run probe first, screen, then full. Going straight to --phase full works but
spends most of its time on microcaps you will discard.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.theta_stock import StockEODClient, SubscriptionError, ThetaError  # noqa: E402
from data.throttle import RATE_LIMITS, RateLimiter  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / "cache" / "bars"


def bars_path(symbol: str) -> Path:
    # Symbols contain '/' and '.' (".PR.S/WI"); keep them off the filesystem.
    safe = symbol.replace("/", "_").replace(".", "_")
    return CACHE / f"{safe}.parquet"


def load_universe(path: Path) -> list[str]:
    return pd.read_parquet(path)["symbol"].tolist()


def liquidity_screen(min_dollar_volume: float, min_price: float, min_days: int) -> pd.DataFrame:
    """Median daily dollar volume per symbol, from whatever is already cached."""
    rows = []
    for f in sorted(CACHE.glob("*.parquet")):
        df = pd.read_parquet(f)
        if len(df) < min_days:
            continue
        dollar = (df["close"] * df["volume"]).median()
        rows.append({
            "symbol": df["symbol"].iloc[0],
            "days": len(df),
            "median_close": df["close"].median(),
            "median_dollar_volume": dollar,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    keep = (out["median_dollar_volume"] >= min_dollar_volume) & (out["median_close"] >= min_price)
    out["passes"] = keep
    return out.sort_values("median_dollar_volume", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="cache/universe.parquet")
    ap.add_argument("--phase", choices=["probe", "full"], default="probe")
    ap.add_argument("--start", default="2023-06-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--tier", default="FREE", choices=list(RATE_LIMITS))
    ap.add_argument("--symbols", help="comma-separated override instead of the universe file")
    ap.add_argument("--limit", type=int, help="stop after N symbols (for a timed trial)")
    ap.add_argument("--refresh", action="store_true", help="re-fetch symbols already cached")
    args = ap.parse_args()

    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if args.phase == "probe":
        # Most recent single window only. 364 days, not 365 and not a calendar
        # year: an inclusive 366-day span tips over the endpoint's cap and
        # silently doubles the cost of the whole phase.
        start = max(start, end - timedelta(days=364))

    symbols = (args.symbols.split(",") if args.symbols
               else load_universe(Path(args.universe)))
    if args.limit:
        symbols = symbols[:args.limit]

    CACHE.mkdir(parents=True, exist_ok=True)
    client = StockEODClient(tier=args.tier, limiter=RateLimiter(RATE_LIMITS[args.tier]))

    todo = [s for s in symbols if args.refresh or not bars_path(s).exists()]
    windows = max(1, (end - start).days // 365 + 1)
    est_min = len(todo) * windows / RateLimiter(RATE_LIMITS[args.tier]).capacity
    print(f"phase={args.phase} tier={args.tier} {start} -> {end}")
    print(f"{len(todo):,} symbols to fetch ({len(symbols) - len(todo):,} already cached), "
          f"~{windows} req each -> ~{est_min / 60:.1f}h at {RATE_LIMITS[args.tier]} req/min")

    t0, done, empty, failed = time.time(), 0, 0, []
    for i, sym in enumerate(todo, 1):
        try:
            df = client.fetch(sym, start, end)
        except SubscriptionError as exc:
            # Entitlement is a property of the request window, not the symbol --
            # every remaining symbol will fail the same way. Stop, don't grind.
            print(f"\nSubscription limit hit at {sym}: {exc}")
            print(f"Requested history needs the {exc.required_tier} tier.")
            return 2
        except ThetaError as exc:
            failed.append((sym, str(exc)[:80]))
            continue

        if df.empty:
            empty += 1
        else:
            df.to_parquet(bars_path(sym), index=False)
            done += 1

        if i % 50 == 0 or i == len(todo):
            rate = i / max(1e-9, time.time() - t0) * 60
            eta = (len(todo) - i) / max(1e-9, rate)
            print(f"  [{i:>5}/{len(todo)}] saved={done} empty={empty} failed={len(failed)} "
                  f"{rate:.0f} sym/min  ETA {eta:.0f}m", flush=True)

    print(f"\ndone in {(time.time() - t0) / 60:.1f}m -- saved={done} empty={empty} failed={len(failed)}")
    for sym, err in failed[:10]:
        print(f"  FAILED {sym}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
