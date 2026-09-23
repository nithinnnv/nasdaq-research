"""Signal survey over the cached universe.

Ranks every feature by its information coefficient, then re-checks the
survivors out-of-sample. The order matters: a feature is selected on train,
confirmed on validation, and only then looked at on test -- looking at test
first is how you end up fitting it.

Output is deliberately unglamorous. On ~3 years of a single rising market, the
honest result for most features is "indistinguishable from zero", and the
harness is built to say so rather than to find something.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.dataset import load_panel  # noqa: E402
from research.evaluate import (  # noqa: E402
    decile_returns, ic_summary, long_short_spread, walk_forward_split,
)
from research.features import FEATURES, add_features, add_forward_returns  # noqa: E402


def survey(df: pd.DataFrame, horizons=(5, 10, 20)) -> pd.DataFrame:
    rows = []
    for feat in FEATURES:
        for h in horizons:
            s = ic_summary(df, feat, h)
            if s is None:
                continue
            row = s.as_row()
            row["ls_spread_bps"] = long_short_spread(df, feat, h) * 10_000
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.reindex(out["t_stat_nw"].abs().sort_values(ascending=False).index)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-dollar-volume", type=float, default=5e6)
    ap.add_argument("--min-price", type=float, default=5.0)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--seed-only", action="store_true",
                    help="restrict to the full-history seed list; use while a "
                         "probe-phase backfill is still writing 1-year stubs "
                         "into the same cache, which would otherwise dilute "
                         "every 252-day feature")
    ap.add_argument("--min-history", type=int, default=300,
                    help="drop symbols with fewer than N cached sessions")
    args = ap.parse_args()

    symbols = None
    if args.seed_only:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from liquid_seed import SEED
        symbols = SEED
    panel = load_panel(symbols)
    if panel.empty:
        print("No cached bars. Run scripts/backfill.py first.")
        return 1

    # A symbol with only a probe window cannot support a 252-day feature; its
    # rows would be NaN in the long lookbacks and present in the short ones,
    # quietly changing which names each horizon is measured over.
    lengths = panel.groupby("symbol")["date"].transform("size")
    panel = panel[lengths >= args.min_history]

    panel = add_forward_returns(add_features(panel))

    # Liquidity screen on measured turnover, and on the RAW price so the filter
    # reflects what the stock actually traded at rather than a restated level.
    liquid = panel[(panel["dv_21d"] >= args.min_dollar_volume) &
                   (panel["raw_close"] >= args.min_price)]

    n_all, n_liq = panel["symbol"].nunique(), liquid["symbol"].nunique()
    print(f"panel: {len(panel):,} rows, {n_all} symbols, "
          f"{panel['date'].min():%Y-%m-%d} -> {panel['date'].max():%Y-%m-%d}")
    print(f"after liquidity screen (>=${args.min_dollar_volume:,.0f}/day, "
          f">=${args.min_price:.0f}): {len(liquid):,} rows, {n_liq} symbols\n")

    parts = walk_forward_split(liquid)
    for name, part in parts.items():
        print(f"  {name:11} {part['date'].min():%Y-%m-%d} -> {part['date'].max():%Y-%m-%d}  "
              f"({part['date'].nunique()} sessions)")
    print()

    train = survey(parts["train"])
    if train.empty:
        print("Not enough breadth to measure. Need more symbols cached.")
        return 1

    cols = ["feature", "horizon", "n_dates", "mean_ic", "ic_ir",
            "t_stat_naive", "t_stat_nw", "hit_rate", "ls_spread_bps"]
    print("=== TRAIN: ranked by |Newey-West t| ===")
    print(train[cols].head(args.top).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    # Anything that clears |t|>2 in-sample is a candidate, not a finding.
    candidates = train[train["t_stat_nw"].abs() > 2.0]
    print(f"\n{len(candidates)} of {len(train)} feature/horizon pairs clear |t_nw| > 2 in train.")
    if candidates.empty:
        print("Nothing to carry forward. That is a legitimate outcome on this window.")
        return 0

    print("\n=== VALIDATION: same pairs, out of sample ===")
    rows = []
    for _, c in candidates.iterrows():
        s = ic_summary(parts["validation"], c["feature"], int(c["horizon"]))
        if s is None:
            continue
        r = s.as_row()
        r["train_ic"] = c["mean_ic"]
        r["sign_held"] = "yes" if r["mean_ic"] * c["mean_ic"] > 0 else "NO"
        rows.append(r)
    if not rows:
        print("No validation coverage.")
        return 0
    val = pd.DataFrame(rows)
    print(val[["feature", "horizon", "train_ic", "mean_ic", "t_stat_nw",
               "hit_rate", "sign_held"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:8.3f}"))

    survivors = val[(val["sign_held"] == "yes") & (val["t_stat_nw"].abs() > 1.5)]
    print(f"\n{len(survivors)} survived validation with the sign intact.")
    if not survivors.empty:
        best = survivors.iloc[survivors["t_stat_nw"].abs().argmax()]
        print(f"\n=== Decile profile: {best['feature']} @ {int(best['horizon'])}d (TRAIN) ===")
        d = decile_returns(parts["train"], best["feature"], int(best["horizon"]))
        print(d[["bucket", "mean_fwd_bps", "n"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:9.1f}"))
        print("\nTest split deliberately untouched — spend it once, on a final candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
