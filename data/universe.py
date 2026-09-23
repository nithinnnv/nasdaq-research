"""The Nasdaq-listed research universe.

ThetaData knows which symbols it has data for but nothing about them -- no
exchange, sector, or security type -- so the listing venue has to come from
Nasdaq itself. Nasdaq Trader publishes the definitive list daily at a public
URL with no key required.

The universe is the intersection of:
  1. Nasdaq's own listing file (what is actually Nasdaq-listed today), and
  2. ThetaData's stock symbol list (what we can actually pull bars for).

Both halves matter. (1) alone includes symbols the vendor has no data for;
(2) alone is every US equity across all venues, ~26.6k symbols.

Survivorship: the listing file is a point-in-time snapshot of *live* listings.
Anything delisted mid-window is absent, so a universe built today and
backtested over history is survivorship-biased -- it only contains companies
that made it. `SURVIVORSHIP_NOTE` spells this out; treat results from a
snapshot universe as an upper bound on realised performance.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
THETA_SYMBOLS_PATH = "/v3/stock/list/symbols"

# Nasdaq's three listing tiers, most to least stringent.
MARKET_CATEGORY = {"Q": "Global Select", "G": "Global Market", "S": "Capital Market"}

SURVIVORSHIP_NOTE = (
    "Universe is a point-in-time snapshot of currently-listed Nasdaq symbols. "
    "Companies delisted during the backtest window are absent, which biases "
    "any historical study upward. Do not read results as realisable returns."
)


# The listing file mixes common stock with the rest of an issuer's capital
# structure. Units, warrants and rights are SPAC scaffolding whose price is a
# function of the trust and the deal clock, not the business; preferreds track
# rates. None of them belong in a directional equity study, and they are a
# quarter of the file.
SECURITY_TYPE_PATTERNS = [
    ("warrant", r"\bwarrant"),
    ("unit", r"\bunits?\b"),
    ("right", r"\brights?\b"),
    ("preferred", r"preferred|\bpfd\b|% series|depositary shares? each represent"),
    ("note", r"\bnotes?\b|debenture|\bbond\b"),
    ("subunit", r"subordinate voting"),
]


def classify_security(name: str) -> str:
    """Bucket a listing by what the security actually is, from its name."""
    import re
    n = (name or "").lower()
    for label, pattern in SECURITY_TYPE_PATTERNS:
        if re.search(pattern, n):
            return label
    return "common"


@dataclass(frozen=True)
class UniverseFilter:
    exclude_etfs: bool = True
    exclude_test_issues: bool = True
    # 'D' flags issuers in default/delinquent filing status -- tradeable, but
    # their price action is dominated by the listing process, not fundamentals.
    exclude_delinquent: bool = True
    market_categories: tuple[str, ...] = ("Q", "G", "S")
    min_round_lot: int = 100
    # Keep only ordinary equity. Turn this off to study the scaffolding itself.
    common_stock_only: bool = True
    # Pre-deal SPACs are cash trusts quoted near $10; they have no operating
    # business for a directional signal to read. Off by default because the
    # name heuristic also catches a handful of real operating companies.
    exclude_spacs: bool = False


def fetch_nasdaq_listed(timeout: int = 30) -> pd.DataFrame:
    """Raw Nasdaq listing file, parsed. One row per listed security."""
    resp = requests.get(NASDAQ_LISTED_URL, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    # The file ends with a "File Creation Time:" trailer that is not a record.
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    # keep_default_na=False is load-bearing, not defensive tidying. Pandas
    # treats the strings "NA", "NAN", "NULL", "N/A" and friends as missing by
    # default, and those are real Nasdaq tickers -- "NA" is Nano Labs. Parsing
    # normally silently converts that symbol to NaN, which then crashes or, far
    # worse, quietly drops the row from the universe.
    df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str,
                     keep_default_na=False, na_values=[])
    df.columns = [c.strip() for c in df.columns]
    df["Round Lot Size"] = pd.to_numeric(df["Round Lot Size"], errors="coerce").fillna(0).astype(int)
    return df


def fetch_theta_symbols(base_url: str = "http://127.0.0.1:25503", timeout: int = 60) -> set[str]:
    """Every stock symbol the local Terminal will serve bars for."""
    resp = requests.get(f"{base_url.rstrip('/')}{THETA_SYMBOLS_PATH}", timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), dtype=str,
                     keep_default_na=False, na_values=[])
    return set(df["symbol"].astype(str).str.strip().str.strip('"'))


def build_universe(
    filt: Optional[UniverseFilter] = None,
    base_url: str = "http://127.0.0.1:25503",
) -> pd.DataFrame:
    """Nasdaq-listed common stocks that ThetaData can actually serve."""
    filt = filt or UniverseFilter()
    listed = fetch_nasdaq_listed()
    theta = fetch_theta_symbols(base_url)

    df = listed.rename(columns={
        "Symbol": "symbol", "Security Name": "name", "Market Category": "market_category",
        "Test Issue": "test_issue", "Financial Status": "financial_status",
        "Round Lot Size": "round_lot", "ETF": "etf",
    })[["symbol", "name", "market_category", "test_issue", "financial_status", "round_lot", "etf"]]
    df["symbol"] = df["symbol"].str.strip()

    steps: list[tuple[str, int]] = [("nasdaq listing file", len(df))]
    if filt.exclude_test_issues:
        df = df[df["test_issue"] != "Y"]
        steps.append(("drop test issues", len(df)))
    if filt.exclude_etfs:
        df = df[df["etf"] != "Y"]
        steps.append(("drop ETFs", len(df)))
    if filt.exclude_delinquent:
        df = df[df["financial_status"].isin(["N", None]) | df["financial_status"].isna()]
        steps.append(("drop delinquent filers", len(df)))
    df = df[df["market_category"].isin(filt.market_categories)]
    steps.append(("market category filter", len(df)))
    df = df[df["round_lot"] >= filt.min_round_lot]
    steps.append(("round lot filter", len(df)))

    df = df.copy()
    df["security_type"] = df["name"].map(classify_security)
    if filt.common_stock_only:
        df = df[df["security_type"] == "common"]
        steps.append(("common stock only", len(df)))
    if filt.exclude_spacs:
        df = df[~df["name"].str.contains(r"Acquisition Corp", case=False, na=False)]
        steps.append(("drop pre-deal SPACs", len(df)))

    df = df[df["symbol"].isin(theta)]
    steps.append(("intersect ThetaData coverage", len(df)))

    df = df.copy()
    df["tier"] = df["market_category"].map(MARKET_CATEGORY)
    df.attrs["funnel"] = steps
    df.attrs["survivorship"] = SURVIVORSHIP_NOTE
    df.attrs["as_of"] = date.today().isoformat()
    return df.sort_values("symbol").reset_index(drop=True)


def save(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
