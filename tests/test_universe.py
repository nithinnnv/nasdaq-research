"""Universe parsing tests.

The NA case is not hypothetical: "NA" is Nano Labs' real Nasdaq ticker, and
pandas' default missing-value handling turns it into NaN. That silently
removed a live symbol from the universe and crashed the backfill on a float
where a string was expected. Symbols that collide with missing-value tokens
are exactly the ones no one thinks to check.
"""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.universe import classify_security  # noqa: E402

LISTING = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
NA|Nano Labs Ltd - Class A Ordinary Shares|G|N|N|100|N|N
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
NULL|Fake Corp - Common Stock|Q|N|N|100|N|N
File Creation Time: 0904202621:31|||||||
"""


def parse(text: str, **kwargs) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str, **kwargs)


def test_default_parsing_would_lose_na_ticker():
    """Documents the trap the real parser has to avoid."""
    assert parse(LISTING)["Symbol"].isna().sum() == 2


def test_na_and_null_tickers_survive_correct_parsing():
    df = parse(LISTING, keep_default_na=False, na_values=[])
    assert set(df["Symbol"]) == {"NA", "AAPL", "NULL"}
    assert df["Symbol"].map(lambda s: isinstance(s, str)).all()


def test_file_creation_trailer_is_not_a_record():
    df = parse(LISTING, keep_default_na=False, na_values=[])
    assert not df["Symbol"].str.startswith("File Creation").any()
    assert len(df) == 3


@pytest.mark.parametrize("name,expected", [
    ("Armada Acquisition Corp. III - Warrant", "warrant"),
    ("Abony Acquisition Corp. I - Units", "unit"),
    ("Some Corp - Rights", "right"),
    ("Bank X - 6.5% Series A Preferred Stock", "preferred"),
    ("Apple Inc. - Common Stock", "common"),
    ("ATA Creativity Global - American Depositary Shares, each representing two common shares", "common"),
])
def test_security_classification(name, expected):
    assert classify_security(name) == expected
