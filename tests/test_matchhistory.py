"""Tests for the match-history spine.

Every match in the dataset passes through this module, and until now nothing
tested it. It is also where the quiet failures live: a date parsed
month-first instead of day-first moves a match to a different season without
erroring, and an encoding change upstream drops whole files -- the ``latin-1``
fallback in `read_csv_bytes` exists because that happened.

So these pin the parsing and the orientation, not the fetching. The network
paths are left alone; what matters is that bytes on disk become the right rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.ingest.matchhistory import (
    current_season_label,
    read_csv_bytes,
    to_matches,
    to_team_match_history,
)

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"


def _csv(*rows: str, encoding: str = "utf-8") -> bytes:
    return "\n".join([HEADER, *rows]).encode(encoding)


# --- season labelling -------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("2026-09-30", "2627"),
        ("2026-07-01", "2627"),   # July belongs to the season about to start
        ("2026-06-30", "2526"),   # June still belongs to the one ending
        ("2027-01-15", "2627"),   # January is the back half of 26/27
        ("2019-12-31", "1920"),
        ("2020-01-01", "1920"),
    ],
)
def test_season_label_splits_the_year_in_july(date: str, expected: str) -> None:
    """A season runs August to May, so the boundary is mid-year, not New Year.

    Getting this wrong files August matches under the previous season, which
    then joins them to the wrong squad-strength row.
    """
    assert current_season_label(pd.Timestamp(date)) == expected


# --- decoding ---------------------------------------------------------------


def test_a_byte_order_mark_does_not_hide_the_first_column() -> None:
    """A BOM turns 'Div' into '\\ufeffDiv' and the file looks column-less.

    Files upstream have been re-saved with one, and the symptom is a whole
    season silently missing rather than an error.
    """
    df = read_csv_bytes(_csv("E0,19/09/2026,Arsenal,Chelsea,2,1,H", encoding="utf-8-sig"))
    assert list(df["Div"]) == ["E0"]


def test_latin1_accents_still_parse() -> None:
    """Older files carry accented club names in Latin-1, which is not UTF-8."""
    df = read_csv_bytes(_csv("SP1,19/09/2026,Alavés,Málaga,1,0,H", encoding="latin-1"))
    assert df["HomeTeam"].iloc[0].startswith("Alav")


def test_trailing_blank_rows_are_dropped() -> None:
    """These files routinely end in empty lines; they are not matches."""
    df = read_csv_bytes(_csv("E0,19/09/2026,Arsenal,Chelsea,2,1,H", ",,,,,,", ",,,,,,"))
    assert len(df) == 1


def test_a_file_without_a_div_column_raises() -> None:
    """Better to stop than to return an empty frame that reads as 'no matches'."""
    with pytest.raises(ValueError, match="Div"):
        read_csv_bytes(b"Date,HomeTeam\n19/09/2026,Arsenal\n")


# --- wide table -------------------------------------------------------------


@pytest.fixture
def matches() -> pd.DataFrame:
    raw = read_csv_bytes(
        _csv(
            "E0,19/09/2026,Arsenal,Chelsea,2,1,H",
            "E0,20/09/2026,Everton,Fulham,0,0,D",
            "E0,21/09/2026,Leeds,Brentford,1,3,A",
        )
    ).assign(_season="2627")
    return to_matches(raw)


def test_dates_are_read_day_first(matches: pd.DataFrame) -> None:
    """19/09 is the 19th of September, not an invalid 9th of the 19th month.

    Month-first parsing silently moves matches across seasons wherever the day
    is 12 or lower, which is most of them.
    """
    assert matches["Date"].min() == pd.Timestamp("2026-09-19")
    assert matches["Date"].max() == pd.Timestamp("2026-09-21")


def test_goals_are_numeric(matches: pd.DataFrame) -> None:
    """Coerced, not left as strings -- everything downstream sums them."""
    assert matches["FTHG"].dtype.kind in "iuf"
    assert matches["FTHG"].sum() == 3


def test_unplayed_fixtures_are_dropped() -> None:
    """A postponed match ships with blank scores and is not a result.

    Keeping it would feed a NaN-scored row into the rolling form windows.
    """
    raw = read_csv_bytes(
        _csv(
            "E0,19/09/2026,Arsenal,Chelsea,2,1,H",
            "E0,26/09/2026,Arsenal,Everton,,,",
        )
    ).assign(_season="2627")
    played = to_matches(raw)
    assert list(played["AwayTeam"]) == ["Chelsea"]


def test_a_two_digit_year_lands_in_the_right_century() -> None:
    """Seasons before 2002 are dated '17/08/02' and must not become 1902."""
    raw = read_csv_bytes(_csv("E0,17/08/02,Arsenal,Chelsea,2,1,H")).assign(_season="0203")
    assert to_matches(raw)["Date"].iloc[0] == pd.Timestamp("2002-08-17")


def test_a_day_past_the_twelfth_is_not_read_as_a_month() -> None:
    """25/12 is Boxing-Day-eve, not a nonexistent 12th of month 25.

    Month-first parsing would raise here but pass silently on 05/12, which is
    why the whole file is parsed day-first rather than inferred per row.
    """
    raw = read_csv_bytes(_csv("E0,25/12/2026,Arsenal,Chelsea,2,1,H")).assign(_season="2627")
    assert to_matches(raw)["Date"].iloc[0] == pd.Timestamp("2026-12-25")


# --- long table -------------------------------------------------------------


def test_every_match_becomes_exactly_two_rows(matches: pd.DataFrame) -> None:
    long_df = to_team_match_history(matches)
    assert len(long_df) == 2 * len(matches)
    assert set(long_df["Venue"]) == {"Home", "Away"}


def test_the_two_sides_mirror_each_other(matches: pd.DataFrame) -> None:
    """One team's goals for are the other's goals against.

    The rolling form features read this table, so a swap here would train the
    model on every team's opponent's record.
    """
    long_df = to_team_match_history(matches)
    pair = long_df[long_df["Date"] == pd.Timestamp("2026-09-19")]
    home = pair[pair["Venue"] == "Home"].iloc[0]
    away = pair[pair["Venue"] == "Away"].iloc[0]

    assert home["Team"] == away["Opponent"] == "Arsenal"
    assert away["Team"] == home["Opponent"] == "Chelsea"
    assert (home["GoalsFor"], home["GoalsAgainst"]) == (2, 1)
    assert (away["GoalsFor"], away["GoalsAgainst"]) == (1, 2)


def test_results_are_oriented_per_team(matches: pd.DataFrame) -> None:
    """Derived from goals rather than read from FTR, so an away win is a W for
    the away side rather than the 'A' the source recorded."""
    long_df = to_team_match_history(matches).set_index(["Team", "Date"])
    d = pd.Timestamp("2026-09-21")
    assert long_df.loc[("Brentford", d), "Result"] == "W"   # won 3-1 away
    assert long_df.loc[("Leeds", d), "Result"] == "L"
    draw = pd.Timestamp("2026-09-20")
    assert long_df.loc[("Everton", draw), "Result"] == "D"
    assert long_df.loc[("Fulham", draw), "Result"] == "D"


def test_rows_come_back_in_date_order(matches: pd.DataFrame) -> None:
    """Rolling form is computed over this order; unsorted input would let a
    window see a later match."""
    dates = to_team_match_history(matches)["Date"]
    assert dates.is_monotonic_increasing
