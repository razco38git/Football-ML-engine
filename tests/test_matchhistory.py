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
import requests

from footballml.ingest import matchhistory
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


# --- retrying a source that resets under load ---------------------------------
#
# On 2026-10-19 football-data.co.uk reset every connection for over an hour --
# not a block and not a TLS problem, since the same client succeeded minutes
# later unchanged. A single `requests.get` with no retry meant one reset killed
# the weekly refresh at its first step, and the site went a weekend stale.


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.content = b"Div,Date\nE0,01/01/2026\n"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code), response=self)


def test_a_reset_is_retried_and_the_retry_is_returned(monkeypatch) -> None:
    calls = []

    def flaky(url, **kwargs):
        calls.append(url)
        if len(calls) < 3:
            raise requests.ConnectionError("Connection aborted.")
        return _Response(200)

    monkeypatch.setattr(matchhistory.requests, "get", flaky)
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    assert matchhistory._get("https://example.test/E0.csv").status_code == 200
    assert len(calls) == 3


def test_the_last_error_is_raised_once_the_attempts_are_spent(monkeypatch) -> None:
    """It must fail loudly. The weekly stops on a failed fetch by design --
    "a partial refresh looks like a complete one"."""

    def always_reset(url, **kwargs):
        raise requests.ConnectionError("Connection aborted.")

    monkeypatch.setattr(matchhistory.requests, "get", always_reset)
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    with pytest.raises(requests.ConnectionError):
        matchhistory._get("https://example.test/E0.csv")


def test_a_404_is_an_answer_and_is_not_retried(monkeypatch) -> None:
    """The source simply has not published that file. Retrying it would turn a
    known absence into five pointless requests and a minute of backoff."""
    calls = []

    def missing(url, **kwargs):
        calls.append(url)
        return _Response(404)

    monkeypatch.setattr(matchhistory.requests, "get", missing)
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    assert matchhistory._get("https://example.test/E0.csv").status_code == 404
    assert len(calls) == 1


def test_a_server_error_is_retried(monkeypatch) -> None:
    calls = []

    def flaky(url, **kwargs):
        calls.append(url)
        return _Response(503 if len(calls) < 2 else 200)

    monkeypatch.setattr(matchhistory.requests, "get", flaky)
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    assert matchhistory._get("https://example.test/E0.csv").status_code == 200
    assert len(calls) == 2


def test_the_backoff_grows_and_is_bounded(monkeypatch) -> None:
    """Five waits of 2, 4, 8, 16, 32 seconds -- about a minute per file at
    worst, against a 70-minute job."""
    waits = []

    monkeypatch.setattr(
        matchhistory.requests, "get",
        lambda url, **kw: (_ for _ in ()).throw(requests.ConnectionError("reset")),
    )
    monkeypatch.setattr(matchhistory.time, "sleep", waits.append)

    with pytest.raises(requests.ConnectionError):
        matchhistory._get("https://example.test/E0.csv")

    assert len(waits) == matchhistory.DOWNLOAD_ATTEMPTS - 1
    assert waits == sorted(waits), "backoff must not shrink"
    assert waits[0] == matchhistory.BACKOFF_BASE_SECONDS
    assert sum(waits) < 120, "a flaky file must not stall the job for minutes"


# --- surviving an outage when yesterday's copy is on disk ---------------------


def test_a_failed_download_falls_back_to_the_cached_season(tmp_path, monkeypatch, caplog) -> None:
    """The whole point of 2026-10-19: the file was on disk and the run died anyway."""
    cached = tmp_path / "E0_2627.csv"
    cached.write_bytes(b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,01/01/2026,A,B,1,0,H\n")

    def always_reset(url, **kwargs):
        raise requests.ConnectionError("Connection aborted.")

    monkeypatch.setattr(matchhistory.requests, "get", always_reset)
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    with caplog.at_level("WARNING"):
        out = matchhistory.fetch_match_history(["E0"], ["2627"], cache_dir=tmp_path)

    assert len(out) == 1, "the cached season should still have been read"
    assert any("falling back to the cached copy" in r.message for r in caplog.records)


def test_an_outage_with_no_cache_still_raises(tmp_path, monkeypatch) -> None:
    """There is nothing to fall back *to*, and inventing an empty season would
    be the silent-approximation failure this project refuses."""

    def always_reset(url, **kwargs):
        raise requests.ConnectionError("Connection aborted.")

    monkeypatch.setattr(matchhistory.requests, "get", always_reset)
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    with pytest.raises(requests.ConnectionError):
        matchhistory.fetch_match_history(["E0"], ["2627"], cache_dir=tmp_path)


def test_a_successful_download_overwrites_the_cache(tmp_path, monkeypatch) -> None:
    """The fallback must not become sticky: when the source is healthy the run
    is exactly what it always was."""
    cached = tmp_path / "E0_2627.csv"
    cached.write_bytes(b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nE0,01/01/2026,A,B,1,0,H\n")
    fresh = (
        b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        b"E0,01/01/2026,A,B,1,0,H\nE0,02/01/2026,C,D,2,2,D\n"
    )

    class _Ok:
        status_code = 200
        content = fresh

        def raise_for_status(self):
            return None

    monkeypatch.setattr(matchhistory.requests, "get", lambda url, **kw: _Ok())
    monkeypatch.setattr(matchhistory.time, "sleep", lambda _: None)

    out = matchhistory.fetch_match_history(["E0"], ["2627"], cache_dir=tmp_path)
    assert len(out) == 2
    assert cached.read_bytes() == fresh
