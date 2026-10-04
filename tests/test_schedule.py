"""Tests for the forward fixture schedule.

These exist because the previous forward source failed in a way that looked
like success. football-data.co.uk publishes one rolling file of scheduled
matches; on a Monday it routinely still holds the round just played. The job
fetched it, correctly skipped every already-played fixture, stored nothing, and
exited zero. The visible symptom was a live track record of eighteen
predictions that were *all Spanish* -- four leagues had never had a single
fixture stored, because the feed had never happened to be ahead of the job for
them.

So the properties worth pinning are about what is and is not in the window, and
about telling "nothing is scheduled" apart from "no schedule arrived".
"""

from __future__ import annotations

import os
import time
from datetime import timedelta

import pandas as pd
import pytest

from footballml.ingest.schedule import (
    _SCORE,
    DEFAULT_HORIZON_DAYS,
    SCHEDULE_MAX_AGE,
    _cache_is_fresh,
    window_fixtures,
)


def _schedule(rows: list[tuple[str, str, str, str, bool]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": lg, "Season": "2627", "Date": pd.Timestamp(date),
             "HomeTeam": home, "AwayTeam": away, "played": played}
            for lg, date, home, away, played in rows
        ]
    )


TODAY = pd.Timestamp("2026-10-05")


@pytest.fixture
def schedule() -> pd.DataFrame:
    return _schedule([
        ("E0", "2026-09-20", "Arsenal", "Chelsea", True),     # already played
        ("E0", "2026-10-09", "Man City", "Everton", False),   # in window
        ("SP1", "2026-10-10", "Barcelona", "Betis", False),   # in window
        ("D1", "2026-10-11", "Bayern Munich", "Mainz", False),
        ("I1", "2026-11-20", "Inter", "Roma", False),         # beyond horizon
    ])


def test_only_unplayed_fixtures_in_the_window(schedule):
    window = window_fixtures(schedule, today=TODAY)
    assert "Arsenal" not in set(window["HomeTeam"])


def test_fixtures_beyond_the_horizon_are_excluded(schedule):
    window = window_fixtures(schedule, horizon_days=10, today=TODAY)
    assert "Inter" not in set(window["HomeTeam"])
    assert len(window) == 3


def test_every_league_with_a_fixture_is_represented(schedule):
    """The failure this module exists for: coverage must not depend on luck."""
    window = window_fixtures(schedule, today=TODAY)
    assert set(window["League"]) == {"E0", "SP1", "D1"}


def test_a_wider_horizon_reaches_further(schedule):
    assert len(window_fixtures(schedule, horizon_days=60, today=TODAY)) == 4


def test_fixtures_already_in_the_past_are_excluded():
    """A fixture that kicked off yesterday is not upcoming, played or not.

    football-data's file keeps publishing a round after it is played, and taking
    that as upcoming is precisely the bug.
    """
    schedule = _schedule([("E0", "2026-10-01", "Arsenal", "Chelsea", False)])
    assert window_fixtures(schedule, today=TODAY).empty


def test_an_empty_schedule_gives_an_empty_window_not_an_error():
    empty = pd.DataFrame(
        columns=["League", "Season", "Date", "HomeTeam", "AwayTeam", "played"]
    )
    window = window_fixtures(empty, today=TODAY)
    assert window.empty
    assert "played" not in window.columns


def test_the_window_drops_the_played_flag(schedule):
    """Callers feed this straight to the feature builder, which has its own
    notion of what has been played; two sources of that truth is one too many."""
    assert "played" not in window_fixtures(schedule, today=TODAY).columns


def test_default_horizon_covers_a_weekend_round():
    """Ten days must reach a round that starts nine days out, or a Monday job
    would miss the following weekend entirely."""
    schedule = _schedule([("E0", "2026-10-14", "Arsenal", "Chelsea", False)])
    assert len(window_fixtures(schedule, horizon_days=DEFAULT_HORIZON_DAYS, today=TODAY)) == 1


@pytest.mark.parametrize(
    "score", ["4\u20132", "4-2", "4 \u2013 2", "0\u20130", "10\u201311"]
)
def test_scores_parse_whatever_dash_the_source_used(score):
    """A score means the fixture has been played, so a parse that fails marks
    the whole season unplayed -- and FBref uses an en-dash, not a hyphen."""
    assert pd.Series([score]).str.extract(_SCORE)[0].notna().all()


def test_a_fixture_with_no_score_is_unplayed():
    assert pd.Series(["", "nan", "None"]).str.extract(_SCORE)[0].isna().all()


class _Reader:
    """Stands in for a soccerdata reader: `_cache_is_fresh` reads three fields."""

    def __init__(self, data_dir, leagues, seasons):
        self.data_dir = data_dir
        self.leagues = leagues
        self.seasons = seasons


def _page(tmp_path, league, season, age=timedelta()):
    path = tmp_path / f"schedule_{league}_{season}.html"
    path.write_text("<html></html>", encoding="utf-8")
    when = time.time() - age.total_seconds()
    os.utime(path, (when, when))
    return path


def test_a_young_cache_is_reused(tmp_path):
    """The whole point: no scrape when the pages on disk are current.

    soccerdata refuses its own cache for a season in progress, which made every
    call a ~90s headless-browser fetch of pages that had not changed.
    """
    _page(tmp_path, "ENG-Premier League", "2627", age=timedelta(hours=1))
    reader = _Reader(tmp_path, ["ENG-Premier League"], ["2627"])
    assert _cache_is_fresh(reader, SCHEDULE_MAX_AGE)


def test_an_aged_cache_is_refetched(tmp_path):
    _page(tmp_path, "ENG-Premier League", "2627", age=SCHEDULE_MAX_AGE + timedelta(minutes=1))
    reader = _Reader(tmp_path, ["ENG-Premier League"], ["2627"])
    assert not _cache_is_fresh(reader, SCHEDULE_MAX_AGE)


def test_one_stale_league_refetches_the_call(tmp_path):
    """All pages or none. `force_cache` is per call, not per page.

    Returning True here would serve a league whose schedule had aged out, and
    the fixture it was missing would simply not appear on the page.
    """
    _page(tmp_path, "ENG-Premier League", "2627", age=timedelta(hours=1))
    _page(tmp_path, "ESP-La Liga", "2627", age=SCHEDULE_MAX_AGE + timedelta(hours=1))
    reader = _Reader(tmp_path, ["ENG-Premier League", "ESP-La Liga"], ["2627"])
    assert not _cache_is_fresh(reader, SCHEDULE_MAX_AGE)


def test_a_missing_page_is_not_fresh(tmp_path):
    """A cold machine must still fetch. soccerdata downloads what it lacks
    whatever `force_cache` says, so this only decides whether the *other* pages
    are reused -- and with one missing there is a scrape to pay for anyway."""
    reader = _Reader(tmp_path, ["ENG-Premier League"], ["2627"])
    assert not _cache_is_fresh(reader, SCHEDULE_MAX_AGE)


def test_max_age_none_always_refetches(tmp_path):
    """What the weekly job passes: it is the thing that refreshes the cache."""
    _page(tmp_path, "ENG-Premier League", "2627")
    reader = _Reader(tmp_path, ["ENG-Premier League"], ["2627"])
    assert not _cache_is_fresh(reader, None)


def test_a_reader_with_no_leagues_is_not_fresh(tmp_path):
    """Vacuous truth would claim a cache hit for pages that do not exist."""
    assert not _cache_is_fresh(_Reader(tmp_path, [], []), SCHEDULE_MAX_AGE)
