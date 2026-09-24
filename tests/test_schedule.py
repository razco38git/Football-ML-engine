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

import pandas as pd
import pytest

from footballml.ingest.schedule import _SCORE, DEFAULT_HORIZON_DAYS, window_fixtures


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
