"""`/fixtures/upcoming` under a source outage.

This endpoint is the slowest thing the API does -- it can scrape for over a
minute -- and it is what the Match Predictor and Discover tabs wait on. Two
failures are worth pinning because both were real:

A cosmetic odds fetch sat unguarded between two guarded ones, so football-data
resetting a connection took the whole endpoint down with a 500: the fixtures
were already built and scored, and a decoration failed the thing it decorated.

And the request path rebuilt the cache itself. When the warmer was failing --
precisely when the sources were unreachable -- every visitor re-attempted the
same doomed scrape and waited out the same timeout, instead of being served the
board that was already in memory.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from footballml.ingest import schedule

api = importlib.import_module("footballml.api.app")


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    """Each test owns the fixture cache outright."""
    monkeypatch.setattr(api.state, "_upcoming", None)
    monkeypatch.setattr(api.state, "_upcoming_at", None)


def _prediction(home: str = "Arsenal", away: str = "Chelsea"):
    """A minimal scored fixture, built the way the endpoint returns them."""
    return api.Prediction(
        league="E0", date="2026-10-17", home_team=home, away_team=away,
        expected_goals_home=1.6, expected_goals_away=1.1,
        prob_home_win=0.5, prob_draw=0.26, prob_away_win=0.24,
        predicted_outcome="H", competition="domestic",
        modal_score_home=1, modal_score_away=1,
        prob_over_2_5=0.52, prob_btts=0.54,
    )


# --- the odds fetch must not take the endpoint down ---------------------------

def test_failing_odds_still_returns_fixtures(monkeypatch) -> None:
    """A dead odds source costs the market row, not the board."""
    fixtures = pd.DataFrame({
        "League": ["E0"], "Date": [pd.Timestamp("2026-10-17")],
        "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
    })
    monkeypatch.setattr(schedule, "upcoming_fixtures", lambda *a, **k: fixtures)
    monkeypatch.setattr(api, "_upcoming_european", lambda: pd.DataFrame())

    def dead(*_a, **_k):
        raise ConnectionError("connection forcibly closed by the remote host")

    monkeypatch.setattr(api, "fetch_fixtures", dead)

    out = api._upcoming_fixture_list()

    assert len(out) == 1
    assert out.iloc[0]["HomeTeam"] == "Arsenal"
    # No market row rather than a broken request.
    assert not [c for c in api.ODDS_COLUMNS if c in out.columns]


def test_working_odds_are_still_merged(monkeypatch) -> None:
    """The guard must not have disabled the feature it protects."""
    fixtures = pd.DataFrame({
        "League": ["E0"], "Date": [pd.Timestamp("2026-10-17")],
        "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
    })
    odds = fixtures.assign(B365H=1.8, B365D=3.6, B365A=4.2)
    monkeypatch.setattr(schedule, "upcoming_fixtures", lambda *a, **k: fixtures)
    monkeypatch.setattr(api, "_upcoming_european", lambda: pd.DataFrame())
    monkeypatch.setattr(api, "fetch_fixtures", lambda *a, **k: odds)

    out = api._upcoming_fixture_list()
    assert out.iloc[0]["B365H"] == pytest.approx(1.8)


# --- the request path must not rebuild when it has something to serve ---------

def test_a_stale_cache_is_served_rather_than_rebuilt(monkeypatch) -> None:
    """Freshness belongs to the warmer; a visitor should never wait on a scrape."""
    api.state._upcoming = [_prediction()]
    # Older than the TTL, so the old code would have rebuilt here.
    api.state._upcoming_at = datetime.now(UTC) - 10 * api.FIXTURE_CACHE_TTL
    assert api.state.upcoming_is_stale

    def explode():
        raise AssertionError("the request path must not rebuild an existing cache")

    monkeypatch.setattr(api, "_score_upcoming", explode)

    out = api.upcoming(league=None, explain=False)
    assert [p.home_team for p in out] == ["Arsenal"]


def test_an_outage_with_a_cache_is_invisible(monkeypatch) -> None:
    """The failure mode that broke the tabs: sources down, board still served."""
    api.state._upcoming = [_prediction()]
    api.state._upcoming_at = datetime.now(UTC) - timedelta(days=3)

    def dead():
        raise ConnectionError("remote host closed the connection")

    monkeypatch.setattr(api, "_score_upcoming", dead)

    assert len(api.upcoming(league=None, explain=False)) == 1


def test_a_cached_board_never_serves_a_match_already_played(monkeypatch) -> None:
    """The bug a visitor saw: yesterday's fixtures listed as upcoming.

    The board is built with `Date >= today` baked in, so it is only correct on
    the day it was built. When the warmer could not refresh it overnight the
    request path served it unchanged -- correctly, since re-attempting a dead
    source helps nobody -- and seventeen matches played the previous evening
    appeared under "Upcoming". Getting this right needs no scrape.
    """
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC).date()
    api.state._upcoming = [
        _prediction("Yesterday FC", "Old Town"),
        _prediction("Today FC", "Now United"),
        _prediction("Tomorrow FC", "Later City"),
    ]
    api.state._upcoming[0].date = today - timedelta(days=1)
    api.state._upcoming[1].date = today
    api.state._upcoming[2].date = today + timedelta(days=1)
    api.state._upcoming_at = datetime.now(UTC)

    out = api.upcoming(league=None, explain=False)
    assert [p.home_team for p in out] == ["Today FC", "Tomorrow FC"]


def test_an_entirely_past_board_empties_rather_than_misleads(monkeypatch) -> None:
    """Shrinking to nothing reads as "nothing scheduled"; the alternative lies."""
    from datetime import UTC, datetime, timedelta

    old = datetime.now(UTC).date() - timedelta(days=3)
    api.state._upcoming = [_prediction("Long Ago", "United")]
    api.state._upcoming[0].date = old
    api.state._upcoming_at = datetime.now(UTC)

    assert api.upcoming(league=None, explain=False) == []


def test_no_cache_and_a_failing_build_is_a_503(monkeypatch) -> None:
    """A 503 tells the caller to retry; a 500 tells them it is broken."""
    def dead():
        raise ConnectionError("remote host closed the connection")

    monkeypatch.setattr(api, "_score_upcoming", dead)

    with pytest.raises(api.HTTPException) as caught:
        api.upcoming(league=None, explain=False)
    assert caught.value.status_code == 503
    assert "try again" in caught.value.detail.lower()


def test_no_cache_builds_once(monkeypatch) -> None:
    """A cold start still produces a board rather than an error."""
    calls = []

    def build():
        calls.append(1)
        return [_prediction("Liverpool", "Everton")]

    monkeypatch.setattr(api, "_score_upcoming", build)

    first = api.upcoming(league=None, explain=False)
    second = api.upcoming(league=None, explain=False)

    assert [p.home_team for p in first] == ["Liverpool"]
    # Second call is served from the cache the first one filled.
    assert len(calls) == 1
    assert len(second) == 1


def test_league_filter_still_applies_to_a_served_cache(monkeypatch) -> None:
    """Serving from cache must not bypass the per-request filtering."""
    api.state._upcoming = [_prediction(), _prediction("Bayern Munich", "Dortmund")]
    api.state._upcoming_at = datetime.now(UTC)
    monkeypatch.setattr(api, "_score_upcoming", lambda: pytest.fail("no rebuild"))

    assert len(api.upcoming(league="E0", explain=False)) == 2  # both are E0 here
    assert api.upcoming(league="SP1", explain=False) == []
