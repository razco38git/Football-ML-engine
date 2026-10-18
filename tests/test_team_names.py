"""`/predict` must accept a team by whichever name the caller has.

The obvious way to find a team name for `/predict` is `/teams`, which serves
the ratings table -- spelled the way Understat does, "Manchester City". The
match history the endpoint validates against is spelled the way
football-data.co.uk does, "Man City". The two never met, so every name taken
from the site's own team list came back 404.

`config/team_aliases.yaml` already reconciles the sources, so the endpoint
resolves through it rather than making the caller guess which spelling wins.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

from footballml.entities import load_aliases

api = importlib.import_module("footballml.api.app")

KNOWN = {"Man City", "Man United", "Ath Madrid", "Arsenal"}


def test_a_name_the_history_uses_passes_straight_through() -> None:
    assert api._canonical_team("Man City", KNOWN) == "Man City"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("Manchester City", "Man City"),
        ("Manchester United", "Man United"),
        ("Atletico Madrid", "Ath Madrid"),
    ],
)
def test_a_ratings_table_name_resolves_to_the_history_name(given: str, expected: str) -> None:
    assert api._canonical_team(given, KNOWN) == expected


def test_an_unknown_name_is_a_404_not_a_silent_pass() -> None:
    """Passing it through would build features for a team with no matches."""
    with pytest.raises(HTTPException) as exc:
        api._canonical_team("Nowhere FC", KNOWN)
    assert exc.value.status_code == 404


def test_an_alias_for_a_team_outside_this_dataset_is_still_unknown() -> None:
    """The alias config covers seasons we no longer load; being in it is not
    the same as being in the match history."""
    with pytest.raises(HTTPException):
        api._canonical_team("Manchester City", {"Arsenal"})


# --- the Elo chart must reach every club the ratings table lists ---------------
#
# `_canonical_team` was written for `/predict` and applied only there. The
# team-strength panel's Elo sparkline looked up `state.features` with the name
# it had been handed -- the ratings-table spelling -- so Barcelona charted and
# Paris Saint Germain 404'd, and the component renders nothing on an error. The
# chart was simply absent for 44 of 168 clubs with no message saying why.


def test_the_elo_endpoint_resolves_the_ratings_table_spelling() -> None:
    """Whatever `/teams` serves must be accepted by `/teams/{name}/elo`."""
    source = inspect.getsource(api.team_elo)
    assert "_canonical_team(name" in source, "the Elo lookup must resolve the name"


def test_every_rated_club_resolves_to_a_name_the_match_data_uses() -> None:
    """The real guard: a promoted club with no alias would silently lose its
    chart again, and nothing else in the suite would notice.

    Data-driven rather than a fixture, because the failure is always a *new*
    name that no handwritten case anticipated.
    """
    strength = Path("data/processed/team_strength.csv")
    matches = Path("data/processed/matches_all.csv")
    if not (strength.exists() and matches.exists()):
        pytest.skip("processed data not built in this checkout")

    rated = set(pd.read_csv(strength)["Team"].dropna())
    played = pd.read_csv(matches, usecols=["HomeTeam", "AwayTeam"], low_memory=False)
    known = set(played["HomeTeam"].dropna()) | set(played["AwayTeam"].dropna())

    unreachable = sorted(
        name for name in rated
        if name not in known and load_aliases("understat").get(name) not in known
    )
    assert not unreachable, (
        f"{len(unreachable)} rated club(s) have no name the match data knows, so "
        f"their Elo chart will not render: {unreachable}"
    )
