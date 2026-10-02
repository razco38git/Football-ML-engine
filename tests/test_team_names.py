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

import pytest
from fastapi import HTTPException

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
