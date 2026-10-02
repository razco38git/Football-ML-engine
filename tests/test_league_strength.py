"""The league comparison behind the Team Strength and What-If panels.

It exists to answer the one question a cross-league prediction raises and the
model cannot: the fixture list never pairs Paris with Manchester, so the honest
framing is how far apart the two *divisions* rate rather than a forecast.

The numbers are comparable because a player rating is not league-relative --
EA's overall is a global scale, and `rating.py` ranks performance within role
and season across all five leagues pooled (`by = ["role", "Season"]`), not
within a division.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest
from fastapi.testclient import TestClient

api = importlib.import_module("footballml.api.app")

TEAMS = pd.DataFrame(
    [
        # Two seasons, so "latest by default" is actually exercised.
        ("E0", "2526", "Man City", 81.0), ("E0", "2526", "Arsenal", 79.0),
        ("E0", "2526", "Burnley", 67.0),
        ("F1", "2526", "Paris SG", 82.0), ("F1", "2526", "Metz", 64.0),
        ("E0", "2425", "Man City", 50.0),
    ],
    columns=["League", "Season", "Team", "strength_overall"],
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api.state, "teams", TEAMS)
    return TestClient(api.app)


def _by_league(payload: list[dict]) -> dict[str, dict]:
    return {row["league"]: row for row in payload}


def test_leagues_come_back_strongest_first(client: TestClient) -> None:
    rows = client.get("/leagues/strength").json()
    assert [r["league"] for r in rows] == ["E0", "F1"]


def test_the_mean_is_over_that_league_only(client: TestClient) -> None:
    rows = _by_league(client.get("/leagues/strength").json())
    assert rows["E0"]["mean_strength"] == pytest.approx((81 + 79 + 67) / 3, abs=0.05)
    assert rows["F1"]["mean_strength"] == pytest.approx(73.0, abs=0.05)


def test_only_the_latest_season_is_used(client: TestClient) -> None:
    """A stale 2425 row for Man City would drag the mean down by ten points."""
    rows = _by_league(client.get("/leagues/strength").json())
    assert rows["E0"]["season"] == "2526"
    assert rows["E0"]["n_teams"] == 3


def test_an_earlier_season_can_be_asked_for(client: TestClient) -> None:
    rows = client.get("/leagues/strength?season=2425").json()
    assert len(rows) == 1
    assert rows[0]["mean_strength"] == 50.0


def test_the_extremes_name_the_right_clubs(client: TestClient) -> None:
    rows = _by_league(client.get("/leagues/strength").json())
    assert (rows["E0"]["strongest_team"], rows["E0"]["strongest_strength"]) == ("Man City", 81.0)
    assert (rows["E0"]["weakest_team"], rows["E0"]["weakest_strength"]) == ("Burnley", 67.0)


def test_spread_separates_a_top_heavy_league_from_an_even_one(client: TestClient) -> None:
    """Both means are close; the spread is what says Ligue 1 is two clubs."""
    rows = _by_league(client.get("/leagues/strength").json())
    assert rows["F1"]["spread"] > rows["E0"]["spread"]


def test_a_season_with_no_ratings_is_a_404_not_an_empty_list(client: TestClient) -> None:
    assert client.get("/leagues/strength?season=9999").status_code == 404


def test_no_ratings_at_all_says_what_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.state, "teams", pd.DataFrame())
    response = TestClient(api.app).get("/leagues/strength")
    assert response.status_code == 404
    assert "build_players" in response.json()["detail"]
