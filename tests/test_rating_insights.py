"""The two rating questions the Discover tab answers, and the page's two columns.

Each test is a claim the page makes out loud, so a change that quietly breaks
one of them fails here rather than on the site.
"""

from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

api = importlib.import_module("footballml.api.app")

MEASURED = pd.DataFrame(
    [
        ("E0", "2526", "Man City", 81.0, 80.0, 79.0, 82.0, 83.0, 15),
        ("E0", "2526", "Arsenal", 79.0, 78.0, 80.0, 78.0, 79.0, 15),
        ("E0", "2627", "Man City", 77.0, 76.0, 75.0, 78.0, 79.0, 12),
    ],
    columns=[
        "League",
        "Season",
        "Team",
        "strength_overall",
        "strength_goalkeeper",
        "strength_defence",
        "strength_midfield",
        "strength_attack",
        "n_players",
    ],
)

#: The pre-season rating covers the clubs the measured one cannot: every club
#: in the season in progress, including one with no measured row at all.
PRE_SEASON = pd.DataFrame(
    [
        ("E0", "2526", "Man City", 80.0, 79.0, 78.0, 81.0, 82.0, 15, 14, "Manchester City"),
        ("E0", "2627", "Man City", 78.0, 77.0, 76.0, 79.0, 80.0, 15, 15, "Manchester City"),
        ("E0", "2627", "Burnley", 70.0, 69.0, 68.0, 71.0, 72.0, 15, 6, "Burnley"),
    ],
    columns=[
        "League",
        "Season",
        "Team",
        "squad_overall",
        "squad_goalkeeper",
        "squad_defence",
        "squad_midfield",
        "squad_attack",
        "n_players",
        "n_from_history",
        "ea_label",
    ],
)


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api.state, "teams", MEASURED)
    monkeypatch.setattr(api.state, "squads", PRE_SEASON)
    # The insight payload is cached for the life of the process, so one test's
    # answer would otherwise be served to the next.
    monkeypatch.setattr(api.state, "_rating_insights", None)
    return TestClient(api.app)


def _teams(client: TestClient, **params) -> dict:
    response = client.get("/teams", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_both_ratings_arrive_on_one_row(client: TestClient) -> None:
    rows = {r["team"]: r for r in _teams(client, season="2526")["teams"]}

    assert rows["Man City"]["strength_overall"] == 81.0
    assert rows["Man City"]["squad_overall"] == 80.0
    assert rows["Man City"]["squad_from_history"] == 14


def test_a_club_with_no_measured_rating_still_appears(client: TestClient) -> None:
    """The whole point: 78 of 96 clubs have no measured rating in September."""
    rows = {r["team"]: r for r in _teams(client, season="2627")["teams"]}

    assert set(rows) == {"Man City", "Burnley"}
    assert rows["Burnley"]["strength_overall"] is None
    assert rows["Burnley"]["squad_overall"] == 70.0
    # `n_players` describes whichever rating the row is showing.
    assert rows["Burnley"]["n_players"] == 15


def test_a_thin_season_is_ordered_by_the_pre_season_rating_and_says_so(
    client: TestClient,
) -> None:
    page = _teams(client, season="2627")

    assert page["ordered_by"] == "pre-season"
    assert [r["team"] for r in page["teams"]] == ["Man City", "Burnley"]


def test_a_settled_season_is_ordered_by_the_measured_rating(client: TestClient) -> None:
    page = _teams(client, season="2526")

    assert page["ordered_by"] == "measured"
    assert [r["team"] for r in page["teams"]] == ["Man City", "Arsenal"]


def test_the_season_in_progress_is_offered_on_the_pre_season_rating_alone(
    client: TestClient,
) -> None:
    """Without this the newest season is simply missing from the dropdown."""
    assert "2627" in _teams(client)["seasons"]


MATCHES = pd.DataFrame(
    [
        ("E0", "2526", "Man City", 2, 0),
        ("E0", "2526", "Man City", 1, 1),
        ("E0", "2526", "Arsenal", 0, 3),
    ],
    columns=["League", "Season", "Team", "GoalsFor", "GoalsAgainst"],
)


def test_a_line_correlation_needs_a_body_of_club_seasons(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three clubs cannot produce a correlation worth publishing."""
    monkeypatch.setattr(api.state, "tmh", MATCHES)

    payload = client.get("/insights/ratings").json()

    assert payload["lines"] == []
    assert payload["overall"] is None


def test_the_one_player_figure_comes_from_the_precomputed_file(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(api.state, "tmh", MATCHES)
    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)
    (tmp_path / "rating_insights.json").write_text(
        json.dumps(
            {
                "one_player": {
                    "median_drop": 0.87,
                    "q1_drop": 0.62,
                    "q3_drop": 1.15,
                    "max_drop": 4.64,
                    "max_team": "FC Cologne",
                    "max_season": "2223",
                    "n": 1177,
                }
            }
        ),
        encoding="utf-8",
    )

    player = client.get("/insights/ratings").json()["one_player"]

    assert player["median_drop"] == 0.87
    assert player["max_team"] == "FC Cologne"
    # The recorded ablation travels with it, because one number without the
    # other invites exactly the wrong conclusion.
    assert player["squad_rps_low"] < 0 < player["squad_rps_high"]


def test_a_missing_file_leaves_the_card_off_rather_than_failing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(api.state, "tmh", MATCHES)
    monkeypatch.setattr(api, "PROCESSED_DIR", tmp_path)

    response = client.get("/insights/ratings")

    assert response.status_code == 200
    assert response.json()["one_player"] is None


FORM_MATCHES_PLAYED = pd.DataFrame(
    [
        # Six matches, so the five-match window actually drops the oldest.
        ("E0", "2627", "Man City", "Rival A", "home", "2026-08-01", 1, 0),
        ("E0", "2627", "Man City", "Rival B", "away", "2026-08-08", 0, 0),
        ("E0", "2627", "Man City", "Rival C", "home", "2026-08-15", 2, 1),
        ("E0", "2627", "Man City", "Rival D", "away", "2026-08-22", 3, 0),
        ("E0", "2627", "Man City", "Rival E", "home", "2026-08-29", 1, 2),
        ("E0", "2627", "Man City", "Rival F", "away", "2026-09-05", 2, 2),
    ],
    columns=[
        "League",
        "Season",
        "Team",
        "Opponent",
        "Venue",
        "Date",
        "GoalsFor",
        "GoalsAgainst",
    ],
)


@pytest.fixture
def played(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.state, "tmh", FORM_MATCHES_PLAYED)
    monkeypatch.setattr(api.state, "features", pd.DataFrame())
    monkeypatch.setattr(api.state, "retrodicted", pd.DataFrame())


def test_form_is_the_last_five_not_the_whole_season(played: None) -> None:
    """The first win is outside the window, so it must not be counted."""
    row = api._season_form("E0", "2627").set_index("Team").loc["Man City"]

    assert row["form_matches"] == api.FORM_MATCHES
    # Won one, drew two, lost one, drew one across the last five.
    assert row["form_points"] == 3 + 1 + 3 + 0 + 1


def test_opponent_quality_is_absent_rather_than_invented(played: None) -> None:
    """With no feature table there is no Elo, and the column says nothing."""
    row = api._season_form("E0", "2627").set_index("Team").loc["Man City"]

    assert pd.isna(row["form_opponent_elo"])


def test_expected_points_come_from_the_chances_in_each_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A club that took three points from a match its chances made a coin toss.

    The gap is the point of the column: results and performance disagree, and
    a league table only shows the first.
    """
    monkeypatch.setattr(api.state, "tmh", FORM_MATCHES_PLAYED.head(1))
    monkeypatch.setattr(api.state, "features", pd.DataFrame())
    monkeypatch.setattr(
        api.state,
        "retrodicted",
        pd.DataFrame(
            [
                {
                    "Date": pd.Timestamp("2026-08-01"),
                    "HomeTeam": "Man City",
                    "AwayTeam": "Rival A",
                    "prob_home": 0.4,
                    "prob_draw": 0.3,
                    "prob_away": 0.3,
                }
            ]
        ),
    )

    row = api._season_form("E0", "2627").set_index("Team").loc["Man City"]

    assert row["expected_points"] == pytest.approx(3 * 0.4 + 0.3)
    assert row["points_taken"] == 3
    assert row["xg_matches"] == 1


def test_a_league_season_with_nothing_played_has_no_form(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api.state, "tmh", FORM_MATCHES_PLAYED)
    monkeypatch.setattr(api.state, "features", pd.DataFrame())
    monkeypatch.setattr(api.state, "retrodicted", pd.DataFrame())

    assert api._season_form("E0", "2526").empty


GEM_PLAYERS = pd.DataFrame(
    [
        # Rated far above EA, and EA agrees a year later.
        ("E0", "2324", "Cole Palmer", "Chelsea", "AMW", 2640, 73.0, 66.0, 52.0, 94.0),
        ("E0", "2425", "Cole Palmer", "Chelsea", "AMW", 2600, 85.0, 85.0, 85.0, 85.0),
        # Rated below EA.
        ("E0", "2324", "Someone Else", "Everton", "CB", 2500, 70.0, 80.0, 80.0, 55.0),
        ("E0", "2425", "Someone Else", "Everton", "CB", 2500, 70.0, 78.0, 78.0, 60.0),
        # Too few minutes to be published.
        ("E0", "2324", "Cameo", "Burnley", "FWD", 500, 70.0, 60.0, 50.0, 95.0),
        # A season whose performance half is not comparable.
        ("E0", "1415", "Old Timer", "Villa", "MID", 3000, 70.0, 60.0, 50.0, 95.0),
    ],
    columns=[
        "League", "Season", "Player", "Team", "role", "minutes",
        "rating", "fifa_overall", "fifa_on_our_scale", "performance_rating",
    ],
).assign(rated=True)
GEM_PLAYERS["fifa_name"] = GEM_PLAYERS["Player"]


@pytest.fixture
def gems_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api.state, "players", GEM_PLAYERS)
    monkeypatch.setattr(api.state, "_gems", None)
    return TestClient(api.app)


def test_the_board_leads_with_the_biggest_gap(gems_client: TestClient) -> None:
    payload = gems_client.get("/insights/hidden-gems").json()

    assert payload["players"][0]["player"] == "Cole Palmer"
    assert payload["players"][0]["gap"] == pytest.approx(42.0)


def test_each_row_carries_eas_own_later_verdict(gems_client: TestClient) -> None:
    """The row is the test, not an illustration of one."""
    row = gems_client.get("/insights/hidden-gems").json()["players"][0]

    assert row["ea_next"] == 85
    assert row["ea_move"] == 19


def test_the_thin_and_the_incomparable_are_absent(gems_client: TestClient) -> None:
    named = {p["player"] for p in gems_client.get("/insights/hidden-gems").json()["players"]}

    assert "Cameo" not in named, "a 500-minute season is too thin for a gap"
    assert "Old Timer" not in named, "1415 served no misc table"


def test_the_page_is_told_what_was_excluded(gems_client: TestClient) -> None:
    """A board that silently drops seasons invites the reader to infer a trend."""
    payload = gems_client.get("/insights/hidden-gems").json()

    assert payload["excluded_seasons"] == ["1415", "1516"]
    assert payload["min_minutes"] == 900


def test_a_role_filter_narrows_the_board(gems_client: TestClient) -> None:
    payload = gems_client.get("/insights/hidden-gems", params={"role": "CB"}).json()

    assert {p["role"] for p in payload["players"]} == {"CB"}


def test_the_claim_is_absent_rather_than_invented_on_thin_data(
    gems_client: TestClient,
) -> None:
    """Six players cannot support a correlation, and none is published."""
    assert gems_client.get("/insights/hidden-gems").json()["claim"] is None


def test_no_ratings_is_a_404_naming_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api.state, "players", pd.DataFrame())
    monkeypatch.setattr(api.state, "_gems", None)
    response = TestClient(api.app).get("/insights/hidden-gems")

    assert response.status_code == 404
    assert "build_players" in response.json()["detail"]


ACH_STRENGTH = pd.DataFrame(
    [("E0", "2324", f"club {i}", 60.0 + i) for i in range(12)]
    + [("E0", "2425", f"club {i}", 60.0 + i) for i in range(12)],
    columns=["League", "Season", "Team", "strength_overall"],
)

#: Points sit exactly on a line through the ratings, except club 0 in 2324,
#: which took thirty more than its squad implies.
#: Points sit on a line through the ratings, except club 0 in 2324, which won
#: ten more than its squad implies. Seasons are long enough to count: a points
#: total from a part-played season is not comparable with a finished one.
def _season_matches(season: str) -> pd.DataFrame:
    rows = []
    for i in range(12):
        wins = i + (10 if season == "2324" and i == 0 else 0)
        for match in range(24):
            won = match < wins
            rows.append(
                {
                    "League": "E0",
                    "Season": season,
                    "Team": f"club {i}",
                    "GoalsFor": 3 if won else 0,
                    "GoalsAgainst": 0 if won else 1,
                }
            )
    return pd.DataFrame(rows)


ACH_MATCHES = pd.concat(
    [_season_matches("2324"), _season_matches("2425")], ignore_index=True
)


@pytest.fixture
def achievement_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api.state, "teams", ACH_STRENGTH)
    monkeypatch.setattr(api.state, "tmh", ACH_MATCHES)
    monkeypatch.setattr(api.state, "_achievement", None)
    return TestClient(api.app)


def test_the_overachiever_leads_the_board(achievement_client: TestClient) -> None:
    payload = achievement_client.get("/insights/achievement").json()

    top = payload["over"][0]
    assert top["team"] == "club 0"
    assert top["season"] == "2324"
    assert top["over"] > 0


def test_every_row_carries_the_clubs_own_run(achievement_client: TestClient) -> None:
    """Without it, one large residual reads as a verdict on the club."""
    top = achievement_client.get("/insights/achievement").json()["over"][0]

    assert [point["season"] for point in top["history"]] == ["2324", "2425"]


def test_the_scale_the_numbers_sit_on_is_published(
    achievement_client: TestClient,
) -> None:
    payload = achievement_client.get("/insights/achievement").json()

    assert payload["club_seasons"] == 24
    assert payload["residual_sd"] > 0


def test_both_ends_come_back_and_disagree_in_sign(
    achievement_client: TestClient,
) -> None:
    payload = achievement_client.get("/insights/achievement").json()

    assert payload["over"][0]["over"] > 0
    assert payload["under"][0]["over"] < 0


def test_no_team_ratings_is_a_404_naming_the_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api.state, "teams", pd.DataFrame())
    monkeypatch.setattr(api.state, "_achievement", None)
    response = TestClient(api.app).get("/insights/achievement")

    assert response.status_code == 404
    assert "build_players" in response.json()["detail"]
