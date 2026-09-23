"""Tests for the season projection.

The fixture list is *reconstructed* rather than fetched, so the assumption it
rests on -- that a league season is an exact double round-robin -- is asserted
here against real data. If a league ever changes format, this fails loudly
instead of silently projecting a season with missing matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.data import PROCESSED_DIR, load_team_match_history
from footballml.projection import current_table, remaining_fixtures, simulate


@pytest.fixture(scope="module")
def tmh() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_match_history_all.csv"
    if not path.exists():
        pytest.skip("run `python -m pipelines.build_dataset` first")
    return load_team_match_history(path)


def _two_team_history() -> pd.DataFrame:
    """One played match between two teams, in long form."""
    return pd.DataFrame(
        [
            {"League": "E0", "Season": "2627", "Date": pd.Timestamp("2026-08-01"),
             "Team": "A", "Opponent": "B", "Venue": "Home",
             "GoalsFor": 2, "GoalsAgainst": 1},
            {"League": "E0", "Season": "2627", "Date": pd.Timestamp("2026-08-01"),
             "Team": "B", "Opponent": "A", "Venue": "Away",
             "GoalsFor": 1, "GoalsAgainst": 2},
        ]
    )


def test_a_completed_season_has_nothing_left(tmh: pd.DataFrame) -> None:
    """2025/26 is finished, so every pairing has been played."""
    for league in ("E0", "D1", "SP1", "I1", "F1"):
        assert remaining_fixtures(tmh, league, "2526").empty, league


def test_remaining_counts_match_the_round_robin(tmh: pd.DataFrame) -> None:
    """Teams x (teams - 1), minus what has been played."""
    for league in ("E0", "D1", "SP1", "I1", "F1"):
        rows = tmh[(tmh["League"] == league) & (tmh["Season"].astype(str) == "2627")]
        teams = rows["Team"].nunique()
        played = len(rows) // 2
        assert len(remaining_fixtures(tmh, league, "2627")) == teams * (teams - 1) - played


def test_a_repeated_pairing_is_refused() -> None:
    """Two matches with the same home side means this is not a round-robin."""
    doubled = pd.concat([_two_team_history(), _two_team_history()], ignore_index=True)
    with pytest.raises(ValueError, match="repeated pairing"):
        remaining_fixtures(doubled, "E0", "2627")


def test_the_return_fixture_is_still_to_come() -> None:
    history = _two_team_history()
    left = remaining_fixtures(history, "E0", "2627")
    assert len(left) == 1
    assert left.iloc[0]["HomeTeam"] == "B" and left.iloc[0]["AwayTeam"] == "A"


def test_current_table_counts_points() -> None:
    table = current_table(_two_team_history(), "E0", "2627").set_index("Team")
    assert table.loc["A", "points"] == 3
    assert table.loc["B", "points"] == 0
    assert table.loc["A", "goal_difference"] == 1


def _certain_home_win(n_fixtures: int, size: int = 6) -> np.ndarray:
    """Matrices where 1-0 is the only possible score."""
    matrices = np.zeros((n_fixtures, size, size))
    matrices[:, 1, 0] = 1.0
    return matrices


def test_points_add_up_exactly() -> None:
    """A guaranteed home win must be worth three points, every time."""
    history = _two_team_history()
    fixtures = remaining_fixtures(history, "E0", "2627")
    table = current_table(history, "E0", "2627")

    out = simulate(fixtures, _certain_home_win(len(fixtures)), table, n_sims=50).set_index("Team")
    # A won the first match; B is certain to win the return.
    assert out.loc["A", "projected_points"] == 3
    assert out.loc["B", "projected_points"] == 3
    assert out.loc["A", "points_low"] == out.loc["A", "points_high"] == 3


def test_the_same_seed_gives_the_same_table(tmh: pd.DataFrame) -> None:
    """The API serves a stored file; a projection that drifted between
    identical runs would look like a real change in the forecast."""
    fixtures = remaining_fixtures(tmh, "E0", "2627").head(20)
    table = current_table(tmh, "E0", "2627")
    matrices = np.full((len(fixtures), 4, 4), 1 / 16)

    first = simulate(fixtures, matrices, table, n_sims=200, seed=3)
    second = simulate(fixtures, matrices, table, n_sims=200, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_probabilities_are_shares_of_the_league() -> None:
    """Exactly one team wins each simulated season, so the title chances sum to 1."""
    history = _two_team_history()
    fixtures = remaining_fixtures(history, "E0", "2627")
    table = current_table(history, "E0", "2627")
    matrices = np.full((len(fixtures), 4, 4), 1 / 16)

    out = simulate(fixtures, matrices, table, n_sims=500, relegation_places=1)
    assert out["title_pct"].sum() == pytest.approx(1.0, abs=0.001)
    assert out["relegation_pct"].sum() == pytest.approx(1.0, abs=0.001)


def test_every_team_plays_a_full_season(tmh: pd.DataFrame) -> None:
    """Played plus remaining must equal the full fixture count for each side."""
    for league in ("E0", "D1"):
        table = current_table(tmh, league, "2627")
        fixtures = remaining_fixtures(tmh, league, "2627")
        teams = len(table)
        appearances = (
            fixtures["HomeTeam"].value_counts() + fixtures["AwayTeam"].value_counts()
        )
        total = table.set_index("Team")["played"] + appearances
        assert (total == 2 * (teams - 1)).all(), league
