"""Tests for the team-strength join.

This leak is invisible to ``test_leakage.py::test_truncation_invariance``. That
test removes future *matches* and checks the features do not move; team strength
arrives from a separate file, so truncating matches would not change it at all
and the test would pass on thoroughly leaky data.

The guard therefore has to be asserted directly: a match must never see strength
computed from its own season or any later one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.features.build import _next_season, add_team_strength


@pytest.fixture
def strength() -> pd.DataFrame:
    """Two seasons of strength, with values that identify their season."""
    rows = []
    for season, base in (("2425", 70.0), ("2526", 80.0)):
        for team in ("Arsenal", "Chelsea"):
            rows.append(
                {
                    "League": "E0",
                    "Season": season,
                    "Team": team,
                    "strength_overall": base,
                    "strength_attack": base + 1,
                    "strength_defence": base + 2,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "League": ["E0", "E0"],
            "Season": ["2526", "2425"],
            "Date": pd.to_datetime(["2025-10-01", "2024-10-01"]),
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea"],
        }
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [("2425", "2526"), ("2526", "2627"), ("1920", "2021"), ("2021", "2122")],
)
def test_next_season(label: str, expected: str) -> None:
    """Season labels roll over correctly, including across the century mark."""
    assert _next_season(label) == expected


def test_match_never_sees_its_own_season(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """The defining guard: a 2025/26 match must use 2024/25 strength.

    Values are seeded so the season is recoverable from the number -- 70 means
    2024/25, 80 means 2025/26. A 2025/26 match showing 80 would mean
    end-of-season squad data was used to predict an October fixture.
    """
    joined = add_team_strength(matches, strength, previous_season=True)

    current = joined[joined["Season"] == "2526"].iloc[0]
    assert current["home_strength_overall"] == 70.0, (
        "a 2025/26 match must see 2024/25 strength, not its own season's"
    )
    assert current["away_strength_overall"] == 70.0


def test_earliest_season_gets_no_strength(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """With nothing before it, the first season must be NaN rather than its own.

    Falling back to the match's own season here would be the leak arriving by a
    side door, and would look like the feature simply had better coverage.
    """
    joined = add_team_strength(matches, strength, previous_season=True)
    earliest = joined[joined["Season"] == "2425"].iloc[0]
    assert pd.isna(earliest["home_strength_overall"])


def test_live_mode_uses_the_current_season(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """Live prediction is the one legitimate exception: today's squad plays next."""
    joined = add_team_strength(matches, strength, previous_season=False)
    current = joined[joined["Season"] == "2526"].iloc[0]
    assert current["home_strength_overall"] == 80.0


def test_attack_and_defence_are_carried_separately(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """Attack and defence must stay distinct -- they feed opposite goal rates."""
    joined = add_team_strength(matches, strength, previous_season=True)
    row = joined[joined["Season"] == "2526"].iloc[0]

    assert row["home_strength_attack"] == 71.0
    assert row["home_strength_defence"] == 72.0
    assert row["away_strength_attack"] == 71.0


def test_unknown_team_gets_nan_not_a_wrong_row(strength: pd.DataFrame) -> None:
    """A promoted side with no strength row must come back empty, not mismatched."""
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Luton"],
            "AwayTeam": ["Chelsea"],
        }
    )
    joined = add_team_strength(matches, strength, previous_season=True)
    assert pd.isna(joined["home_strength_overall"].iloc[0])
    assert joined["away_strength_overall"].iloc[0] == 70.0


def test_missing_strength_file_is_survivable() -> None:
    """An empty strength frame must leave the matches untouched, not raise.

    The match pipeline has to run before `build_players` ever has.
    """
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
        }
    )
    joined = add_team_strength(matches, pd.DataFrame(), previous_season=True)
    assert len(joined) == 1
    assert "home_strength_overall" not in joined.columns


def test_join_does_not_duplicate_matches(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """A many-to-one join that silently fans out would corrupt every metric."""
    assert len(add_team_strength(matches, strength, previous_season=True)) == len(matches)
