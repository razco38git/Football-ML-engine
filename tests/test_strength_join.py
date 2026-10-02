"""Squad strength must reach both sides, including a cross-league pairing.

`add_team_strength` used to key on league as well as season and team. For a
real fixture that is harmless -- both sides are in the fixture's league -- but
`/predict` scores a hypothetical tie in the *home* side's league, so the away
side matched nothing and was scored with no squad strength at all. Roughly a
fifth of the model's influence on expected goals, silently missing, with only a
note on the page to say so.

Dropping league from the key is safe because a club plays in one league per
season: no team name appears in two leagues, and no (season, team) pair repeats.
Both properties are asserted below, since the join's correctness rests on them.
"""

from __future__ import annotations

import pandas as pd

from footballml.data import load_team_strength
from footballml.features.build import add_team_strength


def _strength() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": "E0", "Season": "2526", "Team": "Man City", "strength_overall": 80.9,
             "strength_attack": 83.7, "strength_defence": 77.3, "strength_goalkeeper": 88.0},
            {"League": "SP1", "Season": "2526", "Team": "Real Madrid", "strength_overall": 81.6,
             "strength_attack": 83.8, "strength_defence": 79.4, "strength_goalkeeper": 84.0},
        ]
    )


def _match(league: str, home: str, away: str) -> pd.DataFrame:
    return pd.DataFrame([{"League": league, "Season": "2627", "HomeTeam": home, "AwayTeam": away}])


def test_a_cross_league_pairing_gets_strength_for_both_sides() -> None:
    """Real Madrid at home to Man City is scored in SP1; City is a Premier
    League club and must still bring its rating."""
    out = add_team_strength(
        _match("SP1", "Real Madrid", "Man City"), _strength(), previous_season=True
    )
    assert out["home_strength_overall"].iloc[0] == 81.6
    assert out["away_strength_overall"].iloc[0] == 80.9


def test_a_same_league_fixture_is_unaffected() -> None:
    out = add_team_strength(
        _match("E0", "Man City", "Real Madrid"), _strength(), previous_season=True
    )
    assert out["home_strength_overall"].iloc[0] == 80.9


def test_the_join_does_not_duplicate_rows() -> None:
    """A second row for the same (season, team) would multiply the match out."""
    out = add_team_strength(
        _match("SP1", "Real Madrid", "Man City"), _strength(), previous_season=True
    )
    assert len(out) == 1


def test_an_unrated_team_still_gets_nan_rather_than_a_wrong_row() -> None:
    out = add_team_strength(
        _match("E0", "Man City", "Nowhere FC"), _strength(), previous_season=True
    )
    assert pd.isna(out["away_strength_overall"].iloc[0])


# --- the properties the key rests on ---------------------------------------


def test_no_club_appears_in_two_leagues() -> None:
    """Dropping league from the join key is only safe while this holds."""
    real = load_team_strength()
    leagues = real.groupby("Team")["League"].nunique()
    assert leagues.max() == 1, sorted(leagues[leagues > 1].index)


def test_season_and_team_identify_one_row() -> None:
    real = load_team_strength()
    assert not real.duplicated(subset=["Season", "Team"]).any()
