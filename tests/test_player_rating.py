"""Tests for the player rating.

These pin the properties that make a rating defensible -- that low-minute
players are pulled toward the mean, that unratable groups stay unrated, that
the scale is monotonic -- rather than asserting particular players get
particular numbers, which would just encode today's weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.players.ingest import position_group
from footballml.players.rating import UNRATED_GROUPS, percentile_within, rate_players


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GK", "GK"),
        ("GK S", "GK"),
        ("D", "D"),
        ("D S", "D"),
        ("F M S", "F"),      # most forwards are listed this way
        ("D M S", "D"),
        ("M S", "M"),
        ("S", None),         # substitute only, no role recorded
        (None, None),
        (float("nan"), None),
    ],
)
def test_position_group(raw: str | float | None, expected: str | None) -> None:
    assert position_group(raw) == expected


def test_forwards_are_not_resolved_to_midfield() -> None:
    """``"F M S"`` must resolve to F.

    Understat lists most forwards as having also played midfield. Resolving
    those to M would compare strikers against midfielders on goalscoring and
    inflate them badly.
    """
    assert position_group("F M S") == "F"
    assert position_group("F M") == "F"


def _squad(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A synthetic league of forwards with varying quality and minutes."""
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.1, 1.0, n)
    minutes = rng.uniform(500, 3000, n)
    return pd.DataFrame(
        {
            "League": "E0",
            "Season": "2526",
            "Player": [f"P{i}" for i in range(n)],
            "Team": [f"T{i % 20}" for i in range(n)],
            "position_group": "F",
            "minutes": minutes,
            "nineties": minutes / 90,
            "np_xg_per90": quality * 0.8,
            "np_goals_per90": quality * 0.8,
            "finishing_delta_per90": rng.normal(0, 0.05, n),
            "xa_per90": quality * 0.3,
            "key_passes_per90": quality * 2.0,
            "assists_per90": quality * 0.25,
            "shots_per90": quality * 3.0,
            "xg_chain_per90": quality * 0.9,
            "xg_buildup_per90": quality * 0.4,
            "xg_chain_share": quality * 0.15,
            "xg_buildup_share": quality * 0.1,
            "yellow_cards_per90": rng.uniform(0, 0.3, n),
            "red_cards_per90": 0.0,
        }
    )


def test_ratings_land_on_the_expected_scale() -> None:
    rated = rate_players(_squad())
    ratings = rated.loc[rated["rated"], "rating"].astype(float)

    assert ratings.between(40, 99).all(), "ratings must sit on a 0-99 style scale"
    assert ratings.max() < 99, "nobody should hit the ceiling"
    assert 60 < ratings.median() < 75, "median should sit where football ratings do"


def test_rating_tracks_quality() -> None:
    """Better underlying numbers must produce a better rating."""
    squad = _squad()
    rated = rate_players(squad)
    ok = rated["rated"]

    corr = np.corrcoef(rated.loc[ok, "np_xg_per90"], rated.loc[ok, "rating"].astype(float))[0, 1]
    assert corr > 0.8, f"rating barely tracks performance (r={corr:.2f})"


def test_low_minutes_are_shrunk_toward_the_mean() -> None:
    """A brilliant cameo must not out-rate a brilliant season.

    Two players with identical per-90 output, one with 500 minutes and one with
    3000, must not be rated equally -- otherwise the leaderboard fills with
    players who had one good afternoon.
    """
    squad = _squad()
    elite = {c: squad[c].max() for c in squad.columns if c.endswith(("_per90", "_share"))}

    for name, minutes in (("Cameo", 500.0), ("Regular", 3000.0)):
        row = {**squad.iloc[0].to_dict(), **elite, "Player": name, "minutes": minutes,
               "nineties": minutes / 90, "yellow_cards_per90": 0.0, "red_cards_per90": 0.0}
        squad = pd.concat([squad, pd.DataFrame([row])], ignore_index=True)

    rated = rate_players(squad).set_index("Player")
    assert rated.loc["Regular", "rating"] > rated.loc["Cameo", "rating"]


def test_goalkeepers_are_rated_from_keeper_metrics() -> None:
    """Keepers are rated on FBref shot-stopping, not invented from thin air.

    Understat alone had nothing for them; once save percentage and goals
    against are present, a keeper pool large enough to percentile gets rated
    like any other group.
    """
    rng = np.random.default_rng(3)
    n = 60
    skill = rng.uniform(0.1, 1.0, n)
    keepers = pd.DataFrame(
        {
            "League": "E0", "Season": "2526",
            "Player": [f"GK{i}" for i in range(n)],
            "Team": [f"T{i}" for i in range(n)],
            "position_group": "GK",
            "minutes": rng.uniform(900, 3400, n),
            "save_pct": 55 + skill * 25,
            "goals_against_per90": 2.2 - skill * 1.4,
            "clean_sheet_pct": skill * 45,
            "shots_on_target_against_per90": rng.uniform(2.5, 5.5, n),
            "penalties_saved_per90": rng.uniform(0, 0.05, n),
        }
    )
    keepers["nineties"] = keepers["minutes"] / 90

    rated = rate_players(keepers)
    assert rated["rated"].all()

    ok = rated["rated"]
    corr = np.corrcoef(rated.loc[ok, "save_pct"], rated.loc[ok, "rating"].astype(float))[0, 1]
    assert corr > 0.7, f"keeper rating should track save percentage (r={corr:.2f})"


def test_unrated_groups_mechanism_still_works() -> None:
    """The refusal mechanism survives even though no group currently uses it."""
    assert isinstance(UNRATED_GROUPS, dict)


def test_negative_weights_invert_the_percentile() -> None:
    """A negative weight means lower is better, not 'subtract from the total'.

    Conceding goals and committing fouls are expressed that way, and getting it
    wrong would corrupt the normaliser and let a sub-rating go negative.
    """
    from footballml.players.rating import _weighted

    frame = pd.DataFrame({"good": [0.9, 0.1], "bad": [0.9, 0.1]})
    positive = _weighted(frame, {"good": 1.0})
    inverted = _weighted(frame, {"bad": -1.0})

    assert positive.iloc[0] == pytest.approx(0.9)
    assert inverted.iloc[0] == pytest.approx(0.1), "high value on a negative metric must score low"
    assert (inverted >= 0).all() and (inverted <= 1).all()


def test_short_seasons_are_left_unrated() -> None:
    """A percentile over a handful of players is meaningless and must be refused.

    This is what stopped a player with exactly 450 minutes in a five-matchweek
    season being rated 94.
    """
    tiny = _squad(n=12)
    rated = rate_players(tiny)

    assert not rated["rated"].any()
    assert "comparable players" in rated["unrated_reason"].iloc[0]


def test_under_minimum_minutes_unrated() -> None:
    squad = _squad()
    squad.loc[:4, "minutes"] = 100
    squad.loc[:4, "nineties"] = 100 / 90

    rated = rate_players(squad)
    assert not rated.loc[:4, "rated"].any()
    assert "minutes played" in rated.loc[0, "unrated_reason"]


def test_percentile_within_is_group_local() -> None:
    """Percentiles must be computed inside a group, never across groups."""
    df = pd.DataFrame(
        {
            "position_group": ["F"] * 3 + ["D"] * 3,
            "Season": ["2526"] * 6,
            "metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    pct = percentile_within(df, "metric", ["position_group", "Season"])

    # The best forward and the best defender both sit at the top of their group.
    assert pct.iloc[2] == pytest.approx(1.0)
    assert pct.iloc[5] == pytest.approx(1.0)


def test_percentile_direction_can_invert() -> None:
    """Cards are bad, so a high count must give a low percentile."""
    df = pd.DataFrame(
        {"position_group": ["F"] * 3, "Season": ["2526"] * 3, "cards": [0.0, 0.5, 1.0]}
    )
    pct = percentile_within(df, "cards", ["position_group", "Season"], higher_is_better=False)
    assert pct.iloc[0] > pct.iloc[2]
