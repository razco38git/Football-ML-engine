"""The squad-aware rating: what it reads, and what it must never read."""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.players.squad_rating import (
    MIN_CURVE_POINTS,
    _latest_before,
    rosters,
    squad_strength,
    weight_by_line,
)
from footballml.players.team_strength import LINE_BY_ROLE

#: A roster big enough to clear `team_strength`'s eleven-player floor, with
#: every line filled so the overall is not computed off one of them.
_LINES = ["GK"] + ["CB"] * 5 + ["FB"] * 4 + ["DM"] * 2 + ["MID"] * 4 + ["AMW"] * 3 + ["FWD"] * 3
_POSITION = {
    "GK": "GK",
    "CB": "CB",
    "FB": "LB",
    "DM": "CDM",
    "MID": "CM",
    "AMW": "RW",
    "FWD": "ST",
}


def write_edition(directory, season: str, squads: dict[str, list[tuple[str, int, str]]]) -> None:
    """One edition on disk: ``{label: [(name, overall, role), ...]}``."""
    rows = [
        {
            "name": name,
            "overall": overall,
            "club": label,
            "positions": _POSITION[role],
        }
        for label, squad in squads.items()
        for name, overall, role in squad
    ]
    pd.DataFrame(rows).to_csv(directory / f"fifa_{season}.csv", index=False)


def squad(prefix: str, overalls: list[int] | None = None) -> list[tuple[str, int, str]]:
    """A full-shaped squad of made-up players, named from ``prefix``."""
    return [
        (f"{prefix} {i}", (overalls[i] if overalls else 70), role) for i, role in enumerate(_LINES)
    ]


def rated_rows(*rows: dict) -> pd.DataFrame:
    """Our own rated player-seasons, with the columns the module reads."""
    frame = pd.DataFrame(rows)
    defaults = {
        "League": "E0",
        "rated": True,
        "minutes": 2000.0,
        "role": "MID",
        "fifa_overall": 70.0,
        "fifa_on_our_scale": 70.0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    frame["fifa_name"] = frame.get("fifa_name", frame["Player"])
    frame["rating"] = frame["rating"].astype(float)
    return frame


def votes(season: str, team: str, label: str, names: list[str], **extra) -> list[dict]:
    """Enough rated players at one club to resolve its EA label by vote."""
    return [
        {
            "Season": season,
            "Team": team,
            "Player": name,
            "fifa_name": name,
            "fifa_club": label,
            "rating": 70.0,
            **extra,
        }
        for name in names
    ]


def curve_rows(season: str) -> list[dict]:
    """Published `(fifa_overall, fifa_on_our_scale)` pairs for one season.

    Enough distinct overalls to clear :data:`MIN_CURVE_POINTS`, mapping EA's
    60-80 onto 50-90 so the two scales are visibly different -- which is the
    whole reason the mapping exists.
    """
    return [
        {
            "Season": season,
            "Team": "Scale Pool",
            "Player": f"pool {season} {overall}",
            "fifa_name": f"pool {season} {overall}",
            "fifa_club": "Scale Pool FC",
            "rating": 70.0,
            "role": role,
            "fifa_overall": float(overall),
            "fifa_on_our_scale": float(2 * overall - 70),
        }
        for role in sorted(LINE_BY_ROLE)
        for overall in range(60, 81, 4)
    ]


@pytest.fixture
def editions(tmp_path):
    """Two editions in which one player moves club between them."""
    write_edition(
        tmp_path,
        "2021",
        {"FC Old": squad("old"), "FC New": squad("new")},
    )
    write_edition(
        tmp_path,
        "2122",
        {
            "FC Old": squad("old")[:-1] + [("mover 0", 90, "FWD")],
            "FC New": squad("new"),
        },
    )
    return tmp_path


def test_the_roster_is_eas_not_ours(editions) -> None:
    """A player EA puts elsewhere is not in the squad, however he is rated.

    This is the whole point of the module: the roster comes from the edition,
    so a departure is absent and an arrival is present without either being
    computed.
    """
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        # Rated at a club outside our five leagues last season, so he is in the
        # roster and in no vote of ours.
        *votes("2021", "Elsewhere", "FC Nowhere", ["mover 0"]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )

    roster, _ = rosters(players, editions)
    old = roster[(roster["Season"] == "2122") & (roster["Team"] == "Old")]

    assert "mover 0" in set(old["Player"])
    assert squad("old")[-1][0] not in set(old["Player"])


def test_a_prior_season_rating_is_used_before_eas_overall(editions) -> None:
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )
    players.loc[players["Player"] == "old 0", "rating"] = 88.0

    roster, _ = rosters(players, editions)
    keeper = roster[(roster["Season"] == "2122") & (roster["Player"] == "old 0")]

    assert float(keeper["rating"].iloc[0]) == 88.0
    assert bool(keeper["from_history"].iloc[0])


def test_a_player_we_have_never_rated_comes_through_the_published_curve(
    editions,
) -> None:
    """EA's 90 is read off our own published mapping, not used raw."""
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )

    roster, _ = rosters(players, editions)
    mover = roster[(roster["Season"] == "2122") & (roster["Player"] == "mover 0")]

    assert not bool(mover["from_history"].iloc[0])
    # The curve maps 60-80 onto 50-90 and clamps above its own range, so EA's
    # 90 is published as the 80-rated player's 90 rather than extrapolated.
    assert float(mover["rating"].iloc[0]) == pytest.approx(90.0)


def test_the_curve_clamps_rather_than_extrapolating(editions) -> None:
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )
    roster, _ = rosters(players, editions)
    mover = roster[(roster["Season"] == "2122") & (roster["Player"] == "mover 0")]
    # The pool's best is EA 80, published at 90. A 90 cannot exceed it.
    assert float(mover["rating"].iloc[0]) <= 90.0


def test_a_thin_role_pool_falls_back_to_the_seasons_own_curve(editions) -> None:
    """2015/16 keepers are unrated, so their role has no curve of its own."""
    rows = curve_rows("2122")
    thin = [r for r in rows if r["role"] != "FWD"]
    thin += [r for r in rows if r["role"] == "FWD"][: MIN_CURVE_POINTS - 1]

    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *thin,
    )

    roster, _ = rosters(players, editions)
    mover = roster[(roster["Season"] == "2122") & (roster["Player"] == "mover 0")]

    assert mover["rating"].notna().all()


def test_the_rating_never_reads_the_season_it_describes(editions) -> None:
    """The leakage property, as `test_strength_leakage` states it for the model.

    A rating published for season S must not be built from S's own results. A
    player rated 99 in 2122 keeps his 2021 rating in the 2122 squad.
    """
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )
    players.loc[(players["Player"] == "old 0") & (players["Season"] == "2122"), "rating"] = 99.0

    roster, _ = rosters(players, editions)
    keeper = roster[(roster["Season"] == "2122") & (roster["Player"] == "old 0")]

    assert float(keeper["rating"].iloc[0]) == 70.0


def test_include_current_reads_it_and_is_not_the_default(editions) -> None:
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )
    players.loc[(players["Player"] == "old 0") & (players["Season"] == "2122"), "rating"] = 99.0

    roster, _ = rosters(players, editions, include_current=True)
    keeper = roster[(roster["Season"] == "2122") & (roster["Player"] == "old 0")]

    assert float(keeper["rating"].iloc[0]) == 99.0


def test_an_unresolved_club_season_gets_no_roster(editions) -> None:
    """Two votes do not resolve a label, so there is no squad to rate."""
    players = rated_rows(
        *votes("2122", "Ghost", "FC Old", ["old 0", "old 1"]),
        *curve_rows("2122"),
    )

    roster, resolutions = rosters(players, editions)

    assert "Ghost" not in set(roster.get("Team", pd.Series(dtype=object)))
    assert [r.resolved for r in resolutions if r.team == "Ghost"] == [False]


def test_prior_minutes_are_the_weight_and_unknowns_take_their_lines_mean() -> None:
    roster = pd.DataFrame(
        {
            "Season": ["2122"] * 4,
            "Team": ["Old"] * 4,
            "role": ["CB", "CB", "CB", "FWD"],
            "minutes": [1000.0, 2000.0, None, None],
        }
    )

    weights = weight_by_line(roster, LINE_BY_ROLE)

    assert list(weights[:3]) == [1000.0, 2000.0, 1500.0]
    # Nobody in the attack has prior minutes, which `_weighted_mean` reads as
    # "no weights" and averages plainly.
    assert weights.iloc[3] == 0.0


def test_squad_strength_publishes_one_row_per_club_season(editions) -> None:
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2021", "New", "FC New", [name for name, _, _ in squad("new")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *votes("2122", "New", "FC New", [name for name, _, _ in squad("new")]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )

    strength, _ = squad_strength(players, editions, best_n=15)

    # `Scale Pool` exists only to carry the published mapping; its made-up EA
    # club is in no edition, so it resolves to nothing and is rated not at all.
    assert set(strength["Team"]) == {"Old", "New"}
    for column in ("squad_overall", "squad_defence", "squad_attack", "n_from_history"):
        assert column in strength.columns
    assert (strength["n_players"] == 15).all()
    assert "method" not in strength.columns


def test_n_from_history_counts_the_selected_squad(editions) -> None:
    """Not the whole roster: the column says how much of the *rating* is ours."""
    players = rated_rows(
        *votes("2021", "Old", "FC Old", [name for name, _, _ in squad("old")]),
        *votes("2122", "Old", "FC Old", [name for name, _, _ in squad("old")[:-1]]),
        *curve_rows("2021"),
        *curve_rows("2122"),
    )

    strength, _ = squad_strength(players, editions, best_n=15)
    old = strength[(strength["Season"] == "2122") & (strength["Team"] == "Old")]

    assert int(old["n_from_history"].iloc[0]) == 14


def test_latest_before_takes_the_most_recent_and_stops_at_the_season() -> None:
    seasons = [("1920", 60.0, 100.0), ("2021", 70.0, 200.0), ("2122", 80.0, 300.0)]

    assert _latest_before(seasons, "2122") == (70.0, 200.0)
    assert _latest_before(seasons, "2122", include_current=True) == (80.0, 300.0)
    assert _latest_before(seasons, "1920") is None
