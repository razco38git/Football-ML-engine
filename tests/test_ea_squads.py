"""The EA club-label resolver: each test is a case that broke the naive rule."""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.players.ea_squads import MIN_VOTES, is_reserve, resolve


def write_edition(directory, season: str, squads: dict[str, int]) -> None:
    """One edition on disk: ``{label: squad size}``, names made unique."""
    rows = [
        {"name": f"{label} player {i}", "overall": 70, "club": label, "positions": "CM"}
        for label, size in squads.items()
        for i in range(size)
    ]
    pd.DataFrame(rows).to_csv(directory / f"fifa_{season}.csv", index=False)


def players(*rows: tuple[str, str, str]) -> pd.DataFrame:
    """Rated player-seasons as ``(season, team, fifa_club)`` triples."""
    return pd.DataFrame(
        [{"Season": s, "Team": t, "fifa_club": c} for s, t, c in rows]
    )


@pytest.mark.parametrize(
    "label",
    [
        "FC Barcelona B",
        "Villarreal II",
        "FC Bayern München II",
        "Real Sociedad B",
        "Sevilla Atlético",
        "Hannover 96 II",
    ],
)
def test_reserve_sides_are_recognised(label: str) -> None:
    assert is_reserve(label)


@pytest.mark.parametrize(
    "label",
    [
        # Ends in a Roman two and is a first team in its own right.
        "Willem II",
        # Contain a bare B or II mid-name, which must not count.
        "Bayer 04 Leverkusen",
        "Borussia Dortmund",
        "Brighton & Hove Albion",
        "FC Barcelona",
    ],
)
def test_first_teams_are_not_mistaken_for_reserves(label: str) -> None:
    assert not is_reserve(label)


def test_vote_resolves_a_club_with_enough_agreement(tmp_path) -> None:
    write_edition(tmp_path, "2122", {"Paris Saint-Germain": 30})
    frame = players(*[("2122", "Paris SG", "Paris Saint-Germain")] * MIN_VOTES)

    [result] = resolve(frame, tmp_path)

    assert result.label == "Paris Saint-Germain"
    assert result.how == "vote"


def test_a_club_absent_from_the_edition_resolves_to_nothing(tmp_path) -> None:
    """Spezia 2021: 22 players scattered one vote each, because EA has no Spezia.

    The naive rule took the first name out of that tie and rated Spezia's squad
    as Milan's. One vote is the absence of the club, not a plurality.
    """
    write_edition(tmp_path, "2021", {"AC Milan": 30, "Cagliari": 30, "Inter": 30})
    frame = players(
        ("2021", "Spezia", "AC Milan"),
        ("2021", "Spezia", "Cagliari"),
        ("2021", "Spezia", "Inter"),
    )

    [result] = resolve(frame, tmp_path)

    assert result.label is None
    assert not result.resolved
    assert "1 vote" in result.how


def test_history_outranks_a_thin_vote(tmp_path) -> None:
    """PSG 2627: a six-player sample voted it Rennes' label.

    The club was `Paris SG` last edition and this edition still has a
    `Paris SG`, which is better evidence than the sample.
    """
    write_edition(tmp_path, "2526", {"Paris SG": 30, "Stade Rennais FC": 30})
    write_edition(tmp_path, "2627", {"Paris SG": 29, "Stade Rennais FC": 28})
    frame = players(
        *[("2526", "Paris Saint Germain", "Paris SG")] * 10,
        # Thin, and wrong: more of the matched players sit at Rennes.
        *[("2627", "Paris Saint Germain", "Stade Rennais FC")] * 7,
        *[("2627", "Paris Saint Germain", "Paris SG")] * 6,
    )

    results = {(r.season, r.team): r for r in resolve(frame, tmp_path)}

    assert results[("2627", "Paris Saint Germain")].label == "Paris SG"
    assert results[("2627", "Paris Saint Germain")].how == "history"


def test_history_is_ignored_when_the_label_is_gone(tmp_path) -> None:
    """EA renames clubs, so last edition's label may not exist in this one."""
    write_edition(tmp_path, "2425", {"Paris SG": 30})
    write_edition(tmp_path, "2526", {"Paris Saint-Germain": 30})
    frame = players(
        *[("2425", "Paris Saint Germain", "Paris SG")] * 5,
        *[("2526", "Paris Saint Germain", "Paris Saint-Germain")] * 5,
    )

    results = {r.season: r for r in resolve(frame, tmp_path)}

    assert results["2526"].label == "Paris Saint-Germain"
    assert results["2526"].how == "vote"


def test_two_clubs_cannot_hold_one_label(tmp_path) -> None:
    """The weaker claim yields and re-resolves to its own best candidate."""
    write_edition(tmp_path, "2122", {"Inter": 30, "AC Milan": 30})
    frame = players(
        *[("2122", "Inter", "Inter")] * 20,
        # Milan's players mostly point at Inter -- it must not win it.
        *[("2122", "AC Milan", "Inter")] * 8,
        *[("2122", "AC Milan", "AC Milan")] * 5,
    )

    results = {r.team: r for r in resolve(frame, tmp_path)}

    assert results["Inter"].label == "Inter"
    assert results["AC Milan"].label == "AC Milan"
    assert results["AC Milan"].how == "vote after yielding"


def test_yielding_with_no_second_candidate_resolves_to_nothing(tmp_path) -> None:
    write_edition(tmp_path, "2122", {"Inter": 30})
    frame = players(
        *[("2122", "Inter", "Inter")] * 20,
        *[("2122", "Internazionale", "Inter")] * 4,
    )

    results = {r.team: r for r in resolve(frame, tmp_path)}

    assert results["Inter"].label == "Inter"
    assert results["Internazionale"].label is None
    assert "yielded" in results["Internazionale"].how


def test_a_reserve_side_never_wins_even_with_every_vote(tmp_path) -> None:
    write_edition(tmp_path, "2122", {"FC Barcelona": 30, "FC Barcelona B": 25})
    frame = players(*[("2122", "Barcelona", "FC Barcelona B")] * 20)

    [result] = resolve(frame, tmp_path)

    assert result.label is None


def test_labels_from_another_edition_do_not_vote(tmp_path) -> None:
    """A player matched across editions carries a label this one may not have."""
    write_edition(tmp_path, "2122", {"Paris Saint-Germain": 30})
    frame = players(*[("2122", "Paris SG", "Paris SG")] * 10)

    [result] = resolve(frame, tmp_path)

    assert result.label is None


def test_a_missing_edition_leaves_every_club_unresolved(tmp_path) -> None:
    frame = players(*[("1314", "Barcelona", "FC Barcelona")] * 20)

    [result] = resolve(frame, tmp_path)

    assert result.label is None
