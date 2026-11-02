"""The hidden-gems board: what it publishes, and what it refuses to.

Each test is a claim `docs/PROPOSAL-historical-insights.md` makes, or a filter
it demands evidence for, so a change that quietly drops one fails here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.players.gems import (
    INCOMPLETE_SEASONS,
    MIN_MINUTES,
    agreement,
    gaps,
    hidden_gems,
    with_ea_verdict,
)


def player_rows(*rows: dict) -> pd.DataFrame:
    """Rated player-seasons carrying the columns the board reads."""
    frame = pd.DataFrame(rows)
    defaults = {
        "League": "E0",
        "Team": "Somewhere",
        "role": "MID",
        "rated": True,
        "minutes": 2000,
        "rating": 70.0,
        "fifa_overall": 70.0,
        "fifa_on_our_scale": 70.0,
        "performance_rating": 70.0,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
    if "fifa_name" not in frame.columns:
        frame["fifa_name"] = frame["Player"]
    frame["Season"] = frame["Season"].astype(str)
    return frame


def test_the_gap_is_the_two_published_halves_subtracted() -> None:
    frame = gaps(
        player_rows(
            {"Season": "2324", "Player": "someone",
             "performance_rating": 94.0, "fifa_on_our_scale": 52.0},
        )
    )
    assert float(frame["gap"].iloc[0]) == pytest.approx(42.0)


def test_a_season_a_player_did_not_clear_is_not_on_the_board() -> None:
    """The tails of a difference fill with players who barely played."""
    board, _ = hidden_gems(
        player_rows(
            {"Season": "2324", "Player": "thin", "minutes": MIN_MINUTES - 1,
             "performance_rating": 94.0, "fifa_on_our_scale": 52.0},
            {"Season": "2324", "Player": "solid", "minutes": MIN_MINUTES,
             "performance_rating": 88.0, "fifa_on_our_scale": 60.0},
        )
    )
    assert list(board["Player"]) == ["solid"]


@pytest.mark.parametrize("season", sorted(INCOMPLETE_SEASONS))
def test_the_seasons_with_missing_fbref_tables_are_left_out(season: str) -> None:
    """2014/15 served no `misc` table and 2015/16 no `keeper` table.

    `measured_share` falls to 0.38 for a centre back in them, so the
    performance half is not comparable with a modern one and a gap built from
    it is not either.
    """
    board, _ = hidden_gems(
        player_rows(
            {"Season": season, "Player": "incomparable",
             "performance_rating": 94.0, "fifa_on_our_scale": 50.0},
            {"Season": "2324", "Player": "comparable",
             "performance_rating": 80.0, "fifa_on_our_scale": 60.0},
        )
    )
    assert list(board["Player"]) == ["comparable"]


def test_one_ea_entry_behind_two_clubs_is_refused() -> None:
    """Bruno Saltor and Bruno Gonzalez are both matched to Villarreal's Bruno.

    They are the two largest negative gaps in the whole file. The same shape
    can inflate a positive one, so the guard runs on this board too even
    though it is the polluted tail that needed it.
    """
    frame = gaps(
        player_rows(
            {"Season": "1718", "Player": "Bruno", "Team": "Brighton",
             "fifa_name": "Bruno Soriano Llido", "performance_rating": 48.0,
             "fifa_on_our_scale": 82.0},
            {"Season": "1718", "Player": "Bruno", "Team": "Getafe",
             "fifa_name": "Bruno Soriano Llido", "performance_rating": 49.0,
             "fifa_on_our_scale": 82.0},
        )
    )
    assert frame["shared_entry"].all()


def test_a_shared_entry_never_reaches_the_board() -> None:
    board, _ = hidden_gems(
        player_rows(
            {"Season": "2324", "Player": "mismatched", "Team": "Here",
             "fifa_name": "one entry", "performance_rating": 99.0,
             "fifa_on_our_scale": 40.0},
            {"Season": "2324", "Player": "mismatched", "Team": "There",
             "fifa_name": "one entry", "performance_rating": 99.0,
             "fifa_on_our_scale": 40.0},
            {"Season": "2324", "Player": "genuine", "fifa_name": "genuine",
             "performance_rating": 80.0, "fifa_on_our_scale": 60.0},
        )
    )
    assert list(board["Player"]) == ["genuine"]


def test_one_club_in_a_season_is_not_flagged() -> None:
    frame = gaps(
        player_rows({"Season": "2324", "Player": "solo", "fifa_name": "solo"})
    )
    assert not frame["shared_entry"].any()


def test_the_verdict_is_eas_next_edition_and_only_that() -> None:
    players = player_rows(
        {"Season": "2324", "Player": "palmer", "fifa_overall": 66.0},
        {"Season": "2425", "Player": "palmer", "fifa_overall": 85.0},
        {"Season": "2526", "Player": "palmer", "fifa_overall": 88.0},
    )
    judged = with_ea_verdict(gaps(players), players)
    row = judged[judged["Season"] == "2324"].iloc[0]

    assert float(row["ea_next"]) == 85.0
    assert float(row["ea_move"]) == 19.0


def test_no_next_entry_is_absent_rather_than_zero() -> None:
    """A zero would read as EA having looked at him and declined to move."""
    players = player_rows({"Season": "2526", "Player": "gone", "fifa_overall": 63.0})
    judged = with_ea_verdict(gaps(players), players)

    assert pd.isna(judged["ea_next"].iloc[0])
    assert pd.isna(judged["ea_move"].iloc[0])


def test_the_claim_is_withheld_when_no_stratum_is_big_enough() -> None:
    """Better no number than a correlation read off four players."""
    players = player_rows(
        *[{"Season": "2324", "Player": f"p{i}", "fifa_overall": 70.0 + i}
          for i in range(4)]
    )
    assert agreement(with_ea_verdict(gaps(players), players)) == {}


def test_the_claim_compares_only_players_ea_rated_identically() -> None:
    """The point of the stratification: EA's mean reversion cannot leak in.

    Every player here starts on the same EA overall, so any correlation left
    is between our rating and EA's later move, with nothing else to explain
    it. The data is built so the two agree.
    """
    rows = []
    for i in range(20):
        rows.append(
            {
                "Season": "2324",
                "Player": f"p{i}",
                "role": "MID",
                "fifa_overall": 70.0,
                "fifa_on_our_scale": 60.0,
                "performance_rating": 50.0 + i,
            }
        )
        # EA moves next edition exactly with how we rated him.
        rows.append({"Season": "2425", "Player": f"p{i}", "fifa_overall": 70.0 + i})
    players = player_rows(*rows)

    claim = agreement(with_ea_verdict(gaps(players), players))

    assert claim["strata"] == 1
    assert claim["players"] == 20
    assert claim["correlation"] == pytest.approx(1.0, abs=1e-6)
    assert claim["top_fifth_move"] > claim["bottom_fifth_move"]


def test_mean_reversion_alone_does_not_produce_a_claim() -> None:
    """EA raising every low-rated player is not evidence about our rating.

    EA's move here is decided entirely by EA's own overall -- up five for the
    group it rated 65, down five for the group it rated 75 -- and *within*
    each group it is uncorrelated with our rating by construction, because
    every performance value is paired with one move above the group mean and
    one below.

    The unstratified correlation is large, and the stratified one is exactly
    zero. That difference is the only reason the published claim is worth
    anything.
    """
    rows = []
    for overall in (65.0, 75.0):
        base = 5.0 if overall < 70 else -5.0
        for i in range(12):
            rows.append(
                {
                    "Season": "2324",
                    "Player": f"p{overall}{i}",
                    "role": "MID",
                    "fifa_overall": overall,
                    "fifa_on_our_scale": overall,
                    "performance_rating": 50.0 + (i // 2),
                }
            )
            rows.append(
                {
                    "Season": "2425",
                    "Player": f"p{overall}{i}",
                    "fifa_overall": overall + base + (1.0 if i % 2 == 0 else -1.0),
                }
            )
    players = player_rows(*rows)
    judged = with_ea_verdict(gaps(players), players)

    # The test has teeth only if the naive figure really is misleading here.
    assert abs(judged["gap"].corr(judged["ea_move"])) > 0.9

    claim = agreement(judged)
    assert claim["strata"] == 2
    assert claim["correlation"] == pytest.approx(0.0, abs=1e-9)


def test_the_board_is_biggest_gap_first_and_respects_its_limit() -> None:
    players = player_rows(
        *[
            {"Season": "2324", "Player": f"p{i}", "fifa_name": f"p{i}",
             "performance_rating": 60.0 + i, "fifa_on_our_scale": 50.0}
            for i in range(8)
        ]
    )
    board, _ = hidden_gems(players, limit=3)

    assert len(board) == 3
    assert list(board["Player"]) == ["p7", "p6", "p5"]
    assert board["gap"].is_monotonic_decreasing


def test_nothing_rated_yields_an_empty_board_rather_than_an_error() -> None:
    board, claim = hidden_gems(pd.DataFrame())
    assert board.empty
    assert claim == {}
