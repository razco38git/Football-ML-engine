"""Name matching must not resolve a tie by row order.

`match_players` widens its name key until something matches: exact, then
first-plus-last, then surname alone. The last of those is loose enough that two
players in the same league-season can share it -- both Onanas played in the
Premier League in 2023/24, one a goalkeeper and one a midfielder.

The original code took whichever row `iterrows` reached first. That is not just
arbitrary, it is *unstably* arbitrary: two builds of identical data gave 197
players a different rating and moved 312 team strengths, because the keeper kept
inheriting the midfielder's tackles. An ambiguous key is not a match.
"""

from __future__ import annotations

import pandas as pd

from footballml.players.fbref import match_players

ON = ["League", "Season", "Team"]
FALLBACK = ["League", "Season"]


def _left(*players: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"League": "E0", "Season": "2324", "Team": team, "Player": name} for name, team in players]
    )


def _right(*rows: tuple[str, str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": "E0", "Season": "2324", "Team": team, "Player": name, "Int": value}
            for name, team, value in rows
        ]
    )


def test_an_initial_shared_by_two_players_matches_neither() -> None:
    """"A. Onana" reaches both men and can choose between neither.

    The source abbreviates the given name, so the only key that reaches either
    row is initial-plus-surname -- and both Onanas answer to it.
    """
    left = _left(("A. Onana", "Manchester United"))
    # Different club spellings, so only the league-season fallback can reach them.
    right = _right(("André Onana", "Manchester Utd", 19), ("Amadou Onana", "Everton", 71))

    out = match_players(left, right, on=ON, fallback_on=FALLBACK)
    assert pd.isna(out["Int"].iloc[0])


def test_row_order_does_not_decide_the_winner() -> None:
    """The same inputs in either order give the same answer."""
    left = _left(("A. Onana", "Manchester United"))
    a = _right(("André Onana", "Manchester Utd", 19), ("Amadou Onana", "Everton", 71))
    b = a.iloc[::-1].reset_index(drop=True)

    first = match_players(left, a, on=ON, fallback_on=FALLBACK)["Int"].iloc[0]
    second = match_players(left, b, on=ON, fallback_on=FALLBACK)["Int"].iloc[0]
    assert pd.isna(first) and pd.isna(second)


def test_a_full_name_wins_over_a_shared_initial() -> None:
    """André is still matched, because his own name is tried before "a onana"."""
    left = _left(("Andre Onana", "Manchester United"))
    right = _right(("André Onana", "Manchester Utd", 19), ("Amadou Onana", "Everton", 71))

    assert match_players(left, right, on=ON, fallback_on=FALLBACK)["Int"].iloc[0] == 19


def test_an_unambiguous_surname_still_matches() -> None:
    """Dropping ambiguous keys must not cost the ordinary case."""
    left = _left(("Bukayo Saka", "Arsenal"))
    right = _right(("Bukayo Saka", "Arsenal FC", 12))

    assert match_players(left, right, on=ON, fallback_on=FALLBACK)["Int"].iloc[0] == 12


def test_one_player_listed_twice_is_not_ambiguous() -> None:
    """A midseason transfer puts the same man in two rows, which is not a tie."""
    left = _left(("Kevin Danso", "Tottenham"))
    right = _right(("Kevin Danso", "Lens", 30), ("Kevin Danso", "Tottenham Hotspur", 8))

    out = match_players(left, right, on=ON, fallback_on=FALLBACK)
    assert out["Int"].iloc[0] in (30, 8)


def test_an_exact_team_match_beats_an_ambiguous_league_key() -> None:
    """The tighter pass runs first, so a shared initial never gets the chance."""
    left = _left(("A. Onana", "Manchester United"))
    right = _right(("A. Onana", "Manchester United", 19), ("Amadou Onana", "Everton", 71))

    assert match_players(left, right, on=ON, fallback_on=FALLBACK)["Int"].iloc[0] == 19
