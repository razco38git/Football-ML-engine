"""Which players make the squad when they are tied, and why it has to be fixed.

Ratings are integers on the 0-99 scale -- 47 distinct values across 34,778
player-seasons -- so the cut at `best_n` almost always falls inside a tie.
`team_strength` used to rank with `sort_values("rating")`, whose default
quicksort is not stable, so the tie was broken by the input array's layout.

That mattered because the season in progress is fetched live and grows on every
run. Adding four of its rows re-selected squads in 260 of 1,173 **finished**
team-seasons and moved a line rating by up to 6.18 points -- seasons that were
over, about which nothing had changed. It is the propagation path
`EXPERIMENTS.md` recorded as open after the squad-size sweep drifted between
processes.

These pin the property that closes it: a finished team-season's strength must
depend on that team-season's players and nothing else.
"""

from __future__ import annotations

import pandas as pd

from footballml.players.team_strength import team_strength


def _player(name: str, rating: float, minutes: int, role: str = "MID", **over) -> dict:
    return {
        "League": "E0", "Season": "2425", "Team": "Arsenal", "Player": name,
        "role": role, "rating": rating, "minutes": minutes, "rated": True,
        **over,
    }


def _squad(ratings: list[tuple[str, float, int]], **over) -> pd.DataFrame:
    return pd.DataFrame([_player(n, r, m, **over) for n, r, m in ratings])


def test_a_tie_at_the_cut_is_broken_by_minutes() -> None:
    """The rule `select_eleven` already states: who actually played is better
    evidence of who will play than our opinion of who deserves to.

    The two candidates sit in different lines, so *which* one was chosen is
    visible in the output -- asserting on a rating alone could not tell them
    apart, since both are rated 60.
    """
    frame = pd.DataFrame([
        _player("plays-every-week", 60, 3000, role="MID"),
        _player("one-cup-game", 60, 90, role="FWD"),
    ])
    picked = team_strength(frame, method="best_n", best_n=1, min_players=1)

    assert picked["n_players"].iloc[0] == 1
    assert "strength_midfield" in picked.columns, "the 90-minute player was chosen"
    assert "strength_attack" not in picked.columns

    # And the other way round, so this is the minutes rule and not the order
    # the roles happen to be listed in.
    flipped = pd.DataFrame([
        _player("one-cup-game", 60, 90, role="MID"),
        _player("plays-every-week", 60, 3000, role="FWD"),
    ])
    picked = team_strength(flipped, method="best_n", best_n=1, min_players=1)
    assert "strength_attack" in picked.columns
    assert "strength_midfield" not in picked.columns


def test_the_selection_does_not_depend_on_row_order() -> None:
    """The actual bug. Quicksort's output within a tie depends on the array it
    was handed, so the same squad in a different order chose differently."""
    rows = [("a", 60, 3000), ("b", 60, 90), ("c", 60, 1500), ("d", 59, 2000)]
    forward = team_strength(_squad(rows), method="best_n", best_n=2, min_players=1)
    reverse = team_strength(_squad(rows[::-1]), method="best_n", best_n=2, min_players=1)

    pd.testing.assert_frame_equal(
        forward.reset_index(drop=True), reverse.reset_index(drop=True)
    )


def test_identical_rating_and_minutes_still_resolve() -> None:
    """Minutes tie too, so the order has to bottom out somewhere. Without a
    final key this is back to depending on layout."""
    rows = [("yusuf", 60, 1000), ("adam", 60, 1000), ("zoe", 59, 2000)]
    forward = team_strength(_squad(rows), method="best_n", best_n=1, min_players=1)
    reverse = team_strength(_squad(rows[::-1]), method="best_n", best_n=1, min_players=1)

    assert forward["strength_overall"].iloc[0] == reverse["strength_overall"].iloc[0]


def test_a_finished_season_is_untouched_by_a_season_still_in_progress() -> None:
    """The property that was violated, stated directly.

    The in-progress season cannot contribute to a finished team-season by any
    legitimate route -- different `(League, Season, Team)` group entirely -- so
    its presence must leave finished strength bit-identical. It did not: the
    extra rows changed the array, quicksort partitioned differently, and 260 of
    1,173 settled team-seasons were re-selected.
    """
    finished = pd.DataFrame(
        [_player(f"p{i}", 60 - (i % 3), 2000 + i * 7) for i in range(20)]
    )
    in_progress = pd.DataFrame(
        [
            _player(f"new{i}", 60 - (i % 3), 300 + i, Season="2526", Team="Spurs")
            for i in range(40)
        ]
    )

    alone = team_strength(finished, method="best_n", best_n=15)
    together = team_strength(
        pd.concat([in_progress, finished, in_progress], ignore_index=True),
        method="best_n",
        best_n=15,
    )
    together = together[together["Season"] == "2425"]

    pd.testing.assert_frame_equal(
        alone.reset_index(drop=True),
        together.reset_index(drop=True),
        check_like=True,
    )


def test_growing_the_in_progress_season_one_row_at_a_time_changes_nothing() -> None:
    """How it was actually observed: Understat rewrote its 2026 files mid-run
    and the frame grew by four rows in twenty-five minutes."""
    finished = pd.DataFrame(
        [_player(f"p{i}", 60 - (i % 4), 1500 + i * 11) for i in range(25)]
    )
    growing = [
        _player(f"new{i}", 58 + (i % 5), 200 + i, Season="2526", Team="Spurs")
        for i in range(12)
    ]

    baseline = None
    for n in range(len(growing) + 1):
        frame = pd.concat([finished, pd.DataFrame(growing[:n])], ignore_index=True)
        settled = team_strength(frame, method="best_n", best_n=15)
        settled = settled[settled["Season"] == "2425"].reset_index(drop=True)
        if baseline is None:
            baseline = settled
        else:
            pd.testing.assert_frame_equal(baseline, settled, check_like=True)


def test_a_player_column_is_not_required() -> None:
    """`team_strength` documents its input as `rate_players`' output, but the
    sort must not hard-fail on a frame assembled by hand in a test or a sweep.
    """
    frame = _squad([("a", 60, 3000), ("b", 59, 2000)]).drop(columns=["Player"])
    assert not team_strength(frame, method="best_n", best_n=1, min_players=1).empty
