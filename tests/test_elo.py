"""Tests for the Elo rating.

These pin the properties that make a rating trustworthy -- that it is zero-sum,
that it never sees the result of the match it describes, that beating a strong
side is worth more than beating a weak one -- rather than asserting particular
numbers, which would encode today's K and nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.features.elo import START, add_elo, final_ratings


def _match(date, home, away, hg, ag, season="2627", league="E0"):
    return [
        {"League": league, "Season": season, "Date": pd.Timestamp(date),
         "Team": home, "Opponent": away, "Venue": "Home",
         "GoalsFor": hg, "GoalsAgainst": ag},
        {"League": league, "Season": season, "Date": pd.Timestamp(date),
         "Team": away, "Opponent": home, "Venue": "Away",
         "GoalsFor": ag, "GoalsAgainst": hg},
    ]


def _frame(rows):
    return pd.DataFrame([r for match in rows for r in match])


def test_everyone_starts_level():
    rated = add_elo(_frame([_match("2026-08-01", "A", "B", 1, 0)]))
    assert (rated["elo"] == START).all()


def test_a_match_never_sees_its_own_result():
    """The property the whole design rests on.

    Flipping a result must not change the ratings the two sides carried into
    it -- only what they carry into the next one.
    """
    played = _frame([_match("2026-08-01", "A", "B", 3, 0)])
    reversed_ = _frame([_match("2026-08-01", "A", "B", 0, 3)])
    assert add_elo(played)["elo"].tolist() == add_elo(reversed_)["elo"].tolist()


def test_the_update_is_zero_sum():
    rows = _frame([
        _match("2026-08-01", "A", "B", 2, 0),
        _match("2026-08-08", "A", "B", 1, 1),
        _match("2026-08-15", "B", "A", 3, 1),
    ])
    final = final_ratings(rows)
    assert final.sum() == pytest.approx(2 * START, abs=1e-6)


def test_winning_raises_the_rating_and_losing_lowers_it():
    rows = _frame([
        _match("2026-08-01", "A", "B", 2, 0),
        _match("2026-08-08", "A", "B", 2, 0),
    ])
    final = final_ratings(rows)
    assert final["A"] > START > final["B"]


def test_a_bigger_win_moves_the_rating_further():
    narrow = final_ratings(_frame([_match("2026-08-01", "A", "B", 1, 0)]))
    thrashing = final_ratings(_frame([_match("2026-08-01", "A", "B", 5, 0)]))
    assert thrashing["A"] > narrow["A"]


def test_a_draw_still_moves_a_mismatch():
    """A margin of zero must not freeze the rating.

    Without a floor on the multiplier, a side could draw every match for a
    season and finish exactly where it started, however strong the opposition.
    """
    rows = [_match(f"2026-08-{d:02d}", "A", "B", 1, 1) for d in range(1, 9)]
    strong = _frame([_match("2026-07-01", "A", "C", 5, 0)] + rows)
    final = final_ratings(strong)
    assert final["B"] > START, "drawing repeatedly with a stronger side taught nothing"


def test_beating_a_strong_side_is_worth_more_than_beating_a_weak_one():
    build_up = [_match(f"2026-07-{d:02d}", "STRONG", "FODDER", 4, 0) for d in range(1, 9)]

    over_strong = final_ratings(_frame([*build_up, _match("2026-08-01", "X", "STRONG", 1, 0)]))
    over_weak = final_ratings(_frame([*build_up, _match("2026-08-01", "X", "FODDER", 1, 0)]))
    assert over_strong["X"] > over_weak["X"]


def test_ratings_regress_toward_the_mean_between_seasons():
    """Squads turn over in the summer, so last May is worth less by August."""
    first = [_match(f"2026-08-{d:02d}", "A", "B", 3, 0) for d in range(1, 9)]
    end_of_season = final_ratings(_frame(first))

    carried = _frame([*first, _match("2027-08-01", "A", "B", 0, 0, season="2728")])
    rated = add_elo(carried)
    august = rated[(rated["Team"] == "A") & (rated["Season"] == "2728")]["elo"].iloc[0]

    assert START < august < end_of_season["A"]


def test_unplayed_fixtures_get_a_rating_but_do_not_change_one():
    """Placeholder rows must be scored, and must teach the model nothing."""
    played = _frame([_match("2026-08-01", "A", "B", 3, 0)])
    upcoming = _frame([_match("2026-08-08", "A", "B", np.nan, np.nan)])
    rated = add_elo(pd.concat([played, upcoming], ignore_index=True))

    future = rated[rated["Date"] == pd.Timestamp("2026-08-08")]
    assert future["elo"].notna().all()
    assert final_ratings(pd.concat([played, upcoming], ignore_index=True)).sum() == pytest.approx(
        2 * START, abs=1e-6
    )


def test_empty_history_returns_an_elo_column():
    empty = pd.DataFrame(
        columns=["League", "Season", "Date", "Team", "Opponent", "Venue",
                 "GoalsFor", "GoalsAgainst"]
    )
    assert "elo" in add_elo(empty).columns
