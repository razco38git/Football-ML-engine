"""Points taken against points the squad implied: the fit, and what it refuses.

The board's claim is comparative, so every test here is about the *comparison*
being made inside the right pool rather than about any particular number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.achievement import (
    MIN_CLUBS,
    achievement,
    club_histories,
    expected_points,
)


def league_season(
    league: str,
    season: str,
    clubs: int = MIN_CLUBS,
    slope: float = 2.0,
    intercept: float = -100.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A division whose points sit exactly on a line through its ratings."""
    ratings = [60.0 + i for i in range(clubs)]
    strength = pd.DataFrame(
        {
            "League": league,
            "Season": season,
            "Team": [f"{league}{season} club {i}" for i in range(clubs)],
            "strength_overall": ratings,
        }
    )
    outcomes = pd.DataFrame(
        {
            "League": league,
            "Season": season,
            "Team": strength["Team"],
            "pts": [slope * r + intercept for r in ratings],
        }
    )
    return strength, outcomes


def test_a_division_on_its_own_line_has_no_residuals() -> None:
    strength, outcomes = league_season("E0", "2324")

    fitted = expected_points(strength, outcomes)

    assert len(fitted) == MIN_CLUBS
    assert fitted["over"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_the_residual_is_points_above_the_line() -> None:
    strength, outcomes = league_season("E0", "2324")
    outcomes.loc[outcomes.index[0], "pts"] += 30.0

    fitted = expected_points(strength, outcomes).set_index("Team")
    over = fitted.loc["E02324 club 0", "over"]

    # A club pulls the line it is being measured against, so it keeps part of
    # its 30 rather than all of it -- and this one sits at the end of the
    # rating range, where the leverage is greatest. Over half and under all is
    # the claim; the exact share is a property of least squares, not of us.
    assert 15.0 < over < 30.0
    # It is comfortably the largest residual in its division, and the division
    # still balances: least squares leaves the residuals summing to zero, so
    # one club's gain is spread as everybody else's small loss.
    assert over == fitted["over"].max()
    assert over > 2 * fitted.drop(index="E02324 club 0")["over"].abs().max()
    assert fitted["over"].sum() == pytest.approx(0.0, abs=1e-9)


def test_each_league_season_is_fitted_against_itself() -> None:
    """Points are league-relative even though ratings are not.

    Two divisions whose clubs rate identically but whose points differ by a
    constant must both come back flat. One line across the pair would charge
    every club in the lower-scoring division for the gap.
    """
    a_strength, a_outcomes = league_season("E0", "2324", intercept=-100.0)
    b_strength, b_outcomes = league_season("F1", "2324", intercept=-130.0)

    fitted = expected_points(
        pd.concat([a_strength, b_strength], ignore_index=True),
        pd.concat([a_outcomes, b_outcomes], ignore_index=True),
    )

    assert fitted["over"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_a_season_is_fitted_against_its_own_season() -> None:
    """The relationship moves between years, so each one gets its own line."""
    a_strength, a_outcomes = league_season("E0", "2324", slope=2.0)
    b_strength, b_outcomes = league_season("E0", "2425", slope=3.0)

    fitted = expected_points(
        pd.concat([a_strength, b_strength], ignore_index=True),
        pd.concat([a_outcomes, b_outcomes], ignore_index=True),
    )

    assert fitted["over"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_a_part_built_division_is_not_fitted() -> None:
    strength, outcomes = league_season("E0", "2627", clubs=MIN_CLUBS - 1)

    assert expected_points(strength, outcomes).empty


def test_a_division_with_no_spread_in_rating_is_skipped() -> None:
    """There is no line through a column of identical ratings."""
    strength, outcomes = league_season("E0", "2324")
    strength["strength_overall"] = 70.0

    assert expected_points(strength, outcomes).empty


def test_a_club_carries_its_whole_run() -> None:
    """A single spike is an event, not a property, and the page needs both."""
    frames = [league_season("E0", season) for season in ("2223", "2324", "2425")]
    strength = pd.concat([f[0] for f in frames], ignore_index=True)
    outcomes = pd.concat([f[1] for f in frames], ignore_index=True)
    # Same club name across all three seasons.
    strength["Team"] = strength["Team"].str.replace(r"^E0\d{4} ", "", regex=True)
    outcomes["Team"] = outcomes["Team"].str.replace(r"^E0\d{4} ", "", regex=True)

    histories = club_histories(expected_points(strength, outcomes))

    assert [season for season, _ in histories[("E0", "club 0")]] == ["2223", "2324", "2425"]


def test_the_board_reports_the_scale_its_numbers_sit_on() -> None:
    """ "+30 points" means nothing without the spread it is being read against."""
    strength, outcomes = league_season("E0", "2324", clubs=20)
    outcomes.loc[outcomes.index[0], "pts"] += 30.0

    over, under, summary = achievement(strength, outcomes, limit=3)

    assert summary["club_seasons"] == 20
    assert summary["residual_sd"] > 0
    assert len(over) == 3
    assert over["over"].iloc[0] > 0
    assert under["over"].iloc[0] < 0


def test_nothing_to_fit_is_an_empty_board_rather_than_an_error() -> None:
    over, under, summary = achievement(pd.DataFrame(), pd.DataFrame())

    assert over.empty and under.empty and summary == {}


def test_a_club_season_with_no_rating_is_dropped_not_zeroed() -> None:
    strength, outcomes = league_season("E0", "2324", clubs=MIN_CLUBS + 1)
    strength.loc[strength.index[0], "strength_overall"] = np.nan

    fitted = expected_points(strength, outcomes)

    assert len(fitted) == MIN_CLUBS
    assert "E02324 club 0" not in set(fitted["Team"])


def test_an_unfittable_frame_yields_no_sparklines_rather_than_raising() -> None:
    """`expected_points` returns a bare frame, which has no columns to group."""
    assert club_histories(expected_points(pd.DataFrame(), pd.DataFrame())) == {}
