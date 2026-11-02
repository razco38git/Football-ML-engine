"""The pre-season fallback A/B: its arms, its relabelling, and its isolation.

The experiment's conclusion rests on two things being right that no amount of
careful reading of the headline number would catch.

The first is the **relabelling** in `effective_strength`. `next_season_validity`
correlates a season-S row against season S+1's outcomes, because S strength is
what an S+1 match reads. A filled side's value comes from S+1's *own*
pre-season table, so it has to be written back a season to sit in the slot the
join actually reads it in. Off by one and the secondary criterion silently
grades every fallback value against the wrong season -- and still prints three
plausible correlations.

The second is that the arms differ **only** in the fallback. `baseline` must be
production, and `control` must be `baseline` again, or a 0.0001 movement cannot
be told from run-to-run drift.
"""

from __future__ import annotations

import inspect

import pandas as pd

from footballml.features.build import _next_season
from pipelines import preseason_fallback_ab as ab


def _measured(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "League": "E0",
                "Season": season,
                "Team": team,
                "strength_overall": value,
                "strength_attack": value,
                "strength_defence": value,
            }
            for season, team, value in rows
        ]
    )


def _preseason(*rows: tuple[str, str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "League": "E0",
                "Season": season,
                "Team": team,
                "squad_overall": value,
                "squad_attack": value,
                "squad_defence": value,
            }
            for season, team, value in rows
        ]
    )


# --- the relabelling ----------------------------------------------------------


def test_previous_season_inverts_next_season() -> None:
    """The two must agree, including across the century mark."""
    for label in ("1415", "1920", "2021", "2526", "2627"):
        assert _next_season(ab._previous_season(label)) == label


def test_a_filled_side_is_written_back_one_season() -> None:
    """The whole point of `effective_strength`, and the easiest thing to get wrong.

    Luton has no 2024/25 measured row, so a 2025/26 match reads Luton's
    2025/26 pre-season rating. For the correlation to grade that value against
    2025/26's results, it must appear here labelled **2425** -- the slot the
    measured table would have filled.
    """
    effective = ab.effective_strength(
        _measured(("2425", "Chelsea", 70.0)),
        _preseason(("2526", "Luton", 60.0), ("2526", "Chelsea", 65.0)),
    )
    luton = effective[effective["Team"] == "Luton"]
    assert len(luton) == 1
    assert luton["Season"].iloc[0] == "2425"
    assert luton["strength_overall"].iloc[0] == 60.0


def test_a_measured_side_is_not_displaced_by_its_pre_season_row() -> None:
    """Mirrors `_fill_from_preseason`: only holes are filled.

    Chelsea has a 2024/25 measured rating of 70 and a 2025/26 pre-season
    rating of 65. The table must carry 70 exactly once. If the pre-season row
    were appended anyway, the correlation would be computed over a frame with
    two rows per club-season -- which `next_season_validity` would merge
    happily, double-weighting every covered club.
    """
    effective = ab.effective_strength(
        _measured(("2425", "Chelsea", 70.0)),
        _preseason(("2526", "Chelsea", 65.0)),
    )
    chelsea = effective[effective["Team"] == "Chelsea"]
    assert len(chelsea) == 1
    assert chelsea["strength_overall"].iloc[0] == 70.0


def test_the_effective_table_never_duplicates_a_club_season() -> None:
    """The property the correlation depends on, asserted directly."""
    effective = ab.effective_strength(
        _measured(("2425", "Chelsea", 70.0), ("2526", "Chelsea", 72.0)),
        _preseason(
            ("2526", "Chelsea", 65.0),
            ("2627", "Chelsea", 66.0),
            ("2526", "Luton", 60.0),
        ),
    )
    assert not effective.duplicated(["Season", "Team"]).any()


def test_only_the_measured_columns_are_carried_over() -> None:
    """`squad_midfield` and `n_players` have no measured counterpart.

    Concatenating the raw pre-season frame would add columns that exist on 14%
    of rows and NaN elsewhere -- harmless to the correlation, but it would make
    the "effective" table a different shape from the one the join produces,
    which is the one claim this helper is making.
    """
    preseason = _preseason(("2526", "Luton", 60.0)).assign(squad_midfield=61.0, n_players=15)
    effective = ab.effective_strength(_measured(("2425", "Chelsea", 70.0)), preseason)
    assert "squad_midfield" not in effective.columns
    assert "n_players" not in effective.columns


# --- the arms -----------------------------------------------------------------


def test_the_baseline_arm_passes_no_pre_season_table() -> None:
    """`baseline` must be production: the fallback off, exactly as it ships.

    Measured against anything else, every delta below would describe a
    configuration nobody runs.
    """
    preseason = _preseason(("2526", "Luton", 60.0))
    built = ab.arms(preseason)

    assert [name for name, _ in built] == ["baseline", "control", "fallback"]
    by_name = dict(built)
    assert by_name["baseline"] is None
    assert by_name["fallback"] is preseason


def test_the_control_arm_is_the_baseline_again() -> None:
    """The guard against reading run-to-run drift as a result.

    The two must be configured identically, so a non-zero difference between
    them is a statement about the harness rather than about the change.
    """
    built = dict(ab.arms(_preseason(("2526", "Luton", 60.0))))
    assert built["control"] is built["baseline"]


def test_the_margin_is_the_documented_measurement_floor() -> None:
    """0.0005 is EXPERIMENTS.md's figure for this harness's resolution.

    Tightening it would claim a precision the backtest does not have, and is
    the one knob that could turn this run's verdict from inconclusive to ship
    without any model changing.
    """
    assert ab.MARGIN == 0.0005


def test_the_experiment_writes_nothing() -> None:
    """Both arms' features live in memory; an interrupted run must be harmless.

    `write_csv`/`write_json` anywhere in here would mean a half-finished
    experiment could leave the production predictions pointing at an arm.
    """
    source = inspect.getsource(ab)
    for forbidden in ("write_csv", "write_json", "to_csv", "to_json"):
        assert forbidden not in source
