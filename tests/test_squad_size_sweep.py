"""The squad-size sweep: its arms, its validity metric, and its isolation.

Production builds a team rating from each club's top 15 players, a number that
was never measured -- the library default says 14, production passes 15, and no
table anywhere explains either. These pin the sweep that settles it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from pipelines import squad_size_sweep as sweep

# --- the arms ----------------------------------------------------------------


def test_the_baseline_is_the_setting_production_ships() -> None:
    """Otherwise every delta is measured against something nobody runs."""
    name, size = sweep.variants()[0]
    assert name == "baseline"
    assert size == sweep.SHIPPED == 15


def test_there_is_a_determinism_control_at_the_shipped_setting() -> None:
    """A second arm identical to baseline, which must score identically.

    This pipeline is exactly deterministic within one data vintage and
    demonstrably not between vintages -- rebuilding on a later day moved the
    baseline RPS 0.36508 to 0.36498 on a byte-identical config. Without an arm
    that *cannot* differ, a sweep has no way to tell a real difference from
    that drift.
    """
    arms = sweep.variants()
    assert arms[1] == ("control-15", sweep.SHIPPED)
    assert [size for _, size in arms].count(sweep.SHIPPED) == 2


def test_the_sizes_bracket_the_shipped_number() -> None:
    """Smaller and larger, so a monotone trend is visible if there is one.

    A single better point next door to 15 is noise; the same direction either
    side of it is a finding.
    """
    sizes = [size for _, size in sweep.variants()]
    assert min(sizes) < sweep.SHIPPED < max(sizes)
    assert sorted(set(sizes)) == [13, 14, 15, 16, 17]


def test_every_arm_is_tried_exactly_once_apart_from_the_control() -> None:
    names = [name for name, _ in sweep.variants()]
    assert len(names) == len(set(names)), "a repeated arm name would overwrite its own result"


# --- next-season validity -----------------------------------------------------


def _strength(rows: list[tuple[str, str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "League": lg,
                "Season": s,
                "Team": t,
                "strength_attack": atk,
                "strength_defence": dfc,
                "strength_overall": ovr,
            }
            for lg, s, t, atk, dfc, ovr in rows
        ]
    )


def _outcomes(rows: list[tuple[str, str, str, float, float, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"League": lg, "Season": s, "Team": t, "gf": gf, "ga": ga, "pts": pts, "played": 38}
            for lg, s, t, gf, ga, pts in rows
        ]
    )


def test_the_correlations_point_the_way_round_football_does() -> None:
    """Attack and overall positive, defence negative.

    The signs differ by design: a better attack scores *more*, a better defence
    concedes *fewer*. A sweep that reported all three the same way round would
    rank the leakiest side top on defence and nobody would notice until it
    shipped.
    """
    strength, outcomes = [], []
    for i in range(60):
        # Strong clubs score more, concede less, and take more points.
        quality = i / 60
        strength.append(("E0", "2425", f"T{i}", 60 + 30 * quality, 60 + 30 * quality,
                         60 + 30 * quality))
        outcomes.append(("E0", "2526", f"T{i}", 1.0 + 2.0 * quality, 2.0 - 1.5 * quality,
                         int(30 + 50 * quality)))

    got = sweep.next_season_validity(_strength(strength), _outcomes(outcomes))
    assert got["atk_vs_gf"] > 0.9
    assert got["ovr_vs_pts"] > 0.9
    assert got["def_vs_ga"] < -0.9


def test_validity_looks_at_the_following_season_not_this_one() -> None:
    """Same-season would flatter any rating built from that season's results.

    It is also the only direction the model reads strength: `add_team_strength`
    joins season S onto season S+1.
    """
    strength, outcomes = [], []
    for i in range(60):
        quality = i / 60
        strength.append(("E0", "2425", f"T{i}", 60 + 30 * quality, 60 + 30 * quality,
                         60 + 30 * quality))
        # Next season agrees with the rating.
        outcomes.append(("E0", "2526", f"T{i}", 1.0 + 2.0 * quality, 2.0 - 1.5 * quality,
                         int(30 + 50 * quality)))
        # The *same* season is deliberately inverted: a function reading these
        # would come back with the opposite sign.
        outcomes.append(("E0", "2425", f"T{i}", 3.0 - 2.0 * quality, 0.5 + 1.5 * quality,
                         int(80 - 50 * quality)))

    got = sweep.next_season_validity(_strength(strength), _outcomes(outcomes))
    assert got["atk_vs_gf"] > 0.9, "it read the same season, not the next one"
    assert got["def_vs_ga"] < -0.9


def test_too_few_club_seasons_returns_nan_rather_than_a_number() -> None:
    """A correlation off a handful of clubs is worse than no correlation."""
    strength = _strength([("E0", "2425", "A", 80.0, 80.0, 80.0)])
    outcomes = _outcomes([("E0", "2526", "A", 2.0, 1.0, 70)])
    got = sweep.next_season_validity(strength, outcomes)
    assert all(np.isnan(v) for v in got.values())


def test_a_missing_strength_column_is_reported_as_absent() -> None:
    """Rather than raising and losing the arms that did compute."""
    strength = _strength([("E0", "2425", f"T{i}", 70.0, 70.0, 70.0) for i in range(60)])
    got = sweep.next_season_validity(strength.drop(columns=["strength_attack"]),
                                     _outcomes([("E0", "2526", f"T{i}", 1.5, 1.5, 50)
                                                for i in range(60)]))
    assert np.isnan(got["atk_vs_gf"])


def test_points_are_derived_from_goals_not_from_the_result_column() -> None:
    """So the figure does not depend on how `Result` happens to be encoded."""
    history = pd.DataFrame(
        [{"League": "E0", "Season": "2425", "Team": "A", "GoalsFor": g, "GoalsAgainst": a}
         for g, a in [(2, 1)] * 10 + [(1, 1)] * 10 + [(0, 3)] * 10]
    )
    out = sweep._season_outcomes(history)
    assert len(out) == 1
    assert int(out["pts"].iloc[0]) == 10 * 3 + 10 * 1
    assert "Result" not in history.columns, "the fixture proves the column is not needed"


def test_a_season_too_short_to_compare_is_excluded() -> None:
    """A season in progress has a fine goals-per-match and a meaningless points
    total, and the overall correlation is against points."""
    history = pd.DataFrame(
        [{"League": "E0", "Season": "2627", "Team": "A", "GoalsFor": 2, "GoalsAgainst": 1}] * 5
    )
    assert sweep._season_outcomes(history).empty


# --- isolation from production ------------------------------------------------


def test_the_sweep_cannot_write_anything() -> None:
    """A killed run must not be able to leave a production file behind.

    This is not hypothetical. The first version of `defending_sweep` swapped
    `team_strength.csv` per arm and restored it in a `finally`; a `finally` does
    not survive the process being killed, and a timeout left the shipped file
    holding a variant's numbers -- 1,160 of 1,170 rows wrong, with the API
    serving them. The guarantee has to be that there is no write to interrupt.
    """
    source = "\n".join(
        line
        for line in inspect.getsource(sweep).split("\n")
        if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "to_csv", "to_pickle", "to_parquet", "to_feather", "to_json", "to_hdf",
        "shutil", "os.replace", "os.remove", "os.rename", "unlink(", "write_text",
        "write_bytes", "mkdir(", "savetxt", "np.save",
    ):
        assert forbidden not in source, f"the sweep must not call {forbidden}"
    for line in source.splitlines():
        if "open(" in line and "fileno" not in line:
            assert not any(m in line for m in ('"w"', "'w'", '"a"', "'a'", '"x"', "'x'")), line


def test_the_sweep_does_not_import_the_thing_that_writes_production() -> None:
    """`build_players` owns both artifacts; importing it would route around
    every string check above.

    Checked against the import graph rather than the text, because the module
    docstring names `build_players.py` deliberately -- that is where the
    unmeasured 15 is passed from, and saying so is the point.
    """
    import ast

    tree = ast.parse(inspect.getsource(sweep))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported += [f"{node.module}.{alias.name}" for alias in node.names]
    assert not any("build_players" in name for name in imported), imported
    assert sweep.__dict__.get("build_players") is None


def test_the_sweep_holds_strength_in_memory_rather_than_on_disk() -> None:
    """`build_match_features` takes the frame, which is why no write is needed."""
    source = inspect.getsource(sweep)
    assert "build_match_features(tmh, strength=strength" in source
    assert "team_strength(rated, method=\"best_n\", best_n=size)" in source


def test_players_are_rated_once_for_every_arm() -> None:
    """`best_n` selects from the ratings and changes nothing upstream of them.

    Re-rating per arm would burn minutes reproducing an identical frame. This
    pins the saving so a later edit does not quietly move the call inside the
    loop.
    """
    source = inspect.getsource(sweep.main)
    before, _, after = source.partition("for name, size in variants():")
    assert "rate_players(players)" in before
    assert "rate_players" not in after


def test_the_module_lives_outside_the_data_directory() -> None:
    """A sanity check on the path, so the no-write guarantee is about the right
    file."""
    assert Path(sweep.__file__).parent.name == "pipelines"
