"""Positional quotas: the selection rule, the pre-declared bar, and isolation.

Production takes a club's top 15 by rating with no per-line floor, so nothing
stops a squad being attack-heavy and `strength_defence` being computed from
whoever is left. These pin the experiment that asks whether that matters.
"""

from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from pipelines import positional_quota_ab as quota

KEYS = ["League", "Season", "Team"]


def _pool(**lines: int) -> pd.DataFrame:
    """One club, with `lines` players per line, rated best-first in that order."""
    rows = []
    rating = 99.0
    for line, count in lines.items():
        for _ in range(count):
            rows.append(
                {"League": "E0", "Season": "2425", "Team": "A", "line": line, "rating": rating}
            )
            rating -= 1.0
    return pd.DataFrame(rows)


# --- the selection rule -------------------------------------------------------


def test_no_quota_is_exactly_the_top_n_by_rating() -> None:
    """The baseline arm must reproduce `method="best_n"`, or nothing is readable."""
    pool = _pool(attack=8, defence=8, midfield=4, goalkeeper=2)
    got = quota.select_squad(pool, None, total=15)
    assert len(got) == 15
    assert got["rating"].min() == pool["rating"].nlargest(15).min()


def test_a_quota_is_a_floor_not_a_cap() -> None:
    """A club whose best players are all defenders still fields them.

    The point is to stop a line being starved, not to impose a shape on a club
    that genuinely has depth somewhere.
    """
    pool = _pool(defence=12, attack=4, midfield=3, goalkeeper=1)
    got = quota.select_squad(pool, {"goalkeeper": 1, "defence": 4, "midfield": 3, "attack": 3}, 15)
    assert (got["line"] == "defence").sum() > 4


def test_a_floor_rescues_a_line_the_top_fifteen_would_starve() -> None:
    """The pathology this experiment exists for.

    An attack-heavy club contributes twelve forwards, three defenders and *no
    goalkeeper* under the shipped rule -- so `strength_goalkeeper` is absent and
    `strength_defence` is computed from three players.
    """
    pool = _pool(attack=12, defence=6, midfield=4, goalkeeper=2)

    shipped = quota.select_squad(pool, None, total=15)
    assert (shipped["line"] == "goalkeeper").sum() == 0, "fixture no longer shows the pathology"
    assert (shipped["line"] == "defence").sum() == 3

    floored = quota.select_squad(pool, quota.QUOTAS["formation"], total=15)
    assert (floored["line"] == "goalkeeper").sum() >= 1
    assert (floored["line"] == "defence").sum() >= 4


def test_a_line_with_fewer_players_than_its_floor_contributes_what_it_has() -> None:
    """Rather than raising, or inventing players to meet the number."""
    pool = _pool(attack=10, defence=2, midfield=3, goalkeeper=0)
    got = quota.select_squad(pool, quota.QUOTAS["formation"], total=15)
    assert (got["line"] == "defence").sum() == 2
    assert (got["line"] == "goalkeeper").sum() == 0


def test_the_squad_never_exceeds_the_total() -> None:
    for name, q in quota.QUOTAS.items():
        got = quota.select_squad(_pool(attack=9, defence=9, midfield=6, goalkeeper=3), q, 15)
        assert len(got) <= 15, name


def test_clubs_are_selected_independently() -> None:
    """A quota applies within a club-season, never across the league."""
    a = _pool(attack=10, defence=5, midfield=3, goalkeeper=2)
    b = a.assign(Team="B", rating=a["rating"] - 50)
    got = quota.select_squad(pd.concat([a, b]), quota.QUOTAS["formation"], 15)
    sizes = got.groupby(KEYS, observed=True).size()
    assert (sizes == sizes.iloc[0]).all()
    assert set(got["Team"]) == {"A", "B"}


def test_selection_leaves_the_rating_column_untouched() -> None:
    """Only *which* players are chosen may differ between arms."""
    pool = _pool(attack=8, defence=8, midfield=4, goalkeeper=2)
    got = quota.select_squad(pool, quota.QUOTAS["full"], 15)
    pd.testing.assert_series_equal(
        got["rating"], pool.loc[got.index, "rating"], check_names=False
    )


# --- the arms and the bar -----------------------------------------------------


def test_production_is_the_first_arm_and_has_no_quota() -> None:
    first = next(iter(quota.QUOTAS))
    assert first == "production"
    assert quota.QUOTAS["production"] is None


def test_the_quotas_are_declared_as_constants() -> None:
    """Readable in the source before any number exists."""
    for name in ("formation", "balanced", "full"):
        assert isinstance(quota.QUOTAS[name], dict)
    assert quota.QUOTAS["formation"] == {
        "goalkeeper": 1, "defence": 4, "midfield": 3, "attack": 3
    }


def test_no_quota_reserves_more_places_than_the_squad_has() -> None:
    """A floor summing past 15 would leave no room for the best-remaining rule
    and silently turn every arm into a fixed formation."""
    for name, q in quota.QUOTAS.items():
        if q is None:
            continue
        assert sum(q.values()) <= quota.DEFAULT_BEST_N, name


def test_the_thresholds_are_tied_to_the_squad_sweep() -> None:
    """0.010 is the whole 13-to-17 range of def>GA; 0.005 about half the others'.

    Picking a bar after seeing the result is the failure mode these exist to
    prevent, so they are asserted rather than left in prose.
    """
    assert quota.DEF_GAIN == 0.010
    assert quota.OTHER_TOLERANCE == 0.005
    assert quota.MARGIN == 0.0005
    assert quota.MOVABLE_SEASONS == 11


def test_a_season_that_cannot_move_is_not_counted() -> None:
    """The bug from the extended-Elo run, pinned here before it can recur."""
    import numpy as np

    keys = pd.Series(["dead"] * 4 + ["live"] * 4)
    base = np.full(8, 0.2)
    arm = np.array([0.2] * 4 + [0.3] * 4)
    assert quota._seasons_ok(keys, base, arm, minimum=1) == 0


# --- isolation ----------------------------------------------------------------


def test_the_experiment_cannot_write_anything() -> None:
    source = "\n".join(
        line for line in inspect.getsource(quota).split("\n")
        if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "to_csv", "to_pickle", "to_parquet", "to_json", "to_hdf", "shutil",
        "os.replace", "os.remove", "os.rename", "unlink(", "write_text",
        "write_bytes", "mkdir(", "savetxt", "np.save",
    ):
        assert forbidden not in source, f"the experiment must not call {forbidden}"


def test_the_experiment_does_not_import_the_thing_that_writes_production() -> None:
    tree = ast.parse(inspect.getsource(quota))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not any("build_players" in n for n in names), names


def test_selection_happens_outside_team_strength() -> None:
    """Every arm must run production's own aggregation.

    Selecting outside and passing `method="squad"` means the only thing that
    varies between arms is which fifteen players go in -- the weighting, the
    line means and the `MIN_RATED_PLAYERS` gate are production's.
    """
    source = inspect.getsource(quota.main)
    assert 'team_strength(picked, method="squad")' in source


def test_the_harness_asserts_it_reproduces_production() -> None:
    """If the baseline arm does not equal `best_n`, no arm is readable."""
    source = inspect.getsource(quota.main)
    assert "harness does not reproduce production" in source
    assert 'method="best_n"' in source


@pytest.mark.parametrize("name", ["production", "formation", "balanced", "full"])
def test_each_declared_arm_exists(name: str) -> None:
    assert name in quota.QUOTAS
