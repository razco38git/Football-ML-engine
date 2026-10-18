"""The conditional-rho experiment: fold discipline, its margins, and isolation.

Rho is fitted in `MatchPredictor.fit` and applied to every league. Conditioning
it is cheap to try and easy to get wrong in one specific way -- fitting a
conditioned rho on the scorelines it is about to be scored on would inflate
every arm, and inflate the finely-conditioned ones most, producing a clean
monotone story out of a leak.

Nothing here runs the experiment.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from pipelines import conditional_rho_ab as rho

# --- fold discipline ----------------------------------------------------------


def test_fitting_rho_on_the_scored_fold_raises() -> None:
    """The leak this experiment is most likely to commit."""
    frame = pd.DataFrame({"FTHG": [1, 2, 0, 1]}, index=[0, 1, 2, 3])
    with pytest.raises(rho.FoldLeakError, match="strictly earlier"):
        rho.fit_rho_within_fold(frame, pd.Index([2, 3]), lambda f: -0.05)


def test_a_disjoint_training_frame_is_allowed_through() -> None:
    frame = pd.DataFrame({"FTHG": [1, 2]}, index=[0, 1])
    assert rho.fit_rho_within_fold(frame, pd.Index([2, 3]), lambda f: -0.05) == -0.05


def test_the_leak_error_names_the_overlapping_rows() -> None:
    """A guard that says only "leak" gets disabled; one that points at rows gets
    fixed."""
    frame = pd.DataFrame({"FTHG": [1, 2, 3]}, index=[7, 8, 9])
    with pytest.raises(rho.FoldLeakError) as exc:
        rho.fit_rho_within_fold(frame, pd.Index([8, 9]), lambda f: 0.0)
    assert "2 row(s)" in str(exc.value)
    assert "8" in str(exc.value)


# --- the arms -----------------------------------------------------------------


def test_production_is_the_first_arm() -> None:
    """An arm that is anything other than the shipped global rho would measure a
    configuration nobody runs."""
    assert rho.ARMS[0] == "production"


def test_every_declared_arm_is_present() -> None:
    assert set(rho.ARMS) == {"production", "per-league", "shrunk", "per-lambda"}


PER_LEAGUE = pd.Series({"D1": -0.130, "E0": -0.043, "F1": -0.045,
                        "I1": -0.052, "SP1": -0.033})
COUNTS = pd.Series({"D1": 3402, "E0": 4239, "F1": 3911,
                    "I1": 4230, "SP1": 4256})


def test_shrinkage_pulls_a_noisy_league_toward_the_pool() -> None:
    variances = pd.Series({k: 3.5e-4 for k in PER_LEAGUE.index})
    shrunk = rho.shrink(PER_LEAGUE, -0.0567, COUNTS, variances)
    assert abs(shrunk["D1"] - -0.0567) < abs(PER_LEAGUE["D1"] - -0.0567)
    assert ((shrunk - -0.0567).abs() <= (PER_LEAGUE - -0.0567).abs()).all()


def test_shrinkage_is_partial_not_total() -> None:
    """The bug this arm shipped with on its first run.

    The original factor was `1 - (k - 2) / (k * spread)` with no sampling
    variance in it, which implicitly sets sigma^2 = 1. Rho deviations are ~0.07
    so `spread` is ~0.001, the factor hit its floor of zero on every fold, and
    the `shrunk` arm was production in disguise -- reporting a delta of exactly
    0.000000 against itself across all 18,212 fixtures and all five leagues.

    A collapse to the pooled value must now require the data to say so, not the
    units.
    """
    variances = pd.Series({k: 3.5e-4 for k in PER_LEAGUE.index})
    shrunk = rho.shrink(PER_LEAGUE, -0.0567, COUNTS, variances)
    assert not np.allclose(shrunk.to_numpy(), -0.0567), "collapsed to pooled again"
    assert not np.allclose(shrunk.to_numpy(), PER_LEAGUE.to_numpy())


def test_a_huge_sampling_variance_does_collapse_to_pooled() -> None:
    """When the per-league estimates really are indistinguishable from noise."""
    variances = pd.Series({k: 10.0 for k in PER_LEAGUE.index})
    assert rho.shrink(PER_LEAGUE, -0.0567, COUNTS, variances).eq(-0.0567).all()


def test_omitting_the_variances_applies_no_shrinkage() -> None:
    """Rather than shrinking by an assumed one, which is what went wrong."""
    assert rho.shrink(PER_LEAGUE, -0.0567, COUNTS).equals(PER_LEAGUE.astype("float64"))


def test_shrinkage_of_an_already_pooled_set_is_a_no_op() -> None:
    per_league = pd.Series({"A": -0.05, "B": -0.05, "C": -0.05})
    counts = pd.Series({"A": 100, "B": 100, "C": 100})
    variances = pd.Series({"A": 1e-4, "B": 1e-4, "C": 1e-4})
    assert rho.shrink(per_league, -0.05, counts, variances).eq(-0.05).all()


# --- the margins --------------------------------------------------------------


def test_the_exact_score_margin_is_the_declared_constant() -> None:
    """One sampling sd of a 12.8% hit rate over ~20,000 fixtures is 0.236pp, so
    the margin sits at the resolution of the measurement."""
    assert rho.EXACT_SCORE_MARGIN_PP == 0.25


def test_exact_score_is_judged_on_the_interval_not_the_point_estimate() -> None:
    """"Does not fall" would reject an arm over one match in twenty thousand."""
    assert rho.exact_score_passes(0.10)      # a decline inside the margin
    assert rho.exact_score_passes(-0.30)     # an improvement
    assert not rho.exact_score_passes(0.40)  # a decline past it


def test_a_tiny_decline_does_not_fail_the_arm() -> None:
    """The specific behaviour the margin exists to produce."""
    assert rho.exact_score_passes(0.01)


def test_the_rps_margin_is_the_projects_measured_floor() -> None:
    from pipelines import positional_quota_ab as quota

    assert rho.MARGIN == quota.MARGIN == 0.0005


def test_rps_is_not_assumed_insensitive_to_rho() -> None:
    """Rho moves mass between cells sitting on both sides of the draw/win
    boundary, so an RPS effect is possible and must be measured."""
    source = inspect.getsource(rho)
    assert "MARGIN" in source
    assert "insensitive" in source.lower()


def test_four_of_five_leagues_is_the_declared_requirement() -> None:
    assert rho.LEAGUES_REQUIRED == 4


# --- isolation ----------------------------------------------------------------


def test_the_experiment_cannot_write_anything() -> None:
    source = "\n".join(
        line
        for line in inspect.getsource(rho).split("\n")
        if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "to_csv", "to_pickle", "to_parquet", "to_json", "to_hdf", "shutil",
        "os.replace", "os.remove", "os.rename", "unlink(", "write_text",
        "write_bytes", "mkdir(", "savetxt", "np.save",
    ):
        assert forbidden not in source, f"the experiment must not call {forbidden}"
    for line in source.splitlines():
        if "open(" in line and "fileno" not in line:
            assert not any(
                mode in line for mode in ('"w"', "'w'", '"a"', "'a'", '"x"', "'x'")
            ), line


def test_the_experiment_does_not_import_the_thing_that_writes_production() -> None:
    tree = ast.parse(inspect.getsource(rho))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("build_players" in name for name in imported), imported


def test_the_two_experiments_are_isolated_from_each_other() -> None:
    """Direction 3 must not pick up Direction 2's state, or vice versa."""
    tree = ast.parse(inspect.getsource(rho))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("rating_travel" in name for name in imported), imported

    from pipelines import rating_travel_ab as travel

    other = ast.parse(inspect.getsource(travel))
    back: list[str] = []
    for node in ast.walk(other):
        if isinstance(node, ast.Import):
            back += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            back.append(node.module or "")
    assert not any("conditional_rho" in name for name in back), back
