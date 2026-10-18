"""The rating-travel experiment: its constants, its estimand, and its isolation.

The question is whether a defender's rating describes him or the side he just
left. The design is fragile in two specific ways and these pin both: it is easy
to condition on post-transfer information without noticing, and it is easy to
manufacture the finding by comparing a shrunken residual against an unshrunken
raw predictor.

Nothing here runs the experiment.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import rating_travel_ab as travel


def _stratum(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    club = rng.normal(size=n)
    return pd.DataFrame(
        {
            "role": "CB",
            "Season": "2425",
            "Player": [f"p{i}" for i in range(n)],
            "team_xga_per_match": club,
            "team_ga_per_match": club * 0.8 + rng.normal(scale=0.3, size=n),
            "team_clean_sheet_pct": -club * 0.6 + rng.normal(scale=0.3, size=n),
            travel.PREDICTOR: 70 + 4 * club + rng.normal(scale=2.0, size=n),
        }
    )


# --- the predictor and the outcome scale --------------------------------------


def test_the_predictor_is_the_unblended_rating() -> None:
    """`rating` is half `fifa_overall`, so scoring it against EA's *next*
    edition would be substantially EA predicting itself."""
    assert travel.PREDICTOR == "performance_rating"
    assert travel.PREDICTOR != "rating"


def test_the_incomplete_edition_cannot_be_an_outcome() -> None:
    """2627 carries 504 big-five players against ~1,900 and is still filling."""
    assert "2627" in travel.INCOMPLETE_EDITIONS


def test_both_outcome_scales_are_declared() -> None:
    """The rank form respects the bounded, ordinal nature of an EA rating and is
    run in parallel, not reached for if the z-score answer disappoints."""
    assert set(travel.OUTCOME_SCALES) == {"zscore", "rank"}


def test_an_unknown_scale_raises_rather_than_silently_z_scoring() -> None:
    with pytest.raises(ValueError, match="scale must be one of"):
        travel.standardise_within(_stratum(), travel.PREDICTOR, ["role"], scale="tobit")


# --- the bar ------------------------------------------------------------------


def test_the_equivalence_bounds_are_the_declared_constants() -> None:
    assert travel.EQUIVALENCE == (0.85, 1.18)


def test_the_equivalence_interval_is_symmetric_in_log_ratio() -> None:
    low, high = travel.EQUIVALENCE
    assert low * high == pytest.approx(1.0, abs=0.005)


def test_alternative_bounds_bracket_the_declared_ones() -> None:
    """One looser, one stricter. If the verdict flips across them it depends on
    the threshold rather than the data, which is itself the finding."""
    looser, stricter = travel.ALTERNATIVE_BOUNDS
    assert looser[0] < travel.EQUIVALENCE[0] < stricter[0]
    assert looser[1] > travel.EQUIVALENCE[1] > stricter[1]


def test_the_guards_match_the_quota_experiment() -> None:
    """So the two are directly comparable rather than each having its own bar."""
    from pipelines import positional_quota_ab as quota

    assert travel.DEF_GAIN == quota.DEF_GAIN
    assert travel.OTHER_TOLERANCE == quota.OTHER_TOLERANCE
    assert travel.MARGIN == quota.MARGIN


def test_only_the_three_team_defence_aggregates_are_projected_out() -> None:
    """Projecting more would remove legitimate player signal that merely
    correlates with club quality; the question is about *these* inputs."""
    assert set(travel.PROJECTED_COLUMNS) == {
        "team_xga_per_match",
        "team_ga_per_match",
        "team_clean_sheet_pct",
    }


def test_the_projected_columns_are_the_configs_team_defence_inputs() -> None:
    """Asserted against the config so widening one silently fails here."""
    import yaml

    config = yaml.safe_load(
        Path("config/player_rating.yaml").read_text(encoding="utf-8")
    )
    declared = set(config["positions"]["CB"]["sub_ratings"]["team_defence"])
    assert set(travel.PROJECTED_COLUMNS) == declared


# --- the residualisation, and the way it could manufacture a result -----------


def test_residualising_removes_the_club_signal() -> None:
    frame = _stratum()
    residual, _ = travel.residualise(frame)
    for column in travel.PROJECTED_COLUMNS:
        assert abs(np.corrcoef(residual, frame[column])[0, 1]) < 1e-8


def test_the_residual_has_smaller_variance_than_the_raw_predictor() -> None:
    """The reason the scaling test below exists."""
    frame = _stratum()
    residual, _ = travel.residualise(frame)
    assert residual.var() < frame[travel.PREDICTOR].var()


def test_both_predictors_are_rescaled_to_unit_variance_before_use() -> None:
    """Otherwise a smaller residual coefficient would be an artefact of a
    shrunken predictor rather than evidence about the player.

    This is the single easiest way to manufacture the result this experiment is
    looking for, so it is pinned rather than trusted.
    """
    frame = _stratum()
    residual, _ = travel.residualise(frame)
    frame = frame.assign(residual=residual)

    raw = travel.standardise_within(frame, travel.PREDICTOR, ["role", "Season"])
    res = travel.standardise_within(frame, "residual", ["role", "Season"])
    assert raw.std() == pytest.approx(1.0, abs=1e-6)
    assert res.std() == pytest.approx(1.0, abs=1e-6)


def test_the_projection_reports_r_squared_per_stratum() -> None:
    """Where it is near zero the decomposition has nothing to remove, and a
    residual slope equal to the raw one is uninformative, not reassuring."""
    _, diagnostics = travel.residualise(_stratum())
    assert "r_squared" in diagnostics.columns
    assert diagnostics["r_squared"].between(0.0, 1.0).all()


def test_a_stratum_too_small_to_fit_is_skipped_not_guessed() -> None:
    residual, diagnostics = travel.residualise(_stratum(n=3))
    assert residual.isna().all()
    assert diagnostics.empty


def test_residualise_refuses_a_frame_missing_its_inputs() -> None:
    with pytest.raises(KeyError):
        travel.residualise(_stratum().drop(columns=["team_ga_per_match"]))


# --- the ratio, and its denominator -------------------------------------------


def test_the_ratio_is_undefined_when_the_denominator_flips_sign() -> None:
    """The documented defender EA-move correlation is 0.04-0.06, so a stayer
    slope straddling zero is a realistic outcome -- and a ratio built on one
    would read as a dramatic attenuation while meaning nothing."""
    rng = np.random.default_rng(0)
    straddles = rng.normal(loc=0.0, scale=1.0, size=5000)
    assert not travel.ratio_is_interpretable(straddles)

    clear = rng.normal(loc=5.0, scale=1.0, size=5000)
    assert travel.ratio_is_interpretable(clear)


def test_the_sign_flip_threshold_is_the_declared_constant() -> None:
    assert travel.SIGN_FLIP_MAX == 0.05


def test_a_zero_stayer_slope_gives_nan_rather_than_an_exception() -> None:
    assert np.isnan(travel.travel_ratio(0.0, -0.1))


# --- the three-way verdict ----------------------------------------------------


def test_a_null_stayer_slope_is_uninformative_not_equivalent() -> None:
    """The attenuation of a slope indistinguishable from zero is undefined.

    This fires before every other branch because the alternative -- reading a
    tight-looking ratio off a null denominator -- is the most likely way this
    experiment produces a confident wrong answer.
    """
    assert travel.verdict(
        delta_ci=(-0.01, 0.01),
        ratio=1.0,
        ratio_ci=(0.9, 1.1),
        stayer_ci=(-0.02, 0.05),
    ) == "uninformative"


def test_an_interval_containing_one_is_not_by_itself_equivalence() -> None:
    """A wide interval is ignorance, not evidence of no effect."""
    assert travel.verdict(
        delta_ci=(-0.30, 0.20),
        ratio=0.95,
        ratio_ci=(0.40, 1.60),
        stayer_ci=(0.03, 0.09),
    ) == "inconclusive"


def test_equivalence_requires_the_interval_inside_the_bounds() -> None:
    assert travel.verdict(
        delta_ci=(-0.01, 0.01),
        ratio=0.99,
        ratio_ci=(0.90, 1.10),
        stayer_ci=(0.03, 0.09),
    ) == "equivalent"


def test_contamination_needs_both_a_clear_delta_and_a_ratio_past_the_bound() -> None:
    assert travel.verdict(
        delta_ci=(-0.08, -0.02),
        ratio=0.55,
        ratio_ci=(0.30, 0.80),
        stayer_ci=(0.03, 0.09),
    ) == "contamination"

    # Ratio past the bound but delta straddling zero is not a finding.
    assert travel.verdict(
        delta_ci=(-0.08, 0.02),
        ratio=0.55,
        ratio_ci=(0.20, 1.30),
        stayer_ci=(0.03, 0.09),
    ) == "inconclusive"


# --- isolation from production ------------------------------------------------


def test_the_experiment_cannot_write_anything() -> None:
    source = "\n".join(
        line
        for line in inspect.getsource(travel).split("\n")
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
    tree = ast.parse(inspect.getsource(travel))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("build_players" in name for name in imported), imported


def test_nothing_here_touches_the_rating_config() -> None:
    """Production weights are not changed by this experiment under any outcome.

    A candidate reweighting is *measured* and reported against the guards;
    adopting one is a separate decision taken by a person.
    """
    tree = ast.parse(inspect.getsource(travel))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "yaml" not in imported, "the experiment has no business parsing the config"

    # The config is named in the docstrings, which is the point -- the constants
    # here are justified against it. What must not exist is a path that rewrites
    # it, so the check is on serialisation rather than on the filename.
    source = inspect.getsource(travel)
    for forbidden in ("yaml.dump", "yaml.safe_dump", "safe_dump", "dump("):
        assert forbidden not in source, f"the experiment must not call {forbidden}"


# --- the lookup must outlive the eligibility filter ---------------------------


def test_a_season_ineligible_as_predictor_still_serves_as_an_outcome() -> None:
    """The bug that destroyed a fold.

    2526 is dropped as a *predictor* season because its own t+1 edition is
    still filling. When the tier-1 lookup was built from that same filtered
    frame, the 2425 -> 2526 pairs lost tier 1 entirely: that fold recovered
    36.2% against 94-96% everywhere else, and what survived was whatever the
    raw name match happened to find -- a heavily selected sixth of a season.

    A season can be ineligible to be predicted *from* while remaining perfectly
    good to be predicted *to*.
    """
    rated = pd.DataFrame(
        {
            "Player": ["A", "A"],
            "Season": ["2425", "2526"],
            "role": ["CB", "CB"],
            "fifa_name": ["A", "A"],
            "fifa_age": [25.0, 26.0],
            "fifa_overall": [80.0, 82.0],
            "fifa_club": ["Club X", "Club Y"],
        }
    )
    eligible = rated[rated["Season"] == "2425"]

    # Lookup restricted to the eligible rows -- the bug.
    broken = travel.recover_outcomes(eligible.copy())
    assert not broken["recovered"].iloc[0], "fixture no longer reproduces the bug"

    # Lookup built from every rated season -- the fix.
    fixed = travel.recover_outcomes(eligible.copy(), lookup=rated)
    assert fixed["recovered"].iloc[0]
    assert fixed["next_overall"].iloc[0] == 82.0
    assert bool(fixed["mover"].iloc[0]) is True


def test_build_sample_passes_the_unfiltered_lookup() -> None:
    source = inspect.getsource(travel.build_sample)
    assert "lookup = rated" in source
    assert "recover_outcomes(eligible, lookup)" in source
