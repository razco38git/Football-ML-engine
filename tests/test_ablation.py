"""Tests for the ablation ladder.

The experiment's conclusions are only worth as much as two guarantees: that the
families cover the production model exactly, and that every rung predicts the
same fixtures so the paired bootstrap is valid. Both are asserted here rather
than inspected.

The leak-safety requirements in the brief -- rolling features cannot see the
future, Elo is pre-match, strength uses only prior seasons -- are already
enforced for every feature the ladder uses by `tests/test_leakage.py`, which
rebuilds the whole table from truncated data. The ladder selects columns from
that same table and adds no features of its own, so those properties are
inherited. What is tested here is what the ladder itself could get wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.experiments.ablation import (
    FAMILY_PATTERNS,
    ID_COLUMNS,
    LADDER,
    LEAVE_ONE_OUT,
    classify,
    family_of,
    rung_columns,
)
from footballml.models.evaluate import (
    paired_bootstrap,
    ranked_probability_score,
    rps_per_match,
)
from footballml.models.match_model import feature_columns

# A sample of real production column names, one per stem family, including the
# awkward ones: stems that themselves contain "diff", venue splits, and all
# three column shapes (home_, away_, _diff).
REAL_COLUMNS = {
    "league_code": "base",
    "home_days_since_last_match": "base",
    "away_matches_last_14_days": "base",
    "days_since_last_match_diff": "base",
    "home_elo": "elo",
    "elo_diff": "elo",
    "home_points_last_5": "form",
    "away_points_last_19_venue": "form",
    "goal_diff_last_5_diff": "form",
    "home_goal_diff_last_19_venue": "form",
    "home_wins_last_5": "form",
    "away_losses_last_19_venue": "form",
    "home_shots_on_target_for_last_19_venue": "form",
    "shot_accuracy_last_5_diff": "form",
    "home_matches_used_last_19": "form",
    "home_xg_for_last_5": "xg",
    "xg_diff_last_19_venue_diff": "xg",
    "npxg_diff_last_5_diff": "xg",
    "home_xg_overperformance_last_19": "xg",
    "away_xg_per_shot_last_5_venue": "xg",
    "home_deep_completions_last_5": "xg",
    "ppda_last_19_diff": "xg",
    "home_strength_overall": "squad",
    "strength_goalkeeper_diff": "squad",
}


@pytest.mark.parametrize(("column", "expected"), sorted(REAL_COLUMNS.items()))
def test_family_of_places_real_columns(column: str, expected: str) -> None:
    assert family_of(column) == expected


def test_unknown_feature_is_fatal() -> None:
    """A new feature must break the experiment rather than vanish from it.

    If this ever becomes a silent default, the ladder starts reporting the
    contribution of a *subset* of the model while claiming to cover all of it.
    """
    with pytest.raises(KeyError, match="matches no family"):
        family_of("home_possession_last_5")


def test_stems_containing_diff_are_not_mangled() -> None:
    """Only a trailing `_diff` is a differenced column; `goal_diff` is a stem."""
    assert family_of("home_goal_diff_last_5") == "form"
    assert family_of("goal_diff_last_5_diff") == "form"
    assert family_of("home_npxg_diff_last_19") == "xg"
    assert family_of("npxg_diff_last_19_diff") == "xg"


def test_families_are_disjoint() -> None:
    """No column may match two families, or the partition double-counts."""
    import re

    from footballml.experiments.ablation import _stem

    for column in REAL_COLUMNS:
        hits = [
            family
            for family, patterns in FAMILY_PATTERNS.items()
            if any(re.match(p, _stem(column)) for p in patterns)
        ]
        assert len(hits) == 1, f"{column} matched {hits}"


def test_ladder_is_nested_and_ends_at_full() -> None:
    """Each rung must be a superset of the one before, and the last is everything."""
    sets = [set(f) for _, f in LADDER[:-1]]
    for smaller, larger in zip(sets, sets[1:], strict=False):
        assert smaller < larger, f"{smaller} is not a strict subset of {larger}"
    assert sets[-1] == set(FAMILY_PATTERNS), "the last named rung must hold every family"
    assert LADDER[-1] == ("full", ()), "the final rung must be the production sentinel"


def test_leave_one_out_families_exist() -> None:
    assert set(LEAVE_ONE_OUT) <= set(FAMILY_PATTERNS)
    assert "base" not in LEAVE_ONE_OUT, "base is the floor; dropping it is not an ablation"


def test_rung_columns_preserves_production_order() -> None:
    cols = ["league_code", "home_elo", "home_points_last_5", "home_strength_overall"]
    assert rung_columns(cols, ("base", "elo")) == ["league_code", "home_elo"]
    assert rung_columns(cols, ()) == cols  # full rung is unfiltered
    with pytest.raises(ValueError, match="unknown families"):
        rung_columns(cols, ("nonsense",))


# --- the partition against the real feature table -----------------------------

@pytest.fixture(scope="module")
def production_columns() -> list[str]:
    """Model inputs from the real built feature table, or skip."""
    from footballml.data import PROCESSED_DIR, load_team_match_history, load_team_strength
    from footballml.features.build import build_match_features

    path = PROCESSED_DIR / "team_match_history_all.csv"
    if not path.exists():
        pytest.skip("team_match_history_all.csv not built; run `pipelines.build_dataset`")
    tmh = load_team_match_history(path)
    recent = tmh[tmh["Season"].astype(str).isin(["2324", "2425"])].copy()
    return feature_columns(build_match_features(recent, strength=load_team_strength()))


def test_partition_covers_every_production_feature(production_columns: list[str]) -> None:
    """The headline guarantee: families sum to the production model, exactly."""
    grouped = classify(production_columns)
    covered = sum(len(c) for c in grouped.values())
    assert covered == len(production_columns)
    # And no column appears twice.
    flat = [c for cols in grouped.values() for c in cols]
    assert len(set(flat)) == len(flat)


def test_full_rung_is_every_production_feature(production_columns: list[str]) -> None:
    """`full` must be the production model, not the union of what we classified.

    These are equal only while the partition is exhaustive, which is the point:
    if a family stops covering something, the ladder's top rung still runs the
    real model and the discrepancy surfaces.
    """
    assert rung_columns(production_columns, ()) == production_columns
    union = tuple(FAMILY_PATTERNS)
    assert set(rung_columns(production_columns, union)) == set(production_columns)


def test_every_ladder_rung_is_a_real_subset(production_columns: list[str]) -> None:
    previous: set[str] = set()
    for name, families in LADDER:
        cols = set(rung_columns(production_columns, families))
        assert cols, f"rung {name} selected no features"
        assert previous <= cols, f"rung {name} dropped features the previous rung had"
        assert cols <= set(production_columns), f"rung {name} invented features"
        previous = cols


def test_id_columns_are_not_features(production_columns: list[str]) -> None:
    """Identifiers and targets must never be selectable as model inputs."""
    assert not set(ID_COLUMNS) & set(production_columns)


def test_rung_frame_exposes_only_its_own_features(production_columns: list[str]) -> None:
    """Building a rung's frame must not let an excluded column back in.

    This is the check that `run_rung` performs at runtime, exercised here on a
    frame shaped like the real one -- it is the single mistake that would
    invalidate every number the experiment produces.
    """
    wanted = rung_columns(production_columns, ("base", "elo"))
    frame = pd.DataFrame(
        {c: np.zeros(3) for c in production_columns}
        | {c: ["x", 1, "2024-01-01", "A", "B", 1, 0, "H"][i % 8] for i, c in enumerate(ID_COLUMNS)}
    )
    restricted = frame[[*ID_COLUMNS, *wanted]]
    assert set(feature_columns(restricted)) == set(wanted)


# --- the evaluation primitives ------------------------------------------------

def test_rps_per_match_means_to_the_headline_score() -> None:
    """One definition of RPS: the bootstrap and the headline cannot diverge."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2, 2], size=500)
    y = rng.choice(["H", "D", "A"], size=500)
    assert rps_per_match(y, probs).mean() == pytest.approx(
        ranked_probability_score(y, probs)
    )


def test_rps_per_match_known_values() -> None:
    """Hand-computed, so the formula itself is pinned.

    For a home win with p = (0.5, 0.3, 0.2): cumulative predicted (0.5, 0.8),
    cumulative observed (1, 1), so RPS = ((0.5-1)^2 + (0.8-1)^2) / 2 = 0.145.
    A perfect forecast scores 0; the worst possible scores 1.
    """
    assert rps_per_match(["H"], np.array([[0.5, 0.3, 0.2]]))[0] == pytest.approx(0.145)
    assert rps_per_match(["H"], np.array([[1.0, 0.0, 0.0]]))[0] == pytest.approx(0.0)
    assert rps_per_match(["A"], np.array([[1.0, 0.0, 0.0]]))[0] == pytest.approx(1.0)


def test_paired_bootstrap_finds_a_real_difference() -> None:
    rng = np.random.default_rng(1)
    baseline = rng.normal(0.2, 0.05, 4000)
    variant = baseline - 0.01  # uniformly better
    out = paired_bootstrap(baseline, variant, n_boot=500)
    assert out["delta"] == pytest.approx(-0.01, abs=1e-9)
    assert out["excludes_zero"]
    assert out["hi"] < 0


def test_paired_bootstrap_reports_no_difference_when_there_is_none() -> None:
    """A difference that is real per match but zero on average must not register.

    The offsets are exactly +/-0.01 in alternation, so the mean difference is
    exactly 0 however noisy the individual matches are. Drawing the offsets at
    random instead would *not* test this: the bootstrap is sensitive enough to
    resolve a mean difference far below the per-match noise -- paired differences
    with sd 1e-6 over 4,000 fixtures give a standard error near 1.6e-8 -- so a
    sample mean that merely happens to be small still reads as significant, and
    correctly so.
    """
    rng = np.random.default_rng(2)
    baseline = rng.normal(0.2, 0.05, 4000)
    variant = baseline + np.tile([0.01, -0.01], 2000)
    out = paired_bootstrap(baseline, variant, n_boot=500)
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)
    assert not out["excludes_zero"]
    assert out["lo"] < 0 < out["hi"]


def test_paired_bootstrap_is_deterministic() -> None:
    """A reported interval must be reproducible from the committed seed."""
    a, b = np.linspace(0, 1, 200), np.linspace(0, 1, 200) - 0.02
    assert paired_bootstrap(a, b, n_boot=200) == paired_bootstrap(a, b, n_boot=200)


def test_train_from_default_changes_nothing() -> None:
    """`train_from=None` must be the production path, byte for byte.

    The training-window A/B compares against production, so if adding the option
    perturbed the default the experiment would be measuring its own change. A
    cutoff older than the data is also a no-op, which pins that the filter is
    inclusive at the boundary rather than off by a season.
    """
    from pipelines.backtest import run_backtest

    rng = np.random.default_rng(3)
    n = 600
    frame = pd.DataFrame(
        {
            "League": "E0",
            "Season": rng.choice([1011, 1112, 1213], n),
            "Date": pd.date_range("2010-08-01", periods=n, freq="D"),
            "HomeTeam": [f"H{i}" for i in range(n)],
            "AwayTeam": [f"A{i}" for i in range(n)],
            "FTHG": rng.poisson(1.5, n),
            "FTAG": rng.poisson(1.2, n),
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
        }
    )
    frame["FTR"] = np.where(
        frame.FTHG > frame.FTAG, "H", np.where(frame.FTHG < frame.FTAG, "A", "D")
    )

    baseline, base_preds = run_backtest(frame, start_season=1112, calibrate=False)
    for cutoff in (None, 1011, 900):
        per_season, preds = run_backtest(
            frame, start_season=1112, calibrate=False, train_from=cutoff
        )
        pd.testing.assert_frame_equal(per_season, baseline)
        pd.testing.assert_frame_equal(preds, base_preds)


def test_train_from_actually_restricts_training() -> None:
    """And a cutoff inside the data must shrink the training set."""
    from pipelines.backtest import run_backtest

    rng = np.random.default_rng(4)
    n = 900
    frame = pd.DataFrame(
        {
            "League": "E0",
            "Season": np.repeat([1011, 1112, 1213], n // 3),
            "Date": pd.date_range("2010-08-01", periods=n, freq="D"),
            "HomeTeam": [f"H{i}" for i in range(n)],
            "AwayTeam": [f"A{i}" for i in range(n)],
            "FTHG": rng.poisson(1.5, n),
            "FTAG": rng.poisson(1.2, n),
            "feat_a": rng.normal(size=n),
        }
    )
    frame["FTR"] = np.where(
        frame.FTHG > frame.FTAG, "H", np.where(frame.FTHG < frame.FTAG, "A", "D")
    )

    wide, wide_preds = run_backtest(frame, start_season=1213, calibrate=False)
    narrow, narrow_preds = run_backtest(
        frame, start_season=1213, calibrate=False, train_from=1112
    )
    assert narrow["n_train"].iloc[0] < wide["n_train"].iloc[0]
    # Same fixtures predicted either way -- the precondition for pairing.
    pd.testing.assert_frame_equal(
        wide_preds[["HomeTeam", "AwayTeam"]], narrow_preds[["HomeTeam", "AwayTeam"]]
    )


def test_paired_bootstrap_demands_matched_fixtures() -> None:
    """Unequal lengths mean the pairing is wrong, which must fail loudly."""
    with pytest.raises(ValueError, match="matched fixtures"):
        paired_bootstrap(np.zeros(10), np.zeros(9))
    with pytest.raises(ValueError, match="per-match"):
        paired_bootstrap(np.zeros((10, 3)), np.zeros((10, 3)))
