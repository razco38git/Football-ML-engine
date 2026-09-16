"""Tests for model persistence and feature labelling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footballml import registry
from footballml.labels import humanise
from footballml.models.match_model import MatchPredictor


@pytest.fixture
def trained() -> MatchPredictor:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((120, 4)), columns=["a", "b", "c", "d"])
    return MatchPredictor(gbm_params={"loss": "poisson", "max_iter": 10}).fit(
        X, rng.poisson(1.5, 120), rng.poisson(1.1, 120)
    )


def test_save_load_roundtrip(trained: MatchPredictor, tmp_path: Path) -> None:
    meta = registry.ModelMetadata(
        version="test-1",
        trained_at="2026-09-15T00:00:00+00:00",
        trained_through="2026-09-14",
        n_train=120,
        leagues=["E0"],
        feature_names=["a", "b", "c", "d"],
        rho=-0.06,
        metrics={"rps": 0.205},
    )
    registry.save(trained, meta, tmp_path)

    model, loaded = registry.load("latest", tmp_path)
    assert loaded.version == "test-1"
    assert loaded.n_features == 4
    assert loaded.metrics["rps"] == 0.205

    # The restored model must predict identically to the original.
    X = pd.DataFrame(np.random.default_rng(1).random((5, 4)), columns=["a", "b", "c", "d"])
    assert np.allclose(model.predict_proba(X), trained.predict_proba(X))


def test_missing_model_raises_with_guidance(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pipelines.train"):
        registry.load("latest", tmp_path)


def test_versions_are_unique_per_build() -> None:
    """Two builds of the same data window must be distinguishable.

    Otherwise a retrain after a feature change silently collides with the old
    version, and stored predictions can no longer be attributed correctly.
    """
    assert registry.make_version("2026-09-14") != registry.make_version("2026-09-14")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("home_xg_for_last_5", "home team xG created (last 5)"),
        ("away_ppda_last_5_venue", "away team pressing intensity (last 5, at this venue)"),
        ("points_last_5_diff", "points (last 5, edge over opponent)"),
        ("league_code", "league"),
        ("days_since_last_match", "days of rest"),
        ("home_strength_overall", "home team squad rating"),
        ("away_strength_attack", "away team attack rating"),
        ("home_strength_defence", "home team defence rating"),
        ("strength_overall_diff", "squad rating (edge over opponent)"),
    ],
)
def test_humanise(raw: str, expected: str) -> None:
    assert humanise(raw) == expected


def test_humanise_handles_double_diff() -> None:
    """``goal_diff_last_5_diff`` contains ``_diff`` twice, meaning different things.

    The first is part of the metric name, the second marks a home-minus-away
    comparison. Rewriting both produced "goal edge, last 5 edge".
    """
    assert humanise("goal_diff_last_5_diff") == "goal difference (last 5, edge over opponent)"
    assert humanise("home_goal_diff_last_5_venue") == (
        "home team goal difference (last 5, at this venue)"
    )
