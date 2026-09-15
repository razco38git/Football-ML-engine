"""Tests for prediction storage.

The append-only guarantee is what makes published accuracy trustworthy, so it
gets tested directly rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from footballml import store


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "League": ["E0", "E0"],
            "Date": pd.to_datetime(["2026-09-20", "2026-09-20"]),
            "HomeTeam": ["Liverpool", "Arsenal"],
            "AwayTeam": ["Arsenal", "Chelsea"],
            "prob_home_win": [0.5, 0.4],
            "prob_draw": [0.25, 0.3],
            "prob_away_win": [0.25, 0.3],
        }
    )


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "predictions.csv"


def test_append_then_load(predictions: pd.DataFrame, path: Path) -> None:
    assert store.append(predictions, "v1", path) == 2

    loaded = store.load(path)
    assert len(loaded) == 2
    assert set(loaded["model_version"]) == {"v1"}
    assert loaded["predicted_at"].notna().all()
    assert loaded["actual_result"].isna().all(), "new predictions must start unsettled"


def test_append_is_idempotent(predictions: pd.DataFrame, path: Path) -> None:
    """Re-running the scoring job must not duplicate or overwrite forecasts."""
    store.append(predictions, "v1", path)
    assert store.append(predictions, "v1", path) == 0
    assert len(store.load(path)) == 2


def test_a_new_model_version_predicts_the_same_fixture_again(
    predictions: pd.DataFrame, path: Path
) -> None:
    """Different versions are distinct forecasts and must both be kept.

    Otherwise a retrained model would overwrite the record of what the previous
    one actually said before kickoff.
    """
    store.append(predictions, "v1", path)
    assert store.append(predictions, "v2", path) == 2
    assert len(store.load(path)) == 4


def test_settle_attaches_results(predictions: pd.DataFrame, path: Path) -> None:
    store.append(predictions, "v1", path)

    results = pd.DataFrame(
        {
            "League": ["E0"],
            "Date": pd.to_datetime(["2026-09-20"]),
            "HomeTeam": ["Liverpool"],
            "AwayTeam": ["Arsenal"],
            "FTHG": [2.0],
            "FTAG": [1.0],
            "FTR": ["H"],
        }
    )
    assert store.settle(results, path) == 1

    done = store.settled(path)
    assert len(done) == 1
    assert done["actual_result"].iloc[0] == "H"
    assert done["actual_home_goals"].iloc[0] == 2

    # The unplayed fixture is untouched.
    assert len(store.load(path)) == 2


def test_settle_never_rewrites_history(predictions: pd.DataFrame, path: Path) -> None:
    """A settled prediction is immutable, even if the result feed changes."""
    store.append(predictions, "v1", path)
    key = {
        "League": ["E0"],
        "Date": pd.to_datetime(["2026-09-20"]),
        "HomeTeam": ["Liverpool"],
        "AwayTeam": ["Arsenal"],
    }
    store.settle(pd.DataFrame({**key, "FTHG": [2.0], "FTAG": [1.0], "FTR": ["H"]}), path)

    # A contradictory later feed must be ignored, not applied.
    assert store.settle(
        pd.DataFrame({**key, "FTHG": [0.0], "FTAG": [3.0], "FTR": ["A"]}), path
    ) == 0
    assert store.settled(path)["actual_result"].iloc[0] == "H"


def test_empty_store_reads_cleanly(path: Path) -> None:
    assert store.load(path).empty
    assert store.settled(path).empty
    assert store.settle(pd.DataFrame(), path) == 0
