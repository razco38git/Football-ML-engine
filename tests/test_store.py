"""Tests for prediction storage.

The append-only guarantee is what makes published accuracy trustworthy, so it
gets tested directly rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from footballml import store

#: A match date far enough ahead that `append`, which stamps `predicted_at`
#: with the current time, writes a forecast `settled` can prove came first.
#: Fixed dates in the past silently became post-kickoff forecasts and were
#: refused, which is the rule working rather than a broken test.
KICKOFF = pd.Timestamp.today().normalize() + pd.Timedelta(days=3)


@pytest.fixture
def predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "League": ["E0", "E0"],
            "Date": [KICKOFF, KICKOFF],
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
            "Date": [KICKOFF],
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
        "Date": [KICKOFF],
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


def test_settled_counts_each_fixture_once(tmp_path):
    """A fixture re-predicted after a retrain must not count twice.

    `append` de-duplicates per model version, so a fixture forecast before
    kickoff and forecast again after a retrain -- still before kickoff -- is
    stored twice. Nine La Liga fixtures became an eighteen-match "record".
    """
    import pandas as pd

    from footballml import store

    path = tmp_path / "predictions.csv"
    fixture = pd.DataFrame(
        [{
            "League": "SP1", "Date": str(KICKOFF.date()), "HomeTeam": "Betis",
            "AwayTeam": "Getafe", "expected_goals_home": 1.4,
            "expected_goals_away": 1.0, "prob_home_win": 0.5, "prob_draw": 0.3,
            "prob_away_win": 0.2, "predicted_outcome": "H",
            "modal_score_home": 1, "modal_score_away": 1,
            "prob_over_2_5": 0.4, "prob_btts": 0.5,
        }]
    )
    store.append(fixture, "model-a", path=path)
    store.append(fixture, "model-b", path=path)
    assert len(store.load(path)) == 2, "both versions are kept in the store"

    raw = pd.read_csv(path)
    raw["actual_result"] = "H"
    raw["actual_home_goals"] = 1
    raw["actual_away_goals"] = 0
    raw.to_csv(path, index=False)

    assert len(store.settled(path)) == 1, "the published record counts it once"


# --- which of several forecasts gets published --------------------------------
#
# A fixture is forecast repeatedly: `score_upcoming` runs daily and stores one
# row per model version. The published record keeps exactly one, and which one
# is a claim about what we said, so it is pinned here rather than left to a
# `keep=` argument nobody reads.
#
# It used to keep the earliest, as "the most conservative claim". That was the
# wrong axis: the Match Predictor serves the *current* model, so a reader saw
# 44/25/32 for Roma v Real Madrid before kickoff and 47/24/29 afterwards -- a
# ten-day-old forecast from a model three versions superseded. All 61 settled
# fixtures held more than one forecast and 58 disagreed, by up to sixteen
# points.

def _forecast(date, home="Roma", away="Real Madrid", **over) -> dict:
    row = {
        "League": "I1", "Date": str(date), "HomeTeam": home, "AwayTeam": away,
        "expected_goals_home": 1.7, "expected_goals_away": 1.4,
        "prob_home_win": 0.47, "prob_draw": 0.24, "prob_away_win": 0.29,
        "predicted_outcome": "H", "modal_score_home": 1, "modal_score_away": 1,
        "prob_over_2_5": 0.52, "prob_btts": 0.54,
    }
    return {**row, **over}


def _write(path: Path, rows: list[dict]) -> None:
    """Write the store directly, so `predicted_at` can be placed deliberately."""
    frame = pd.DataFrame(rows)
    frame["actual_result"] = "A"
    frame["actual_home_goals"] = 3
    frame["actual_away_goals"] = 4
    frame.to_csv(path, index=False)


def test_the_last_forecast_before_kickoff_is_published(path: Path) -> None:
    """What the site actually showed, not the oldest thing it ever said."""
    kickoff = pd.Timestamp("2026-10-14")
    _write(path, [
        _forecast(kickoff, prob_home_win=0.47, prob_draw=0.24, prob_away_win=0.29,
                  model_version="a", predicted_at="2026-10-04T17:57:35+00:00"),
        _forecast(kickoff, prob_home_win=0.49, prob_draw=0.23, prob_away_win=0.28,
                  model_version="b", predicted_at="2026-10-05T08:41:52+00:00"),
        _forecast(kickoff, prob_home_win=0.44, prob_draw=0.25, prob_away_win=0.32,
                  model_version="c", predicted_at="2026-10-12T12:22:46+00:00"),
    ])

    out = store.settled(path)
    assert len(out) == 1
    assert out["prob_home_win"].iloc[0] == pytest.approx(0.44)
    assert out["model_version"].iloc[0] == "c"


def test_a_forecast_made_after_kickoff_is_refused(path: Path) -> None:
    """`score_upcoming` forecasts whatever still looks unplayed.

    When the results feed lags, that includes matches already played: fourteen
    stored rows postdate their fixture, one by a day and a half. Publishing one
    as a prediction committed in advance would be false.
    """
    kickoff = pd.Timestamp("2026-09-15")
    _write(path, [
        _forecast(kickoff, prob_home_win=0.47, predicted_at="2026-09-10T06:00:00+00:00",
                  model_version="before"),
        _forecast(kickoff, prob_home_win=0.90, predicted_at="2026-09-16T14:36:11+00:00",
                  model_version="after"),
    ])

    out = store.settled(path)
    assert len(out) == 1
    assert out["model_version"].iloc[0] == "before"
    assert out["prob_home_win"].iloc[0] == pytest.approx(0.47)


def test_a_fixture_with_only_late_forecasts_is_dropped(path: Path) -> None:
    """Rather than counted on one of them. Three real fixtures do this.

    All three La Liga matches of 2026-09-15 were forecast at 13:02 that day and
    again the next afternoon, and never before. There is no honest number to
    publish for them, so they leave the record -- which makes the claim smaller
    and truer.
    """
    kickoff = pd.Timestamp("2026-09-15")
    _write(path, [
        _forecast(kickoff, predicted_at="2026-09-15T13:02:46+00:00", model_version="a"),
        _forecast(kickoff, predicted_at="2026-09-16T14:36:11+00:00", model_version="b"),
    ])
    assert store.settled(path).empty


def test_the_bound_is_the_match_date_because_we_have_no_kickoff_times(path: Path) -> None:
    """A forecast made the previous evening counts; one made that morning does not.

    The store holds dates, not kickoff times, so midnight is the latest instant
    we can prove came first. A lunchtime forecast on the day of an evening
    kickoff is genuinely pre-match and is still refused -- the cost of a bound
    that can be proved.
    """
    kickoff = pd.Timestamp("2026-10-14")
    _write(path, [
        _forecast(kickoff, predicted_at="2026-10-13T23:59:00+00:00", model_version="eve"),
    ])
    assert len(store.settled(path)) == 1

    _write(path, [
        _forecast(kickoff, predicted_at="2026-10-14T00:01:00+00:00", model_version="morn"),
    ])
    assert store.settled(path).empty


def test_fixtures_are_still_counted_once_each(path: Path) -> None:
    """The original guarantee, unchanged by picking a different row."""
    kickoff = pd.Timestamp("2026-10-14")
    _write(path, [
        _forecast(kickoff, predicted_at="2026-10-04T06:00:00+00:00", model_version="a"),
        _forecast(kickoff, predicted_at="2026-10-05T06:00:00+00:00", model_version="b"),
        _forecast(kickoff, home="Arsenal", away="Lille",
                  predicted_at="2026-10-05T06:00:00+00:00", model_version="b"),
    ])
    out = store.settled(path)
    assert len(out) == 2
    assert set(out["HomeTeam"]) == {"Roma", "Arsenal"}


def test_a_store_without_timestamps_still_works(path: Path) -> None:
    """Nothing can be proved about ordering, so the rows are left alone."""
    kickoff = pd.Timestamp("2026-10-14")
    rows = [
        _forecast(kickoff, prob_home_win=0.47, model_version="a"),
        _forecast(kickoff, prob_home_win=0.44, model_version="b"),
    ]
    frame = pd.DataFrame(rows)
    frame["actual_result"] = "A"
    frame.to_csv(path, index=False)

    out = store.settled(path)
    assert len(out) == 1
    assert out["prob_home_win"].iloc[0] == pytest.approx(0.44), "last on file order"
