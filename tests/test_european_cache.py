"""The feature cache that makes a weekly refit affordable.

Scoring 822 European ties costs ~92 minutes, nearly all of it rebuilding
features. That is intolerable weekly and unnecessary: a past tie's features are
built from the domestic matches *before* it, and history does not move. So they
are cached, and only genuinely new matchdays are built.

The risk a cache like this carries is silence -- serving a stale row that looks
exactly like a fresh one. These pin the two things that stop that: it must be
invalidated when squad strength for a *completed* season changes, and it must
not be invalidated merely because the current season moved, which happens every
week and would put the 92 minutes straight back.
"""

from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest

validate = importlib.import_module("pipelines.validate_european")


def _strength(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["League", "Season", "Team", "strength_overall"])


BASE = [
    ("E0", "2324", "Arsenal", 78.0),
    ("E0", "2324", "Chelsea", 75.0),
    ("SP1", "2324", "Barcelona", 82.0),
    ("E0", "2425", "Arsenal", 79.0),   # the current season, still moving
    ("E0", "2425", "Chelsea", 76.0),
]


def _write(tmp_path, rows) -> object:
    path = tmp_path / "team_strength.csv"
    _strength(rows).to_csv(path, index=False)
    return path


# --- what the fingerprint must and must not notice --------------------------


def test_the_current_season_moving_does_not_invalidate_the_cache(tmp_path) -> None:
    """`build_players` runs every Monday and rewrites the whole file.

    Strength is joined from the *previous* season, so this week's ratings reach
    no past European tie. Hashing the file whole would throw the cache away
    every week and put the 92-minute rebuild back.
    """
    before = validate.strength_fingerprint(_write(tmp_path, BASE))
    moved = [*BASE[:3], ("E0", "2425", "Arsenal", 81.5), ("E0", "2425", "Chelsea", 74.0)]
    assert validate.strength_fingerprint(_write(tmp_path, moved)) == before


def test_a_completed_season_changing_does_invalidate_it(tmp_path) -> None:
    """That means the ratings were rebuilt differently -- a config change, a new
    source -- and every cached row really is stale."""
    before = validate.strength_fingerprint(_write(tmp_path, BASE))
    rebuilt = [("E0", "2324", "Arsenal", 71.0), *BASE[1:]]
    assert validate.strength_fingerprint(_write(tmp_path, rebuilt)) != before


def test_a_club_added_to_a_completed_season_invalidates_it(tmp_path) -> None:
    before = validate.strength_fingerprint(_write(tmp_path, BASE))
    added = [*BASE, ("E0", "2324", "Everton", 70.0)]
    assert validate.strength_fingerprint(_write(tmp_path, added)) != before


def test_row_order_alone_does_not_invalidate_it(tmp_path) -> None:
    """The file is regenerated weekly; a reordering is not a change."""
    before = validate.strength_fingerprint(_write(tmp_path, BASE))
    assert validate.strength_fingerprint(_write(tmp_path, list(reversed(BASE)))) == before


def test_a_missing_strength_file_is_its_own_fingerprint(tmp_path) -> None:
    assert validate.strength_fingerprint(tmp_path / "nope.pkl") == "absent"


# --- loading it -------------------------------------------------------------


def _cached(fingerprint: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": pd.Timestamp("2024-09-17"), "HomeTeam": "Arsenal",
                "AwayTeam": "Barcelona", "home_elo": 1600.0,
                validate.FINGERPRINT_COLUMN: fingerprint,
            }
        ]
    )


def test_a_cache_from_a_different_strength_build_is_refused(tmp_path) -> None:
    path = tmp_path / "features.pkl"
    _cached("old").to_pickle(path)
    assert validate.load_cache("new", path).empty


def test_a_matching_cache_loads(tmp_path) -> None:
    path = tmp_path / "features.pkl"
    _cached("same").to_pickle(path)
    loaded = validate.load_cache("same", path)
    assert len(loaded) == 1
    assert loaded["Date"].iloc[0] == pd.Timestamp("2024-09-17")


def test_a_cache_without_the_fingerprint_column_is_refused(tmp_path) -> None:
    """An older or hand-edited file must not be trusted rather than assumed
    current -- there is no way to tell what it was built from."""
    path = tmp_path / "features.pkl"
    _cached("x").drop(columns=[validate.FINGERPRINT_COLUMN]).to_pickle(path)
    assert validate.load_cache("x", path).empty


def test_a_missing_cache_is_not_an_error(tmp_path) -> None:
    assert validate.load_cache("anything", tmp_path / "nope.pkl").empty


def test_a_cache_holding_two_fingerprints_is_refused(tmp_path) -> None:
    """A half-written file from an interrupted run. Reusing the valid half and
    silently dropping the rest would be worse than rebuilding.
    """
    path = tmp_path / "features.pkl"
    pd.concat([_cached("a"), _cached("b")]).to_pickle(path)
    assert validate.load_cache("a", path).empty


# --- using it ---------------------------------------------------------------


def test_only_uncached_matches_are_rebuilt(monkeypatch) -> None:
    """The property the weekly job depends on: a week with no new European
    football must build nothing at all.
    """
    european = pd.DataFrame(
        [
            {"League": "UCL", "Season": "2425", "Date": pd.Timestamp("2024-09-17"),
             "HomeTeam": "Arsenal", "AwayTeam": "Barcelona", "FTHG": 2, "FTAG": 1,
             "FTR": "H", "home_league": "E0", "away_league": "SP1"},
        ]
    )
    cached = pd.DataFrame(
        [
            {"Date": pd.Timestamp("2024-09-17"), "HomeTeam": "Arsenal",
             "AwayTeam": "Barcelona", "League": "E0", "Season": "2425",
             "FTHG": float("nan"), "FTAG": float("nan"), "FTR": None,
             "home_elo": 1600.0, "away_elo": 1580.0},
        ]
    )

    called = []
    monkeypatch.setattr(
        validate, "build_upcoming_features",
        lambda *a, **k: called.append(1) or pd.DataFrame(),
    )
    scored, raw = validate.build_features(
        european, pd.DataFrame(), pd.DataFrame(), cached=cached
    )
    assert not called, "rebuilt a match that was already cached"
    assert len(scored) == 1
    assert scored["home_elo"].iloc[0] == 1600.0
    # And the results came from the fixture list, not the cache's empty ones.
    assert scored["FTHG"].iloc[0] == 2
    assert scored["competition"].iloc[0] == "UCL"


def test_a_cached_match_missing_from_the_fixture_list_is_dropped() -> None:
    """The cache is a store, not a source: it must not resurrect a match that
    the fetch no longer returns."""
    european = pd.DataFrame(
        [
            {"League": "UCL", "Season": "2425", "Date": pd.Timestamp("2024-09-17"),
             "HomeTeam": "Arsenal", "AwayTeam": "Barcelona", "FTHG": 2, "FTAG": 1,
             "FTR": "H", "home_league": "E0", "away_league": "SP1"},
        ]
    )
    cached = pd.DataFrame(
        [
            {"Date": pd.Timestamp("2024-09-17"), "HomeTeam": "Arsenal",
             "AwayTeam": "Barcelona", "home_elo": 1600.0},
            {"Date": pd.Timestamp("2020-01-01"), "HomeTeam": "Gone",
             "AwayTeam": "Vanished", "home_elo": 1500.0},
        ]
    )
    _, raw = validate.build_features(european, pd.DataFrame(), pd.DataFrame(), cached=cached)
    assert len(raw) == 1
    assert "Gone" not in set(raw["HomeTeam"])


def test_no_cache_falls_back_to_building(monkeypatch) -> None:
    european = pd.DataFrame(
        [
            {"League": "UCL", "Season": "2425", "Date": pd.Timestamp("2024-09-17"),
             "HomeTeam": "Arsenal", "AwayTeam": "Barcelona", "FTHG": 2, "FTAG": 1,
             "FTR": "H", "home_league": "E0", "away_league": "SP1"},
        ]
    )
    called = []

    def fake(tmh, fixtures, **kwargs):
        called.append(1)
        return fixtures.assign(FTHG=float("nan"), FTAG=float("nan"), FTR=None, home_elo=1.0)

    monkeypatch.setattr(validate, "build_upcoming_features", fake)
    scored, _ = validate.build_features(european, pd.DataFrame(), pd.DataFrame(), cached=None)
    assert called, "nothing was built and nothing was cached"
    assert len(scored) == 1


# --- the format itself ------------------------------------------------------


def test_the_cache_round_trips_floats_exactly(tmp_path) -> None:
    """Why this file is a pickle and not a CSV.

    CSV loses about 2 ULP of a float64, and the booster *bins* its inputs -- a
    value on a split boundary then lands in a different bin and the prediction
    moves for real. Round-tripping this cache through CSV changed 151 of 822
    predictions, two of them enough to flip the predicted outcome. An in-memory
    equivalence test cannot see that, which is exactly how it got shipped once.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-05-22"] * 500),
            "HomeTeam": ["Atalanta"] * 500,
            "AwayTeam": ["Leverkusen"] * 500,
            "home_elo": rng.normal(1500, 80, 500),
            "xg_for_last_5_diff": rng.normal(0, 1.5, 500),
            validate.FINGERPRINT_COLUMN: ["f"] * 500,
        }
    )
    path = tmp_path / "features.pkl"
    frame.to_pickle(path)
    back = validate.load_cache("f", path)

    for column in ("home_elo", "xg_for_last_5_diff"):
        assert np.array_equal(frame[column].to_numpy(), back[column].to_numpy()), (
            f"{column} did not survive the round trip bit-for-bit"
        )


def test_an_unreadable_cache_rebuilds_rather_than_crashing(tmp_path) -> None:
    """A pickle written by another pandas must not take the weekly job down."""
    path = tmp_path / "features.pkl"
    path.write_bytes(b"not a pickle")
    assert validate.load_cache("f", path).empty


# --- surviving an interruption ---------------------------------------------
#
# The cache was written once, after the last of 438 matchdays. Anything that
# stopped the run first threw away every minute of it: one run was killed at
# matchday 200, ~33 minutes in, and the next started again from zero.


def test_progress_is_written_before_the_run_finishes(tmp_path, monkeypatch) -> None:
    """A kill partway through must cost minutes, not the whole rebuild."""
    monkeypatch.setattr(validate, "FEATURES_CACHE", tmp_path / "features.pkl")
    monkeypatch.setattr(validate, "CHECKPOINT_EVERY", 2)

    days = pd.date_range("2024-09-17", periods=5, freq="7D")
    european = pd.DataFrame(
        [{"League": "UCL", "Season": "2425", "Date": d, "HomeTeam": f"H{i}",
          "AwayTeam": f"A{i}", "FTHG": 1, "FTAG": 0, "FTR": "H",
          "home_league": "E0", "away_league": "SP1"} for i, d in enumerate(days)]
    )

    seen = []

    def fake(tmh, fixtures, **kwargs):
        seen.append(len(seen))
        # Fail on the fifth matchday, after two checkpoints have been written.
        if len(seen) == 5:
            raise KeyboardInterrupt
        return fixtures.assign(FTHG=float("nan"), FTAG=float("nan"), FTR=None, home_elo=1.0)

    monkeypatch.setattr(validate, "build_upcoming_features", fake)
    with pytest.raises(KeyboardInterrupt):
        validate.build_features(european, pd.DataFrame(), pd.DataFrame(),
                                cached=None, fingerprint="f")

    kept = validate.load_cache("f", tmp_path / "features.pkl")
    assert len(kept) == 4, "the work done before the interruption was thrown away"


def test_a_checkpoint_cannot_leave_a_half_written_cache(tmp_path, monkeypatch) -> None:
    """Truncated but readable is the dangerous outcome: it still carries a
    valid fingerprint, so the next run loads and trusts it. The write goes
    through a temporary file and a replace."""
    monkeypatch.setattr(validate, "FEATURES_CACHE", tmp_path / "features.pkl")
    good = pd.DataFrame([{"Date": pd.Timestamp("2024-09-17"), "HomeTeam": "A",
                          "AwayTeam": "B", "home_elo": 1500.0}])
    validate._checkpoint([good], "f")
    assert len(validate.load_cache("f", tmp_path / "features.pkl")) == 1
    assert not (tmp_path / "features.pkl.tmp").exists(), "temporary file left behind"


def test_no_cache_writes_nothing(tmp_path, monkeypatch) -> None:
    """`--no-cache` must not quietly start writing one from a checkpoint."""
    monkeypatch.setattr(validate, "FEATURES_CACHE", tmp_path / "features.pkl")
    validate._checkpoint([pd.DataFrame([{"Date": pd.Timestamp("2024-09-17")}])], None)
    assert not (tmp_path / "features.pkl").exists()


# --- the stamp that keeps the correction honest ------------------------------
#
# `validate_european` is optional in the weekly, so when it fails the job
# continues to `fit_league_adjustment` with the *previous* model's predictions
# still on disk. Refitting those and stamping them as the serving model's
# defeats the whole point of the version check: the API compares equal and
# reports a current correction while applying offsets measured against a model
# that no longer exists. These pin the refusal.


def _predictions(version: str | None, n: int = 4) -> pd.DataFrame:
    frame = pd.DataFrame({
        "home_league": ["E0"] * n, "away_league": ["SP1"] * n,
        "expected_goals_home": [1.4] * n, "expected_goals_away": [1.2] * n,
        "FTHG": [1] * n, "FTAG": [0] * n, "FTR": ["H"] * n,
        "Season": ["2425"] * n,
        "prob_home_win": [0.4] * n, "prob_draw": [0.3] * n, "prob_away_win": [0.3] * n,
    })
    if version is not None:
        frame[validate.MODEL_VERSION_COLUMN] = version
    return frame


def _fit_module(monkeypatch, tmp_path, predictions: pd.DataFrame, serving: str):
    """`fit_league_adjustment.main` wired to a throwaway predictions file."""
    fit_mod = importlib.import_module("pipelines.fit_league_adjustment")
    path = tmp_path / "european_predictions.csv"
    predictions.to_csv(path, index=False)
    monkeypatch.setattr(fit_mod, "PREDICTIONS", path)

    class _Meta:
        version = serving

    monkeypatch.setattr(fit_mod.registry, "load", lambda *a, **k: (object(), _Meta()))
    monkeypatch.setattr(sys, "argv", ["fit_league_adjustment"])
    return fit_mod


def test_predictions_from_another_model_are_refused(monkeypatch, tmp_path, capsys):
    """The 12 October failure path: a killed `validate_european` leaves last
    week's scores behind, and this must not stamp them as this week's."""
    fit_mod = _fit_module(monkeypatch, tmp_path, _predictions("old-model"), "new-model")
    written: list = []
    monkeypatch.setattr(fit_mod, "save", lambda *a, **k: written.append(a))

    assert fit_mod.main() == 1
    assert not written, "a correction was fitted on another model's residuals"


def test_predictions_with_no_recorded_model_are_refused(monkeypatch, tmp_path):
    """A file written before the column existed is indistinguishable from one
    written by another model, and that has to mean no rather than yes."""
    fit_mod = _fit_module(monkeypatch, tmp_path, _predictions(None), "new-model")
    written: list = []
    monkeypatch.setattr(fit_mod, "save", lambda *a, **k: written.append(a))

    assert fit_mod.main() == 1
    assert not written


def test_a_mixture_of_model_versions_is_refused(monkeypatch, tmp_path):
    """Half-overwritten: some rows rescored, some not. Equality against a set
    catches this where `in` would have passed it."""
    frame = _predictions("new-model")
    frame.loc[0, validate.MODEL_VERSION_COLUMN] = "old-model"
    fit_mod = _fit_module(monkeypatch, tmp_path, frame, "new-model")
    written: list = []
    monkeypatch.setattr(fit_mod, "save", lambda *a, **k: written.append(a))

    assert fit_mod.main() == 1
    assert not written


def test_a_partly_stamped_file_is_refused(monkeypatch, tmp_path):
    """Found in the pre-push audit. The check read `set(...dropna())`, so a file
    whose rows were *half* stamped with the serving model and half blank passed:
    the blanks were dropped and the remainder matched.

    Nothing writes that shape today -- `validate_european` stamps the whole
    frame in one assignment -- but scoring incrementally would make it the
    normal shape, and rows from an unknown model are the thing being guarded
    against rather than an exception to it.
    """
    frame = _predictions("new-model")
    frame.loc[2:, validate.MODEL_VERSION_COLUMN] = None
    fit_mod = _fit_module(monkeypatch, tmp_path, frame, "new-model")
    written: list = []
    monkeypatch.setattr(fit_mod, "save", lambda *a, **k: written.append(a))

    assert fit_mod.main() == 1
    assert not written, "fitted on rows whose model is unknown"


def test_an_empty_predictions_file_is_refused(monkeypatch, tmp_path):
    """Zero rows carry zero evidence about which model scored them."""
    fit_mod = _fit_module(
        monkeypatch, tmp_path, _predictions("new-model").iloc[:0], "new-model"
    )
    written: list = []
    monkeypatch.setattr(fit_mod, "save", lambda *a, **k: written.append(a))

    assert fit_mod.main() == 1
    assert not written


def test_predictions_from_the_serving_model_are_accepted(monkeypatch, tmp_path):
    """The guard must not block the normal week, or it is just an outage."""
    fit_mod = _fit_module(monkeypatch, tmp_path, _predictions("new-model", n=200), "new-model")
    saved: list = []
    monkeypatch.setattr(fit_mod, "save", lambda fitted, *a: saved.append(fitted) or tmp_path)
    monkeypatch.setattr(
        fit_mod, "fit", lambda *a, **k: {"offsets": {"E0": 0.0}, "n_matches": 200, "holdout": {}}
    )

    assert fit_mod.main() == 0
    assert saved and saved[0]["model_version"] == "new-model"


def test_the_scored_file_records_the_model_that_scored_it(monkeypatch, tmp_path):
    """The producing half of the contract, and the half that is easy to forget:
    without this column the consumer above refuses every week, so the round trip
    is what is worth testing rather than either side alone."""
    scored = _predictions("ignored", n=6).drop(columns=[validate.MODEL_VERSION_COLUMN])
    scored["competition"] = "UCL"
    scored["Date"] = pd.to_datetime("2025-02-11")
    scored["HomeTeam"], scored["AwayTeam"] = "Arsenal", "Real Madrid"

    matches = tmp_path / "european_matches.csv"
    scored.assign(League="UCL").to_csv(matches, index=False)
    out = tmp_path / "european_predictions.csv"

    class _Meta:
        version = "the-model-that-scored-it"
        trained_through = "2026-10-18"

    monkeypatch.setattr(validate, "MATCHES_IN", matches)
    monkeypatch.setattr(validate, "PREDICTIONS_OUT", out)
    monkeypatch.setattr(validate, "BACKTEST", tmp_path / "absent.csv")
    monkeypatch.setattr(validate, "FEATURES_CACHE", tmp_path / "features.pkl")
    monkeypatch.setattr(validate, "load_team_match_history", lambda *a: pd.DataFrame())
    monkeypatch.setattr(validate, "load_team_strength", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(validate.registry, "load", lambda *a, **k: (object(), _Meta()))
    monkeypatch.setattr(validate, "strength_fingerprint", lambda *a, **k: "fp")
    monkeypatch.setattr(validate, "load_cache", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(validate, "build_features", lambda *a, **k: (scored, scored))
    monkeypatch.setattr(validate, "predict_with", lambda frame, *a: frame)
    monkeypatch.setattr(
        sys, "argv", ["validate_european", "--no-domestic-reference"]
    )

    assert validate.main() == 0

    written = pd.read_csv(out)
    assert list(written[validate.MODEL_VERSION_COLUMN].unique()) == [
        "the-model-that-scored-it"
    ]

    # And the consumer accepts exactly what the producer wrote.
    fit_mod = _fit_module(monkeypatch, tmp_path, written, "the-model-that-scored-it")
    monkeypatch.setattr(fit_mod, "save", lambda *a, **k: tmp_path / "adj.json")
    monkeypatch.setattr(
        fit_mod, "fit", lambda *a, **k: {"offsets": {"E0": 0.0}, "n_matches": 6, "holdout": {}}
    )
    assert fit_mod.main() == 0
