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

import pandas as pd

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
