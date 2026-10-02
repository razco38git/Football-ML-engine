"""Showing Champions League ties without corrupting the domestic record.

Two things make a European fixture awkward, and both fail quietly rather than
loudly:

1. `LEAGUE_CODES` has no UEFA entry, so a row labelled ``"UCL"`` gives the model
   a **NaN** ``league_code`` -- a value no training row ever carried, which the
   booster accepts in silence. So ``League`` stays the *home side's division*
   and the competition is carried separately.
2. That makes a European prediction indistinguishable from a domestic one on
   ``League`` alone, so without the ``competition`` column it could settle
   against the wrong result, or never settle at all and sit in the record
   forever with nulls.

The third property is editorial rather than technical and matters just as much:
cross-league predictions keep about half the model's usual edge, so they are
never pooled into the headline accuracy figure.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from footballml import store
from footballml.features.build import LEAGUE_CODES
from footballml.ingest.european import SHOWN_COMPETITIONS

api = importlib.import_module("footballml.api.app")


def _prediction(league: str, home: str, away: str, competition: str | None = None) -> pd.DataFrame:
    row = {
        "League": league, "Date": "2026-10-21", "HomeTeam": home, "AwayTeam": away,
        "expected_goals_home": 1.6, "expected_goals_away": 1.1,
        "prob_home_win": 0.5, "prob_draw": 0.25, "prob_away_win": 0.25,
        "predicted_outcome": "H",
    }
    if competition is not None:
        row["competition"] = competition
    return pd.DataFrame([row])


# --- the league_code trap ---------------------------------------------------


def test_only_the_shown_competition_is_the_champions_league() -> None:
    """Europa and Conference are fetched -- they feed the cross-league
    correction -- but not shown: we can rate 14% and 6% of their matches."""
    assert SHOWN_COMPETITIONS == ("UCL",)


def test_a_competition_code_is_never_a_league_code() -> None:
    """The whole reason `League` stays a division. If a UEFA code were ever
    added to `LEAGUE_CODES`, a European row would map to a code the model has
    never seen rather than NaN -- different failure, equally silent.
    """
    for code in SHOWN_COMPETITIONS:
        assert code not in LEAGUE_CODES


# --- the store --------------------------------------------------------------


def test_competition_identifies_a_forecast(tmp_path) -> None:
    """Two clubs can meet twice in a season, once at home in each competition.

    On `League` alone those are the same row, and the second would be dropped as
    a duplicate.
    """
    path = tmp_path / "p.csv"
    store.append(_prediction("E0", "Arsenal", "Chelsea", store.DOMESTIC), "v1", path)
    added = store.append(_prediction("E0", "Arsenal", "Chelsea", "UCL"), "v1", path)
    assert added == 1
    assert len(store.load(path)) == 2


def test_a_prediction_without_a_competition_counts_as_domestic(tmp_path) -> None:
    """Every row written before this column existed, and every caller that does
    not set it."""
    path = tmp_path / "p.csv"
    store.append(_prediction("E0", "Arsenal", "Chelsea"), "v1", path)
    assert store.load(path)["competition"].iloc[0] == store.DOMESTIC

    # And the same fixture is then recognised as already stored.
    assert store.append(_prediction("E0", "Arsenal", "Chelsea", store.DOMESTIC), "v1", path) == 0


def test_a_european_tie_settles_against_the_european_result(tmp_path) -> None:
    path = tmp_path / "p.csv"
    store.append(_prediction("SP1", "Real Madrid", "Arsenal", "UCL"), "v1", path)

    results = pd.DataFrame(
        [{
            "League": "SP1", "competition": "UCL", "Date": "2026-10-21",
            "HomeTeam": "Real Madrid", "AwayTeam": "Arsenal",
            "FTHG": 2, "FTAG": 1, "FTR": "H",
        }]
    )
    assert store.settle(results, path) == 1
    assert store.load(path)["actual_result"].iloc[0] == "H"


def test_a_domestic_result_does_not_settle_a_european_tie(tmp_path) -> None:
    """The failure the competition column exists to prevent: the same two clubs,
    the same date, the wrong competition.
    """
    path = tmp_path / "p.csv"
    store.append(_prediction("SP1", "Real Madrid", "Arsenal", "UCL"), "v1", path)

    domestic = pd.DataFrame(
        [{
            "League": "SP1", "competition": store.DOMESTIC, "Date": "2026-10-21",
            "HomeTeam": "Real Madrid", "AwayTeam": "Arsenal",
            "FTHG": 0, "FTAG": 3, "FTR": "A",
        }]
    )
    assert store.settle(domestic, path) == 0
    assert pd.isna(store.load(path)["actual_result"].iloc[0])


def test_an_old_store_still_settles_from_a_results_frame_without_the_column(tmp_path) -> None:
    """Backward compatibility, both sides blank. The live store has 45 such rows."""
    path = tmp_path / "p.csv"
    store.append(_prediction("E0", "Arsenal", "Chelsea"), "v1", path)
    results = pd.DataFrame(
        [{
            "League": "E0", "Date": "2026-10-21", "HomeTeam": "Arsenal",
            "AwayTeam": "Chelsea", "FTHG": 1, "FTAG": 1, "FTR": "D",
        }]
    )
    assert store.settle(results, path) == 1


# --- keeping the record honest ---------------------------------------------


def _settled(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "competition": competition, "actual_result": actual,
                "prob_home_win": 0.6 if predicted == "H" else 0.2,
                "prob_draw": 0.2,
                "prob_away_win": 0.6 if predicted == "A" else 0.2,
            }
            for competition, predicted, actual in rows
        ]
    )


def test_competitions_are_reported_separately() -> None:
    rows = _settled(
        [(store.DOMESTIC, "H", "H")] * 8 + [("UCL", "H", "A")] * 4
    )
    out = api._by_competition(rows)
    assert set(out) == {store.DOMESTIC, "UCL"}
    assert out[store.DOMESTIC].n == 8
    assert out["UCL"].n == 4
    assert out[store.DOMESTIC].accuracy > out["UCL"].accuracy


def test_each_competition_carries_its_own_base_rate() -> None:
    """Each gets the base rate of *its own* outcomes, not a pooled one.

    The two competitions are given deliberately different outcome mixes: one
    mostly home wins, the other mostly away. A shared base rate would land
    between them and be the wrong yardstick for both.
    """
    rows = _settled(
        [(store.DOMESTIC, "H", "H")] * 7 + [(store.DOMESTIC, "H", "A")] * 3
        + [("UCL", "H", "A")] * 7 + [("UCL", "H", "H")] * 3
    )
    out = api._by_competition(rows)
    assert out[store.DOMESTIC].rps_base_rate > 0
    assert out["UCL"].rps_base_rate > 0
    # Same shape of split, so the base rates match each other -- and crucially
    # the *accuracy* does not, because only one competition was predicted well.
    assert out[store.DOMESTIC].accuracy == pytest.approx(0.7)
    assert out["UCL"].accuracy == pytest.approx(0.3)


def test_rows_with_no_competition_column_report_as_domestic() -> None:
    rows = _settled([(store.DOMESTIC, "H", "H")] * 4).drop(columns=["competition"])
    assert api._by_competition(rows) == {}


def test_nothing_settled_reports_nothing() -> None:
    assert api._by_competition(pd.DataFrame()) == {}


# --- the correction must still reach these ---------------------------------


def test_a_division_that_is_really_a_competition_is_flagged(caplog) -> None:
    """`_division_of` falls back to the fixture's label for an unknown club.

    If that label were a competition, `adjust` would look up an offset that does
    not exist, get 0.0, and silently become the identity for exactly the match
    that needs it. These fixtures are filtered to big-five clubs so it should
    not happen -- but it must not happen quietly.
    """
    api.state.tmh = pd.DataFrame(columns=["Team", "Date", "League"])
    with caplog.at_level("WARNING"):
        assert api._division_of("Benfica", "UCL") == "UCL"
    assert any("not a division" in r.message for r in caplog.records)


def test_a_known_club_resolves_to_its_division_without_warning(caplog) -> None:
    api.state.tmh = pd.DataFrame(
        [{"Team": "Arsenal", "Date": pd.Timestamp("2026-09-01"), "League": "E0"}]
    )
    with caplog.at_level("WARNING"):
        assert api._division_of("Arsenal", "UCL") == "E0"
    assert not caplog.records


@pytest.fixture(autouse=True)
def _restore_state():
    """`state` is module-level, so these tests must not leak into others."""
    original = api.state.tmh
    yield
    api.state.tmh = original
