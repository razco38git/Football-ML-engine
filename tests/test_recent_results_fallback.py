"""Recent results when the results feed falls behind.

football-data is the results source and it lags. In October 2026 the match
history ended 2026-09-20 while the leagues had played on to 2026-10-10, so
"Recent results" stopped three weeks short with nothing to tell a reader that a
stale feed was not a quiet fortnight.

Two things close that gap, and both are pinned here. The schedule pages carry
the scores and were already being scraped for fixtures -- the goals were parsed
and then thrown away. And the forecasts for those matches were already stored,
so once settled they can be served directly.

The risk being guarded is mixing the two silently: a row re-scored by today's
model and a row committed to before kickoff are different claims, so the second
is flagged rather than quietly blended in.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

from footballml.ingest import schedule

api = importlib.import_module("footballml.api.app")


# --- the schedule must stop discarding the score ------------------------------

def test_score_regex_captures_both_goals() -> None:
    """`_SCORE` always had two groups; only group 0 was ever used."""
    found = pd.Series(["2–1", "0–0", "10–3"]).str.extract(schedule._SCORE)
    assert list(found[0]) == ["2", "0", "10"]
    assert list(found[1]) == ["1", "0", "3"]


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "‒", "–", "—", "―"])
def test_every_dash_fbref_might_use_parses(dash: str) -> None:
    """A dash this misses would mark a played match unplayed, and lose its score."""
    found = pd.Series([f"3{dash}2"]).str.extract(schedule._SCORE)
    assert found.iloc[0].tolist() == ["3", "2"]


def test_a_fixture_list_never_carries_result_columns() -> None:
    """`FTHG`/`FTAG` are the training table's name for the truth.

    An unplayed fixture has NaN in both, and letting those ride along would put
    empty truth columns beside rows the model is about to predict.
    """
    assert set(schedule._RESULT_ONLY) == {"FTHG", "FTAG", "played"}
    empty = schedule._empty()
    for column in schedule._RESULT_ONLY:
        assert column in empty.columns


# --- /matches falls back to the store -----------------------------------------

def _stored(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["E0"] * len(dates),
        "competition": ["domestic"] * len(dates),
        "Date": pd.to_datetime(dates),
        "HomeTeam": [f"Home{i}" for i in range(len(dates))],
        "AwayTeam": [f"Away{i}" for i in range(len(dates))],
        "expected_goals_home": [1.5] * len(dates),
        "expected_goals_away": [1.1] * len(dates),
        "prob_home_win": [0.5] * len(dates),
        "prob_draw": [0.26] * len(dates),
        "prob_away_win": [0.24] * len(dates),
        "predicted_outcome": ["H"] * len(dates),
        "modal_score_home": [1] * len(dates),
        "modal_score_away": [1] * len(dates),
        "prob_over_2_5": [0.52] * len(dates),
        "prob_btts": [0.54] * len(dates),
        "actual_home_goals": [2.0] * len(dates),
        "actual_away_goals": [1.0] * len(dates),
        "actual_result": ["H"] * len(dates),
    })


def test_only_matches_newer_than_the_feature_table_are_added(monkeypatch) -> None:
    """The store must top up, never duplicate what has already been re-scored."""
    monkeypatch.setattr(
        api.store, "settled",
        lambda *a, **k: _stored(["2026-09-19", "2026-09-20", "2026-10-10"]),
    )
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)
    assert [str(p.date) for p in out] == ["2026-10-10"]


def test_added_rows_are_flagged_as_pre_kickoff(monkeypatch) -> None:
    """A stored forecast and a re-scored one are different claims."""
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-10"]))
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)
    assert out[0].predicted_before_kickoff is True
    assert out[0].actual_result == "H"
    assert out[0].actual_home_goals == 2


def test_rows_are_newest_first(monkeypatch) -> None:
    monkeypatch.setattr(
        api.store, "settled",
        lambda *a, **k: _stored(["2026-10-01", "2026-10-10", "2026-10-05"]),
    )
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)
    assert [str(p.date) for p in out] == ["2026-10-10", "2026-10-05", "2026-10-01"]


def test_the_league_filter_applies(monkeypatch) -> None:
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-10"]))
    assert len(api._recent_from_store(pd.Timestamp("2026-09-20"), "E0")) == 1
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), "SP1") == []


def test_a_caught_up_feature_table_adds_nothing(monkeypatch) -> None:
    """When football-data recovers, the fallback must get out of the way."""
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-10"]))
    assert api._recent_from_store(pd.Timestamp("2026-10-10"), None) == []


def test_an_unreadable_store_does_not_break_the_page(monkeypatch) -> None:
    """Recent results without the top-up beats no recent results at all."""
    def broken(*_a, **_k):
        raise OSError("predictions.csv is locked")

    monkeypatch.setattr(api.store, "settled", broken)
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), None) == []


def test_an_empty_store_adds_nothing(monkeypatch) -> None:
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: pd.DataFrame())
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), None) == []
