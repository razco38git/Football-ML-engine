"""The `/elo` ranking endpoint.

Two properties matter and neither is obvious from reading the handler. The
ratings it publishes must be the *pre-match* ones the model was given — a
"current" rating that had seen its own match would be a different, leakier
number than the one the rest of the site reports. And "peak" must be each team's
own best across the whole span, not the best rows in a recent slice, which is
what a naive sort would give.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

api = importlib.import_module("footballml.api.app")


@pytest.fixture
def features(monkeypatch) -> pd.DataFrame:
    """Three clubs whose ratings rise, fall and peak at different times."""
    rows = [
        # Arsenal peaks early then declines; Chelsea climbs to finish highest.
        ("E0", "2425", "2025-01-01", "Arsenal", "Chelsea", 1800.0, 1500.0),
        ("E0", "2526", "2026-01-01", "Arsenal", "Liverpool", 1700.0, 1550.0),
        ("E0", "2627", "2026-09-01", "Liverpool", "Arsenal", 1600.0, 1650.0),
        ("E0", "2627", "2026-09-08", "Chelsea", "Arsenal", 1750.0, 1640.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=["League", "Season", "Date", "HomeTeam", "AwayTeam", "home_elo", "away_elo"],
    )
    frame["Date"] = pd.to_datetime(frame["Date"])
    monkeypatch.setattr(api.state, "features", frame)
    return frame


def _call(limit: int = 10):
    return api.elo_ranking(limit=limit)


def test_current_uses_each_team_s_most_recent_match(features) -> None:
    """Not their best match, and not the newest row overall."""
    current = {e.team: e.elo for e in _call().current}
    # Arsenal's last appearance is 2026-09-08 away, rating 1640 -- not the 1800
    # it carried in 2425, and not the 1650 from 2026-09-01.
    assert current["Arsenal"] == pytest.approx(1640.0)
    assert current["Chelsea"] == pytest.approx(1750.0)
    assert current["Liverpool"] == pytest.approx(1600.0)


def test_current_is_ordered_strongest_first(features) -> None:
    elos = [e.elo for e in _call().current]
    assert elos == sorted(elos, reverse=True)
    assert _call().current[0].team == "Chelsea"


def test_peak_is_each_team_s_own_best_across_the_whole_span(features) -> None:
    """Arsenal's 1800 is three seasons old and must still win the peak board."""
    peak = {e.team: e.elo for e in _call().peak}
    assert peak["Arsenal"] == pytest.approx(1800.0)
    assert peak["Chelsea"] == pytest.approx(1750.0)
    assert _call().peak[0].team == "Arsenal"


def test_peak_reports_when_the_peak_happened(features) -> None:
    """A peak rating is meaningless without its date, and the season labels it."""
    arsenal = next(e for e in _call().peak if e.team == "Arsenal")
    assert arsenal.season == "2425"
    assert arsenal.date == "2025-01-01"


def test_each_team_appears_once_per_board(features) -> None:
    for board in (_call().current, _call().peak):
        names = [e.team for e in board]
        assert len(names) == len(set(names))


def test_limit_is_respected(features) -> None:
    assert len(_call(limit=2).current) == 2
    assert len(_call(limit=2).peak) == 2


def test_start_is_the_elo_anchor(features) -> None:
    """The UI renders ratings as a distance from this, so it must be the real one."""
    from footballml.features.elo import START

    assert _call().start == START


def test_empty_feature_table_is_a_503(monkeypatch) -> None:
    """Not a 500, and not an empty board that looks like a real answer."""
    monkeypatch.setattr(api.state, "features", pd.DataFrame())
    with pytest.raises(api.HTTPException) as caught:
        _call()
    assert caught.value.status_code == 503


def test_missing_elo_columns_is_a_503(monkeypatch) -> None:
    """A feature table built before Elo existed must fail loudly."""
    frame = pd.DataFrame({
        "League": ["E0"], "Season": ["2627"], "Date": [pd.Timestamp("2026-09-01")],
        "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"],
    })
    monkeypatch.setattr(api.state, "features", frame)
    with pytest.raises(api.HTTPException) as caught:
        _call()
    assert caught.value.status_code == 503
