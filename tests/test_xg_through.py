"""How far the shot data reaches, and why a match card needs to be told.

An absent retrodiction has two very different causes that look identical on
screen. Usually the match was played last night and Understat has not got to it
yet, so the honest advice is to wait a day. But on 2026-11-01 understat.com
began refusing connections while still resolving normally, `build_dataset`
carried forward the xG already on disk, and every card from that day on went on
saying the data "usually lands a day or two after the result" -- true of the
mechanism, and three days later still telling people to wait for something that
was not coming.

`xg_through` is the fact that separates them: results run on to today while the
shot data stays put on the last day the source answered.
"""

from __future__ import annotations

import importlib
from datetime import date

import pandas as pd

api = importlib.import_module("footballml.api.app")


def history(rows: list[tuple[str, str, str, float | None]]) -> pd.DataFrame:
    """A team-match history, home rows only -- what `_match_xg_lookup` reads."""
    return pd.DataFrame(
        [
            {
                "Venue": "Home",
                "Date": pd.Timestamp(day),
                "Team": home,
                "Opponent": away,
                "xGFor": xg,
                "xGAgainst": None if xg is None else xg / 2,
            }
            for day, home, away, xg in rows
        ]
    )


def through(rows: list[tuple[str, str, str, float | None]]) -> date | None:
    return api._xg_through(api._match_xg_lookup(history(rows)))


def test_it_reports_the_newest_day_with_shot_data() -> None:
    assert through(
        [
            ("2026-10-24", "Arsenal", "Chelsea", 1.8),
            ("2026-10-31", "Leeds", "Burnley", 0.9),
        ]
    ) == date(2026, 10, 31)


def test_a_newer_match_without_xg_does_not_move_it() -> None:
    """The whole point: the results advanced, the shot data did not."""
    assert through(
        [
            ("2026-10-31", "Arsenal", "Chelsea", 1.8),
            ("2026-11-01", "Leeds", "Burnley", None),
            ("2026-11-02", "Everton", "Fulham", None),
        ]
    ) == date(2026, 10, 31)


def test_a_first_ever_build_reports_nothing_rather_than_a_date() -> None:
    """No xG at all is an empty database, not an outage."""
    assert through([("2026-11-01", "Leeds", "Burnley", None)]) is None
    assert api._xg_through({}) is None


def test_a_history_without_the_xg_columns_is_not_a_crash() -> None:
    bare = pd.DataFrame({"Venue": ["Home"], "Team": ["Leeds"], "Date": ["2026-11-01"]})
    assert api._xg_through(api._match_xg_lookup(bare)) is None


def test_health_publishes_it(monkeypatch) -> None:
    """A caller reading only `/health` should not have to infer this."""
    monkeypatch.setattr(api.state, "xg_through", date(2026, 10, 31))
    monkeypatch.setattr(api, "_require_model", lambda: (None, _metadata()), raising=True)
    monkeypatch.setattr(api.state, "league_offsets_stale", False)

    assert api.health().xg_through == date(2026, 10, 31)


def test_health_reports_nothing_as_null_not_as_a_guess(monkeypatch) -> None:
    monkeypatch.setattr(api.state, "xg_through", None)
    monkeypatch.setattr(api, "_require_model", lambda: (None, _metadata()), raising=True)
    monkeypatch.setattr(api.state, "league_offsets_stale", False)

    assert api.health().xg_through is None


def _metadata():
    from types import SimpleNamespace

    return SimpleNamespace(
        version="test",
        trained_through=pd.Timestamp("2026-11-02"),
        n_train=29345,
        rho=-0.05,
    )
