"""Pinning and fingerprinting an experiment's inputs.

The squad-size sweep produced two different tables from what looked like the
same inputs. The inputs were not the same: the season in progress is re-fetched
live on every run and grew by four player-seasons inside half an hour. These pin
the helper that stops that, and the one that makes it visible when it happens
anyway.
"""

from __future__ import annotations

import pandas as pd

from pipelines import vintage


def _players(*seasons: str, rows: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Season": s, "Player": f"p{i}", "League": "E0", "minutes": 900 + i}
            for s in seasons
            for i in range(rows)
        ]
    )


# --- identifying the season that is still moving ------------------------------


def test_a_season_past_the_cached_window_is_the_current_one() -> None:
    assert vintage.current_season(_players("2425", "2526", "2627")) == "2627"


def test_a_frame_of_finished_seasons_has_no_current_season() -> None:
    """So this is a no-op out of season, rather than something to remember to
    switch off."""
    assert vintage.current_season(_players("2425", "2526")) is None


def test_an_empty_frame_does_not_raise() -> None:
    assert vintage.current_season(pd.DataFrame()) is None


# --- pinning ------------------------------------------------------------------


def test_pinning_drops_only_the_in_progress_season() -> None:
    pinned, dropped = vintage.pin_finished_seasons(_players("2425", "2526", "2627"))
    assert dropped == "2627"
    assert set(pinned["Season"]) == {"2425", "2526"}


def test_pinning_a_finished_frame_changes_nothing() -> None:
    frame = _players("2425", "2526")
    pinned, dropped = vintage.pin_finished_seasons(frame)
    assert dropped is None
    pd.testing.assert_frame_equal(pinned, frame)


def test_pinning_is_idempotent() -> None:
    once, _ = vintage.pin_finished_seasons(_players("2526", "2627"))
    twice, dropped = vintage.pin_finished_seasons(once)
    assert dropped is None
    pd.testing.assert_frame_equal(once, twice)


def test_a_pinned_frame_is_unchanged_when_the_live_season_grows() -> None:
    """The property the whole module exists for.

    Two runs differing only in how much of the in-progress season has been
    played must present identical inputs to everything downstream.
    """
    monday, _ = vintage.pin_finished_seasons(_players("2526", "2627", rows=3))
    friday_frame = pd.concat(
        [_players("2526", "2627", rows=3), _players("2627", rows=7)]
    ).drop_duplicates()
    friday, dropped = vintage.pin_finished_seasons(friday_frame)
    assert dropped == "2627"
    assert vintage.fingerprint(monday) == vintage.fingerprint(friday)


# --- fingerprinting -----------------------------------------------------------


def test_the_fingerprint_ignores_row_order() -> None:
    """"Is this the same data", not "is this the same file"."""
    frame = _players("2425", "2526")
    assert vintage.fingerprint(frame) == vintage.fingerprint(frame.iloc[::-1])


def test_the_fingerprint_ignores_column_order() -> None:
    frame = _players("2425")
    assert vintage.fingerprint(frame) == vintage.fingerprint(
        frame[list(reversed(frame.columns))]
    )


def test_the_fingerprint_changes_when_a_value_changes() -> None:
    frame = _players("2425")
    other = frame.copy()
    other.loc[0, "minutes"] = 1
    assert vintage.fingerprint(frame) != vintage.fingerprint(other)


def test_the_fingerprint_changes_when_a_row_is_added() -> None:
    """Four extra player-seasons is exactly the perturbation that was missed."""
    frame = _players("2425", rows=3)
    bigger = pd.concat([frame, _players("2425", rows=4)]).drop_duplicates()
    assert vintage.fingerprint(frame) != vintage.fingerprint(bigger)


def test_describe_inputs_names_every_frame_and_its_hash() -> None:
    text = vintage.describe_inputs(players=_players("2425"), strength=_players("2526"))
    assert "players" in text and "strength" in text
    assert vintage.fingerprint(_players("2425")) in text


# --- isolation ----------------------------------------------------------------


def test_production_does_not_import_this() -> None:
    """It is a research-harness helper. If `src/footballml` ever depends on it,
    pinning stops being an experiment's choice and becomes model behaviour."""
    import ast
    import pathlib

    for path in pathlib.Path("src/footballml").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "vintage" not in (node.module or ""), path
            elif isinstance(node, ast.Import):
                assert not any("vintage" in a.name for a in node.names), path
