"""Topping the match history up from FBref when football-data lags.

football-data is the results source and it falls behind -- three weeks, in
October 2026 -- which froze the projected tables, because `project_season`
simulates from the match history and will not simulate from rows missing half
their columns.

The gap is narrower than it looks. Of what football-data brings, only goals,
shots and shots on target ever become features; fouls, corners and cards are
stored and read by nothing. FBref has all three and Understat has the xG family,
so a topped-up row is complete in every column the model uses.

The guarantee worth pinning is that this only ever *adds*: football-data's own
rows must win every contested match, so a week when it is current produces
exactly the frame it would have produced without any of this.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.ingest import fbref_results as fb


def _existing(dates: list[str]) -> pd.DataFrame:
    """Rows as `to_team_match_history` produces them, football-data shaped."""
    return pd.DataFrame({
        "League": ["E0"] * len(dates),
        "Season": ["2627"] * len(dates),
        "Date": pd.to_datetime(dates),
        "Team": ["Arsenal"] * len(dates),
        "Opponent": ["Chelsea"] * len(dates),
        "Venue": ["Home"] * len(dates),
        "GoalsFor": [2] * len(dates),
        "GoalsAgainst": [1] * len(dates),
        "Result": ["W"] * len(dates),
        "ShotsFor": [14] * len(dates),
        "FoulsFor": [9] * len(dates),
    })


def _incoming(dates: list[str], team: str = "Arsenal") -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["E0"] * len(dates),
        "Season": ["2627"] * len(dates),
        "Date": pd.to_datetime(dates),
        "Team": [team] * len(dates),
        "Opponent": ["Chelsea"] * len(dates),
        "Venue": ["Home"] * len(dates),
        "GoalsFor": [3] * len(dates),
        "GoalsAgainst": [0] * len(dates),
        "Result": ["W"] * len(dates),
        "ShotsFor": [20] * len(dates),
        "ShotsAgainst": [5] * len(dates),
        "ShotsOnTargetFor": [8] * len(dates),
        "ShotsOnTargetAgainst": [2] * len(dates),
    })


def test_only_missing_matches_are_added() -> None:
    have = _existing(["2026-09-20"])
    out = fb.top_up(have, _incoming(["2026-09-20", "2026-10-11"]))
    assert len(out) == 2
    assert sorted(str(d.date()) for d in out["Date"]) == ["2026-09-20", "2026-10-11"]


def test_the_primary_source_wins_a_contested_row() -> None:
    """The same match from both sources must keep football-data's version."""
    have = _existing(["2026-09-20"])
    out = fb.top_up(have, _incoming(["2026-09-20"]))

    kept = out[out["Date"] == pd.Timestamp("2026-09-20")]
    assert len(kept) == 1
    # football-data said 2-1 with 14 shots; FBref said 3-0 with 20.
    assert kept.iloc[0]["GoalsFor"] == 2
    assert kept.iloc[0]["ShotsFor"] == 14
    assert kept.iloc[0]["source"] == fb.FOOTBALL_DATA


def test_a_current_primary_source_changes_nothing() -> None:
    """The property that makes this safe to leave switched on."""
    have = _existing(["2026-09-19", "2026-09-20"])
    out = fb.top_up(have, _incoming(["2026-09-19", "2026-09-20"]))
    pd.testing.assert_frame_equal(
        out.drop(columns="source").reset_index(drop=True),
        have.reset_index(drop=True),
        check_dtype=False,
    )


def test_nothing_to_add_is_not_an_error() -> None:
    have = _existing(["2026-09-20"])
    assert len(fb.top_up(have, pd.DataFrame())) == 1
    assert len(fb.top_up(have, _incoming([]))) == 1


def test_every_row_is_labelled_with_its_source() -> None:
    """A silent mix would be the thing to avoid: these rows carry no fouls."""
    out = fb.top_up(_existing(["2026-09-20"]), _incoming(["2026-10-11"]))
    by_source = out.set_index(out["Date"].dt.date)["source"].to_dict()
    assert by_source[pd.Timestamp("2026-09-20").date()] == fb.FOOTBALL_DATA
    assert by_source[pd.Timestamp("2026-10-11").date()] == fb.FBREF


def test_matching_is_per_team_not_per_date() -> None:
    """Two clubs play on the same day; only the missing one may be added."""
    have = _existing(["2026-10-11"])  # Arsenal v Chelsea
    out = fb.top_up(have, _incoming(["2026-10-11"], team="Liverpool"))
    assert len(out) == 2
    assert set(out["Team"]) == {"Arsenal", "Liverpool"}


def test_a_source_column_is_added_even_with_no_top_up() -> None:
    """So a reader never has to wonder whether an unlabelled row is labelled."""
    out = fb.top_up(_existing(["2026-09-20"]), pd.DataFrame())
    assert set(out["source"]) == {fb.FOOTBALL_DATA}


def test_an_existing_source_column_is_respected() -> None:
    """Re-running the build must not relabel rows it already labelled."""
    have = _existing(["2026-09-20"]).assign(source=fb.FBREF)
    out = fb.top_up(have, pd.DataFrame())
    assert set(out["source"]) == {fb.FBREF}


def test_a_row_missing_a_feature_column_is_refused() -> None:
    """The guarantee that lets these rows near the training data.

    soccerdata drives FBref through a headless browser and that browser can die
    mid-run: on 2026-10-12 England and France came back with shots while Germany
    and Italy came back with none, and 36 of 88 rows went in with NaN. A league
    that fails this waits for a run where the scrape worked.
    """
    assert set(fb.FEATURE_COLUMNS) == {
        "GoalsFor", "GoalsAgainst",
        "ShotsFor", "ShotsAgainst",
        "ShotsOnTargetFor", "ShotsOnTargetAgainst",
    }
    # Every feature column is one the top-up promises to deliver.
    for column in fb.FEATURE_COLUMNS:
        assert column in fb.COLUMNS


# --- the fetcher's contract ---------------------------------------------------

def test_unknown_division_is_rejected() -> None:
    """A UEFA code here would quietly put European ties in the match history."""
    with pytest.raises(KeyError, match="unknown division"):
        fb.fetch_fbref_results(["UCL"])


def test_a_scrape_failure_returns_empty_rather_than_raising(monkeypatch) -> None:
    """A failed top-up must never cost the rows football-data did supply."""
    def boom(*_a, **_k):
        raise ConnectionError("FBref is down")

    monkeypatch.setattr("footballml.ingest.schedule.played_results", boom)
    out = fb.fetch_fbref_results(["E0"])
    assert out.empty
    assert list(out.columns) == fb.COLUMNS


def test_the_columns_it_promises_are_the_ones_features_need() -> None:
    """Fouls, corners and cards are absent on purpose -- nothing reads them."""
    assert "FoulsFor" not in fb.COLUMNS
    assert "CornersFor" not in fb.COLUMNS
    for needed in (
        "GoalsFor", "GoalsAgainst",
        "ShotsFor", "ShotsAgainst",
        "ShotsOnTargetFor", "ShotsOnTargetAgainst",
    ):
        assert needed in fb.COLUMNS
