"""Publishing results when Understat cannot be reached.

The rule this bends is a real one -- `build_dataset` stops rather than publish
a partial refresh, because a half-updated table looks exactly like a complete
one. These tests pin the distinction that makes the exception safe: the
*results* are complete, and what is missing is an optional column family worth
-0.00010 RPS [-0.00045, +0.00023] to the model.

What must never happen is the failure mode that would be worse than the outage:
a run that reaches the fallback and quietly wipes twelve seasons of xG.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.ingest.understat import XG_COLUMNS, merge_xg, xg_from_published


def published(rows: list[dict]) -> pd.DataFrame:
    """A `team_match_history_all.csv` as the pipeline writes one."""
    frame = pd.DataFrame(rows)
    for column in XG_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame["GoalsFor"] = 1
    frame["GoalsAgainst"] = 0
    return frame


def match(team: str, opponent: str, date: str, xg: float | None, season: str = "2627") -> dict:
    return {
        "League": "E0",
        "Season": season,
        "Date": date,
        "Team": team,
        "Opponent": opponent,
        "Venue": "Home",
        "xGFor": xg,
        "xGAgainst": None if xg is None else xg / 2,
        "npxGFor": xg,
        "npxGAgainst": None if xg is None else xg / 2,
        "PPDAFor": None if xg is None else 9.0,
        "DeepCompletionsFor": None if xg is None else 5.0,
    }


def test_the_published_xg_comes_back_in_the_fetchs_shape(tmp_path) -> None:
    """So `merge_xg` needs no special case for it."""
    path = tmp_path / "team_match_history_all.csv"
    published([match("Arsenal", "Chelsea", "2026-08-15", 1.8)]).to_csv(path, index=False)

    carried = xg_from_published(path)

    assert list(carried.columns) == XG_COLUMNS
    assert float(carried["xGFor"].iloc[0]) == 1.8


def test_rows_that_never_had_xg_are_not_carried(tmp_path) -> None:
    """They would take up the one-to-one join holding nothing."""
    path = tmp_path / "team_match_history_all.csv"
    published(
        [
            match("Arsenal", "Chelsea", "2026-08-15", 1.8),
            match("Leeds", "Burnley", "2026-08-16", None),
        ]
    ).to_csv(path, index=False)

    assert len(xg_from_published(path)) == 1


def test_a_first_ever_build_still_works(tmp_path) -> None:
    carried = xg_from_published(tmp_path / "nothing-here.csv")

    assert carried.empty
    assert list(carried.columns) == XG_COLUMNS


def test_a_history_predating_the_xg_columns_is_not_a_crash(tmp_path) -> None:
    path = tmp_path / "team_match_history_all.csv"
    pd.DataFrame(
        {"League": ["E0"], "Season": ["1112"], "Team": ["Arsenal"], "Date": ["2011-08-15"]}
    ).to_csv(path, index=False)

    assert xg_from_published(path).empty


def test_old_matches_keep_their_xg_and_new_ones_come_back_empty(tmp_path) -> None:
    """The whole point: an outage costs the matches played during it, no more."""
    path = tmp_path / "team_match_history_all.csv"
    published([match("Arsenal", "Chelsea", "2026-08-15", 1.8)]).to_csv(path, index=False)

    # The fresh results carry a match played since, which Understat never served.
    fresh = pd.DataFrame(
        [
            {"League": "E0", "Season": "2627", "Date": pd.Timestamp("2026-08-15"),
             "Team": "Arsenal", "Opponent": "Chelsea", "Venue": "Home"},
            {"League": "E0", "Season": "2627", "Date": pd.Timestamp("2026-11-01"),
             "Team": "Arsenal", "Opponent": "Everton", "Venue": "Home"},
        ]
    )

    merged = merge_xg(fresh, xg_from_published(path)).set_index("Opponent")

    assert float(merged.loc["Chelsea", "xGFor"]) == 1.8
    assert pd.isna(merged.loc["Everton", "xGFor"])
    # And the results themselves are all there, which is the thing that was
    # being thrown away.
    assert len(merged) == 2


def test_carrying_forward_never_drops_a_result(tmp_path) -> None:
    """The failure that would be worse than the outage it works around."""
    path = tmp_path / "team_match_history_all.csv"
    published(
        [match("Arsenal", "Chelsea", "2026-08-15", 1.8, season=s) for s in ("2425", "2526")]
    ).to_csv(path, index=False)

    fresh = pd.DataFrame(
        [
            {"League": "E0", "Season": season, "Date": pd.Timestamp("2026-08-15"),
             "Team": "Arsenal", "Opponent": "Chelsea", "Venue": "Home"}
            for season in ("2425", "2526", "2627")
        ]
    )

    merged = merge_xg(fresh, xg_from_published(path))

    assert len(merged) == 3
    assert int(merged["xGFor"].notna().sum()) == 2


def test_the_join_stays_one_to_one(tmp_path) -> None:
    """`merge_xg` validates it, so a duplicated key would raise rather than fan out."""
    path = tmp_path / "team_match_history_all.csv"
    published(
        [
            match("Arsenal", "Chelsea", "2026-08-15", 1.8),
            match("Arsenal", "Chelsea", "2026-08-15", 2.1),
        ]
    ).to_csv(path, index=False)

    fresh = pd.DataFrame(
        [{"League": "E0", "Season": "2627", "Date": pd.Timestamp("2026-08-15"),
          "Team": "Arsenal", "Opponent": "Chelsea", "Venue": "Home"}]
    )

    with pytest.raises(pd.errors.MergeError):
        merge_xg(fresh, xg_from_published(path))
