"""Refetching one season must not delete the other ten.

`fetch_european` ended with ``kept.to_csv(MATCHES_OUT)``, which made
``--start-season`` a destructive flag dressed as a narrowing one: asking for
2026/27 alone rewrote the file with 2026/27 alone. On 2026-10-14 that run was
started and stopped seconds before it wrote, which would have discarded ten
seasons of European results.

The loss would not have been obvious. Nothing reads `european_matches.csv` on
the site; `fit_league_adjustment` reads it to fit the cross-league correction,
and it would simply have refitted on a fraction of the ties, producing a
plausible number from almost no data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipelines.fetch_european import _read, merge_seasons


def _matches(season: str, teams: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["UCL"] * len(teams),
        "Season": [season] * len(teams),
        "Date": [f"20{season[:2]}-10-{10 + i:02d}" for i in range(len(teams))],
        "HomeTeam": [h for h, _ in teams],
        "AwayTeam": [a for _, a in teams],
        "FTHG": [1] * len(teams),
        "FTAG": [0] * len(teams),
    })


# --- the property the flag needs ----------------------------------------------

def test_seasons_not_fetched_are_kept() -> None:
    """The whole point: a narrow refetch is narrow."""
    existing = pd.concat([
        _matches("2425", [("Arsenal", "Bayern Munich")]),
        _matches("2526", [("Real Madrid", "Inter")]),
        _matches("2627", [("Napoli", "Arsenal")]),
    ], ignore_index=True)
    fetched = _matches("2627", [("Napoli", "Arsenal"), ("Roma", "Real Madrid")])

    out = merge_seasons(existing, fetched, ["2627"])
    assert sorted(out["Season"].unique()) == ["2425", "2526", "2627"]
    assert len(out) == 4


def test_the_fetched_season_is_replaced_wholesale() -> None:
    """A refetch is authoritative for the seasons it covers.

    Row-by-row merging would be wrong: a tie dropped from the source, or
    recognised as a duplicate, has to disappear from our copy too.
    """
    existing = _matches("2627", [("Napoli", "Arsenal"), ("Stale", "Fixture")])
    fetched = _matches("2627", [("Napoli", "Arsenal")])

    out = merge_seasons(existing, fetched, ["2627"])
    assert len(out) == 1
    assert "Stale" not in set(out["HomeTeam"])


def test_a_full_refetch_replaces_everything() -> None:
    """The weekly job asks for every season, so nothing should survive it."""
    existing = pd.concat([
        _matches("2526", [("Old", "Row")]),
        _matches("2627", [("Another", "Old")]),
    ], ignore_index=True)
    fetched = _matches("2627", [("Napoli", "Arsenal")])

    out = merge_seasons(existing, fetched, ["2526", "2627"])
    assert len(out) == 1
    assert set(out["HomeTeam"]) == {"Napoli"}


def test_season_labels_compare_as_strings() -> None:
    """A CSV round trip can make `Season` an int; the filter must still bite."""
    existing = _matches("2627", [("Stale", "Fixture")]).assign(Season=2627)
    fetched = _matches("2627", [("Napoli", "Arsenal")])

    out = merge_seasons(existing, fetched, ["2627"])
    assert len(out) == 1
    assert set(out["HomeTeam"]) == {"Napoli"}


# --- not losing data on the unhappy paths -------------------------------------

def test_an_empty_fetch_keeps_what_is_there() -> None:
    """A scrape that came back with nothing is not a reason to delete history."""
    existing = _matches("2526", [("Real Madrid", "Inter")])
    out = merge_seasons(existing, pd.DataFrame(), ["2627"])
    pd.testing.assert_frame_equal(out, existing)


def test_a_first_run_writes_what_it_fetched() -> None:
    fetched = _matches("2627", [("Napoli", "Arsenal")])
    pd.testing.assert_frame_equal(merge_seasons(pd.DataFrame(), fetched, ["2627"]), fetched)


def test_a_missing_file_reads_as_empty(tmp_path) -> None:
    assert _read(tmp_path / "nope.csv").empty


def test_an_unreadable_file_refuses_rather_than_overwrites(tmp_path) -> None:
    """Failing closed, deliberately, and the opposite of the cache's choice.

    A cache that cannot be read costs a refetch. This file may hold seasons
    this run is not fetching, and overwriting it would destroy exactly what
    the merge exists to protect -- so an unreadable file stops the run.
    """
    path = tmp_path / "european_matches.csv"
    # `pd.read_csv` raises on very little -- handed this it returns a cheerful
    # one-column frame -- so the schema is what has to be checked, not the parse.
    path.write_text("not,a,european,file\n1,2,3,4\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Not overwriting"):
        _read(path)


def test_a_file_with_the_right_schema_is_accepted(tmp_path) -> None:
    path = tmp_path / "european_matches.csv"
    _matches("2627", [("Napoli", "Arsenal")]).to_csv(path, index=False)
    assert len(_read(path)) == 1


# --- ordering -----------------------------------------------------------------

def test_the_merged_file_is_ordered_by_season_and_date() -> None:
    """So a human opening it sees a timeline rather than two stacked blocks."""
    existing = _matches("2425", [("Arsenal", "Bayern Munich")])
    fetched = _matches("2627", [("Napoli", "Arsenal")])

    out = merge_seasons(existing, fetched, ["2627"])
    assert list(out["Season"]) == sorted(out["Season"])
