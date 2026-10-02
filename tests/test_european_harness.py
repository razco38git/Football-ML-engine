"""The path that scores a European tie must behave like the ordinary one.

``pipelines/validate_european.py`` scores a cross-league match by handing it to
:func:`build_upcoming_features` as a fixture, one matchday at a time, against
the domestic history. Two properties make that measurement trustworthy, and
both are easy to lose silently:

1. **Nothing from on or after the match date may reach its features.** A leak
   here would make our cross-league predictions look good for the worst reason.
2. **The features must be the ones the model was trained on.** If scoring a
   match as a *fixture* differs from scoring it as *history*, then the European
   numbers are not comparable to any domestic number and the whole exercise
   measures nothing.

Both are checked here against real domestic matches, where the answer is known,
rather than against European ones, where it is not.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.data import PROCESSED_DIR, load_team_match_history
from footballml.features.build import build_match_features, build_upcoming_features


@pytest.fixture(scope="module")
def tmh() -> pd.DataFrame:
    """Two seasons: enough history for the 19-match window to be populated.

    The multi-league file, not the default: the legacy EPL-only one has no
    `League` column, which `build_upcoming_features` needs.
    """
    df = load_team_match_history(PROCESSED_DIR / "team_match_history_all.csv")
    return df[df["Season"].astype(str).isin(["2324", "2425"])].copy()


@pytest.fixture(scope="module")
def target(tmh: pd.DataFrame) -> pd.Series:
    """One real match, late enough that both sides have a full form history."""
    late = tmh[(tmh["Venue"] == "Home") & (tmh["Date"] > pd.Timestamp("2025-02-01"))]
    return late.sort_values(["Date", "Team"]).iloc[0]


def _fixture_frame(row: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "League": row.get("League", "E0"),
                "Season": str(row["Season"]),
                "Date": pd.Timestamp(row["Date"]),
                "HomeTeam": row["Team"],
                "AwayTeam": row["Opponent"],
            }
        ]
    )


def _shared_feature_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    ignore = {"League", "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    return [
        c
        for c in a.columns
        if c in b.columns and c not in ignore and pd.api.types.is_numeric_dtype(a[c])
    ]


def _without(tmh: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Drop both rows of one match, so it can be re-added as a fixture."""
    pair = {row["Team"], row["Opponent"]}
    drop = (tmh["Date"] == row["Date"]) & tmh["Team"].isin(pair) & tmh["Opponent"].isin(pair)
    return tmh[~drop].copy()


def test_scoring_a_match_as_a_fixture_matches_scoring_it_as_history(
    tmh: pd.DataFrame, target: pd.Series
) -> None:
    """The equivalence test.

    Remove a real match from the history, hand it back as a fixture, and the
    features must be exactly what the normal build produced for it. Removing it
    cannot change its own features -- they look only at earlier matches -- so
    any difference is the fixture path doing something the training path did
    not.
    """
    as_history = build_match_features(tmh)
    key = (
        (as_history["Date"] == target["Date"])
        & (as_history["HomeTeam"] == target["Team"])
        & (as_history["AwayTeam"] == target["Opponent"])
    )
    expected = as_history[key]
    assert len(expected) == 1, "test match not found in the ordinary build"

    as_fixture = build_upcoming_features(_without(tmh, target), _fixture_frame(target))
    assert len(as_fixture) == 1, "the fixture path produced no row"

    columns = _shared_feature_columns(expected, as_fixture)
    assert len(columns) > 50, f"only {len(columns)} shared feature columns; check the shapes"

    mismatched = [
        c
        for c in columns
        if not pd.Series(expected[c].to_numpy()).equals(pd.Series(as_fixture[c].to_numpy()))
    ]
    assert not mismatched, f"fixture path differs on {len(mismatched)}: {mismatched[:10]}"


def test_no_feature_sees_the_match_date_or_later(
    tmh: pd.DataFrame, target: pd.Series
) -> None:
    """The leak test.

    Truncating the history at the match date must leave the fixture's features
    untouched. If a single column moves, something downstream of the date is
    reaching the prediction.
    """
    history = _without(tmh, target)
    fixtures = _fixture_frame(target)

    full = build_upcoming_features(history, fixtures)
    truncated = build_upcoming_features(
        history[history["Date"] < target["Date"]].copy(), fixtures
    )
    assert len(full) == len(truncated) == 1

    for column in _shared_feature_columns(full, truncated):
        pd.testing.assert_series_equal(
            full[column].reset_index(drop=True),
            truncated[column].reset_index(drop=True),
            check_names=False,
            obj=f"{column} changed when later matches were removed",
        )


def test_a_cross_league_fixture_gets_strength_for_both_sides(tmh: pd.DataFrame) -> None:
    """The reason the whole exercise is possible.

    `add_team_strength` is keyed on (season, team) rather than league, so a
    fixture stamped with one side's division still finds the other side's
    rating. Without this a European tie would be scored with half its squad
    information missing -- which is exactly what `/predict` did until recently.

    Teams are taken from the match history, not the ratings table: the ratings
    carry Understat's spellings ("Manchester City") and the fixture needs the
    canonical ones ("Man City") that `add_team_strength` resolves to.
    """
    from footballml.data import load_team_strength

    strength = load_team_strength()
    recent = tmh[tmh["Season"].astype(str) == "2425"]
    english = recent[recent["League"] == "E0"]["Team"]
    spanish = recent[recent["League"] == "SP1"]["Team"]
    if english.empty or spanish.empty:
        pytest.skip("this slice does not hold both leagues")

    home, away = english.iloc[0], spanish.iloc[0]
    fixtures = pd.DataFrame(
        [
            {
                "League": "E0", "Season": "2425",
                "Date": pd.Timestamp("2025-02-19"),
                "HomeTeam": home, "AwayTeam": away,
            }
        ]
    )
    built = build_upcoming_features(tmh, fixtures, strength=strength)
    assert len(built) == 1, f"no features for {home} v {away}"

    assert built["home_strength_overall"].notna().all(), f"no rating for {home}"
    assert built["away_strength_overall"].notna().all(), (
        f"no rating for {away} -- the away side's rating went missing, so "
        "add_team_strength is keyed on league again"
    )
