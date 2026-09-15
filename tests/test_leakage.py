"""The most important tests in the repo.

A leaky feature pipeline produces a model that looks excellent in backtests and
is worthless in production. These tests catch that by construction rather than
by inspection.

The central idea: a feature for match M must depend only on matches *before* M.
So computing it over the full dataset and computing it over a dataset truncated
just before M must give identical answers. If they differ, the future leaked in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.data import load_team_match_history
from footballml.features.build import build_match_features
from footballml.features.rolling import add_team_form, prepare_team_match


@pytest.fixture(scope="module")
def tmh() -> pd.DataFrame:
    """Two recent seasons -- enough to exercise the windows, fast enough to iterate."""
    df = load_team_match_history()
    return df[df["Season"].isin([2324, 2425])].copy()


@pytest.fixture(scope="module")
def features(tmh: pd.DataFrame) -> pd.DataFrame:
    return build_match_features(tmh)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    ignore = {"Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    return [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]


def test_truncation_invariance(tmh: pd.DataFrame, features: pd.DataFrame) -> None:
    """Features must not change when future matches are removed from the input.

    This is the definitive leakage check. We pick a cutoff, rebuild features
    using only data up to that cutoff, and require the overlapping rows to match
    the full-data build exactly.
    """
    cutoff = pd.Timestamp("2025-03-01")
    truncated = build_match_features(tmh[tmh["Date"] < cutoff].copy())

    key = ["Date", "HomeTeam", "AwayTeam"]
    full_subset = features[features["Date"] < cutoff].sort_values(key).reset_index(drop=True)
    trunc_subset = truncated.sort_values(key).reset_index(drop=True)

    assert len(full_subset) == len(trunc_subset), "truncation changed the number of matches"

    for col in _feature_cols(full_subset):
        pd.testing.assert_series_equal(
            full_subset[col],
            trunc_subset[col],
            check_names=False,
            obj=f"feature {col!r} changed when future data was removed -- leakage",
        )


def test_form_excludes_current_match(tmh: pd.DataFrame) -> None:
    """A team's rolling form must never include the match it is attached to.

    Checked directly: the ``points_last_5`` on a team's Nth match must equal the
    points they actually took in matches N-5..N-1.
    """
    prepared = prepare_team_match(tmh)
    rolled = add_team_form(prepared, window=5, venue_split=False)

    team = rolled["Team"].iloc[0]
    hist = rolled[rolled["Team"] == team].sort_values("Date").reset_index(drop=True)

    for idx in range(6, min(len(hist), 20)):
        expected = hist.loc[idx - 5 : idx - 1, "Points"].sum()
        actual = hist.loc[idx, "points_last_5"]
        assert actual == pytest.approx(expected), (
            f"{team} match {idx}: points_last_5={actual} but prior 5 matches "
            f"totalled {expected} -- window is misaligned"
        )


def test_venue_form_uses_only_same_venue(tmh: pd.DataFrame) -> None:
    """Venue-split form must draw only on prior matches at that same venue."""
    prepared = prepare_team_match(tmh)
    rolled = add_team_form(prepared, window=5, venue_split=True)

    team = rolled["Team"].iloc[0]
    home = (
        rolled[(rolled["Team"] == team) & (rolled["Venue"] == "Home")]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    for idx in range(6, min(len(home), 15)):
        expected = home.loc[idx - 5 : idx - 1, "Points"].sum()
        actual = home.loc[idx, "points_last_5_venue"]
        assert actual == pytest.approx(expected), (
            f"{team} home match {idx}: venue form drew on non-home matches"
        )


def test_first_match_has_no_form(tmh: pd.DataFrame) -> None:
    """A team's very first match in the dataset must have undefined form, not zero.

    Zero would read to the model as 'lost every one of their last five', which is
    a real and very different claim from 'we have no idea'.
    """
    prepared = prepare_team_match(tmh)
    rolled = add_team_form(prepared, window=5, venue_split=False)

    firsts = rolled.groupby("Team", sort=False).head(1)
    assert firsts["points_last_5"].isna().all(), (
        "first-ever match should have NaN form, not a filled value"
    )


def test_no_target_columns_in_features(features: pd.DataFrame) -> None:
    """Nothing describing the current match may appear as a feature."""
    banned = {"Points", "Win", "Draw", "Loss", "GoalDiff", "ShotAccuracy", "Result"}
    for side in ("home_", "away_"):
        leaked = {c for c in features.columns if c.removeprefix(side) in banned}
        assert not leaked, f"current-match outcome columns leaked into features: {leaked}"


def test_features_are_finite(features: pd.DataFrame) -> None:
    """Guard against inf from division by zero (0-shot matches, etc.)."""
    numeric = features[_feature_cols(features)]
    infinite = numeric.columns[np.isinf(numeric.to_numpy(dtype="float64")).any(axis=0)]
    assert len(infinite) == 0, f"infinite values in {list(infinite)}"
