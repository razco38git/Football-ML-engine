"""Assemble the match-level, pre-kickoff feature table.

Form is computed on the long team-match shape (see
:mod:`footballml.features.rolling`), then pivoted here into one row per match
with ``home_``/``away_`` prefixed columns plus a handful of differences.

Differences matter more than you'd expect: tree models can in principle learn
``home_x - away_x`` themselves, but handing it over directly consistently helps,
because the *relative* strength of the two sides is what actually decides a
football match.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from footballml.features.rolling import (
    add_rest_features,
    add_team_form,
    prepare_team_match,
)

#: Feature stems differenced into ``*_diff`` columns when present on both sides.
DIFF_STEMS: tuple[str, ...] = (
    "points_last_5",
    "points_last_5_venue",
    "goal_diff_last_5",
    "goal_diff_last_5_venue",
    "goals_for_last_5",
    "goals_against_last_5",
    "shots_on_target_for_last_5",
    "shot_accuracy_last_5",
    "xg_for_last_5",
    "xg_against_last_5",
    "xg_diff_last_5",
    "xg_diff_last_5_venue",
    "xg_overperformance_last_5",
    "npxg_diff_last_5",
    "ppda_last_5",
    "days_since_last_match",
)

#: Columns identifying a match rather than describing it.
_ID_COLS = ("League", "Season", "Date", "HomeTeam", "AwayTeam")

#: Stable numeric codes for the league feature. Scoring rates and home advantage
#: genuinely differ between leagues -- the Bundesliga is higher scoring than
#: Serie A -- so the model is told which competition it is looking at. Codes are
#: fixed rather than derived from the data so that a model trained on four
#: leagues still interprets the feature correctly when scoring a fifth.
LEAGUE_CODES = {"E0": 0, "SP1": 1, "D1": 2, "I1": 3, "F1": 4}


def build_team_features(
    tmh: pd.DataFrame,
    windows: Sequence[int] = (5,),
    congestion_days: int = 14,
) -> pd.DataFrame:
    """Run the full long-shape feature build: prepare, roll, venue-split, rest."""
    df = prepare_team_match(tmh)
    for window in windows:
        df = add_team_form(df, window=window, venue_split=False)
        df = add_team_form(df, window=window, venue_split=True)
    return add_rest_features(df, congestion_days=congestion_days)


def build_match_features(
    tmh: pd.DataFrame,
    windows: Sequence[int] = (5,),
    congestion_days: int = 14,
) -> pd.DataFrame:
    """Build the wide, model-ready match feature table from long team-match rows.

    Args:
        tmh: Raw long team-match history (one row per team per match, with a
            ``Venue`` column of ``"Home"``/``"Away"``).
        windows: Rolling window sizes to compute form over.
        congestion_days: Lookback for the fixture-congestion count.

    Returns:
        One row per match: identifiers, the actual result (``FTHG``, ``FTAG``,
        ``FTR``) as targets, and ``home_*``/``away_*``/``*_diff`` features that
        are all strictly knowable before kickoff.
    """
    if "League" not in tmh.columns:
        # Single-league inputs (the original EPL-only files) predate the League
        # column. Default rather than fail, so older data still builds.
        tmh = tmh.assign(League="E0")

    long_df = build_team_features(tmh, windows=windows, congestion_days=congestion_days)

    feature_cols = _feature_columns(long_df, tmh)

    home = long_df[long_df["Venue"] == "Home"].copy()
    away = long_df[long_df["Venue"] == "Away"].copy()

    home = home.rename(columns={"Team": "HomeTeam", "Opponent": "AwayTeam"})
    away = away.rename(columns={"Team": "AwayTeam", "Opponent": "HomeTeam"})

    # Targets come from the home row, where GoalsFor/Against are already
    # oriented home-first.
    targets = home[[*_ID_COLS, "GoalsFor", "GoalsAgainst"]].rename(
        columns={"GoalsFor": "FTHG", "GoalsAgainst": "FTAG"}
    )
    # Unplayed fixtures have null goals and must keep a null result. Without the
    # final `.where`, NaN comparisons are all False and every upcoming match
    # would be silently labelled an away win.
    played = targets["FTHG"].notna() & targets["FTAG"].notna()
    targets["FTR"] = (
        pd.Series("D", index=targets.index)
        .where(targets["FTHG"] == targets["FTAG"])
        .fillna(pd.Series("H", index=targets.index).where(targets["FTHG"] > targets["FTAG"]))
        .fillna("A")
        .where(played)
    )

    home_feats = home[[*_ID_COLS, *feature_cols]].add_prefix("home_")
    away_feats = away[[*_ID_COLS, *feature_cols]].add_prefix("away_")
    home_feats = home_feats.rename(columns={f"home_{c}": c for c in _ID_COLS})
    away_feats = away_feats.rename(columns={f"away_{c}": c for c in _ID_COLS})

    matches = targets.merge(home_feats, on=list(_ID_COLS), how="inner", validate="1:1")
    matches = matches.merge(away_feats, on=list(_ID_COLS), how="inner", validate="1:1")

    matches = _add_diffs(matches)
    matches["league_code"] = matches["League"].map(LEAGUE_CODES).astype("float64")
    return matches.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)


def build_upcoming_features(
    tmh: pd.DataFrame, fixtures: pd.DataFrame, **kwargs: object
) -> pd.DataFrame:
    """Attach pre-kickoff features to fixtures that have not been played.

    The trick is that no separate code path is needed. Each fixture is appended
    to the history as a pair of team-match rows with every statistic missing,
    then the normal builder runs over the combined frame. Because rolling form
    only ever looks *backwards*, those placeholder rows pick up each side's real
    recent form while contributing nothing themselves -- which is precisely the
    leak-safe behaviour the tests already guarantee.

    Args:
        tmh: Long team-match history of played matches.
        fixtures: ``League``, ``Season``, ``Date``, ``HomeTeam``, ``AwayTeam``.
        **kwargs: Passed through to :func:`build_match_features`.

    Returns:
        One row per fixture with features populated and ``FTHG``/``FTAG`` null.
    """
    if fixtures.empty:
        return pd.DataFrame()

    placeholder = pd.concat(
        [
            _fixture_side(fixtures, "HomeTeam", "AwayTeam", "Home"),
            _fixture_side(fixtures, "AwayTeam", "HomeTeam", "Away"),
        ],
        ignore_index=True,
    )
    # Match the history's columns so the concat lines up. Missing statistics must
    # be float NaN rather than pd.NA: a pd.NA leaves the column as object dtype,
    # and the arithmetic in prepare_team_match then fails on it.
    for col in tmh.columns:
        if col not in placeholder.columns:
            placeholder[col] = (
                np.nan if pd.api.types.is_numeric_dtype(tmh[col]) else None
            )

    combined = pd.concat([tmh, placeholder[tmh.columns]], ignore_index=True)
    for col in tmh.columns:
        if pd.api.types.is_numeric_dtype(tmh[col]):
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    built = build_match_features(combined, **kwargs)  # type: ignore[arg-type]

    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    wanted = fixtures[key].assign(Date=pd.to_datetime(fixtures["Date"]))
    return built.merge(wanted, on=key, how="inner").reset_index(drop=True)


def _fixture_side(
    fixtures: pd.DataFrame, own: str, opp: str, venue: str
) -> pd.DataFrame:
    """One side of each unplayed fixture, with no match statistics."""
    return pd.DataFrame(
        {
            "League": fixtures["League"],
            "Season": fixtures["Season"].astype(str),
            "Date": pd.to_datetime(fixtures["Date"]),
            "Team": fixtures[own],
            "Opponent": fixtures[opp],
            "Venue": venue,
            "Result": pd.NA,
        }
    )


def _feature_columns(long_df: pd.DataFrame, original: pd.DataFrame) -> list[str]:
    """Columns added by the feature build -- i.e. everything not in the input.

    Derived per-match quantities (``Points``, ``GoalDiff``, ...) are excluded:
    they describe the match being predicted, so carrying them forward would leak
    the result.
    """
    derived_in_match = {
        "Points", "Win", "Draw", "Loss", "GoalDiff", "ShotAccuracy",
        "xGDiff", "xGOverperformance", "xGPerShot", "npxGDiff",
    }
    original_cols = set(original.columns) | derived_in_match
    return [c for c in long_df.columns if c not in original_cols]


def _add_diffs(matches: pd.DataFrame) -> pd.DataFrame:
    """Add ``{stem}_diff`` columns for every stem present on both sides."""
    diffs = {
        f"{stem}_diff": matches[f"home_{stem}"] - matches[f"away_{stem}"]
        for stem in DIFF_STEMS
        if f"home_{stem}" in matches.columns and f"away_{stem}" in matches.columns
    }
    return matches.assign(**diffs)
