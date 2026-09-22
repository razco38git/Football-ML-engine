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

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from footballml.entities import load_aliases
from footballml.features.rolling import (
    add_rest_features,
    add_team_form,
    prepare_team_match,
)

logger = logging.getLogger(__name__)

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
    "strength_overall",
    "strength_attack",
    "strength_defence",
    "strength_goalkeeper",
)

#: Squad-strength columns joined onto each match.
#:
#: The goalkeeper is carried separately rather than folded into the defence
#: line, because he is the best single defensive signal we have: across 1,052
#: completed team-seasons his rating tracks goals conceded at -0.62, against
#: -0.52 for the whole back line. Blending him into `strength_defence` would
#: also double-count him, since he already contributes to `strength_overall`.
STRENGTH_COLUMNS = (
    "strength_overall",
    "strength_attack",
    "strength_defence",
    "strength_goalkeeper",
)

#: Columns identifying a match rather than describing it.
_ID_COLS = ("League", "Season", "Date", "HomeTeam", "AwayTeam")

#: Stable numeric codes for the league feature. Scoring rates and home advantage
#: genuinely differ between leagues -- the Bundesliga is higher scoring than
#: Serie A -- so the model is told which competition it is looking at. Codes are
#: fixed rather than derived from the data so that a model trained on four
#: leagues still interprets the feature correctly when scoring a fifth.
LEAGUE_CODES = {"E0": 0, "SP1": 1, "D1": 2, "I1": 3, "F1": 4}


def add_team_strength(
    matches: pd.DataFrame,
    strength: pd.DataFrame,
    previous_season: bool = True,
) -> pd.DataFrame:
    """Attach squad strength to each match, from the player ratings.

    Three numbers per side -- overall, attack and defence -- which map onto the
    model's two goal rates directly: a side's attack is set against the
    opposition's defence.

    Args:
        matches: Wide match table with ``League``, ``Season`` and both teams.
        strength: ``team_strength.csv``, one row per team-season.
        previous_season: Join the *preceding* season's strength rather than the
            match's own. This is the leakage guard and the default.

    Returns:
        ``matches`` with ``home_``/``away_`` strength columns. Teams with no
        strength row get NaN, which the gradient booster handles natively.

    .. warning::
        Strength is computed from **whole-season** player ratings, so joining a
        match to its own season lets end-of-season information predict an
        October fixture. That leak is invisible to the truncation test in
        ``tests/test_leakage.py``: it truncates *match* data, and strength comes
        from a separate file that would not change.

        ``previous_season=True`` is therefore correct for training and
        backtesting -- and for live prediction too, for two reasons that each
        suffice on their own:

        - **Current-season strength does not exist when it is needed.** Ratings
          require a minimum number of minutes, so four weeks into 2026/27 not
          one player was rated and ``team_strength.csv`` had no rows for it.
          ``False`` would hand every live fixture all-NaN strength, silently
          identical to not using the feature.
        - **Train/serve consistency.** The model learns "this match, given last
          season's squad". Serving it this season's squad is a different
          quantity, and skew even where the data exists.

        ``False`` remains for analysis only. A promoted side has no
        previous-season row in its new league and gets NaN; the model was
        trained with exactly those gaps, so it degrades rather than breaks.
    """
    out = matches.copy()
    available = [c for c in STRENGTH_COLUMNS if c in strength.columns]
    if strength.empty or not available:
        return out

    # Match on a temporary string key rather than casting `Season` in place.
    # Callers compare seasons numerically (`run_backtest` filters `s >= start`),
    # so silently turning the column into strings breaks them well downstream of
    # here, with an error that points nowhere near this function.
    lookup = strength[["League", "Season", "Team", *available]].copy()

    # Strength is built from player data, which carries Understat's club names
    # ("Borussia Dortmund", "Atletico Madrid"); matches carry football-data's
    # ("Dortmund", "Ath Madrid"). Without translating, the join fails silently --
    # a left merge just leaves NaN -- and before this line it reached only 40% of
    # Bundesliga and 52% of La Liga team-seasons, missing Dortmund, Leverkusen,
    # Leipzig and both Madrid clubs. Resolved here rather than in the CSV so the
    # Team Strength page keeps the fuller display names. Canonical names are not
    # alias keys, so this is safe on an already-resolved frame.
    lookup["Team"] = lookup["Team"].replace(load_aliases("understat"))
    lookup["_season_key"] = lookup["Season"].astype(str)
    lookup = lookup.drop(columns=["Season"])
    out["_season_key"] = out["Season"].astype(str)

    if previous_season:
        # Shift the strength forward a season so a 2425 squad rating is what a
        # 2526 match sees. Season labels are "2425"-style, so the successor of
        # season YYZZ is ZZ(ZZ+1).
        lookup["_season_key"] = lookup["_season_key"].map(_next_season)

    for side in ("Home", "Away"):
        prefixed = lookup.rename(
            columns={"Team": f"{side}Team", **{c: f"{side.lower()}_{c}" for c in available}}
        )
        out = out.merge(prefixed, on=["League", "_season_key", f"{side}Team"], how="left")

    return out.drop(columns=["_season_key"])


def _next_season(label: str) -> str:
    """``"2425"`` -> ``"2526"``. Returns the input unchanged if unparseable."""
    text = str(label)
    if len(text) != 4 or not text.isdigit():
        return text
    start = int(text[:2]) + 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


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
    strength: pd.DataFrame | None = None,
    previous_season_strength: bool = True,
) -> pd.DataFrame:
    """Build the wide, model-ready match feature table from long team-match rows.

    Args:
        tmh: Raw long team-match history (one row per team per match, with a
            ``Venue`` column of ``"Home"``/``"Away"``).
        windows: Rolling window sizes to compute form over.
        congestion_days: Lookback for the fixture-congestion count.
        strength: Optional ``team_strength.csv`` to join squad quality from.
        previous_season_strength: Join the preceding season's strength. Leave
            True for training and backtesting; pass False only for live
            prediction, where the current squad is the right one.

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

    if strength is not None:
        matches = add_team_strength(
            matches, strength, previous_season=previous_season_strength
        )

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

    # Drop fixtures the history already contains. football-data's fixture list
    # keeps publishing a round after it has been played, so once results are
    # refreshed every "upcoming" fixture can already be in `tmh` -- appending a
    # placeholder for one then duplicates a real match, and the 1:1 merge in
    # `build_match_features` fails with a MergeError that points nowhere near
    # here. Predicting a played match through this path is meaningless anyway;
    # `/matches` scores those retrospectively.
    played = tmh[tmh["Venue"] == "Home"][["League", "Date", "Team", "Opponent"]]
    played = played.rename(columns={"Team": "HomeTeam", "Opponent": "AwayTeam"})
    played = played.assign(Date=pd.to_datetime(played["Date"]), _played=True)

    key = ["League", "Date", "HomeTeam", "AwayTeam"]
    fixtures = fixtures.assign(Date=pd.to_datetime(fixtures["Date"]))
    marked = fixtures.merge(played.drop_duplicates(key), on=key, how="left")
    already = marked["_played"].notna()
    if already.any():
        logger.info(
            "Skipping %d of %d fixture(s) already played and in the history",
            int(already.sum()), len(fixtures),
        )
        fixtures = fixtures[~already.to_numpy()]
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
