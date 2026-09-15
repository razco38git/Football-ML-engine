"""Leak-safe rolling form primitives.

Every function here answers one question: *what did we know about this team
before this match kicked off?*

The ``.shift(1)`` inside :func:`prev_window` is what enforces that. It drops the
current row out of the window before any aggregation happens. Delete it and the
match result leaks into the features used to predict that same match -- the
model would score brilliantly in backtests and fall over in production.

Features are computed on the *long* team-match shape (one row per team per
match, as in ``data/processed/team_match_history.csv``) rather than the wide
match shape, because rolling windows are natural per-team and awkward per-match.
:func:`footballml.features.build.build_match_features` pivots back to wide.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

#: Points awarded per result code.
POINTS = {"W": 3, "D": 1, "L": 0}

#: Metrics rolled into form features, mapping each source column to its
#: ``(aggregation, feature_name)``. Columns absent from the input are skipped, so
#: this table can name xG fields that exist only after Understat ingestion.
#:
#: Feature names are written out explicitly rather than derived from the column
#: name. Deriving them looks tidier but breaks on consecutive capitals -- an
#: automatic converter turns ``xGFor`` into ``x_gfor``, which then fails to match
#: anything downstream and silently drops the feature instead of raising.
FORM_METRICS: dict[str, tuple[str, str]] = {
    "Points": ("sum", "points"),
    "Win": ("sum", "wins"),
    "Draw": ("sum", "draws"),
    "Loss": ("sum", "losses"),
    "GoalsFor": ("mean", "goals_for"),
    "GoalsAgainst": ("mean", "goals_against"),
    "GoalDiff": ("mean", "goal_diff"),
    "ShotsFor": ("mean", "shots_for"),
    "ShotsAgainst": ("mean", "shots_against"),
    "ShotsOnTargetFor": ("mean", "shots_on_target_for"),
    "ShotsOnTargetAgainst": ("mean", "shots_on_target_against"),
    "ShotAccuracy": ("mean", "shot_accuracy"),
    # Understat only.
    "xGFor": ("mean", "xg_for"),
    "xGAgainst": ("mean", "xg_against"),
    "xGDiff": ("mean", "xg_diff"),
    "xGOverperformance": ("mean", "xg_overperformance"),
    "xGPerShot": ("mean", "xg_per_shot"),
    "npxGFor": ("mean", "npxg_for"),
    "npxGAgainst": ("mean", "npxg_against"),
    "npxGDiff": ("mean", "npxg_diff"),
    "PPDAFor": ("mean", "ppda"),
    "DeepCompletionsFor": ("mean", "deep_completions"),
}


def prev_window(
    df: pd.DataFrame,
    keys: Sequence[str],
    cols: Sequence[str],
    window: int,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Aggregate ``cols`` over the previous ``window`` rows within each group.

    The current row is always excluded. Groups with fewer than ``window`` prior
    rows aggregate over whatever history exists (``min_periods=1``); rows with no
    history at all come back as NaN, which callers should treat as "unknown"
    rather than filling with zero -- a promoted side's first fixture genuinely
    has no prior form, and zero would read as "terrible form" to the model.

    Args:
        df: Long team-match frame, **must already be sorted by date**.
        keys: Grouping columns, e.g. ``["Team"]`` or ``["Team", "Venue"]``.
        cols: Numeric columns to aggregate.
        window: Number of prior matches in the window.
        aggfunc: Aggregation applied over the window.

    Returns:
        Frame indexed like ``df`` with one column per entry in ``cols``.
    """
    keys = list(keys)
    cols = list(cols)
    if not cols:
        return pd.DataFrame(index=df.index)

    grouped = df.groupby(keys, sort=False, observed=True)[cols]
    shifted = grouped.shift(1)

    # Regroup the shifted values on the same keys so the rolling window never
    # spans a team (or venue) boundary.
    group_series = [df[k] for k in keys]
    rolled = (
        shifted.groupby(group_series, sort=False, observed=True)
        .rolling(window, min_periods=1)
        .agg(aggfunc)
    )
    # `.rolling` on a groupby prepends the group keys to the index; drop them so
    # the result realigns with the original frame.
    rolled = rolled.reset_index(level=list(range(len(keys))), drop=True)
    return rolled.reindex(df.index)


def prepare_team_match(tmh: pd.DataFrame) -> pd.DataFrame:
    """Derive the per-match quantities that form features roll over.

    Adds points, result indicators, goal difference, shot accuracy and -- when
    xG columns are present -- xG difference, overperformance and xG per shot.
    """
    df = tmh.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["Points"] = df["Result"].map(POINTS).astype("float64")
    df["Win"] = (df["Result"] == "W").astype("float64")
    df["Draw"] = (df["Result"] == "D").astype("float64")
    df["Loss"] = (df["Result"] == "L").astype("float64")
    df["GoalDiff"] = df["GoalsFor"] - df["GoalsAgainst"]

    # Guard against 0-shot matches rather than emitting inf.
    df["ShotAccuracy"] = (
        df["ShotsOnTargetFor"].div(df["ShotsFor"]).where(df["ShotsFor"] > 0)
    )

    if "xGFor" in df.columns and "xGAgainst" in df.columns:
        df["xGDiff"] = df["xGFor"] - df["xGAgainst"]
        # Positive means the team scored more than the chances warranted, which
        # historically regresses toward zero -- one of the strongest signals
        # available for spotting a team whose recent form flatters them.
        # Compared against total xG, since goals include penalties too.
        df["xGOverperformance"] = df["GoalsFor"] - df["xGFor"]
        df["xGPerShot"] = df["xGFor"].div(df["ShotsFor"]).where(df["ShotsFor"] > 0)

    if "npxGFor" in df.columns and "npxGAgainst" in df.columns:
        df["npxGDiff"] = df["npxGFor"] - df["npxGAgainst"]

    return df.sort_values(["Date", "Team"]).reset_index(drop=True)


def add_team_form(
    tmh: pd.DataFrame,
    window: int = 5,
    venue_split: bool = False,
    suffix: str | None = None,
) -> pd.DataFrame:
    """Attach rolling form columns to a prepared long team-match frame.

    Args:
        tmh: Output of :func:`prepare_team_match`.
        window: Matches in the rolling window.
        venue_split: When True, group by ``(Team, Venue)`` so a home fixture sees
            only previous *home* matches. This is the "last 5 at the next venue"
            signal -- home advantage varies enormously by club, and a side's
            overall form can badly misrepresent how they play at home.
        suffix: Column suffix. Defaults to ``_last_{window}`` or
            ``_last_{window}_venue`` when ``venue_split`` is set.

    Returns:
        ``tmh`` with added form columns and a ``matches_used`` count.
    """
    keys = ["Team", "Venue"] if venue_split else ["Team"]
    if suffix is None:
        suffix = f"_last_{window}_venue" if venue_split else f"_last_{window}"

    df = tmh.copy()
    available = {c: spec for c, spec in FORM_METRICS.items() if c in df.columns}

    out_cols: dict[str, pd.Series] = {}
    # Group metrics by aggregation so each distinct agg is one vectorised pass.
    for agg in sorted({spec[0] for spec in available.values()}):
        cols = [c for c, spec in available.items() if spec[0] == agg]
        rolled = prev_window(df, keys, cols, window, agg)
        for c in cols:
            out_cols[f"{available[c][1]}{suffix}"] = rolled[c]

    # How much history actually backed these numbers. Lets the model discount
    # early-season form and lets the UI say "based on 2 matches".
    counted = prev_window(df, keys, ["Points"], window, "count")
    out_cols[f"matches_used{suffix}"] = counted["Points"]

    return df.assign(**out_cols)


def add_rest_features(tmh: pd.DataFrame, congestion_days: int = 14) -> pd.DataFrame:
    """Add days since the team's previous match and recent fixture congestion."""
    df = tmh.copy()
    prev_date = df.groupby("Team", sort=False, observed=True)["Date"].shift(1)
    df["days_since_last_match"] = (df["Date"] - prev_date).dt.days

    # Matches played in the trailing window, excluding this one. Counted by
    # comparing each team's match dates against a lookback cutoff.
    def _congestion(g: pd.DataFrame) -> pd.Series:
        dates = g["Date"]
        cutoff = dates - pd.Timedelta(days=congestion_days)
        counts = [
            int(((dates < d) & (dates >= c)).sum())
            for d, c in zip(dates, cutoff, strict=True)
        ]
        return pd.Series(counts, index=g.index, dtype="float64")

    df[f"matches_last_{congestion_days}_days"] = (
        df.groupby("Team", sort=False, observed=True, group_keys=False)
        .apply(_congestion, include_groups=False)
        .reindex(df.index)
    )
    return df
