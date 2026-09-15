"""Understat xG ingestion.

Understat covers exactly the five leagues this project targets, back to 2014/15,
and is the only free source of shot-quality data at this coverage. It supplies
the piece football-data.co.uk fundamentally cannot: how good the chances were,
not just how many shots were taken.

Beyond xG it carries two metrics worth having:

- **npxG** -- non-penalty xG. The better predictor of future performance, since
  penalties are rare, converted at a stable rate, and not a repeatable team
  skill. Total xG is kept too, because goal *overperformance* must compare like
  with like against total goals.
- **PPDA** -- opponent passes allowed per defensive action. A pressing-intensity
  proxy; lower means more aggressive pressing.

.. warning::
    Season labels must be passed unambiguously as ``"2122"``, not ``"2021"``.
    soccerdata reads a bare ``"2021"`` as the *2020-21* season, so requesting
    ``range(2014, 2025)`` silently collapses two inputs onto one season and drops
    2021-22 without error.
"""

from __future__ import annotations

import logging

import pandas as pd

from footballml.entities.teams import resolve_teams

logger = logging.getLogger(__name__)

UNDERSTAT_LEAGUES = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "F1": "FRA-Ligue 1",
}

#: Understat's earliest season.
FIRST_SEASON = "1415"

#: Per-side columns, mapped to the ``For``/``Against`` long-format names.
_SIDE_METRICS = {
    "xg": "xG",
    "np_xg": "npxG",
    "ppda": "PPDA",
    "deep_completions": "DeepCompletions",
}


def season_labels(start: str = FIRST_SEASON, end: str | None = None) -> list[str]:
    """Unambiguous four-digit season labels from ``start`` to ``end`` inclusive.

    Defaults to ending at whichever season is currently underway, so the
    pipeline stays current without an annual code change.
    """
    from footballml.ingest.matchhistory import current_season_label

    end = end or current_season_label()
    first, last = int(start[:2]), int(end[:2])
    return [f"{y:02d}{y + 1:02d}" for y in range(first, last + 1)]


def fetch_understat_team_match(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    known_teams: set[str] | None = None,
) -> pd.DataFrame:
    """Fetch Understat team-match stats in the long one-row-per-team-match shape.

    Args:
        leagues: Division codes such as ``["E0", "SP1"]``. Defaults to all five.
        seasons: Four-digit season labels. Defaults to everything available.
        known_teams: Canonical names to validate the alias mapping against.

    Returns:
        Columns ``League``, ``Season``, ``Date``, ``Team``, ``Opponent``,
        ``Venue`` plus ``xGFor``/``xGAgainst`` and the other metrics, for both
        sides of every match.
    """
    import soccerdata as sd  # imported lazily: pulls a heavy browser stack

    codes = leagues or sorted(UNDERSTAT_LEAGUES)
    seasons = seasons or season_labels()
    raw = (
        sd.Understat(leagues=[UNDERSTAT_LEAGUES[c] for c in codes], seasons=seasons)
        .read_team_match_stats()
        .reset_index()
    )

    missing = set(seasons) - set(raw["season"].unique())
    if missing:
        # Loud, because a silently absent season leaves a hole in the training
        # data that no downstream check would catch.
        logger.warning("Understat returned no data for seasons: %s", sorted(missing))

    home = _side_frame(raw, "home", "away", "Home")
    away = _side_frame(raw, "away", "home", "Away")
    long_df = pd.concat([home, away], ignore_index=True)

    long_df = resolve_teams(
        long_df, ["Team", "Opponent"], source="understat", known=known_teams
    )
    return long_df.sort_values(["Date", "League", "Team"]).reset_index(drop=True)


def _side_frame(raw: pd.DataFrame, side: str, other: str, venue: str) -> pd.DataFrame:
    """Extract one side of each match into the long format."""
    inverse = {v: k for k, v in UNDERSTAT_LEAGUES.items()}
    out = pd.DataFrame(
        {
            "League": raw["league"].map(inverse),
            "Season": raw["season"].astype(str),
            "Date": pd.to_datetime(raw["date"]).dt.normalize(),
            "Team": raw[f"{side}_team"],
            "Opponent": raw[f"{other}_team"],
            "Venue": venue,
        }
    )
    for col, name in _SIDE_METRICS.items():
        out[f"{name}For"] = pd.to_numeric(raw[f"{side}_{col}"], errors="coerce")
        out[f"{name}Against"] = pd.to_numeric(raw[f"{other}_{col}"], errors="coerce")
    # PPDA and deep completions describe the team itself, not something conceded,
    # so the For/Against framing would be misleading. Keep only the team's own.
    return out.drop(columns=["PPDAAgainst", "DeepCompletionsAgainst"], errors="ignore")


def merge_xg(
    tmh: pd.DataFrame, xg: pd.DataFrame, validate_dates: bool = True
) -> pd.DataFrame:
    """Attach Understat metrics to the football-data team-match history.

    Joined on ``(League, Season, Team, Opponent, Venue)`` rather than on date. In
    a double round-robin league that key is unique, and it is immune to the
    one-day disagreements that kick-off timezones produce between sources.

    Args:
        tmh: Long team-match history from football-data.co.uk.
        xg: Output of :func:`fetch_understat_team_match`.
        validate_dates: Warn when matched rows disagree on date by more than a
            day, which would indicate the key matched the wrong fixture.
    """
    key = ["League", "Season", "Team", "Opponent", "Venue"]
    left = tmh.copy()
    left["Season"] = left["Season"].astype(str)

    right = xg.rename(columns={"Date": "Date_xg"})
    merged = left.merge(right, on=key, how="left", validate="1:1")

    matched = merged["Date_xg"].notna()
    if validate_dates and matched.any():
        drift = (merged.loc[matched, "Date"] - merged.loc[matched, "Date_xg"]).abs()
        off = drift > pd.Timedelta(days=1)
        if off.any():
            logger.warning(
                "%d matched rows disagree on date by more than a day", int(off.sum())
            )

    coverage = matched.mean()
    logger.info("xG coverage: %.1f%% of %d team-matches", coverage * 100, len(merged))
    return merged.drop(columns=["Date_xg"])
