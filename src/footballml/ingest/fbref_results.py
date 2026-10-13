"""FBref match results, for the weeks football-data has not published.

football-data is the results source and it falls behind: on 2026-10-12 its files
still ended 2026-09-20 while three rounds had been played. That froze the
projected tables, because `project_season` simulates from the match history and
will not simulate from rows that are missing half their columns.

The gap turns out to be smaller than it looks. Of the columns football-data
brings, only **goals, shots and shots on target** ever reach the model -- fouls,
corners and cards are stored and never read by anything outside the code that
writes them. FBref publishes all three on the same pages the schedule already
comes from, and Understat supplies xG, npxG, PPDA and deep completions
separately and stays current.

So a row built here is *complete in every column the model uses*. It is not an
approximation standing in for one, which is the only reason it is allowed near
the training data:

    goals, shots, shots on target   here
    xG, npxG, PPDA, deep            Understat, merged by `build_dataset`
    fouls, corners, cards           absent, and never a feature

Strictly a top-up. football-data remains the primary source and its own rows
always win -- see `top_up` -- so this changes nothing on a week when it is
current, and the mix is recorded in a `source` column rather than left to be
inferred.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footballml.entities import load_aliases
from footballml.ingest.matchhistory import current_season_label
from footballml.ingest.understat import UNDERSTAT_LEAGUES

logger = logging.getLogger(__name__)

#: Marks where a row came from. football-data rows carry `FOOTBALL_DATA`.
FBREF = "fbref"
FOOTBALL_DATA = "football-data"

#: Identifies one side's half of one match.
KEY = ["League", "Season", "Date", "Team", "Opponent"]

#: The columns here that become model features. Every one must be present on a
#: row before it is allowed into the match history -- see `fetch_fbref_results`.
#: The xG family is merged separately by `build_dataset` and checked there.
FEATURE_COLUMNS = [
    "GoalsFor", "GoalsAgainst",
    "ShotsFor", "ShotsAgainst",
    "ShotsOnTargetFor", "ShotsOnTargetAgainst",
]

#: What this module can fill. Everything else football-data brings is either
#: supplied by Understat or never used as a feature.
COLUMNS = [
    *KEY, "Venue", "GoalsFor", "GoalsAgainst", "Result",
    "ShotsFor", "ShotsAgainst", "ShotsOnTargetFor", "ShotsOnTargetAgainst",
]


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop FBref's column MultiIndex, keeping the leaf name.

    `read_team_match_stats` returns ``('Standard', 'Sh')`` beside a plain
    ``('date', '')``. Taking the leaf where there is one and the head otherwise
    gives `Sh`, `SoT` and `date` without a lookup table.
    """
    frame = frame.copy()
    frame.columns = [
        (c[1] if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c))
        for c in frame.columns
    ]
    return frame.loc[:, ~frame.columns.duplicated()]


def _shooting(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    """Shots and shots on target, keyed by date, venue and *opponent*.

    Not by team, because soccerdata's `read_team_match_stats` leaves the team
    name unparsed: every one of the 118 Premier League rows for 2026/27 came
    back with ``team`` as NaN, and even the game label read
    ``2026-08-21 Arsenal-nan``. `opponent` and `venue` survive intact, which is
    enough -- a row is then "whoever played *against* Y, at home or away", and
    the schedule says who that was.
    """
    import soccerdata as sd

    reader = sd.FBref(leagues=[UNDERSTAT_LEAGUES[c] for c in leagues], seasons=list(seasons))
    raw = _flatten(reader.read_team_match_stats(stat_type="shooting").reset_index())

    needed = {"league", "season", "date", "venue", "opponent", "Sh", "SoT"}
    missing = needed - set(raw.columns)
    if missing:
        raise KeyError(f"FBref shooting table is missing {sorted(missing)}")

    aliases = load_aliases("fbref")
    inverse = {v: k for k, v in UNDERSTAT_LEAGUES.items()}
    return pd.DataFrame(
        {
            # League and Season are part of the key, not decoration. Without
            # them the join below matched across divisions: two clubs can share
            # an opponent name and a kick-off date in different leagues, and
            # Dortmund v Werder Bremen came back three times.
            "League": raw["league"].map(inverse),
            "Season": raw["season"].astype(str),
            "Date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "Venue": raw["venue"].astype(str).str.title(),
            "Against": raw["opponent"].astype(str).str.strip().replace(aliases),
            "Shots": pd.to_numeric(raw["Sh"], errors="coerce"),
            "ShotsOnTarget": pd.to_numeric(raw["SoT"], errors="coerce"),
        }
    ).dropna(subset=["League", "Date", "Against"])


def fetch_fbref_results(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    known: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Played matches in the long team-match shape, from FBref.

    Args:
        leagues: Division codes, e.g. ``["E0"]``. Defaults to the five domestic
            divisions; UEFA codes are not accepted, since European ties are
            deliberately kept out of the match history.
        seasons: Season labels. Defaults to the current one.
        known: The match history as it already stands. **Pass this.** It is what
            keeps the common case cheap -- see below.

    Returns:
        :data:`COLUMNS`, one row per team per played match, or an empty frame.
        Empty rather than raising: this is a top-up, and a failed top-up must
        never cost the rows the primary source did supply.

    The expensive call here is the shooting table: soccerdata fetches one page
    per team and refuses its own cache for the current season, so a five-league
    request is about 98 pages at seven seconds each -- twelve minutes, every
    run, on a job that otherwise takes twenty seconds.

    Almost every day that work is wasted, because football-data is current and
    there is nothing to top up. So the schedule decides first: it is five pages,
    cached for twelve hours, and already fetched elsewhere in the same run. If
    it shows no played match that ``known`` is missing, this returns immediately
    and the shooting table is never touched. Only leagues that are actually
    behind get scraped.
    """
    codes = list(leagues or UNDERSTAT_LEAGUES)
    unknown = [c for c in codes if c not in UNDERSTAT_LEAGUES]
    if unknown:
        raise KeyError(f"unknown division code(s): {unknown}")
    seasons = list(seasons or [current_season_label(pd.Timestamp.today())])

    try:
        from footballml.ingest.schedule import played_results

        results = played_results(codes, seasons)
        if results.empty:
            return pd.DataFrame(columns=COLUMNS)

        results = results.copy()
        results["Date"] = pd.to_datetime(results["Date"])
        results["Season"] = results["Season"].astype(str)

        behind = _leagues_behind(results, known)
        if not behind:
            logger.info(
                "The match history has every played match the schedule knows "
                "about; skipping the shooting scrape"
            )
            return pd.DataFrame(columns=COLUMNS)

        results = results[results["League"].isin(behind)]
        logger.info("Behind in %s; fetching shooting stats for those only", sorted(behind))
        shots = _shooting(sorted(behind), seasons)
    except Exception as exc:  # noqa: BLE001 - a top-up must not break the build
        logger.warning("Could not read FBref results (%s)", exc)
        return pd.DataFrame(columns=COLUMNS)

    # The home side's row is the one played at home against the away side, and
    # vice versa. This is what recovers the identity `team` lost.
    on_home = ["League", "Season", "Date", "AwayTeam"]
    on_away = ["League", "Season", "Date", "HomeTeam"]
    home = shots[shots["Venue"] == "Home"].rename(
        columns={"Against": "AwayTeam", "Shots": "HS", "ShotsOnTarget": "HST"}
    )[[*on_home, "HS", "HST"]].drop_duplicates(on_home)
    away = shots[shots["Venue"] == "Away"].rename(
        columns={"Against": "HomeTeam", "Shots": "AS", "ShotsOnTarget": "AST"}
    )[[*on_away, "AS", "AST"]].drop_duplicates(on_away)

    results["Season"] = results["Season"].astype(str)
    wide = (
        results.merge(home, on=on_home, how="left", validate="one_to_one")
        .merge(away, on=on_away, how="left", validate="one_to_one")
    )
    resolved = wide[["HS", "AS"]].notna().all(axis=1).sum()
    logger.info(
        "FBref results: %d played match(es) to %s, shots resolved for %d",
        len(wide), wide["Date"].max().date(), resolved,
    )

    sides = []
    for venue, (gf, ga, sf, sa, stf, sta, team, opp) in {
        "Home": ("FTHG", "FTAG", "HS", "AS", "HST", "AST", "HomeTeam", "AwayTeam"),
        "Away": ("FTAG", "FTHG", "AS", "HS", "AST", "HST", "AwayTeam", "HomeTeam"),
    }.items():
        sides.append(
            pd.DataFrame(
                {
                    "League": wide["League"], "Season": wide["Season"].astype(str),
                    "Date": wide["Date"], "Team": wide[team], "Opponent": wide[opp],
                    "Venue": venue,
                    "GoalsFor": wide[gf], "GoalsAgainst": wide[ga],
                    "ShotsFor": wide[sf], "ShotsAgainst": wide[sa],
                    "ShotsOnTargetFor": wide[stf], "ShotsOnTargetAgainst": wide[sta],
                }
            )
        )

    out = pd.concat(sides, ignore_index=True)
    out["Result"] = np.where(
        out["GoalsFor"] > out["GoalsAgainst"], "W",
        np.where(out["GoalsFor"] < out["GoalsAgainst"], "L", "D"),
    )

    # A row missing a feature column does not go in. The whole argument for
    # letting these near the training data is that they are *complete* in
    # everything the model reads -- a half-filled row would be the quiet
    # degradation this is supposed to avoid, and it would be invisible once
    # written.
    #
    # It is not hypothetical. soccerdata drives FBref through a headless
    # browser, and on 2026-10-12 that browser died partway through: England,
    # France and most of Spain came back with shots, Germany and Italy came
    # back with none, and 36 of 88 rows went in with NaN. A league that fails
    # this check simply waits for a run where the scrape worked.
    complete = out[FEATURE_COLUMNS].notna().all(axis=1)
    if not complete.all():
        dropped = out[~complete]
        logger.warning(
            "Dropping %d of %d team-match row(s) missing a feature column "
            "(league(s) %s). They will be picked up by a later run; a partial "
            "row must not reach the model.",
            len(dropped), len(out), sorted(dropped["League"].dropna().unique()),
        )
    return out.loc[complete, COLUMNS].reset_index(drop=True)


def _leagues_behind(results: pd.DataFrame, known: pd.DataFrame | None) -> set[str]:
    """Divisions where the schedule knows a played match the history does not.

    Cheap by design: it compares the schedule -- five cached pages -- against
    rows already on disk, and decides whether the twelve-minute scrape is worth
    starting. With ``known`` missing there is nothing to compare, so everything
    is treated as behind rather than silently skipping a real gap.
    """
    if known is None or known.empty:
        return set(results["League"].dropna().unique())

    have = known.assign(
        Date=pd.to_datetime(known["Date"]), Season=known["Season"].astype(str)
    )
    # The history is one row per team; a match is present if either side is.
    seen = set(
        zip(have["League"], have["Season"], have["Date"], have["Team"], strict=True)
    )
    behind = {
        row.League
        for row in results.itertuples()
        if (row.League, row.Season, row.Date, row.HomeTeam) not in seen
    }
    return behind


def top_up(tmh: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Add FBref rows for matches the primary source has not published.

    football-data wins every contested row. This only ever *adds*, so a week
    when it is current produces exactly the frame it would have produced
    without this module -- which is the property that makes the top-up safe to
    leave switched on.

    Both frames gain a ``source`` column. A silent mix would be the thing to
    avoid here: these rows carry no fouls, corners or cards, and while nothing
    reads those today, a reader of the file deserves to know which rows have
    them.
    """
    if "source" not in tmh.columns:
        tmh = tmh.assign(source=FOOTBALL_DATA)
    if extra.empty:
        return tmh

    extra = extra.copy()
    extra["Date"] = pd.to_datetime(extra["Date"])
    extra["Season"] = extra["Season"].astype(str)
    have = tmh.assign(Date=pd.to_datetime(tmh["Date"]), Season=tmh["Season"].astype(str))

    fresh = extra.merge(have[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
    fresh = fresh[fresh["_merge"] == "left_only"].drop(columns="_merge")
    if fresh.empty:
        logger.info("FBref has no team-match rows the primary source is missing")
        return tmh

    logger.info(
        "Topping up %d team-match row(s) from FBref (to %s, primary source ends %s)",
        len(fresh), fresh["Date"].max().date(), have["Date"].max().date(),
    )
    return pd.concat([have, fresh.assign(source=FBREF)], ignore_index=True)
