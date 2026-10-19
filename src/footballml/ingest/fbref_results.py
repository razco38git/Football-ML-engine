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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from footballml.data import PROCESSED_DIR
from footballml.entities import load_aliases
from footballml.ingest.matchhistory import current_season_label
from footballml.ingest.understat import UNDERSTAT_LEAGUES

logger = logging.getLogger(__name__)

#: Marks where a row came from. football-data rows carry `FOOTBALL_DATA`.
FBREF = "fbref"
FOOTBALL_DATA = "football-data"

#: Identifies one side's half of one match.
KEY = ["League", "Season", "Date", "Team", "Opponent"]

#: Identifies a whole match, as the schedule presents it.
MATCH_KEY = ["League", "Season", "Date", "HomeTeam", "AwayTeam"]

#: Matches a working scrape could not complete, so the gate can stop paying for
#: them. Local state beside the other processed files, not a tracked artifact:
#: it is derived, and losing it costs one wasted scrape.
DEFERRED_PATH = PROCESSED_DIR / "fbref_deferred.csv"

DEFERRED_COLUMNS = [*MATCH_KEY, "attempts", "last_attempt"]

#: How long a parked match is left alone before it is tried again.
#:
#: The gate is otherwise memoryless, which is a real cost: on 2026-10-13 La Liga
#: was behind by exactly one match -- Vallecano 0-0 Ath Bilbao, 10 October --
#: that *neither* source could supply. football-data published nine of the
#: round's ten and skipped it; FBref's schedule page had the score, but the
#: shooting matchlogs for both clubs ended three weeks earlier, so there was no
#: shot line to merge and the completeness check refused the row. Nothing was
#: wrong, and nothing would change tomorrow -- yet the league counted as behind
#: every morning, and `build_dataset` paid 6m 43s for a row it would refuse
#: again.
#:
#: A week, rather than never, because FBref does backfill match reports: a
#: permanent skip would trade the daily cost for silently losing shots that
#: arrive late. Whichever run next finds the record stale pays for one retry, so
#: the worst case is one scrape a week per unfillable match instead of seven.
RETRY_AFTER = timedelta(days=7)

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


#: Competition names FBref uses in a club's **match logs**, which are not the
#: ones it uses on its competition index.
#:
#: soccerdata's league table knows only the index spelling, and
#: `read_team_match_stats` takes each row's league from the match log's `Comp`
#: column. `_translate_league` then maps anything it does not recognise to NaN
#: and the final `.loc[self.leagues]` drops it, so every German and Italian row
#: was discarded inside soccerdata before this module ever saw it. England,
#: France and Spain are spelled identically in both places and were unaffected,
#: which is why the loss looked like a scraper fault for a week.
#:
#: Measured on the cached pages for 2026/27: with this correction
#: `read_team_match_stats` returns 108 Bundesliga and 136 Serie A rows where it
#: previously returned none, and the other three leagues are unchanged.
#:
#: Overriding the *index* spelling instead does not work and is worse than
#: doing nothing: setting ``GER-Bundesliga`` to ``Bundesliga`` makes
#: `read_team_season_stats` find no German teams at all, so the match logs are
#: never requested either.
MATCHLOG_LEAGUE_ALIASES = {
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
}


def _matchlog_reader(leagues: list[str], seasons: list[str]):
    """An FBref reader that understands match-log competition names.

    Subclassed rather than patched globally: nothing else in the project reads
    match logs, and mutating soccerdata's module-level league table would reach
    every other reader in the process.
    """
    import soccerdata as sd

    class _FBrefWithMatchlogLeagues(sd.FBref):
        @classmethod
        def _all_leagues(cls) -> dict[str, str]:
            # soccerdata keys its league table on `cls.__name__`, so a subclass
            # of its own finds nothing and every league looks invalid. Defer.
            return sd.FBref._all_leagues()

        @classmethod
        def _translate_league(cls, df: pd.DataFrame, col: str = "league") -> pd.DataFrame:
            known = {v: k for k, v in cls._all_leagues().items()}
            known.update(MATCHLOG_LEAGUE_ALIASES)
            df.loc[~df[col].isin(known), col] = np.nan
            df[col] = df[col].replace(known)
            return df

    return _FBrefWithMatchlogLeagues(leagues=leagues, seasons=seasons)


def _shooting(leagues: list[str], seasons: list[str]) -> pd.DataFrame:
    """Shots and shots on target, keyed by date, venue and *opponent*.

    Not by team, because soccerdata's `read_team_match_stats` leaves the team
    name unparsed: every one of the 118 Premier League rows for 2026/27 came
    back with ``team`` as NaN, and even the game label read
    ``2026-08-21 Arsenal-nan``. `opponent` and `venue` survive intact, which is
    enough -- a row is then "whoever played *against* Y, at home or away", and
    the schedule says who that was.
    """
    reader = _matchlog_reader(
        [UNDERSTAT_LEAGUES[c] for c in leagues], list(seasons)
    )
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

    The gate also remembers. A match neither source can supply -- FBref has the
    score on the schedule page but no shooting line to merge -- would otherwise
    keep its division "behind" for good, and buy the same refusal every morning.
    Such a match is parked in :data:`DEFERRED_PATH` for :data:`RETRY_AFTER` and
    does not count towards being behind while it is, so the cost falls from
    every run to one a week. See :func:`_park_what_failed` for why a league that
    returns *nothing* is never parked.
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

        now = _today()
        seen = _seen(known)
        missing = _missing_matches(results, known)
        if missing.empty:
            logger.info(
                "The match history has every played match the schedule knows "
                "about; skipping the shooting scrape"
            )
            return pd.DataFrame(columns=COLUMNS)

        deferred = _load_deferred(seen=seen)
        eligible = missing[~_key_series(missing).isin(_parked(deferred, now))]
        if eligible.empty:
            logger.info(
                "All %d missing match(es) were already tried and could not be "
                "completed (%s); next attempt once a record is %d days old. "
                "Skipping the shooting scrape.",
                len(missing), _describe(missing), RETRY_AFTER.days,
            )
            return pd.DataFrame(columns=COLUMNS)

        behind = set(eligible["League"].dropna().unique())
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
    _park_what_failed(wide, eligible, deferred, now)

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
    # It is not hypothetical. On 2026-10-12 England, France and most of Spain
    # came back with shots while Germany and Italy came back with none, and 36
    # of 88 rows arrived with NaN. This guard is what kept them out.
    #
    # The cause was *not* the headless browser dying partway through, which is
    # what a comment here claimed until the fixtures were traced one by one. It
    # was soccerdata's league naming: FBref's match logs say "Bundesliga" and
    # "Serie A" where its league table says "GER-Bundesliga" and "ITA-Serie A",
    # so `_translate_league` set those two to NaN and the shot columns never
    # joined. Fixed in `db98e26`; a league that still fails this check waits for
    # a run where the scrape worked.
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


def _today() -> pd.Timestamp:
    """Date-granular "now". Its own function so a test can move it."""
    return pd.Timestamp(datetime.now(UTC).date())


def _key_series(frame: pd.DataFrame) -> pd.Series:
    """One hashable key per row, for membership tests against another frame."""
    if frame.empty:
        return pd.Series([], dtype=object, index=frame.index)
    return pd.Series(
        list(
            zip(
                frame["League"],
                frame["Season"].astype(str),
                pd.to_datetime(frame["Date"]),
                frame["HomeTeam"],
                frame["AwayTeam"],
                strict=True,
            )
        ),
        index=frame.index,
        dtype=object,
    )


def _keys(frame: pd.DataFrame) -> set[tuple]:
    return set(_key_series(frame))


def _seen(known: pd.DataFrame | None) -> set[tuple] | None:
    """``(league, season, date, team)`` for every match already in the history.

    ``None`` when there is nothing to compare, which callers read as "assume
    everything is missing" rather than silently skipping a real gap.
    """
    if known is None or known.empty:
        return None

    have = known.assign(
        Date=pd.to_datetime(known["Date"]), Season=known["Season"].astype(str)
    )
    # The history is one row per team; a match is present if either side is.
    return set(
        zip(have["League"], have["Season"], have["Date"], have["Team"], strict=True)
    )


def _missing_matches(results: pd.DataFrame, known: pd.DataFrame | None) -> pd.DataFrame:
    """Played matches the schedule knows about and the history does not.

    Cheap by design: it compares the schedule -- five cached pages -- against
    rows already on disk, and is what decides whether the twelve-minute scrape
    is worth starting.

    Returns the matches rather than just their divisions, because the divisions
    alone are not enough to decide. A league is "behind" for as long as one
    match is absent, and :func:`_load_deferred` needs to know *which* match that
    is to tell a gap worth scraping for from one that has already been tried.
    """
    seen = _seen(known)
    if seen is None:
        return results.copy()

    absent = pd.Series(
        [
            (row.League, row.Season, row.Date, row.HomeTeam) not in seen
            for row in results.itertuples()
        ],
        index=results.index,
        dtype=bool,
    )
    return results[absent].copy()


def _load_deferred(
    path: Path | None = None, seen: set[tuple] | None = None
) -> pd.DataFrame:
    """Matches a working scrape has already failed to complete.

    Pruned against the history on the way in: once a match arrives from either
    source its record is dead weight, and dropping it here is what keeps the
    file from growing a permanent row for every hole ever seen.

    Unreadable or malformed is treated as empty -- the cost of forgetting is one
    wasted scrape, and the cost of raising is a build that fails over a cache.
    """
    path = path or DEFERRED_PATH
    empty = pd.DataFrame(columns=DEFERRED_COLUMNS)
    if not path.exists():
        return empty

    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        logger.warning(
            "Could not read %s (%s); every missing match is eligible again",
            path.name, exc,
        )
        return empty

    absent = [c for c in DEFERRED_COLUMNS if c not in frame.columns]
    if absent:
        logger.warning(
            "%s is missing column(s) %s; every missing match is eligible again",
            path.name, absent,
        )
        return empty

    frame = frame.assign(
        Date=pd.to_datetime(frame["Date"], errors="coerce"),
        Season=frame["Season"].astype(str),
        last_attempt=pd.to_datetime(frame["last_attempt"], errors="coerce"),
        attempts=pd.to_numeric(frame["attempts"], errors="coerce"),
    ).dropna(subset=["Date", "last_attempt", "attempts"])

    if seen is not None and not frame.empty:
        arrived = pd.Series(
            [
                (row.League, row.Season, row.Date, row.HomeTeam) in seen
                for row in frame.itertuples()
            ],
            index=frame.index,
            dtype=bool,
        )
        if arrived.any():
            logger.info(
                "%d parked match(es) have since arrived from a source; forgetting them",
                int(arrived.sum()),
            )
        frame = frame[~arrived]

    return frame.reset_index(drop=True)


def _parked(deferred: pd.DataFrame, now: pd.Timestamp) -> set[tuple]:
    """Keys still inside their retry window, so not worth scraping for today."""
    if deferred.empty:
        return set()
    return _keys(deferred[deferred["last_attempt"] > now - RETRY_AFTER])


def _record_deferred(
    unfillable: pd.DataFrame,
    deferred: pd.DataFrame,
    path: Path | None = None,
    now: pd.Timestamp | None = None,
) -> None:
    """Park matches a working scrape could not complete, and rewrite the file.

    ``attempts`` is incremented rather than overwritten. It is not used to
    decide anything -- :func:`_parked` goes on the date alone -- but a count of
    four on a match from six weeks ago is how somebody reading the file learns
    that this is a permanent hole in both sources rather than a page FBref has
    not got round to yet.
    """
    path = path or DEFERRED_PATH
    now = now if now is not None else _today()

    record: dict[tuple, tuple[int, pd.Timestamp]] = {
        (row.League, row.Season, row.Date, row.HomeTeam, row.AwayTeam): (
            int(row.attempts), row.last_attempt,
        )
        for row in deferred.itertuples()
    }
    for key in _keys(unfillable):
        attempts, _ = record.get(key, (0, now))
        record[key] = (attempts + 1, now)

    out = pd.DataFrame(
        [
            {
                "League": league, "Season": season,
                "Date": pd.Timestamp(date).date().isoformat(),
                "HomeTeam": home, "AwayTeam": away, "attempts": attempts,
                "last_attempt": pd.Timestamp(last).date().isoformat(),
            }
            for (league, season, date, home, away), (attempts, last) in sorted(
                record.items(), key=lambda item: (item[0][0], str(item[0][2]))
            )
        ],
        columns=DEFERRED_COLUMNS,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _park_what_failed(
    wide: pd.DataFrame,
    eligible: pd.DataFrame,
    deferred: pd.DataFrame,
    now: pd.Timestamp,
) -> None:
    """Park the matches this scrape was for and still could not complete.

    Only matches in a division whose scrape plainly *worked*. The distinction is
    the whole safety of this: a league that comes back with no shots at all is
    the headless browser having died, which happens, and parking on that would
    hide a real gap for a week. A league that comes back with shots for every
    match but one has told us something true about that one.

    Never raises. Forgetting to park costs a repeated scrape; failing the build
    over a cache file costs the day's results.
    """
    try:
        unresolved = wide[wide[["HS", "AS"]].isna().any(axis=1)]
        worked = {
            league
            for league, rows in wide.groupby("League")
            if bool(rows[["HS", "AS"]].notna().all(axis=1).any())
        }
        # Only what this run was actually trying to add -- `wide` carries the
        # league's whole season, most of which the history already has.
        unfillable = unresolved[
            unresolved["League"].isin(worked)
            & _key_series(unresolved).isin(_keys(eligible))
        ]
        if unfillable.empty and deferred.empty:
            return

        _record_deferred(unfillable, deferred, now=now)
        if not unfillable.empty:
            logger.warning(
                "FBref has no shooting line for %d missing match(es) (%s). "
                "Parked for %d days: the row cannot be completed, and without "
                "this the division counts as behind every morning.",
                len(unfillable), _describe(unfillable), RETRY_AFTER.days,
            )
    except Exception as exc:  # noqa: BLE001 - a bookkeeping file must not break the build
        logger.warning("Could not record deferred matches (%s)", exc)


def _describe(matches: pd.DataFrame, limit: int = 4) -> str:
    """``SP1 2026-10-10 Vallecano v Ath Bilbao`` for a log line."""
    shown = [
        f"{row.League} {pd.Timestamp(row.Date).date()} {row.HomeTeam} v {row.AwayTeam}"
        for row in matches.head(limit).itertuples()
    ]
    if len(matches) > limit:
        shown.append(f"and {len(matches) - limit} more")
    return "; ".join(shown)


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
