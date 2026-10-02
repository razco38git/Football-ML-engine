"""Forward fixtures from FBref's season schedule.

football-data.co.uk publishes a single rolling file of scheduled matches, and
for a long time that was the only forward-looking source here. It has two
problems that only show up in production:

**It is a snapshot, not a schedule.** On 2026-09-24 it held 18-20 September --
fixtures already played. The weekly job runs on a Monday, so it routinely sees
the round that has just finished rather than the one coming, stores nothing, and
reports success.

**Coverage is therefore luck.** Whatever happens to be in that file when the job
runs is what gets predicted. Of the eighteen predictions in the live record at
the time of writing, *all eighteen* were Spanish: four leagues had never once
had a fixture stored, not because anything failed but because the feed never
happened to be ahead of the job for them.

FBref publishes the whole season instead -- every fixture, with its date, months
ahead. That turns "what does this feed happen to hold today" into "what is
actually scheduled between now and next Tuesday", which is a question with a
stable answer.

Cross-checked against :func:`footballml.projection.remaining_fixtures`, which
derives the same thing independently by assuming a double round-robin: both give
330 (E0), 311 (SP1), 261 (F1), 270 (D1) and 330 (I1) unplayed for 2026/27.

One request per league-season, cached by soccerdata under ``~/soccerdata``.
"""

from __future__ import annotations

import logging

import pandas as pd

from footballml.entities.teams import load_aliases
from footballml.ingest.matchhistory import current_season_label
from footballml.ingest.understat import UNDERSTAT_LEAGUES

logger = logging.getLogger(__name__)

#: How far ahead to look by default. A week and a half covers a normal weekend
#: round plus any midweek fixtures, without predicting matches whose team news
#: is a fortnight from being known.
DEFAULT_HORIZON_DAYS = 10

#: Longer, for UEFA competitions. Their rounds are a fortnight apart rather than
#: weekly, and an international break stretches the gap further -- so a ten-day
#: window leaves the Champions League tab empty most of the time, which reads as
#: a broken page rather than as the calendar. Three weeks covers a European
#: round plus a break. The cost is predicting slightly further from team news,
#: which matters less here: these are cross-league ties the model already judges
#: with about half its usual edge.
COMPETITION_HORIZON_DAYS = 21

#: FBref writes scores with an en-dash. A row with no score is unplayed, which
#: is exactly what this module is for -- but a parse that quietly fails would
#: mark *every* fixture unplayed, so the dash characters are matched explicitly.
_SCORE = r"(\d+)\s*[-‐‑‒–—―]\s*(\d+)"


def fetch_schedule(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> pd.DataFrame:
    """Every scheduled fixture, played or not, in canonical team names.

    Args:
        leagues: Division codes, e.g. ``["E0"]``. A UEFA competition code from
            :data:`footballml.ingest.european.EUROPEAN_LEAGUES` -- ``"UCL"`` and
            friends -- is accepted too. Defaults to the five domestic divisions.
        seasons: Season labels. Defaults to the current one.

    Returns:
        ``League``, ``Season``, ``Date``, ``HomeTeam``, ``AwayTeam`` and
        ``played``. Empty if soccerdata is not installed, so the caller can fall
        back rather than crash -- FBref lives in the optional ingest extra.
    """
    # Imported here, not at module scope: `european` reads `_SCORE` from this
    # module, so a top-level import would be circular.
    from footballml.ingest.european import EUROPEAN_LEAGUES, _register

    known = {**UNDERSTAT_LEAGUES, **EUROPEAN_LEAGUES}
    codes = list(leagues or UNDERSTAT_LEAGUES)
    unknown = [c for c in codes if c not in known]
    if unknown:
        raise KeyError(f"unknown competition code(s): {unknown}")
    seasons = seasons or [current_season_label(pd.Timestamp.today())]

    try:
        import soccerdata as sd
    except ImportError:  # pragma: no cover - depends on the optional extra
        logger.warning("soccerdata not installed; no schedule available")
        return _empty()

    # soccerdata ships no UEFA club competitions, so they must be registered
    # before the reader will accept their names.
    uefa = [c for c in codes if c in EUROPEAN_LEAGUES]
    if uefa:
        _register(uefa)

    reader = sd.FBref(leagues=[known[c] for c in codes], seasons=list(seasons))
    raw = reader.read_schedule().reset_index()
    if raw.empty:
        return _empty()

    inverse = {v: k for k, v in known.items()}
    aliases = load_aliases("fbref")

    out = pd.DataFrame(
        {
            "League": raw["league"].map(inverse),
            "Season": raw["season"].astype(str),
            "Date": pd.to_datetime(raw["date"], errors="coerce").dt.normalize(),
            "HomeTeam": raw["home_team"].astype(str).str.strip().replace(aliases),
            "AwayTeam": raw["away_team"].astype(str).str.strip().replace(aliases),
            "played": raw["score"].astype(str).str.extract(_SCORE)[0].notna(),
        }
    )
    out = out.dropna(subset=["Date", "League"])
    logger.info(
        "Schedule: %d fixtures across %d league-season(s), %d already played",
        len(out), out.groupby(["League", "Season"]).ngroups, int(out["played"].sum()),
    )
    return out.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)


def window_fixtures(
    schedule: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Unplayed fixtures in ``schedule`` kicking off within ``horizon_days``.

    Separate from :func:`fetch_schedule` so a caller can tell an empty window
    ("nothing is scheduled for the next ten days") from an empty schedule ("the
    source gave us nothing"). Those need opposite responses -- the first is
    normal and the second wants somebody's attention -- and they are
    indistinguishable once both have become an empty frame.
    """
    if schedule.empty:
        return _empty().drop(columns=["played"])

    today = (today or pd.Timestamp.today()).normalize()
    horizon = today + pd.Timedelta(days=horizon_days)
    window = schedule[
        (~schedule["played"]) & (schedule["Date"] >= today) & (schedule["Date"] <= horizon)
    ]
    logger.info(
        "%d unplayed fixture(s) between %s and %s",
        len(window), today.date(), horizon.date(),
    )
    return window.drop(columns=["played"]).reset_index(drop=True)


def upcoming_fixtures(
    leagues: list[str] | None = None,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Unplayed fixtures kicking off within ``horizon_days``.

    An empty result is a legitimate answer, not a failure: at the time of
    writing the big five had no fixture between 20 September and 9 October
    because of an international break.
    """
    return window_fixtures(fetch_schedule(leagues), horizon_days, today)


def next_fixture_date(leagues: list[str] | None = None) -> pd.Timestamp | None:
    """When the next unplayed fixture is, or None if the season is over.

    Lets a caller distinguish "nothing scheduled for a fortnight" from "the
    fixture source is broken", which otherwise look identical.
    """
    schedule = fetch_schedule(leagues)
    pending = schedule[~schedule["played"]] if not schedule.empty else schedule
    return None if pending.empty else pending["Date"].min()


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["League", "Season", "Date", "HomeTeam", "AwayTeam", "played"]
    )
