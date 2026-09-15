"""football-data.co.uk ingestion and reshaping.

This is the spine of the match dataset: results, shots, shots on target, cards,
corners and closing odds, in one consistent schema across all five target
leagues and back to 2010.

The files are read directly rather than through ``soccerdata``. That library
picks its encoding from the season number -- ``latin-1`` before 2024/25,
``utf-8-sig`` after -- but the source does not actually follow that rule.
``E0_2122.csv`` was re-saved upstream with a UTF-8 BOM, so reading it as latin-1
turns the first header into ``ï»¿Div``, the ``Div -> league`` rename silently
does nothing, and the whole season is dropped without an error. Detecting the
encoding per file instead costs a few lines and removes both that bug and a
heavyweight browser dependency from the most important data source.

The other transformation here is **wide to long**. football-data.co.uk publishes
one row per match with ``Home*``/``Away*`` column pairs. Rolling form is
naturally computed per team, so everything downstream works on a long frame with
one row per team per match and a ``Venue`` column. Splitting a match into two
mirrored rows is what makes "last five matches at this venue" a simple groupby
rather than a special case.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"

#: Upcoming fixtures across all divisions, refreshed by the source continuously.
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

#: Courtesy delay between downloads. The files are small and static, but there
#: is no reason to hammer a free public resource.
REQUEST_DELAY_SECONDS = 0.5


def current_season_label(today: pd.Timestamp | None = None) -> str:
    """The season label covering ``today``, e.g. ``"2627"`` in September 2026.

    European seasons run August to May, so anything from July onward belongs to
    the season starting that calendar year.
    """
    today = today or pd.Timestamp.today()
    start_year = today.year if today.month >= 7 else today.year - 1
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"

#: football-data.co.uk division codes to soccerdata league identifiers.
LEAGUES = {
    "E0": "ENG-Premier League",
    "SP1": "ESP-La Liga",
    "D1": "GER-Bundesliga",
    "I1": "ITA-Serie A",
    "F1": "FRA-Ligue 1",
}

#: Wide column stems, mapped to their long-format ``For``/``Against`` names.
#: Keys are the football-data.co.uk prefixes (H = home, A = away).
_STAT_COLUMNS = {
    "S": "Shots",
    "ST": "ShotsOnTarget",
    "F": "Fouls",
    "C": "Corners",
    "Y": "YellowCards",
    "R": "RedCards",
}


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse a football-data.co.uk CSV, detecting its encoding.

    ``utf-8-sig`` transparently handles both BOM-prefixed and plain UTF-8. Older
    files carrying accented characters in Latin-1 fail that decode, so they fall
    back. Trailing blank rows are common in these files and are dropped.
    """
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(data), encoding=encoding, on_bad_lines="warn", low_memory=False
            )
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - both encodings failing would be a source change
        raise ValueError("could not decode CSV as utf-8-sig or latin-1")

    if "Div" not in df.columns:
        raise ValueError(f"no 'Div' column; got {list(df.columns)[:5]}")
    return df.dropna(how="all").dropna(subset=["Div"])


def fetch_match_history(
    leagues: list[str] | None = None,
    seasons: list[str] | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download (or read from cache) raw wide match rows.

    Args:
        leagues: Division codes such as ``["E0", "SP1"]``. Defaults to all five.
        seasons: Four-digit season labels such as ``["2425"]``.
        cache_dir: Where CSVs are stored. Defaults to ``data/raw``.
        refresh: Re-download even when a cached copy exists. Needed for the
            current season, which gains rows every week.
    """
    from footballml.data import RAW_DIR

    codes = leagues or sorted(LEAGUES)
    seasons = seasons or []
    cache_dir = cache_dir or RAW_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # The in-progress season gains rows every week, so a cached copy is stale by
    # definition. Completed seasons never change and are safe to cache forever.
    live_season = current_season_label()

    frames = []
    for code in codes:
        for season in seasons:
            path = cache_dir / f"{code}_{season}.csv"
            if refresh or season >= live_season or not path.exists():
                url = f"{BASE_URL}/{season}/{code}.csv"
                logger.info("Downloading %s", url)
                response = requests.get(url, timeout=60, headers={"User-Agent": "footballml"})
                if response.status_code == 404:
                    logger.warning("No data published for %s %s", code, season)
                    continue
                response.raise_for_status()
                path.write_bytes(response.content)
                time.sleep(REQUEST_DELAY_SECONDS)

            try:
                frames.append(read_csv_bytes(path.read_bytes()).assign(_season=season))
            except ValueError as exc:
                logger.error("Skipping %s: %s", path.name, exc)

    if not frames:
        raise ValueError("no match data could be read")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    logger.info("Read %d matches across %d leagues", len(raw), raw["Div"].nunique())
    return raw


def fetch_fixtures(leagues: list[str] | None = None) -> pd.DataFrame:
    """Fetch upcoming, unplayed fixtures.

    football-data.co.uk publishes a single rolling file of scheduled matches
    across every division it covers, typically the next week or two. This is the
    only part of the pipeline that is genuinely *forward*-looking -- everything
    else is history.

    Never cached: the whole point is that it changes.

    Returns:
        ``League``, ``Season``, ``Date``, ``HomeTeam``, ``AwayTeam`` and, where
        published, 1X2 odds. Goal columns are absent because these have not
        been played.
    """
    codes = set(leagues or LEAGUES)
    logger.info("Fetching fixtures from %s", FIXTURES_URL)
    response = requests.get(FIXTURES_URL, timeout=60, headers={"User-Agent": "footballml"})
    response.raise_for_status()

    raw = read_csv_bytes(response.content)
    raw = raw[raw["Div"].isin(codes)]
    if raw.empty:
        logger.warning("No upcoming fixtures for %s", sorted(codes))
        return pd.DataFrame(columns=["League", "Season", "Date", "HomeTeam", "AwayTeam"])

    out = pd.DataFrame(
        {
            "League": raw["Div"],
            "Date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed").dt.normalize(),
            "HomeTeam": raw["HomeTeam"].str.strip(),
            "AwayTeam": raw["AwayTeam"].str.strip(),
        }
    )
    out["Season"] = [current_season_label(d) for d in out["Date"]]
    for col in ("B365H", "B365D", "B365A"):
        if col in raw.columns:
            out[col] = pd.to_numeric(raw[col], errors="coerce")

    logger.info("Fetched %d upcoming fixtures", len(out))
    return out.sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)


def to_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw rows into the wide match table."""
    out = pd.DataFrame(
        {
            "League": raw["Div"],
            "Season": raw["_season"].astype(str),
            # Day-first throughout; older files use two-digit years.
            "Date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed").dt.normalize(),
            "HomeTeam": raw["HomeTeam"].str.strip(),
            "AwayTeam": raw["AwayTeam"].str.strip(),
            "FTHG": pd.to_numeric(raw["FTHG"], errors="coerce"),
            "FTAG": pd.to_numeric(raw["FTAG"], errors="coerce"),
            "FTR": raw["FTR"],
        }
    )
    for prefix, name in _STAT_COLUMNS.items():
        out[f"H{name}"] = pd.to_numeric(raw.get(f"H{prefix}"), errors="coerce")
        out[f"A{name}"] = pd.to_numeric(raw.get(f"A{prefix}"), errors="coerce")

    # Matches without a result are fixtures that have not been played.
    played = out["FTHG"].notna() & out["FTAG"].notna()
    if (~played).any():
        logger.info("Dropping %d unplayed fixtures", int((~played).sum()))
    return out[played].sort_values(["Date", "League", "HomeTeam"]).reset_index(drop=True)


def to_team_match_history(matches: pd.DataFrame) -> pd.DataFrame:
    """Explode the wide match table into one row per team per match."""
    home = _side(matches, "H", "A", "Home")
    away = _side(matches, "A", "H", "Away")
    long_df = pd.concat([home, away], ignore_index=True)
    return long_df.sort_values(["Date", "League", "Team"]).reset_index(drop=True)


def _side(matches: pd.DataFrame, own: str, opp: str, venue: str) -> pd.DataFrame:
    """Build one side's rows, orienting every stat from that team's perspective."""
    is_home = venue == "Home"
    goals_for = matches["FTHG"] if is_home else matches["FTAG"]
    goals_against = matches["FTAG"] if is_home else matches["FTHG"]

    out = pd.DataFrame(
        {
            "League": matches["League"],
            "Season": matches["Season"],
            "Date": matches["Date"],
            "Team": matches["HomeTeam"] if is_home else matches["AwayTeam"],
            "Opponent": matches["AwayTeam"] if is_home else matches["HomeTeam"],
            "Venue": venue,
            "GoalsFor": goals_for,
            "GoalsAgainst": goals_against,
        }
    )
    # Derive the result from goals rather than reading FTR, so it is always
    # oriented from this team's point of view.
    out["Result"] = pd.Series("D", index=out.index).where(
        goals_for == goals_against,
        pd.Series("W", index=out.index).where(goals_for > goals_against, "L"),
    )
    for name in _STAT_COLUMNS.values():
        out[f"{name}For"] = matches[f"{own}{name}"]
        out[f"{name}Against"] = matches[f"{opp}{name}"]
    return out
