"""Filesystem layout and loaders for local datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TEAM_MATCH_HISTORY = PROCESSED_DIR / "team_match_history.csv"
MATCHES = PROCESSED_DIR / "matches.csv"


def load_team_match_history(path: Path | None = None) -> pd.DataFrame:
    """Load the long team-match history (one row per team per match)."""
    df = pd.read_csv(path or TEAM_MATCH_HISTORY)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_matches(path: Path | None = None) -> pd.DataFrame:
    """Load the wide match table (one row per match)."""
    df = pd.read_csv(path or MATCHES)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


#: Bet365 1X2 decimal odds columns in the football-data.co.uk schema.
ODDS_COLUMNS = ("B365H", "B365D", "B365A")


def load_raw_odds(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load bookmaker 1X2 odds from the raw football-data.co.uk CSVs.

    Odds are used **only** as an evaluation benchmark -- the market is the
    standard every football model is measured against. They are deliberately
    excluded from the feature set (see
    :func:`footballml.models.match_model.feature_columns`).

    Returns:
        Frame of ``Date``, ``HomeTeam``, ``AwayTeam`` and the odds columns, for
        matches where a full set of odds is available.
    """
    from footballml.ingest.matchhistory import read_csv_bytes

    raw_dir = raw_dir or RAW_DIR
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        try:
            # Shared reader: encoding varies file to file, including one
            # BOM-prefixed outlier that naive decoding silently mangles.
            df = read_csv_bytes(path.read_bytes())
        except ValueError:
            continue
        keep = ["Date", "HomeTeam", "AwayTeam", *ODDS_COLUMNS]
        if not set(keep).issubset(df.columns):
            continue
        sub = df[keep].copy()
        sub["HomeTeam"] = sub["HomeTeam"].str.strip()
        sub["AwayTeam"] = sub["AwayTeam"].str.strip()
        # football-data.co.uk uses day-first dates, two-digit years in older files.
        sub["Date"] = pd.to_datetime(sub["Date"], dayfirst=True, format="mixed")
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", *ODDS_COLUMNS])

    odds = pd.concat(frames, ignore_index=True)
    return odds.dropna(subset=list(ODDS_COLUMNS)).reset_index(drop=True)
