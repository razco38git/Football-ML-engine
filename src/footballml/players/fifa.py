"""EA FC / FIFA ratings ingestion.

Read from a CSV you download rather than scraped. EA's ratings page is backed by
an undocumented internal API, SoFIFA and cmtracker both sit behind Cloudflare,
and none of the three is published for programmatic use. A Kaggle export is
stable, legitimate, and arrives with the one thing every free performance source
lacks: **detailed positions**.

That last point matters more than the ratings themselves. Understat knows only
GK/D/M/F, which is why a centre-back was compared against attacking full-backs
and a holding midfielder against a winger. EA's ``CB``, ``LB``, ``CDM``, ``CAM``,
``RW``, ``ST`` give the five outfield roles directly.

Column names differ between Kaggle publishers, so the loader detects them rather
than demanding one exact schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from footballml.data import RAW_DIR
from footballml.players.fbref import normalise_name

logger = logging.getLogger(__name__)

#: Where a downloaded EA FC export is looked for.
FIFA_DIR = RAW_DIR / "fifa"

#: Candidate column names per field, tried in order. Covers the naming used by
#: the common Kaggle EA FC exports without forcing one publisher's schema.
COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "name": ("short_name", "long_name", "name", "player_name", "Name", "Player", "PLAYER"),
    "overall": ("overall", "overall_rating", "Overall", "OVR", "ovr", "rating"),
    "club": ("club_name", "club", "Club", "team", "Team", "TEAM", "club_team"),
    "positions": (
        "player_positions", "positions", "position", "Position", "POS", "best_position",
    ),
    "age": ("age", "Age", "AGE"),
    "pace": ("pace", "Pace", "PAC"),
    "shooting": ("shooting", "Shooting", "SHO"),
    "passing": ("passing", "Passing", "PAS"),
    "dribbling": ("dribbling", "Dribbling", "DRI"),
    "defending": ("defending", "Defending", "DEF"),
    "physical": ("physic", "physical", "Physicality", "PHY"),
}

#: EA position codes to the five outfield roles, plus goalkeeper.
#:
#: Centre-backs and full-backs are separated because they do completely
#: different jobs; lumping them together is what let overlapping full-backs
#: out-rate every genuine centre-back. Wingers sit with attacking midfielders
#: because their output profiles are near-identical.
ROLE_BY_POSITION = {
    "GK": "GK",
    "CB": "CB",
    "LCB": "CB", "RCB": "CB",
    "LB": "FB", "RB": "FB", "LWB": "FB", "RWB": "FB",
    "CDM": "MID", "CM": "MID", "LDM": "MID", "RDM": "MID", "LCM": "MID", "RCM": "MID",
    "CAM": "AMW", "LM": "AMW", "RM": "AMW", "LW": "AMW", "RW": "AMW",
    "ST": "FWD", "CF": "FWD", "LS": "FWD", "RS": "FWD", "LF": "FWD", "RF": "FWD",
}

#: Display names for the roles.
ROLE_LABELS = {
    "GK": "Goalkeeper",
    "CB": "Centre back",
    "FB": "Full back",
    "MID": "Midfielder",
    "AMW": "Attacking midfielder / winger",
    "FWD": "Forward",
}

#: Understat's coarse groups, used when a player has no EA entry.
FALLBACK_ROLE = {"GK": "GK", "D": "CB", "M": "MID", "F": "FWD"}


class FifaDataMissingError(FileNotFoundError):
    """Raised with download instructions when no EA FC export is present."""


def _find_column(df: pd.DataFrame, field: str) -> str | None:
    """Locate a field in a frame whose exact column naming is unknown."""
    for candidate in COLUMN_CANDIDATES[field]:
        if candidate in df.columns:
            return candidate
    # Case-insensitive second pass, since publishers differ on capitalisation.
    lowered = {c.lower(): c for c in df.columns}
    for candidate in COLUMN_CANDIDATES[field]:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def primary_role(positions: str | float) -> str | None:
    """Map an EA position string to one role.

    EA lists positions best-first (``"ST, LW"``), so the first recognised entry
    is the player's primary role.
    """
    if not isinstance(positions, str):
        return None
    for token in positions.replace("|", ",").split(","):
        role = ROLE_BY_POSITION.get(token.strip().upper())
        if role:
            return role
    return None


def load_fifa(directory: Path | None = None) -> pd.DataFrame:
    """Load an EA FC export from ``data/raw/fifa/``.

    Any CSV in that directory is accepted; the first with a detectable name and
    overall column wins.

    Raises:
        FifaDataMissingError: When no usable CSV is found, with instructions.
    """
    directory = directory or FIFA_DIR
    candidates = sorted(directory.glob("*.csv")) if directory.exists() else []

    for path in candidates:
        df = pd.read_csv(path, low_memory=False)
        name_col = _find_column(df, "name")
        overall_col = _find_column(df, "overall")
        if not name_col or not overall_col:
            logger.warning("%s has no recognisable name/overall column, skipping", path.name)
            continue

        out = pd.DataFrame(
            {
                "fifa_name": df[name_col].astype(str),
                "fifa_overall": pd.to_numeric(df[overall_col], errors="coerce"),
            }
        )
        for field in ("club", "positions", "age", "pace", "shooting", "passing",
                      "dribbling", "defending", "physical"):
            column = _find_column(df, field)
            if column is not None:
                values = df[column]
                out[f"fifa_{field}"] = (
                    values.astype(str) if field in {"club", "positions"}
                    else pd.to_numeric(values, errors="coerce")
                )

        out["role"] = out.get("fifa_positions", pd.Series(dtype="object")).map(primary_role)
        out["_norm"] = out["fifa_name"].map(normalise_name)
        out = out.dropna(subset=["fifa_overall"])
        # Keep the best-rated entry per name: exports often carry several
        # versions of the same player across rating updates.
        out = out.sort_values("fifa_overall", ascending=False).drop_duplicates("_norm")

        logger.info(
            "Loaded %d FIFA players from %s (%d with a mapped role)",
            len(out), path.name, int(out["role"].notna().sum()),
        )
        return out.reset_index(drop=True)

    raise FifaDataMissingError(
        f"No EA FC export found in {directory}.\n"
        "Download one of these and drop the CSV in that folder:\n"
        "  https://www.kaggle.com/datasets/justdhia/ea-sports-fc-26-player-ratings\n"
        "  https://www.kaggle.com/datasets/flynn28/eafc26-player-database\n"
        "  https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats\n"
        "Any schema works: the loader detects the name, overall and position columns."
    )


def attach_fifa(players: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Attach FIFA ratings and roles to performance rows, matched by name.

    Matched on normalised name across the whole export rather than within a
    club, because EA's club field reflects one moment in a transfer window and
    our rows span whole seasons. Full names are distinctive enough that this is
    safe; club-constrained matching loses far more than it protects.
    """
    out = players.copy()
    out["_norm"] = out["Player"].map(normalise_name)

    columns = [c for c in fifa.columns if c.startswith("fifa_") or c == "role"]
    merged = out.merge(fifa[["_norm", *columns]], on="_norm", how="left")

    rate = merged["fifa_overall"].notna().mean()
    logger.info("FIFA match rate: %.0f%% of %d player-seasons", rate * 100, len(merged))

    # Players with no EA entry keep a coarse role from Understat, so they are
    # still rated rather than dropped.
    merged["role"] = merged["role"].fillna(merged["position_group"].map(FALLBACK_ROLE))
    return merged.drop(columns=["_norm"])
