"""Split every EA edition we hold into one canonical file per season.

Five sources, four different schemas, twelve seasons. Rather than teach the
loader all four, normalise once here and write a uniform file per season into
``data/raw/fifa/``, which is what :func:`footballml.players.fifa.load_fifa`
reads.

**Edition N ships in September of year N-1 and describes season (N-1)/N**, which
the exports' own ``fifa_update_date`` confirms: FIFA 16 is dated 2015-09-21.
Getting this wrong is indistinguishable from the leakage the per-season split
exists to prevent, so the file is named for the season rather than the edition.

The name column is chosen explicitly rather than left to detection. sofifa's
``short_name`` ("K. Mbappe") cannot be matched against a performance source's
"Kylian Mbappe" -- the loose matcher's surname fallback rejects it because Ethan
Mbappe exists too -- so ``long_name`` is used, which the first-plus-last pass
resolves correctly.

Run with::

    python -m pipelines.split_fifa
    python -m pipelines.split_fifa --sources "C:/Users/me/Downloads"
    python -m pipelines.split_fifa --editions 26 27

.. note::
    The raw exports are large (~107 MB) and are **not** in the repository --
    ``data/raw/`` is gitignored. Keep them somewhere durable and point
    ``--sources`` at it; see ``SOURCES`` below for what each file is. Without
    them the twelve per-season files cannot be rebuilt, and every player rating
    depends on them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footballml.data import RAW_DIR  # noqa: E402

OUT = RAW_DIR / "fifa"

#: Where the raw exports are expected, unless --sources says otherwise.
DEFAULT_SOURCES = OUT / "sources"

#: FIFA 15 -> season 2014/15, the oldest season we can rate.
#:
#: The binding constraint is Understat, whose coverage starts there -- a player
#: rating blends an EA overall with performance percentiles, and without the
#: second half there is nothing to rate. Match data goes back to 2010/11, so it
#: is not the limit; an earlier comment here claimed it was.
EARLIEST_EDITION = 15

#: Goalkeeper attributes. EA has no pace or shooting for keepers, so without
#: these a goalkeeper carries no attributes at all and cannot be compared to
#: anyone -- which is how the Player Similarity tab first shipped.
GK_CANON = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]

#: Canonical schema. Names match COLUMN_CANDIDATES in `players/fifa.py` exactly.
#:
#: `alt_name` carries the form a player is actually known by. EA stores legal
#: names -- Vitinha is "Vitor Machado Ferreira", Casemiro is "Carlos Henrique
#: Venancio Casimiro" -- which share no token at all with the name every other
#: source uses, so no amount of key-building can bridge them. sofifa's
#: `short_name` is that familiar form, and throwing it away left 63 of 2025/26's
#: 176 unmatched players unmatchable in principle.
CANON = [
    "name", "alt_name", "overall", "club", "positions", "age",
    "pace", "shooting", "passing", "dribbling", "defending", "physical",
    *GK_CANON,
]

logger = logging.getLogger("split_fifa")


@dataclass(frozen=True)
class Source:
    """One raw export and what it covers."""

    filename: str
    editions: str
    schema: str
    provenance: str


#: The raw exports, by the schema they use. Recorded because the filenames are
#: whatever the download happened to be called and say nothing about what they
#: hold -- "male_players.csv" and "male_players (1).csv" are different editions
#: from different years.
SOURCES: tuple[Source, ...] = (
    Source("male_players (legacy)_23.csv", "15-23", "sofifa",
           "sofifa's combined FIFA 15-23 export; split by its `fifa_version` column"),
    Source("FC26_20250921.csv", "26", "sofifa", "sofifa FC26 export, dated 2025-09-21"),
    Source("male_players.csv", "24", "ea-fc24", "EA's own FC24 site export"),
    Source("male_players (1).csv", "25", "ea-fc25", "EA's own FC25 site export"),
    Source("players.csv", "27", "ea-fc27", "EA FC27 export; already in data/raw/fifa/"),
)


def season_for(edition: int) -> str:
    """Edition 16 -> ``"1516"``; edition 24 -> ``"2324"``."""
    return f"{(edition - 1) % 100:02d}{edition % 100:02d}"


def emit(frame: pd.DataFrame, edition: int, source: str) -> Path | None:
    """Write one season's canonical file."""
    out = frame.reindex(columns=CANON)
    out = out[out["name"].notna() & out["overall"].notna()]
    if out.empty:
        logger.warning("edition %d produced no usable rows from %s", edition, source)
        return None

    path = OUT / f"fifa_{season_for(edition)}.csv"
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8")

    keepers = out[GK_CANON[0]].notna().sum() if GK_CANON[0] in out else 0
    logger.info(
        "  %s  edition %2d  %6d players  %6d with positions  %6d with keeper attrs  <- %s",
        path.name, edition, len(out), out["positions"].notna().sum(), keepers, source,
    )
    return path


def from_sofifa(df: pd.DataFrame) -> pd.DataFrame:
    """sofifa's long format: the legacy FIFA 15-23 file and the FC26 export."""
    return pd.DataFrame({
        "name": df["long_name"],
        "alt_name": df.get("short_name"),
        "overall": pd.to_numeric(df["overall"], errors="coerce"),
        "club": df.get("club_name"),
        "positions": df.get("player_positions"),
        "age": pd.to_numeric(df.get("age"), errors="coerce"),
        "pace": pd.to_numeric(df.get("pace"), errors="coerce"),
        "shooting": pd.to_numeric(df.get("shooting"), errors="coerce"),
        "passing": pd.to_numeric(df.get("passing"), errors="coerce"),
        "dribbling": pd.to_numeric(df.get("dribbling"), errors="coerce"),
        "defending": pd.to_numeric(df.get("defending"), errors="coerce"),
        "physical": pd.to_numeric(df.get("physic"), errors="coerce"),
        **{
            c: pd.to_numeric(df.get(f"goalkeeping_{c[3:]}"), errors="coerce")
            for c in GK_CANON
        },
    })


def from_fc24(df: pd.DataFrame) -> pd.DataFrame:
    """EA's FC24 site export. Carries no goalkeeper attributes at all."""
    return pd.DataFrame({
        "name": df["Name"], "alt_name": df["Name"],
        "overall": pd.to_numeric(df["Overall"], errors="coerce"),
        "club": df.get("Club"), "positions": df.get("Position"),
        "age": pd.to_numeric(df.get("Age"), errors="coerce"),
        "pace": df.get("Pace"), "shooting": df.get("Shooting"),
        "passing": df.get("Passing"), "dribbling": df.get("Dribbling"),
        "defending": df.get("Defending"), "physical": df.get("Physicality"),
    })


def from_fc25(df: pd.DataFrame) -> pd.DataFrame:
    """EA's FC25 site export."""
    # Primary position first, alternates after, so `primary_role` picks the primary.
    pos = df["Position"].fillna("").astype(str)
    alt = df.get("Alternative positions", pd.Series("", index=df.index)).fillna("").astype(str)
    return pd.DataFrame({
        "name": df["Name"], "alt_name": df["Name"],
        "overall": pd.to_numeric(df["OVR"], errors="coerce"),
        "club": df.get("Team"), "positions": (pos + ", " + alt).str.strip(", "),
        "age": pd.to_numeric(df.get("Age"), errors="coerce"),
        "pace": df.get("PAC"), "shooting": df.get("SHO"), "passing": df.get("PAS"),
        "dribbling": df.get("DRI"), "defending": df.get("DEF"), "physical": df.get("PHY"),
        # EA-site exports fill these only for keepers, unlike sofifa which gives
        # every player a (meaninglessly low) diving score. So a populated
        # `gk_diving` does not identify a keeper -- use the role.
        **{c: df.get("GK " + c[3:].capitalize()) for c in GK_CANON},
    })


def from_fc27(df: pd.DataFrame) -> pd.DataFrame:
    """EA's FC27 export, which uses yet another shape."""
    if "gender" in df.columns:
        gender = df["gender"].astype(str)
        male = gender.str.contains("men", case=False, na=True) & ~gender.str.contains(
            "women", case=False, na=False
        )
        dropped = int((~male).sum())
        if dropped:
            logger.info("  dropped %d non-men's entries", dropped)
        df = df[male]

    # `common_name` is blank for most players, so fall back to first + last.
    common = df["common_name"].astype("string").str.strip().replace("", pd.NA)
    joined = (
        df["first_name"].fillna("").astype(str).str.strip() + " "
        + df["last_name"].fillna("").astype(str).str.strip()
    ).str.strip().replace("", pd.NA)
    pos = df["position"].fillna("").astype(str)
    alt = df.get("alternate_positions", pd.Series("", index=df.index)).fillna("").astype(str)

    return pd.DataFrame({
        "name": common.fillna(joined),
        "alt_name": common,
        "overall": pd.to_numeric(df["overall_rating"], errors="coerce"),
        "club": df.get("club"), "positions": (pos + ", " + alt).str.strip(", "),
        "age": pd.NA,
        **{
            c: pd.to_numeric(df.get(f"goalkeeping_{c[3:]}"), errors="coerce")
            for c in GK_CANON
        },
    })


def _locate(source: Source, sources_dir: Path) -> Path | None:
    """Find one export, preferring the sources directory over data/raw/fifa."""
    for candidate in (sources_dir / source.filename, OUT / source.filename):
        if candidate.exists():
            return candidate
    logger.warning(
        "missing %s (editions %s) -- %s. Looked in %s and %s",
        source.filename, source.editions, source.provenance, sources_dir, OUT,
    )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", type=Path, default=DEFAULT_SOURCES,
        help=f"Directory holding the raw exports (default: {DEFAULT_SOURCES}).",
    )
    parser.add_argument(
        "--editions", type=int, nargs="+",
        help="Only rebuild these editions, e.g. 26 27. Default: all available.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    wanted = set(args.editions) if args.editions else None
    written: list[Path] = []

    for source in SOURCES:
        path = _locate(source, args.sources)
        if path is None:
            continue

        if source.schema == "sofifa" and source.editions == "15-23":
            logger.info("=== %s ===", source.provenance)
            frame = pd.read_csv(path, low_memory=False)
            for edition, group in frame.groupby("fifa_version"):
                edition = int(edition)
                if edition < EARLIEST_EDITION:
                    # FIFA 14 and earlier have no counterpart in the performance
                    # data: Understat's coverage begins with 2014/15, so there
                    # would be nothing to blend an EA overall against.
                    continue
                if wanted and edition not in wanted:
                    continue
                written.append(emit(from_sofifa(group), edition, source.filename))
            continue

        edition = int(source.editions)
        if wanted and edition not in wanted:
            continue
        logger.info("=== %s ===", source.provenance)
        reader = {
            "sofifa": from_sofifa, "ea-fc24": from_fc24,
            "ea-fc25": from_fc25, "ea-fc27": from_fc27,
        }[source.schema]
        written.append(emit(reader(pd.read_csv(path, low_memory=False)), edition, source.filename))

    written = [p for p in written if p is not None]
    if not written:
        logger.error(
            "Nothing written. The raw exports are not in the repository -- put "
            "them in %s, or pass --sources. See SOURCES in this file for what "
            "each one is.", args.sources,
        )
        return 1

    logger.info("=== wrote %d season file(s) ===", len(written))
    for path in sorted(OUT.glob("fifa_*.csv")):
        logger.info("  %s  %.1f MB", path.name, path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
