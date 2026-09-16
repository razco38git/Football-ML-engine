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
import re
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
    # `long_name` outranks `short_name` deliberately. sofifa exports carry both,
    # and the short form is an initial plus surname -- "K. Mbappe" -- which the
    # loose matcher cannot resolve: the first-plus-last key never matches, and
    # the surname fallback rejects it because Ethan Mbappe exists too. The long
    # form ("Kylian Mbappe Lottin") does match. `short_name` stays last as a
    # fallback for exports that carry nothing else.
    "name": (
        "long_name", "common_name", "name", "player_name",
        "Name", "Player", "PLAYER", "short_name",
    ),
    "first_name": ("first_name", "firstname", "given_name"),
    "last_name": ("last_name", "lastname", "surname", "family_name"),
    "overall": ("overall", "overall_rating", "Overall", "OVR", "ovr", "rating"),
    "club": ("club_name", "club", "Club", "team", "Team", "TEAM", "club_team"),
    "positions": (
        "player_positions", "positions", "position", "Position", "POS", "best_position",
    ),
    "gender": ("gender", "Gender", "sex"),
    "league": ("league", "League", "league_name"),
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


def _build_names(df: pd.DataFrame) -> pd.Series | None:
    """Assemble a display name, whichever columns the export provides.

    Exports disagree about this more than any other field. The FC27 database
    carries ``common_name`` but leaves it blank for most players -- Mbappé has
    an empty one and only ``first_name`` / ``last_name`` filled -- so a single
    column is never enough. Falls back to joining first and last, and fills any
    remaining blanks from the other source.
    """
    name_col = _find_column(df, "name")
    first_col = _find_column(df, "first_name")
    last_col = _find_column(df, "last_name")

    combined = None
    if first_col and last_col:
        combined = (
            df[first_col].fillna("").astype(str).str.strip()
            + " "
            + df[last_col].fillna("").astype(str).str.strip()
        ).str.strip()
        combined = combined.replace("", pd.NA)

    if name_col:
        primary = df[name_col].astype("string").str.strip().replace("", pd.NA)
        return primary.fillna(combined) if combined is not None else primary.fillna("")

    return combined


#: Per-season exports, named by the season they describe rather than by edition.
#:
#: Naming by season is deliberate. EA editions are named for the year *after*
#: release -- FIFA 16 shipped in September 2015 and describes 2015/16 -- so a
#: file called ``fifa_16.csv`` is ambiguous in exactly the way that produces an
#: off-by-one-season error, which is indistinguishable from the leakage this
#: whole arrangement exists to prevent.
SEASON_FILE_GLOB = "fifa_[0-9][0-9][0-9][0-9].csv"

#: Characters outside Latin and its extensions, which some exports append to a
#: name with no separator at all.
#:
#: The FC26 database stores Mohamed Salah as ``"Mohamed Salah Hamed Ghalyمحمد صلاح"``
#: -- the Arabic runs straight on from "Ghaly" -- and does this to 9.8% of its
#: entries, which is enough on its own to drop a season's match rate by twenty
#: points. The ranges kept are ASCII, Latin-1 Supplement, Latin Extended-A/B and
#: Latin Extended Additional, so accented names survive untouched:
#: ``"Willum Þór Willumsson"`` and ``"Tjaš Begić"`` pass through unchanged.
_FOREIGN_SCRIPT = re.compile(r"[^\x00-\x7FÀ-ɏḀ-ỿ]")


def _strip_foreign_script(name: str) -> str:
    """Drop non-Latin script from a name and tidy the whitespace it leaves."""
    return re.sub(r"\s+", " ", _FOREIGN_SCRIPT.sub("", str(name))).strip()


def _normalise_export(df: pd.DataFrame, source: str) -> pd.DataFrame | None:
    """Reduce one export to the canonical ``fifa_*`` columns.

    Returns ``None`` when the frame has no detectable name or overall column,
    so the caller can skip it rather than fail the whole load.
    """
    overall_col = _find_column(df, "overall")
    if not overall_col:
        logger.warning("%s has no recognisable overall column, skipping", source)
        return None

    names = _build_names(df)
    if names is None:
        logger.warning("%s has no recognisable name column, skipping", source)
        return None

    # Women's players share the file but never our leagues, and their names
    # can collide with men's. Drop them rather than risk a wrong join.
    gender_col = _find_column(df, "gender")
    if gender_col is not None:
        mens = df[gender_col].astype(str).str.contains("men", case=False, na=True)
        womens = df[gender_col].astype(str).str.contains("women", case=False, na=False)
        keep = mens & ~womens
        if keep.any():
            logger.info("Filtered out %d non-men's entries", int((~keep).sum()))
            df, names = df[keep], names[keep]

    out = pd.DataFrame(
        {
            "fifa_name": names.astype(str).map(_strip_foreign_script),
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
    # Keep the best-rated entry per name: exports often carry several versions
    # of the same player across rating updates.
    out = out.sort_values("fifa_overall", ascending=False).drop_duplicates("_norm")
    return out.reset_index(drop=True)


def load_fifa(directory: Path | None = None) -> pd.DataFrame:
    """Load EA FC exports from ``data/raw/fifa/``.

    Prefers one file per season (``fifa_1516.csv`` ... ``fifa_2627.csv``), and
    returns them stacked with a ``Season`` column so each season can be matched
    against the ratings that were current *at the time*.

    Falls back to the old behaviour -- any single CSV, no season -- when no
    per-season files are present. That fallback is a real compromise, not a
    convenience: a single export describes one moment, so using it for every
    season means a player carries today's rating back through his whole career.
    The 2026 database rated Lamine Yamal 90, which applied to 2015/16 would
    describe an eight-year-old. Per-season files are strongly preferred.

    Raises:
        FifaDataMissingError: When no usable CSV is found, with instructions.
    """
    directory = directory or FIFA_DIR
    seasonal = sorted(directory.glob(SEASON_FILE_GLOB)) if directory.exists() else []

    if seasonal:
        frames = []
        for path in seasonal:
            season = path.stem.split("_")[-1]
            normalised = _normalise_export(pd.read_csv(path, low_memory=False), path.name)
            if normalised is None:
                continue
            normalised["Season"] = season
            frames.append(normalised)

        if frames:
            out = pd.concat(frames, ignore_index=True)
            logger.info(
                "Loaded %d FIFA player-seasons across %d editions (%s..%s)",
                len(out), len(frames),
                out["Season"].min(), out["Season"].max(),
            )
            return out

    # Largest first: a partial export sitting alongside a full one should not
    # win just because it sorts earlier.
    candidates = (
        sorted(directory.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        if directory.exists()
        else []
    )
    for path in candidates:
        normalised = _normalise_export(pd.read_csv(path, low_memory=False), path.name)
        if normalised is None:
            continue
        logger.warning(
            "Using %s for every season -- no per-season files found, so ratings "
            "will not reflect the season they are applied to", path.name,
        )
        return normalised

    raise FifaDataMissingError(
        f"No EA FC export found in {directory}.\n"
        "Download one of these and drop the CSV in that folder:\n"
        "  https://www.kaggle.com/datasets/justdhia/ea-sports-fc-26-player-ratings\n"
        "  https://www.kaggle.com/datasets/flynn28/eafc26-player-database\n"
        "  https://www.kaggle.com/datasets/nyagami/ea-sports-fc-25-database-ratings-and-stats\n"
        "Any schema works: the loader detects the name, overall and position columns.\n"
        "Name each file for the season it describes -- fifa_1516.csv for FIFA 16 --\n"
        "so ratings are matched to the season they were current in."
    )


#: Minimum exact-name matches before a team's EA club name is trusted, and the
#: share of those matches the most common club must hold.
CLUB_EVIDENCE_MIN = 3
CLUB_EVIDENCE_SHARE = 0.5


def _club_names(merged: pd.DataFrame) -> dict[str, str]:
    """Learn what EA calls each of our teams, from players already matched.

    The two sources spell clubs nothing alike -- ``"Bayern Munich"`` against
    ``"FC Bayern München"``, ``"FC Cologne"`` against ``"1. FC Köln"`` -- and
    comparing the strings would happily confuse ``Real Betis`` with
    ``Real Sociedad``. The exact name matches already say which is which: if
    most of a team's matched players carry one EA club, that is its EA name.
    Transfers put a few players under their old club, hence the majority rule
    rather than unanimity.
    """
    if "Team" not in merged.columns or "fifa_club" not in merged.columns:
        return {}
    # The loader stringifies clubs, so a missing one arrives as "nan".
    clubs = merged["fifa_club"].astype("string")
    known = merged["fifa_overall"].notna() & clubs.notna() & (clubs != "nan")
    matched = merged[known]
    names: dict[str, str] = {}
    for team, clubs in matched.groupby("Team", observed=True)["fifa_club"]:
        counts = clubs.value_counts()
        top = counts.iloc[0]
        if top >= CLUB_EVIDENCE_MIN and top / len(clubs) >= CLUB_EVIDENCE_SHARE:
            names[str(team)] = counts.index[0]
    return names


def _match_loosely(
    merged: pd.DataFrame, fifa: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Second pass for names the two sources spell differently.

    Four kinds of mismatch remain after exact comparison, each handled by its
    own key and tried strongest first:

    - **Extra given names.** ``"Ionuț Andrei Radu"`` against ``"Ionut Radu"``,
      matched on first-plus-last.
    - **Trailing family names.** The mirror image, and the common case in
      Iberian, French and Brazilian naming: EA stores the full legal name
      ``"Kylian Mbappé Lottin"`` where the performance source has
      ``"Kylian Mbappe"``. First-plus-last builds ``"kylian lottin"`` here and
      never matches, so the first *two* tokens are tried as well. sofifa exports
      carry three or more tokens for roughly half their entries, against 8% of
      the EA-site exports, which is why those seasons matched ~14 points worse.
    - **Short names.** EA often stores a Spanish or Portuguese player under a
      surname alone -- ``"De Gea"``, ``"Sivera"``, ``"Álex Remiro"`` -- where the
      performance source has the full name.
    - **Surname only.** Last resort, and accepted *only where that surname is
      unique across the whole EA database*. Without that guard, every Petrović
      and every Silva would collapse onto one player.

    The last two tiers are guarded by uniqueness for the same reason: a key that
    identifies several players identifies none of them. Measured on the 2025/26
    edition, the leading-pair key is unique for 256 of the 269 names it resolves,
    so the guard costs little and removes the guesswork.

    **Club breaks ties, and only ties.** ``"David García"`` at Las Palmas shares
    a leading pair with both David García Santana (Las Palmas) and David García
    Zubiría (Osasuna), so neither is taken on name alone -- but exactly one is
    at his club. Where a key names several players and exactly one of them is
    at the row's club, that one is taken. Club is never *required*: EA's club
    is one transfer-window snapshot, so demanding it would lose every player
    who moved (see :func:`attach_fifa`). The surname tier also demands the
    given name appear somewhere in the EA name, since "same surname, same club"
    alone is not proof -- brothers and namesakes share dressing rooms.
    """
    unmatched = merged["fifa_overall"].isna()
    if not unmatched.any():
        return merged

    club_names = _club_names(merged)

    def keys_for(name: str) -> list[str]:
        parts = normalise_name(name).split()
        if not parts:
            return []
        keys = [" ".join(parts)]
        if len(parts) > 2:
            keys.append(f"{parts[0]} {parts[-1]}")
        return keys

    # Every candidate per key, in export order -- highest-rated first, since
    # the loader sorts that way.
    strong: dict[str, list[pd.Series]] = {}
    leading: dict[str, list[pd.Series]] = {}
    surnames: dict[str, list[pd.Series]] = {}

    for _, row in fifa.iterrows():
        for key in keys_for(str(row["fifa_name"])):
            strong.setdefault(key, []).append(row)
        parts = normalise_name(str(row["fifa_name"])).split()
        if len(parts) > 2:
            leading.setdefault(f"{parts[0]} {parts[1]}", []).append(row)
        if parts:
            surnames.setdefault(parts[-1], []).append(row)

    def at_club(candidates: list[pd.Series], team: object) -> pd.Series | None:
        """The single candidate at the row's club, if there is exactly one."""
        club = club_names.get(str(team))
        if club is None or "fifa_club" not in fifa.columns:
            return None
        here = [c for c in candidates if c["fifa_club"] == club]
        return here[0] if len(here) == 1 else None

    has_team = "Team" in merged.columns
    for idx in merged.index[unmatched]:
        name = str(merged.at[idx, "Player"])
        team = merged.at[idx, "Team"] if has_team else None
        parts = normalise_name(name).split()
        hit = None

        # Strong keys may map to several players. Prefer the one at his club;
        # otherwise keep the highest-rated, which is overwhelmingly the one a
        # top-five-league dataset means.
        for key in keys_for(name):
            if key in strong:
                candidates = strong[key]
                hit = candidates[0]
                if len(candidates) > 1:
                    local = at_club(candidates, team)
                    hit = local if local is not None else hit
                break

        if hit is None:
            # A two-token name against an EA entry carrying extra family names.
            # Unique leading pairs only: "jose maria" names several players.
            candidates = leading.get(" ".join(parts), [])
            if len(candidates) == 1:
                hit = candidates[0]
            elif candidates:
                hit = at_club(candidates, team)

        if hit is None and parts:
            # Unique surnames only: anything shared is too risky to guess at,
            # unless the club and given name both confirm it.
            candidates = surnames.get(parts[-1], [])
            if len(candidates) == 1:
                hit = candidates[0]
            elif candidates:
                given = parts[0]
                named = [
                    c for c in candidates
                    if given in normalise_name(str(c["fifa_name"])).split()
                ]
                hit = at_club(named, team) if named else None

        if hit is not None:
            for col in columns:
                merged.at[idx, col] = hit[col]

    return merged


def _attach_per_season(
    players: pd.DataFrame, fifa: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Match each season's players against the edition current that season.

    This is the whole point of holding per-season exports. Matching every season
    against one export gives a player a single rating for his entire career:
    Aaron Cresswell was 69 in 2015/16 and still 69 in 2024/25, and a player who
    was seventeen and ordinary in 2016 carried his 2026 rating backwards. Both
    are future knowledge, and both inflate a backtest while being worthless in
    production.
    """
    fifa = fifa.copy()
    fifa["Season"] = fifa["Season"].astype(str)
    by_season = {season: group for season, group in fifa.groupby("Season")}

    pieces: list[pd.DataFrame] = []
    for season, group in players.groupby(players["Season"].astype(str), sort=False):
        edition = by_season.get(season)
        if edition is None:
            # Better an unrated season than one rated from the wrong year.
            logger.warning(
                "No FIFA export for season %s; %d players keep performance only",
                season, len(group),
            )
            piece = group.copy()
            for column in columns:
                piece[column] = pd.NA
            pieces.append(piece.reset_index(drop=True))
            continue

        edition = edition.reset_index(drop=True)
        piece = group.reset_index(drop=True).merge(
            edition[["_norm", *columns]], on="_norm", how="left"
        )
        piece = _match_loosely(piece, edition, columns)
        logger.info(
            "  %s: %.0f%% of %d player-seasons matched",
            season, piece["fifa_overall"].notna().mean() * 100, len(piece),
        )
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True)


def attach_fifa(players: pd.DataFrame, fifa: pd.DataFrame) -> pd.DataFrame:
    """Attach FIFA ratings and roles to performance rows, matched by name.

    Matched on normalised name rather than within a club, because EA's club
    field reflects one moment in a transfer window and our rows span whole
    seasons. Full names are distinctive enough that this is safe;
    club-constrained matching loses far more than it protects.

    When ``fifa`` carries a ``Season`` column, each season is matched against
    its own edition -- see :func:`_attach_per_season`. Otherwise every season is
    matched against the one export available, which is the legacy behaviour and
    applies one year's ratings to all of them.
    """
    out = players.copy()
    out["_norm"] = out["Player"].map(normalise_name)

    columns = [c for c in fifa.columns if c.startswith("fifa_") or c == "role"]

    if "Season" in fifa.columns and "Season" in out.columns:
        merged = _attach_per_season(out, fifa, columns)
    else:
        merged = out.merge(fifa[["_norm", *columns]], on="_norm", how="left")
        logger.info("FIFA exact match: %.0f%%", merged["fifa_overall"].notna().mean() * 100)
        merged = _match_loosely(merged, fifa, columns)

    logger.info(
        "FIFA match rate: %.0f%% of %d player-seasons",
        merged["fifa_overall"].notna().mean() * 100, len(merged),
    )

    # Players with no EA entry keep a coarse role from Understat, so they are
    # still rated rather than dropped.
    merged["role"] = merged["role"].fillna(merged["position_group"].map(FALLBACK_ROLE))
    return merged.drop(columns=["_norm"])
