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
    # The form a player is actually known by, where the export carries one
    # alongside the legal name. EA lists Vitinha as "Vitor Machado Ferreira"
    # and Casemiro as "Carlos Henrique Venancio Casimiro" -- no shared token
    # with the name every other source uses, so this is the only way to match
    # them at all.
    "alt_name": ("alt_name", "short_name", "common_name", "display_name"),
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
    # Keepers have no pace or shooting; EA rates them on these instead. Without
    # them a goalkeeper carries no attributes at all and cannot be compared
    # with anyone, which is why the similarity page needs them.
    "gk_diving": ("gk_diving", "goalkeeping_diving", "GK Diving"),
    "gk_handling": ("gk_handling", "goalkeeping_handling", "GK Handling"),
    "gk_kicking": ("gk_kicking", "goalkeeping_kicking", "GK Kicking"),
    "gk_positioning": ("gk_positioning", "goalkeeping_positioning", "GK Positioning"),
    "gk_reflexes": ("gk_reflexes", "goalkeeping_reflexes", "GK Reflexes"),
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
    # A holder and a number eight are as different as a centre-back and a
    # full-back, and splitting those two is the precedent. Pooled, 2,028
    # CDM-primary player-seasons were ranked for chance creation against 3,081
    # CM-primary ones and lost: they averaged 65.3 against 70.5 and held 18%
    # of each season's top fifty while being 40% of the pool. Given their own
    # pool -- same weights, only a fair comparison -- the gap closes to 68.5
    # against 68.6 and their share of the top fifty reaches 38%. Busquets
    # gains 2-5 a season, Casemiro 1-7, Rodri 8 in 2020/21.
    #
    # The *weights* were tried too and are deliberately unchanged. Shifting
    # them toward defending and build-up made the midfield line predict next
    # season's points worse, 0.646 to 0.634, so the pool was the problem and
    # the weights were not.
    "CDM": "DM", "LDM": "DM", "RDM": "DM",
    "CM": "MID", "LCM": "MID", "RCM": "MID",
    "CAM": "AMW", "LM": "AMW", "RM": "AMW", "LW": "AMW", "RW": "AMW",
    "ST": "FWD", "CF": "FWD", "LS": "FWD", "RS": "FWD", "LF": "FWD", "RF": "FWD",
}

#: Display names for the roles.
ROLE_LABELS = {
    "GK": "Goalkeeper",
    "CB": "Centre back",
    "FB": "Full back",
    "DM": "Defensive midfielder",
    "MID": "Midfielder",
    "AMW": "Attacking midfielder / winger",
    "FWD": "Forward",
}

#: Understat's coarse groups, used when a player has no EA entry.
FALLBACK_ROLE = {"GK": "GK", "D": "CB", "M": "MID", "F": "FWD"}

#: EA roles each Understat group may plausibly carry. A match outside this is
#: taken as evidence of the wrong *person*, not of a versatile one.
#:
#: Deliberately generous, because the two sources disagree about position all
#: the time and almost none of it means anything. Understat files wing-backs
#: as "D" where EA says LM or RW -- 1,073 player-seasons, nearly all correct --
#: and calls plenty of forwards "D" or midfielders "F". Only two boundaries
#: survive as real:
#:
#: **Goalkeeper, in both directions.** Nobody is both, and it is the most
#: damaging mismatch there is: it takes a club's keeper line away and inflates
#: an outfield one. Aston Villa's Emiliano Martínez was matched to Emiliano
#: Martínez Toranza of Club Nacional and spent two seasons, 6,500 minutes, as
#: a *midfielder*. Atalanta's Éderson took Manchester City's Ederson.
#:
#: **Forward against centre back.** 24 player-seasons, every one a different
#: man: Espanyol's Sergio García held Sergio Ramos' 87, 90 and 91 across three
#: seasons, and Anthony Martial was filed as Johan Martial of Troyes.
#:
#: Forward-to-midfield is *not* here and must not be added: false nines and
#: withdrawn forwards cross it constantly.
#: Every outfield role must appear in every outfield group except where a
#: boundary above says otherwise. Leaving one out rejects wholesale rather than
#: narrowly: splitting `DM` out of `MID` and forgetting to list it here dropped
#: the FIFA match rate from 92% to 87% in one run, because every CDM-primary
#: entry in the database suddenly contradicted every player.
_OUTFIELD = {"CB", "FB", "DM", "MID", "AMW", "FWD"}
COMPATIBLE_ROLES = {
    "GK": {"GK"},
    "D": _OUTFIELD,
    "M": _OUTFIELD,
    "F": _OUTFIELD - {"CB"},
}


def contradicts_position(group: object, role: object) -> bool:
    """Whether an EA entry's role rules it out as this player.

    Unknown on either side is not a contradiction -- it is an absence.
    """
    if not isinstance(group, str) or not isinstance(role, str):
        return False
    allowed = COMPATIBLE_ROLES.get(group)
    return allowed is not None and role not in allowed


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
                  "dribbling", "defending", "physical",
                  "gk_diving", "gk_handling", "gk_kicking", "gk_positioning",
                  "gk_reflexes"):
        column = _find_column(df, field)
        if column is not None:
            values = df[column]
            out[f"fifa_{field}"] = (
                values.astype(str) if field in {"club", "positions"}
                else pd.to_numeric(values, errors="coerce")
            )

    alt_col = _find_column(df, "alt_name")
    if alt_col is not None:
        alt = df[alt_col].astype("string").str.strip().map(
            lambda v: _strip_foreign_script(v) if isinstance(v, str) else v
        )
        # Only worth carrying when it differs from the legal name.
        out["fifa_alt_name"] = alt.where(alt.fillna("") != out["fifa_name"])

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
    #: First initial plus surname -- "o watkins". EA's alternate name is often
    #: exactly this shape ("O. Watkins"), and it is the only key that bridges a
    #: diminutive to a legal name: ours says "Ollie", EA says "Oliver George
    #: Arthur Watkins". It also absorbs accent differences for free, since
    #: "Djordje" and "Đorđe" both normalise to "d".
    initials: dict[str, list[pd.Series]] = {}
    #: Token set per EA entry, for the containment tier.
    token_sets: list[tuple[frozenset[str], pd.Series]] = []

    # A row must appear at most once per key. Indexing two names for the same
    # player can otherwise put him in a bucket twice -- "Fabian Ruiz Pena" and
    # "Fabian Ruiz" both end in "ruiz" -- which reads as two candidates and
    # defeats the uniqueness guards the tiers below rely on.
    seen: dict[str, dict[str, set[object]]] = {
        "strong": {}, "leading": {}, "surnames": {}, "initials": {},
    }

    def add(
        bucket: dict[str, list[pd.Series]], which: str, key: str, row: pd.Series
    ) -> None:
        marked = seen[which].setdefault(key, set())
        if row.name in marked:
            return
        marked.add(row.name)
        bucket.setdefault(key, []).append(row)

    has_alt = "fifa_alt_name" in fifa.columns
    for _, row in fifa.iterrows():
        # Index the legal name *and* the familiar one. Without the second,
        # "Vitinha" can never reach "Vitor Machado Ferreira": they share no
        # token, so every key tier below is looking for something that is not
        # there. 63 of 2025/26's 176 unmatched players were single-word names.
        variants = [str(row["fifa_name"])]
        if has_alt and isinstance(row.get("fifa_alt_name"), str):
            variants.append(row["fifa_alt_name"])

        tokens_added: set[frozenset[str]] = set()
        for variant in variants:
            for key in keys_for(variant):
                add(strong, "strong", key, row)
            parts = normalise_name(variant).split()
            if len(parts) > 2:
                add(leading, "leading", f"{parts[0]} {parts[1]}", row)
            if parts:
                add(surnames, "surnames", parts[-1], row)
                if len(parts) > 1 and parts[0]:
                    add(initials, "initials", f"{parts[0][0]} {parts[-1]}", row)
                token_set = frozenset(parts)
                if token_set not in tokens_added:
                    tokens_added.add(token_set)
                    token_sets.append((token_set, row))

    def _tokens_of(row: pd.Series) -> set[str]:
        """Every token of both names EA carries for a player.

        The index is built from the legal name *and* the familiar one, so the
        confirmation has to read both or it rejects the very rows the familiar
        name put in the bucket. EA files Girona's Alex Granell as "Alejandro
        Granell Nogue" with "Alex Granell" alongside: the surname key comes
        from the second, and checking only the first threw him out -- along
        with Javi Puado, Fede San Emeterio, Tosin Adarabioyo and Bote Baku,
        all correct matches losing to a diminutive.
        """
        tokens: set[str] = set()
        for column in ("fifa_name", "fifa_alt_name"):
            value = row.get(column)
            if isinstance(value, str):
                tokens.update(normalise_name(value).split())
        return tokens

    def at_club(candidates: list[pd.Series], team: object) -> pd.Series | None:
        """The single candidate at the row's club, if there is exactly one."""
        club = club_names.get(str(team))
        if club is None or "fifa_club" not in fifa.columns:
            return None
        here = [c for c in candidates if c["fifa_club"] == club]
        return here[0] if len(here) == 1 else None

    has_team = "Team" in merged.columns
    has_group = "position_group" in merged.columns
    for idx in merged.index[unmatched]:
        name = str(merged.at[idx, "Player"])
        team = merged.at[idx, "Team"] if has_team else None
        parts = normalise_name(name).split()
        hit = None

        # Entries this player cannot be are removed before any tier sees them,
        # not after one picks: a bucket holding the right man and a keeper is
        # ambiguous only until the keeper is dropped, and the uniqueness guards
        # below should get to count the candidates that are actually possible.
        group = merged.at[idx, "position_group"] if has_group else None

        def possible(rows: list[pd.Series], group: object = group) -> list[pd.Series]:
            return [r for r in rows if not contradicts_position(group, r.get("role"))]

        # Strong keys may map to several players. Prefer the one at his club;
        # otherwise keep the highest-rated, which is overwhelmingly the one a
        # top-five-league dataset means.
        for key in keys_for(name):
            if key in strong:
                candidates = possible(strong[key])
                if not candidates:
                    continue
                hit = candidates[0]
                if len(candidates) > 1:
                    local = at_club(candidates, team)
                    hit = local if local is not None else hit
                break

        if hit is None:
            # A two-token name against an EA entry carrying extra family names.
            # Unique leading pairs only: "jose maria" names several players.
            candidates = possible(leading.get(" ".join(parts), []))
            if len(candidates) == 1:
                hit = candidates[0]
            elif candidates:
                hit = at_club(candidates, team)

        if hit is None and len(parts) >= 2:
            # Extra given names *and* extra family names at once, which no
            # ordered key catches. EA stores Moises Caicedo as "Moises Isaac
            # Caicedo Corozo": first-plus-last gives "moises corozo", the
            # leading pair "moises isaac", and the surname tier looks for
            # "corozo". Every token we have is present though, so fall back to
            # containment -- guarded, like the tiers above, by being unique or
            # resolved by club.
            # Collapse by identity first: one player indexed under both his
            # legal and familiar name ("Fabian Ruiz Pena" and "Fabian Ruiz")
            # contains the query twice and would otherwise look ambiguous.
            wanted = frozenset(parts)
            by_row = {
                row.name: row for tokens, row in token_sets if wanted <= tokens
            }
            candidates = possible(list(by_row.values()))
            if len(candidates) == 1:
                hit = candidates[0]
            elif candidates:
                hit = at_club(candidates, team)

        if hit is None and len(parts) > 1:
            # Containment the other way round: EA's name inside *ours*.
            #
            # Every tier above asks whether our tokens are present in EA's, and
            # none of them asks the reverse, so an EA entry that is a shorter
            # form of our name is unreachable however obvious it looks.
            # Understat writes Kylian Mbappe as "Mbappe-Lottin" and EA's FC24
            # and FC25 exports simply say "Kylian Mbappe" -- the right player,
            # at the right club, with the right position, and not one key
            # reaches him:
            #
            #     ours   {kylian, mbappe, lottin}
            #     theirs {kylian, mbappe}        ours <= theirs  False
            #                                    theirs <= ours  True
            #
            # So the best player in the league carried no EA rating in two
            # seasons out of three. Fabian Ruiz fails the same way.
            #
            # Two tokens minimum, because a one-token EA name inside ours is a
            # much weaker claim -- a bare "Silva" sits inside "Thiago Silva"
            # and inside two dozen others. Those are already served by the
            # surname tier, which has the guards for them.
            ours = frozenset(parts)
            by_row = {
                row.name: row
                for tokens, row in token_sets
                if len(tokens) > 1 and tokens <= ours
            }
            candidates = possible(list(by_row.values()))
            if len(candidates) == 1:
                hit = candidates[0]
            elif candidates:
                hit = at_club(candidates, team)

        if hit is None and len(parts) > 1 and parts[0]:
            # Diminutives and accent differences: "Ollie Watkins" against
            # "Oliver George Arthur Watkins", "Djordje" against "Đorđe". Safe
            # where a club-only fallback is not, because the initial still
            # separates two different men who share a surname and a dressing
            # room: Pedro Lopes is "p lopes", Hugo Miguel Lopes "h lopes".
            #
            # The club is *required* here, not just a tiebreak. An initial and
            # a surname are weak evidence on their own, and being the only
            # candidate proves nothing when the right man is filed under a
            # different surname entirely: EA stores Barcelona's Alex Balde as
            # "Alejandro Balde Martinez", whose last token is the maternal
            # surname, so the only player keyed "a balde" was Aliou Balde of
            # St. Gallen. Taken unopposed, that swapped an 83-rated starter for
            # a 66-rated stranger and dropped him to 54.
            candidates = possible(initials.get(f"{parts[0][0]} {parts[-1]}", []))
            hit = at_club(candidates, team) if candidates else None

        if hit is None and parts:
            # Unique surnames only: anything shared is too risky to guess at,
            # unless the club and given name both confirm it.
            candidates = possible(surnames.get(parts[-1], []))
            given = parts[0]
            named = [c for c in candidates if given in _tokens_of(c)]
            if len(candidates) == 1:
                # Sole holder of the surname, which proves nothing on its own:
                # the right man is often filed under a different surname
                # entirely. Understat writes Kylian Mbappe as "Mbappe-Lottin",
                # EA's 2024/25 export does not, and the only entry keyed
                # "lottin" was Albert-Nicolas Lottin of CD Castellon. Taken
                # unopposed, a 91-rated striker became a 65-rated defensive
                # midfielder -- which also moved him out of Real Madrid's
                # attack line and into its midfield.
                #
                # The given name or the club has to confirm it; either alone
                # is enough. "San Jose" reaches "Mikel San Jose Dominguez" on
                # the name, and a legal name sharing no given name with ours
                # still lands when the club agrees.
                #
                # A third confirmation is needed because EA often stores
                # nothing but the surname -- "Alena", "Reguilon", "Terrats",
                # "Granell" -- and a given name that is not there can never
                # agree with ours. Those players are also the ones most likely
                # to have moved, so the club cannot rescue them either. Where
                # EA's name is a strict *subset* of ours it is taken: it adds
                # no claim we cannot already see. "Alena" within "Carles
                # Alena" is accepted; "Ridle Baku" against "Bote Baku",
                # "Edgar Guerra" against "Javier Guerra" and "Albert-Nicolas
                # Lottin" against "Kylian Mbappe-Lottin" all introduce a given
                # name of their own and are refused.
                contained = _tokens_of(candidates[0]) <= set(parts)
                confirmed = (
                    bool(named) or contained or at_club(candidates, team) is not None
                )
                hit = candidates[0] if confirmed else None
            elif candidates:
                hit = at_club(named, team) if named else None
                # Deliberately not relaxed to "unique at the club" when the
                # given name does not match. That would match Ollie Watkins to
                # "Oliver George Arthur Watkins" -- but it would equally match
                # Pedro Lopes to Hugo Miguel Lopes, a different man at the same
                # club, and the two cases are indistinguishable by club alone.
                # Diminutives (Ollie/Oliver, Mat/Mathew) therefore stay
                # unmatched: ~5 players, against silently wrong ratings.

        if hit is not None:
            for col in columns:
                merged.at[idx, col] = hit[col]

    return merged


def _match_across_editions(
    merged: pd.DataFrame, fifa: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Carry a confirmed match sideways into editions that renamed the player.

    EA does not spell a player the same way twice. The 2025/26 export stores
    legal names with the familiar one alongside -- ``"Vinicius José Paixão de
    Oliveira Junior"`` / ``"Vini Jr."`` -- while 2024/25 and 2026/27 store only
    ``"Vini Jr."``. Every tier in :func:`_match_loosely` works on the tokens of
    one name against the tokens of another, and ``"Vini Jr."`` shares none with
    Understat's ``"Vinícius Júnior"``: no key reaches it, so the best player at
    Real Madrid carried no EA rating in two of the three seasons. ``"Fermín"``
    against ``"Fermín López"`` fails for the opposite reason -- EA's name is a
    strict *subset*, and containment only looks for supersets.

    Both are solvable without guessing, because the match already exists in
    another season. A player who matched anywhere gives us the EA names he goes
    by; those names are then looked for in the editions where he did not match.
    This is evidence rather than string similarity, which is the same standard
    the tiers above hold to.

    Two guards keep it honest, and both are needed. A name variant claimed by
    more than one performance-source player is discarded -- it identifies
    neither. And the club is **required**, not merely a tiebreak, which is the
    opposite of the rule everywhere else in this module.

    The reason is that the performance source's name is the only identity a
    player has here, and two men can share one. Understat calls both Luis
    Alberto Suárez Díaz and Luis Javier Suárez Charris "Luis Suárez", so the
    Colombian's EA name is in the Uruguayan's variant set and vice versa.
    Taken on the name alone this tier gave Atlético's 2020/21 Suárez the
    Colombian's 75. It also handed Arsenal's Emiliano Martínez an unrelated
    Martínez rated 61. A variant is weak evidence by construction -- it is a
    name EA chose in a *different* edition -- so it only counts when the
    edition also places that player at the club he actually played for. Where
    the club cannot be resolved, the row stays unmatched.
    """
    unmatched = merged["fifa_overall"].isna()
    if not unmatched.any() or "Season" not in merged.columns:
        return merged

    variant_columns = [c for c in ("fifa_name", "fifa_alt_name") if c in merged.columns]
    if not variant_columns or "fifa_club" not in fifa.columns:
        return merged

    # --- the names each player has already been matched under ----------------
    owners: dict[str, set[str]] = {}
    matched = merged[~unmatched]
    for column in variant_columns:
        for player, value in zip(matched["Player"], matched[column], strict=True):
            if isinstance(value, str) and (key := normalise_name(value)):
                owners.setdefault(key, set()).add(str(player))

    variants: dict[str, set[str]] = {}
    for key, claimants in owners.items():
        if len(claimants) == 1:
            variants.setdefault(next(iter(claimants)), set()).add(key)
    if not variants:
        return merged

    # --- every edition's entries, indexed by the same keys -------------------
    editions = fifa.copy()
    editions["Season"] = editions["Season"].astype(str)
    # Positions, not rows: materialising ~220,000 Series to build an index that
    # is then asked about a few hundred of them is the slow way round.
    index: dict[tuple[str, str], set[int]] = {}
    seasons = editions["Season"].to_numpy()
    for column in variant_columns:
        if column not in editions.columns:
            continue
        for position, value in enumerate(editions[column].to_numpy()):
            if isinstance(value, str) and (key := normalise_name(value)):
                index.setdefault((seasons[position], key), set()).add(position)

    edition_clubs = editions["fifa_club"].to_numpy()
    has_team = "Team" in merged.columns

    # Per edition, not pooled. EA renames clubs between editions as licences
    # come and go, so a majority taken over twelve seasons answers for none of
    # them: "Real Madrid" is "Real Madrid CF" up to FC 23 and plain "Real
    # Madrid" after, and the pooled majority rejected Vinícius at his own club
    # in exactly the three editions this tier exists to repair. The pooled map
    # stays as a fallback for a season with too few matched players to decide.
    pooled = _club_names(merged)
    by_season = {
        str(season): {**pooled, **_club_names(group)}
        for season, group in merged.groupby(merged["Season"].astype(str), sort=False)
    }
    repaired = 0

    for idx in merged.index[unmatched]:
        player = str(merged.at[idx, "Player"])
        known = variants.get(player)
        if not known:
            continue
        season = str(merged.at[idx, "Season"])

        positions: set[int] = set()
        for key in known:
            positions |= index.get((season, key), set())
        if not positions:
            continue

        club = (
            by_season.get(season, pooled).get(str(merged.at[idx, "Team"]))
            if has_team
            else None
        )
        if club is None:
            continue
        here = [
            editions.iloc[position]
            for position in sorted(positions)
            if edition_clubs[position] == club
        ]
        if len(here) != 1:
            continue
        hit = here[0]
        # The same check the tiers apply. This tier fires on a name EA used in
        # a *different* edition, which is the weakest evidence in the module,
        # so it must not be the one route that skips it.
        if contradicts_position(merged.at[idx, "position_group"]
                                if "position_group" in merged.columns else None,
                                hit.get("role")):
            continue

        for column in columns:
            merged.at[idx, column] = hit[column]
        repaired += 1

    if repaired:
        logger.info(
            "Cross-edition names recovered %d player-season(s) EA had renamed",
            repaired,
        )
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

        # The exact pass needs the same check as the loose tiers, because two
        # different men genuinely share a normalised name: Atalanta's midfielder
        # Éderson and Manchester City's keeper Ederson both reduce to
        # "ederson", and the merge handed the midfielder an 88-rated
        # goalkeeper. Cleared here rather than filtered, so the tiers below get
        # a chance to find the right man instead.
        if "position_group" in piece.columns and "role" in piece.columns:
            wrong = [
                contradicts_position(g, r)
                for g, r in zip(piece["position_group"], piece["role"], strict=True)
            ]
            if any(wrong):
                piece.loc[wrong, columns] = pd.NA
                logger.info(
                    "  %s: %d exact name match(es) cleared, the EA entry plays "
                    "a position the player cannot", season, sum(wrong),
                )
        # Twice, deliberately. `_match_loosely` learns what EA calls each club
        # from the matches made so far, and on the first call that is the
        # exact-name pass alone -- the pass Spanish squads fail, because EA
        # stores their legal names with extra family names. Measured across
        # the editions, 7 to 11 clubs a season had no resolvable EA name, and
        # they were Girona, Espanyol, Villarreal, Athletic Club, Real
        # Valladolid. So the club check was blind for a third of La Liga,
        # which is exactly where the weakest keys lean on it hardest.
        #
        # The second call rebuilds that map from everything the first matched
        # and retries only the rows still empty. The name-only tiers return
        # the same answer twice and cost a few seconds; the two club-dependent
        # tiers get a map worth consulting.
        for _ in range(2):
            piece = _match_loosely(piece, edition, columns)
        logger.info(
            "  %s: %.0f%% of %d player-seasons matched",
            season, piece["fifa_overall"].notna().mean() * 100, len(piece),
        )
        pieces.append(piece)

    # Only now, with every season matched, is there a body of confirmed names
    # to carry into the editions that spell a player differently.
    return _match_across_editions(
        pd.concat(pieces, ignore_index=True), fifa, columns
    )


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
    #
    # `season_position` is tried first and `position_group` only as a backstop.
    # The latter is a *career* position (see `_career_position` in ingest.py):
    # right for describing a player, wrong for "what did he play this season".
    # Moisés Caicedo has no EA entry, a season position of M and a career
    # position of D, so the career value made him a centre-back rated 90 and
    # inflated Chelsea's defence rating.
    fallback = merged["position_group"]
    if "season_position" in merged.columns:
        fallback = merged["season_position"].fillna(fallback)
    merged["role"] = merged["role"].fillna(fallback.map(FALLBACK_ROLE))
    return merged.drop(columns=["_norm"])
