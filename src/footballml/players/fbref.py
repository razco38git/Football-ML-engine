"""FBref player statistics: the defensive and goalkeeping metrics Understat lacks.

Understat carries no tackles, no interceptions and nothing at all for keepers,
which is why an Understat-only rating judged defenders on their attacking
involvement and refused to rate goalkeepers. FBref supplies both, through two
of the five stat tables soccerdata exposes:

- ``misc`` -- interceptions, tackles won, fouls, crosses. Actual defending.
- ``keeper`` -- saves, save percentage, goals against, clean sheets, penalties.

This is deliberately used *instead of* FIFA ratings for goalkeepers. FIFA's
numbers are a scout's opinion; save percentage is what happened. SoFIFA is also
behind Cloudflare and returns 503 to any non-browser request, and working around
bot protection is not something to build a pipeline on.

The cost is real: FBref drives a headless Chrome and takes 10-15 seconds per
league-season against Understat's sub-second. It runs in the batch job only.
"""

from __future__ import annotations

import logging
import unicodedata

import pandas as pd

from footballml.ingest.understat import UNDERSTAT_LEAGUES

logger = logging.getLogger(__name__)

#: ``misc`` columns worth keeping, mapped to readable names.
MISC_COLUMNS = {
    "Int": "interceptions",
    "TklW": "tackles_won",
    "Fls": "fouls",
    "Fld": "fouled",
    "Crs": "crosses",
    "Recov": "recoveries",
}

#: ``keeper`` columns worth keeping.
KEEPER_COLUMNS = {
    "GA": "goals_against",
    "GA90": "goals_against_per90",
    "SoTA": "shots_on_target_against",
    "Saves": "saves",
    "Save%": "save_pct",
    "CS": "clean_sheets",
    "CS%": "clean_sheet_pct",
    "PKsv": "penalties_saved",
    "PKatt": "penalties_faced",
}


#: Letters Unicode decomposition cannot handle.
#:
#: NFKD splits ``é`` into ``e`` plus a combining accent, which strips cleanly.
#: But ``Đ``, ``ł`` and ``ø`` are distinct letters, not decorated ones, so they
#: survive decomposition and never match their plain spellings. That is why
#: ``"Đorđe Petrović"`` failed to match ``"Djordje Petrovic"``.
TRANSLITERATIONS = str.maketrans({
    "Đ": "D", "đ": "d", "Ð": "D", "ð": "d",
    "Ł": "L", "ł": "l",
    "Ø": "O", "ø": "o",
    "Æ": "AE", "æ": "ae",
    "Œ": "OE", "œ": "oe",
    "ß": "ss",
    "Þ": "Th", "þ": "th",
    "Ħ": "H", "ħ": "h",
    "Ŀ": "L", "ŀ": "l",
    "ı": "i", "İ": "I",
})


def normalise_name(name: str) -> str:
    """Strip accents, punctuation and case so spellings can be compared.

    Two passes are needed. Explicit transliteration handles letters that are
    their own characters rather than decorated ones (see
    :data:`TRANSLITERATIONS`); NFKD decomposition then strips ordinary accents.
    Running only the second, as this did originally, silently failed on every
    Slavic, Polish and Scandinavian name.
    """
    if not isinstance(name, str):
        return ""
    transliterated = name.translate(TRANSLITERATIONS)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in ascii_only.lower() if c.isalnum() or c.isspace()).strip()


def _name_keys(name: str) -> set[str]:
    """Candidate keys for a name: the whole thing, and first+last.

    Sources disagree about middle names -- ``"Amad Diallo Traore"`` against
    ``"Amad Diallo"`` -- so a first-plus-last key catches those without
    resorting to a similarity threshold that could pair the wrong players.
    """
    norm = normalise_name(name)
    if not norm:
        return set()
    parts = norm.split()
    keys = {norm}
    if len(parts) > 2:
        keys.add(f"{parts[0]} {parts[-1]}")
    if len(parts) >= 2:
        keys.add(f"{parts[0][0]} {parts[-1]}")  # "a robertson"
    return keys


def match_players(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: list[str],
    fallback_on: list[str] | None = None,
) -> pd.DataFrame:
    """Join two player frames on name, within the groups given by ``on``.

    Three passes, each looser than the last but all constrained enough that a
    wrong pairing is implausible:

    1. Exact normalised name within ``on`` (league, season and team).
    2. Looser name keys -- first-plus-last, initial-plus-last -- still within
       ``on``. This catches ``"Amad Diallo Traore"`` against ``"Amad Diallo"``.
    3. The same keys within ``fallback_on`` (league and season only).

    The third pass exists because the two sources disagree about *club* names as
    well as player names -- FBref's ``Manchester Utd`` against Understat's
    ``Manchester United`` -- which silently cost 30% of matches when team was a
    hard requirement. Player names are close to unique inside a league-season,
    so dropping team is safe where keeping it loses real data.

    Returns:
        ``left`` with ``right``'s extra columns attached where matched.
    """
    left = left.copy()
    right = right.copy()
    left["_norm"] = left["Player"].map(normalise_name)
    right["_norm"] = right["Player"].map(normalise_name)

    # Only the new metric columns are carried across. Merging `right` whole
    # brings its `Player` column along as `Player_fb`, and a second call then
    # fails outright because that suffixed name already exists. Overlapping
    # metric names are dropped too -- `Int` and `TklW` appear in both the misc
    # and defense tables with identical values, so first source wins.
    extra = [
        c
        for c in right.columns
        if c not in {*on, "Player", "_norm"} and c not in left.columns
    ]
    dropped = [
        c
        for c in right.columns
        if c not in {*on, "Player", "_norm"} and c in left.columns
    ]
    if dropped:
        logger.debug("Already present, not re-merged: %s", dropped)

    merged = left.merge(right[[*on, "_norm", *extra]], on=[*on, "_norm"], how="left")
    if not extra:
        return merged.drop(columns=["_norm"], errors="ignore")

    def _rate() -> float:
        return float(merged[extra[0]].notna().mean())

    logger.info("Name match, exact within %s: %.0f%%", "+".join(on), _rate() * 100)

    for keys in ([on] + ([fallback_on] if fallback_on else [])):
        unmatched = merged[extra[0]].isna()
        if not unmatched.any():
            break

        lookup: dict[tuple, dict] = {}
        for _, row in right.iterrows():
            group = tuple(row[c] for c in keys)
            for key in _name_keys(str(row["Player"])):
                lookup.setdefault((*group, key), row)

        for idx in merged.index[unmatched]:
            group = tuple(merged.at[idx, c] for c in keys)
            for key in _name_keys(str(merged.at[idx, "Player"])):
                hit = lookup.get((*group, key))
                if hit is not None:
                    for col in extra:
                        merged.at[idx, col] = hit[col]
                    break

        logger.info("After fuzzy pass on %s: %.0f%%", "+".join(keys), _rate() * 100)

    return merged.drop(columns=["_norm"], errors="ignore")


def _flatten_columns(columns: pd.Index) -> list[str]:
    """Flatten FBref's two-level headers, keeping names unique.

    The same leaf name appears under different groups -- ``Save%`` sits under
    both ``Performance`` and ``Penalty Kicks`` -- and collapsing them blindly
    produces duplicate columns, after which ``df[name]`` silently returns a
    DataFrame and every downstream numeric call fails.

    First occurrence keeps the plain name (that is the one the stat is usually
    known by); later collisions are qualified with their group.
    """
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            group, leaf = (col + ("",))[:2]
            name = leaf or group
            if name in seen and group and leaf:
                name = f"{group}_{leaf}"
        else:
            name = str(col)
        # Still colliding: disambiguate positionally rather than lose a column.
        base, suffix = name, 2
        while name in seen:
            name = f"{base}_{suffix}"
            suffix += 1
        seen.add(name)
        out.append(name)
    return out


def fetch_fbref_stats(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> pd.DataFrame:
    """Fetch and combine FBref ``misc`` and ``keeper`` per-90 stats.

    Returns:
        ``League``, ``Season``, ``Team``, ``Player`` plus per-90 defensive and
        goalkeeping rates.
    """
    import soccerdata as sd  # imported lazily: pulls a heavy browser stack

    codes = leagues or sorted(UNDERSTAT_LEAGUES)
    names = [UNDERSTAT_LEAGUES[c] for c in codes]
    inverse = {v: k for k, v in UNDERSTAT_LEAGUES.items()}
    fb = sd.FBref(leagues=names, seasons=seasons)

    frames = []
    for stat_type, wanted in (("misc", MISC_COLUMNS), ("keeper", KEEPER_COLUMNS)):
        raw = fb.read_player_season_stats(stat_type).reset_index()
        raw.columns = _flatten_columns(raw.columns)

        out = pd.DataFrame(
            {
                "League": raw["league"].map(inverse),
                "Season": raw["season"].astype(str),
                "Team": raw["team"],
                "Player": raw["player"],
            }
        )
        nineties = pd.to_numeric(raw.get("90s"), errors="coerce")
        for source, name in wanted.items():
            if source not in raw.columns:
                continue
            values = pd.to_numeric(raw[source], errors="coerce")
            # Percentages and rates are already normalised; counts are not.
            if source.endswith("%") or source.endswith("90"):
                out[name] = values
            else:
                out[f"{name}_per90"] = values.div(nineties).where(nineties > 0)
                out[name] = values
        frames.append(out)
        logger.info("FBref %s: %d rows", stat_type, len(out))

    misc, keeper = frames
    return misc.merge(
        keeper, on=["League", "Season", "Team", "Player"], how="outer", suffixes=("", "_gk")
    )
