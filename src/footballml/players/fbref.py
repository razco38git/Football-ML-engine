"""FBref player statistics: the defensive and goalkeeping metrics Understat lacks.

Understat carries no tackles, no interceptions and nothing at all for keepers,
which is why an Understat-only rating judged defenders on their attacking
involvement and refused to rate goalkeepers. FBref supplies both, through two
of the five stat tables soccerdata exposes:

- ``misc`` -- interceptions, tackles won, fouls, crosses. Actual defending,
  though only two of the four things a defender does; see ``MISC_COLUMNS``.
- ``keeper`` -- saves, save percentage, goals against, clean sheets, penalties.

This is deliberately used *instead of* FIFA ratings for goalkeepers. FIFA's
numbers are a scout's opinion; save percentage is what happened. SoFIFA is also
behind Cloudflare and returns 503 to any non-browser request, and working around
bot protection is not something to build a pipeline on.

The cost is real: FBref drives a headless Chrome and takes 10-15 seconds per
league-season against Understat's sub-second. It runs in the batch job only.

.. note::
    **The other three tables are a dead end -- do not retry them.** FBref's
    ``defense``, ``possession`` and ``passing`` pages carry everything missing
    from the rating: carries, progressive passes, take-ons, touches by zone,
    blocks, clearances, duels. They are fetched successfully and they arrive
    **with the values stripped** for players, though the squad totals on the
    same page are intact -- 48 populated cells against 1,236::

        <td class="right iz" data-stat="touches" ></td>
        <td class="right iz" data-stat="carries" ></td>

    Column headers, player names and minutes come through; every statistic is
    empty. Verified three ways: a fresh fetch through this same browser session
    (2.6 MB, table found inside a 1.2 MB HTML comment, all 528 players present,
    every stat cell blank), 195 cached copies spanning 2014/15 to 2026/27 (4-9%
    of cells populated), and against ``misc`` as a control on the same day
    (1056 of 1056 rows populated). A plain request is 403 regardless.

    A ``players/fbref_extended.py`` existed for this and was removed: it cost
    40 seconds on every ``build_players`` run and returned three usable columns
    -- ``tackles_won_per90`` and ``interceptions_per90``, which ``misc`` above
    already supplies, and ``fb_assists_per90``, which nothing read.

    Those metrics need event data. ``socceraction`` computes VAEP and xThreat
    from it for free, but only from Opta, StatsBomb or Wyscout, and StatsBomb's
    open data covers no current big-five season.
"""

from __future__ import annotations

import logging
import unicodedata

import pandas as pd

from footballml.ingest.understat import UNDERSTAT_LEAGUES

logger = logging.getLogger(__name__)

#: ``misc`` columns worth keeping, mapped to readable names.
#:
#: ``Recov`` and the three Aerial Duels columns are deliberately absent, having
#: been tried: FBref's ``misc`` table has twenty-one columns and carries
#: neither, in any of the thirteen cached seasons. ``data-stat="ball_recoveries"``
#: and ``data-stat="aerials_won"`` appear zero times in the HTML -- not blank,
#: as the gated tables are, but simply not served. A name here that the table
#: does not carry is skipped in silence, so ``recoveries`` was configured,
#: weighted in the centre-back score, and never built.
#:
#: Aerial duels are the loss that matters. They are the one freely available
#: measure of the thing a centre-back is actually judged on, and the one
#: measure that would not read Van Dijk -- 6th percentile for tackles won in
#: every Liverpool season, because he does not need to make them -- as a poor
#: defender. Nothing reachable here replaces them.
MISC_COLUMNS = {
    "Int": "interceptions",
    "TklW": "tackles_won",
    "Fls": "fouls",
    "Fld": "fouled",
    "Crs": "crosses",
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
    # Hyphens separate names; deleting them welds two apart into one token.
    # Understat spells Mbappé "Kylian Mbappe-Lottin", which became the single
    # token "mbappelottin" and could never reach EA's "Kylian Mbappé Lottin".
    # Apostrophes are left to be stripped: "N'Golo" is one name, not two.
    name = name.replace("-", " ").replace("‐", " ").replace("–", " ")
    transliterated = name.translate(TRANSLITERATIONS)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in ascii_only.lower() if c.isalnum() or c.isspace()).strip()


def _name_keys(name: str) -> list[str]:
    """Candidate keys for a name, **most specific first**.

    Sources disagree about middle names -- ``"Amad Diallo Traore"`` against
    ``"Amad Diallo"`` -- so a first-plus-last key catches those without
    resorting to a similarity threshold that could pair the wrong players.

    Order is part of the contract. Callers take the first key that hits, so the
    full name must be tried before ``"a onana"``, which cannot tell André from
    Amadou. Returning a set left that order to string hashing, which Python
    randomises per process -- the same data matched differently run to run.
    """
    norm = normalise_name(name)
    if not norm:
        return []
    parts = norm.split()
    keys = [norm]
    if len(parts) > 2:
        keys.append(f"{parts[0]} {parts[-1]}")
    if len(parts) >= 2:
        keys.append(f"{parts[0][0]} {parts[-1]}")  # "a robertson"
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

        # A loose key can name two different players: "a onana" reaches both
        # André and Amadou, who were in the Premier League together. Taking the
        # first row seen resolved that by *row order*, which made the build
        # irreproducible -- two runs over the same data gave 197 players a
        # different rating and moved 312 team strengths, because a goalkeeper
        # kept inheriting a midfielder's tackles. An ambiguous key is not a
        # match, so drop it and let the player fall through to the next key or
        # go unmatched.
        lookup: dict[tuple, dict] = {}
        ambiguous: set[tuple] = set()
        for _, row in right.iterrows():
            group = tuple(row[c] for c in keys)
            for key in _name_keys(str(row["Player"])):
                full = (*group, key)
                seen = lookup.get(full)
                if seen is None:
                    lookup[full] = row
                elif seen["_norm"] != row["_norm"]:
                    ambiguous.add(full)
        for key in ambiguous:
            del lookup[key]
        if ambiguous:
            logger.info(
                "Ambiguous on %s: %d name keys matched more than one player, left unmatched",
                "+".join(keys), len(ambiguous),
            )

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
        # One season at a time. soccerdata fetches every requested league-season
        # in a single call, so a single unparseable page -- older seasons use a
        # different layout and raise "not enough values to unpack" -- aborts the
        # entire run. Per-season means a bad page costs that season, not the
        # eleven that parsed fine.
        seasonal: list[pd.DataFrame] = []
        for season in seasons or [None]:
            reader = fb if season is None else sd.FBref(leagues=names, seasons=season)
            try:
                raw = reader.read_player_season_stats(stat_type).reset_index()
            except Exception as exc:  # noqa: BLE001 - keep going on a bad page
                logger.warning("FBref %s %s unavailable: %s", stat_type, season, exc)
                continue

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
            seasonal.append(out)

        if not seasonal:
            logger.warning("FBref %s: no seasons could be read", stat_type)
            frames.append(pd.DataFrame(columns=["League", "Season", "Team", "Player"]))
            continue

        combined = pd.concat(seasonal, ignore_index=True)
        frames.append(combined)
        logger.info(
            "FBref %s: %d rows across %d season(s)",
            stat_type, len(combined), combined["Season"].nunique(),
        )

    misc, keeper = frames
    return misc.merge(
        keeper, on=["League", "Season", "Team", "Player"], how="outer", suffixes=("", "_gk")
    )
