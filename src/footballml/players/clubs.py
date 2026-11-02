"""Which club does a player-season belong to, when Understat names several?

Understat's league payload carries one row per player per season, and its
``team_title`` is a **comma-joined list** of every club the player turned out
for -- ``"Napoli,Verona"``, ``"Genoa,Juventus"``. soccerdata reduces that to one
name by taking the first (``understat.py``: *"pick first team if multiple teams
are listed"*), and the list arrives alphabetically sorted, so the club a
player-season is filed under is decided by the spelling of his employers.

Measured across the twelve cached seasons, 927 of 34,856 player-seasons (2.7%)
carry such a list. For most of them the alphabetical pick is merely *coarse*:
the player really did play for both clubs, Understat's row is one aggregate
across the two, and filing it under either is defensible. This module does not
touch those.

It exists for the case where the pick is not coarse but **false** -- the player
never appeared for the club he was filed under:

    Understat, Ligue 1 2026/27, ``team_title``
        ``"Paris Saint Germain,Rennes"``  ... 31 players

Thirty-one is not a transfer window. Every one of those 31 is a participant in
one fixture -- Paris Saint-Germain 2-2 Rennes, 23 August 2026, Understat match
31948 -- and Understat tags all of them with both clubs. The per-match data for
that same fixture is correct (16 players a side, each at his own club); only the
season-level ``team_title`` is wrong. Because ``Paris`` sorts before ``Rennes``,
fifteen Rennes players were published as Paris Saint-Germain players: Charlie
Cresswell, Valentin Rongier, Brice Samba, Przemyslaw Frankowski, Adrien
Thomasson, Esteban Lepaul and nine more. That left 39 player-seasons under
``Paris Saint Germain`` and six under ``Rennes``, against 21 to 27 for every
other Ligue 1 club, and it put both clubs' squad strength out.

**FBref settles it, and costs nothing.** `fetch_fbref_stats` is already fetched
for the defensive and goalkeeping metrics, and FBref attributes per club: a
player who moved mid-season gets one row per squad. So FBref can say which of
Understat's candidates a player actually appeared for. The rule is:

    When Understat lists several clubs, the row may only be filed under a club
    FBref also records the player at. If the filed club is not one of them, move
    the row to whichever candidate FBref gives him the most minutes.

That fires on a false attribution and stays silent on a genuine transfer, where
FBref has a row at the filed club too. **Counted on a full rebuild of all twelve
seasons, it refiles 34 of the 927**: fourteen of them in 2026/27, which is the
PSG case, and two to five a season across 1415-2122. The rest of the 927 keep
Understat's own first choice.

Those are the figures the pipeline reports, and they are the ones to quote. An
earlier estimate in this docstring said 45, from a throwaway script that matched
FBref on exact normalised names only; the implementation also uses the
first-plus-last, initial-plus-last and containment tiers, so it finds evidence
for far more players and refiles fewer of them. The same script put the
"pick by minutes among genuine transfers" question at a further 393 rows -- that
number is a **lower bound measured the weaker way**, and re-deriving it properly
is part of that open question rather than something settled here. Either way it
is a judgement about data quality, not a false attribution, and is deliberately
left alone.

**The one it does not fix is a name.** Mousa Al Tamari is a Rennes player, and
FBref spells him ``Musa Al-Taamari``. Normalised, that is ``musa al taamari``
against ``mousa al tamari``: neither contains the other and no first-plus-last
or initial-plus-last key reaches across it, so FBref has no row for him under
any key and the row keeps Understat's answer. He is the residual -- one
player-season still filed at Paris Saint-Germain, where the audit in
:func:`footballml.players.ea_squads.resolve` now reports a runner-up of one
rather than fifteen.

Absence of FBref evidence is always left alone rather than guessed at, which on
a full rebuild is 17 of the 927 -- FBref has no row this module can reach for
them under any tier. A false negative here costs nothing, because
it leaves the behaviour exactly as it was; a false positive would move a row
that was right. That asymmetry is why presence is what the rule turns on.

.. note::
    Reading the club list needs Understat's raw payload, which soccerdata parses
    and discards. :func:`club_candidates` goes back to it through
    ``Understat._read_league_season``, a private method, with ``no_cache=False``
    so it reads the file `read_player_season_stats` stored moments earlier --
    the same bytes, no second request. It is guarded: if a soccerdata upgrade
    moves or renames it, the candidates come back empty, every row keeps the
    club it has today, and the warning says so.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from footballml.cells import as_float, as_int
from footballml.entities import load_aliases
from footballml.ingest.understat import UNDERSTAT_LEAGUES
from footballml.players.fbref import _is_same_name, _name_keys, _tokens, normalise_name

logger = logging.getLogger(__name__)

#: Understat joins a player's clubs with this, unspaced.
SEPARATOR = ","


def club_candidates(reader, seasons: list[str] | None = None) -> dict[tuple, tuple[str, ...]]:
    """Every club Understat lists for a player-season, where it lists more than one.

    Args:
        reader: The ``soccerdata.Understat`` instance that
            `read_player_season_stats` was just called on.
        seasons: Season labels to restrict to. Defaults to the reader's own.

    Returns:
        ``{(league_code, season, player_id): (club, ...)}``, holding only
        player-seasons with two or more clubs. Empty when the payload cannot be
        reached, which leaves every row's club as it is.
    """
    read_league_season = getattr(reader, "_read_league_season", None)
    if read_league_season is None:
        logger.warning(
            "This soccerdata has no Understat._read_league_season; a player-season "
            "listing several clubs keeps whichever one soccerdata picked."
        )
        return {}

    inverse = {v: k for k, v in UNDERSTAT_LEAGUES.items()}
    wanted = set(seasons) if seasons else None
    candidates: dict[tuple, tuple[str, ...]] = {}

    try:
        catalogue = reader.read_seasons()
    except Exception as exc:  # noqa: BLE001 - a missing list must not fail the build
        logger.warning("Could not list Understat seasons (%s); clubs left as found", exc)
        return {}

    for (league, season), row in catalogue.iterrows():
        code = inverse.get(str(league))
        season = str(season)
        if code is None or (wanted is not None and season not in wanted):
            continue
        try:
            # no_cache=False: `read_player_season_stats` has already fetched and
            # stored this exact payload, so this is a read of those bytes.
            data = read_league_season(str(row["url"]), row["league_id"], row["season_id"], False)
            players = data["playersData"]
        except Exception as exc:  # noqa: BLE001 - one bad season, not the run
            logger.warning("Understat %s %s club list unavailable: %s", code, season, exc)
            continue

        for player in players:
            title = str(player.get("team_title") or "")
            if SEPARATOR not in title:
                continue
            clubs = tuple(part.strip() for part in title.split(SEPARATOR) if part.strip())
            if len(clubs) > 1:
                candidates[(code, season, int(player["id"]))] = clubs

    if candidates:
        logger.info("Understat lists several clubs for %d player-season(s)", len(candidates))
    return candidates


def _fbref_minutes(fbref: pd.DataFrame) -> pd.Series:
    """Minutes behind each FBref row, from a count and its per-90 rate.

    FBref's ``90s`` column is consumed as a divisor in `fetch_fbref_stats` and
    not kept, but every count it stores sits beside its own rate, so the minutes
    divide back out. Any one pair would do; several are tried because a player
    with no fouls has no quotient there.

    Only used to order two clubs that FBref records the player at *both* of,
    which is why an estimate is enough -- the rule itself turns on presence.
    """
    pairs = (
        ("fouls", "fouls_per90"),
        ("interceptions", "interceptions_per90"),
        ("tackles_won", "tackles_won_per90"),
        ("fouled", "fouled_per90"),
        ("crosses", "crosses_per90"),
        ("saves", "saves_per90"),
    )
    minutes = pd.Series(0.0, index=fbref.index, dtype=float)
    for count, rate in pairs:
        if count not in fbref.columns or rate not in fbref.columns:
            continue
        counts = pd.to_numeric(fbref[count], errors="coerce")
        rates = pd.to_numeric(fbref[rate], errors="coerce")
        estimate = (counts / rates.where(rates > 0)) * 90.0
        minutes = minutes.mask((minutes <= 0) & estimate.notna(), estimate)
    return minutes.fillna(0.0)


def _clubs_by_name_key(fbref: pd.DataFrame, minutes: pd.Series) -> dict[tuple, dict[str, float]]:
    """Canonical clubs FBref records each player at, with estimated minutes.

    Keyed by ``(League, Season, name key)`` for each of a name's keys, most
    specific first, exactly as `match_players` keys its own fuzzy pass. A key
    that names two different players is dropped rather than resolved by row
    order, for the reason that pass gives: taking the first row seen made the
    build irreproducible.
    """
    aliases = load_aliases("fbref")

    by_key: dict[tuple, dict[str, float]] = defaultdict(dict)
    owner: dict[tuple, str] = {}
    ambiguous: set[tuple] = set()

    for idx, row in fbref.iterrows():
        name = str(row["Player"])
        norm = normalise_name(name)
        club = str(row["Team"]).strip()
        club = aliases.get(club, club)
        group = (row["League"], str(row["Season"]))
        for key in _name_keys(name):
            full = (*group, key)
            seen = owner.get(full)
            if seen is None:
                owner[full] = norm
            elif seen != norm:
                ambiguous.add(full)
            by_key[full][club] = max(by_key[full].get(club, 0.0), as_float(minutes.at[idx]))

    for ambiguous_key in ambiguous:
        by_key.pop(ambiguous_key, None)
    if ambiguous:
        logger.info(
            "Club attribution: %d FBref name key(s) named more than one player, ignored",
            len(ambiguous),
        )
    return dict(by_key)


def _clubs_by_containment(
    block: pd.DataFrame, name: str, aliases: dict[str, str], minutes: pd.Series
) -> dict[str, float]:
    """Clubs for a name only one FBref player in ``block`` can be.

    The last tier of `match_players`, for the same reason it has one: the
    sources print one word where the other prints two. Understat's ``Fabián`` is
    FBref's ``Fabián Ruiz Peña`` and its ``William Pacho`` is ``Willian Pacho``,
    and no name key reaches either. Containment, not overlap, and only when
    exactly one player in the league-season can be meant.
    """
    mine = _tokens(name)
    if not mine:
        return {}
    hits = [
        (idx, row)
        for idx, row in block.iterrows()
        if _is_same_name(mine, _tokens(str(row["Player"])))
    ]
    if len({normalise_name(str(row["Player"])) for _, row in hits}) != 1:
        return {}
    clubs: dict[str, float] = {}
    for idx, row in hits:
        club = str(row["Team"]).strip()
        club = aliases.get(club, club)
        clubs[club] = max(clubs.get(club, 0.0), as_float(minutes.at[idx]))
    return clubs


def resolve_club_attribution(
    players: pd.DataFrame,
    fbref: pd.DataFrame,
    candidates: dict[tuple, tuple[str, ...]],
    team_column: str = "team",
) -> pd.DataFrame:
    """Refile player-seasons Understat filed under a club they never played for.

    Args:
        players: The Understat player-season frame, carrying ``League``,
            ``Season``, ``player_id``, ``player`` and ``team_column``.
        fbref: Output of `fetch_fbref_stats`, which attributes per club.
        candidates: Output of :func:`club_candidates`.
        team_column: Column holding Understat's club name.

    Returns:
        ``players`` with ``team_column`` corrected. Unchanged where there is
        nothing to decide, or no FBref evidence to decide it with.
    """
    needed = {"League", "Season", "player", "player_id", team_column}
    if not candidates or fbref.empty:
        return players
    if not needed <= set(players.columns):
        logger.warning(
            "Club attribution needs %s; left as found",
            sorted(needed - set(players.columns)),
        )
        return players

    understat_aliases = load_aliases("understat")
    fbref_aliases = load_aliases("fbref")
    minutes = _fbref_minutes(fbref)
    by_key = _clubs_by_name_key(fbref, minutes)
    seasons = fbref["Season"].astype(str)

    def canonical(club: str) -> str:
        club = club.strip()
        return understat_aliases.get(club, club)

    out = players.copy()
    moves: list[tuple[str, str, str, str, str]] = []
    blind = 0

    for idx in out.index:
        try:
            player_id = as_int(out.at[idx, "player_id"])
        except (TypeError, ValueError):
            continue
        league = str(out.at[idx, "League"])
        season = str(out.at[idx, "Season"])
        listed = candidates.get((league, season, player_id))
        if not listed:
            continue

        name = str(out.at[idx, "player"])
        evidence: dict[str, float] = {}
        for name_key in _name_keys(name):
            found = by_key.get((league, season, name_key))
            if found:
                evidence = found
                break
        if not evidence:
            block = fbref[(fbref["League"] == league) & (seasons == season)]
            evidence = _clubs_by_containment(block, name, fbref_aliases, minutes)
        if not evidence:
            blind += 1
            continue

        # Only clubs Understat itself put forward are eligible.
        present = {
            club: evidence[canonical(club)] for club in listed if canonical(club) in evidence
        }
        filed = str(out.at[idx, team_column])
        if not present or filed in present:
            continue

        # Understat's own order breaks a tie, so the answer cannot depend on
        # dict iteration order.
        best = max(present, key=lambda club: (present[club], -listed.index(club)))
        if best == filed:
            continue
        out.at[idx, team_column] = best
        moves.append((season, str(league), name, filed, best))

    for season, league, name, filed, best in sorted(moves):
        logger.info(
            "%s %s: %s refiled to %s -- no FBref appearance for %s",
            season,
            league,
            name,
            best,
            filed,
        )
    logger.info(
        "Club attribution: %d of %d multi-club player-season(s) refiled, %d with no FBref evidence",
        len(moves),
        len(candidates),
        blind,
    )
    return out
