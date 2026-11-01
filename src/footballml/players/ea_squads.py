"""Which EA club label is this team, in this edition?

The squad-aware rating needs a *roster*: the players EA says are at a club in
a given edition. That needs the club's label in that edition's export, and the
label is not stable. EA renames clubs between editions -- Paris Saint-Germain
is ``Paris SG`` in 24/25, ``Paris Saint-Germain`` in 25/26 and ``Paris SG``
again in 26/27 -- so the label cannot be hard-coded, and `config/team_aliases.yaml`
(182 derived sofifa aliases, called by no production code) carries errors that
would hand us the wrong squad: `Falkirk -> Swansea` is a different club in a
different country, and `FC Barcelona B -> Barcelona` folds a youth side into
the first team.

So the label is *inferred from the players*, who we have already matched to EA
rows one at a time: for a club-season, look at where EA puts the players we
know played there, and take the label they point at. Three things make that
work, and each of them is here because it failed without it.

**History first.** A plurality vote alone breaks where a club's rated players
do not all belong to it. Paris Saint-Germain's 2026/27 vote splits
``Stade Rennais FC`` 7 against ``Paris SG`` 6, so the naive rule hands PSG
Rennes' label and both squads are wrong.

The split turned out not to be sampling noise, and finding out why is the
better argument for the rule. **Rennes' squad is merged into PSG's in our own
player file**: 15 of the 39 player-seasons filed under `Paris Saint Germain`
in 2026/27 carry `Stade Rennais FC` as their EA club, while `Rennes` itself
holds six rows against every other Ligue 1 club's 21 to 27. Charlie Cresswell,
Valentin Rongier and Brice Samba are published as PSG players. That is an
ingest defect, it predates this module, and it is the only club-season in
1,266 where the runner-up EA club holds more than three rated players -- the
audit that says so is :func:`resolve`'s own `runner_up`.

So the vote is not merely thin here, it is *contaminated*, and no threshold on
it would help. Preferring the label a club used most recently, when that label
is still in this edition, is evidence of a different kind: a club that was
``Paris SG`` last edition and has a ``Paris SG`` this edition is that club,
whatever half of a borrowed squad says.

**A vote needs evidence behind it.** Spezia's 2020/21 promotion season has 22
rated players whose EA clubs are AEK Athens, Cagliari, Empoli, Sampdoria,
Inter and Milan -- one vote each, because *Spezia is not in the 2021 edition
at all*. It arrives in 2122. A plurality of one is not a plurality; it is the
absence of the club, and the honest answer is no label rather than the first
name out of a tie. :data:`MIN_VOTES` is what makes that answer available, and
an unresolved club-season simply has no squad-aware rating -- which is true,
and better than rating Spezia's squad as Milan's.

**A label belongs to one club.** Two clubs resolving to one label means at
least one is wrong, so the claims are compared and the weaker one gives it up
and re-resolves. Without this, the history rule's own fallback reintroduced
the Spezia collision from the other direction.

Reserve sides are dropped from candidacy rather than mapped to their first
team. `FC Barcelona B` is a different squad from `FC Barcelona`; folding them
together -- which `team_aliases.yaml` does -- produces a 53-man "squad" from
which a best-15 could pick a youth player who never appeared in the league. A
youth player who steps up shows up in the first-team squad in the next
edition, which is where he should count.

**That filter is a guard, not a fix.** Measured: of the 18 reserve labels
across the twelve editions, **none** wins a club-season's vote even with the
filter removed, because a reserve side holds none of the players our league
data records. It stays because the failure it prevents is real the moment
anyone resolves labels through the alias map instead, and because a silent
`FC Barcelona B` would be very hard to notice downstream.

**Bare names are shared between real clubs, and that one is measured rather
than filtered.** `Everton` in the 2223 edition holds 57 players: Everton FC
*and* Everton de Viña del Mar. Six club-seasons are inflated this way (45-57
players, against a median of 30). They are left alone because the contamination
does not reach the selection -- the best-15 cut-off sits at 75-81 overall while
the Chilean squad tops out at 73, and all six best-15s are players our own
big-five data knows. If a later edition narrows that margin, the audit's
size report is what catches it, so the sizes are reported rather than clipped.

What this deliberately does *not* do is diff editions to find transfers. The
roster is the data: `fifa_2122.csv` says Messi is at Paris Saint-Germain, so
asking who left Barcelona is a harder version of a question already answered
-- and it would need a `fifa_1314.csv`, which does not exist.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from footballml.data import RAW_DIR

logger = logging.getLogger(__name__)

FIFA_DIR = RAW_DIR / "fifa"

#: Fewest players that must agree before a vote decides a label. One is what
#: resolved Spezia 2021 to Milan; two is a coin toss between two wrong
#: answers, and raising it further starts discarding real thin-squad editions.
MIN_VOTES = 3

#: Reserve, youth and academy sides. Excluded from candidacy, never remapped:
#: see the module docstring. Anchored at the end of the label, so `Barcelona B`
#: is caught and `Bayer 04 Leverkusen` is not.
_RESERVE = re.compile(
    r"\s(?:B|II|2|U1[6-9]|U2[0-3]|Reserves?|Academy)\s*$",
    re.IGNORECASE,
)

#: A reserve side the pattern cannot see: Sevilla's second team carries a name
#: of its own rather than a suffix.
_RESERVE_LABELS = frozenset({"Sevilla Atlético", "Sevilla Atletico"})

#: A first team the pattern wrongly catches. Willem II Tilburg is an Eredivisie
#: club whose name ends in a Roman two; there is no `Willem` for it to be the
#: reserves of. Tried and rejected: deciding this by looking for a parent label
#: in the same edition, which gets Willem II right but then misreads
#: `B. Dortmund II`, `Freiburg II` and `SC Freiburg II` as first teams, because
#: those editions spell the parent club differently (`Borussia Dortmund`).
#: Two named labels that can each be checked in one line beat a rule that is
#: wrong three ways.
_FIRST_TEAM_LABELS = frozenset({"Willem II"})


def is_reserve(label: str) -> bool:
    """Whether ``label`` names a club's reserve, youth or academy side."""
    label = label.strip()
    if label in _FIRST_TEAM_LABELS:
        return False
    return label in _RESERVE_LABELS or bool(_RESERVE.search(label))


@dataclass(frozen=True)
class Resolution:
    """One club-season's EA label, and how it was decided."""

    season: str
    team: str
    label: str | None
    how: str
    votes: int
    runner_up: tuple[str, int] | None = None

    @property
    def resolved(self) -> bool:
        return self.label is not None


def edition_labels(directory: Path | None = None) -> dict[str, set[str]]:
    """Every club label present in each edition on disk, reserves included."""
    directory = Path(directory or FIFA_DIR)
    labels: dict[str, set[str]] = {}
    for path in sorted(directory.glob("fifa_*.csv")):
        season = path.stem.split("_", 1)[1]
        clubs = pd.read_csv(path, usecols=["club"], low_memory=False)["club"]
        labels[season] = set(clubs.dropna().astype(str).str.strip())
    return labels


def _votes(block: pd.DataFrame, available: set[str]) -> Counter:
    """How many of a club-season's players EA places at each candidate label.

    Restricted to labels that exist in *this* edition -- a player matched
    across editions can carry a label this one does not have -- and to first
    teams.
    """
    clubs = block["fifa_club"].dropna().astype(str).str.strip()
    return Counter(club for club in clubs if club in available and not is_reserve(club))


def _claim_strength(resolution: Resolution) -> tuple[int, int]:
    """Ordering for two clubs claiming one label: history outranks a vote."""
    return (1 if resolution.how == "history" else 0, resolution.votes)


def resolve(players: pd.DataFrame, directory: Path | None = None) -> list[Resolution]:
    """Resolve every ``(Season, Team)`` in ``players`` to an EA club label.

    Args:
        players: Rated player-seasons, with `Season`, `Team` and `fifa_club`.
        directory: Where the editions live. Defaults to `data/raw/fifa`.

    Returns:
        One :class:`Resolution` per club-season, season then team. An
        unresolved one carries ``label=None`` and a `how` saying why.
    """
    available = edition_labels(directory)
    frame = players.copy()
    frame["Season"] = frame["Season"].astype(str)

    resolutions: list[Resolution] = []
    history: dict[str, str] = {}

    for season in sorted(frame["Season"].unique()):
        here = available.get(season)
        if not here:
            logger.warning("no fifa_%s.csv -- every %s club is unresolved", season, season)
            here = set()

        season_rows = frame[frame["Season"] == season]

        # First pass: each club's own best claim, history preferred over a vote.
        claims: dict[str, Resolution] = {}
        for team, block in season_rows.groupby("Team", observed=True):
            team = str(team)
            votes = _votes(block, here)
            ranked = votes.most_common()
            runner_up = ranked[1] if len(ranked) > 1 else None
            previous = history.get(team)

            if previous and previous in here:
                claims[team] = Resolution(
                    season, team, previous, "history", votes.get(previous, 0), runner_up
                )
            elif ranked and ranked[0][1] >= MIN_VOTES:
                label, count = ranked[0]
                claims[team] = Resolution(season, team, label, "vote", count, runner_up)
            else:
                best = ranked[0] if ranked else None
                claims[team] = Resolution(
                    season,
                    team,
                    None,
                    f"only {best[1]} vote(s)" if best else "not in edition",
                    best[1] if best else 0,
                    runner_up,
                )

        # Second pass: a label belongs to one club. The weaker claim yields and
        # re-resolves to its best unclaimed candidate that clears the bar.
        for label in {r.label for r in claims.values() if r.label}:
            rivals = sorted(
                (t for t, r in claims.items() if r.label == label),
                key=lambda t: _claim_strength(claims[t]),
                reverse=True,
            )
            for team in rivals[1:]:
                taken = {r.label for t, r in claims.items() if r.label and t != team}
                votes = _votes(season_rows[season_rows["Team"] == team], here)
                nxt = next(
                    (
                        (club, count)
                        for club, count in votes.most_common()
                        if club not in taken and count >= MIN_VOTES
                    ),
                    None,
                )
                logger.info("%s: %s yielded %r to %s", season, team, label, rivals[0])
                claims[team] = Resolution(
                    season,
                    team,
                    nxt[0] if nxt else None,
                    "vote after yielding" if nxt else f"yielded {label!r}",
                    nxt[1] if nxt else 0,
                    claims[team].runner_up,
                )

        for team, resolution in sorted(claims.items()):
            resolutions.append(resolution)
            if resolution.label:
                history[team] = resolution.label

    return resolutions
