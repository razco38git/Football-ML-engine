"""Squad and starting-eleven strength from player ratings.

Item 3 of the original brief: turn player ratings into a team rating the match
model can use.

Predicted line-ups for unplayed matches are not freely available anywhere, so
the eleven has to be inferred. Minutes are the evidence: a player who has
started every match will almost certainly start the next one, and one who has
played 200 minutes almost certainly will not. Picking the highest-minute player
for each slot in a formation reconstructs the likely eleven without needing a
teamsheet.

Three methods are offered, because they answer slightly different questions:

``eleven``
    Most likely starting eleven -- top players by minutes, filling a formation.
    Best estimate of who actually takes the field.
``squad``
    Every rated player, weighted by minutes. Smoother across rotation and
    injuries, and a better description of a squad's depth over a season.
``best_n``
    The N highest-rated players regardless of minutes. Describes the talent
    available rather than what gets used.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

#: Roles grouped into the lines a team rating is reported by.
LINE_BY_ROLE = {
    "GK": "goalkeeper",
    "CB": "defence",
    "FB": "defence",
    "DM": "midfield",
    "MID": "midfield",
    "AMW": "attack",
    "FWD": "attack",
    # Fallbacks for when no FIFA export supplies detailed roles.
    "D": "defence",
    "M": "midfield",
    "F": "attack",
}

#: Slots per line in a 4-3-3. Used to build the eleven.
FORMATION = {"goalkeeper": 1, "defence": 4, "midfield": 3, "attack": 3}

#: Lines that count toward the overall rating, weighted by how many are on the
#: pitch.
#:
#: Keepers were originally excluded, on the argument that save percentage and
#: goals conceded mostly describe the defence in front of them.
#:
#: They are included now, but the margin is narrow and the first justification
#: written here was wrong. It cited -0.62 for the keeper against -0.49 for the
#: back line, both measured against the keeper's *own* season -- which flatters
#: him, because save percentage is partly a consequence of the goals that
#: season rather than a stable trait. Re-measured the way the model actually
#: uses strength, joined from the previous season, the keeper predicts next
#: season's goals conceded at -0.474: slightly *worse* than the back line's
#: -0.483.
#:
#: What still supports including him is the overall: with the keeper it
#: predicts next season's points at +0.694, against +0.690 for the outfield
#: ten. A real edge, but a hair rather than the gulf originally claimed.
OVERALL_LINES = {"goalkeeper": 1, "defence": 4, "midfield": 3, "attack": 3}

#: Roles are filled in this order when a line is short, so a team missing a
#: recognised full-back borrows a centre-back rather than leaving a hole.
_LINE_ORDER = ("goalkeeper", "defence", "midfield", "attack")


def select_eleven(
    players: pd.DataFrame, formation: dict[str, int] | None = None
) -> pd.DataFrame:
    """Pick the most likely starting eleven for one team-season.

    Slots are filled by minutes played, not by rating. That distinction matters:
    the question is who *will* play, and a manager's team selection is better
    evidence of that than our opinion of who deserves to.

    Short lines are topped up from whoever is left over, so a team with only
    three rated defenders still returns eleven players.
    """
    formation = formation or FORMATION
    chosen: list[pd.DataFrame] = []
    used: set = set()

    for line in _LINE_ORDER:
        slots = formation.get(line, 0)
        if slots <= 0:
            continue
        pool = players[(players["line"] == line) & (~players.index.isin(used))]
        picked = pool.nlargest(slots, "minutes")
        used.update(picked.index)
        chosen.append(picked)

    eleven = pd.concat(chosen) if chosen else players.head(0)

    shortfall = sum(formation.values()) - len(eleven)
    if shortfall > 0:
        # Fill from whoever has played most, regardless of line.
        spare = players[~players.index.isin(used)].nlargest(shortfall, "minutes")
        eleven = pd.concat([eleven, spare])

    return eleven


#: Rated players a team-season needs before it gets a rating at all.
#:
#: Ratings require a minutes threshold, so early in a season almost nobody
#: qualifies. Five matchweeks into 2026/27 a fresh build produced 72 team
#: ratings from a *median of two* rated players -- numbers that would be
#: published on the site and later fed to the model as a previous-season
#: feature. Eleven is the obvious floor: fewer than a team's worth of rated
#: players is not a team rating.
MIN_RATED_PLAYERS = 11


def team_strength(
    rated: pd.DataFrame,
    method: str = "eleven",
    best_n: int = 14,
    formation: dict[str, int] | None = None,
    min_players: int = MIN_RATED_PLAYERS,
) -> pd.DataFrame:
    """Team rating per team and season.

    Args:
        rated: Output of :func:`footballml.players.rating.rate_players`.
        method: ``"eleven"``, ``"squad"`` or ``"best_n"``.
        best_n: Squad size when ``method="best_n"``.
        formation: Slots per line. Defaults to a 4-3-3.

    Returns:
        One row per team-season: an overall rating, one per line, and how many
        players it was computed from.
    """
    players = rated[rated["rated"] & rated["rating"].notna()].copy()
    if players.empty:
        logger.warning("No rated players; cannot compute team strength")
        return pd.DataFrame()

    group_col = "role" if "role" in players.columns else "position_group"
    players["line"] = players[group_col].map(LINE_BY_ROLE)
    players = players.dropna(subset=["line"])
    players["rating"] = players["rating"].astype(float)

    keys = ["League", "Season", "Team"]

    if method == "eleven":
        # A plain loop rather than groupby.apply: apply's handling of the group
        # keys varies with include_groups, and concatenating the returned frames
        # keeps every original column intact without any index gymnastics.
        picks = [select_eleven(g, formation) for _, g in players.groupby(keys, observed=True)]
        selected = pd.concat(picks) if picks else players.head(0)
    elif method == "best_n":
        selected = (
            players.sort_values("rating", ascending=False)
            .groupby(keys, observed=True)
            .head(best_n)
        )
    elif method == "squad":
        selected = players
    else:
        raise ValueError(f"unknown method {method!r}; use eleven, squad or best_n")

    lines = (
        selected.groupby([*keys, "line"], observed=True)
        .apply(_weighted_mean, include_groups=False)
        .reset_index(name="rating")
    )
    wide = lines.pivot_table(index=keys, columns="line", values="rating").reset_index()
    wide.columns.name = None

    counts = selected.groupby(keys, observed=True).size().reset_index(name="n_players")
    wide = wide.merge(counts, on=keys, how="left")

    available = [line for line in OVERALL_LINES if line in wide.columns]
    weights = pd.Series({line: OVERALL_LINES[line] for line in available})
    present = wide[available]
    # Normalise by the weights actually present, so a team with no rated keeper
    # is not dragged down by a missing line.
    wide["overall"] = (present * weights).sum(axis=1) / (present.notna() * weights).sum(axis=1)

    wide = wide.rename(
        columns={
            **{line: f"strength_{line}" for line in FORMATION if line in wide.columns},
            "overall": "strength_overall",
        }
    )
    wide["method"] = method

    thin = wide["n_players"] < min_players
    if thin.any():
        seasons = sorted(wide.loc[thin, "Season"].astype(str).unique())
        logger.info(
            "Dropping %d team-season(s) with fewer than %d rated players "
            "(seasons %s) -- too early in the season to rate a squad",
            int(thin.sum()), min_players, ", ".join(seasons),
        )
        wide = wide[~thin]

    logger.info("Computed %s strength for %d team-seasons", method, len(wide))
    return wide.sort_values(["Season", "strength_overall"], ascending=[False, False])


def _weighted_mean(group: pd.DataFrame) -> float:
    """Mean rating weighted by minutes played."""
    weights = group["minutes"].clip(lower=0)
    if weights.sum() <= 0:
        return float(group["rating"].mean())
    return float((group["rating"] * weights).sum() / weights.sum())
