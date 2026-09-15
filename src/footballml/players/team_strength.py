"""Squad strength from player ratings.

Item 3 of the original brief: turn player ratings into a team rating that can
feed the match model.

The honest framing is that this is a **squad** rating, not a starting eleven.
Predicted line-ups for matches that have not been played are not freely
available anywhere -- nobody publishes Saturday's teamsheet in a form that can
be ingested. What can be measured is who has actually been playing, which in
practice is the same eleven: minutes are the best available proxy for selection.

Weighting by minutes does the work. A player who started every match counts
roughly ten times a fringe squad player, so the number reflects the side that
actually takes the field rather than the depth chart. It cannot anticipate a
surprise rotation or an injury announced on the morning of the match, and that
gap is part of why bookmakers stay ahead of us.
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
    "MID": "midfield",
    "AMW": "attack",
    "FWD": "attack",
    # Fallbacks for when no FIFA export supplies detailed roles.
    "D": "defence",
    "M": "midfield",
    "F": "attack",
}

#: How many players of each line are on the pitch, used to weight the overall.
#: Roughly a 4-3-3: one keeper, four defenders, three midfielders, three ahead.
LINE_SIZES = {"goalkeeper": 1, "defence": 4, "midfield": 3, "attack": 3}


def team_strength(
    rated: pd.DataFrame, top_n_per_line: int | None = None
) -> pd.DataFrame:
    """Minutes-weighted squad rating per team and season.

    Args:
        rated: Output of :func:`footballml.players.rating.rate_players`.
        top_n_per_line: When set, use only this many highest-rated players per
            line. Approximates a first eleven more tightly than weighting the
            whole squad, at the cost of ignoring rotation.

    Returns:
        One row per team-season with an overall rating and one per line.
    """
    players = rated[rated["rated"] & rated["rating"].notna()].copy()
    if players.empty:
        logger.warning("No rated players; cannot compute team strength")
        return pd.DataFrame()

    group_col = "role" if "role" in players.columns else "position_group"
    players["line"] = players[group_col].map(LINE_BY_ROLE)
    players = players.dropna(subset=["line"])
    players["rating"] = players["rating"].astype(float)

    if top_n_per_line:
        players = (
            players.sort_values("rating", ascending=False)
            .groupby(["League", "Season", "Team", "line"], observed=True)
            .head(top_n_per_line)
        )

    keys = ["League", "Season", "Team"]
    lines = (
        players.groupby([*keys, "line"], observed=True)
        .apply(_weighted_mean, include_groups=False)
        .reset_index(name="rating")
    )
    wide = lines.pivot_table(index=keys, columns="line", values="rating").reset_index()
    wide.columns.name = None

    # Overall weights each line by how many of them are on the pitch, so a
    # brilliant keeper cannot carry a poor outfield.
    available = [line for line in LINE_SIZES if line in wide.columns]
    weights = pd.Series({line: LINE_SIZES[line] for line in available})
    present = wide[available]
    wide["overall"] = (present * weights).sum(axis=1) / (
        present.notna() * weights
    ).sum(axis=1)

    renamed = {line: f"strength_{line}" for line in available}
    wide = wide.rename(columns={**renamed, "overall": "strength_overall"})

    logger.info("Computed strength for %d team-seasons", len(wide))
    return wide.sort_values(["Season", "strength_overall"], ascending=[False, False])


def _weighted_mean(group: pd.DataFrame) -> float:
    """Mean rating weighted by minutes played."""
    weights = group["minutes"].clip(lower=0)
    if weights.sum() <= 0:
        return float(group["rating"].mean())
    return float((group["rating"] * weights).sum() / weights.sum())
