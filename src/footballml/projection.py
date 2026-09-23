"""Project a league table by simulating the rest of the season.

The match model already produces a full scoreline distribution per fixture, so
a projected table is mostly bookkeeping: play out every remaining match a few
thousand times, count the points, and report what happened across those seasons
rather than a single tidy number.

Two design points worth stating plainly.

**The remaining fixtures are derived, not fetched.** football-data publishes
only the next round or two. But the big five are exact double round-robins --
every completed 2025/26 season holds precisely one match per ordered pair, with
no duplicates -- so "every ordered pair that has not been played" reconstructs
the fixture list without another source. :func:`remaining_fixtures` asserts that
assumption rather than trusting it.

**Scorelines are sampled from the Dixon-Coles matrix**, not from two independent
Poisson draws. Independent draws would quietly discard the low-score correction
the model was fitted with, and 0-0 and 1-1 are exactly the results a league
table turns on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Points for a win and a draw. Every one of the five leagues uses these.
WIN, DRAW = 3, 1


def remaining_fixtures(
    tmh: pd.DataFrame, league: str, season: str
) -> pd.DataFrame:
    """Ordered pairs in this league-season that have not been played yet.

    Args:
        tmh: Long team-match history -- two rows per match, one per side.
        league: Division code, e.g. ``"E0"``.
        season: Season label, e.g. ``"2627"``.

    Returns:
        ``HomeTeam`` / ``AwayTeam`` for each unplayed fixture. Empty once the
        season is complete.

    Raises:
        ValueError: If a pairing appears twice, which would mean the season is
            not the double round-robin this reconstruction assumes.
    """
    season = str(season)
    rows = tmh[(tmh["League"] == league) & (tmh["Season"].astype(str) == season)]
    home = rows[rows["Venue"] == "Home"]

    played = set(zip(home["Team"], home["Opponent"], strict=True))
    if len(played) != len(home):
        raise ValueError(
            f"{league} {season} has a repeated pairing; the fixture list cannot "
            "be reconstructed by assuming a double round-robin"
        )

    teams = sorted(set(rows["Team"]))
    fixtures = [
        {"League": league, "Season": season, "HomeTeam": h, "AwayTeam": a}
        for h in teams
        for a in teams
        if h != a and (h, a) not in played
    ]
    return pd.DataFrame(fixtures, columns=["League", "Season", "HomeTeam", "AwayTeam"])


def current_table(tmh: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Points, goal difference and matches played so far."""
    rows = tmh[
        (tmh["League"] == league) & (tmh["Season"].astype(str) == str(season))
    ].copy()
    rows["points"] = np.where(
        rows["GoalsFor"] > rows["GoalsAgainst"], WIN,
        np.where(rows["GoalsFor"] == rows["GoalsAgainst"], DRAW, 0),
    )
    table = (
        rows.groupby("Team")
        .agg(
            played=("points", "size"),
            points=("points", "sum"),
            goals_for=("GoalsFor", "sum"),
            goals_against=("GoalsAgainst", "sum"),
        )
        .reset_index()
    )
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    return table.sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False
    ).reset_index(drop=True)


def simulate(
    fixtures: pd.DataFrame,
    matrices: np.ndarray,
    table: pd.DataFrame,
    n_sims: int = 10_000,
    seed: int = 7,
    relegation_places: int = 3,
) -> pd.DataFrame:
    """Play the remaining fixtures ``n_sims`` times and summarise the outcomes.

    Args:
        fixtures: Unplayed fixtures, aligned with ``matrices``.
        matrices: ``(n_fixtures, G+1, G+1)`` scoreline probabilities.
        table: Points and goal difference already banked, from
            :func:`current_table`.
        n_sims: Seasons to simulate.
        seed: Fixed so the same inputs give the same table. The API serves a
            stored file, and a projection that drifted between identical runs
            would be indistinguishable from a real change in the forecast.

    Returns:
        One row per team: projected points (mean), a 10th-90th percentile band,
        and the share of simulated seasons finishing first, in the top four, and
        in the relegation places.
    """
    teams = sorted(table["Team"])
    index = {team: i for i, team in enumerate(teams)}
    rng = np.random.default_rng(seed)

    banked_points = np.array(
        [table.set_index("Team").loc[t, "points"] for t in teams], dtype="int32"
    )
    banked_gd = np.array(
        [table.set_index("Team").loc[t, "goal_difference"] for t in teams],
        dtype="int32",
    )

    points = np.repeat(banked_points[None, :], n_sims, axis=0)
    goal_diff = np.repeat(banked_gd[None, :], n_sims, axis=0)

    if len(fixtures):
        size = matrices.shape[1]
        # One flat categorical draw per fixture per season. Sampling the matrix
        # keeps the Dixon-Coles low-score correction the model was fitted with;
        # two independent Poisson draws would silently discard it.
        flat = matrices.reshape(len(fixtures), -1)
        flat = flat / flat.sum(axis=1, keepdims=True)
        cdf = flat.cumsum(axis=1)

        home_idx = np.array([index[t] for t in fixtures["HomeTeam"]])
        away_idx = np.array([index[t] for t in fixtures["AwayTeam"]])

        for f in range(len(fixtures)):
            cell = np.searchsorted(cdf[f], rng.random(n_sims))
            hg, ag = np.divmod(cell, size)
            home, away = home_idx[f], away_idx[f]

            # Direct indexing rather than `np.add.at`: one fixture touches one
            # column per side, so there are no repeated indices to accumulate.
            points[:, home] += np.where(hg > ag, WIN, np.where(hg == ag, DRAW, 0))
            points[:, away] += np.where(ag > hg, WIN, np.where(hg == ag, DRAW, 0))
            goal_diff[:, home] += hg - ag
            goal_diff[:, away] += ag - hg

    # Rank within each simulated season: points first, goal difference as the
    # tiebreak, which is how all five leagues separate level teams (Spain uses
    # head-to-head first, a refinement not worth carrying here).
    order = np.lexsort((-goal_diff, -points), axis=1)
    position = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    position[rows, order] = np.arange(len(teams))[None, :] + 1

    return pd.DataFrame(
        {
            "Team": teams,
            "projected_points": points.mean(axis=0).round(1),
            "points_low": np.percentile(points, 10, axis=0).round(0),
            "points_high": np.percentile(points, 90, axis=0).round(0),
            "projected_position": position.mean(axis=0).round(1),
            "title_pct": (position == 1).mean(axis=0).round(4),
            "top_four_pct": (position <= 4).mean(axis=0).round(4),
            "relegation_pct": (
                position > len(teams) - relegation_places
            ).mean(axis=0).round(4),
        }
    )
