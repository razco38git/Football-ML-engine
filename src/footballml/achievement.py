"""Points a club took against the points its squad rating implied.

The squad rating says how good the players are. The table says what happened.
Subtracting one from the other is the only way the site can say *this club
got more out of its season than the players explain*, which is the question a
league table provokes and never answers.

The comparison is made **inside each league-season**, by least squares over
that division's own clubs. Three reasons, and the first two are the ones that
would otherwise produce nonsense:

- Ratings are not league-relative but *points* are. A 70-rated side in Ligue 1
  plays different opponents from a 70-rated side in the Premier League, and
  eighteen clubs take 38 fewer points between them than twenty do.
- The relationship moves between seasons. Fitting one line across twelve years
  would charge every club in a tight season for the spread of a loose one.
- It needs no constant anyone has to defend. The line is whatever that league
  that season actually did.

What the residuals look like
----------------------------

Over 762 finished club-seasons the residual standard deviation is **7.93
points**, and the extremes are the seasons a football follower would name:

    1516 Leicester         rated 68.6   81 points   expected 45.8   +35.2
    2324 Girona            rated 72.0   81 points   expected 54.1   +26.9
    1819 Lille             rated 69.8   75 points   expected 53.1   +21.9
    2223 Arsenal           rated 78.9   84 points   expected 65.7   +18.3
    1920 Sheffield United  rated 67.6   54 points   expected 36.7   +17.3

and at the other end Aston Villa 2015/16 at -26.6, Málaga 2017/18 at -24.9,
Tottenham 2024/25 at -24.1 and Napoli 2023/24 at -21.6.

**Leicester 2015/16 is +35.2, which is 4.4 standard deviations and the largest
of the 762.** That is the board's smoke test rather than its tuning target:
any change to the fit that loses it has broken something.

Why a club's own history ships with every row
---------------------------------------------

A single large residual reads as a verdict on the club. It is not one, and
Leicester is the proof: every other Leicester season in the data sits inside
±8, so the 2015/16 spike is an event and not a property.

    1415 +0.8   1516 +35.2   1617 -1.9   1718 -1.1   1819 -2.9
    1920 +3.4   2021 +7.8    2122 -7.3   2223 -14.2  2425 -4.1

So :func:`achievement` returns each club's whole run beside the row, and the
page draws it. A board that showed only the spike would be inviting the
reader to the wrong conclusion.

This also settles an old tension in the repo rather than ducking it.
:mod:`footballml.players.rating` says a squad-quality rating that *liked*
Leicester in 2015/16 would be the broken one -- EA had them 18th of 20 and the
bookmakers said 5000-1. This agrees, and makes it the point: the rating was
right about the squad, and the season was the most extreme overachievement in
twelve years of data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Clubs a league-season needs before a line is fitted through it. A division
#: is 18 to 20, so this only ever excludes a partially built season.
MIN_CLUBS = 10

KEYS = ["League", "Season", "Team"]


def expected_points(strength: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Fit points against squad rating within each league-season.

    Args:
        strength: `team_strength.csv`, carrying `strength_overall`.
        outcomes: One row per club-season with `pts`, from a caller that has
            already dropped seasons too short to compare (a points *total*
            from a season in progress is not one).

    Returns:
        The joined rows plus `expected` and `over`, the residual in points.
        Empty when nothing can be fitted.
    """
    if strength.empty or outcomes.empty:
        return pd.DataFrame()

    merged = strength.assign(Season=strength["Season"].astype(str)).merge(
        outcomes.assign(Season=outcomes["Season"].astype(str)), on=KEYS, how="inner"
    )
    merged = merged.dropna(subset=["strength_overall", "pts"])
    if merged.empty:
        return pd.DataFrame()

    fitted: list[pd.DataFrame] = []
    for (league, season), block in merged.groupby(["League", "Season"], observed=True):
        if len(block) < MIN_CLUBS:
            continue
        ratings = block["strength_overall"].to_numpy(dtype=float)
        # A division where every club rates the same has no line through it.
        if np.ptp(ratings) <= 0:
            logger.info("%s %s: no spread in squad rating; skipped", league, season)
            continue
        slope, intercept = np.polyfit(ratings, block["pts"].to_numpy(dtype=float), 1)
        fitted.append(block.assign(expected=slope * ratings + intercept))

    if not fitted:
        return pd.DataFrame()

    out = pd.concat(fitted, ignore_index=True)
    out["over"] = out["pts"] - out["expected"]
    return out


def club_histories(fitted: pd.DataFrame) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """Every club's residual by season, oldest first, keyed by league and team.

    Empty in, empty out: :func:`expected_points` returns a bare frame when
    nothing could be fitted -- no finished season, or none with enough clubs --
    and grouping that by a column it does not have would raise where the
    caller only wants no sparklines.
    """
    histories: dict[tuple[str, str], list[tuple[str, float]]] = {}
    if fitted.empty or "League" not in fitted.columns:
        return histories
    for (league, team), block in fitted.groupby(["League", "Team"], observed=True):
        ordered = block.sort_values("Season")
        histories[(str(league), str(team))] = list(
            zip(ordered["Season"].astype(str), ordered["over"].astype(float), strict=True)
        )
    return histories


def achievement(
    strength: pd.DataFrame, outcomes: pd.DataFrame, limit: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """The clubs furthest above and below what their squad rating implied.

    Returns:
        The overachievers, the underachievers, and a summary carrying the
        residual standard deviation and how many club-seasons it rests on --
        without which "+35 points" is a number with no scale attached.
    """
    fitted = expected_points(strength, outcomes)
    if fitted.empty:
        logger.info("No league-season could be fitted; no achievement board")
        return pd.DataFrame(), pd.DataFrame(), {}

    summary = {
        # Sample sd, matching the figure quoted in the docstring and in
        # `docs/PROPOSAL-historical-insights.md`.
        "residual_sd": float(fitted["over"].std()),
        "club_seasons": int(len(fitted)),
    }
    return (
        fitted.nlargest(limit, "over"),
        fitted.nsmallest(limit, "over"),
        summary,
    )
