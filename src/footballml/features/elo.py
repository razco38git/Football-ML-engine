"""Elo ratings: the long-run team quality the rolling windows cannot express.

Every form feature in this project is a window -- five matches or nineteen.
Windows forget. A side that has been excellent for three years and has just
drawn two looks, to a nineteen-match window, much like a mid-table side on a
good run, and the only persistent input is the previous season's squad rating,
joined once and then static until August.

That shows up against the market. On 20,013 matches, where the market says 84%
we say 79% and the home side wins 85%; where it says 65% we say 61% and the
result comes in at 67%. Our RPS is 0.2009 against the market's 0.1950.
Decomposing real league tables into skill and luck, our predicted spread is
13.0 points against the market's 14.2.

It is not a calibration problem, which was worth ruling out before writing any
of this: sharpening the output probabilities was swept from 0.6 to 1.2 and the
optimum is ~0.95, with no improvement in RPS. The model is not uniformly timid,
it is timid exactly where the market is confident -- which is a shortage of
information, and the missing information is persistent quality.

Elo carries that. A rating only moves when a result disagrees with what the
rating expected, so it accumulates across seasons instead of expiring with a
window.

**Leak-safety.** Each match records the ratings the two sides carried *going
in*, and only then applies the update. A match never sees its own result, which
is the same discipline as the ``.shift(1)`` in the rolling features and is
pinned by ``tests/test_leakage.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Every team starts here. The scale is arbitrary; only differences matter.
START = 1500.0

#: How far a rating moves per match. 20 is the conventional football value --
#: roughly a 30-point swing for beating a well-matched opponent, so a season
#: can separate two teams by a division's worth of rating without a single
#: result dominating.
K = 20.0

#: Rating points of home advantage, added to the home side when forming the
#: expectation. Not tuned here: it is worth about 0.55 expected score at equal
#: ratings, which is close to the home win rate in these five leagues.
HOME_ADVANTAGE = 60.0

#: Fraction of the way each rating is pulled back to `START` between seasons.
#:
#: Squads turn over in the summer and promoted sides arrive with no history, so
#: carrying a rating forward untouched overstates how much last May still says
#: about this August. A quarter is the usual choice and keeps most of the
#: signal.
SEASON_REGRESSION = 0.25


def _expected(rating: float, opponent: float) -> float:
    """Expected score (1 win, 0.5 draw, 0 loss) on the logistic Elo curve."""
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def _margin_multiplier(goal_difference: int) -> float:
    """Weight the update by how convincing the result was.

    A 4-0 says more than a 1-0, but not four times as much -- the logarithm
    keeps a thrashing from dominating, which matters because blowouts are noisy
    and often happen against ten men.
    """
    return float(np.log1p(abs(goal_difference)))


def _walk(
    df: pd.DataFrame,
    k: float,
    home_advantage: float,
    season_regression: float,
) -> tuple[dict[tuple, tuple[float, float]], dict[str, float]]:
    """Play the history forward once.

    Returns the ratings each side carried *into* every match, keyed by match,
    and the ratings everyone holds after the last one. Both come from the same
    pass so they can never disagree.
    """
    # One row per match, taken from the home side so goals are oriented
    # home-first. Ordering by date is what makes the walk causal.
    home = df[df["Venue"] == "Home"].sort_values(["Date", "League", "Team"])

    ratings: dict[str, float] = {}
    seen_season: dict[str, str] = {}
    pre: dict[tuple, tuple[float, float]] = {}

    for row in home.itertuples():
        season = str(row.Season)
        pair = []
        for team in (row.Team, row.Opponent):
            rating = ratings.get(team, START)
            # A new season for this team: pull back toward the mean before the
            # rating is used, so the regression is visible to the first match
            # rather than applied retroactively.
            if seen_season.get(team) not in (None, season):
                rating = rating + season_regression * (START - rating)
            ratings[team] = rating
            seen_season[team] = season
            pair.append(rating)

        home_rating, away_rating = pair
        pre[(row.League, row.Date, row.Team, row.Opponent)] = (home_rating, away_rating)

        if pd.isna(row.GoalsFor) or pd.isna(row.GoalsAgainst):
            continue  # Unplayed placeholder: it has no result to learn from.

        goals_for, goals_against = float(row.GoalsFor), float(row.GoalsAgainst)
        actual = 1.0 if goals_for > goals_against else 0.5 if goals_for == goals_against else 0.0
        expected = _expected(home_rating + home_advantage, away_rating)

        # Draws carry no margin, so the multiplier floors at 1 rather than 0 --
        # otherwise a draw would never move a rating and an unbeaten run of
        # draws would leave a side looking exactly as it did in August.
        weight = max(1.0, _margin_multiplier(int(goals_for - goals_against)))
        change = k * weight * (actual - expected)

        ratings[row.Team] = home_rating + change
        ratings[row.Opponent] = away_rating - change

    return pre, ratings


def add_elo(
    tmh: pd.DataFrame,
    k: float = K,
    home_advantage: float = HOME_ADVANTAGE,
    season_regression: float = SEASON_REGRESSION,
) -> pd.DataFrame:
    """Attach each team's pre-match Elo rating to a long team-match frame.

    Args:
        tmh: Long team-match history, two rows per match, with ``Team``,
            ``Opponent``, ``Venue``, ``GoalsFor``, ``GoalsAgainst`` and
            ``Date``. Need not be sorted.
        k: Update size per match.
        home_advantage: Rating points granted to the home side.
        season_regression: Fraction pulled back toward ``START`` each new season.

    Returns:
        ``tmh`` with an ``elo`` column holding the rating the team carried
        *into* that match -- never one that has seen the result.
    """
    if tmh.empty:
        return tmh.assign(elo=pd.Series(dtype="float64"))

    df = tmh.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    pre, _ = _walk(df, k, home_advantage, season_regression)

    def lookup(row: pd.Series) -> float:
        if row["Venue"] == "Home":
            key = (row["League"], row["Date"], row["Team"], row["Opponent"])
            index = 0
        else:
            key = (row["League"], row["Date"], row["Opponent"], row["Team"])
            index = 1
        found = pre.get(key)
        return np.nan if found is None else found[index]

    df["elo"] = df.apply(lookup, axis=1).astype("float64")
    return df


def final_ratings(
    tmh: pd.DataFrame,
    k: float = K,
    home_advantage: float = HOME_ADVANTAGE,
    season_regression: float = SEASON_REGRESSION,
) -> pd.Series:
    """Every team's rating *after* the last match in ``tmh``, strongest first.

    Note the difference from the ``elo`` column, which is deliberately the
    rating carried *into* a match: reading the last row per team would give a
    rating that has not seen that team's final result.

    Not used by the feature build -- this is for inspecting whether the ratings
    say anything a football supporter would recognise.
    """
    if tmh.empty:
        return pd.Series(dtype="float64")

    df = tmh.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    _, ratings = _walk(df, k, home_advantage, season_regression)
    return pd.Series(ratings, dtype="float64").sort_values(ascending=False)
