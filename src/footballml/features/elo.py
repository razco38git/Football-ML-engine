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

**MEASURED (2026-09-30): feeding UEFA ties in works, and is not enough.**

822 big-five vs big-five European matches now inform the walk (``extra``).
Elo itself got better at the job: on the 788 cross-league ties its own implied
probability improved from Brier 0.1772 to **0.1735**, and its correlation with
the result from +0.260 to **+0.295**. Domestically it costs nothing -- pooled
walk-forward RPS **0.2006 either way**, six seasons better and four worse.

But the model barely uses it. Regressing its predicted home-win probability
against the league-rating gap across those ties:

- before: slope **-0.0002** (p 0.94) -- no relationship at all
- after: slope **+0.0034** (p 0.22) -- right sign, still not significant
- what the truth does: slope **+0.0402**

So the model now moves about **8%** as much as it should, up from nothing, and
European RPS improves 0.2182 -> 0.2177. Real, directionally right, small.

**Why a better feature was never going to be enough.** Every match the model
trains on is domestic, so the league gap in training is *always zero*. The
booster has never seen a row where ``elo_diff`` carries league strength, and
applies the weight it learned where that meaning does not exist. The limit is
the training set, not the feature -- which means the remaining ~92% of the gap
needs either European matches in training, or a correction applied outside the
model, where the 822-match measurement can be used directly.

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

#: Fraction of the way each rating is pulled back between seasons.
#:
#: Squads turn over in the summer and promoted sides arrive with no history, so
#: carrying a rating forward untouched overstates how much last May still says
#: about this August. A quarter is the usual choice and keeps most of the
#: signal.
#:
#: **Still toward `START`, not toward the team's league mean.** Anchoring to the
#: league instead looks right -- squad turnover is a club property, and a global
#: anchor does decay a league gap 25% a summer -- but it was tried and rejected
#: on measurement. Removing the global anchor removes the only restoring force,
#: so league means become a random walk: with *no* European data at all it
#: drifted the five leagues 38 points apart, pure churn artefact, and inverted
#: the ordering (Spearman -0.20 against measured European strength). Against
#: that it bought 0.0011 of Brier on 788 cross-league ties, which is nothing.
#: A self-correcting anchor is worth more than that in a job that runs weekly
#: forever.
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

    Args:
    """
    # One row per match, taken from the home side so goals are oriented
    # home-first. Ordering by date is what makes the walk causal.
    #
    # Unplayed rows first within a date. A team plays once a day, so ordering
    # inside a date is normally irrelevant -- except for the one case where a
    # team really does appear twice: a placeholder for a fixture being scored
    # alongside the real, played row for that same fixture, which is exactly
    # what happens when a European tie is both fed to Elo and predicted. Read
    # the placeholder first and it cannot see its own result. This held by
    # accident before, because every domestic league code sorts before "UCL".
    home = df[df["Venue"] == "Home"].copy()
    home["_unplayed"] = home["GoalsFor"].isna() | home["GoalsAgainst"].isna()
    home = home.sort_values(
        ["Date", "_unplayed", "League", "Team"], ascending=[True, False, True, True]
    )

    ratings: dict[str, float] = {}
    seen_season: dict[str, str] = {}
    pre: dict[tuple, tuple[float, float]] = {}

    for row in home.itertuples():
        season = str(row.Season)
        pair = []
        for team in (row.Team, row.Opponent):
            rating = ratings.get(team, START)
            # A new season for this team: pull back toward the anchor before the
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
        # A final is at a neutral ground, so neither side gets the home term.
        # Compared against True explicitly: concatenating a frame that has this
        # column with one that does not fills the gaps with NaN, and `if NaN`
        # is *truthy*, which silently stripped home advantage from all 29,143
        # domestic matches. `add_elo` now normalises the column, and this is the
        # belt to that braces.
        edge = 0.0 if getattr(row, "Neutral", False) is True else home_advantage
        expected = _expected(home_rating + edge, away_rating)

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
    extra: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach each team's pre-match Elo rating to a long team-match frame.

    Args:
        tmh: Long team-match history, two rows per match, with ``Team``,
            ``Opponent``, ``Venue``, ``GoalsFor``, ``GoalsAgainst`` and
            ``Date``. Need not be sorted.
        k: Update size per match.
        home_advantage: Rating points granted to the home side.
        season_regression: Fraction pulled back toward ``START`` each new season.
        extra: Further matches, in the same long shape, that the walk should
            **learn from but not return rows for** -- UEFA ties. They are
            evidence about the teams, not part of the domestic record: putting
            them in ``tmh`` would pull them into every rolling form window and
            the rest features too, and they carry no xG, so the model would be
            fed a different thing from the one it was trained on. Feeding them
            only to Elo is the whole point, since Elo is the only feature that
            can carry strength across a league boundary.

    Returns:
        ``tmh`` with an ``elo`` column holding the rating the team carried
        *into* that match -- never one that has seen the result. One row per
        input row: ``extra`` never appears in the output.
    """
    if tmh.empty:
        return tmh.assign(elo=pd.Series(dtype="float64"))

    df = tmh.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    walked = df
    if extra is not None and not extra.empty:
        other = extra.copy()
        other["Date"] = pd.to_datetime(other["Date"])
        # Concatenated, then sorted inside `_walk` by date -- so a European
        # result informs every later match and no earlier one.
        walked = pd.concat([df, other], ignore_index=True)
        # Domestic rows have no `Neutral` column, so the concat leaves them NaN.
        # Normalise to real booleans before the walk reads them.
        walked["Neutral"] = (
            walked.get("Neutral", pd.Series(False, index=walked.index))
            .fillna(False)
            .astype(bool)
        )

    pre, _ = _walk(walked, k, home_advantage, season_regression)

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
