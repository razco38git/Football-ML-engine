"""What European results say the five leagues are worth.

Squad ratings put the Premier League 3.6 above Ligue 1, and Elo says nothing at
all -- its five league pools are disconnected and all sit at 1500. Neither
number was ever checked against a match between the two divisions, because
until now we held none.

This fits league strength directly from the ties that did happen. The model is
deliberately the simplest thing that answers the question::

    goal difference = strength(home league) - strength(away league)
                      + home advantage + noise

one parameter per league against a reference, solved by least squares. A
Poisson or Bradley-Terry fit would be more fashionable and, on a few hundred
matches with four free parameters, would land in the same place; goal difference
has the advantage that the answer is already in goals, which is the unit the
rest of the project argues in.

**MEASURED (2026-09-30), on 822 big-five vs big-five ties since 2016/17.**

League strength, goals per match against the Premier League, 90% bootstrap
intervals -- every one excludes zero:

===========  ========  ================  ==============  ============
league       measured  interval          squad ratings   Elo
===========  ========  ================  ==============  ============
E0            0.000    reference          73.2            1521
SP1          -0.346    -0.543 .. -0.154   71.5            1504
D1           -0.423    -0.646 .. -0.215   70.6            1514
I1           -0.469    -0.670 .. -0.265   70.6            1519
F1           -0.530    -0.761 .. -0.319   69.6            1519
===========  ========  ================  ==============  ============

**The squad ratings were right.** Their ordering reproduces the measured one at
Spearman **+0.975**, from an entirely independent source -- 822 actual results
rather than player data. Elo scores **+0.051**, i.e. nothing, exactly as its
disconnected league pools predict. The home-advantage term comes out at
**0.405** goals (0.299 .. 0.508), close to the domestic value, which is a
useful sign the fit is not picking up something strange.

D1 and I1 are not separated: their intervals overlap almost entirely, matching
the squad ratings, which tie them at 70.6.

**And the model could not see any of it.** Across the same ties, regressing the
home side's league-rating advantage against what the model predicted. These are
the figures *before* European results were fed into Elo; that change lifted the
first line to **+0.0034** (p 0.22), and
:mod:`footballml.league_adjust` closes the rest from outside the model:

- predicted home-win probability vs league gap: slope **-0.0002**, r -0.003,
  p 0.94 -- no relationship whatsoever
- actual home-win rate vs the same gap: slope **+0.0402**, p 1.1e-05
- so the residual tracks the gap at r +0.163 (p 4.3e-06), and across the
  observed 7.2-point range the model is missing about **29 percentage points**
  of home-win probability

That is what a missing feature looks like. The model says ~45% home win whether
the home side is from a league 3.6 points stronger or 3.6 points weaker, while
the truth runs from 36% to 61%. Its skill over a base rate halves accordingly:
12.8% of base on domestic matches, **6.2%** on cross-league ones.

**What this can and cannot measure.** European entrants are their league's
better clubs, so this is a statement about the top of each division, not its
depth. Two-legged ties distort effort -- a side three up rests players -- and
finals are at neutral venues where the home term does not apply. The confidence
intervals are bootstrapped and are the point, not decoration: with this sample
an interval that straddles zero means the leagues are not separable, and that is
a finding rather than a failure.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Reference league, whose strength is pinned to zero. Every other league is
#: reported relative to it. The Premier League is the natural choice because
#: both the squad ratings and the coefficient of variation put it on top, so
#: the other four come out negative and read as "goals behind England".
REFERENCE = "E0"

#: Bootstrap resamples. A thousand is enough to place a 90% interval to about a
#: hundredth of a goal, which is finer than anything we would act on.
N_BOOTSTRAP = 1000

#: Interval width reported. 90% rather than 95%: the sample is small enough that
#: a 95% interval on a per-league term is usually uninformative, and pretending
#: otherwise would be the same false precision this module exists to avoid.
INTERVAL = 90.0


def _design(matches: pd.DataFrame, leagues: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Design matrix and goal differences for the least-squares fit.

    One column per non-reference league holding ``+1`` when it is the home
    side's division and ``-1`` when it is the away side's, plus a constant
    column for home advantage. A match between two clubs of the *same* league
    contributes a row of zeros to the league terms -- it carries no information
    about the gap between divisions, but it does help pin home advantage.
    """
    others = [lg for lg in leagues if lg != REFERENCE]
    rows = np.zeros((len(matches), len(others) + 1), dtype=float)
    for i, league in enumerate(others):
        rows[:, i] = (matches["home_league"] == league).to_numpy(dtype=float) - (
            matches["away_league"] == league
        ).to_numpy(dtype=float)
    rows[:, -1] = 1.0  # home advantage
    gd = (matches["FTHG"] - matches["FTAG"]).to_numpy(dtype=float)
    return rows, gd


def fit_league_strength(
    matches: pd.DataFrame, n_bootstrap: int = N_BOOTSTRAP, seed: int = 7
) -> pd.DataFrame:
    """Estimate each league's strength in goals, relative to :data:`REFERENCE`.

    Args:
        matches: Kept European matches with ``home_league``, ``away_league``,
            ``FTHG`` and ``FTAG``.
        n_bootstrap: Resamples for the interval. Zero skips it.
        seed: Fixed so a reported table can be reproduced exactly.

    Returns:
        One row per league with ``strength`` (goals per match against the
        reference), ``low``/``high`` bootstrap bounds, and ``n_matches``
        (cross-league ties involving that league). Includes a ``home_advantage``
        row, which is a useful sanity check in its own right: it should land
        near the 0.3-0.4 goals seen in domestic play.
    """
    leagues = sorted(set(matches["home_league"]) | set(matches["away_league"]))
    if len(leagues) < 2 or matches.empty:
        return pd.DataFrame()

    design, gd = _design(matches, leagues)
    coefficients, *_ = np.linalg.lstsq(design, gd, rcond=None)

    samples = np.empty((n_bootstrap, design.shape[1]), dtype=float)
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        for i in range(n_bootstrap):
            pick = rng.integers(0, len(gd), len(gd))
            try:
                samples[i], *_ = np.linalg.lstsq(design[pick], gd[pick], rcond=None)
            except np.linalg.LinAlgError:  # pragma: no cover - degenerate resample
                samples[i] = np.nan

    lo, hi = (100 - INTERVAL) / 2, 100 - (100 - INTERVAL) / 2
    others = [lg for lg in leagues if lg != REFERENCE]

    cross = matches[matches["home_league"] != matches["away_league"]]
    involved = pd.concat([cross["home_league"], cross["away_league"]]).value_counts()

    rows = [
        {
            "league": REFERENCE, "strength": 0.0, "low": 0.0, "high": 0.0,
            "n_matches": int(involved.get(REFERENCE, 0)),
        }
    ]
    for i, league in enumerate(others):
        draws = samples[:, i] if n_bootstrap else np.array([np.nan])
        rows.append(
            {
                "league": league,
                "strength": float(coefficients[i]),
                "low": float(np.nanpercentile(draws, lo)) if n_bootstrap else np.nan,
                "high": float(np.nanpercentile(draws, hi)) if n_bootstrap else np.nan,
                "n_matches": int(involved.get(league, 0)),
            }
        )

    draws = samples[:, -1] if n_bootstrap else np.array([np.nan])
    rows.append(
        {
            "league": "home_advantage",
            "strength": float(coefficients[-1]),
            "low": float(np.nanpercentile(draws, lo)) if n_bootstrap else np.nan,
            "high": float(np.nanpercentile(draws, hi)) if n_bootstrap else np.nan,
            "n_matches": len(matches),
        }
    )

    out = pd.DataFrame(rows)
    ordering = out["league"] == "home_advantage"
    return pd.concat(
        [out[~ordering].sort_values("strength", ascending=False), out[ordering]]
    ).reset_index(drop=True)


def head_to_head(matches: pd.DataFrame) -> pd.DataFrame:
    """Raw record of each league against each other league.

    The fit above is a summary; this is the evidence under it. If a league's
    estimated strength is not visible here as points won, the fit is leaning on
    something subtle and should be distrusted.
    """
    cross = matches[matches["home_league"] != matches["away_league"]].copy()
    if cross.empty:
        return pd.DataFrame()

    home = pd.DataFrame(
        {
            "league": cross["home_league"], "opponent": cross["away_league"],
            "gf": cross["FTHG"], "ga": cross["FTAG"],
            "points": cross["FTR"].map({"H": 3, "D": 1, "A": 0}),
        }
    )
    away = pd.DataFrame(
        {
            "league": cross["away_league"], "opponent": cross["home_league"],
            "gf": cross["FTAG"], "ga": cross["FTHG"],
            "points": cross["FTR"].map({"A": 3, "D": 1, "H": 0}),
        }
    )
    both = pd.concat([home, away], ignore_index=True)

    out = both.groupby("league").agg(
        played=("points", "size"),
        points_per_match=("points", "mean"),
        goals_for=("gf", "sum"),
        goals_against=("ga", "sum"),
    )
    out["goal_diff_per_match"] = (out["goals_for"] - out["goals_against"]) / out["played"]
    return out.sort_values("points_per_match", ascending=False).round(3)
