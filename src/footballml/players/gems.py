"""Players we rated far above EA, and whether EA agreed a year later.

The two halves of a published rating are already comparable: `rating.py` ranks
EA's overall *within role and season* and maps that percentile onto our 0-99
scale, so `fifa_on_our_scale` and `performance_rating` live on one axis by
construction. Subtracting them is the whole idea here, and the result is
centred (mean +0.06, sd 9.37 over 23,683 rated player-seasons) and stable
across both seasons and roles, so no further normalisation is needed.

A board of big gaps would be a curiosity. What makes it worth publishing is
that the gap is **checkable**: a player we rate far above EA should be someone
EA raises in the next edition, and that can be looked up rather than argued.

What the check says
-------------------

Over 12,776 player-seasons from 2016/17 on, joined to the same player's EA
overall one edition later:

    gap decile      mean gap   EA's next move   share EA raised
    1 (lowest)        -16.0           -0.93            17.4%
    5                  -1.2           +0.48            42.9%
    10 (highest)      +17.7           +4.95            91.3%

Raw correlation +0.58. **The obvious objection is that this is EA reverting to
its own mean** -- `corr(fifa_overall, next move)` is -0.55, EA raises the
low-rated whoever they are, and the gap is mechanically anti-correlated with
`fifa_overall`. Two tests say it is not that:

- partialling `fifa_overall`, age and minutes out of both sides: **+0.375**
- comparing only players **EA rated identically** -- same season, same role,
  same overall, 487 strata holding 7,744 players, no model at all: **+0.370**

A regression and a model-free stratification agreeing to three decimals is as
good as this gets. :func:`agreement` recomputes the second one, and it is a
published number rather than a comment because it is the board's entire claim.

What has to be filtered, and why only one side of it
----------------------------------------------------

**The negative tail is polluted and the positive tail is not.** Rows whose EA
entry is attached to two different clubs in one season are 1.72% of the data,
but **6 of the 30 largest negative gaps and 0 of the 30 largest positive**.
The asymmetry is structural: matching a player to a *highly rated* wrong entry
manufactures a huge negative gap, while the reverse needs a star to match a
low-rated entry, which the matcher's uniqueness guards make rare. The two
worst negatives are exactly this -- Bruno Saltor of Brighton and Bruno
González of Getafe are **both** matched to `Bruno Soriano Llido`, Villarreal's
Bruno, EA 84.

So :func:`hidden_gems` publishes the positive side and still applies
:func:`_shared_entry` to it, because the guard costs nothing there and the
failure it catches is invisible without it.

Two exclusions come from `rating.py`'s own documented gaps: 2014/15 and
2015/16 served no `misc` and no `keeper` table, so `measured_share` falls to
0.38 for a centre back and the performance half is not comparable with a
modern one; and a minutes floor above the rating's own 450, because the tails
of a gap are where thin seasons land.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Seasons whose performance half is not comparable. FBref served no `misc`
#: table for the first and no `keeper` table for the second -- see
#: :data:`footballml.players.rating.MIN_METRIC_COVERAGE`.
INCOMPLETE_SEASONS = frozenset({"1415", "1516"})

#: Minutes a season needs before its gap is published, above the 450 the
#: rating already requires. A gap is a difference of two percentile-derived
#: numbers, so its tails fill with players who barely played.
MIN_MINUTES = 900

#: Members a `(season, role, EA overall)` stratum needs before it can carry a
#: within-stratum correlation. Ten is what 487 strata clear.
MIN_STRATUM = 10


def gaps(players: pd.DataFrame) -> pd.DataFrame:
    """Every rated player-season with both halves, and the gap between them.

    Adds `gap` (`performance_rating - fifa_on_our_scale`, positive meaning we
    rate the player above EA) and `shared_entry`, which flags the mismatch
    signature described in the module docstring.
    """
    needed = {"performance_rating", "fifa_on_our_scale", "rated", "rating"}
    if players.empty or not needed <= set(players.columns):
        return pd.DataFrame()

    frame = players[players["rated"] & players["rating"].notna()].copy()
    frame = frame.dropna(subset=["performance_rating", "fifa_on_our_scale"])
    if frame.empty:
        return frame

    frame["Season"] = frame["Season"].astype(str)
    frame["gap"] = frame["performance_rating"] - frame["fifa_on_our_scale"]
    frame["shared_entry"] = _shared_entry(frame)
    return frame


def _shared_entry(frame: pd.DataFrame) -> pd.Series:
    """Whether this row's EA entry is attached to two clubs in one season.

    The mismatch signature, and deliberately an *upper* bound: a player who
    moves in January produces the same shape. Used to drop rows from a board
    rather than to accuse the matcher of anything, so over-flagging costs a
    row and under-flagging costs a wrong name in front of a reader.
    """
    if "fifa_name" not in frame.columns or "Team" not in frame.columns:
        return pd.Series(False, index=frame.index)
    key = frame["fifa_name"].astype(str).str.strip().str.casefold()
    clubs = frame.groupby([frame["Season"], key], observed=True)["Team"].transform("nunique")
    return clubs > 1


def _next_season(season: str) -> str:
    """`2324` -> `2425`. Mirrors `features.build._next_season` on our labels."""
    start = int(season[:2])
    return f"{(start + 1) % 100:02d}{(start + 2) % 100:02d}"


def with_ea_verdict(frame: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach what EA did to each player in the *next* edition.

    `ea_next` is the overall EA gave the same player one edition later and
    `ea_move` the change. Both are `NaN` where EA has no next entry -- a
    player who left the big five, retired, or simply went unmatched -- and
    that is published as "no entry" rather than as a zero, because a zero
    would read as EA having looked and declined to move him.
    """
    if frame.empty or "fifa_overall" not in players.columns:
        return frame.assign(ea_next=np.nan, ea_move=np.nan)

    rated = players[players["rated"]].copy()
    rated["Season"] = rated["Season"].astype(str)
    later = (
        rated.dropna(subset=["fifa_overall"])
        .sort_values("minutes", ascending=False)
        .drop_duplicates(["Player", "Season"])[["Player", "Season", "fifa_overall"]]
        .rename(columns={"Season": "_next", "fifa_overall": "ea_next"})
    )
    out = frame.assign(_next=frame["Season"].map(_next_season)).merge(
        later, on=["Player", "_next"], how="left"
    )
    out["ea_move"] = out["ea_next"] - out["fifa_overall"]
    return out.drop(columns=["_next"])


def agreement(frame: pd.DataFrame) -> dict[str, float]:
    """Does EA move toward us, among players EA rated *identically*?

    The board's claim, recomputed from the data behind the board rather than
    quoted from this docstring. Within each `(season, role, EA overall)`
    stratum both our rating and EA's next move are standardised, which removes
    EA's mean reversion completely: every player compared started on the same
    EA number.

    Returns the correlation, how many players and strata it rests on, and the
    mean EA move in the top and bottom fifth. An empty dict when nothing
    clears :data:`MIN_STRATUM`, so a caller can leave the claim off the page
    rather than print a number from four players.
    """
    usable = frame.dropna(subset=["ea_move", "performance_rating", "fifa_overall"])
    usable = usable[~usable["Season"].isin(INCOMPLETE_SEASONS)]
    if usable.empty or "role" not in usable.columns:
        return {}

    keys = ["Season", "role", "fifa_overall"]
    grouped = usable.groupby(keys, observed=True)
    kept = grouped.filter(lambda block: len(block) >= MIN_STRATUM).copy()
    if kept.empty:
        return {}

    def standardise(column: str) -> pd.Series:
        spread = kept.groupby(keys, observed=True)[column].transform("std")
        centre = kept.groupby(keys, observed=True)[column].transform("mean")
        return (kept[column] - centre) / spread.replace(0.0, np.nan)

    kept["_ours"] = standardise("performance_rating")
    kept["_theirs"] = standardise("ea_move")
    inner = kept.dropna(subset=["_ours", "_theirs"])
    if len(inner) < MIN_STRATUM:
        return {}

    cut = inner["_ours"].quantile([0.2, 0.8])
    top = inner[inner["_ours"] >= cut.iloc[1]]
    bottom = inner[inner["_ours"] <= cut.iloc[0]]
    return {
        "correlation": float(inner["_ours"].corr(inner["_theirs"])),
        "players": int(len(inner)),
        "strata": int(inner.groupby(keys, observed=True).ngroups),
        "top_fifth_move": float(top["ea_move"].mean()),
        "bottom_fifth_move": float(bottom["ea_move"].mean()),
        "top_fifth_raised": float((top["ea_move"] > 0).mean()),
        "bottom_fifth_raised": float((bottom["ea_move"] > 0).mean()),
    }


def hidden_gems(
    players: pd.DataFrame,
    limit: int = 20,
    min_minutes: int = MIN_MINUTES,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """The biggest positive gaps, with EA's later verdict and the claim behind it.

    Args:
        players: Output of :func:`footballml.players.rating.rate_players`.
        limit: How many rows to return.
        min_minutes: Minutes floor, above the rating's own 450.

    Returns:
        The board, biggest gap first, and :func:`agreement`'s summary of
        whether EA moves toward us -- computed over the whole eligible
        population, not over the board, because a statistic read off the
        twenty rows it is being used to justify would be circular.
    """
    frame = gaps(players)
    if frame.empty:
        logger.info("No rated player-seasons carry both halves; no gems board")
        return pd.DataFrame(), {}

    eligible = frame[
        ~frame["Season"].isin(INCOMPLETE_SEASONS)
        & ~frame["shared_entry"]
        & (pd.to_numeric(frame["minutes"], errors="coerce") >= min_minutes)
    ]
    if eligible.empty:
        return pd.DataFrame(), {}

    judged = with_ea_verdict(eligible, players)
    board = judged.nlargest(limit, "gap")
    logger.info(
        "Hidden gems: %d eligible of %d rated; board spans %.1f to %.1f",
        len(eligible),
        len(frame),
        float(board["gap"].min()),
        float(board["gap"].max()),
    )
    return board, agreement(judged)
