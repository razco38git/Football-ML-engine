"""Player season statistics from Understat.

Understat is used rather than FBref for three reasons found by measurement:
its player endpoint returns in under a second where FBref takes 45 and needs a
headless Chrome; soccerdata exposes only five of FBref's stat tables, omitting
passing, defending and possession; and FBref's ``standard`` table comes back
without any xG columns at all, leaving goals, assists and cards.

What Understat gives is genuinely well suited to rating attacking contribution:

- **npxG / xA** -- chance quality created, stripped of penalties
- **key passes / shots** -- volume behind that quality
- **xGChain** -- total xG of every possession the player was involved in
- **xGBuildup** -- the same, excluding shots and key passes, so it isolates
  contribution to moves the player did not finish

That last pair matters most. Goals and assists reward only the final two touches;
xGBuildup credits the defender who started the move and the midfielder who
carried it, which is exactly the contribution a naive rating misses.

.. warning::
    Goalkeepers cannot be meaningfully rated from this source. Understat carries
    no saves, no post-shot xG, nothing defensive -- a keeper's row is essentially
    empty. They are ingested but deliberately left unrated; see
    :mod:`footballml.players.rating`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from footballml.ingest.understat import UNDERSTAT_LEAGUES, season_labels

logger = logging.getLogger(__name__)

#: Understat encodes position as space-separated letters covering every role the
#: player occupied, plus ``S`` for substitute appearances. Precedence below is
#: ordered by how much the role defines a player's statistical profile.
#:
#: ``D`` outranks ``F`` because a defender who occasionally pushes up is still
#: judged as a defender, while ``F`` outranks ``M`` because most forwards are
#: listed "F M S" -- resolving those to midfield would compare strikers against
#: midfielders and inflate them badly.
POSITION_PRECEDENCE = ("GK", "D", "F", "M")

#: Counting stats converted to per-90 rates before any comparison. Raw totals
#: would simply rank by minutes played.
PER_90_METRICS = (
    "np_xg", "xa", "shots", "key_passes", "xg_chain", "xg_buildup",
    "np_goals", "assists", "yellow_cards", "red_cards",
)


def position_group(position: str | float) -> str | None:
    """Reduce an Understat position string to one group.

    ``"F M S"`` -> ``"F"``, ``"D M S"`` -> ``"D"``, ``"GK"`` -> ``"GK"``.
    Returns ``None`` for players listed only as ``"S"``, who have no recorded
    role at all.
    """
    if not isinstance(position, str):
        return None
    tokens = set(position.split())
    for group in POSITION_PRECEDENCE:
        if group in tokens:
            return group
    return None


#: Involvement metrics expressed as a share of the team's own output.
TEAM_RELATIVE_METRICS = ("xg_chain", "xg_buildup", "np_xg", "xa")


def _add_team_relative(df: pd.DataFrame) -> pd.DataFrame:
    """Express involvement as a share of the team's attacking output.

    xGChain and xGBuildup credit a player for every possession they touched that
    ended in a shot. That makes them heavily team-dependent: a full-back at a
    side with 65% possession accumulates enormous involvement without being
    especially good, which is why an unadjusted rating fills its top twenty with
    Bayern and Leverkusen defenders.

    Dividing by the team's own per-90 total asks a better question -- *of
    everything this team created, how much ran through this player?* -- and that
    is a property of the player rather than of their employer.
    """
    out = df.copy()
    per90 = {m: f"{m}_per90" for m in TEAM_RELATIVE_METRICS if f"{m}_per90" in out.columns}
    if not per90:
        return out

    # Team baseline: minutes-weighted mean across the squad, which approximates
    # the team's own per-90 rate without needing a separate team dataset.
    weights = out["nineties"].clip(lower=0)
    for metric, column in per90.items():
        totals = out.groupby(["League", "Season", "team"], observed=True)[metric].transform("sum")
        team_nineties = weights.groupby(
            [out["League"], out["Season"], out["team"]]
        ).transform("sum")
        # 11 players on the pitch, so squad nineties are ~11x team nineties.
        team_rate = (totals / team_nineties.replace(0, np.nan)) * 11.0
        out[f"{metric}_share"] = out[column].div(team_rate).where(team_rate > 0)

    return out


def _career_position(df: pd.DataFrame) -> pd.Series:
    """Resolve each player to one position across their whole career.

    Per-season labels are noisy in a way that badly distorts the rating.
    Understat lists every role a player occupied, so a midfielder who covered at
    full-back for three matches comes back as ``"D M S"`` and, under the
    precedence rule alone, is judged against defenders for the entire season.
    That is how Szoboszlai, Laimer and Zaïre-Emery ended up rated as defenders.

    Taking the label the player carried across the most minutes fixes it: an
    occasional role change cannot outweigh the position they actually play.
    """
    minutes = pd.to_numeric(df["minutes"], errors="coerce").fillna(0)
    tally = (
        df.assign(_minutes=minutes)
        .dropna(subset=["season_position"])
        .groupby(["player", "season_position"], observed=True)["_minutes"]
        .sum()
    )
    if tally.empty:
        return df["season_position"]

    dominant = tally.reset_index().sort_values("_minutes", ascending=False)
    lookup = dominant.drop_duplicates(subset=["player"], keep="first").set_index("player")[
        "season_position"
    ]
    # Fall back to the season label for players with no usable career history.
    return df["player"].map(lookup).fillna(df["season_position"])


def fetch_player_seasons(
    leagues: list[str] | None = None, seasons: list[str] | None = None
) -> pd.DataFrame:
    """Fetch per-season player totals and derive per-90 rates.

    Returns:
        One row per player per team per season, with raw totals, ``*_per90``
        rates, ``position_group`` and ``nineties`` (minutes / 90).
    """
    import soccerdata as sd  # imported lazily: pulls a heavy browser stack

    codes = leagues or sorted(UNDERSTAT_LEAGUES)
    seasons = seasons or season_labels()

    raw = (
        sd.Understat(
            leagues=[UNDERSTAT_LEAGUES[c] for c in codes], seasons=seasons
        )
        .read_player_season_stats()
        .reset_index()
    )
    logger.info("Fetched %d player-seasons", len(raw))

    inverse = {v: k for k, v in UNDERSTAT_LEAGUES.items()}
    df = raw.assign(
        League=raw["league"].map(inverse),
        Season=raw["season"].astype(str),
        season_position=raw["position"].map(position_group),
    )
    df["position_group"] = _career_position(df)

    numeric = [
        "minutes", "matches", "goals", "xg", "np_goals", "np_xg", "assists",
        "xa", "shots", "key_passes", "yellow_cards", "red_cards",
        "xg_chain", "xg_buildup",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["nineties"] = df["minutes"] / 90.0
    for metric in PER_90_METRICS:
        # Guard against a zero-minute row producing inf.
        df[f"{metric}_per90"] = df[metric].div(df["nineties"]).where(df["nineties"] > 0)

    # Finishing relative to chance quality. Positive means the player scored
    # more than the chances warranted -- a real skill in small part, and noise
    # in large part, which is why it carries little weight in the rating.
    df["finishing_delta_per90"] = (
        (df["np_goals"] - df["np_xg"]).div(df["nineties"]).where(df["nineties"] > 0)
    )

    df = _add_team_relative(df)

    keep = [
        "League", "Season", "player", "team", "position", "position_group",
        "season_position", "nineties", *numeric,
        *[f"{m}_per90" for m in PER_90_METRICS], "finishing_delta_per90",
        *[f"{m}_share" for m in TEAM_RELATIVE_METRICS],
    ]
    # dict.fromkeys preserves order while dropping repeats: `numeric` already
    # contains minutes and matches, and a duplicated name makes df[col] return a
    # DataFrame rather than a Series, which fails much later and confusingly.
    unique = [c for c in dict.fromkeys(keep) if c in df.columns]
    out = df[unique].rename(columns={"player": "Player", "team": "Team"})
    return out.sort_values(["Season", "League", "Player"]).reset_index(drop=True)
