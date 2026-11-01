"""A team rating built from the squad that is actually at the club.

`team_strength.py` rates **last season's squad, carried forward**. That is the
right input for the match model -- a feature read in season S+1 must not see
season S+1 -- but it is the wrong thing to *publish*. A 2021/22 Barcelona
rating built from 2020/21 players still contains Messi, and PSG's does not. A
promoted side has no rating at all, because the join that feeds the model is a
left join and they arrive as `NaN`.

This module answers the published question instead: *who is at this club now,
and how good are they?* It is a second number beside the first, not a
replacement -- see :func:`squad_strength`.

The roster is EA's, taken rather than computed
----------------------------------------------

The obvious design is to diff two editions for arrivals and departures. That
is the hard version of an easy problem. `fifa_2122.csv` simply *says* Messi is
at Paris Saint-Germain, so a roster needs no predecessor edition -- which also
avoids a dead end, since there is no `fifa_1314.csv` and 2014/15 therefore has
no arrivals baseline. :mod:`footballml.players.ea_squads` works out which EA
club label a team carries in each edition; this module reads the squad under
that label.

EA is the only source that covers a promoted side. Measured over 138 promoted
club-seasons, a median of **43.5%** of their players have a prior top-flight
rated season, and the half we do have is the more experienced half -- rating
them from our own history alone would be a selection bias dressed as a gap.
Every promoted club checked across four editions is present in EA with a 26-36
man squad.

A player's rating, in two cases
-------------------------------

1. **He has a rated season before this one.** Use the most recent, which is
   what the squad-strength feature would have used anyway.
2. **He does not** -- a youth player, a signing from outside the big five, or
   anyone under the 450-minute gate. Map EA's `overall` onto our scale through
   the curve the published ratings already use.

The second case needs care, and reuses rather than re-derives. `rating.py`
blends a performance rating with EA's overall, and it does not average the raw
numbers: EA's cluster in the sixties and seventies, ours are spread by
construction, so EA's overall is first ranked *within (role, season)* and that
percentile is interpolated onto 0-99. The published `fifa_on_our_scale` column
is the result. So the mapping here is read straight off the published pairs --
every distinct `fifa_overall` in a (role, season) pool and the
`fifa_on_our_scale` it was published at -- rather than re-ranking a different
population and getting a slightly different curve.

Two consequences of that, both deliberate. The curve is clamped at both ends,
so a 95-rated player in a pool whose best is 91 is rated as a 91: our own data
has nothing to say above its own range, and inventing an extrapolation there
would be the least defensible number on the page. And a player whose
(role, season) pool has fewer than :data:`MIN_CURVE_POINTS` published pairs
falls back to the season's whole-population curve -- that is 2015/16
goalkeepers, who are not rated at all because FBref served no `keeper` table.

Minutes, which a pre-season squad does not have
-----------------------------------------------

`team_strength` weights each line's mean by minutes played. A September roster
has none, and this season's minutes are not knowable in August -- nor allowed,
since `tests/test_strength_leakage.py` forbids a rating that sees its own
season. So the weight is the player's **previous** season's minutes: knowable
on the day the edition ships, and enough to stop a fringe signing counting the
same as an ever-present.

Players with no prior minutes take the mean of their line's, so they are
weighted like an average squad member rather than dropped. Weighting them zero
was the alternative and is worse than it looks: it does not merely ignore them,
it silently removes every promoted side's unknowns from a rating whose whole
purpose is to cover them, and a squad with *no* prior minutes anywhere -- which
`_weighted_mean` already handles by falling back to a plain mean -- would then
be rated while a mixed squad quietly lost half its players.

What this is not
----------------

It is a **September snapshot**. It misses January transfers, and the page says
so. It is also not a model feature and must not become one: removing squad
strength from the model entirely costs +0.00015 RPS [-0.00001, +0.00032], which
does not clear zero over 20,013 matches, so there is no prediction gain
available here to chase. The model keeps reading `team_strength.csv`, untouched.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from footballml.players.ea_squads import FIFA_DIR, Resolution, resolve
from footballml.players.fbref import normalise_name
from footballml.players.fifa import _normalise_export
from footballml.players.team_strength import DEFAULT_BEST_N, team_strength

logger = logging.getLogger(__name__)

#: Published `(fifa_overall, fifa_on_our_scale)` pairs a (role, season) pool
#: needs before its own curve is used instead of the season's.
#:
#: Four points cannot describe the shape of a mapping that spans the whole
#: scale, and the pools this protects are the ones with a documented hole in
#: them rather than merely small ones: 2015/16 goalkeepers are unrated because
#: FBref served no `keeper` table that season, so the pool is empty, not thin.
MIN_CURVE_POINTS = 5


def _history(players: pd.DataFrame) -> dict[str, list[tuple[str, float, float]]]:
    """Each EA-matched player's rated seasons, oldest first.

    Keyed by the normalised EA name, because that is the identifier the roster
    carries: our own `Player` spelling is Understat's and the roster's is EA's,
    and the two agree on neither accents nor given names. `fifa_name` is the EA
    name this player-season was matched to, so normalising it is the join.

    A player who moved mid-season appears twice in one season; the row with
    more minutes is the better summary of that season and wins.
    """
    rated = players[players["rated"] & players["rating"].notna()]
    rated = rated[rated["fifa_name"].notna()]
    if rated.empty:
        return {}

    frame = pd.DataFrame(
        {
            "_norm": rated["fifa_name"].astype(str).map(normalise_name),
            "Season": rated["Season"].astype(str),
            "rating": rated["rating"].astype(float),
            "minutes": pd.to_numeric(rated["minutes"], errors="coerce").fillna(0.0),
        }
    )
    frame = frame.sort_values("minutes", ascending=False).drop_duplicates(
        ["_norm", "Season"]
    )

    out: dict[str, list[tuple[str, float, float]]] = {}
    for key, block in frame.sort_values("Season").groupby("_norm", observed=True):
        out[str(key)] = list(
            zip(block["Season"], block["rating"], block["minutes"], strict=True)
        )
    return out


def _latest_before(
    seasons: list[tuple[str, float, float]],
    season: str,
    include_current: bool = False,
) -> tuple[float, float] | None:
    """The most recent ``(rating, minutes)`` strictly before ``season``.

    Season labels sort chronologically as strings over our range -- `1415` to
    `2627` -- which is what `ea_squads.resolve` already relies on.

    ``include_current`` allows ``season`` itself, which is **not** what gets
    published and exists because the alternative had to be measured. A rating
    that reads the season it describes is no longer a pre-season projection --
    and for the one case this module is for, the season in progress, there is
    almost nothing there to read. See the `EXPERIMENTS.md` section on this.
    """
    best: tuple[float, float] | None = None
    for had, rating, minutes in seasons:
        if had > season or (had == season and not include_current):
            break
        best = (rating, minutes)
    return best


def _curves(players: pd.DataFrame) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """The published EA-overall-to-our-scale mapping, per (role, season).

    Read off `fifa_on_our_scale` rather than recomputed: see the module
    docstring. The season's whole-population curve is stored under role
    ``""`` as the fallback for a pool too thin to carry its own.
    """
    needed = {"fifa_overall", "fifa_on_our_scale", "role", "Season"}
    if not needed.issubset(players.columns):
        logger.warning("No published EA scale mapping; unrated players get no rating")
        return {}

    frame = players[list(needed)].dropna(subset=["fifa_overall", "fifa_on_our_scale"])
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for keys, block in frame.groupby(["role", "Season"], observed=True):
        role, season = (str(k) for k in keys)
        curves[(role, season)] = _curve(block)
    for season, block in frame.groupby("Season", observed=True):
        curves[("", str(season))] = _curve(block)
    return curves


def _curve(block: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One monotone curve: distinct EA overalls against our scale."""
    grouped = block.groupby("fifa_overall", observed=True)["fifa_on_our_scale"].mean()
    grouped = grouped.sort_index()
    return grouped.index.to_numpy(dtype="float64"), grouped.to_numpy(dtype="float64")


def _on_our_scale(
    overall: pd.Series,
    role: pd.Series,
    season: str,
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
) -> pd.Series:
    """Map EA overalls onto our scale, each through its own role's curve."""
    out = pd.Series(np.nan, index=overall.index, dtype="float64")
    fallback = curves.get(("", season))
    for value, block in role.groupby(role.fillna(""), observed=True):
        curve = curves.get((str(value), season))
        if curve is None or len(curve[0]) < MIN_CURVE_POINTS:
            curve = fallback
        if curve is None:
            continue
        xs, ys = curve
        out.loc[block.index] = np.interp(
            overall.loc[block.index].to_numpy(dtype="float64"), xs, ys
        )
    return out


def rosters(
    players: pd.DataFrame,
    directory: Path | None = None,
    include_current: bool = False,
) -> tuple[pd.DataFrame, list[Resolution]]:
    """Every resolved club-season's EA squad, each player rated.

    Args:
        players: Output of :func:`footballml.players.rating.rate_players`.
        directory: Where the EA editions live. Defaults to `data/raw/fifa`.

    Returns:
        A frame shaped like ``players`` as far as
        :func:`footballml.players.team_strength.team_strength` is concerned --
        `League`, `Season`, `Team`, `Player`, `role`, `rating`, `minutes`,
        `rated` -- plus `ea_label`, `fifa_overall` and `from_history`, which
        records whether the rating came from a prior season or from EA. And the
        resolutions, so a caller can report what did not resolve.
    """
    directory = Path(directory or FIFA_DIR)
    frame = players.copy()
    frame["Season"] = frame["Season"].astype(str)

    resolutions = resolve(frame[frame["rated"]], directory)
    history = _history(frame)
    curves = _curves(frame)

    # A club's league in a season, from our own data: the roster comes from EA,
    # which carries no league column at all, and `team_strength` keys on it.
    leagues = (
        frame.groupby(["Season", "Team"], observed=True)["League"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    blocks: list[pd.DataFrame] = []
    for season, group in _by_season(resolutions):
        path = directory / f"fifa_{season}.csv"
        if not path.exists():
            continue
        edition = pd.read_csv(path, low_memory=False)
        club = _club_column(edition)
        if club is None:
            logger.warning("%s has no club column; no rosters from it", path.name)
            continue
        by_label = {
            str(label).strip(): block
            for label, block in edition.groupby(edition[club].astype(str).str.strip())
        }

        for resolution in group:
            raw = by_label.get(resolution.label)
            if raw is None or raw.empty:
                continue
            # Normalised one club at a time, so the loader's drop of duplicate
            # names cannot take a squad member away in favour of a better-rated
            # namesake at some other club in the same edition.
            squad = _normalise_export(raw, f"fifa_{season}.csv {resolution.label}")
            if squad is None or squad.empty:
                continue
            blocks.append(
                _rate_squad(
                    squad, resolution, leagues, history, curves, include_current
                )
            )

    if not blocks:
        logger.warning("No EA rosters built; every club-season was unresolved")
        return pd.DataFrame(), resolutions

    roster = pd.concat(blocks, ignore_index=True)
    logger.info(
        "Built %d rosters covering %d players; %.1f%% rated from a prior season",
        roster.groupby(["Season", "Team"], observed=True).ngroups,
        len(roster),
        100.0 * float(roster["from_history"].mean()),
    )
    return roster, resolutions


def _by_season(resolutions: list[Resolution]) -> list[tuple[str, list[Resolution]]]:
    """Resolved club-seasons grouped by season, so each edition is read once."""
    grouped: dict[str, list[Resolution]] = {}
    for resolution in resolutions:
        if resolution.resolved:
            grouped.setdefault(resolution.season, []).append(resolution)
    return sorted(grouped.items())


def _club_column(edition: pd.DataFrame) -> str | None:
    """The club column, whichever of the export spellings this file uses."""
    for candidate in ("club", "club_name", "team", "Club"):
        if candidate in edition.columns:
            return candidate
    return None


def _rate_squad(
    squad: pd.DataFrame,
    resolution: Resolution,
    leagues: dict[tuple[str, str], str],
    history: dict[str, list[tuple[str, float, float]]],
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    include_current: bool = False,
) -> pd.DataFrame:
    """One club-season's roster, with a rating and a weight per player."""
    season, team = resolution.season, resolution.team
    out = pd.DataFrame(
        {
            "League": leagues.get((season, team)),
            "Season": season,
            "Team": team,
            "ea_label": resolution.label,
            "Player": squad["fifa_name"].astype(str),
            "role": squad["role"],
            "fifa_overall": squad["fifa_overall"].astype(float),
        }
    )

    prior = [
        _latest_before(history.get(key, []), season, include_current)
        for key in squad["_norm"]
    ]
    out["rating"] = [p[0] if p else np.nan for p in prior]
    out["minutes"] = [p[1] if p else np.nan for p in prior]
    out["from_history"] = out["rating"].notna()

    missing = ~out["from_history"]
    if missing.any():
        out.loc[missing, "rating"] = _on_our_scale(
            out.loc[missing, "fifa_overall"], out.loc[missing, "role"], season, curves
        )

    out["rated"] = out["rating"].notna()
    return out


def weight_by_line(roster: pd.DataFrame, line_by_role: dict[str, str]) -> pd.Series:
    """Prior minutes, with the unknowns set to their line's mean.

    `_weighted_mean` reads `minutes`, and a player with none would otherwise
    weigh nothing at all -- see the module docstring on why that is worse than
    it looks. A line where nobody has prior minutes comes back all zero, which
    `_weighted_mean` already treats as "no weights" and averages plainly.
    """
    minutes = pd.to_numeric(roster["minutes"], errors="coerce")
    line = roster["role"].map(line_by_role)
    means = minutes.groupby([roster["Season"], roster["Team"], line]).transform("mean")
    return minutes.fillna(means).fillna(0.0)


def squad_strength(
    players: pd.DataFrame,
    directory: Path | None = None,
    best_n: int = DEFAULT_BEST_N,
    include_current: bool = False,
) -> tuple[pd.DataFrame, list[Resolution]]:
    """Publishable squad rating per club-season, strongest first.

    The aggregation is `team_strength`'s own, called with the roster in place
    of our rated player-seasons: the same `best_n` cut, the same total order
    over ties, the same per-line means and the same `OVERALL_LINES` weighting.
    Nothing about how a team rating is formed is re-decided here -- only *which
    players* it is formed from -- so the two columns on the page differ by
    their roster and by nothing else.

    Returns:
        One row per club-season with `squad_overall` and a `squad_*` column per
        line, `n_players`, and `n_from_history` -- how many of the selected
        players were rated from a season of ours rather than from EA alone.
        And the resolutions, for reporting.
    """
    from footballml.players.team_strength import LINE_BY_ROLE

    roster, resolutions = rosters(players, directory, include_current)
    if roster.empty:
        return pd.DataFrame(), resolutions

    weighted = roster.copy()
    weighted["minutes"] = weight_by_line(weighted, LINE_BY_ROLE)

    strength = team_strength(weighted, method="best_n", best_n=best_n)
    if strength.empty:
        return strength, resolutions

    # Which players the aggregation actually took, so `n_from_history` counts
    # the selected squad rather than the whole roster. Repeating the selection
    # is cheaper and less fragile than making `team_strength` return it.
    picked = _selected(weighted, best_n)
    provenance = (
        picked.groupby(["League", "Season", "Team"], observed=True)
        .agg(
            n_from_history=("from_history", "sum"),
            ea_label=("ea_label", "first"),
        )
        .reset_index()
    )

    out = strength.merge(provenance, on=["League", "Season", "Team"], how="left")
    out = out.rename(
        columns={
            c: c.replace("strength_", "squad_")
            for c in out.columns
            if c.startswith("strength_")
        }
    )
    out["n_from_history"] = out["n_from_history"].astype("Int64")
    return out.drop(columns=["method"], errors="ignore"), resolutions


def _selected(roster: pd.DataFrame, best_n: int) -> pd.DataFrame:
    """The players `team_strength(method="best_n")` keeps, by its own order."""
    from footballml.players.team_strength import _BEST_N_ORDER, LINE_BY_ROLE

    usable = roster[roster["rated"] & roster["rating"].notna()].copy()
    usable["line"] = usable["role"].map(LINE_BY_ROLE)
    usable = usable.dropna(subset=["line"])
    by = [c for c in _BEST_N_ORDER if c in usable.columns]
    return (
        usable.sort_values(
            by, ascending=[_BEST_N_ORDER[c] for c in by], kind="stable"
        )
        .groupby(["League", "Season", "Team"], observed=True)
        .head(best_n)
    )
