"""The 0-99 player rating.

Four steps, each of which can be inspected on its own:

1. **Percentiles.** Every per-90 metric is ranked within its position group and
   season. This is what makes the rating comparable: a defender is only ever
   measured against defenders, and a high-scoring season inflates nobody because
   everyone's percentile moves together.
2. **Sub-ratings.** Percentiles combine into named groups -- finishing,
   creation, involvement -- using weights from ``config/player_rating.yaml``.
   These are what the UI shows as attribute bars.
3. **Shrinkage.** The composite is pulled toward the position mean in proportion
   to how little the player has played. Without it the leaderboard fills with
   players who had one good afternoon.
4. **Scale.** The shrunk composite's percentile maps onto 0-99 through the
   anchors in the config.

A fifth step sits outside those four: the scaled performance rating is blended
with EA's overall, half and half, by ``_blend_with_fifa``. ``rating`` is that
blend; ``performance_rating`` is the number this module computes on its own, and
both are carried so a reader can see which half is doing the work.

Everything debatable is a number in the YAML, not a decision buried in code.

.. note::
    **What this can and cannot see.** Goalkeepers *are* rated, on save
    percentage, goals against, clean sheets and penalties saved -- 72% of a
    keeper's composite is shot-stopping. Outfield defending is real too:
    possession-adjusted interceptions and tackles carry 62% of a centre-back's
    composite and reach 92% of them, via FBref.

    What is genuinely missing is anything needing event-level data -- every
    touch with its location and outcome. So there is no xThreat, no VAEP, no
    carries or progressive actions, no duels, no post-shot xG, and no
    open-play/set-piece split. Those are not gaps in the method but in the
    sources: Understat and FBref publish season aggregates, and no weighting of
    aggregates reconstructs them.

    The consequence worth knowing: defending is measured by *volume* of
    defensive actions, not by their quality or by chances prevented. A
    centre-back who positions well and rarely needs to tackle rates below one
    who tackles constantly. Measured against next-season goals conceded, the
    defence line reaches -0.48 where squad overall reaches -0.52 -- the defence
    number is the weaker of the two, and the match model has learned to
    discount it.

    Van Dijk is the clearest case and worth stating plainly, because it looks
    like a bug and is not. His interception share is respectable and swings
    with his role (15th to 80th percentile across his Liverpool seasons); his
    *tackles won* share sits at the 3rd to 10th percentile in every one of
    them, and tackles carry 2.5 of the 6.0 defending weight. He does not
    tackle because he does not have to. Two fixes were measured and both made
    the rating worse, which is why neither was taken:

    ====================================  =========  ========
    centre-back defending weights         def > GA   Van Dijk
    ====================================  =========  ========
    int 3.0 + tackles 2.5 (this)            -0.439     77
    tackles halved                          -0.433     83
    interceptions only                      -0.426     92
    one combined ball-winning metric        -0.430     73
    ====================================  =========  ========

    Tackles are the *better* half, not the worse one: dropping them raises Van
    Dijk fifteen points and costs the only thing the number is for. The limit
    is the data. Aerial duels would settle it and FBref no longer publishes
    them -- see :data:`footballml.players.fbref.MISC_COLUMNS`.

    Partly addressed, and worth knowing how far. `defending` now carries two
    *quality* terms beside the volume ones -- what share of a player's
    challenges ended with the ball rather than a free kick, and how much he
    wins per foul given away. Van Dijk is 3rd to 11th percentile on volume and
    70th to 97th on those, which is the whole complaint in two numbers, and his
    published rating moves 79 to 82 across his Liverpool seasons.

    It is a proxy, not the measurement anyone would choose. FBref strips tackle
    *attempts* from player rows, so a true success rate is out of reach and a
    foul stands in for a failed challenge. Three things that would settle it
    outright are not available at player level in any of the 65 cached
    league-seasons: tackle success rate, passing accuracy and clearances are
    served at 4% -- the squad summary rows only -- and aerial duels and
    recoveries are absent from the HTML entirely.

    The `playing_time` table was probed for the same reason and rejected on
    measurement. It *is* served, with goals conceded on the pitch and an
    On-Off column, but neither works: on-pitch goals conceded varies nearly
    twice as much between clubs as within them, so it says which side a player
    turns out for, and On-Off is wildest for exactly the ever-presents it would
    need to judge -- a median absolute value of 1.25 for players who sat out
    fewer than four matches, against 0.44 for those who sat out fourteen, with
    one reading of 30.29.

    Where a metric is missing rather than merely crude, the rating now says
    so instead of filling the gap: see :data:`MIN_METRIC_COVERAGE` and
    :data:`MAX_MISSING_WEIGHT`. 2014/15 centre-backs carry no defending score
    at all, and 2015/16 goalkeepers are not rated, because FBref served no
    ``misc`` table for the first and no ``keeper`` table for the second.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from footballml.data import PROJECT_ROOT

logger = logging.getLogger(__name__)

RATING_CONFIG = PROJECT_ROOT / "config" / "player_rating.yaml"

#: Position groups with no configured rating, keyed to the reason shown to
#: users. Empty now that FBref supplies goalkeeping stats, but kept as the
#: mechanism for refusing to invent a number when data is missing.
UNRATED_GROUPS: dict[str, str] = {}


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read the rating configuration."""
    return yaml.safe_load((path or RATING_CONFIG).read_text(encoding="utf-8"))


def percentile_within(
    df: pd.DataFrame, metric: str, by: list[str], higher_is_better: bool = True
) -> pd.Series:
    """Rank a metric into 0-1 percentiles within each group.

    Uses average ranking so tied players share a percentile, and normalises by
    the group size so groups of different sizes stay comparable.
    """
    ranked = df.groupby(by, observed=True)[metric].rank(pct=True, method="average")
    return ranked if higher_is_better else 1.0 - ranked


#: How much of a sub-rating's configured weight may be missing before the
#: sub-rating is refused rather than computed from what is left.
#:
#: Half. A metric or two short of a well-specified group is the case the 0.5
#: fill was written for, and renormalising over the rest is honest there. Most
#: of the group missing is a different situation entirely -- see `_weighted`.
MAX_MISSING_WEIGHT = 0.5

#: How much of a (role, season) pool a metric must cover to count as present.
#:
#: A percentile is a ranking against peers, and a peer with no value is filled
#: with 0.5 -- fine for the odd player, meaningless when most of the pool is
#: filled that way. FBref served only La Liga's `misc` table for 2015/16, so
#: interceptions and tackles reached 16-22% of each outfield pool: the Spanish
#: players spread out and everyone else sat on the median, which is how that
#: season's centre-back defending came back with a spread of 9 against ~20
#: everywhere else. Coverage in this dataset is bimodal -- every other
#: role-season-metric is at 87% or above -- so the threshold has plenty of room
#: either side of it.
MIN_METRIC_COVERAGE = 0.5

#: Warnings already emitted, so a message fires once per run rather than once
#: per (role, season) group -- 66 groups would bury it.
_WARNED: set[str] = set()


def _warn_once(message: str, *args: object) -> None:
    key = message % args
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning("%s", key)


def _weighted(
    frame: pd.DataFrame,
    weights: dict[str, float],
    max_missing_weight: float = MAX_MISSING_WEIGHT,
) -> pd.Series:
    """Weighted mean of percentile columns, ignoring any that are absent.

    A **negative weight means lower is better**: the percentile is flipped and
    the magnitude used as the weight. That is how ``goals_against_per90: -2.0``
    and ``fouls_per90: -0.5`` are expressed. Subtracting them instead would both
    corrupt the normaliser and let a metric drag a sub-rating below zero.

    Missing values default to 0.5 -- the middle of the distribution -- so a
    player with one unavailable metric is treated as average on it rather than
    worst.
    """
    # A column that is mostly missing carries little information, and filling
    # it with 0.5 would drag every player toward the middle of a sub-rating
    # while looking like a contribution. FBref serves several of its tables
    # with the values stripped -- headers present, cells empty -- so this is
    # not hypothetical: without the guard, ~25 empty columns quietly diluted
    # the metrics that did have data. See `MIN_METRIC_COVERAGE` for why the bar
    # is a share of the pool rather than "any value at all".
    usable = {
        c: w
        for c, w in weights.items()
        if c in frame.columns
        and w != 0
        and frame[c].notna().mean() >= MIN_METRIC_COVERAGE
    }

    # Skipping quietly is how a third of the centre-back defending weight went
    # missing unnoticed: `recoveries_per90` and `recoveries_padj` were weighted
    # 1.5 each in the config but never built, so `defending` silently reduced to
    # interceptions and tackles. Name what was dropped and how much weight went
    # with it, once per distinct set rather than once per group.
    dropped = {c: w for c, w in weights.items() if w != 0 and c not in usable}
    share = 0.0
    if dropped:
        share = sum(abs(w) for w in dropped.values()) / sum(
            abs(w) for w in weights.values() if w != 0
        )
        _warn_once(
            "Ignoring %d configured metric(s) with no data (%.0f%% of the "
            "weight): %s", len(dropped), 100 * share, ", ".join(sorted(dropped)),
        )

    # What is left has to be most of what was asked for. Renormalising over a
    # small remainder does not recover the sub-rating, it replaces it: with no
    # FBref `misc` table for 2014/15, centre-back defending kept only
    # `fouls_per90` -- a *negative* weight -- so 92% of the weight went missing
    # and the survivor ranked defenders by who fouled least. That is worse than
    # admitting the number cannot be computed, because it looks like defending.
    if not usable or share > max_missing_weight:
        return pd.Series(np.nan, index=frame.index)

    total = sum(abs(w) for w in usable.values())
    stacked = sum(
        (frame[c].fillna(0.5) if w > 0 else 1.0 - frame[c].fillna(0.5)) * abs(w)
        for c, w in usable.items()
    )
    return stacked / total


def rate_players(
    players: pd.DataFrame, config: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Compute ratings for every eligible player-season.

    Args:
        players: Output of :func:`footballml.players.ingest.fetch_player_seasons`.
        config: Parsed rating config. Loaded from disk when omitted.

    Returns:
        ``players`` plus ``rating`` (0-99), ``composite`` (0-1 before scaling),
        each ``sub_*`` rating, and ``rated`` / ``unrated_reason`` flags.
    """
    config = config or load_config()
    min_minutes = config["min_minutes"]
    k = float(config["shrinkage_nineties"])

    df = players.copy()
    # Ratings are computed per role. `role` carries the five outfield roles plus
    # goalkeeper and comes from EA positions; without a FIFA export we fall back
    # to Understat's coarse GK/D/M/F, which cannot tell a centre-back from a
    # full-back but still rates everyone.
    if "role" not in df.columns:
        from footballml.players.fifa import FALLBACK_ROLE

        df["role"] = df["position_group"].map(FALLBACK_ROLE)
    df["rated"] = False
    df["unrated_reason"] = pd.Series([None] * len(df), dtype="object")

    df.loc[df["role"].isna(), "unrated_reason"] = "No position recorded"
    for group, reason in UNRATED_GROUPS.items():
        df.loc[df["role"] == group, "unrated_reason"] = reason
    df.loc[
        (df["minutes"] < min_minutes) & df["unrated_reason"].isna(), "unrated_reason"
    ] = f"Under {min_minutes} minutes played"

    # Percentiles need a real distribution behind them. Drop pools too small to
    # provide one -- in practice the season currently in progress.
    min_group = int(config.get("min_group_size", 0))
    if min_group:
        qualified = df[df["unrated_reason"].isna()]
        sizes = qualified.groupby(["role", "Season"], observed=True).size()
        sparse = {key for key, size in sizes.items() if size < min_group}
        if sparse:
            keys = list(zip(df["role"], df["Season"], strict=True))
            too_small = pd.Series([k in sparse for k in keys], index=df.index)
            df.loc[too_small & df["unrated_reason"].isna(), "unrated_reason"] = (
                f"Fewer than {min_group} comparable players this season"
            )
            logger.info("Skipped %d sparse position-season pools", len(sparse))

    eligible = df["unrated_reason"].isna()
    if not eligible.any():
        logger.warning("No eligible players to rate")
        return df

    work = df[eligible].copy()
    by = ["role", "Season"]

    # --- 1 & 2: percentiles into sub-ratings -------------------------------
    # Declared from the config up front, in config order, and insertion-ordered
    # rather than a set. Three reasons, all about the column layout being the
    # same on every run: a set iterates in string-hash order, which Python
    # randomises per process, so two builds of identical data wrote the same
    # numbers under a different layout; discovering the columns as the groups
    # are walked would make the layout depend on which (role, season) came
    # first; and each column needs to start as plain float64, because assigning
    # a nullable Float64 block into one another group created as float64 raises.
    sub_columns: dict[str, None] = {
        f"sub_{name}": None
        for spec in config["positions"].values()
        for name in spec["sub_ratings"]
    }
    for column in sub_columns:
        work[column] = np.nan
    composites = pd.Series(np.nan, index=work.index)

    # Per role *and season*, not per role. `_weighted` drops a metric with no
    # data anywhere in the frame it is handed, and renormalises what is left --
    # but handed a whole role's twelve seasons it only ever sees the metric as
    # present, so a season where nobody has it falls through to the
    # fill-with-0.5 path instead. FBref serves no `misc` table at all for
    # 2014/15 and only La Liga's for 2015/16, which meant every centre-back in
    # those two seasons shared one invented defending score: the spread across
    # the pool was 2.3 points against ~20 in every other season, and defending
    # is 62% of a centre-back's composite. Splitting by season puts the guard
    # where the data actually varies. Percentiles are already ranked within
    # (role, season), so no other number moves.
    for (group, _season), block in work.groupby(by, observed=True, sort=False):
        spec = config["positions"].get(group)
        if spec is None:
            continue
        sub_values: dict[str, pd.Series] = {}
        for sub_name, metrics in spec["sub_ratings"].items():
            percentiles = pd.DataFrame(
                {
                    metric: percentile_within(block, metric, by)
                    for metric in metrics
                    if metric in block.columns
                }
            )
            sub_values[sub_name] = _weighted(percentiles, metrics)

        # `.astype(float)` on every assignment below, and it is load-bearing.
        # The ingest hands over nullable Float64 columns, so a group whose
        # metrics are all absent comes back as a Float64 block of pd.NA --
        # which pandas refuses to write into the plain float64 column the
        # previous group created. Reading the same data back from CSV flattens
        # the dtypes and hides it, so a dry run against the built file cannot
        # catch this; only the real pipeline can.
        for sub_name, values in sub_values.items():
            work.loc[block.index, f"sub_{sub_name}"] = values.astype(float)

        # No floor here, unlike the metric calls above. A missing *metric* is
        # absent evidence and renormalising over a sliver of what was asked for
        # invents a number; a missing *sub-rating* has already been refused on
        # that ground, and renormalising over the rest is what the sub-rating
        # weights are for. Refusing again would drop every 2014/15 centre-back
        # -- defending is 62% of the composite -- and with them EA's half of
        # their rating, which is present and is the better half for a defender
        # anyway. What is left is weaker, visibly so: `sub_defending` comes
        # back blank rather than as a plausible 50.
        composite = _weighted(
            pd.DataFrame(sub_values), spec["sub_rating_weights"], max_missing_weight=1.0
        )

        # Discipline only ever costs: a clean player is not rewarded, a
        # frequently sent-off one is penalised.
        discipline = config.get("discipline", {})
        if discipline:
            penalties = pd.DataFrame(
                {
                    metric: percentile_within(block, metric, by, higher_is_better=False)
                    for metric in discipline["metrics"]
                    if metric in block.columns
                }
            )
            if not penalties.empty:
                score = _weighted(penalties, discipline["metrics"])
                weight = float(discipline["weight"])
                # Guarded like the peak blend below, and for the same reason.
                # `_weighted` returns all-NaN when the cards are missing or
                # cover less than half the pool, and an unguarded blend would
                # multiply every composite in the block by that -- unrating a
                # whole role-season over a 3% term. Cards are served for every
                # season in this dataset, so this has never fired; it is the
                # kind of thing that fires the first time a source changes.
                composite = np.where(
                    score.notna(),
                    composite * (1 - weight) + score * weight,
                    composite,
                )
                composite = pd.Series(composite, index=block.index)

        composites.loc[block.index] = composite.astype(float)

    work["composite_raw"] = composites

    # Credit being outstanding at something, not just being tidy at everything.
    #
    # A weighted mean of percentiles cannot distinguish a player who is 97th
    # percentile at the thing his position exists for from one who is middling
    # across the board, and it ranks the second above the first whenever the
    # weights happen to favour what the specialist does not do. Mane's 22-goal
    # 2018/19 rated below his 11-goal 2020/21 for exactly that reason. See
    # `peak_weight` in the config for the sweep behind the number.
    peak_weight = float(config.get("peak_weight", 0.0))
    if peak_weight > 0 and sub_columns:
        present = [c for c in sub_columns if c in work.columns]
        # Sub-ratings are 0-1 here -- they are only scaled to 0-99 at the very
        # end -- so the peak is already on the composite's own scale.
        peak = work[present].astype(float).max(axis=1)
        # A row with no sub-rating at all keeps its composite rather than being
        # pulled toward a NaN: `max` of an empty row is NaN, and blending that
        # in would silently unrate the player.
        work["composite_raw"] = np.where(
            peak.notna(),
            (1 - peak_weight) * work["composite_raw"] + peak_weight * peak,
            work["composite_raw"],
        )

    # A group whose sub-ratings could not be computed at all -- every 2015/16
    # goalkeeper, because FBref served no `keeper` table that season -- must
    # not come back as a rated player with a blank rating, which is the one
    # outcome `unrated_reason` exists to prevent.
    #
    # But only where there is nothing else. EA's overall is half the published
    # rating and is present for ~95% of these players, so refusing outright
    # would throw away a real number to avoid publishing a missing one, and
    # take every 2015/16 team's goalkeeper line with it. Those players keep a
    # rating from EA alone and a blank `performance_rating`, which is exactly
    # what the two columns are carried separately to show.
    hollow = work["composite_raw"].isna()
    if hollow.any():
        missing = sorted(
            f"{role} {season}"
            for role, season in work.loc[hollow, by].drop_duplicates().itertuples(index=False)
        )
        rescued = hollow & work.get(
            "fifa_overall", pd.Series(np.nan, index=work.index)
        ).notna()
        logger.warning(
            "No usable performance data for %d player-season(s) (%s); %d keep a "
            "rating from EA alone, %d are left unrated",
            int(hollow.sum()), ", ".join(missing),
            int(rescued.sum()), int((hollow & ~rescued).sum()),
        )
        df.loc[work.index[hollow & ~rescued], "unrated_reason"] = (
            "No performance data this season"
        )
        work = work[~(hollow & ~rescued)]
        if work.empty:
            return df

    # --- 3: shrink toward the positional mean ------------------------------
    work["composite"] = shrink_toward_mean(work, "composite_raw", by, k)

    # --- 4: map onto 0-99 --------------------------------------------------
    # Ranked *within position*, not across all players. Ranking globally lets
    # whichever group has the widest spread dominate both tails: goalkeepers
    # play whole seasons, so shrinkage barely moves them (median weight 0.78
    # against 0.66 for forwards), their composites spread twice as wide, and
    # they took 30% of the top fifty while being 7% of the pool. Midfielders,
    # the most tightly clustered group, were squeezed out at 10% of 21%.
    #
    # Within-position ranking means the best keeper and the best forward land
    # on the same rating, which is both what the eye expects and how EA's own
    # scale behaves.
    work["performance_rating"] = _to_scale(
        work.groupby(by, observed=True)["composite"].rank(pct=True),
        config["scale"],
    )
    work["rating"] = _blend_with_fifa(work, config)
    work["rated"] = True

    # `performance_rating` is carried through alongside the blended `rating` so
    # the two can be compared. Where they disagree is the interesting part: a
    # player well above their EA overall is in form, one well below is coasting
    # on reputation.
    carried = [
        *sub_columns, "composite_raw", "composite", "performance_rating",
        "fifa_on_our_scale", "rating",
    ]
    for column in carried:
        if column in work.columns:
            df.loc[work.index, column] = work[column]
    df.loc[work.index, "rated"] = True

    # Sub-ratings are nicer to read on the same 0-99 scale as the overall.
    for column in sub_columns:
        df[column] = (df[column] * 99).round().astype("Float64")

    df["rating"] = df["rating"].round().astype("Int64")
    logger.info(
        "Rated %d of %d player-seasons (%d unrated)",
        int(df["rated"].sum()), len(df), int((~df["rated"]).sum()),
    )
    return df


def shrink_toward_mean(
    frame: pd.DataFrame, column: str, by: list[str], k: float
) -> pd.Series:
    """Empirical-Bayes shrinkage of ``column`` toward its group mean.

    Each value keeps weight ``n / (n + k)`` of itself, where ``n`` is the
    player's 90-minute appearances; the rest comes from the mean of his
    ``by`` group. ``k`` is how many nineties of evidence it takes to be
    trusted halfway.
    """
    prior = frame.groupby(by, observed=True)[column].transform("mean")
    n = frame["nineties"].clip(lower=0)
    weight = n / (n + k)
    return weight * frame[column] + (1 - weight) * prior


def _blend_with_fifa(work: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    """Combine the performance rating with EA's overall.

    The two measure genuinely different things and each covers the other's gap.
    EA's rating is balanced across positions by people who watch players, and it
    does not care how many minutes someone played. Ours is grounded in what
    actually happened on the pitch this season, and moves when form does.

    Blending needs care on one point: the two scales are not the same. EA's
    ratings cluster in the sixties and seventies with a long thin top, ours are
    spread by construction. Averaging them raw would let whichever is wider
    dominate, so EA's overall is first mapped onto our scale by percentile --
    within position, so a keeper is compared with keepers.

    Players with no EA entry are not penalised for the gap, but they are shrunk
    toward their role's mean by ``unmatched_shrinkage_nineties``. They have
    strictly less evidence behind them, and left alone they were the most
    extreme ratings in the table: the blend pulls every matched player's
    percentile toward EA's clustered middle, so an unmatched Las Palmas
    centre-back with one strong season out-rated nearly everyone. ~16% of
    rated player-seasons were unmatched, yet they held 11-18 of each season's
    top fifty.
    """
    weight = float(config.get("fifa_weight", 0.0))
    performance = work["performance_rating"]

    if weight <= 0 or "fifa_overall" not in work.columns:
        return performance

    present = work["fifa_overall"].notna()
    if not present.any():
        logger.warning("fifa_weight set but no FIFA ratings matched; using performance only")
        return performance

    group = "role" if "role" in work.columns else "position_group"
    fifa_percentile = work.groupby([group, "Season"], observed=True)["fifa_overall"].rank(
        pct=True
    )
    fifa_on_our_scale = _to_scale(fifa_percentile, config["scale"])
    # Kept, not just used. Published as `fifa_on_our_scale`, because without it
    # the page shows three numbers that cannot be reconciled: Chema Andres is
    # EA 63, performance 78, rating 64, and no reader can get 64 out of 63 and
    # 78. The blend is of 78.5 and *49.4* -- EA's 63 is the 1st percentile
    # among 181 defensive midfielders, and that is what 63 means once both
    # numbers are on one scale.
    work["fifa_on_our_scale"] = fifa_on_our_scale.where(present)

    blended = performance.copy()
    blended[present] = (
        performance[present] * (1 - weight) + fifa_on_our_scale[present] * weight
    )
    # No performance half to blend: the season's source table was missing for
    # this whole group (see `rate_players`). EA's judgement is the entire
    # rating rather than half of it.
    hollow = present & performance.isna()
    if hollow.any():
        blended[hollow] = fifa_on_our_scale[hollow]

    k = float(config.get("unmatched_shrinkage_nineties", 0.0))
    if k > 0 and (~present).any():
        # The prior is the mean over the whole role-season, matched players
        # included: it is the pool the performance percentile was ranked in.
        shrunk = shrink_toward_mean(work, "performance_rating", [group, "Season"], k)
        blended[~present] = shrunk[~present]
    logger.info(
        "Blended %d/%d ratings with FIFA at weight %.2f",
        int(present.sum()), len(work), weight,
    )
    return blended


def _to_scale(percentiles: pd.Series, anchors: list[list[float]]) -> pd.Series:
    """Interpolate percentiles onto the 0-99 scale defined by ``anchors``."""
    xs = [float(p) for p, _ in anchors]
    ys = [float(v) for _, v in anchors]
    return pd.Series(
        np.interp(percentiles.to_numpy(dtype="float64"), xs, ys),
        index=percentiles.index,
    )


def latest_ratings(rated: pd.DataFrame) -> pd.DataFrame:
    """One row per player: their most recent rated season.

    Players move clubs mid-season and appear twice; the latest row by season,
    then by minutes, is the best single summary of where they are now.
    """
    ordered = rated.sort_values(["Season", "minutes"], ascending=[False, False])
    return ordered.drop_duplicates(subset=["Player"], keep="first").reset_index(drop=True)
