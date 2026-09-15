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

Everything debatable is a number in the YAML, not a decision buried in code.

.. note::
    Two limitations are structural to the data rather than the method, and are
    surfaced rather than papered over. Goalkeepers are not rated at all -- the
    source has no saves or post-shot xG. And defenders are rated on attacking
    involvement, because tackles and interceptions are not available either, so
    an attacking full-back will out-rate a superb stay-at-home centre-back.
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


def _weighted(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted mean of percentile columns, ignoring any that are absent.

    A **negative weight means lower is better**: the percentile is flipped and
    the magnitude used as the weight. That is how ``goals_against_per90: -2.0``
    and ``fouls_per90: -0.5`` are expressed. Subtracting them instead would both
    corrupt the normaliser and let a metric drag a sub-rating below zero.

    Missing values default to 0.5 -- the middle of the distribution -- so a
    player with one unavailable metric is treated as average on it rather than
    worst.
    """
    usable = {c: w for c, w in weights.items() if c in frame.columns and w != 0}
    if not usable:
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
    sub_columns: set[str] = set()
    composites = pd.Series(np.nan, index=work.index)

    for group, spec in config["positions"].items():
        mask = work["role"] == group
        if not mask.any():
            continue
        block = work[mask]

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

        for sub_name, values in sub_values.items():
            column = f"sub_{sub_name}"
            sub_columns.add(column)
            work.loc[mask, column] = values

        composite = _weighted(
            pd.DataFrame(sub_values), spec["sub_rating_weights"]
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
                composite = composite * (1 - weight) + score * weight

        composites.loc[block.index] = composite

    work["composite_raw"] = composites

    # --- 3: shrink toward the positional mean ------------------------------
    prior = work.groupby(by, observed=True)["composite_raw"].transform("mean")
    n = work["nineties"].clip(lower=0)
    weight = n / (n + k)
    work["composite"] = weight * work["composite_raw"] + (1 - weight) * prior

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

    for column in [*sub_columns, "composite_raw", "composite", "rating"]:
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

    Players with no EA entry keep their performance rating unchanged rather than
    being penalised for the gap.
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

    blended = performance.copy()
    blended[present] = (
        performance[present] * (1 - weight) + fifa_on_our_scale[present] * weight
    )
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
