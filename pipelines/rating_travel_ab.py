"""Does a defender's rating travel with him when he changes club?

A centre back's rating correlates +0.67 with his own club's defensive record,
and `config/player_rating.yaml` gives CB ``team_defence: 0.30`` against FB's
``0.0`` -- so 30% of a centre back's composite *is* his club's expected goals
against, goals against and clean sheet rate. The config states the consequence
("Saliba and Gabriel both take Arsenal's 98 and both land on 85"); what is
unknown is whether that makes the number wrong about the *player*.

A transfer separates the two. If the rating is about him it should predict his
next EA assessment equally well whether or not he moves; if it is about the side
he just left, it should attenuate.

The full protocol, its acceptance criteria and the reasoning behind every
constant below live in ``docs/PROTOCOLS.md``. This module holds the constants so
they are readable in the source before any number exists, which is the standard
the extended-Elo and positional-quota experiments set.

.. warning::
    **Writes nothing.** Every frame lives in memory and the production rating
    and strength files are never opened for writing. ``tests/test_rating_travel_ab.py``
    asserts that by scanning this source, because a ``finally`` does not survive
    the process being killed and one experiment already left ``team_strength.csv``
    holding its own numbers with the API serving them.

.. note::
    **Nothing here changes production weights under any outcome.** A candidate
    reweighting is measured and reported against the guards below; adopting one
    is a separate decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- the roles under test -----------------------------------------------------

#: Analysed separately, never pooled into a single "defender" estimate. They
#: differ in the variable under test -- CB carries `team_defence` at 0.30, FB at
#: 0.0 -- which is what makes the contrast informative at all.
ROLES = ("CB", "FB")

# --- the predictor ------------------------------------------------------------

#: `performance_rating`, never `rating`. `rating` is a 50/50 blend with
#: `fifa_overall`, so scoring it against EA's *next* edition would be largely EA
#: predicting itself. This is the number `rating.py` computes before
#: `_blend_with_fifa`.
PREDICTOR = "performance_rating"

#: Projected out of the predictor for the decomposition. Exactly the three club
#: aggregates that constitute the `team_defence` sub-rating in the config --
#: adding more would project out legitimate player signal that merely correlates
#: with club quality, and the question is about *these* inputs.
PROJECTED_COLUMNS = (
    "team_xga_per_match",
    "team_ga_per_match",
    "team_clean_sheet_pct",
)

# --- the outcome --------------------------------------------------------------

#: EA's scale drifts: mean overall runs 73.41 in the 1415 edition to 75.98 in
#: 1920, with the sd between 5.14 and 5.60. A raw difference of overalls would
#: pool that drift with the player's actual move, so each edition is
#: standardised over its own full worldwide export first.
OUTCOME_SCALES = ("zscore", "rank")

#: Excluded as an *outcome* edition: it carries 504 big-five players against
#: ~1,900 in a complete edition and is still being populated.
INCOMPLETE_EDITIONS = ("2627",)

#: EA ages are integers and a player is one year older per edition; ±1.5 admits
#: the rounding without admitting a different footballer. Measured on the 6,209
#: departed player-seasons this recovers 77.9% uniquely with 3 ambiguous.
AGE_TOLERANCE = 1.5

#: Above this, a defender can only move down, so the thin ceiling (per-edition
#: max 91-94, under 1% at 90+) distorts the move. Sensitivity only; the headline
#: controls for the level instead.
TOP_BAND = 88

#: Where the unrecovered share is 3.7-6.8% rather than 25.5%.
AGE_CAP = 30

# --- the bar, fixed before the run --------------------------------------------

#: Equivalence bounds on the travel ratio: a **declared practical threshold**,
#: not a derivation. Nothing implies a 30% composite weight yields a 30% slope
#: attenuation. What the weight establishes is the order of magnitude -- if club
#: contamination runs through `team_defence` at all, the attenuation is
#: plausibly tens of percent rather than single digits -- so 15% marks roughly
#: half that scale, below which an effect is too small to attribute to the
#: mechanism or to justify a weight change. 1.18 is 1/0.85, so the interval is
#: symmetric in log-ratio.
EQUIVALENCE = (0.85, 1.18)

#: The same decision re-run at a looser and a stricter threshold. If the verdict
#: flips across these, it depends on the bound rather than the data, and that is
#: reported as inconclusive rather than resolved.
ALTERNATIVE_BOUNDS = ((0.80, 1.25), (0.90, 1.11))

#: Above this share of bootstrap replicates flipping the sign of the stayer
#: slope, the ratio has a denominator indistinguishable from zero and is
#: reported as undefined rather than summarised.
SIGN_FLIP_MAX = 0.05

#: Resamples draw *players*, not rows: one player contributes up to 11
#: consecutive-season pairs, and a row-level bootstrap would understate every
#: interval. `player_rating.yaml` makes the same point for the FB sweep -- "the
#: paired bootstrap's effective sample is nearer 12 than 20,060".
BOOTSTRAP_RESAMPLES = 10_000
SEED = 7

# --- the guards a candidate reweighting must clear ----------------------------
#
# Identical to the positional-quota experiment's, so the two are directly
# comparable. Production weights are not changed here under any outcome.

#: `def_vs_ga` may not worsen by more than this -- the whole 13-to-17 range of
#: `def>GA` from the squad-size sweep.
DEF_GAIN = 0.010

#: `ovr_vs_pts` and `atk_vs_gf` may not worsen by more than this.
OTHER_TOLERANCE = 0.005

#: Walk-forward RPS upper CI bound. The measured floor of this harness.
MARGIN = 0.0005


def standardise_within(
    frame: pd.DataFrame, column: str, by: list[str], scale: str = "zscore"
) -> pd.Series:
    """Standardise ``column`` within each ``by`` group.

    Both the raw and the residualised predictor go through this before entering
    any model. Residualising necessarily shrinks variance, and a
    smaller-variance predictor yields a smaller coefficient *for that reason
    alone* -- so comparing an unscaled raw slope against an unscaled residual
    slope would manufacture the finding the experiment is looking for.

    Args:
        frame: Source frame.
        column: Column to standardise.
        by: Grouping keys, normally ``["role", "Season"]``.
        scale: ``"zscore"`` for mean 0 / sd 1, or ``"rank"`` for a within-group
            percentile. The rank form respects the bounded, ordinal nature of an
            EA rating and is immune to compression at the top.

    Returns:
        The standardised column, aligned to ``frame``'s index.
    """
    if scale not in OUTCOME_SCALES:
        raise ValueError(f"scale must be one of {OUTCOME_SCALES}, got {scale!r}")
    grouped = frame.groupby(by, observed=True)[column]
    if scale == "rank":
        return grouped.rank(pct=True)
    mean = grouped.transform("mean")
    sd = grouped.transform("std")
    # A stratum with one member has no spread; it contributes no information and
    # must not become an infinity.
    return (frame[column] - mean) / sd.replace(0.0, np.nan)


def residualise(
    frame: pd.DataFrame,
    column: str = PREDICTOR,
    projected: tuple[str, ...] = PROJECTED_COLUMNS,
    by: tuple[str, ...] = ("role", "Season"),
) -> tuple[pd.Series, pd.DataFrame]:
    """Project the club defensive aggregates out of the predictor.

    Fitted **within each (role, season) stratum on the full rated population**,
    not on the analysis sample. That superset is deliberate: it uses only
    season-`t` information and does not condition on a player having an EA
    outcome at `t+1`, so which rows survive into the regression cannot change
    what the residual means.

    No outcome of any kind enters this, so there is no path by which it leaks.

    Returns:
        The residual, and a per-stratum frame carrying ``r_squared`` -- which is
        the quantity saying how much of the rating these three club numbers
        account for at all. Where it is near zero the decomposition has nothing
        to remove, and a residual slope equal to the raw one is uninformative
        rather than reassuring.
    """
    missing = [c for c in (column, *projected) if c not in frame.columns]
    if missing:
        raise KeyError(f"residualise needs {missing}, which the frame does not carry")

    residual = pd.Series(np.nan, index=frame.index, dtype="float64")
    diagnostics: list[dict[str, object]] = []

    for key, group in frame.groupby(list(by), observed=True):
        usable = group.dropna(subset=[column, *projected])
        if len(usable) <= len(projected) + 1:
            continue
        x = np.column_stack(
            [np.ones(len(usable)), *(usable[c].to_numpy() for c in projected)]
        )
        y = usable[column].to_numpy()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ beta
        residual.loc[usable.index] = y - fitted
        total = float(((y - y.mean()) ** 2).sum())
        diagnostics.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True)),
                "n": len(usable),
                "r_squared": float(1.0 - ((y - fitted) ** 2).sum() / total)
                if total > 0
                else np.nan,
            }
        )

    return residual, pd.DataFrame(diagnostics)


def travel_ratio(stayer: float, mover_delta: float) -> float:
    """``(beta + delta) / beta`` -- reported, never the decision variable.

    `METHODOLOGY.md` records the defender EA-move correlation at 0.04-0.06, so
    the denominator is expected to be small and the ratio correspondingly
    unstable. The decision rests on ``delta``, which is a difference and is
    stable however small ``beta`` is.
    """
    if stayer == 0.0:
        return float("nan")
    return (stayer + mover_delta) / stayer


def ratio_is_interpretable(stayer_draws: np.ndarray) -> bool:
    """False when the bootstrap flips the denominator's sign too often.

    A ratio whose denominator straddles zero has no stable interpretation and no
    meaningful interval; reporting one anyway is how a near-zero slope becomes a
    dramatic-looking attenuation.
    """
    draws = np.asarray(stayer_draws, dtype="float64")
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return False
    minority = min((draws > 0).mean(), (draws < 0).mean())
    return bool(minority <= SIGN_FLIP_MAX)


def verdict(
    delta_ci: tuple[float, float],
    ratio: float,
    ratio_ci: tuple[float, float],
    stayer_ci: tuple[float, float],
    bounds: tuple[float, float] = EQUIVALENCE,
) -> str:
    """The three-way decision, computed from the declared constants.

    A confidence interval containing 1.0 is **not** evidence of no effect, which
    is why ``"equivalent"`` requires the interval to sit inside the bounds rather
    than merely to cover them.

    ``"uninformative"`` fires first: the attenuation of a slope indistinguishable
    from zero is undefined, and given the documented 0.04-0.06 baseline that is a
    realistic outcome. Pre-declaring it is what stops it being rationalised after
    the fact.
    """
    low, high = stayer_ci
    if low <= 0.0 <= high:
        return "uninformative"

    d_low, d_high = delta_ci
    if not (d_low <= 0.0 <= d_high) and ratio < bounds[0]:
        return "contamination"

    r_low, r_high = ratio_ci
    if bounds[0] <= r_low and r_high <= bounds[1]:
        return "equivalent"

    return "inconclusive"
