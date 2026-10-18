"""Does every league want the same Dixon-Coles rho?

``rho`` is a single scalar fitted in ``MatchPredictor.fit`` and applied to all
five leagues. Profiling the Dixon-Coles likelihood per league on the committed
backtest puts the Bundesliga at **-0.1297 [-0.1784, -0.0787]**, an interval that
excludes the shipped **-0.0566**; rho also trends with scoring level, -0.038 in
the lowest total-lambda quartile against -0.070 in the highest.

So one rho across five leagues is currently an undocumented assumption rather
than a measured choice. This asks whether conditioning it survives out of sample.

The full protocol and the reasoning behind every constant live in
``docs/PROTOCOLS.md``.

.. warning::
    **Writes nothing**, and is **isolated from the rating-travel experiment** --
    the two share nothing but the backtest file on disk and may not be imported
    into one process. ``tests/test_conditional_rho_ab.py`` asserts both.

.. note::
    Every rho in every arm is fitted **strictly within the training fold**. That
    is the leak this experiment is most likely to commit: fitting rho on the
    scored fold would flatter every conditioned arm, and exactly in proportion to
    how finely it is conditioned.
"""

from __future__ import annotations

import pandas as pd

#: Arms, declared before the run. `production` is first and must equal the
#: shipped global rho -- an arm that is anything other than what
#: `MatchPredictor` already does would measure a configuration nobody runs.
ARMS = ("production", "per-league", "shrunk", "per-lambda")

#: Quartiles of predicted total goals for the `per-lambda` arm.
LAMBDA_BINS = 4

#: Scoreline log-likelihood must improve in at least this many of the five
#: leagues. Four of five, so one adverse league is survivable and two are not.
LEAGUES_REQUIRED = 4

#: Exact-score hit rate is ~12.8% over ~20,000 fixtures, so one sampling sd is
#: sqrt(0.128 * 0.872 / 20038) = 0.236 percentage points. The margin is set at
#: the resolution of the measurement, the same way the RPS floor was.
#:
#: The criterion reads the **upper bound of the 95% paired-bootstrap CI on the
#: decline**, not the point estimate: "does not fall" would reject an arm over
#: one match in twenty thousand, which is noise. The paired interval is much
#: narrower than 0.236pp since the arms share lambda and differ only in rho, so
#: this is permissive by construction -- it exists to stop a noise-level decline
#: vetoing a real log-likelihood gain, not to wave a real decline through.
EXACT_SCORE_MARGIN_PP = 0.25

#: Walk-forward RPS upper CI bound. **Not** assumed to be insensitive to rho:
#: rho moves mass between the 0-0, 0-1, 1-0 and 1-1 cells, and those sit on both
#: sides of the draw/win boundary, so an RPS effect is possible and is measured
#: rather than argued away.
MARGIN = 0.0005

BOOTSTRAP_RESAMPLES = 10_000
SEED = 7


class FoldLeakError(RuntimeError):
    """Raised when a rho fit is handed rows from the fold it will score."""


def fit_rho_within_fold(
    train: pd.DataFrame, score_index: pd.Index, fitter, **kwargs
) -> float:
    """Fit rho on ``train`` only, refusing any overlap with the scored fold.

    The guard is the point of this function. Every arm routes through it so that
    a conditioned rho cannot be fitted on the scorelines it is about to be
    judged on -- which would not merely inflate the result but would inflate the
    finely-conditioned arms most, producing a clean-looking monotone story.
    """
    overlap = train.index.intersection(score_index)
    if len(overlap) > 0:
        raise FoldLeakError(
            f"{len(overlap)} row(s) are in both the rho training frame and the "
            f"fold being scored, e.g. {list(overlap[:3])}. Rho must be fitted "
            "on strictly earlier matches."
        )
    return float(fitter(train, **kwargs))


def shrink(
    per_group: pd.Series, pooled: float, counts: pd.Series
) -> pd.Series:
    """James-Stein shrinkage of per-league rho toward the pooled estimate.

    A league contributes a fifth of the data, so its rho is noisier than the
    pooled one by roughly sqrt(5). Shrinkage is what keeps the `per-league` arm
    from being beaten by its own variance, and it is why `shrunk` is a declared
    arm rather than a fallback reached for afterwards.
    """
    weight = counts / counts.sum()
    spread = float(((per_group - pooled) ** 2 * weight).sum())
    if spread <= 0.0:
        return pd.Series(pooled, index=per_group.index, dtype="float64")
    factor = max(0.0, 1.0 - (len(per_group) - 2) / (len(per_group) * spread))
    return pooled + factor * (per_group - pooled)


def exact_score_passes(decline_ci_upper_pp: float) -> bool:
    """Non-inferiority on exact-score hit rate, read off the interval.

    Args:
        decline_ci_upper_pp: Upper bound of the 95% paired-bootstrap CI on the
            decline, in **percentage points**, positive meaning worse.
    """
    return bool(decline_ci_upper_pp < EXACT_SCORE_MARGIN_PP)
