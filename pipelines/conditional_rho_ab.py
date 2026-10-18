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

import numpy as np
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
    per_group: pd.Series,
    pooled: float,
    counts: pd.Series,
    variances: pd.Series | None = None,
) -> pd.Series:
    """Empirical-Bayes shrinkage of per-league rho toward the pooled estimate.

    A league contributes a fifth of the data, so its rho is noisier than the
    pooled one. Shrinkage is what keeps the `per-league` arm from being beaten
    by its own variance, and it is why `shrunk` is a declared arm rather than a
    fallback reached for afterwards.

    .. warning::
        **The first version of this function was wrong**, and wrong in a way
        that produced a clean-looking result. It computed
        ``1 - (k - 2) / (k * spread)`` with no sampling variance in it, which
        implicitly sets sigma^2 = 1. Rho deviations are ~0.07, so ``spread`` is
        ~0.001 and the factor hit its floor of zero on every fold: the `shrunk`
        arm was *production in disguise*, reporting a delta of exactly 0.000000
        against itself. That is why the variances are now a required input
        rather than an assumption.

    Args:
        per_group: Per-league rho estimates.
        pooled: The pooled estimate to shrink toward.
        counts: Rows behind each league, for the between-group variance weights.
        variances: Sampling variance of each per-league estimate. When omitted,
            no shrinkage is applied and the per-league values are returned
            unchanged -- the honest default, since shrinking by an assumed
            variance is what went wrong the first time.

    Returns:
        ``pooled + w_g * (rho_g - pooled)`` with
        ``w_g = tau^2 / (tau^2 + sigma_g^2)``, tau^2 estimated by method of
        moments as the excess of observed spread over mean sampling variance.
    """
    if variances is None:
        return per_group.astype("float64")

    weight = counts.reindex(per_group.index).astype("float64")
    weight = weight / weight.sum()
    sigma2 = variances.reindex(per_group.index).astype("float64")

    observed = float((((per_group - pooled) ** 2) * weight).sum())
    within = float((sigma2 * weight).sum())
    tau2 = max(0.0, observed - within)
    if tau2 <= 0.0:
        return pd.Series(pooled, index=per_group.index, dtype="float64")

    w = tau2 / (tau2 + sigma2)
    return pooled + w * (per_group - pooled)


def rho_variance(frame: pd.DataFrame, rho_hat: float, step: float = 5e-3) -> float:
    """Sampling variance of a fitted rho, from the curvature of its likelihood.

    The inverse of the second derivative of the negative log-likelihood at the
    optimum -- the observed-information estimate. Computed numerically because
    the likelihood runs through `score_matrix` and has no closed form here.
    """
    mu_h = frame["mu_home"].to_numpy()
    mu_a = frame["mu_away"].to_numpy()
    hg = frame["FTHG"].to_numpy().clip(0, 10).astype(int)
    ag = frame["FTAG"].to_numpy().clip(0, 10).astype(int)
    rows = np.arange(mu_h.size)

    def nll(rho: float) -> float:
        m = _matrix(mu_h, mu_a, np.full(mu_h.size, rho))
        return float(-np.log(m[rows, hg, ag] + 1e-12).sum())

    curvature = (nll(rho_hat + step) - 2.0 * nll(rho_hat) + nll(rho_hat - step)) / step**2
    if not np.isfinite(curvature) or curvature <= 0.0:
        return float("inf")
    return 1.0 / curvature


def exact_score_passes(decline_ci_upper_pp: float) -> bool:
    """Non-inferiority on exact-score hit rate, read off the interval.

    Args:
        decline_ci_upper_pp: Upper bound of the 95% paired-bootstrap CI on the
            decline, in **percentage points**, positive meaning worse.
    """
    return bool(decline_ci_upper_pp < EXACT_SCORE_MARGIN_PP)


# --- scoring ------------------------------------------------------------------

BACKTEST = "data/processed/backtest_predictions.csv"

#: A league needs at least this many training scorelines before it gets its own
#: rho; below it the arm falls back to the pooled value rather than fitting a
#: parameter on a handful of matches.
MIN_FIT_ROWS = 500


def _matrix(mu_h, mu_a, rho, max_goals: int = 10):
    """Scoreline matrices with a **per-match** rho.

    `score_matrix` takes a scalar -- its `if rho != 0.0` guard is ambiguous for
    an array -- so production's Poisson construction is reused at rho=0 and the
    correction applied here. The arithmetic is `_tau`'s, including its clip: a
    large rho against large goal rates can drive a cell negative, which is not a
    probability.
    """
    from footballml.models.dixon_coles import score_matrix

    matrix = score_matrix(mu_h, mu_a, rho=0.0, max_goals=max_goals)
    rho = np.broadcast_to(np.asarray(rho, dtype="float64"), mu_h.shape)
    tau = np.empty((mu_h.size, 2, 2), dtype="float64")
    tau[:, 0, 0] = 1.0 - mu_h * mu_a * rho
    tau[:, 0, 1] = 1.0 + mu_h * rho
    tau[:, 1, 0] = 1.0 + mu_a * rho
    tau[:, 1, 1] = 1.0 - rho
    matrix[:, :2, :2] *= np.clip(tau, 1e-10, None)
    matrix /= matrix.sum(axis=(1, 2), keepdims=True)
    return matrix


def _fit_pooled(frame: pd.DataFrame) -> float:
    from footballml.models.dixon_coles import fit_rho

    return fit_rho(
        frame["mu_home"].to_numpy(), frame["mu_away"].to_numpy(),
        frame["FTHG"].to_numpy(), frame["FTAG"].to_numpy(),
    )


def _fit_linear_in_lambda(frame: pd.DataFrame) -> tuple[float, float, float]:
    """rho = a + b * (total lambda - centre), by maximum likelihood."""
    from scipy.optimize import minimize

    mu_h = frame["mu_home"].to_numpy()
    mu_a = frame["mu_away"].to_numpy()
    hg = frame["FTHG"].to_numpy().clip(0, 10).astype(int)
    ag = frame["FTAG"].to_numpy().clip(0, 10).astype(int)
    total = mu_h + mu_a
    centre = float(total.mean())
    rows = np.arange(mu_h.size)

    def neg_log_lik(theta):
        rho = np.clip(theta[0] + theta[1] * (total - centre), -0.2, 0.2)
        m = _matrix(mu_h, mu_a, rho)
        return -np.log(m[rows, hg, ag] + 1e-12).sum()

    best = minimize(neg_log_lik, x0=[-0.05, 0.0], method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-3, "maxiter": 200})
    return float(best.x[0]), float(best.x[1]), centre


def rho_for_arm(arm: str, train: pd.DataFrame, score: pd.DataFrame) -> np.ndarray:
    """The per-match rho an arm assigns to the fold it is about to score.

    Every branch reads `train` only. `score` is consulted for its league labels
    and goal rates, never its scorelines.
    """
    pooled = _fit_pooled(train)
    n = len(score)

    if arm == "production":
        return np.full(n, pooled)

    if arm in {"per-league", "shrunk"}:
        counts = train.groupby("League").size()
        per = {}
        for league, group in train.groupby("League"):
            per[league] = _fit_pooled(group) if len(group) >= MIN_FIT_ROWS else pooled
        series = pd.Series(per, dtype="float64")
        if arm == "shrunk":
            variances = pd.Series(
                {
                    league: rho_variance(group, per[league])
                    for league, group in train.groupby("League")
                },
                dtype="float64",
            )
            series = shrink(
                series, pooled, counts.reindex(series.index).fillna(0), variances
            )
        return score["League"].map(series).fillna(pooled).to_numpy()

    if arm == "per-lambda":
        a, b, centre = _fit_linear_in_lambda(train)
        total = (score["mu_home"] + score["mu_away"]).to_numpy()
        return np.clip(a + b * (total - centre), -0.2, 0.2)

    raise ValueError(f"unknown arm {arm!r}")


def score_fold(frame: pd.DataFrame, rho: np.ndarray) -> pd.DataFrame:
    """Per-match log-likelihood, RPS and exact-score hit for one arm."""
    from footballml.models.evaluate import rps_per_match

    mu_h = frame["mu_home"].to_numpy()
    mu_a = frame["mu_away"].to_numpy()
    hg = frame["FTHG"].to_numpy().clip(0, 10).astype(int)
    ag = frame["FTAG"].to_numpy().clip(0, 10).astype(int)
    matrix = _matrix(mu_h, mu_a, rho)
    rows = np.arange(len(frame))

    draw = np.trace(matrix, axis1=1, axis2=2)
    home = np.tril(matrix, -1).sum(axis=(1, 2))
    probs = np.column_stack([home, draw, 1.0 - home - draw])

    flat = matrix.reshape(len(frame), -1).argmax(axis=1)
    side = matrix.shape[1]
    return pd.DataFrame(
        {
            "loglik": np.log(matrix[rows, hg, ag] + 1e-12),
            "rps": rps_per_match(frame["FTR"], probs),
            "exact": ((flat // side) == hg) & ((flat % side) == ag),
        },
        index=frame.index,
    )


def walk(data: pd.DataFrame) -> pd.DataFrame:
    """Walk forward over test seasons, fitting every rho on strictly earlier folds.

    The earliest backtest season is excluded as a *scored* fold: there is no
    earlier out-of-sample data from which to fit a causal rho for it, so no arm
    can be read there. That is the same convention the extended-Elo experiment
    uses for its unmovable fold.
    """
    seasons = sorted(data["Season"].astype(str).unique())
    out = []
    for season in seasons[1:]:
        train = data[data["Season"].astype(str) < season]
        fold = data[data["Season"].astype(str) == season]
        # The guard, on every arm, every fold.
        fit_rho_within_fold(train, fold.index, lambda f: 0.0)
        scored = fold[["League", "Season"]].copy()
        for arm in ARMS:
            rho = rho_for_arm(arm, train, fold)
            marks = score_fold(fold, rho)
            scored[f"loglik_{arm}"] = marks["loglik"]
            scored[f"rps_{arm}"] = marks["rps"]
            scored[f"exact_{arm}"] = marks["exact"]
            scored[f"rho_{arm}"] = rho
        out.append(scored)
    return pd.concat(out)


def _paired(base: np.ndarray, arm: np.ndarray, seed: int = SEED):
    """Paired bootstrap on the per-match difference. Positive favours `base`."""
    diff = np.asarray(arm, dtype="float64") - np.asarray(base, dtype="float64")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    draws = diff[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)
