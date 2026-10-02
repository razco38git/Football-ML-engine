"""The cross-league correction the model cannot learn for itself.

Every match the model trains on is domestic, so the gap between two *leagues*
is always zero in its training data. It therefore never learns what a
cross-league rating difference means, and measurement says so plainly: across
788 UEFA ties between big-five clubs, its predicted home-win probability moves
at **+0.0034** per point of league-rating gap where the truth moves at
**+0.0402**. Feeding European results into Elo lifted that from -0.0002, which
is the right direction and about 8% of the distance.

That is a limit of the training set, not of the feature, and no better feature
fixes it. What does exist is the evidence: 822 matches in which the five
leagues actually played each other. This module fits a per-league offset on
those matches and applies it to the goal rates, outside the model.

**Fitted on the model's own residuals, not on the results alone.** The offsets
answer "where is this model wrong, and by how much, when a Spanish club plays
an English one" -- which is the question -- rather than "how strong is LaLiga",
which a standalone fit would answer and which the model has already partly
accounted for through squad strength and form. So the model's predicted rates
enter the fit as a Poisson offset and only the residual league effect is
estimated.

**Four free parameters on ~1,570 observations**, one league pinned as the
reference. `fit` reports a leave-one-season-out score precisely because an
in-sample improvement here would be worth nothing: a correction fitted and
judged on the same matches will always look good.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from footballml.data import PROCESSED_DIR

logger = logging.getLogger(__name__)

ADJUSTMENT_PATH = PROCESSED_DIR / "league_adjustment.json"

#: Pinned to zero; every other league is fitted relative to it. Which league is
#: chosen does not change the *differences*, which are all the correction uses.
REFERENCE = "E0"

#: Half the gap is added to the home rate and half taken from the away rate, in
#: log space, so the correction shifts the balance of a match without inflating
#: the total goals expected. A cross-league tie should not become higher
#: scoring just because the sides are unevenly matched.
SPLIT = 0.5


def _negative_log_likelihood(
    params: np.ndarray,
    home_index: np.ndarray,
    away_index: np.ndarray,
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    goals_home: np.ndarray,
    goals_away: np.ndarray,
) -> float:
    """Poisson NLL of both scorelines under the adjusted rates."""
    offsets = np.concatenate([[0.0], params])  # reference pinned at zero
    gap = offsets[home_index] - offsets[away_index]
    adjusted_home = mu_home * np.exp(SPLIT * gap)
    adjusted_away = mu_away * np.exp(-SPLIT * gap)
    return float(
        np.sum(adjusted_home - goals_home * np.log(adjusted_home))
        + np.sum(adjusted_away - goals_away * np.log(adjusted_away))
    )


def _fit_offsets(frame: pd.DataFrame, leagues: list[str]) -> dict[str, float]:
    order = {lg: i for i, lg in enumerate(leagues)}
    home_index = frame["home_league"].map(order).to_numpy()
    away_index = frame["away_league"].map(order).to_numpy()
    args = (
        home_index,
        away_index,
        frame["expected_goals_home"].to_numpy(dtype=float),
        frame["expected_goals_away"].to_numpy(dtype=float),
        frame["FTHG"].to_numpy(dtype=float),
        frame["FTAG"].to_numpy(dtype=float),
    )
    result = optimize.minimize(
        _negative_log_likelihood, np.zeros(len(leagues) - 1), args=args, method="BFGS"
    )
    return dict(zip(leagues, np.concatenate([[0.0], result.x]), strict=True))


def _rps(probs: np.ndarray, actual: np.ndarray) -> float:
    """Ranked probability score, the project's headline metric."""
    from footballml.models.evaluate import ranked_probability_score

    return ranked_probability_score(actual, probs)


def fit(predictions: pd.DataFrame, reference: str = REFERENCE) -> dict[str, object]:
    """Fit per-league offsets from scored European ties.

    Args:
        predictions: ``european_predictions.csv`` -- the model's own output on
            those matches, with ``home_league``, ``away_league``,
            ``expected_goals_home``/``away``, ``FTHG``, ``FTAG`` and ``Season``.

    Returns:
        ``{"offsets": {...}, "n_matches": int, "reference": str, "holdout": {...}}``.
        The offsets are log-scale multipliers on the goal rates.
    """
    cross = predictions[predictions["home_league"] != predictions["away_league"]].copy()
    if cross.empty:
        raise ValueError("no cross-league matches to fit on")

    seen = set(cross["home_league"]) | set(cross["away_league"])
    leagues = [reference] + sorted(seen - {reference})
    offsets = _fit_offsets(cross, leagues)

    # Leave-one-season-out: fit without a season, score that season. Anything
    # else would be judging the correction on the matches that produced it.
    from footballml.models.match_model import OUTCOMES

    held_probs, held_base, held_actual = [], [], []
    for season in sorted(cross["Season"].astype(str).unique()):
        mask = cross["Season"].astype(str) == season
        if mask.sum() < 10 or (~mask).sum() < 50:
            continue
        fitted = _fit_offsets(cross[~mask], leagues)
        block = cross[mask]
        held_probs.append(_probabilities(block, fitted))
        held_base.append(block[["prob_home_win", "prob_draw", "prob_away_win"]].to_numpy())
        held_actual.append(block["FTR"].to_numpy())

    holdout: dict[str, float] = {}
    if held_probs:
        actual = np.concatenate(held_actual)
        holdout = {
            "n": int(len(actual)),
            "rps_uncorrected": round(_rps(np.concatenate(held_base), actual), 5),
            "rps_corrected": round(_rps(np.concatenate(held_probs), actual), 5),
        }
        holdout["improvement"] = round(
            holdout["rps_uncorrected"] - holdout["rps_corrected"], 5
        )
        logger.info(
            "Leave-one-season-out: RPS %.5f -> %.5f (%+.5f) on %d matches",
            holdout["rps_uncorrected"], holdout["rps_corrected"],
            -holdout["improvement"], holdout["n"],
        )
        assert OUTCOMES  # the probability column order these scores assume

    return {
        "offsets": {k: round(v, 5) for k, v in offsets.items()},
        "reference": reference,
        "n_matches": int(len(cross)),
        "holdout": holdout,
    }


def _probabilities(block: pd.DataFrame, offsets: dict[str, float]) -> np.ndarray:
    """1X2 probabilities after correcting a block's goal rates."""
    from footballml.models.dixon_coles import outcome_probs, score_matrix

    mu_h, mu_a = adjust(
        block["expected_goals_home"].to_numpy(dtype=float),
        block["expected_goals_away"].to_numpy(dtype=float),
        block["home_league"].to_numpy(),
        block["away_league"].to_numpy(),
        offsets,
    )
    # rho=0: the correction is being judged on the balance it shifts, and the
    # low-score adjustment is the same on both sides of the comparison.
    return outcome_probs(score_matrix(mu_h, mu_a, rho=0.0))


def adjust(
    mu_home: np.ndarray | float,
    mu_away: np.ndarray | float,
    home_league: np.ndarray | str,
    away_league: np.ndarray | str,
    offsets: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Shift goal rates by the measured gap between the two leagues.

    A same-league pairing has a gap of zero and comes back untouched, so this is
    safe to apply everywhere rather than only where it matters.
    """
    home = np.atleast_1d(np.asarray(home_league, dtype=object))
    away = np.atleast_1d(np.asarray(away_league, dtype=object))
    gap = np.array(
        [
            offsets.get(str(h), 0.0) - offsets.get(str(a), 0.0)
            for h, a in zip(home, away, strict=True)
        ]
    )
    return (
        np.atleast_1d(mu_home) * np.exp(SPLIT * gap),
        np.atleast_1d(mu_away) * np.exp(-SPLIT * gap),
    )


def save(fitted: dict[str, object], path: Path | None = None) -> Path:
    path = path or ADJUSTMENT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fitted, indent=2), encoding="utf-8")
    return path


def load(path: Path | None = None) -> dict[str, float]:
    """Fitted offsets, or an empty dict if never fitted.

    Empty means every gap is zero and :func:`adjust` is the identity, so the
    site behaves exactly as it did before this existed.
    """
    path = path or ADJUSTMENT_PATH
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8"))["offsets"])
    except (KeyError, json.JSONDecodeError):  # pragma: no cover - corrupt file
        logger.warning("Could not read league adjustment from %s", path)
        return {}


def fitted_model_version(path: Path | None = None) -> str | None:
    """Which model artifact these offsets were fitted against, if recorded.

    The offsets describe one model's *residual* error, so they do not transfer
    to another. Returning the stamp lets a caller notice rather than silently
    correct the wrong thing.
    """
    path = path or ADJUSTMENT_PATH
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("model_version")
    except json.JSONDecodeError:  # pragma: no cover - corrupt file
        return None
