"""Evaluation metrics for probabilistic match forecasts.

The headline metric is the **Ranked Probability Score**. Accuracy is a poor fit
for this problem: home/draw/away are *ordered* (a draw sits between a home and
an away win), and being confidently wrong should cost more than being tentatively
wrong. RPS captures both; accuracy captures neither.

Calibration is tracked separately and matters just as much for the site's
credibility. A model that says 60% and is right 60% of the time is useful even
if its accuracy is unremarkable. One that says 90% and is right 60% of the time
is actively misleading, however good its hit rate looks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")


def _one_hot(y: np.ndarray | pd.Series) -> np.ndarray:
    """Encode H/D/A labels as ``(n, 3)`` indicators in outcome order."""
    y = np.asarray(y, dtype=object)
    return np.column_stack([(y == o).astype("float64") for o in OUTCOMES])


def rps_per_match(y_true: np.ndarray | pd.Series, probs: np.ndarray) -> np.ndarray:
    """RPS for each fixture separately, as a ``(n,)`` array. Lower is better.

    Split out from :func:`ranked_probability_score` so the pooled score and the
    paired bootstrap cannot drift apart: there is one definition of RPS and the
    mean of this *is* the headline number. Comparing two models needs the
    per-match values rather than the mean -- see :func:`paired_bootstrap`.
    """
    obs = _one_hot(y_true)
    cum_pred = np.cumsum(probs, axis=1)[:, :-1]
    cum_obs = np.cumsum(obs, axis=1)[:, :-1]
    return np.sum((cum_pred - cum_obs) ** 2, axis=1) / (len(OUTCOMES) - 1)


def ranked_probability_score(y_true: np.ndarray | pd.Series, probs: np.ndarray) -> float:
    """Mean RPS over ordered outcomes. Lower is better.

    For reference: bookmaker closing odds land around 0.19 on top-5 league
    football, and a naive base-rate forecast around 0.22. A model in the low 0.20s
    is doing real work; anything below ~0.19 deserves a hard look for leakage.
    """
    return float(np.mean(rps_per_match(y_true, probs)))


def paired_bootstrap(
    baseline: np.ndarray,
    variant: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 7,
    alpha: float = 0.05,
) -> dict[str, float | bool | int]:
    """Bootstrap the mean per-match difference between two sets of scores.

    **Pairing is the whole point.** Football matches differ enormously in how
    predictable they are, and that between-match variance swamps the difference
    between two models: an unpaired comparison of two models 0.001 apart cannot
    resolve it. Resampling *fixtures* and differencing within each one cancels
    the shared difficulty, which is what makes an effect that small measurable.

    Both arrays must be per-match scores for **the same fixtures in the same
    order**, which is checked. Any lower-is-better score works; this project
    uses :func:`rps_per_match`.

    Args:
        baseline: Per-match scores for the reference model.
        variant: Per-match scores for the model being tested.
        n_boot: Bootstrap resamples.
        seed: Fixed so a reported interval can be reproduced exactly.
        alpha: ``0.05`` gives a 95% interval.

    Returns:
        ``delta`` (mean ``variant - baseline``, so negative favours the variant),
        ``lo``/``hi`` percentile bounds, ``excludes_zero`` -- the only thing that
        licenses the word "better" -- and ``n``.
    """
    baseline = np.asarray(baseline, dtype="float64")
    variant = np.asarray(variant, dtype="float64")
    if baseline.shape != variant.shape:
        raise ValueError(
            f"paired bootstrap needs matched fixtures; got {baseline.shape} and {variant.shape}"
        )
    if baseline.ndim != 1:
        raise ValueError(f"expected per-match 1-D scores, got shape {baseline.shape}")

    diff = variant - baseline
    n = diff.size
    rng = np.random.default_rng(seed)
    # One (n_boot, n) index draw would be 10000 x 20013 int64 = 1.6GB. Summing
    # per-resample keeps it to one row at a time.
    means = np.empty(n_boot, dtype="float64")
    for b in range(n_boot):
        means[b] = diff[rng.integers(0, n, n)].mean()

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n": int(n),
    }


def log_loss(y_true: np.ndarray | pd.Series, probs: np.ndarray, eps: float = 1e-15) -> float:
    """Multiclass log loss. Punishes confident mistakes severely."""
    obs = _one_hot(y_true)
    clipped = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.sum(obs * np.log(clipped), axis=1)))


def brier_score(y_true: np.ndarray | pd.Series, probs: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error over the probability vector."""
    return float(np.mean(np.sum((probs - _one_hot(y_true)) ** 2, axis=1)))


def accuracy(y_true: np.ndarray | pd.Series, probs: np.ndarray) -> float:
    """Share of fixtures where the most likely outcome was the actual one."""
    pred = np.array(OUTCOMES, dtype=object)[probs.argmax(axis=1)]
    return float(np.mean(pred == np.asarray(y_true, dtype=object)))


def goal_errors(
    home_goals: np.ndarray, away_goals: np.ndarray, mu_home: np.ndarray, mu_away: np.ndarray
) -> dict[str, float]:
    """MAE and RMSE of expected goals against actual goals, both sides pooled."""
    actual = np.concatenate([np.asarray(home_goals), np.asarray(away_goals)])
    pred = np.concatenate([np.asarray(mu_home), np.asarray(mu_away)])
    err = actual - pred
    return {
        "goals_mae": float(np.mean(np.abs(err))),
        "goals_rmse": float(np.sqrt(np.mean(err**2))),
    }


def evaluate(
    y_true: np.ndarray | pd.Series,
    probs: np.ndarray,
    home_goals: np.ndarray | None = None,
    away_goals: np.ndarray | None = None,
    mu_home: np.ndarray | None = None,
    mu_away: np.ndarray | None = None,
) -> dict[str, float]:
    """Full metric bundle for one set of forecasts."""
    metrics = {
        "rps": ranked_probability_score(y_true, probs),
        "log_loss": log_loss(y_true, probs),
        "brier": brier_score(y_true, probs),
        "accuracy": accuracy(y_true, probs),
        "n": int(len(probs)),
    }
    if all(v is not None for v in (home_goals, away_goals, mu_home, mu_away)):
        metrics.update(goal_errors(home_goals, away_goals, mu_home, mu_away))
    return metrics


def calibration_table(
    y_true: np.ndarray | pd.Series, probs: np.ndarray, bins: int = 10
) -> pd.DataFrame:
    """Predicted vs observed frequency, pooled across all three outcomes.

    Each (fixture, outcome) pair contributes one point: its predicted probability
    and whether it happened. Well-calibrated models sit on the diagonal.
    """
    obs = _one_hot(y_true).ravel()
    pred = np.asarray(probs, dtype="float64").ravel()

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin_lower": edges[b],
                "bin_upper": edges[b + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(pred[mask].mean()),
                "observed_rate": float(obs[mask].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["observed_rate"] - table["mean_predicted"]
    return table


def base_rate_probs(y_train: np.ndarray | pd.Series, n: int) -> np.ndarray:
    """Baseline that always predicts the training-set outcome frequencies.

    Any model that cannot beat this has learned nothing at all.
    """
    freqs = _one_hot(y_train).mean(axis=0)
    return np.tile(freqs, (n, 1))


def odds_implied_probs(odds: pd.DataFrame, columns: tuple[str, str, str]) -> np.ndarray:
    """Convert decimal bookmaker odds to de-vigged probabilities.

    Raw reciprocals sum to more than one -- the bookmaker's margin, typically
    4-6%. Normalising proportionally is the standard simple removal. It slightly
    over-corrects favourites, but is more than good enough as a benchmark.

    Used **only** to benchmark our model, never as a feature.
    """
    raw = 1.0 / odds[list(columns)].to_numpy(dtype="float64")
    return raw / raw.sum(axis=1, keepdims=True)
