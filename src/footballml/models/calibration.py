r"""Probability calibration for match forecasts.

The backtest exposed a real flaw: the model is overconfident on strong
favourites. Where it says 74% it is right 67% of the time; at 84% it is right
78%. Since the site publishes an accuracy tracker, systematically overstating
certainty on exactly the fixtures people care most about is a credibility
problem, not merely a metrics one.

**Temperature scaling** is the standard fix and the right one here:

.. math::

    p'_i \propto p_i^{1/T}

A single parameter :math:`T`. Above 1 it flattens the distribution (less
confident), below 1 it sharpens it. Two properties make it well suited:

1. **It cannot change the argmax.** Raising every probability to the same
   positive power preserves their order, so accuracy is untouched -- this fixes
   confidence without trading away correctness.
2. **One parameter.** Fitted on thousands of matches, the overfitting risk is
   negligible. Isotonic regression is more flexible but needs far more
   calibration data and can produce non-monotonic artefacts on thin bins.

The temperature **must** be fitted on predictions the model has not trained on.
Fitting it in-sample would learn the model's training-set confidence, which is
not the confidence that needs correcting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from footballml.models.evaluate import OUTCOMES


@dataclass
class TemperatureCalibrator:
    """Single-parameter temperature scaling for multiclass probabilities."""

    temperature_: float = 1.0
    fitted_: bool = field(default=False, init=False)

    def fit(
        self,
        probs: np.ndarray,
        y_true: np.ndarray | pd.Series,
        bounds: tuple[float, float] = (0.25, 4.0),
    ) -> TemperatureCalibrator:
        """Fit the temperature by minimising log loss on held-out predictions.

        Log loss is the canonical objective for calibration: it is a strictly
        proper scoring rule, so it is minimised only by honest probabilities, and
        it is smooth in ``T``.

        Args:
            probs: ``(n, 3)`` uncalibrated probabilities from data the model did
                **not** train on.
            y_true: Actual outcomes as H/D/A labels.
            bounds: Search range for ``T``.
        """
        log_p = np.log(np.clip(np.asarray(probs, dtype="float64"), 1e-12, None))
        obs = np.column_stack([(np.asarray(y_true, dtype=object) == o) for o in OUTCOMES])

        def neg_log_lik(temperature: float) -> float:
            scaled = log_p / temperature
            normalised = scaled - logsumexp(scaled, axis=1, keepdims=True)
            return float(-normalised[obs].sum())

        result = minimize_scalar(neg_log_lik, bounds=bounds, method="bounded")
        self.temperature_ = float(result.x)
        self.fitted_ = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature. Row-normalised, argmax preserved."""
        log_p = np.log(np.clip(np.asarray(probs, dtype="float64"), 1e-12, None))
        scaled = log_p / self.temperature_
        return np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))

    def fit_transform(self, probs: np.ndarray, y_true: np.ndarray | pd.Series) -> np.ndarray:
        return self.fit(probs, y_true).transform(probs)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, probs: np.ndarray, bins: int = 10
) -> float:
    """Mean gap between predicted probability and observed frequency.

    Each (fixture, outcome) pair contributes one point, binned by predicted
    probability and weighted by bin population. Zero is perfect. This is the
    single number to watch when judging whether calibration improved -- the
    curve shows *where* the model is wrong, this shows *how much*.
    """
    obs = np.column_stack(
        [(np.asarray(y_true, dtype=object) == o) for o in OUTCOMES]
    ).ravel().astype("float64")
    pred = np.asarray(probs, dtype="float64").ravel()

    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, bins - 1)

    total = 0.0
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.sum() * abs(obs[mask].mean() - pred[mask].mean())
    return float(total / pred.size)
