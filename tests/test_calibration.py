"""Tests for temperature scaling."""

from __future__ import annotations

import numpy as np
import pytest

from footballml.models.calibration import (
    TemperatureCalibrator,
    expected_calibration_error,
)
from footballml.models.evaluate import accuracy, log_loss


@pytest.fixture
def overconfident() -> tuple[np.ndarray, np.ndarray]:
    """Probabilities that are too sharp for the outcomes they accompany.

    Built by sampling outcomes from honest probabilities, then sharpening the
    reported ones -- exactly the failure mode seen in the backtest.
    """
    rng = np.random.default_rng(42)
    n = 4000
    honest = rng.dirichlet([3.0, 2.0, 2.5], size=n)
    labels = np.array(["H", "D", "A"], dtype=object)
    y = np.array([rng.choice(labels, p=row) for row in honest], dtype=object)

    sharp = honest**1.6
    return sharp / sharp.sum(axis=1, keepdims=True), y


def test_preserves_argmax(overconfident: tuple[np.ndarray, np.ndarray]) -> None:
    """Temperature scaling must never change which outcome is favoured.

    This is the property that makes it safe: confidence is corrected without
    trading away any correct predictions.
    """
    probs, y = overconfident
    calibrated = TemperatureCalibrator().fit_transform(probs, y)

    assert (calibrated.argmax(axis=1) == probs.argmax(axis=1)).all()
    assert accuracy(y, calibrated) == accuracy(y, probs)


def test_rows_remain_distributions(overconfident: tuple[np.ndarray, np.ndarray]) -> None:
    probs, y = overconfident
    calibrated = TemperatureCalibrator().fit_transform(probs, y)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert (calibrated >= 0).all()


def test_softens_overconfidence(overconfident: tuple[np.ndarray, np.ndarray]) -> None:
    """Given overconfident input, the fitted temperature must exceed 1."""
    probs, y = overconfident
    calibrator = TemperatureCalibrator().fit(probs, y)

    assert calibrator.temperature_ > 1.0, "should flatten an overconfident model"
    calibrated = calibrator.transform(probs)
    assert calibrated.max(axis=1).mean() < probs.max(axis=1).mean()


def test_improves_calibration_and_log_loss(
    overconfident: tuple[np.ndarray, np.ndarray],
) -> None:
    probs, y = overconfident
    calibrated = TemperatureCalibrator().fit_transform(probs, y)

    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, probs)
    assert log_loss(y, calibrated) < log_loss(y, probs)


def test_leaves_calibrated_input_alone() -> None:
    """An already-honest model should get a temperature near 1."""
    rng = np.random.default_rng(7)
    n = 6000
    honest = rng.dirichlet([3.0, 2.0, 2.5], size=n)
    labels = np.array(["H", "D", "A"], dtype=object)
    y = np.array([rng.choice(labels, p=row) for row in honest], dtype=object)

    assert TemperatureCalibrator().fit(honest, y).temperature_ == pytest.approx(1.0, abs=0.1)


def test_identity_at_unit_temperature() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    assert np.allclose(TemperatureCalibrator(temperature_=1.0).transform(probs), probs)


def test_ece_is_zero_for_perfect_calibration() -> None:
    """A model that always says 100% and is always right has no calibration error."""
    probs = np.tile([1.0, 0.0, 0.0], (500, 1))
    y = np.array(["H"] * 500, dtype=object)
    assert expected_calibration_error(y, probs) == pytest.approx(0.0, abs=1e-9)
