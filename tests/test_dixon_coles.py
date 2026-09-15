"""Tests for the scoreline distribution.

These are the tests that catch a silently wrong probability model. A bug here
wouldn't crash anything -- it would just quietly produce numbers that don't sum
to one, or swap home and away.
"""

from __future__ import annotations

import numpy as np
import pytest

from footballml.models.dixon_coles import (
    both_teams_score,
    fit_rho,
    most_likely_score,
    outcome_probs,
    over_under,
    score_matrix,
)

MU_HOME = np.array([1.5, 2.1, 0.8, 3.0])
MU_AWAY = np.array([1.2, 0.9, 1.6, 0.5])


@pytest.mark.parametrize("rho", [0.0, -0.10, 0.05])
def test_matrix_is_a_probability_distribution(rho: float) -> None:
    matrix = score_matrix(MU_HOME, MU_AWAY, rho=rho)
    assert np.allclose(matrix.sum(axis=(1, 2)), 1.0)
    assert (matrix >= 0).all(), "negative probabilities in the scoreline matrix"


@pytest.mark.parametrize("rho", [0.0, -0.10])
def test_outcome_probs_sum_to_one(rho: float) -> None:
    probs = outcome_probs(score_matrix(MU_HOME, MU_AWAY, rho=rho))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_home_away_orientation() -> None:
    """A far stronger home side must get the higher home-win probability.

    Guards against the classic transposition bug, which is easy to introduce and
    almost impossible to spot from aggregate metrics.
    """
    probs = outcome_probs(score_matrix(np.array([3.0]), np.array([0.4])))
    assert probs[0, 0] > probs[0, 2], "home-win probability should dominate here"

    reversed_probs = outcome_probs(score_matrix(np.array([0.4]), np.array([3.0])))
    assert reversed_probs[0, 2] > reversed_probs[0, 0]


def test_dixon_coles_shifts_mass_into_low_draws() -> None:
    """Negative rho must raise 0-0 and 1-1 and lower 1-0 and 0-1.

    This is the entire point of the correction; if it doesn't hold, the sign
    convention is wrong.
    """
    independent = score_matrix(MU_HOME, MU_AWAY, rho=0.0)
    corrected = score_matrix(MU_HOME, MU_AWAY, rho=-0.10)

    assert (corrected[:, 0, 0] > independent[:, 0, 0]).all()
    assert (corrected[:, 1, 1] > independent[:, 1, 1]).all()
    assert (corrected[:, 1, 0] < independent[:, 1, 0]).all()
    assert (corrected[:, 0, 1] < independent[:, 0, 1]).all()

    # Net effect: more draws.
    assert (outcome_probs(corrected)[:, 1] > outcome_probs(independent)[:, 1]).all()


def test_high_scores_are_untouched() -> None:
    """The correction applies only where both sides score at most once."""
    independent = score_matrix(MU_HOME, MU_AWAY, rho=0.0)
    corrected = score_matrix(MU_HOME, MU_AWAY, rho=-0.10)
    # Compare ratios, since renormalisation rescales everything slightly.
    ratio = corrected[:, 2:, 2:] / independent[:, 2:, 2:]
    assert np.allclose(ratio, ratio[:, :1, :1], rtol=1e-9)


def test_fit_rho_recovers_planted_value() -> None:
    """Simulate from a known rho and check the optimiser finds it."""
    rng = np.random.default_rng(0)
    n = 20_000
    mu_h = rng.uniform(0.7, 2.5, n)
    mu_a = rng.uniform(0.5, 2.0, n)
    true_rho = -0.08

    matrix = score_matrix(mu_h, mu_a, rho=true_rho)
    flat = matrix.reshape(n, -1)
    size = matrix.shape[1]
    draws = np.array([rng.choice(flat.shape[1], p=row) for row in flat])
    hg, ag = np.unravel_index(draws, (size, size))

    assert fit_rho(mu_h, mu_a, hg, ag) == pytest.approx(true_rho, abs=0.02)


def test_derived_markets_are_consistent() -> None:
    """Over/under and BTTS must agree with the matrix they came from."""
    matrix = score_matrix(MU_HOME, MU_AWAY, rho=-0.08)

    over = over_under(matrix, 2.5)
    assert ((over >= 0) & (over <= 1)).all()
    # Over 0.5 must be at least as likely as over 2.5.
    assert (over_under(matrix, 0.5) >= over).all()

    btts = both_teams_score(matrix)
    # BTTS implies at least 2 goals, so it cannot exceed P(over 1.5).
    assert (btts <= over_under(matrix, 1.5) + 1e-12).all()


def test_most_likely_score_matches_argmax() -> None:
    matrix = score_matrix(MU_HOME, MU_AWAY, rho=-0.08)
    modal = most_likely_score(matrix)
    for row, (h, a) in enumerate(modal):
        assert matrix[row, h, a] == matrix[row].max()


def test_mismatched_shapes_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        score_matrix(np.array([1.0, 2.0]), np.array([1.0]))
