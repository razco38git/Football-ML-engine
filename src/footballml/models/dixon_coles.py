r"""Dixon-Coles scoreline distribution.

Given expected goals for each side, the naive model treats the two scores as
independent Poisson draws. That is very nearly right, and wrong in one specific,
well-documented way: it under-predicts low-scoring draws. Real football produces
more 0-0 and 1-1 results than independence allows, because the state of a match
changes how both teams play.

Dixon & Coles (1997) fix this with a multiplicative correction :math:`\tau`
applied to the four cells where both teams score at most once:

.. math::

    \tau(x, y) = \begin{cases}
        1 - \lambda\mu\rho & x = 0, y = 0 \\
        1 + \lambda\rho    & x = 0, y = 1 \\
        1 + \mu\rho        & x = 1, y = 0 \\
        1 - \rho           & x = 1, y = 1 \\
        1                  & \text{otherwise}
    \end{cases}

with :math:`\lambda` the home goal rate and :math:`\mu` the away goal rate. A
negative :math:`\rho` -- which is what real data gives -- shifts mass into 0-0
and 1-1 and out of 1-0 and 0-1.

Once the matrix exists, every other output is a sum over it: 1X2 probabilities,
exact scorelines, over/under, both-teams-to-score. One coherent object, so the
site can never show a 1X2 probability that contradicts its own scoreline table.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import poisson

DEFAULT_MAX_GOALS = 10


def score_matrix(
    mu_home: np.ndarray | float,
    mu_away: np.ndarray | float,
    rho: float = 0.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """Build the normalised scoreline probability matrix.

    Args:
        mu_home: Home expected goals. Scalar or array of length ``n``.
        mu_away: Away expected goals. Scalar or array of length ``n``.
        rho: Dixon-Coles correlation parameter. ``0.0`` reduces to independent
            Poisson.
        max_goals: Highest scoreline modelled per side. Truncated mass is
            recovered by the normalisation step, and at 10 goals it is ~1e-6.

    Returns:
        Array of shape ``(n, max_goals + 1, max_goals + 1)`` where
        ``[n, i, j]`` is P(home scores ``i``, away scores ``j``). Rows sum to 1.
    """
    mu_h = np.atleast_1d(np.asarray(mu_home, dtype="float64"))
    mu_a = np.atleast_1d(np.asarray(mu_away, dtype="float64"))
    if mu_h.shape != mu_a.shape:
        raise ValueError(f"mu_home {mu_h.shape} and mu_away {mu_a.shape} must match")

    goals = np.arange(max_goals + 1)
    # (n, G+1) marginal pmfs, then outer product into (n, G+1, G+1).
    p_home = poisson.pmf(goals[None, :], mu_h[:, None])
    p_away = poisson.pmf(goals[None, :], mu_a[:, None])
    matrix = p_home[:, :, None] * p_away[:, None, :]

    if rho != 0.0:
        matrix[:, :2, :2] *= _tau(mu_h, mu_a, rho)

    # tau breaks the sum-to-one, and truncation loses a sliver of tail mass.
    # Renormalising fixes both at once.
    matrix /= matrix.sum(axis=(1, 2), keepdims=True)
    return matrix


def _tau(mu_h: np.ndarray, mu_a: np.ndarray, rho: float) -> np.ndarray:
    """The 2x2 Dixon-Coles correction block, shaped ``(n, 2, 2)``."""
    tau = np.empty((mu_h.size, 2, 2), dtype="float64")
    tau[:, 0, 0] = 1.0 - mu_h * mu_a * rho
    tau[:, 0, 1] = 1.0 + mu_h * rho
    tau[:, 1, 0] = 1.0 + mu_a * rho
    tau[:, 1, 1] = 1.0 - rho
    # A large rho paired with large goal rates can drive a cell negative, which
    # is not a probability. Clip rather than fail: the optimiser explores freely
    # and the bounds in fit_rho keep the fitted value well inside the safe region.
    return np.clip(tau, 1e-10, None)


def outcome_probs(matrix: np.ndarray) -> np.ndarray:
    """Collapse scoreline matrices into 1X2 probabilities.

    Returns:
        ``(n, 3)`` array of ``[P(home win), P(draw), P(away win)]``.
    """
    draw = np.trace(matrix, axis1=1, axis2=2)
    # Home wins are the cells below the diagonal (home goals > away goals).
    home = np.tril(matrix, k=-1).sum(axis=(1, 2))
    away = np.triu(matrix, k=1).sum(axis=(1, 2))
    return np.column_stack([home, draw, away])


def most_likely_score(matrix: np.ndarray) -> np.ndarray:
    """Modal scoreline per match as an ``(n, 2)`` array of ``[home, away]``."""
    n, size, _ = matrix.shape
    flat = matrix.reshape(n, -1).argmax(axis=1)
    return np.column_stack(np.unravel_index(flat, (size, size)))


def top_scores(matrix: np.ndarray, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """The ``n`` likeliest scorelines per match, likeliest first.

    One scoreline on its own is a misleading summary of this distribution, and
    the reason is structural rather than a quirk of any fixture. An outcome
    probability sums a whole triangle of the matrix -- a home win collects 1-0,
    2-0, 2-1, 3-1 and the rest -- while a draw's mass piles into the few cells on
    the diagonal. So 1-1 is the single likeliest score in 67% of matches even
    though the home side is usually the likeliest *winner*, and a card showing
    only "1-1" beside "Home win 57%" reads as a contradiction it is not.

    Showing three makes the shape visible: the leader rarely clears the runner-up
    by more than a point or two.

    That 67% is partly a property of ``rho`` rather than of football, so re-measure
    it after a refit instead of trusting it: over the same 20,013-match backtest it
    is 52% at ``rho=0``, 66% at ``-0.05`` and 73% at ``-0.10``. The figure quoted
    above is for the fitted ``-0.057``, and every other mention of it in the
    codebase inherits that caveat.

    Args:
        matrix: ``(n_matches, size, size)`` scoreline probabilities.
        n: How many scorelines to return.

    Returns:
        ``(scores, probs)`` where ``scores`` is ``(n_matches, n, 2)`` of
        ``[home, away]`` and ``probs`` is ``(n_matches, n)``. The first entry of
        each row is exactly :func:`most_likely_score`.
    """
    count, size, _ = matrix.shape
    n = min(n, size * size)
    flat = matrix.reshape(count, -1)

    # argpartition finds the n largest without sorting all 121 cells, then only
    # those n are sorted. Ordering matters -- the display leads with the best.
    top = np.argpartition(-flat, n - 1, axis=1)[:, :n]
    rows = np.arange(count)[:, None]
    top = np.take_along_axis(top, np.argsort(-flat[rows, top], axis=1), axis=1)

    home, away = np.unravel_index(top, (size, size))
    return np.stack([home, away], axis=-1), flat[rows, top]


def over_under(matrix: np.ndarray, line: float = 2.5) -> np.ndarray:
    """P(total goals above ``line``). Use a half-goal line to avoid pushes."""
    size = matrix.shape[1]
    totals = np.add.outer(np.arange(size), np.arange(size))
    return matrix[:, totals > line].sum(axis=1)


def both_teams_score(matrix: np.ndarray) -> np.ndarray:
    """P(both teams score at least once)."""
    return matrix[:, 1:, 1:].sum(axis=(1, 2))


def fit_rho(
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> float:
    """Fit ``rho`` by maximum likelihood on observed scorelines.

    Must be fitted on training data only -- it is a model parameter like any
    other, and fitting it on the test set leaks.

    Returns:
        The maximum-likelihood ``rho``, bounded to a numerically safe range.
    """
    mu_h = np.asarray(mu_home, dtype="float64")
    mu_a = np.asarray(mu_away, dtype="float64")
    hg = np.asarray(home_goals, dtype="int64").clip(0, max_goals)
    ag = np.asarray(away_goals, dtype="int64").clip(0, max_goals)
    rows = np.arange(mu_h.size)

    def neg_log_lik(rho: float) -> float:
        matrix = score_matrix(mu_h, mu_a, rho=rho, max_goals=max_goals)
        return -np.log(matrix[rows, hg, ag] + 1e-12).sum()

    # Real-world rho sits near -0.1; these bounds are wide enough to be
    # uninformative and tight enough to keep tau positive.
    result = minimize_scalar(neg_log_lik, bounds=(-0.2, 0.2), method="bounded")
    return float(result.x)
