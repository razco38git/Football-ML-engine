"""Tests for the top-N scoreline list.

The card used to show one scoreline beside the outcome, and a football supporter
read "likeliest exact score 1-1" next to "Newcastle Win 57%" as a contradiction.
It is not one: an outcome probability sums a whole triangle of the scoreline
matrix while a draw's mass sits on the diagonal, so 1-1 is the single likeliest
score in 63% of matches even when the home side is the likeliest winner.

Measured before changing anything: constraining the shown score to agree with
the picked outcome makes it *less* accurate (11.1% exactly right against 12.9%),
so the numbers were left alone and three are shown instead of one.

What matters here is that the list stays consistent with the single score the
record still stores, and that it is genuinely ordered.
"""

from __future__ import annotations

import numpy as np
import pytest

from footballml.models.dixon_coles import most_likely_score, score_matrix, top_scores


@pytest.fixture
def matrix() -> np.ndarray:
    """Three fixtures: a home favourite, a tight game, and a rout."""
    return score_matrix(
        np.array([1.93, 1.20, 3.10]), np.array([1.07, 1.25, 0.40]), rho=-0.061
    )


def test_the_first_entry_is_the_modal_score(matrix):
    """The card leads with this, and the stored record keeps the same number.

    If these ever diverge, the headline score and the one in the prediction
    history would disagree for the same fixture.
    """
    scores, _ = top_scores(matrix)
    assert np.array_equal(scores[:, 0, :], most_likely_score(matrix))


def test_scores_come_back_likeliest_first(matrix):
    _, probs = top_scores(matrix)
    assert (np.diff(probs, axis=1) <= 0).all()


def test_probabilities_cannot_exceed_a_whole(matrix):
    _, probs = top_scores(matrix)
    assert (probs.sum(axis=1) <= 1.0).all()


def test_the_scores_are_distinct(matrix):
    """argpartition returning a cell twice would show "1-1, 1-1, 2-1"."""
    scores, _ = top_scores(matrix)
    for row in scores:
        assert len({tuple(s) for s in row}) == len(row)


def test_each_probability_is_the_matrix_cell_it_names(matrix):
    scores, probs = top_scores(matrix)
    for match in range(len(matrix)):
        for k in range(scores.shape[1]):
            home, away = scores[match, k]
            assert probs[match, k] == pytest.approx(matrix[match, home, away])


def test_no_score_outside_the_list_beats_one_inside_it(matrix):
    """The property the whole function claims."""
    scores, probs = top_scores(matrix)
    for match in range(len(matrix)):
        chosen = {tuple(s) for s in scores[match]}
        smallest = probs[match].min()
        for i in range(matrix.shape[1]):
            for j in range(matrix.shape[2]):
                if (i, j) not in chosen:
                    assert matrix[match, i, j] <= smallest + 1e-12


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_n_is_respected(matrix, n):
    scores, probs = top_scores(matrix, n=n)
    assert scores.shape == (len(matrix), n, 2)
    assert probs.shape == (len(matrix), n)


def test_asking_for_more_than_exists_is_capped(matrix):
    """A caller asking for 500 of 121 cells should get 121, not an error."""
    size = matrix.shape[1]
    scores, _ = top_scores(matrix, n=size * size + 50)
    assert scores.shape[1] == size * size


def test_the_leader_is_often_a_draw_even_when_a_side_is_favoured():
    """Pins the behaviour the display change exists for.

    Newcastle v Hull: 1-1 leads on 10.9% while the home side is a 57% favourite,
    because every home-win scoreline is counted separately.
    """
    matrix = score_matrix(np.array([1.93]), np.array([1.07]), rho=-0.061)
    scores, probs = top_scores(matrix)

    assert tuple(scores[0, 0]) == (1, 1), "leader should be the draw"
    assert probs[0, 0] == pytest.approx(0.109, abs=0.002)
    assert tuple(scores[0, 1]) == (2, 1), "runner-up should be a home win"
    # The gap is about a point -- which is the reason one number misleads.
    assert probs[0, 0] - probs[0, 1] < 0.02
