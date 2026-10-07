"""Tests for the Shapley family attribution.

The Shapley value is worth using only because of the axioms it satisfies, so the
axioms are what get tested: efficiency (the parts sum to the whole), symmetry
(equal players get equal credit), dummy (a useless player gets zero), and
linearity. A bug in the weights would still produce plausible-looking numbers
that quietly fail these.
"""

from __future__ import annotations

import numpy as np
import pytest

from footballml.experiments.shapley import (
    check_efficiency,
    coalitions,
    per_match_contributions,
    shapley_weight,
)

PLAYERS = ("elo", "form", "xg", "squad")


def test_coalitions_are_complete_and_unique() -> None:
    cs = coalitions(PLAYERS)
    assert len(cs) == 2 ** len(PLAYERS) == 16
    assert len(set(cs)) == 16
    assert () in cs
    assert tuple(PLAYERS) in cs


def test_coalitions_ordered_smallest_first() -> None:
    sizes = [len(c) for c in coalitions(PLAYERS)]
    assert sizes == sorted(sizes)


def test_weights_form_a_probability_distribution() -> None:
    """Summed over subsets of each size, the weights must total exactly 1.

    This is the property that makes the values sum to the total gain, so it is
    worth checking directly rather than only through efficiency.
    """
    from math import comb

    for n in range(1, 8):
        total = sum(comb(n - 1, s) * shapley_weight(s, n) for s in range(n))
        assert total == pytest.approx(1.0)


def test_weight_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        shapley_weight(4, 4)


def _scores_from_values(values: dict[tuple[str, ...], float], n: int = 50) -> dict:
    """Constant per-match arrays realising a given coalition value function."""
    return {k: np.full(n, v, dtype="float64") for k, v in values.items()}


def test_efficiency_on_random_value_functions() -> None:
    """Values must sum to the total gain for any value function at all."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        scores = {c: rng.normal(0.2, 0.02, 40) for c in coalitions(PLAYERS)}
        contributions = per_match_contributions(scores, PLAYERS)
        total = check_efficiency(contributions, scores, PLAYERS)
        assert total == pytest.approx(
            scores[()].mean() - scores[tuple(PLAYERS)].mean()
        )


def test_dummy_player_gets_exactly_zero() -> None:
    """A family that changes nothing anywhere must be attributed nothing."""
    rng = np.random.default_rng(1)
    base = {c: rng.normal(0.2, 0.02) for c in coalitions(("elo", "form", "xg"))}
    # `squad` is a dummy: adding it to any coalition leaves the score unchanged.
    values = {}
    for c in coalitions(PLAYERS):
        without_squad = tuple(p for p in c if p != "squad")
        values[c] = base[without_squad]
    contributions = per_match_contributions(_scores_from_values(values), PLAYERS)
    assert contributions["squad"].mean() == pytest.approx(0.0, abs=1e-12)


def test_symmetric_players_get_equal_value() -> None:
    """Two families interchangeable in every coalition must score identically."""
    values = {}
    for c in coalitions(PLAYERS):
        # Value depends only on how many of elo/form are present, so the two are
        # perfectly interchangeable.
        count = len({"elo", "form"} & set(c))
        values[c] = 0.23 - 0.01 * count - 0.002 * len({"xg", "squad"} & set(c))
    contributions = per_match_contributions(_scores_from_values(values), PLAYERS)
    assert contributions["elo"].mean() == pytest.approx(contributions["form"].mean())


def test_perfectly_redundant_pair_splits_the_credit() -> None:
    """Two families carrying the *same* information get half each, not all each.

    This is the behaviour the ladder cannot produce and the reason for running
    this at all: with `elo` and `form` fully redundant, the ladder would give
    everything to whichever came first and nothing to the other.
    """
    gain = 0.03
    values = {}
    for c in coalitions(PLAYERS):
        # Either one alone delivers the whole gain; together they add nothing.
        values[c] = 0.23 - (gain if {"elo", "form"} & set(c) else 0.0)
    contributions = per_match_contributions(_scores_from_values(values), PLAYERS)
    assert contributions["elo"].mean() == pytest.approx(gain / 2)
    assert contributions["form"].mean() == pytest.approx(gain / 2)
    assert contributions["xg"].mean() == pytest.approx(0.0, abs=1e-12)


def test_additive_players_get_their_own_contribution() -> None:
    """With no interaction, Shapley reduces to each player's solo effect."""
    solo = {"elo": 0.02, "form": 0.005, "xg": 0.001, "squad": 0.0005}
    values = {c: 0.23 - sum(solo[p] for p in c) for c in coalitions(PLAYERS)}
    contributions = per_match_contributions(_scores_from_values(values), PLAYERS)
    for player, expected in solo.items():
        assert contributions[player].mean() == pytest.approx(expected)


def test_two_player_case_by_hand() -> None:
    """A worked example small enough to verify without the formula.

    v(empty)=0.20, v(a)=0.18, v(b)=0.19, v(ab)=0.17, as losses.
    a's marginals: 0.20-0.18=0.02 alone, 0.19-0.17=0.02 after b -> 0.02.
    b's marginals: 0.20-0.19=0.01 alone, 0.18-0.17=0.01 after a -> 0.01.
    Total gain 0.03, and 0.02 + 0.01 = 0.03.
    """
    values = {(): 0.20, ("a",): 0.18, ("b",): 0.19, ("a", "b"): 0.17}
    contributions = per_match_contributions(_scores_from_values(values), ("a", "b"))
    assert contributions["a"].mean() == pytest.approx(0.02)
    assert contributions["b"].mean() == pytest.approx(0.01)


def test_missing_coalition_is_fatal() -> None:
    """A missing run must fail rather than silently bias the attribution."""
    scores = {c: np.zeros(5) for c in coalitions(PLAYERS)}
    del scores[("elo", "xg")]
    with pytest.raises(KeyError, match="missing per-match scores"):
        per_match_contributions(scores, PLAYERS)


def test_mismatched_fixture_counts_are_fatal() -> None:
    scores = {c: np.zeros(5) for c in coalitions(PLAYERS)}
    scores[("elo",)] = np.zeros(4)
    with pytest.raises(ValueError, match="disagree on fixture count"):
        per_match_contributions(scores, PLAYERS)


def test_efficiency_failure_is_detected() -> None:
    """check_efficiency must actually catch a broken attribution."""
    scores = {c: np.full(10, 0.2 - 0.01 * len(c)) for c in coalitions(PLAYERS)}
    contributions = per_match_contributions(scores, PLAYERS)
    contributions["elo"] = contributions["elo"] + 1.0  # corrupt one
    with pytest.raises(AssertionError, match="but the total gain is"):
        check_efficiency(contributions, scores, PLAYERS)
