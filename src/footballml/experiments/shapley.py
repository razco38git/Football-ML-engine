r"""Shapley attribution over feature families.

The ablation ladder measures a family's contribution *given the families added
before it*, so whichever of two overlapping families arrives first takes the
credit for what they share. Leave-one-out measures the opposite extreme: what is
lost that nothing else replaces. Neither is "the" value of a family, and for
`elo` and `form` -- both built from match results -- the two disagree sharply.

The Shapley value resolves it by averaging a family's marginal contribution over
**every possible ordering**, which is the unique attribution satisfying:

*efficiency*
    the values sum exactly to the total gain, so nothing is lost or invented;
*symmetry*
    two families that add the same thing to every coalition get the same value;
*dummy*
    a family that never changes anything gets exactly zero;
*linearity*
    over the value function.

For player :math:`i` out of :math:`n`:

.. math::

    \phi_i = \sum_{S \subseteq N \setminus \{i\}}
             \frac{|S|!\,(n-|S|-1)!}{n!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr]

**`base` is the floor, not a player.** It is league identity and rest -- no
information about how good either side is -- so it is present in every coalition
and :math:`v(\emptyset)` is the `base` rung rather than a model with no features
at all. That makes this 2^4 = 16 runs rather than 2^5 = 32, and makes
:math:`v(\emptyset)` something already measured rather than a degenerate model.

The value function is **gain**, not RPS: :math:`v(S) = \text{RPS}_{base} -
\text{RPS}(S)`, so larger is better, :math:`v(\emptyset) = 0`, and the values sum
to the total improvement the full model makes over `base`.

Uncertainty comes free and exactly. Because :math:`\phi_i` is a *linear*
combination of mean per-match differences, it can be rewritten as the mean of a
per-match quantity:

.. math::

    \phi_i = \frac{1}{m} \sum_{\text{matches}}
             \underbrace{\sum_{S} w_{|S|}
             \bigl[r_S - r_{S \cup \{i\}}\bigr]}_{c_i(\text{match})}

so the same paired bootstrap used everywhere else in this project applies
directly to :math:`c_i`, with no extra model fits and no approximation.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from math import factorial

import numpy as np


def coalitions(players: Sequence[str]) -> list[tuple[str, ...]]:
    """Every subset of ``players``, smallest first, each sorted by player order.

    Returned as tuples so they can key a dict. The empty tuple is included: it is
    the floor coalition, whose value is defined as zero.
    """
    index = {p: i for i, p in enumerate(players)}
    out: list[tuple[str, ...]] = []
    for size in range(len(players) + 1):
        for combo in combinations(players, size):
            out.append(tuple(sorted(combo, key=index.__getitem__)))
    return out


def shapley_weight(subset_size: int, n_players: int) -> float:
    """The weight ``|S|!(n-|S|-1)!/n!`` on a marginal contribution to ``S``.

    Equivalently: the probability that, in a uniformly random ordering of the
    players, exactly the members of ``S`` come before player ``i``.
    """
    if not 0 <= subset_size < n_players:
        raise ValueError(f"subset size {subset_size} out of range for {n_players} players")
    return (
        factorial(subset_size)
        * factorial(n_players - subset_size - 1)
        / factorial(n_players)
    )


def per_match_contributions(
    scores: dict[tuple[str, ...], np.ndarray], players: Sequence[str]
) -> dict[str, np.ndarray]:
    """Per-match Shapley contribution for each player.

    Args:
        scores: Per-match loss (lower is better, e.g. RPS) for every coalition,
            keyed exactly as :func:`coalitions` returns them. Every array must
            cover the same fixtures in the same order.
        players: The players, in a stable order.

    Returns:
        ``{player: (n_matches,) array}``. The mean of each array is that
        player's Shapley value in units of *gain*, so positive means the family
        reduces loss. Summing all of them per match and averaging reproduces the
        total gain exactly -- which :func:`check_efficiency` asserts.
    """
    players = list(players)
    n = len(players)
    expected = set(coalitions(players))
    missing = expected - set(scores)
    if missing:
        raise KeyError(
            f"missing per-match scores for {len(missing)} coalition(s): "
            f"{sorted(missing)[:3]}"
        )

    lengths = {arr.shape for arr in scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"coalitions disagree on fixture count: {lengths}")

    index = {p: i for i, p in enumerate(players)}

    def key(members: set[str]) -> tuple[str, ...]:
        return tuple(sorted(members, key=index.__getitem__))

    out: dict[str, np.ndarray] = {}
    for player in players:
        others = [p for p in players if p != player]
        total = np.zeros(next(iter(scores.values())).shape, dtype="float64")
        for size in range(n):
            weight = shapley_weight(size, n)
            for subset in combinations(others, size):
                without = scores[key(set(subset))]
                with_it = scores[key({*subset, player})]
                # Loss falls when the player helps, so (without - with) is the
                # per-match *gain* from adding them.
                total += weight * (without - with_it)
        out[player] = total
    return out


def check_efficiency(
    contributions: dict[str, np.ndarray],
    scores: dict[tuple[str, ...], np.ndarray],
    players: Sequence[str],
    tolerance: float = 1e-10,
) -> float:
    """Assert the values sum to the total gain, and return that total.

    Efficiency is the property that makes a Shapley attribution trustworthy: the
    parts add up to the whole, exactly. If this fails the weights are wrong.
    """
    grand = tuple(players)
    total_gain = float(scores[()].mean() - scores[grand].mean())
    attributed = float(sum(c.mean() for c in contributions.values()))
    if abs(total_gain - attributed) > tolerance:
        raise AssertionError(
            f"Shapley values sum to {attributed:.12f} but the total gain is "
            f"{total_gain:.12f} (difference {attributed - total_gain:.2e})"
        )
    return total_gain
