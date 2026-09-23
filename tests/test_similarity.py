"""Tests for player similarity.

These pin the properties that make a similarity number trustworthy -- that it
is symmetric, that a player is identical to himself, that percentiles are never
compared across roles, and that the scale actually discriminates -- rather than
asserting that particular players come out near each other, which would encode
today's EA export rather than the rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.players.similarity import (
    GK_ATTRIBUTES,
    OUTFIELD_ATTRIBUTES,
    attribute_set,
    similar_players,
)


def _player(name, role, team="A", season="2526", **values):
    row = {
        "Player": name,
        "Team": team,
        "League": "E0",
        "Season": season,
        "role": role,
        "rating": values.pop("rating", 75),
        "minutes": values.pop("minutes", 2000),
    }
    attributes = attribute_set(role)
    base = values.pop("base", 60)
    for attribute in attributes:
        row[f"fifa_{attribute}"] = values.pop(attribute, base)
    subs = ("sub_finishing", "sub_creation", "sub_involvement", "sub_defending")
    if role == "GK":
        subs = ("sub_shot_stopping", "sub_reliability", "sub_workload", "sub_penalties")
    for sub in subs:
        row[sub] = values.pop(sub, base)
    row.update(values)
    return row


def _frame(rows):
    return pd.DataFrame(rows)


@pytest.fixture
def outfielders():
    """A spread of wingers, so the reference distance is not degenerate."""
    rng = np.random.default_rng(0)
    rows = [_player("Subject", "AMW", base=70)]
    for i in range(30):
        spread = {a: float(rng.integers(30, 95)) for a in OUTFIELD_ATTRIBUTES}
        rows.append(_player(f"Player {i}", "AMW", base=60, **spread))
    return _frame(rows)


def test_a_player_is_identical_to_himself(outfielders):
    """A clone scores 100 on both axes -- the fixed point the scale rests on."""
    twin = dict(outfielders.iloc[0])
    twin["Player"] = "Clone"
    frame = _frame([*outfielders.to_dict("records"), twin])

    _, matches = similar_players(frame, "Subject", "2526", limit=1)

    assert matches[0].player == "Clone"
    assert matches[0].fifa_similarity == 100.0
    assert matches[0].percentile_similarity == 100.0
    assert matches[0].combined == 100.0


def test_similarity_is_symmetric(outfielders):
    """`similar(a, b)` must equal `similar(b, a)`.

    This is why the scale is calibrated on the whole population rather than the
    pool minus the subject -- excluding whoever was asked about would move the
    reference distance and quietly break the property.
    """
    forward, *_ = [m for m in similar_players(outfielders, "Subject", limit=99)[1]]
    back = next(
        m for m in similar_players(outfielders, forward.player, limit=99)[1]
        if m.player == "Subject"
    )
    assert forward.fifa_similarity == back.fifa_similarity
    assert forward.percentile_similarity == back.percentile_similarity
    assert forward.combined == back.combined


def test_closer_attributes_score_higher(outfielders):
    """Ordering follows distance, which the first normalisation failed to do."""
    subject = outfielders.iloc[0]
    near = _player("Near", "AMW", **{a: subject[f"fifa_{a}"] + 2 for a in OUTFIELD_ATTRIBUTES})
    far = _player("Far", "AMW", **{a: subject[f"fifa_{a}"] + 30 for a in OUTFIELD_ATTRIBUTES})
    frame = _frame([*outfielders.to_dict("records"), near, far])

    scores = {m.player: m.fifa_similarity for m in similar_players(frame, "Subject", limit=99)[1]}
    assert scores["Near"] > scores["Far"]


def test_the_scale_discriminates_between_random_players(outfielders):
    """Two unrelated players must not score near-identical.

    Dividing by the theoretical maximum put random same-position pairs at
    89-90%, so ranking was decided by noise. A typical pair belongs near 50.
    """
    scores = [m.fifa_similarity for m in similar_players(outfielders, "Subject", limit=99)[1]]
    assert np.median(scores) < 70


def test_percentiles_are_not_compared_across_roles():
    """A centre-back's 80 and a winger's 80 are ranks in different populations."""
    rows = [_player("Defender", "CB", base=70)]
    rows += [_player(f"Winger {i}", "AMW", base=50 + i * 3) for i in range(10)]
    rows += [_player(f"Back {i}", "CB", base=40 + i * 4) for i in range(10)]

    _, matches = similar_players(_frame(rows), "Defender", limit=99, same_role=False)

    assert any(m.role != "CB" for m in matches), "widening should reach other roles"
    assert all(m.percentile_similarity is None for m in matches)
    assert all(m.fifa_similarity is not None for m in matches)


def test_keepers_are_never_compared_with_outfielders():
    """The two attribute sets measure different things, so no comparison exists."""
    rows = [_player(f"Keeper {i}", "GK", base=50 + i * 4) for i in range(8)]
    rows += [_player(f"Winger {i}", "AMW", base=50 + i * 4) for i in range(8)]

    _, matches = similar_players(_frame(rows), "Keeper 0", limit=99, same_role=False)

    assert matches, "keepers should still find other keepers"
    assert all(m.role == "GK" for m in matches)


def test_a_keeper_is_scored_on_the_keeper_attributes():
    assert attribute_set("GK") == GK_ATTRIBUTES
    assert attribute_set("CB") == OUTFIELD_ATTRIBUTES


def test_a_missing_ea_entry_leaves_the_axis_empty_not_zero(outfielders):
    """8.6% of rated players have no EA row. A zero would read as 'opposite'."""
    absent = _player("No EA", "AMW")
    for attribute in OUTFIELD_ATTRIBUTES:
        absent[f"fifa_{attribute}"] = np.nan
    frame = _frame([*outfielders.to_dict("records"), absent])

    match = next(
        m for m in similar_players(frame, "Subject", limit=99)[1] if m.player == "No EA"
    )
    assert match.fifa_similarity is None
    assert match.percentile_similarity is not None
    assert match.combined == match.percentile_similarity


def test_unknown_player_raises_lookup_error(outfielders):
    with pytest.raises(LookupError):
        similar_players(outfielders, "Nobody At All")
