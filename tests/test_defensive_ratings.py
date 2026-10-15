"""Matching a player whose name one source abbreviates, and refusing to guess.

Thirty of 402 centre-backs in 2025/26 reached the rating step with no FBref
defensive data at all, among them Gabriel, Bremer, Kalulu, Konsa and Ndicka --
first-choice defenders at Arsenal, Juventus, Aston Villa and Roma. They did not
come back empty: the weighted mean filled every missing percentile with 0.5, so
each was published with a defending rating of exactly 50 out of 100, invented
end to end, and defending is 62% of a centre-back's composite.

Two separate faults, tested separately here. The names never matched, and the
miss was then papered over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from footballml.players.fbref import _is_same_name, _tokens, match_players
from footballml.players.rating import MAX_MISSING_WEIGHT, _weighted

# --- one name contains the other ----------------------------------------------

def test_a_single_word_name_matches_the_longer_spelling() -> None:
    """Understat prints "Gabriel"; FBref prints "Gabriel Magalhães"."""
    assert _is_same_name(_tokens("Gabriel"), _tokens("Gabriel Magalhães"))
    assert _is_same_name(_tokens("Bremer"), _tokens("Gleison Bremer"))


def test_a_trailing_extra_word_matches() -> None:
    """The first-plus-last key reads this as "pierre kyatengwa" and misses."""
    assert _is_same_name(_tokens("Pierre Kalulu Kyatengwa"), _tokens("Pierre Kalulu"))
    assert _is_same_name(_tokens("Ezri Konsa Ngoyo"), _tokens("Ezri Konsa"))


def test_it_works_in_both_directions() -> None:
    """Understat is not always the longer one: "Alejandro Catena" / "Catena"."""
    assert _is_same_name(_tokens("Alejandro Catena"), _tokens("Catena"))


def test_a_shared_surname_alone_is_refused() -> None:
    """The case that makes a looser rule dangerous.

    Werder Bremen's "Karim Coulibaly" and FBref's "Abdoul Coulibaly" share a
    surname and nothing else, and are not the same man. A shared-token rule
    would pair them and hand a defender another player's numbers.
    """
    assert not _is_same_name(_tokens("Karim Coulibaly"), _tokens("Abdoul Coulibaly"))


def test_a_different_given_name_is_refused_even_when_it_costs_a_match() -> None:
    """Roma's Evan Ndicka is FBref's "Obite N'Dicka" -- the same man, refused.

    The rule cannot tell this from the Coulibalys without knowing who the
    players are, so it declines both. A missing rating is recoverable; another
    player's tackle count is not.
    """
    assert not _is_same_name(_tokens("Evan Ndicka"), _tokens("Obite N'Dicka"))


def test_an_empty_name_matches_nothing() -> None:
    assert not _is_same_name(_tokens(""), _tokens("Gabriel Magalhães"))


# --- and only when one candidate is left --------------------------------------

def _understat(names: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["E0"] * len(names), "Season": ["2526"] * len(names),
        "Team": ["Arsenal"] * len(names), "Player": names,
    })


def _fbref(names: list[str], tackles: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["E0"] * len(names), "Season": ["2526"] * len(names),
        "Team": ["Arsenal"] * len(names), "Player": names,
        "tackles_won_per90": tackles,
    })


def test_an_exact_match_takes_its_player_out_of_contention() -> None:
    """What resolves Gabriel.

    "gabriel" alone cannot tell Magalhães from Jesus, and both are at Arsenal.
    Jesus matches exactly on the first pass, so by the containment pass only
    one Gabriel is left to take.
    """
    left = _understat(["Gabriel", "Gabriel Jesus"])
    right = _fbref(["Gabriel Magalhães", "Gabriel Jesus"], [1.1, 0.2])

    out = match_players(left, right, on=["League", "Season", "Team"])
    got = dict(zip(out["Player"], out["tackles_won_per90"], strict=True))
    assert got["Gabriel Jesus"] == 0.2
    assert got["Gabriel"] == 1.1


def test_two_live_candidates_are_left_alone() -> None:
    """Without the exact match to clear one away, "Gabriel" is ambiguous."""
    left = _understat(["Gabriel"])
    right = _fbref(["Gabriel Magalhães", "Gabriel Jesus"], [1.1, 0.2])

    out = match_players(left, right, on=["League", "Season", "Team"])
    assert pd.isna(out["tackles_won_per90"].iloc[0])


def test_the_ordinary_exact_match_still_works() -> None:
    left = _understat(["William Saliba"])
    right = _fbref(["William Saliba"], [0.79])
    out = match_players(left, right, on=["League", "Season", "Team"])
    assert out["tackles_won_per90"].iloc[0] == 0.79


# --- a player missing the evidence is not given the midpoint ------------------

def _percentiles(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_a_player_missing_every_metric_comes_back_blank() -> None:
    """The invented 50. `fillna(0.5)` is the exact middle of every distribution.

    Published as a defending rating it is indistinguishable from a measured
    average, which is how 30 centre-backs were scored on nothing at all.
    """
    weights = {"a": 3.0, "b": 2.5}
    frame = _percentiles([
        {"a": 0.9, "b": 0.8},
        {"a": np.nan, "b": np.nan},
        {"a": 0.2, "b": 0.3},
    ])
    out = _weighted(frame, weights)
    assert pd.isna(out.iloc[1])
    assert not pd.isna(out.iloc[0])


def test_a_player_missing_a_little_is_renormalised_not_filled() -> None:
    """Absent evidence, so the rest carries the rating -- the module's own rule.

    Filling with 0.5 drags a strong defender toward the middle on a metric
    nobody measured; renormalising asks what the metrics we *do* have say.
    """
    weights = {"a": 3.0, "b": 1.0}
    frame = _percentiles([{"a": 1.0, "b": np.nan}])
    out = _weighted(frame, weights)
    # Renormalised over the weight present, so a 100th-percentile player on the
    # only measured metric scores 1.0 rather than being pulled to 0.875.
    assert out.iloc[0] == 1.0


def test_nothing_changes_when_nothing_is_missing() -> None:
    """The overwhelming majority of rows, and they must be untouched."""
    weights = {"a": 3.0, "b": 1.0}
    frame = _percentiles([{"a": 0.8, "b": 0.4}])
    expected = (0.8 * 3.0 + 0.4 * 1.0) / 4.0
    assert _weighted(frame, weights).iloc[0] == expected


def test_a_negative_weight_still_flips_the_percentile() -> None:
    """`fouls_per90: -0.5` means lower is better, and renormalising must keep that."""
    frame = _percentiles([{"a": 1.0, "fouls": 1.0}])
    out = _weighted(frame, {"a": 1.0, "fouls": -1.0})
    assert out.iloc[0] == 0.5  # best on one, worst on the other


def test_the_refusal_threshold_is_the_one_already_configured() -> None:
    """Rows and columns are judged by the same bar, deliberately."""
    weights = {"a": 1.0, "b": 1.0}
    # Half the weight missing sits exactly on the limit and is kept.
    assert MAX_MISSING_WEIGHT == 0.5
    out = _weighted(_percentiles([{"a": 0.6, "b": np.nan}]), weights)
    assert out.iloc[0] == 0.6
