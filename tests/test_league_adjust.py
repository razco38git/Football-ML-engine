"""The correction applied where the model cannot learn one.

Every match the model trains on is domestic, so the league gap is always zero
in training and it never learns what a cross-league rating difference means.
Measured on 788 UEFA ties, its predicted home-win probability moves at +0.0034
per point of league gap where the truth moves at +0.0402.

These pin the three things that make a correction like this safe rather than a
thumb on the scale: it is the identity within a league, it only ever shifts the
*balance* of a match and never its total, and it is judged on matches it was
not fitted on.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from footballml.league_adjust import adjust, fit, load, save

OFFSETS = {"E0": 0.0, "SP1": -0.18, "F1": -0.35}


# --- applying it ------------------------------------------------------------


def test_a_same_league_pairing_is_untouched() -> None:
    """The correction has to be safe to apply everywhere, including to the
    29,143 domestic matches it knows nothing about."""
    home, away = adjust(1.5, 1.2, "E0", "E0", OFFSETS)
    assert home[0] == pytest.approx(1.5)
    assert away[0] == pytest.approx(1.2)


def test_an_unknown_league_is_treated_as_the_reference() -> None:
    """A division with no fitted offset must not silently become the strongest
    or weakest -- it gets no shift at all."""
    home, away = adjust(1.5, 1.2, "E0", "XX", OFFSETS)
    assert home[0] == pytest.approx(1.5)
    assert away[0] == pytest.approx(1.2)


def test_the_stronger_league_gains_and_the_weaker_loses() -> None:
    home, away = adjust(1.5, 1.5, "E0", "F1", OFFSETS)
    assert home[0] > 1.5 > away[0]

    # And the other way round when the stronger side travels.
    home, away = adjust(1.5, 1.5, "F1", "E0", OFFSETS)
    assert home[0] < 1.5 < away[0]


def test_the_correction_shifts_balance_without_inflating_the_match() -> None:
    """Half the gap up, half down, in log space. A mismatch should not become a
    higher-scoring game just because the sides are unevenly matched -- that
    would quietly move every over/under and both-teams-to-score number too.
    """
    home, away = adjust(1.5, 1.5, "E0", "F1", OFFSETS)
    assert float(home[0] * away[0]) == pytest.approx(1.5 * 1.5)


def test_reversing_the_fixture_reverses_the_shift() -> None:
    forward_home, forward_away = adjust(1.4, 1.4, "E0", "F1", OFFSETS)
    reverse_home, reverse_away = adjust(1.4, 1.4, "F1", "E0", OFFSETS)
    assert forward_home[0] == pytest.approx(reverse_away[0])
    assert forward_away[0] == pytest.approx(reverse_home[0])


def test_it_vectorises_over_a_batch() -> None:
    home, away = adjust(
        np.array([1.5, 1.5]), np.array([1.5, 1.5]),
        np.array(["E0", "E0"]), np.array(["F1", "E0"]), OFFSETS,
    )
    assert home[0] > 1.5
    assert home[1] == pytest.approx(1.5)


def test_no_offsets_means_no_change() -> None:
    """The state of any checkout that has not fitted one."""
    home, away = adjust(1.5, 1.2, "E0", "F1", {})
    assert (home[0], away[0]) == pytest.approx((1.5, 1.2))


# --- fitting it -------------------------------------------------------------


def _synthetic(n: int = 600, true_gap: float = 0.4, seed: int = 5) -> pd.DataFrame:
    """Matches where the model is deliberately blind to a real league gap.

    Both sides are predicted at 1.4 goals whoever they are, while the truth is
    generated with E0 clubs better by `true_gap` in log space -- which is the
    failure the correction exists to repair.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        home_league, away_league = ("E0", "F1") if i % 2 else ("F1", "E0")
        gap = true_gap if home_league == "E0" else -true_gap
        mu_h, mu_a = 1.4 * np.exp(0.5 * gap), 1.4 * np.exp(-0.5 * gap)
        hg, ag = rng.poisson(mu_h), rng.poisson(mu_a)
        rows.append({
            "Season": f"{2000 + i % 6}",
            "home_league": home_league, "away_league": away_league,
            "expected_goals_home": 1.4, "expected_goals_away": 1.4,
            "prob_home_win": 0.4, "prob_draw": 0.27, "prob_away_win": 0.33,
            "FTHG": hg, "FTAG": ag,
            "FTR": "H" if hg > ag else "A" if ag > hg else "D",
        })
    return pd.DataFrame(rows)


def test_the_fit_recovers_a_gap_the_model_was_blind_to() -> None:
    fitted = fit(_synthetic(true_gap=0.4))
    offsets = fitted["offsets"]
    assert offsets["E0"] == 0.0
    assert -0.55 < offsets["F1"] < -0.25


def test_a_model_that_is_already_right_needs_no_correction() -> None:
    """The guard against inventing a gap: when the predicted rates already
    match the truth, the fitted offsets must be about zero."""
    frame = _synthetic(true_gap=0.0)
    offsets = fit(frame)["offsets"]
    assert abs(offsets["F1"]) < 0.15


def test_the_holdout_is_scored_on_seasons_it_was_not_fitted_on() -> None:
    """The only number worth believing here. A correction fitted and judged on
    the same matches always looks good."""
    holdout = fit(_synthetic(true_gap=0.4))["holdout"]
    assert holdout["n"] > 0
    assert holdout["rps_corrected"] < holdout["rps_uncorrected"]
    assert holdout["improvement"] > 0


def test_no_cross_league_matches_raises_rather_than_returning_zeros() -> None:
    """Zeros would look like 'the leagues are level' instead of 'no evidence'."""
    frame = _synthetic()
    frame["away_league"] = frame["home_league"]
    with pytest.raises(ValueError, match="cross-league"):
        fit(frame)


# --- storing it -------------------------------------------------------------


def test_a_missing_file_loads_as_no_correction(tmp_path) -> None:
    assert load(tmp_path / "nope.json") == {}


def test_what_is_saved_is_what_loads(tmp_path) -> None:
    path = tmp_path / "adj.json"
    save({"offsets": OFFSETS, "reference": "E0", "n_matches": 10, "holdout": {}}, path)
    assert load(path) == OFFSETS


def test_a_corrupt_file_loads_as_no_correction(tmp_path) -> None:
    """Degrade to the uncorrected model rather than crash every prediction."""
    path = tmp_path / "adj.json"
    path.write_text("{not json", encoding="utf-8")
    assert load(path) == {}

    path.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
    assert load(path) == {}
