"""Recent results when the results feed falls behind.

football-data is the results source and it lags. In October 2026 the match
history ended 2026-09-20 while the leagues had played on to 2026-10-10, so
"Recent results" stopped three weeks short with nothing to tell a reader that a
stale feed was not a quiet fortnight.

Two things close that gap, and both are pinned here. The schedule pages carry
the scores and were already being scraped for fixtures -- the goals were parsed
and then thrown away. And the forecasts for those matches were already stored,
so once settled they can be served directly.

The risk being guarded is mixing the two silently: a row re-scored by today's
model and a row committed to before kickoff are different claims, so the second
is flagged rather than quietly blended in.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pandas as pd
import pytest

from footballml.ingest import schedule

api = importlib.import_module("footballml.api.app")


# --- the schedule must stop discarding the score ------------------------------

def test_score_regex_captures_both_goals() -> None:
    """`_SCORE` always had two groups; only group 0 was ever used."""
    found = pd.Series(["2–1", "0–0", "10–3"]).str.extract(schedule._SCORE)
    assert list(found[0]) == ["2", "0", "10"]
    assert list(found[1]) == ["1", "0", "3"]


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "‒", "–", "—", "―"])
def test_every_dash_fbref_might_use_parses(dash: str) -> None:
    """A dash this misses would mark a played match unplayed, and lose its score."""
    found = pd.Series([f"3{dash}2"]).str.extract(schedule._SCORE)
    assert found.iloc[0].tolist() == ["3", "2"]


def test_a_fixture_list_never_carries_result_columns() -> None:
    """`FTHG`/`FTAG` are the training table's name for the truth.

    An unplayed fixture has NaN in both, and letting those ride along would put
    empty truth columns beside rows the model is about to predict.
    """
    assert set(schedule._RESULT_ONLY) == {"FTHG", "FTAG", "played"}
    empty = schedule._empty()
    for column in schedule._RESULT_ONLY:
        assert column in empty.columns


# --- /matches falls back to the store -----------------------------------------

def _stored(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "League": ["E0"] * len(dates),
        "competition": ["domestic"] * len(dates),
        "Date": pd.to_datetime(dates),
        "HomeTeam": [f"Home{i}" for i in range(len(dates))],
        "AwayTeam": [f"Away{i}" for i in range(len(dates))],
        "expected_goals_home": [1.5] * len(dates),
        "expected_goals_away": [1.1] * len(dates),
        "prob_home_win": [0.5] * len(dates),
        "prob_draw": [0.26] * len(dates),
        "prob_away_win": [0.24] * len(dates),
        "predicted_outcome": ["H"] * len(dates),
        "modal_score_home": [1] * len(dates),
        "modal_score_away": [1] * len(dates),
        "prob_over_2_5": [0.52] * len(dates),
        "prob_btts": [0.54] * len(dates),
        "actual_home_goals": [2.0] * len(dates),
        "actual_away_goals": [1.0] * len(dates),
        "actual_result": ["H"] * len(dates),
    })


def test_only_matches_newer_than_the_feature_table_are_added(monkeypatch) -> None:
    """The store must top up, never duplicate what has already been re-scored."""
    monkeypatch.setattr(
        api.store, "settled",
        lambda *a, **k: _stored(["2026-09-19", "2026-09-20", "2026-10-10"]),
    )
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)
    assert [str(p.date) for p in out] == ["2026-10-10"]


def test_added_rows_are_flagged_as_pre_kickoff(monkeypatch) -> None:
    """A stored forecast and a re-scored one are different claims."""
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-10"]))
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)
    assert out[0].predicted_before_kickoff is True
    assert out[0].actual_result == "H"
    assert out[0].actual_home_goals == 2


def test_a_stored_row_with_odds_reports_the_market(monkeypatch) -> None:
    """Discover filters its bookmaker list on these fields.

    The store carries B365 prices and this path was dropping them, so eight of
    the fifty-two settled predictions were silently absent from "where we
    disagreed with the bookmakers" while their odds sat unused.
    """
    rows = _stored(["2026-10-12"]).assign(B365H=2.0, B365D=4.0, B365A=4.0)
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: rows)

    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    # De-vigged: 1/2 : 1/4 : 1/4 normalises to 0.5 / 0.25 / 0.25.
    assert out.market_prob_home == pytest.approx(0.5, abs=1e-3)
    assert out.market_prob_draw == pytest.approx(0.25, abs=1e-3)
    assert out.market_prob_away == pytest.approx(0.25, abs=1e-3)
    assert out.market_prob_home + out.market_prob_draw + out.market_prob_away == pytest.approx(
        1.0, abs=1e-3
    )


def test_a_stored_row_without_odds_has_no_market(monkeypatch) -> None:
    """Most rows have none, because football-data was unreachable when stored."""
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-12"]))
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    assert out.market_prob_home is None
    assert out.market_prob_draw is None
    assert out.market_prob_away is None


def test_a_partial_price_is_treated_as_no_price(monkeypatch) -> None:
    """Two of three odds cannot be de-vigged into anything meaningful."""
    rows = _stored(["2026-10-12"]).assign(B365H=2.0, B365D=4.0, B365A=float("nan"))
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: rows)
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0].market_prob_home is None


def test_rows_are_newest_first(monkeypatch) -> None:
    monkeypatch.setattr(
        api.store, "settled",
        lambda *a, **k: _stored(["2026-10-01", "2026-10-10", "2026-10-05"]),
    )
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)
    assert [str(p.date) for p in out] == ["2026-10-10", "2026-10-05", "2026-10-01"]


def test_the_league_filter_applies(monkeypatch) -> None:
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-10"]))
    assert len(api._recent_from_store(pd.Timestamp("2026-09-20"), "E0")) == 1
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), "SP1") == []


def test_a_caught_up_feature_table_adds_nothing(monkeypatch) -> None:
    """When football-data recovers, the fallback must get out of the way."""
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _stored(["2026-10-10"]))
    assert api._recent_from_store(pd.Timestamp("2026-10-10"), None) == []


def test_an_unreadable_store_does_not_break_the_page(monkeypatch) -> None:
    """Recent results without the top-up beats no recent results at all."""
    def broken(*_a, **_k):
        raise OSError("predictions.csv is locked")

    monkeypatch.setattr(api.store, "settled", broken)
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), None) == []


def test_an_empty_store_adds_nothing(monkeypatch) -> None:
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: pd.DataFrame())
    assert api._recent_from_store(pd.Timestamp("2026-09-20"), None) == []


# --- the expanded card, for a row that came from the store --------------------
#
# A store row carries the two goal rates and the modal score and nothing else,
# so the expanded card for a European tie had an empty score grid, no likely
# scores, no modal probability and no squad strength -- while a domestic tie
# beside it, re-scored from the feature table, had all four. Same component,
# same schema, visibly less in it.
#
# These are *reconstructed* from the stored rates rather than back-filled from
# today's model, and the test that matters is the one below pinning that the
# reconstruction agrees with what the prediction itself recorded.

@pytest.fixture
def scoring_model(monkeypatch):
    """A model stub, because the grid is rebuilt through the Dixon-Coles matrix.

    `_scoreline_detail` needs only the two fitted constants, which is the point:
    the goal rates come from the stored prediction and nothing else about
    today's model enters the reconstruction.
    """
    monkeypatch.setattr(
        api.state, "model", SimpleNamespace(rho_=-0.0567642328, max_goals=10)
    )


def _european(home: str = "Roma", away: str = "Real Madrid") -> pd.DataFrame:
    rows = _stored(["2026-10-14"])
    return rows.assign(
        competition="UCL", League="I1", HomeTeam=home, AwayTeam=away,
        expected_goals_home=1.7, expected_goals_away=1.42,
    )


def test_a_stored_row_gets_a_score_grid(monkeypatch, scoring_model) -> None:
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _european())
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]

    size = api.SCORE_GRID_MAX + 1
    assert len(out.score_grid) == size
    assert all(len(row) == size for row in out.score_grid)
    assert sum(sum(row) for row in out.score_grid) == pytest.approx(1.0, abs=0.02)


def test_the_reconstruction_agrees_with_the_stored_modal_score(monkeypatch, scoring_model) -> None:
    """The property that makes rebuilding the grid honest rather than invented.

    The two goal rates are the prediction's own output, so running them back
    through the same Dixon-Coles matrix must recover the scoreline the
    prediction actually recorded. If these ever disagree, the grid being drawn
    is not the distribution the forecast came from.
    """
    rows = _european().assign(modal_score_home=1, modal_score_away=1)
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: rows)
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]

    top = out.likely_scores[0]
    assert (top.home, top.away) == (out.modal_score_home, out.modal_score_away)
    assert out.prob_modal_score == pytest.approx(top.probability, abs=1e-4)


def test_likely_scores_are_ordered_and_capped(monkeypatch, scoring_model) -> None:
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _european())
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]

    assert len(out.likely_scores) == api.TOP_SCORES
    probabilities = [s.probability for s in out.likely_scores]
    assert probabilities == sorted(probabilities, reverse=True)


def test_squad_strength_reaches_both_sides_of_a_european_tie(monkeypatch) -> None:
    """The join keys on season and team without league, which is why this works.

    A UEFA tie has one side in Serie A and the other in La Liga. Keyed on
    league as well, the away side would match nothing and be shown with no
    strength at all.
    """
    strength = pd.DataFrame({
        "League": ["I1", "SP1"],
        "Season": ["2526", "2526"],
        "Team": ["Roma", "Real Madrid"],
        "strength_overall": [75.2, 81.5],
        "strength_attack": [75.2, 84.2],
        "strength_defence": [75.6, 78.9],
    })
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _european())
    monkeypatch.setattr(api.state, "strength", strength)

    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    assert out.strength_home_overall == 75.2
    assert out.strength_away_overall == 81.5
    assert out.strength_away_attack == 84.2


def test_strength_comes_from_the_previous_season(monkeypatch) -> None:
    """Not a nicety: `team_strength.csv` has no rows for the season in progress.

    Ratings need a minimum of minutes, so four weeks into 2026/27 there were
    none. Joining the match's own season would show an empty panel on every
    current fixture, which is what the first attempt at this did.
    """
    strength = pd.DataFrame({
        "League": ["I1"], "Season": ["2526"], "Team": ["Roma"],
        "strength_overall": [75.2], "strength_attack": [75.2], "strength_defence": [75.6],
    })
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _european())
    monkeypatch.setattr(api.state, "strength", strength)

    # The match is in 2627 and the only strength row is 2526.
    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    assert out.strength_home_overall == 75.2


def test_club_names_are_translated_before_the_join(monkeypatch) -> None:
    """Strength carries Understat's names and the store carries football-data's.

    "Man City" against "Manchester City" fails as a left merge -- silently, as
    a NaN -- so without the alias map the biggest clubs are exactly the ones
    that show no strength.
    """
    strength = pd.DataFrame({
        "League": ["E0"], "Season": ["2526"], "Team": ["Manchester City"],
        "strength_overall": [80.8], "strength_attack": [83.7], "strength_defence": [78.0],
    })
    rows = _european(home="Man City", away="Paris SG").assign(League="E0")
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: rows)
    monkeypatch.setattr(api.state, "strength", strength)

    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    assert out.strength_home_overall == 80.8


def test_no_strength_table_is_not_an_error(monkeypatch, scoring_model) -> None:
    """An empty panel beats a 500."""
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _european())
    monkeypatch.setattr(api.state, "strength", pd.DataFrame())

    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    assert out.strength_home_overall is None
    assert len(out.score_grid) == api.SCORE_GRID_MAX + 1


def test_an_unrated_club_simply_has_no_strength(monkeypatch) -> None:
    """A newly promoted side has no previous-season row in its new league."""
    strength = pd.DataFrame({
        "League": ["I1"], "Season": ["2526"], "Team": ["Roma"],
        "strength_overall": [75.2], "strength_attack": [75.2], "strength_defence": [75.6],
    })
    monkeypatch.setattr(api.store, "settled", lambda *a, **k: _european())
    monkeypatch.setattr(api.state, "strength", strength)

    out = api._recent_from_store(pd.Timestamp("2026-09-20"), None)[0]
    assert out.strength_home_overall == 75.2
    assert out.strength_away_overall is None


# --- European ties under "All competitions" ----------------------------------
#
# They were reachable only by picking Champions League from the dropdown. The
# Upcoming board has always merged both sources and filtered per request, so a
# tie appeared there, then vanished the moment it was played unless the reader
# changed the filter. Same page, same dropdown, opposite behaviour.


def _scored_european(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "competition": comp, "Date": pd.Timestamp(date),
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": 2.0, "FTAG": 1.0, "FTR": "H",
            "home_league": "E0", "away_league": "SP1",
            "expected_goals_home": 1.6, "expected_goals_away": 1.1,
            "prob_home_win": 0.5, "prob_draw": 0.26, "prob_away_win": 0.24,
            "predicted_outcome": "H", "modal_score_home": 1, "modal_score_away": 1,
            "prob_over_2_5": 0.5, "prob_btts": 0.5,
        }
        for date, home, away, comp in rows
    ])


def test_a_played_tie_is_reachable_without_changing_the_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        api.state, "european",
        _scored_european([("2026-10-21", "Bayern Munich", "Arsenal", "UCL")]),
    )

    out = api._european_matches(None, 10, 0)

    assert [(p.home_team, p.away_team) for p in out] == [("Bayern Munich", "Arsenal")]


def test_the_unshown_competitions_stay_unshown(monkeypatch) -> None:
    """Europa and Conference are scored into the same file because they feed
    the cross-league correction, not because they are fit to display -- we can
    rate 14% and 6% of their ties. Widening the merge must not widen that."""
    monkeypatch.setattr(
        api.state, "european",
        _scored_european([
            ("2026-10-22", "Bournemouth", "Milan", "UEL"),
            ("2026-10-22", "Monaco", "Freiburg", "UECL"),
            ("2026-10-21", "Bayern Munich", "Arsenal", "UCL"),
        ]),
    )

    out = api._european_matches(None, 10, 0)

    assert {p.competition for p in out} == {"UCL"}


def test_an_unshown_competition_returns_nothing_however_it_is_asked_for(monkeypatch) -> None:
    """Second lock on the same door: the endpoint refuses to pass one, and the
    helper refuses to serve one."""
    monkeypatch.setattr(
        api.state, "european",
        _scored_european([
            ("2026-10-21", "Bayern Munich", "Arsenal", "UCL"),
            ("2026-10-22", "Bournemouth", "Milan", "UEL"),
        ]),
    )

    assert {p.competition for p in api._european_matches("UCL", 10, 0)} == {"UCL"}
    assert api._european_matches("UEL", 10, 0) == []


def test_no_european_results_is_not_an_error(monkeypatch) -> None:
    monkeypatch.setattr(api.state, "european", pd.DataFrame())
    assert api._european_matches(None, 10, 0) == []


# --- how likely was that result, given the chances ---------------------------
#
# A retrodiction, not a forecast: it reads the match's own xG, so it could only
# ever be computed afterwards. The four ways it can be unavailable are all
# different and all must degrade to `None` rather than to a zero, because "this
# result had a 0.0% chance" is a confident number with nothing behind it.


@pytest.fixture
def scored_match(monkeypatch):
    """One settled match whose own xG is known."""
    monkeypatch.setattr(
        api.state, "match_xg",
        {(pd.Timestamp("2026-10-19").date(), "Arsenal", "Chelsea"): (2.40, 0.60)},
    )
    monkeypatch.setattr(api.state, "metadata", SimpleNamespace(rho=-0.0567))
    return pd.Series({
        "Date": pd.Timestamp("2026-10-19"),
        "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "FTR": "H",
    })


def test_the_chances_favour_the_side_that_created_them(scored_match) -> None:
    r = api._retrodiction(scored_match)

    assert r is not None
    assert r.prob_home_win > r.prob_away_win
    assert r.prob_home_win + r.prob_draw + r.prob_away_win == pytest.approx(1.0, abs=1e-3)
    assert r.xg_home == 2.40 and r.xg_away == 0.60


def test_prob_actual_is_the_component_that_happened(scored_match) -> None:
    """The headline number on the card. Reading the wrong component would be
    invisible on a home win and wrong on every away win."""
    assert api._retrodiction(scored_match).prob_actual == pytest.approx(
        api._retrodiction(scored_match).prob_home_win
    )

    away = scored_match.copy()
    away["FTR"] = "A"
    r = api._retrodiction(away)
    assert r.prob_actual == pytest.approx(r.prob_away_win)


@pytest.mark.parametrize(
    ("reason", "mutate"),
    [
        ("an unplayed fixture", lambda s: s.drop(labels=["FTR"])),
        ("a match with no xG on record", lambda s: s.replace({"Arsenal": "Burnley"})),
    ],
)
def test_it_is_absent_rather_than_zero(scored_match, reason, mutate) -> None:
    assert api._retrodiction(mutate(scored_match)) is None, reason


def test_an_abandoned_match_is_gated_out(monkeypatch, scored_match) -> None:
    """Bastia 0-3 Lyon, 2017-04-16: abandoned after crowd trouble and awarded,
    so both sides record 0.00 xG. One match in 21,933, and without the gate the
    page would tell a reader that result had a 0.0% chance."""
    monkeypatch.setattr(
        api.state, "match_xg",
        {(pd.Timestamp("2026-10-19").date(), "Arsenal", "Chelsea"): (0.0, 0.0)},
    )
    assert api._retrodiction(scored_match) is None


def test_the_gate_lets_a_genuinely_quiet_match_through(monkeypatch, scored_match) -> None:
    """The floor has to catch the degenerate case and nothing else. The 0.1%
    quantile of combined match xG is 0.431, well clear of 0.1."""
    monkeypatch.setattr(
        api.state, "match_xg",
        {(pd.Timestamp("2026-10-19").date(), "Arsenal", "Chelsea"): (0.25, 0.20)},
    )
    assert api._retrodiction(scored_match) is not None


def test_the_lookup_reads_both_sides_off_the_home_row() -> None:
    """`xGAgainst` on the home row equals the away row's `xGFor` on all 29,239
    rows, so there is nothing to merge -- and merging would risk duplicating."""
    tmh = pd.DataFrame([
        {"Date": pd.Timestamp("2026-10-19"), "Team": "Arsenal", "Opponent": "Chelsea",
         "Venue": "Home", "xGFor": 2.4, "xGAgainst": 0.6},
        {"Date": pd.Timestamp("2026-10-19"), "Team": "Chelsea", "Opponent": "Arsenal",
         "Venue": "Away", "xGFor": 0.6, "xGAgainst": 2.4},
    ])

    lookup = api._match_xg_lookup(tmh)

    assert lookup == {(pd.Timestamp("2026-10-19").date(), "Arsenal", "Chelsea"): (2.4, 0.6)}


def test_a_history_without_xg_columns_is_survivable() -> None:
    """Every season before 2014/15 -- Understat does not go back further."""
    assert api._match_xg_lookup(pd.DataFrame()) == {}
    assert api._match_xg_lookup(pd.DataFrame({"Team": ["A"]})) == {}
