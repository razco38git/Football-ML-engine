"""Elo learning from matches it does not report on.

Elo's updates are zero-sum and, until now, every match it saw was domestic --
so the five leagues were five disconnected pools each anchored at 1500 and a
rating could not be compared across a border. Measured on 822 UEFA ties, the
model's predicted home-win probability had *no* relationship with which league
was stronger (slope -0.0002, p 0.94) while the result did (slope +0.0402,
p 1.1e-05).

`add_elo(tmh, extra=...)` is the fix: European results inform the ratings
without becoming part of the domestic record. They must not reach a rolling
form window -- they carry no xG and are not league matches -- so the one thing
these tests guard above all is that `extra` changes the *numbers* and never the
*rows*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.features.build import european_elo_rows
from footballml.features.elo import HOME_ADVANTAGE, START, add_elo


def _long(*matches: tuple[str, str, str, int, int], league: str = "E0", season: str = "2425"):
    """Long team-match rows from (date, home, away, hg, ag) tuples."""
    rows = []
    for date, home, away, hg, ag in matches:
        rows.append({
            "League": league, "Season": season, "Date": pd.Timestamp(date),
            "Team": home, "Opponent": away, "Venue": "Home",
            "GoalsFor": hg, "GoalsAgainst": ag,
        })
        rows.append({
            "League": league, "Season": season, "Date": pd.Timestamp(date),
            "Team": away, "Opponent": home, "Venue": "Away",
            "GoalsFor": ag, "GoalsAgainst": hg,
        })
    return pd.DataFrame(rows)


def _elo(df: pd.DataFrame, team: str, date: str) -> float:
    row = df[(df["Team"] == team) & (df["Date"] == pd.Timestamp(date))]
    return float(row["elo"].iloc[0])


# --- the shape contract -----------------------------------------------------


def test_extra_matches_never_become_rows() -> None:
    """The whole point: they inform the rating, they are not part of the record.

    A European tie in the output would flow into every rolling form window and
    the rest features, and it carries no xG -- so the model would be served a
    different thing from the one it was trained on.
    """
    tmh = _long(("2025-01-01", "Arsenal", "Chelsea", 1, 0))
    extra = _long(("2024-12-01", "Arsenal", "Real Madrid", 3, 0), league="UCL")

    out = add_elo(tmh, extra=extra)
    assert len(out) == len(tmh)
    assert set(out["Team"]) == {"Arsenal", "Chelsea"}
    assert "Real Madrid" not in set(out["Team"])


def test_an_extra_match_moves_the_rating_of_a_later_one() -> None:
    """Arsenal beating Real Madrid must leave them rated higher next week."""
    tmh = _long(("2025-01-01", "Arsenal", "Chelsea", 1, 0))
    win = _long(("2024-12-01", "Arsenal", "Real Madrid", 3, 0), league="UCL")
    loss = _long(("2024-12-01", "Arsenal", "Real Madrid", 0, 3), league="UCL")

    after_win = _elo(add_elo(tmh, extra=win), "Arsenal", "2025-01-01")
    after_loss = _elo(add_elo(tmh, extra=loss), "Arsenal", "2025-01-01")
    assert after_win > START > after_loss


def test_no_extra_matches_reproduces_the_old_behaviour_exactly() -> None:
    """The change must be invisible when there is no European data -- which is
    the state of any checkout that has not run `pipelines.fetch_european`."""
    tmh = _long(("2025-01-01", "Arsenal", "Chelsea", 1, 0),
                ("2025-01-08", "Chelsea", "Arsenal", 2, 2))
    pd.testing.assert_frame_equal(add_elo(tmh), add_elo(tmh, extra=None))
    pd.testing.assert_frame_equal(add_elo(tmh), add_elo(tmh, extra=pd.DataFrame()))


# --- leak safety ------------------------------------------------------------


def test_an_extra_match_after_the_fixture_cannot_reach_it() -> None:
    """The ordering guarantee, which is the same one the rolling windows rely
    on: a rating is recorded going *into* a match, so only earlier results can
    have moved it."""
    tmh = _long(("2025-01-01", "Arsenal", "Chelsea", 1, 0))
    later = _long(("2025-06-01", "Arsenal", "Real Madrid", 5, 0), league="UCL")

    assert _elo(add_elo(tmh, extra=later), "Arsenal", "2025-01-01") == pytest.approx(START)


def test_truncating_the_extra_matches_leaves_earlier_ratings_untouched() -> None:
    tmh = _long(("2025-01-01", "Arsenal", "Chelsea", 1, 0),
                ("2025-03-01", "Arsenal", "Chelsea", 1, 0))
    extra = _long(("2024-12-01", "Arsenal", "Real Madrid", 3, 0),
                  ("2025-02-01", "Arsenal", "Inter", 2, 0), league="UCL")

    full = add_elo(tmh, extra=extra)
    trimmed = add_elo(tmh, extra=extra[extra["Date"] < pd.Timestamp("2025-01-01")])
    assert _elo(full, "Arsenal", "2025-01-01") == pytest.approx(
        _elo(trimmed, "Arsenal", "2025-01-01")
    )


# --- neutral venues ---------------------------------------------------------


def test_a_final_grants_no_home_advantage() -> None:
    """A final is at a neutral ground, so the nominal home side is not at home.

    Two evenly matched sides drawing: with a home term the "home" side is
    expected to win and so loses rating on a draw; on neutral ground a draw is
    exactly par and nothing moves.
    """
    tmh = _long(("2025-06-10", "Arsenal", "Chelsea", 0, 0))
    final = _long(("2025-06-01", "Arsenal", "Inter", 1, 1), league="UCL")

    with_home = final.assign(Neutral=False)
    neutral = final.assign(Neutral=True)

    assert _elo(add_elo(tmh, extra=neutral), "Arsenal", "2025-06-10") == pytest.approx(START)
    assert _elo(add_elo(tmh, extra=with_home), "Arsenal", "2025-06-10") < START


# --- the wide-to-long converter --------------------------------------------


def test_european_elo_rows_emits_two_mirrored_rows_per_match() -> None:
    matches = pd.DataFrame(
        [{
            "League": "UCL", "Season": "2425", "Date": pd.Timestamp("2025-02-19"),
            "HomeTeam": "Arsenal", "AwayTeam": "Real Madrid",
            "FTHG": 2, "FTAG": 1, "Round": "Round of 16",
        }]
    )
    rows = european_elo_rows(matches)
    assert len(rows) == 2
    home = rows[rows["Venue"] == "Home"].iloc[0]
    away = rows[rows["Venue"] == "Away"].iloc[0]
    assert (home["GoalsFor"], home["GoalsAgainst"]) == (2, 1)
    assert (away["GoalsFor"], away["GoalsAgainst"]) == (1, 2)
    assert not home["Neutral"]


def test_only_the_final_is_marked_neutral() -> None:
    matches = pd.DataFrame(
        [
            {"League": "UCL", "Season": "2425", "Date": pd.Timestamp("2025-05-31"),
             "HomeTeam": "Paris SG", "AwayTeam": "Inter", "FTHG": 5, "FTAG": 0,
             "Round": "Final"},
            {"League": "UCL", "Season": "2425", "Date": pd.Timestamp("2025-04-30"),
             "HomeTeam": "Barcelona", "AwayTeam": "Inter", "FTHG": 3, "FTAG": 3,
             "Round": "Semi-finals"},
        ]
    )
    rows = european_elo_rows(matches)
    assert rows[rows["Team"] == "Paris SG"]["Neutral"].all()
    assert not rows[rows["Team"] == "Barcelona"]["Neutral"].any()


def test_an_empty_frame_converts_to_an_empty_frame() -> None:
    assert european_elo_rows(pd.DataFrame()).empty


# --- the property this whole exercise exists for ----------------------------


def test_a_win_abroad_carries_across_the_league_boundary() -> None:
    """Two closed leagues, then one cross-league result, and the two pools are
    no longer independent.

    Without the tie, an English and a Spanish club that have played identical
    domestic records are indistinguishable. With it, the winner outranks the
    loser -- which is the only mechanism in the codebase that can express that.
    """
    english = _long(("2025-01-01", "Arsenal", "Chelsea", 1, 0), league="E0")
    spanish = _long(("2025-01-01", "Barcelona", "Sevilla", 1, 0), league="SP1")
    later = _long(("2025-04-01", "Arsenal", "Chelsea", 0, 0), league="E0")
    later_es = _long(("2025-04-01", "Barcelona", "Sevilla", 0, 0), league="SP1")
    tmh = pd.concat([english, spanish, later, later_es], ignore_index=True)

    apart = add_elo(tmh)
    assert _elo(apart, "Arsenal", "2025-04-01") == pytest.approx(
        _elo(apart, "Barcelona", "2025-04-01")
    ), "identical records in separate leagues should be indistinguishable without a tie"

    tie = _long(("2025-02-01", "Arsenal", "Barcelona", 3, 0), league="UCL")
    joined = add_elo(tmh, extra=tie)
    assert _elo(joined, "Arsenal", "2025-04-01") > _elo(joined, "Barcelona", "2025-04-01")


def test_the_points_one_league_gains_are_the_points_another_loses() -> None:
    """Elo is zero-sum, so a cross-league result moves value between pools
    rather than creating it. That is what makes the leagues comparable at all.
    """
    tmh = pd.concat(
        [
            _long(("2025-04-01", "Arsenal", "Chelsea", 0, 0), league="E0"),
            _long(("2025-04-01", "Barcelona", "Sevilla", 0, 0), league="SP1"),
        ],
        ignore_index=True,
    )
    tie = _long(("2025-02-01", "Arsenal", "Barcelona", 3, 0), league="UCL")
    out = add_elo(tmh, extra=tie)

    gained = _elo(out, "Arsenal", "2025-04-01") - START
    lost = START - _elo(out, "Barcelona", "2025-04-01")
    assert gained == pytest.approx(lost)
    assert gained > 0


def test_home_advantage_is_still_applied_to_ordinary_extra_matches() -> None:
    """A guard on the `Neutral` default: a frame without the column must behave
    as it always did, not silently lose home advantage everywhere."""
    tmh = _long(("2025-06-10", "Arsenal", "Chelsea", 0, 0))
    tie = _long(("2025-06-01", "Arsenal", "Inter", 1, 1), league="UCL")
    assert "Neutral" not in tie.columns

    expected = 1.0 / (1.0 + 10.0 ** (-HOME_ADVANTAGE / 400.0))
    assert expected > 0.5
    # Drew when favoured, so the rating must fall.
    assert _elo(add_elo(tmh, extra=tie), "Arsenal", "2025-06-10") < START


def test_ratings_stay_finite_with_a_lopsided_result() -> None:
    tmh = _long(("2025-06-10", "Arsenal", "Chelsea", 0, 0))
    rout = _long(("2025-06-01", "Arsenal", "Inter", 9, 0), league="UCL")
    assert np.isfinite(_elo(add_elo(tmh, extra=rout), "Arsenal", "2025-06-10"))


# --- the bug that hid for a whole backtest ----------------------------------


def test_domestic_matches_keep_home_advantage_when_extra_is_supplied() -> None:
    """Concatenating a frame that has `Neutral` with one that does not fills the
    gaps with NaN -- and ``if NaN`` is *truthy* in Python, so every domestic
    match silently lost its home advantage.

    It was invisible in the aggregate because a team plays about as often at
    home as away, so the error nearly cancels per team; it surfaced only
    because matches from 2010 changed when the first European result was from
    2016. That is the shape of this failure: not a crash, a quiet corruption of
    every row.
    """
    # Two matches: the second carries the rating the first produced, which is
    # where the missing home advantage shows up.
    tmh = _long(("2025-06-10", "Arsenal", "Chelsea", 0, 0),
                ("2025-06-20", "Arsenal", "Chelsea", 0, 0))
    extra = _long(("2025-06-01", "Liverpool", "Inter", 1, 0), league="UCL")

    alone = _elo(add_elo(tmh), "Arsenal", "2025-06-20")
    with_extra = _elo(add_elo(tmh, extra=extra), "Arsenal", "2025-06-20")
    assert alone == pytest.approx(with_extra), (
        "an unrelated European match changed a domestic rating -- "
        "home advantage is being dropped"
    )
    # And it must be the value home advantage gives: Arsenal were favoured at
    # home and only drew, so they come into the second match rated below START.
    assert alone < START


def test_a_european_result_cannot_reach_a_match_that_precedes_it() -> None:
    """The general form of the same check, and the one that caught the bug:
    no match before the first European result may move at all."""
    tmh = _long(("2015-01-01", "Arsenal", "Chelsea", 2, 0),
                ("2015-02-01", "Chelsea", "Arsenal", 1, 1),
                ("2026-01-01", "Arsenal", "Chelsea", 0, 1))
    extra = _long(("2025-09-13", "Arsenal", "Inter", 4, 0), league="UCL")

    before = add_elo(tmh)
    after = add_elo(tmh, extra=extra)
    early = after["Date"] < pd.Timestamp("2025-09-13")
    pd.testing.assert_series_equal(
        before.loc[early, "elo"], after.loc[early, "elo"], check_names=False
    )
    # ... while the later one does move, or the test would pass vacuously.
    assert _elo(before, "Arsenal", "2026-01-01") != _elo(after, "Arsenal", "2026-01-01")


def test_a_fixture_is_read_before_the_played_match_it_duplicates() -> None:
    """Scoring a European tie means its placeholder and its real result sit on
    the same date, for the same teams -- the one case where ordering inside a
    date decides whether a match sees its own outcome.

    The placeholder must be read first. That used to hold only because every
    domestic league code sorts before "UCL"; now it is explicit.
    """
    played = _long(("2025-02-19", "Arsenal", "Inter", 4, 0), league="UCL")
    # The same fixture as an unplayed placeholder, labelled with the home
    # side's domestic league, exactly as `build_upcoming_features` emits it.
    placeholder = _long(("2025-02-19", "Arsenal", "Inter", 0, 0), league="E0")
    placeholder[["GoalsFor", "GoalsAgainst"]] = np.nan

    out = add_elo(placeholder, extra=played)
    assert _elo(out, "Arsenal", "2025-02-19") == pytest.approx(START), (
        "the fixture saw its own result"
    )


def test_ordering_holds_even_when_the_competition_sorts_first() -> None:
    """The guarantee must not depend on the competition's name. "AAA" sorts
    before every domestic code, so this fails if the fix is alphabetical luck.
    """
    played = _long(("2025-02-19", "Arsenal", "Inter", 4, 0), league="AAA")
    placeholder = _long(("2025-02-19", "Arsenal", "Inter", 0, 0), league="E0")
    placeholder[["GoalsFor", "GoalsAgainst"]] = np.nan

    out = add_elo(placeholder, extra=played)
    assert _elo(out, "Arsenal", "2025-02-19") == pytest.approx(START)
