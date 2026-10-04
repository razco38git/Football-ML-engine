"""European ties: canonicalisation, the big-five filter, and the league fit.

These matches are the only cross-league evidence the project has, and the ways
they can go wrong are quiet ones. A club dropped because its name is spelled
differently on a European page looks exactly like a club correctly dropped for
being Portuguese. A league-strength fit on a handful of matches looks exactly
like one on a thousand. So the filter reports what it discarded and the fit
reports an interval, and both are pinned here.

The network paths are not exercised; what matters is that rows which come back
become the right rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.european import fit_league_strength, head_to_head
from footballml.ingest.european import big_five_only, unmatched_clubs

TMH = pd.DataFrame(
    [
        # (Season, Team, League) -- the domestic record the filter checks against.
        ("2425", "Real Madrid", "SP1"),
        ("2425", "Man City", "E0"),
        ("2425", "Bayern Munich", "D1"),
        ("2425", "Milan", "I1"),
        ("2425", "Paris SG", "F1"),
        # Relegated for 2425: in Europe, but with no domestic season to draw on.
        ("2324", "Leicester", "E0"),
    ],
    columns=["Season", "Team", "League"],
)


def _match(home: str, away: str, hg: int, ag: int, season: str = "2425") -> dict:
    return {
        "League": "UCL", "Season": season, "Date": pd.Timestamp("2025-02-11"),
        "HomeTeam": home, "AwayTeam": away,
        "FTHG": hg, "FTAG": ag,
        "FTR": "H" if hg > ag else "A" if ag > hg else "D",
        "home_league": pd.NA, "away_league": pd.NA,
    }


# --- the big-five filter ----------------------------------------------------


def test_a_tie_between_two_big_five_clubs_is_kept_and_labelled() -> None:
    kept, dropped = big_five_only(pd.DataFrame([_match("Real Madrid", "Man City", 3, 1)]), TMH)
    assert len(kept) == 1
    assert dropped.empty
    assert kept["home_league"].iloc[0] == "SP1"
    assert kept["away_league"].iloc[0] == "E0"


def test_a_tie_against_a_foreign_club_is_dropped_with_a_reason() -> None:
    kept, dropped = big_five_only(pd.DataFrame([_match("Real Madrid", "Benfica", 2, 0)]), TMH)
    assert kept.empty
    assert len(dropped) == 1
    assert "Benfica" in dropped["reason"].iloc[0]


def test_a_club_outside_the_big_five_THIS_season_is_dropped() -> None:
    """Leicester were in the Premier League in 23/24 and not in 24/25.

    Form comes from that season's domestic matches and strength from the one
    before, so a club that has dropped out has neither -- matching on the club
    alone would score it on data that does not exist.
    """
    kept, dropped = big_five_only(
        pd.DataFrame([_match("Real Madrid", "Leicester", 1, 0, season="2425")]), TMH
    )
    assert kept.empty
    assert "Leicester" in dropped["reason"].iloc[0]

    kept_earlier, _ = big_five_only(
        pd.DataFrame([_match("Leicester", "Leicester", 1, 0, season="2324")]), TMH
    )
    assert len(kept_earlier) == 1


def test_both_clubs_missing_says_so_rather_than_naming_one() -> None:
    kept, dropped = big_five_only(pd.DataFrame([_match("Ajax", "Benfica", 1, 1)]), TMH)
    assert kept.empty
    assert dropped["reason"].iloc[0] == "neither club in the big five this season"


def test_the_review_list_counts_each_unresolved_club() -> None:
    """The list a human reads to spot an alias gap, so it must not miss one."""
    _, dropped = big_five_only(
        pd.DataFrame(
            [
                _match("Real Madrid", "Benfica", 2, 0),
                _match("Benfica", "Man City", 0, 3),
                _match("Ajax", "Milan", 1, 1),
            ]
        ),
        TMH,
    )
    review = unmatched_clubs(dropped)
    assert review["Benfica"] == 2
    assert review["Ajax"] == 1
    assert "Real Madrid" not in review


def test_an_empty_input_does_not_explode() -> None:
    kept, dropped = big_five_only(pd.DataFrame(), TMH)
    assert kept.empty and dropped.empty


# --- the league-strength fit ------------------------------------------------


def _synthetic(gap: float, n: int = 400, seed: int = 3) -> pd.DataFrame:
    """Matches generated with a known league gap, to check the fit recovers it.

    E0 is `gap` goals better than F1 per match; home advantage is 0.3.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        home_league, away_league = ("E0", "F1") if i % 2 else ("F1", "E0")
        edge = (gap if home_league == "E0" else -gap) + 0.3
        gd = int(np.round(rng.normal(edge, 1.5)))
        hg = max(0, 1 + max(gd, 0))
        ag = hg - gd
        rows.append(
            {
                "home_league": home_league, "away_league": away_league,
                "FTHG": hg, "FTAG": max(ag, 0),
                "FTR": "H" if hg > ag else "A" if ag > hg else "D",
            }
        )
    return pd.DataFrame(rows)


def test_the_fit_recovers_a_known_gap() -> None:
    table = fit_league_strength(_synthetic(gap=0.5), n_bootstrap=200)
    strength = table.set_index("league")["strength"]
    assert strength["E0"] == 0.0  # the reference, pinned
    # F1 should come out about half a goal worse.
    assert -0.75 < strength["F1"] < -0.25


def test_the_fit_recovers_home_advantage() -> None:
    table = fit_league_strength(_synthetic(gap=0.5), n_bootstrap=0)
    advantage = table.set_index("league").loc["home_advantage", "strength"]
    assert 0.1 < advantage < 0.5


def test_a_true_gap_of_zero_gives_an_interval_spanning_zero() -> None:
    """The result that must not be dressed up: leagues that are level."""
    table = fit_league_strength(_synthetic(gap=0.0), n_bootstrap=400)
    row = table.set_index("league").loc["F1"]
    assert row["low"] < 0 < row["high"]


def test_the_interval_narrows_with_more_matches() -> None:
    narrow = fit_league_strength(_synthetic(0.5, n=1600), n_bootstrap=300).set_index("league")
    wide = fit_league_strength(_synthetic(0.5, n=120), n_bootstrap=300).set_index("league")
    assert (narrow.loc["F1", "high"] - narrow.loc["F1", "low"]) < (
        wide.loc["F1", "high"] - wide.loc["F1", "low"]
    )


def test_too_few_leagues_returns_empty_rather_than_a_fake_number() -> None:
    one_league = pd.DataFrame(
        [{"home_league": "E0", "away_league": "E0", "FTHG": 1, "FTAG": 0, "FTR": "H"}]
    )
    assert fit_league_strength(one_league).empty


# --- the raw record ---------------------------------------------------------


def test_head_to_head_counts_both_sides_of_each_match() -> None:
    matches = pd.DataFrame(
        [
            {"home_league": "E0", "away_league": "F1", "FTHG": 2, "FTAG": 0, "FTR": "H"},
            {"home_league": "F1", "away_league": "E0", "FTHG": 1, "FTAG": 1, "FTR": "D"},
        ]
    )
    table = head_to_head(matches)
    assert table.loc["E0", "played"] == 2
    assert table.loc["F1", "played"] == 2
    assert table.loc["E0", "points_per_match"] == pytest.approx(4 / 2)
    assert table.loc["F1", "points_per_match"] == pytest.approx(1 / 2)
    assert table.loc["E0", "goal_diff_per_match"] == pytest.approx((3 - 1) / 2)


def test_same_league_ties_are_excluded_from_the_record() -> None:
    """Two English clubs meeting in Europe says nothing about England."""
    matches = pd.DataFrame(
        [{"home_league": "E0", "away_league": "E0", "FTHG": 2, "FTAG": 0, "FTR": "H"}]
    )
    assert head_to_head(matches).empty


# --- the alias supplement ---------------------------------------------------


def test_the_european_alias_supplement_is_applied() -> None:
    """FBref's European pages spell a few clubs differently from its domestic
    ones, and the generated alias map only knows the domestic spellings.

    Hertha is the case that exists: "Hertha BSC" in Europe against
    football-data's "Hertha". They were relegated after 2022/23, so the alias
    derivation -- which works from domestic fixtures -- never produced it, and
    six Europa League matches were being discarded in silence.
    """
    from footballml.ingest.european import EUROPEAN_ALIASES

    assert EUROPEAN_ALIASES["Hertha BSC"] == "Hertha"


def test_a_supplemented_name_survives_the_big_five_filter() -> None:
    """The supplement is worthless unless the filter then recognises the club."""
    tmh = pd.DataFrame(
        [("1718", "Hertha", "D1"), ("1718", "Milan", "I1")],
        columns=["Season", "Team", "League"],
    )
    # As the fetch would emit it, i.e. already canonicalised.
    match = _match("Hertha", "Milan", 1, 2, season="1718")
    kept, dropped = big_five_only(pd.DataFrame([match]), tmh)
    assert len(kept) == 1
    assert dropped.empty
    assert kept["home_league"].iloc[0] == "D1"


# --- round classification ---------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Group stage", "round_robin"),      # up to 2023/24
        ("League phase", "round_robin"),     # the 2024/25 format change
        ("Round of 32", "two_legged"),
        ("Round of 16", "two_legged"),
        ("Quarter-finals", "two_legged"),
        ("Semi-finals", "two_legged"),
        ("Knockout round play-offs", "two_legged"),
        ("Knockout phase play-offs", "two_legged"),
        ("Final", "neutral"),
        ("Some New UEFA Name", "unclassified"),
    ],
)
def test_rounds_are_classified(name: str, expected: str) -> None:
    """The three kinds behave differently enough that pooling them would hide
    whatever the knockout rounds are doing."""
    from footballml.ingest.european import classify_round

    assert classify_round(name) == expected


def test_an_unknown_round_is_labelled_not_silently_counted_as_normal() -> None:
    """UEFA renames its rounds every few years. A new label must surface as its
    own bucket rather than be counted as ordinary round-robin football."""
    from footballml.ingest.european import classify_round

    assert classify_round("League phase play-in") == "unclassified"
    assert classify_round(None) == "unclassified"


# --- registering a competition after a reader already exists ----------------


def test_registering_works_even_after_a_reader_has_been_built() -> None:
    """The reason the site showed no Champions League matches at all.

    `BaseReader._all_leagues` snapshots `LEAGUE_DICT` onto the *class* the
    first time any reader is built, and every caller builds a domestic reader
    first -- `score_upcoming` fetches the five leagues' schedule before the
    European one. So by the time `_register` ran, the snapshot already existed
    without a UEFA entry and the registration had no effect. soccerdata then
    raised "Invalid league 'UEFA-Champions League'", which was caught, logged
    at warning level and turned into zero fixtures.
    """
    sd = pytest.importorskip("soccerdata")
    from footballml.ingest.european import _register

    # Stand in for the domestic reader that always comes first.
    sd.FBref._all_leagues()
    assert "_all_leagues_dict" in vars(sd.FBref), "test no longer reproduces the setup"
    assert "UEFA-Champions League" not in sd.FBref._all_leagues()

    _register(["UCL"])
    assert "UEFA-Champions League" in sd.FBref._all_leagues(), (
        "registration after the first reader still does not take effect"
    )


def test_registering_is_idempotent() -> None:
    """It runs on every `fetch_schedule` call, so it must not accumulate or
    thrash the snapshot into uselessness."""
    sd = pytest.importorskip("soccerdata")
    from footballml.ingest.european import _register

    _register(["UCL", "UEL"])
    first = dict(sd.FBref._all_leagues())
    _register(["UCL", "UEL"])
    assert sd.FBref._all_leagues() == first
