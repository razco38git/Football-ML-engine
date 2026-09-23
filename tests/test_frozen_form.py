"""Tests for whole-season feature building.

These exist because of a bug that produced a perfectly-shaped frame of silent
NaN. The season projection handed all 330 of a league's remaining fixtures to
``build_upcoming_features`` on a single date, so each team got ~33 placeholder
rows stacked together; every rolling window then looked back at the *other
placeholders* rather than at real matches. 31 of 232 model features came back
entirely empty, the model could no longer tell the teams apart, and the
projected tables bunched into the sixties.

Nothing caught it because every check was on shape, not content. So the test
that matters here is the one that looks at content.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.features.build import (
    build_frozen_form_features,
    build_upcoming_features,
)


def _history(teams: list[str], rounds: int = 12) -> pd.DataFrame:
    """A played history long enough to fill a 19-match window."""
    rng = np.random.default_rng(0)
    rows = []
    date = pd.Timestamp("2026-08-01")
    for _ in range(rounds):
        rng.shuffle(teams)
        for i in range(0, len(teams) - 1, 2):
            home, away = teams[i], teams[i + 1]
            hg, ag = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            for team, opp, venue, gf, ga in (
                (home, away, "Home", hg, ag),
                (away, home, "Away", ag, hg),
            ):
                rows.append({
                    "League": "E0", "Season": "2627", "Date": date,
                    "Team": team, "Opponent": opp, "Venue": venue,
                    "GoalsFor": gf, "GoalsAgainst": ga,
                    "Result": "W" if gf > ga else "D" if gf == ga else "L",
                    "ShotsFor": int(rng.integers(5, 20)),
                    "ShotsAgainst": int(rng.integers(5, 20)),
                    "ShotsOnTargetFor": int(rng.integers(1, 9)),
                    "ShotsOnTargetAgainst": int(rng.integers(1, 9)),
                })
        date += pd.Timedelta(days=7)
    return pd.DataFrame(rows)


def _all_pairs(teams: list[str]) -> pd.DataFrame:
    return pd.DataFrame([
        {"League": "E0", "Season": "2627", "HomeTeam": h, "AwayTeam": a}
        for h in teams for a in teams if h != a
    ])


@pytest.fixture
def teams() -> list[str]:
    return [f"Team {i}" for i in range(10)]


def test_a_whole_season_keeps_its_form_features(teams):
    """The regression that matters: no feature may come back entirely empty.

    Shape assertions passed throughout the original bug. Only looking at the
    values reveals it.
    """
    history = _history(list(teams))
    built = build_frozen_form_features(history, _all_pairs(teams))

    form = [c for c in built.columns if "_last_" in c]
    assert form, "no form columns were produced at all"

    empty = [c for c in form if built[c].isna().all()]
    assert not empty, f"{len(empty)} form feature(s) entirely NaN, e.g. {empty[:5]}"


def test_stacking_a_whole_season_on_one_date_is_what_broke_it(teams):
    """Pin the failure mode itself, so the reason for this module is legible."""
    history = _history(list(teams))
    fixtures = _all_pairs(teams).assign(
        Date=history["Date"].max() + pd.Timedelta(days=7)
    )
    stacked = build_upcoming_features(history, fixtures)

    away_form = [
        c for c in stacked.columns if c.startswith("away_") and "_last_5" in c
    ]
    assert any(stacked[c].isna().all() for c in away_form), (
        "the old failure mode no longer reproduces; if build_upcoming_features "
        "has been fixed to handle repeated teams, this test should go"
    )


def test_every_fixture_gets_a_row(teams):
    fixtures = _all_pairs(teams)
    built = build_frozen_form_features(_history(list(teams)), fixtures)
    assert len(built) == len(fixtures)
    assert set(zip(built["HomeTeam"], built["AwayTeam"], strict=True)) == set(
        zip(fixtures["HomeTeam"], fixtures["AwayTeam"], strict=True)
    )


def test_a_team_carries_the_same_form_into_every_fixture(teams):
    """Form is frozen by design, so one team's numbers must not vary by opponent.

    This is also what makes building it once per team correct rather than a
    shortcut.
    """
    built = build_frozen_form_features(_history(list(teams)), _all_pairs(teams))
    column = "home_points_last_5"
    spread = built.groupby("HomeTeam")[column].nunique(dropna=False)
    assert (spread == 1).all(), "a team's frozen form changed between fixtures"


def test_differences_are_recomputed_from_the_assembled_sides(teams):
    """`_diff` columns pair a home column with its away twin.

    They are only meaningful once the two blocks sit side by side, so assembling
    rows without rebuilding them would leave stale or empty differences.
    """
    built = build_frozen_form_features(_history(list(teams)), _all_pairs(teams))
    diffs = [c for c in built.columns if c.endswith("_diff")]
    assert diffs

    stem = "points_last_5"
    if f"{stem}_diff" in built.columns:
        expected = built[f"home_{stem}"] - built[f"away_{stem}"]
        pd.testing.assert_series_equal(
            built[f"{stem}_diff"], expected, check_names=False
        )


def test_no_fixtures_returns_empty(teams):
    empty = build_frozen_form_features(
        _history(list(teams)), pd.DataFrame(columns=["League", "Season", "HomeTeam", "AwayTeam"])
    )
    assert empty.empty
