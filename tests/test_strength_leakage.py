"""Tests for the team-strength join.

This leak is invisible to ``test_leakage.py::test_truncation_invariance``. That
test removes future *matches* and checks the features do not move; team strength
arrives from a separate file, so truncating matches would not change it at all
and the test would pass on thoroughly leaky data.

The guard therefore has to be asserted directly: a match must never see strength
computed from its own season or any later one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from footballml.features.build import _next_season, add_team_strength


@pytest.fixture
def strength() -> pd.DataFrame:
    """Two seasons of strength, with values that identify their season."""
    rows = []
    for season, base in (("2425", 70.0), ("2526", 80.0)):
        for team in ("Arsenal", "Chelsea"):
            rows.append(
                {
                    "League": "E0",
                    "Season": season,
                    "Team": team,
                    "strength_overall": base,
                    "strength_attack": base + 1,
                    "strength_defence": base + 2,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "League": ["E0", "E0"],
            "Season": ["2526", "2425"],
            "Date": pd.to_datetime(["2025-10-01", "2024-10-01"]),
            "HomeTeam": ["Arsenal", "Arsenal"],
            "AwayTeam": ["Chelsea", "Chelsea"],
        }
    )


@pytest.mark.parametrize(
    ("label", "expected"),
    [("2425", "2526"), ("2526", "2627"), ("1920", "2021"), ("2021", "2122")],
)
def test_next_season(label: str, expected: str) -> None:
    """Season labels roll over correctly, including across the century mark."""
    assert _next_season(label) == expected


def test_match_never_sees_its_own_season(matches: pd.DataFrame, strength: pd.DataFrame) -> None:
    """The defining guard: a 2025/26 match must use 2024/25 strength.

    Values are seeded so the season is recoverable from the number -- 70 means
    2024/25, 80 means 2025/26. A 2025/26 match showing 80 would mean
    end-of-season squad data was used to predict an October fixture.
    """
    joined = add_team_strength(matches, strength, previous_season=True)

    current = joined[joined["Season"] == "2526"].iloc[0]
    assert current["home_strength_overall"] == 70.0, (
        "a 2025/26 match must see 2024/25 strength, not its own season's"
    )
    assert current["away_strength_overall"] == 70.0


def test_earliest_season_gets_no_strength(matches: pd.DataFrame, strength: pd.DataFrame) -> None:
    """With nothing before it, the first season must be NaN rather than its own.

    Falling back to the match's own season here would be the leak arriving by a
    side door, and would look like the feature simply had better coverage.
    """
    joined = add_team_strength(matches, strength, previous_season=True)
    earliest = joined[joined["Season"] == "2425"].iloc[0]
    assert pd.isna(earliest["home_strength_overall"])


def test_own_season_mode_uses_the_current_season(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """``previous_season=False`` joins a match to its own season.

    Kept for analysis only. Live prediction does *not* use it: current-season
    strength rarely exists early in a season, and the model was trained on the
    previous season's. See :func:`test_upcoming_fixture_gets_previous_season_strength`.
    """
    joined = add_team_strength(matches, strength, previous_season=False)
    current = joined[joined["Season"] == "2526"].iloc[0]
    assert current["home_strength_overall"] == 80.0


def test_attack_and_defence_are_carried_separately(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """Attack and defence must stay distinct -- they feed opposite goal rates."""
    joined = add_team_strength(matches, strength, previous_season=True)
    row = joined[joined["Season"] == "2526"].iloc[0]

    assert row["home_strength_attack"] == 71.0
    assert row["home_strength_defence"] == 72.0
    assert row["away_strength_attack"] == 71.0


def test_unknown_team_gets_nan_not_a_wrong_row(strength: pd.DataFrame) -> None:
    """A promoted side with no strength row must come back empty, not mismatched."""
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Luton"],
            "AwayTeam": ["Chelsea"],
        }
    )
    joined = add_team_strength(matches, strength, previous_season=True)
    assert pd.isna(joined["home_strength_overall"].iloc[0])
    assert joined["away_strength_overall"].iloc[0] == 70.0


def test_missing_strength_file_is_survivable() -> None:
    """An empty strength frame must leave the matches untouched, not raise.

    The match pipeline has to run before `build_players` ever has.
    """
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
        }
    )
    joined = add_team_strength(matches, pd.DataFrame(), previous_season=True)
    assert len(joined) == 1
    assert "home_strength_overall" not in joined.columns


def test_join_does_not_duplicate_matches(matches: pd.DataFrame, strength: pd.DataFrame) -> None:
    """A many-to-one join that silently fans out would corrupt every metric."""
    assert len(add_team_strength(matches, strength, previous_season=True)) == len(matches)


# --- The serving path ------------------------------------------------------
#
# Everything above exercises the join directly. The site never calls it
# directly: it goes through `build_upcoming_features`, which appends placeholder
# rows to real history and re-runs the builder. A strength kwarg dropped on that
# path would leave every live prediction without squad quality -- which is
# precisely how the deployed model came to ignore strength while the backtest
# used it.


def _previous(label: str) -> str:
    """``"2627"`` -> ``"2526"``."""
    start = int(label[:2])
    return f"{(start - 1) % 100:02d}{start:02d}"


@pytest.fixture(scope="module")
def recent_history() -> pd.DataFrame:
    """Real match history, trimmed to one league's latest season for speed.

    Reads the five-league file the API serves from; the default single-league
    file carries no ``League`` column, which the strength join keys on.
    """
    from footballml.data import PROCESSED_DIR, load_team_match_history

    path = PROCESSED_DIR / "team_match_history_all.csv"
    if not path.exists():
        pytest.skip("run `python -m pipelines.build_dataset` first")
    df = load_team_match_history(path)
    df = df[df["League"] == "E0"]
    latest = df["Season"].max()
    return df[df["Season"] == latest].copy()


def test_upcoming_fixture_gets_previous_season_strength(
    recent_history: pd.DataFrame,
) -> None:
    """A fixture in season S must receive S-1 strength through the live path.

    Seeded so the season is readable from the value: 80 is last season, 99 is
    this one. 99 would mean the serving path joined the current season, which
    does not exist early on and is not what the model was trained with.
    """
    from footballml.features.build import build_upcoming_features

    season = str(recent_history["Season"].max())
    league = str(recent_history["League"].iloc[0])
    home, away = recent_history["Team"].drop_duplicates().iloc[:2].tolist()

    strength = pd.DataFrame(
        [
            {
                "League": league,
                "Season": s,
                "Team": t,
                "strength_overall": v,
                "strength_attack": v,
                "strength_defence": v,
            }
            for s, v in ((_previous(season), 80.0), (season, 99.0))
            for t in (home, away)
        ]
    )
    fixture = pd.DataFrame(
        [
            {
                "League": league,
                "Season": season,
                "Date": recent_history["Date"].max() + pd.Timedelta(days=7),
                "HomeTeam": home,
                "AwayTeam": away,
            }
        ]
    )

    built = build_upcoming_features(recent_history, fixture, strength=strength)

    assert len(built) == 1
    row = built.iloc[0]
    assert row["home_strength_overall"] == 80.0, "live path used the wrong season"
    assert row["away_strength_defence"] == 80.0
    assert row["strength_overall_diff"] == 0.0


def test_upcoming_fixture_without_strength_has_no_strength_columns(
    recent_history: pd.DataFrame,
) -> None:
    """The kwarg is what switches the feature on -- so omitting it is visible.

    This is the failure a trained-with-strength artifact would hit: the columns
    it expects are simply absent.
    """
    from footballml.features.build import build_upcoming_features

    home, away = recent_history["Team"].drop_duplicates().iloc[:2].tolist()
    fixture = pd.DataFrame(
        [
            {
                "League": str(recent_history["League"].iloc[0]),
                "Season": str(recent_history["Season"].max()),
                "Date": recent_history["Date"].max() + pd.Timedelta(days=7),
                "HomeTeam": home,
                "AwayTeam": away,
            }
        ]
    )
    built = build_upcoming_features(recent_history, fixture)
    assert not any("strength" in c for c in built.columns)


def test_understat_club_names_reach_football_data_fixtures(
    strength: pd.DataFrame,
) -> None:
    """Strength rows named by Understat must join to football-data fixtures.

    Player data -- and so strength -- carries "Borussia Dortmund"; matches carry
    "Dortmund". A left merge on the raw names does not fail, it just leaves NaN,
    which is how the feature silently reached only 40% of Bundesliga
    team-seasons while every test above passed.
    """
    renamed = strength.replace(
        {"Team": {"Arsenal": "Borussia Dortmund", "Chelsea": "Atletico Madrid"}}
    )
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Dortmund"],
            "AwayTeam": ["Ath Madrid"],
        }
    )
    joined = add_team_strength(matches, renamed, previous_season=True)
    assert joined["home_strength_overall"].iloc[0] == 70.0
    assert joined["away_strength_overall"].iloc[0] == 70.0


def test_already_played_fixtures_are_skipped(recent_history: pd.DataFrame) -> None:
    """A fixture already in the history must not be appended as a placeholder.

    football-data keeps publishing a round after it is played, so once results
    are refreshed every "upcoming" fixture can already exist. Appending one then
    duplicates a real match and the 1:1 merge inside `build_match_features`
    fails with a MergeError pointing nowhere near the cause.
    """
    from footballml.features.build import build_upcoming_features

    home = recent_history[recent_history["Venue"] == "Home"].iloc[0]
    fixture = pd.DataFrame(
        [
            {
                "League": home["League"],
                "Season": str(home["Season"]),
                "Date": home["Date"],
                "HomeTeam": home["Team"],
                "AwayTeam": home["Opponent"],
            }
        ]
    )
    assert build_upcoming_features(recent_history, fixture).empty


def test_unplayed_fixtures_still_build(recent_history: pd.DataFrame) -> None:
    """The skip must not swallow genuine fixtures."""
    from footballml.features.build import build_upcoming_features

    home, away = recent_history["Team"].drop_duplicates().iloc[:2].tolist()
    fixture = pd.DataFrame(
        [
            {
                "League": str(recent_history["League"].iloc[0]),
                "Season": str(recent_history["Season"].max()),
                "Date": recent_history["Date"].max() + pd.Timedelta(days=14),
                "HomeTeam": home,
                "AwayTeam": away,
            }
        ]
    )
    assert len(build_upcoming_features(recent_history, fixture)) == 1


# --- The pre-season fallback ----------------------------------------------
#
# `add_team_strength(preseason=...)` fills the sides with no measured rating --
# 14.1% of team-sides, almost all promoted or yo-yo clubs -- from the match's
# **own** season's pre-season rating. That is the one place in this module where
# a match reads its own season on purpose, so it is also the highest-risk line
# in the feature build: a leak here would inflate the backtest and look exactly
# like the change working.
#
# The guard has two halves, and both are asserted below.
#
# 1. The *join* must take the match's own season, not a shifted one, and must
#    never displace a measured value.
# 2. The *table* must contain nothing from the season it is labelled with. That
#    is `squad_rating._latest_before`, strictly-before by default, and the
#    EA-overall curve, which is a rank of EA's own numbers within
#    (role, season) and reads no results at all.


@pytest.fixture
def preseason() -> pd.DataFrame:
    """Pre-season ratings whose season is readable from the value.

    50 means 2024/25, 60 means 2025/26. Luton has no measured row in either
    season, so it is the side the fallback is for; Arsenal and Chelsea have
    both, so they are the sides the fallback must leave alone.
    """
    rows = []
    for season, base in (("2425", 50.0), ("2526", 60.0)):
        for team in ("Arsenal", "Chelsea", "Luton"):
            rows.append(
                {
                    "League": "E0",
                    "Season": season,
                    "Team": team,
                    "squad_overall": base,
                    "squad_attack": base + 1,
                    "squad_defence": base + 2,
                }
            )
    return pd.DataFrame(rows)


def _luton_home(season: str) -> pd.DataFrame:
    """One match in `season`, with a side that is in no measured table."""
    return pd.DataFrame(
        {
            "League": ["E0"],
            "Season": [season],
            "Date": pd.to_datetime([f"20{season[2:]}-10-01"]),
            "HomeTeam": ["Luton"],
            "AwayTeam": ["Chelsea"],
        }
    )


def test_fallback_uses_the_match_own_season_not_the_previous_one(
    strength: pd.DataFrame, preseason: pd.DataFrame
) -> None:
    """The defining guard for the fallback, and it points the *other* way.

    A 2025/26 match must take the 2025/26 pre-season rating -- 60, not 50. That
    is the whole point: the measured rating is shifted back a season because it
    is computed from a season of results, and the pre-season rating is not
    shifted because it is computed from a roster plus ratings from strictly
    before. Shifting it too would hand a promoted side the roster it had in the
    division below, which is the measured rating again by a longer route.
    """
    joined = add_team_strength(_luton_home("2526"), strength, preseason=preseason)
    assert joined["home_strength_overall"].iloc[0] == 60.0
    assert joined["home_strength_attack"].iloc[0] == 61.0
    assert joined["home_strength_defence"].iloc[0] == 62.0


def test_fallback_marks_the_side_it_filled(strength: pd.DataFrame, preseason: pd.DataFrame) -> None:
    """The flag is what stops the model blending two quantities blind."""
    joined = add_team_strength(_luton_home("2526"), strength, preseason=preseason)
    assert joined["home_strength_is_preseason"].iloc[0] == 1.0
    assert joined["away_strength_is_preseason"].iloc[0] == 0.0


def test_fallback_never_displaces_a_measured_rating(
    matches: pd.DataFrame, strength: pd.DataFrame, preseason: pd.DataFrame
) -> None:
    """Strictly additive: a side with a measured rating keeps it, untouched.

    Chelsea's 2024/25 measured rating is 70 and its 2025/26 pre-season rating
    is 60. A 2025/26 match must still see 70. If the fallback overwrote rather
    than filled, every side in the backtest would quietly be scored on a
    pre-season projection instead of on a season of football, and the headline
    RPS would be measuring a different change from the one described.
    """
    joined = add_team_strength(matches, strength, preseason=preseason)
    current = joined[joined["Season"] == "2526"].iloc[0]
    assert current["home_strength_overall"] == 70.0
    assert current["away_strength_overall"] == 70.0
    assert current["home_strength_is_preseason"] == 0.0


def test_fallback_does_fill_the_earliest_season_and_that_is_not_the_leak(
    matches: pd.DataFrame, strength: pd.DataFrame, preseason: pd.DataFrame
) -> None:
    """2024/25 has no 2023/24 measured rating, so the fallback does apply here.

    Worth stating plainly, because it reads like the leak
    `test_earliest_season_gets_no_strength` forbids and is not: the 2024/25
    match takes the 2024/25 **pre-season** rating, 50, which was knowable in
    September 2024. What it must never take is 2024/25's *measured* rating of
    70, which is computed from the season the match is in.
    """
    joined = add_team_strength(matches, strength, preseason=preseason)
    earliest = joined[joined["Season"] == "2425"].iloc[0]
    assert earliest["home_strength_overall"] == 50.0
    assert earliest["home_strength_is_preseason"] == 1.0


def test_fallback_leaves_a_measured_side_with_a_missing_keeper_alone(
    preseason: pd.DataFrame,
) -> None:
    """A whole missing row is the gap; a missing goalkeeper is a different one.

    1,113 team-sides have a measured rating whose ``strength_goalkeeper`` is
    NaN -- mostly 2015/16, where FBref served no `keeper` table. Patching that
    one column from EA would mix a measured attack and a projected keeper inside
    a single row, which is exactly what the flag exists to let the booster
    avoid, and it is not the gap this fallback was measured for.
    """
    strength = pd.DataFrame(
        [
            {
                "League": "E0",
                "Season": "2425",
                "Team": "Chelsea",
                "strength_overall": 70.0,
                "strength_attack": 71.0,
                "strength_goalkeeper": float("nan"),
            }
        ]
    )
    matches = _luton_home("2526").assign(HomeTeam="Chelsea", AwayTeam="Luton")
    joined = add_team_strength(matches, strength, preseason=preseason.assign(squad_goalkeeper=65.0))
    assert joined["home_strength_overall"].iloc[0] == 70.0
    assert pd.isna(joined["home_strength_goalkeeper"].iloc[0])
    assert joined["home_strength_is_preseason"].iloc[0] == 0.0


def test_fallback_publishes_the_flag_even_with_nothing_to_fill_from(
    matches: pd.DataFrame, strength: pd.DataFrame
) -> None:
    """Feature *names* must not depend on whether a file is on disk.

    The reload guard compares names, so a frame that grows two columns only
    when `squad_strength.csv` happens to exist is how a trained artifact comes
    to disagree with the frame it is served -- the quiet version of the failure
    this file's serving section was written for.
    """
    joined = add_team_strength(matches, strength, preseason=pd.DataFrame())
    assert joined["home_strength_is_preseason"].tolist() == [0.0, 0.0]
    assert joined["away_strength_is_preseason"].tolist() == [0.0, 0.0]


def test_fallback_does_not_duplicate_matches(
    matches: pd.DataFrame, strength: pd.DataFrame, preseason: pd.DataFrame
) -> None:
    """Two more left joins, two more chances to fan the frame out silently."""
    assert len(add_team_strength(matches, strength, preseason=preseason)) == len(matches)


def test_side_in_neither_table_stays_nan(strength: pd.DataFrame, preseason: pd.DataFrame) -> None:
    """A side in neither table keeps NaN and a false flag, exactly as before."""
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Ipswich"],
            "AwayTeam": ["Chelsea"],
        }
    )
    joined = add_team_strength(matches, strength, preseason=preseason)
    assert pd.isna(joined["home_strength_overall"].iloc[0])
    assert joined["home_strength_is_preseason"].iloc[0] == 0.0


def test_fallback_translates_club_names_like_the_measured_join(
    strength: pd.DataFrame, preseason: pd.DataFrame
) -> None:
    """The pre-season table is Understat-named too, so it needs the same aliases.

    Both tables come from the same player data. A second hand-rolled lookup
    that skipped `load_aliases` would reach Arsenal and not Dortmund, and a left
    merge does not fail -- it just leaves the NaN the fallback was added to
    remove, while every test above still passes.
    """
    renamed = preseason.replace({"Team": {"Luton": "Borussia Dortmund"}})
    matches = pd.DataFrame(
        {
            "League": ["E0"],
            "Season": ["2526"],
            "Date": pd.to_datetime(["2025-10-01"]),
            "HomeTeam": ["Dortmund"],
            "AwayTeam": ["Chelsea"],
        }
    )
    joined = add_team_strength(matches, strength, preseason=renamed)
    assert joined["home_strength_overall"].iloc[0] == 60.0
    assert joined["home_strength_is_preseason"].iloc[0] == 1.0


# --- The other half: what is in the table ----------------------------------


def test_preseason_rating_reads_nothing_from_the_season_it_describes() -> None:
    """`_latest_before` is strictly-before, which is what makes the join safe.

    The fallback joins a season-S rating onto a season-S match, so this is the
    only thing standing between that and a leak. Seeded so the season is
    readable from the value: 80 is 2025/26's own rating and must not be
    reachable for a 2025/26 squad.
    """
    from footballml.players.squad_rating import _latest_before

    seasons = [("2425", 70.0, 900.0), ("2526", 80.0, 1200.0)]
    assert _latest_before(seasons, "2526") == (70.0, 900.0)
    assert _latest_before(seasons, "2425") is None

    # The unsafe variant exists only because the alternative had to be
    # measured. If it ever became the default, every assertion above would
    # still pass while each rating silently gained its own season.
    assert _latest_before(seasons, "2526", include_current=True) == (80.0, 1200.0)


def test_published_preseason_ratings_are_built_strictly_before() -> None:
    """The published table must take the safe branch, and take it by default.

    `pipelines.build_squad_strength` calls `squad_strength(players, best_n=...)`
    and passes nothing for `include_current`, so the default is what ships --
    and now what the model reads. A flipped default would leak into the model
    through the fallback without a single call site changing.
    """
    import inspect

    from footballml.players.squad_rating import rosters, squad_strength

    for function in (squad_strength, rosters):
        assert inspect.signature(function).parameters["include_current"].default is False


def test_preseason_ea_scale_curve_reads_no_results() -> None:
    """An unrated player is placed by EA's numbers alone, not by performance.

    A player with no season before this one is rated by mapping EA's `overall`
    through the published `fifa_on_our_scale` curve for his (role, season).
    That curve is the *rank of EA overalls within that pool*, so it is a
    function of EA's September judgement and nothing else -- which is the second
    half of why joining a season-S rating onto a season-S match carries no
    result from S.

    The `rating` column here is deliberately anti-correlated with the curve: if
    performance ever reached `_curves`, the mapping could not come back monotone
    in EA's overall.
    """
    import numpy as np

    from footballml.players.squad_rating import _curves

    players = pd.DataFrame(
        {
            "fifa_overall": [60.0, 70.0, 80.0, 90.0, 95.0],
            "fifa_on_our_scale": [20.0, 40.0, 60.0, 80.0, 90.0],
            "role": ["st"] * 5,
            "Season": ["2526"] * 5,
            "rating": [99.0, 10.0, 95.0, 12.0, 50.0],
        }
    )
    xs, ys = _curves(players)[("st", "2526")]
    assert xs.tolist() == [60.0, 70.0, 80.0, 90.0, 95.0]
    assert ys.tolist() == [20.0, 40.0, 60.0, 80.0, 90.0]
    assert np.all(np.diff(ys) > 0)


def test_upcoming_fixture_fallback_uses_the_fixture_own_season(
    recent_history: pd.DataFrame,
) -> None:
    """The serving path, which is the one that actually matters.

    Everything above exercises the join directly; the site goes through
    `build_upcoming_features`. A promoted side is precisely the club a visitor
    looks up, and it is the club that had no squad strength at all before this
    change -- so a `preseason_strength` kwarg dropped on this path would leave
    the feature working in the backtest and absent in production, which is how
    the deployed model came to ignore strength while the backtest used it.
    """
    from footballml.features.build import build_upcoming_features

    season = str(recent_history["Season"].max())
    league = str(recent_history["League"].iloc[0])
    home, away = recent_history["Team"].drop_duplicates().iloc[:2].tolist()

    # `home` is deliberately absent from the measured table: the promoted side.
    strength = pd.DataFrame(
        [
            {
                "League": league,
                "Season": _previous(season),
                "Team": away,
                "strength_overall": 80.0,
                "strength_attack": 80.0,
                "strength_defence": 80.0,
            }
        ]
    )
    preseason = pd.DataFrame(
        [
            {
                "League": league,
                "Season": s,
                "Team": t,
                "squad_overall": v,
                "squad_attack": v,
                "squad_defence": v,
            }
            for s, v in ((_previous(season), 50.0), (season, 60.0))
            for t in (home, away)
        ]
    )
    fixture = pd.DataFrame(
        [
            {
                "League": league,
                "Season": season,
                "Date": recent_history["Date"].max() + pd.Timedelta(days=7),
                "HomeTeam": home,
                "AwayTeam": away,
            }
        ]
    )

    built = build_upcoming_features(
        recent_history, fixture, strength=strength, preseason_strength=preseason
    )

    assert len(built) == 1
    row = built.iloc[0]
    assert row["home_strength_overall"] == 60.0, "serving path used the wrong season"
    assert row["home_strength_is_preseason"] == 1.0
    # The away side has a measured rating and must keep it.
    assert row["away_strength_overall"] == 80.0
    assert row["away_strength_is_preseason"] == 0.0
