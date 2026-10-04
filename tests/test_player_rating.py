"""Tests for the player rating.

These pin the properties that make a rating defensible -- that low-minute
players are pulled toward the mean, that unratable groups stay unrated, that
the scale is monotonic -- rather than asserting particular players get
particular numbers, which would just encode today's weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from footballml.players.ingest import position_group
from footballml.players.rating import (
    UNRATED_GROUPS,
    load_config,
    percentile_within,
    rate_players,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("GK", "GK"),
        ("GK S", "GK"),
        ("D", "D"),
        ("D S", "D"),
        ("F M S", "F"),      # most forwards are listed this way
        ("D M S", "D"),
        ("M S", "M"),
        ("S", None),         # substitute only, no role recorded
        (None, None),
        (float("nan"), None),
    ],
)
def test_position_group(raw: str | float | None, expected: str | None) -> None:
    assert position_group(raw) == expected


def test_forwards_are_not_resolved_to_midfield() -> None:
    """``"F M S"`` must resolve to F.

    Understat lists most forwards as having also played midfield. Resolving
    those to M would compare strikers against midfielders on goalscoring and
    inflate them badly.
    """
    assert position_group("F M S") == "F"
    assert position_group("F M") == "F"


def _squad(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """A synthetic league of forwards with varying quality and minutes."""
    rng = np.random.default_rng(seed)
    quality = rng.uniform(0.1, 1.0, n)
    minutes = rng.uniform(500, 3000, n)
    return pd.DataFrame(
        {
            "League": "E0",
            "Season": "2526",
            "Player": [f"P{i}" for i in range(n)],
            "Team": [f"T{i % 20}" for i in range(n)],
            "position_group": "F",
            "minutes": minutes,
            "nineties": minutes / 90,
            "np_xg_per90": quality * 0.8,
            "np_goals_per90": quality * 0.8,
            "finishing_delta_per90": rng.normal(0, 0.05, n),
            "xa_per90": quality * 0.3,
            "key_passes_per90": quality * 2.0,
            "assists_per90": quality * 0.25,
            "shots_per90": quality * 3.0,
            "xg_chain_per90": quality * 0.9,
            "xg_buildup_per90": quality * 0.4,
            "xg_chain_share": quality * 0.15,
            "xg_buildup_share": quality * 0.1,
            "yellow_cards_per90": rng.uniform(0, 0.3, n),
            "red_cards_per90": 0.0,
        }
    )


def test_ratings_land_on_the_expected_scale() -> None:
    rated = rate_players(_squad())
    ratings = rated.loc[rated["rated"], "rating"].astype(float)

    assert ratings.between(40, 99).all(), "ratings must sit on a 0-99 style scale"
    assert ratings.max() < 99, "nobody should hit the ceiling"
    assert 60 < ratings.median() < 75, "median should sit where football ratings do"


def test_rating_tracks_quality() -> None:
    """Better underlying numbers must produce a better rating."""
    squad = _squad()
    rated = rate_players(squad)
    ok = rated["rated"]

    corr = np.corrcoef(rated.loc[ok, "np_xg_per90"], rated.loc[ok, "rating"].astype(float))[0, 1]
    assert corr > 0.8, f"rating barely tracks performance (r={corr:.2f})"


def test_low_minutes_are_shrunk_toward_the_mean() -> None:
    """A brilliant cameo must not out-rate a brilliant season.

    Two players with identical per-90 output, one with 500 minutes and one with
    3000, must not be rated equally -- otherwise the leaderboard fills with
    players who had one good afternoon.
    """
    squad = _squad()
    elite = {c: squad[c].max() for c in squad.columns if c.endswith(("_per90", "_share"))}

    for name, minutes in (("Cameo", 500.0), ("Regular", 3000.0)):
        row = {**squad.iloc[0].to_dict(), **elite, "Player": name, "minutes": minutes,
               "nineties": minutes / 90, "yellow_cards_per90": 0.0, "red_cards_per90": 0.0}
        squad = pd.concat([squad, pd.DataFrame([row])], ignore_index=True)

    rated = rate_players(squad).set_index("Player")
    assert rated.loc["Regular", "rating"] > rated.loc["Cameo", "rating"]


def test_goalkeepers_are_rated_from_keeper_metrics() -> None:
    """Keepers are rated on FBref shot-stopping, not invented from thin air.

    Understat alone had nothing for them; once save percentage and goals
    against are present, a keeper pool large enough to percentile gets rated
    like any other group.
    """
    rng = np.random.default_rng(3)
    n = 60
    skill = rng.uniform(0.1, 1.0, n)
    keepers = pd.DataFrame(
        {
            "League": "E0", "Season": "2526",
            "Player": [f"GK{i}" for i in range(n)],
            "Team": [f"T{i}" for i in range(n)],
            "position_group": "GK",
            "minutes": rng.uniform(900, 3400, n),
            "save_pct": 55 + skill * 25,
            "goals_against_per90": 2.2 - skill * 1.4,
            "clean_sheet_pct": skill * 45,
            "shots_on_target_against_per90": rng.uniform(2.5, 5.5, n),
            "penalties_saved_per90": rng.uniform(0, 0.05, n),
        }
    )
    keepers["nineties"] = keepers["minutes"] / 90

    rated = rate_players(keepers)
    assert rated["rated"].all()

    ok = rated["rated"]
    corr = np.corrcoef(rated.loc[ok, "save_pct"], rated.loc[ok, "rating"].astype(float))[0, 1]
    assert corr > 0.7, f"keeper rating should track save percentage (r={corr:.2f})"


def test_unrated_groups_mechanism_still_works() -> None:
    """The refusal mechanism survives even though no group currently uses it."""
    assert isinstance(UNRATED_GROUPS, dict)


def test_negative_weights_invert_the_percentile() -> None:
    """A negative weight means lower is better, not 'subtract from the total'.

    Conceding goals and committing fouls are expressed that way, and getting it
    wrong would corrupt the normaliser and let a sub-rating go negative.
    """
    from footballml.players.rating import _weighted

    frame = pd.DataFrame({"good": [0.9, 0.1], "bad": [0.9, 0.1]})
    positive = _weighted(frame, {"good": 1.0})
    inverted = _weighted(frame, {"bad": -1.0})

    assert positive.iloc[0] == pytest.approx(0.9)
    assert inverted.iloc[0] == pytest.approx(0.1), "high value on a negative metric must score low"
    assert (inverted >= 0).all() and (inverted <= 1).all()


def test_short_seasons_are_left_unrated() -> None:
    """A percentile over a handful of players is meaningless and must be refused.

    This is what stopped a player with exactly 450 minutes in a five-matchweek
    season being rated 94.
    """
    tiny = _squad(n=12)
    rated = rate_players(tiny)

    assert not rated["rated"].any()
    assert "comparable players" in rated["unrated_reason"].iloc[0]


def test_under_minimum_minutes_unrated() -> None:
    squad = _squad()
    squad.loc[:4, "minutes"] = 100
    squad.loc[:4, "nineties"] = 100 / 90

    rated = rate_players(squad)
    assert not rated.loc[:4, "rated"].any()
    assert "minutes played" in rated.loc[0, "unrated_reason"]


def test_percentile_within_is_group_local() -> None:
    """Percentiles must be computed inside a group, never across groups."""
    df = pd.DataFrame(
        {
            "position_group": ["F"] * 3 + ["D"] * 3,
            "Season": ["2526"] * 6,
            "metric": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    pct = percentile_within(df, "metric", ["position_group", "Season"])

    # The best forward and the best defender both sit at the top of their group.
    assert pct.iloc[2] == pytest.approx(1.0)
    assert pct.iloc[5] == pytest.approx(1.0)


def test_percentile_direction_can_invert() -> None:
    """Cards are bad, so a high count must give a low percentile."""
    df = pd.DataFrame(
        {"position_group": ["F"] * 3, "Season": ["2526"] * 3, "cards": [0.0, 0.5, 1.0]}
    )
    pct = percentile_within(df, "cards", ["position_group", "Season"], higher_is_better=False)
    assert pct.iloc[0] > pct.iloc[2]


def test_unmatched_players_are_shrunk_but_matched_ones_are_not() -> None:
    """A player with no EA entry has less evidence, so his rating is less extreme.

    Left unblended, unmatched players kept untempered percentiles and crowded
    the tails. They are pulled toward the role mean; matched players are
    blended exactly as before.
    """
    squad = _squad()
    squad["fifa_overall"] = np.where(np.arange(len(squad)) % 2 == 0, 75.0, np.nan)
    squad.loc[squad.index % 4 == 0, "fifa_overall"] = 65.0

    config = load_config()
    shrunk = rate_players(squad, {**config, "unmatched_shrinkage_nineties": 4})
    raw = rate_players(squad, {**config, "unmatched_shrinkage_nineties": 0})

    matched = squad["fifa_overall"].notna()
    assert (shrunk.loc[matched, "rating"] == raw.loc[matched, "rating"]).all()

    missing = ~matched
    mean = raw.loc[missing, "performance_rating"].astype(float).mean()
    distance = lambda frame: (frame.loc[missing, "rating"].astype(float) - mean).abs()  # noqa: E731
    assert distance(shrunk).max() < distance(raw).max()
    assert raw.loc[missing, "rating"].equals(
        raw.loc[missing, "performance_rating"].round().astype("Int64")
    )


def test_absent_metrics_are_reported_not_silently_skipped(caplog):
    """A configured metric with no data must say so.

    `recoveries_per90` and `recoveries_padj` were weighted 1.5 each in the
    centre-back defending score but never built, so a third of that weight
    vanished without a word. Skipping is correct; skipping quietly is not.
    """
    import logging

    import pandas as pd

    from footballml.players import rating

    rating._WARNED.clear()
    frame = pd.DataFrame({"present_metric": [0.2, 0.8]})
    with caplog.at_level(logging.WARNING, logger=rating.logger.name):
        rating._weighted(frame, {"present_metric": 1.0, "absent_metric": 3.0})

    assert "absent_metric" in caplog.text
    assert "75%" in caplog.text, "the share of lost weight should be reported"


# --- a sub-rating built from data that is not there -------------------------
#
# All three of these were live. FBref serves no `misc` table at all for
# 2014/15 and no `keeper` table for 2015/16, and only La Liga's `misc` for
# 2015/16. None of it was visible in the output: every 2014/15 centre-back
# came back with a defending score of ~50 -- the spread across the pool was
# 2.3 points against ~20 in every other season -- because the metrics were
# absent, filled with 0.5, and defending is 62% of a centre-back's composite.


def _defenders(n: int = 120, seasons: tuple[str, ...] = ("2425",), seed: int = 0) -> pd.DataFrame:
    """Centre-backs across one or more seasons, with real defensive metrics."""
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        quality = rng.uniform(0.1, 1.0, n)
        minutes = rng.uniform(900, 3200, n)
        frames.append(
            pd.DataFrame(
                {
                    "League": "E0", "Season": season,
                    "Player": [f"{season}-P{i}" for i in range(n)],
                    "Team": [f"T{i % 20}" for i in range(n)],
                    "position_group": "D", "role": "CB",
                    "minutes": minutes, "nineties": minutes / 90,
                    "interceptions_padj": quality * 0.3,
                    "tackles_won_padj": quality * 0.2,
                    "fouls_per90": rng.uniform(0.1, 1.5, n),
                    "xg_buildup_share": quality * 0.1,
                    "xg_buildup_per90": quality * 0.4,
                    "xg_chain_share": quality * 0.1,
                    "xg_chain_per90": quality * 0.3,
                    "xa_per90": quality * 0.05,
                    "key_passes_per90": quality * 0.4,
                    "np_xg_per90": quality * 0.08,
                    "np_goals_per90": quality * 0.07,
                    "yellow_cards_per90": rng.uniform(0, 0.3, n),
                    "red_cards_per90": 0.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_a_metric_covering_a_fraction_of_the_pool_is_not_a_percentile() -> None:
    """2015/16, where only La Liga's `misc` table came through.

    Interceptions reached 17% of the centre-back pool. The other 83% were
    filled with 0.5 and sat on the median while the Spanish players spread out
    around them -- a ranking of one league, presented as a ranking of five.
    """
    squad = _defenders(n = 120)
    thin = squad.index >= 20  # 17% coverage, as 2015/16 had
    squad.loc[thin, ["interceptions_padj", "tackles_won_padj"]] = np.nan

    rated = rate_players(squad)
    assert rated.loc[rated["rated"], "sub_defending"].isna().all(), (
        "defending was computed from a metric covering a sixth of the pool"
    )


def test_most_of_a_sub_ratings_weight_missing_refuses_the_sub_rating() -> None:
    """What is left has to be most of what was asked for.

    With no interceptions or tackles, centre-back defending keeps only
    `fouls_per90` -- 8% of the weight, and a *negative* one. Renormalised, it
    ranks defenders by who fouls least and calls the answer defending.
    """
    squad = _defenders()
    squad[["interceptions_padj", "tackles_won_padj"]] = np.nan

    rated = rate_players(squad)
    assert rated.loc[rated["rated"], "sub_defending"].isna().all()
    # Blank, not a plausible 50: a reader can see the number is absent.
    assert rated["sub_defending"].notna().sum() == 0


def test_a_refused_sub_rating_renormalises_rather_than_dropping_the_player() -> None:
    """Defending is 62% of a centre-back, but EA's half of his rating is still
    there and is the better half for a defender anyway. Refusing twice would
    cost 2014/15 its entire back line."""
    squad = _defenders()
    squad[["interceptions_padj", "tackles_won_padj"]] = np.nan

    rated = rate_players(squad)
    assert rated["rated"].all(), "a missing sub-rating must not unrate the pool"
    assert rated.loc[rated["rated"], "rating"].notna().all()


_NO_PERFORMANCE = (
    "interceptions_padj", "tackles_won_padj", "fouls_per90",
    "xg_buildup_share", "xg_buildup_per90", "xg_chain_share", "xg_chain_per90",
    "xa_per90", "key_passes_per90", "np_xg_per90", "np_goals_per90",
)


def test_a_pool_with_no_usable_data_and_no_ea_entry_is_unrated() -> None:
    """Every 2015/16 goalkeeper: FBref served no `keeper` table, so all four
    sub-ratings were empty. They came back `rated` with a null rating, which
    is the one outcome `unrated_reason` exists to prevent."""
    squad = _defenders()
    for column in _NO_PERFORMANCE:
        squad[column] = np.nan

    rated = rate_players(squad)
    assert not rated["rated"].any()
    assert rated["unrated_reason"].eq("No performance data this season").all()


def test_no_performance_data_still_keeps_eas_half_of_the_rating() -> None:
    """Refusing outright would throw away a real number to avoid publishing a
    missing one -- and take every 2015/16 team's goalkeeper line with it.

    EA's overall is present for ~95% of them, so it becomes the whole rating
    instead of half of it, and `performance_rating` stays blank to say so.
    """
    squad = _defenders()
    for column in _NO_PERFORMANCE:
        squad[column] = np.nan
    squad["fifa_overall"] = np.linspace(60, 90, len(squad))

    rated = rate_players(squad)
    assert rated["rated"].all()
    assert rated["rating"].notna().all()
    assert rated["performance_rating"].isna().all(), (
        "a performance rating must not be invented from no performance data"
    )
    # EA's ordering survives: the best-rated EA player is the best-rated here.
    best = rated.loc[rated["fifa_overall"].idxmax(), "rating"]
    assert best == rated["rating"].max()


def test_nullable_columns_from_the_ingest_do_not_break_the_build() -> None:
    """The ingest hands over pandas' nullable `Float64`, not plain float.

    A group with no usable sub-ratings gets a float64 column of NaN, which the
    discipline blend then multiplies against a `Float64` card score -- and the
    result is a `Float64` block of pd.NA that pandas refuses to write into the
    plain float64 series collecting the composites. The whole build dies with
    a dtype error after the scrape, which is twenty minutes in.

    Reading the same frame back from CSV flattens every dtype, so a dry run
    against the built file passes while the real pipeline crashes. Only a
    fixture carrying the ingest's own dtypes can catch it.
    """
    squad = _defenders(seasons=("1415", "2425"))
    numeric = squad.select_dtypes("number").columns
    squad[numeric] = squad[numeric].astype("Float64")
    # 2014/15: no performance data at all, but cards -- the discipline score
    # is the one thing still computable, and it is what upcasts the composite.
    early = squad["Season"].eq("1415")
    squad.loc[early, list(_NO_PERFORMANCE)] = pd.NA
    squad["fifa_overall"] = pd.array(
        np.linspace(60, 90, len(squad)), dtype="Float64"
    )

    rated = rate_players(squad)
    assert rated["rated"].all()
    assert rated.loc[rated["Season"].eq("1415"), "performance_rating"].isna().all()
    assert rated.loc[rated["Season"].eq("2425"), "sub_defending"].notna().all()


def test_one_season_missing_a_metric_does_not_borrow_another_seasons() -> None:
    """Sub-ratings are built per role *and* season.

    Computed over a whole role at once, a metric present in 2024/25 counts as
    present for 2014/15 too, and the earlier season falls through to the
    fill-with-0.5 path instead of being refused. That is exactly how two
    seasons of invented defending scores shipped.
    """
    squad = _defenders(seasons=("1415", "2425"))
    early = squad["Season"].eq("1415")
    squad.loc[early, ["interceptions_padj", "tackles_won_padj"]] = np.nan

    rated = rate_players(squad)
    done = rated[rated["rated"]]
    assert done.loc[done["Season"].eq("1415"), "sub_defending"].isna().all()
    assert done.loc[done["Season"].eq("2425"), "sub_defending"].notna().all()


# --- rewarding a player's best attribute ------------------------------------
#
# A weighted mean of percentiles ranks a tidy all-rounder above a player who is
# 97th percentile at the thing his position exists for. Mane's 22-goal 2018/19
# rated below his 11-goal 2020/21, because winger weights put creation above
# finishing and he had one assist.


def test_a_specialist_gains_on_an_all_rounder() -> None:
    """The property the peak term exists for, stated as an ordering.

    Two forwards with the same weighted mean: one outstanding at finishing and
    poor at creating, one middling at both. Without the peak term they score
    the same; with it the specialist is ahead.
    """
    squad = _squad(n=120, seed=3)
    # Give two players deliberately constructed profiles at the same mean.
    squad.loc[0, ["np_xg_per90", "np_goals_per90", "shots_per90"]] = [1.4, 1.4, 6.0]
    squad.loc[0, ["xa_per90", "key_passes_per90", "assists_per90"]] = [0.02, 0.2, 0.02]
    squad.loc[1, ["np_xg_per90", "np_goals_per90", "shots_per90"]] = [0.45, 0.45, 2.4]
    squad.loc[1, ["xa_per90", "key_passes_per90", "assists_per90"]] = [0.18, 1.3, 0.14]
    squad.loc[[0, 1], "minutes"] = 3000
    squad.loc[[0, 1], "nineties"] = 3000 / 90

    config = load_config()
    flat = dict(config, peak_weight=0.0)
    peaked = dict(config, peak_weight=0.30)

    def gap(cfg: dict) -> float:
        rated = rate_players(squad.copy(), cfg).set_index("Player")
        return float(rated.loc["P0", "composite_raw"] - rated.loc["P1", "composite_raw"])

    assert gap(peaked) > gap(flat), (
        "the peak term did not move the specialist toward the all-rounder"
    )


def test_the_peak_term_can_be_switched_off() -> None:
    """`peak_weight: 0` has to reproduce the plain weighted mean exactly, or
    the sweep behind the chosen value cannot be re-run."""
    squad = _squad(seed=4)
    config = load_config()
    off = rate_players(squad.copy(), dict(config, peak_weight=0.0))
    assert off["composite_raw"].notna().any()
    # Recomputing the weighted mean by hand is the point: no peak, no change.
    again = rate_players(squad.copy(), {k: v for k, v in config.items() if k != "peak_weight"})
    pd.testing.assert_series_equal(
        off["composite_raw"], again["composite_raw"], check_names=False
    )


def test_a_player_with_no_sub_ratings_is_not_pulled_toward_nothing() -> None:
    """`max` of an all-empty row is NaN, and blending that in would unrate a
    player the rest of the pipeline went to some trouble to keep."""
    squad = _defenders()
    for column in _NO_PERFORMANCE:
        squad[column] = np.nan
    squad["fifa_overall"] = np.linspace(60, 90, len(squad))

    rated = rate_players(squad, dict(load_config(), peak_weight=0.30))
    assert rated["rated"].all()
    assert rated["rating"].notna().all()


# --- quality per defensive action, not volume of it -------------------------
#
# `defending` asked how *much* defending a player did, which a centre back at a
# dominant side does little of: Van Dijk sat at the 3rd to 11th percentile for
# defensive-action volume in every Liverpool season. Across 5,177 rated
# defender-seasons the score correlated -0.0023 with whether his team actually
# prevented chances -- it was not a crude measure of defending, it was
# uninformative about it.


def test_a_clean_tackler_outranks_a_busy_fouler() -> None:
    """The property the ratios exist for.

    Two defenders winning the ball equally often, one giving away three times
    the fouls doing it. On volume alone they are identical.
    """
    squad = _defenders(n=120, seed=7)
    squad.loc[0, ["tackles_won_per90", "interceptions_per90", "fouls_per90"]] = [1.0, 1.0, 0.3]
    squad.loc[1, ["tackles_won_per90", "interceptions_per90", "fouls_per90"]] = [1.0, 1.0, 1.5]
    for i in (0, 1):
        squad.loc[i, "clean_challenge_rate"] = (
            squad.loc[i, "tackles_won_per90"]
            / (squad.loc[i, "tackles_won_per90"] + squad.loc[i, "fouls_per90"])
        )
        squad.loc[i, "ball_won_per_foul"] = (
            (squad.loc[i, "tackles_won_per90"] + squad.loc[i, "interceptions_per90"])
            / squad.loc[i, "fouls_per90"]
        )
    squad.loc[[0, 1], ["interceptions_padj", "tackles_won_padj"]] = [0.15, 0.10]

    config = load_config()
    rated = rate_players(squad, config).set_index("Player")
    assert rated.loc["2425-P0", "sub_defending"] > rated.loc["2425-P1", "sub_defending"], (
        "the cleaner defender did not rank above the one who fouls five times as often"
    )


def test_the_quality_ratios_are_optional() -> None:
    """A season without them -- 2014/15 has no FBref misc table at all -- must
    still produce a defending score from what is there."""
    import copy

    squad = _defenders(n=120, seed=8)
    config = copy.deepcopy(load_config())
    rated = rate_players(squad, config)
    done = rated[rated["rated"]]
    assert done["sub_defending"].notna().any(), (
        "defending collapsed when the quality ratios were absent from the frame"
    )
