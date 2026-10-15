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

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from footballml.players import ingest, rating
from footballml.players.fbref import _is_same_name, _tokens, match_players
from footballml.players.rating import MAX_MISSING_WEIGHT, _weighted, load_config
from pipelines import build_players
from pipelines import defending_sweep as sweep

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


# --- the club's own defensive record ------------------------------------------
#
# `sub_defending` correlates +0.10 with the goals a team concedes, across 859
# team-seasons: not weak, backwards. Every input counts defensive *actions*, and
# a defender makes more of those when his side defends worse. What the group
# never had is what happened at the other end -- which the goalkeeper has been
# rated on all along, at 97% of his composite, while the defenders in front of
# him are rated on it at 0%.

def _history(rows: list[tuple[str, str, str, list[int]]]) -> pd.DataFrame:
    """League, season, team and the goals conceded in each match."""
    out = []
    for league, season, team, conceded in rows:
        out.extend(
            {
                "League": league, "Season": season, "Team": team,
                "GoalsAgainst": ga, "xGAgainst": float(ga) + 0.1,
            }
            for ga in conceded
        )
    return pd.DataFrame(out)


def _players(rows: list[tuple[str, str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"League": lg, "Season": s, "Team": t, "Player": p} for lg, s, t, p in rows]
    )


def test_the_club_record_reaches_its_defenders() -> None:
    history = _history([("E0", "2526", "Arsenal", [0, 0, 1, 2])])
    out = ingest.add_team_defence(
        _players([("E0", "2526", "Arsenal", "William Saliba")]), history
    )
    row = out.iloc[0]
    assert row["team_ga_per_match"] == pytest.approx(0.75)
    assert row["team_clean_sheet_pct"] == pytest.approx(0.5)
    assert row["team_xga_per_match"] == pytest.approx(0.85)


def test_it_separates_a_good_defence_from_a_bad_one() -> None:
    """The whole point. These two were indistinguishable on `sub_defending`.

    Arsenal conceded 0.71 a game in 2025/26 and Werder Bremen 1.76, and their
    centre-backs rated 53.5 and 60.6 -- the wrong way round.
    """
    history = _history([
        ("E0", "2526", "Arsenal", [0, 0, 1, 1]),
        ("D1", "2526", "Werder Bremen", [2, 2, 3, 1]),
    ])
    out = ingest.add_team_defence(
        _players([
            ("E0", "2526", "Arsenal", "William Saliba"),
            ("D1", "2526", "Werder Bremen", "Karim Coulibaly"),
        ]),
        history,
    )
    arsenal, bremen = out.iloc[0], out.iloc[1]
    assert arsenal["team_ga_per_match"] < bremen["team_ga_per_match"]
    assert arsenal["team_clean_sheet_pct"] > bremen["team_clean_sheet_pct"]


def test_every_defender_at_a_club_gets_the_same_number() -> None:
    """Stated as a limitation, not hidden: this cannot separate teammates.

    It fixes the cross-club problem and not the within-club one, and it makes a
    player's rating depend partly on who he plays with.
    """
    history = _history([("E0", "2526", "Arsenal", [0, 1])])
    out = ingest.add_team_defence(
        _players([
            ("E0", "2526", "Arsenal", "William Saliba"),
            ("E0", "2526", "Arsenal", "Gabriel"),
        ]),
        history,
    )
    teammates = out["team_ga_per_match"]
    assert (teammates == teammates.iloc[0]).all()


def test_the_season_dtype_survives() -> None:
    """`run_backtest` filters seasons numerically.

    `add_team_strength` carries the same warning: casting `Season` to string in
    place breaks callers a long way downstream, with an error pointing nowhere
    near the function that did it.
    """
    players = _players([("E0", "2526", "Arsenal", "William Saliba")])
    players["Season"] = players["Season"].astype(int)
    out = ingest.add_team_defence(players, _history([("E0", "2526", "Arsenal", [0, 1])]))
    assert out["Season"].dtype == players["Season"].dtype
    assert "_season_key" not in out.columns


def test_a_club_with_no_history_is_left_blank_not_guessed() -> None:
    """NaN is absent evidence; `_weighted` renormalises rather than filling."""
    out = ingest.add_team_defence(
        _players([("E0", "2526", "Promoted", "Someone")]),
        _history([("E0", "2526", "Arsenal", [0, 1])]),
    )
    assert pd.isna(out.iloc[0]["team_ga_per_match"])


def test_xg_is_optional_so_pre_2014_seasons_still_work() -> None:
    """Understat starts at 1415; the goals columns carry the earlier ones."""
    history = _history([("E0", "1112", "Arsenal", [0, 2])]).drop(columns=["xGAgainst"])
    out = ingest.add_team_defence(_players([("E0", "1112", "Arsenal", "Someone")]), history)
    assert out.iloc[0]["team_ga_per_match"] == pytest.approx(1.0)
    assert "team_xga_per_match" not in out.columns


def test_a_player_moving_club_gets_each_club_s_record() -> None:
    """One row per player per team per season, so the join must be per row."""
    history = _history([
        ("E0", "2526", "Arsenal", [0, 0]),
        ("E0", "2526", "Burnley", [3, 3]),
    ])
    out = ingest.add_team_defence(
        _players([
            ("E0", "2526", "Arsenal", "Journeyman"),
            ("E0", "2526", "Burnley", "Journeyman"),
        ]),
        history,
    )
    assert sorted(out["team_ga_per_match"]) == [0.0, 3.0]


# --- the definition is inert until it is weighted -----------------------------

def test_a_zero_weighted_sub_rating_changes_nothing() -> None:
    """`team_defence` ships at 0.0 so the config can hold it without publishing it.

    `usable_weights` filters `w != 0` from both the usable set and the total, so
    a zero weight contributes to neither. `GK`'s `workload` has relied on this
    all along.
    """
    frame = pd.DataFrame([{"a": 0.8, "b": 0.4, "unused": 0.1}])
    with_zero = _weighted(frame, {"a": 3.0, "b": 1.0, "unused": 0.0})
    without = _weighted(frame, {"a": 3.0, "b": 1.0})
    assert with_zero.iloc[0] == without.iloc[0]


def test_the_shipped_config_is_what_the_sweep_chose() -> None:
    """0.30 on centre backs, nothing on full backs.

    Chosen on defensive validity, not prediction: the centre-back line against
    next season's conceding moves -0.381 to -0.544, improving in all five
    leagues on all three outcome measures, while **no variant in the sweep was
    significant on RPS**. Two earlier runs were discarded, one for a club-name
    join missing 26.6% of defenders and one because `peak_weight` let
    zero-weighted groups into the published rating.

    Full backs stay at zero deliberately. `--stage fb` scored better on every
    aggregate, including the only significant RPS result in the experiment,
    and that result improved in just 8 of 11 movable seasons -- while the
    weight would leave a full back rated 8% on ball-winning.
    """
    config = load_config()
    cb = config['positions']['CB']['sub_rating_weights']
    assert cb['team_defence'] == pytest.approx(0.30)
    assert cb['defending'] == pytest.approx(0.32)
    fb = config['positions']['FB']['sub_rating_weights']
    assert fb['team_defence'] == 0.0
    assert fb['defending'] == pytest.approx(0.38), 'full backs keep production weights'
    for role in ('CB', 'FB'):
        weights = config['positions'][role]['sub_rating_weights']
        assert sum(weights.values()) == pytest.approx(1.0), role
        # The definition is present for both, whatever it is weighted at.
        assert set(config['positions'][role]['sub_ratings']['team_defence']) == {
            'team_xga_per_match', 'team_ga_per_match', 'team_clean_sheet_pct',
        }


def test_the_defending_group_keeps_a_real_share_of_a_centre_back() -> None:
    """`share-0.00` was rejected, so a centre back is never rated on no defending.

    Deleting the group did not make the model worse, but it leaves the
    composite entirely made of involvement, creation and finishing -- nothing
    defensive -- behind a bar the page labels "Ball-winning". If the weight
    ever reaches zero, that label in `web/src/components/PlayerDatabase.tsx`
    has to change with it.
    """
    assert load_config()["positions"]["CB"]["sub_rating_weights"]["defending"] > 0.0


def test_conceding_counts_against_a_defender() -> None:
    """Sign convention, matching the keeper's `goals_against_per90: -2.0`.

    Positive weights here would rate the leakiest defence in the division top.
    """
    config = load_config()
    metrics = config["positions"]["CB"]["sub_ratings"]["team_defence"]
    assert metrics["team_xga_per_match"] < 0
    assert metrics["team_ga_per_match"] < 0
    assert metrics["team_clean_sheet_pct"] > 0


# --- the sweep itself ---------------------------------------------------------

def test_every_variant_still_sums_to_one() -> None:
    """A composite whose weights do not sum to 1 is not on the same scale."""
    config = load_config()
    for name, variant in sweep.variants(config):
        weights = variant["positions"]["CB"]["sub_rating_weights"]
        assert sum(weights.values()) == pytest.approx(1.0), name


def test_the_share_variant_keeps_the_other_groups_in_proportion() -> None:
    """Only one thing varies: how much of the composite rests on defending."""
    config = load_config()
    base = config["positions"]["CB"]["sub_rating_weights"]
    moved = sweep._share_variant(config, 0.30)["positions"]["CB"]["sub_rating_weights"]
    assert moved["defending"] == pytest.approx(0.30)
    # creation and finishing were equal before and must stay equal.
    assert moved["creation"] == pytest.approx(moved["finishing"])
    assert moved["involvement"] / moved["creation"] == pytest.approx(
        base["involvement"] / base["creation"]
    )


def test_the_outcome_variant_only_moves_defending() -> None:
    """So it is isolated from the share sweep and the two can be told apart."""
    config = load_config()
    base = config["positions"]["CB"]["sub_rating_weights"]
    moved = sweep._outcome_variant(config, 0.30)["positions"]["CB"]["sub_rating_weights"]
    assert moved["team_defence"] == pytest.approx(0.30)
    # Against the pool, not against `base["defending"]`: the two defensive
    # groups share a fixed 0.62 between them, and now that 0.30 of it ships as
    # `team_defence` a builder that subtracted from `defending` again would
    # leave the weights summing to 0.70. `_pool` is what keeps every variant
    # describing the same experiment whatever is currently shipped.
    assert moved["defending"] == pytest.approx(sweep._pool(base) - 0.30)
    assert sweep._pool(moved) == pytest.approx(sweep._pool(base))
    for group in ("involvement", "creation", "finishing"):
        assert moved[group] == pytest.approx(base[group])


def test_the_fb_stage_builds_on_the_setting_it_is_given() -> None:
    """The second stage must never extend a winner the current numbers rejected.

    `CB_WINNER` is a constant recording what one sweep chose. When the first
    stage is re-run and picks differently, the full-back stage has to be told,
    or it quietly measures full backs on top of a centre-back setting nobody
    selected -- which is how the invalidated run would have propagated.
    """
    config = load_config()
    names = [name for name, _ in sweep.variants(config, stage="fb", cb_share=0.15)]
    assert names == ["baseline", "cb-0.15-only", "cb-0.15+fb-0.15", "cb-0.15+fb-0.30"]

    for _, variant in sweep.variants(config, stage="fb", cb_share=0.15)[1:]:
        assert variant["positions"]["CB"]["sub_rating_weights"]["team_defence"] == pytest.approx(
            0.15
        )


def test_the_fb_stage_defaults_to_the_recorded_winner() -> None:
    """Omitting it keeps the documented behaviour rather than silently zeroing."""
    config = load_config()
    names = [name for name, _ in sweep.variants(config, stage="fb")]
    assert names[1] == f"cb-{sweep.CB_WINNER:.2f}-only"


def test_the_baseline_variant_is_the_shipped_config() -> None:
    """Otherwise every delta is measured against something nobody ships."""
    config = load_config()
    name, variant = sweep.variants(config)[0]
    assert name == "baseline"
    assert (
        variant["positions"]["CB"]["sub_rating_weights"]
        == config["positions"]["CB"]["sub_rating_weights"]
    )


def test_the_defence_correlation_looks_at_the_following_season() -> None:
    """Same-season would flatter the outcome variant by construction.

    It is built from a season's own results, so scoring it against that same
    season's goals conceded would be marking its own homework. The model only
    ever reads strength joined from the previous season anyway.
    """
    strength = pd.DataFrame({
        "League": ["E0", "E0"], "Season": ["2425", "2425"],
        "Team": ["Arsenal", "Burnley"], "strength_defence": [85.0, 60.0],
    })
    history = _history([
        ("E0", "2526", "Arsenal", [0] * 20),
        ("E0", "2526", "Burnley", [3] * 20),
        # The same-season rows are deliberately the wrong way round: if the
        # function looked at these, the correlation would come back positive.
        ("E0", "2425", "Arsenal", [3] * 20),
        ("E0", "2425", "Burnley", [0] * 20),
    ])
    assert sweep.next_season_defence(strength, history, min_rows=2) < 0


def test_the_sweep_never_writes_the_shipped_strength_file() -> None:
    """A finally does not survive the process being killed.

    The first version swapped `team_strength.csv` per variant and restored it
    in a `finally`. A timeout cut the smoke test short, the `finally` never
    ran, and the shipped file was left holding a variant's numbers -- 1,160 of
    1,170 rows wrong by up to 5.3 points, with the API serving them.

    It was never necessary: `build_match_features` takes the strength frame, so
    the experiment holds its variants in memory and the file on disk is not
    part of the experiment at all.
    """
    import inspect

    # Checked against the code rather than the prose: the module comment names
    # the file deliberately, to say it does not write it.
    source = "\n".join(
        line for line in inspect.getsource(sweep).split("\n")
        if not line.lstrip().startswith("#")
    )
    assert "to_csv" not in source, "the sweep must not write any frame to disk"
    assert "shutil" not in source, "nothing to copy if nothing is overwritten"
    # Every other way a frame or file reaches disk, because the point is not
    # that one API is banned -- it is that an interrupted experiment cannot
    # leave anything behind. A `finally` does not run when the process is
    # killed, which is exactly how the corruption happened, so the guarantee
    # has to be that there is no write to interrupt.
    for forbidden in (
        "to_pickle", "to_parquet", "to_feather", "to_json", "to_hdf",
        "os.replace", "os.remove", "os.rename", "unlink(", "write_text",
        "write_bytes", "mkdir(", "savetxt", "np.save",
    ):
        assert forbidden not in source, f"the sweep must not call {forbidden}"
    # `open` only ever for reading, if at all.
    for line in source.splitlines():
        if "open(" in line and "fileno" not in line:
            assert not any(m in line for m in ('"w"', "'w'", '"a"', "'a'", '"x"', "'x'")), line


def test_the_sweep_never_imports_a_module_that_writes_production() -> None:
    """`build_players` is the only thing allowed to write the two artifacts.

    An experiment that imported it could call `main()` and overwrite them
    without any of the string checks above noticing.
    """
    import inspect

    source = inspect.getsource(sweep)
    assert "build_players" not in source
    assert "from pipelines.build_players" not in source


# --- the artifacts the winner gets written to ---------------------------------

def test_the_ratings_artifacts_are_written_atomically() -> None:
    """A kill part-way through a write must not leave a readable, wrong file.

    This is the failure that actually happened: a timeout left
    `team_strength.csv` holding an experiment's numbers, 1,160 of 1,170 rows
    wrong, and the API served them. A truncated CSV still parses, so nothing
    downstream can tell it from a complete one.
    """
    source = Path(build_players.__file__).read_text(encoding="utf-8")
    assert "to_csv(OUTPUT" not in source, "write the ratings through _write_csv"
    assert "to_csv(TEAM_OUTPUT" not in source, "write the strength through _write_csv"
    assert "tmp.replace(path)" in source


def test_the_atomic_write_leaves_no_temporary_file_behind(tmp_path: Path) -> None:
    """And the finished file holds exactly what was handed to it."""
    target = tmp_path / "ratings.csv"
    frame = pd.DataFrame({"Player": ["William Saliba"], "rating": [85]})
    build_players._write_csv(frame, target)
    assert target.exists()
    assert list(tmp_path.iterdir()) == [target]
    pd.testing.assert_frame_equal(pd.read_csv(target), frame)


def test_the_atomic_write_replaces_rather_than_appends(tmp_path: Path) -> None:
    """Writing twice must not grow the file, which `mode="a"` by accident would."""
    target = tmp_path / "strength.csv"
    build_players._write_csv(pd.DataFrame({"Team": ["Arsenal"], "v": [1]}), target)
    build_players._write_csv(pd.DataFrame({"Team": ["Inter"], "v": [2]}), target)
    out = pd.read_csv(target)
    assert len(out) == 1
    assert out["Team"].iloc[0] == "Inter"


# --- a zero weight has to mean zero everywhere --------------------------------

def test_a_zero_weighted_sub_rating_stays_out_of_the_peak() -> None:
    """`peak_weight` blends in a player's best sub-rating, and it used to take
    the max over *every* defined sub-rating -- including ones weighted zero.

    So a weight of zero kept a term out of the composite but not out of the
    published rating. Rebuilding with `team_defence` defined at zero weight for
    full backs moved 807 of 3,904 of them by up to 4 points, with no full-back
    weight and no full-back sub-rating changed.

    Here a full back is outstanding at the zero-weighted group and mediocre at
    everything he is actually rated on. His rating must not know about it.
    """
    config = copy.deepcopy(load_config())
    config["peak_weight"] = 0.15
    weights = config["positions"]["FB"]["sub_rating_weights"]
    assert weights["team_defence"] == 0.0, "the fixture depends on this shipping at zero"

    frame = pd.DataFrame({
        "role": ["FB", "FB"],
        "Season": ["2526", "2526"],
        "sub_defending": [0.5, 0.5],
        "sub_creation": [0.5, 0.5],
        "sub_involvement": [0.5, 0.5],
        "sub_finishing": [0.5, 0.5],
        "sub_team_defence": [0.02, 0.98],
    })
    peak = rating._peak_over_rated_sub_ratings(frame, config)
    assert peak.iloc[0] == pytest.approx(peak.iloc[1]), (
        "the club's clean sheets changed a full back's peak"
    )
    assert peak.iloc[0] == pytest.approx(0.5)


def test_the_peak_ignores_every_role_s_zero_weighted_sub_ratings() -> None:
    """Across every role in the config, not just the two this sweep touched.

    `GK` carries a zero-weighted `workload` and has done since long before
    `team_defence` existed, so the keepers were being credited for being
    outstanding at something the config says they are not rated on. Measured:
    459 of 1,799 keepers moved when the peak was narrowed, by up to 5 points,
    while every outfield role came back bit-identical.

    Parameterising over the config rather than naming roles means a new
    zero-weighted group cannot be added without this covering it.
    """
    config = copy.deepcopy(load_config())
    config["peak_weight"] = 0.15
    for role, spec in config["positions"].items():
        weights = spec["sub_rating_weights"]
        zeroed = [name for name, weight in weights.items() if weight == 0]
        if not zeroed:
            continue
        rated_on = [name for name, weight in weights.items() if weight != 0]
        # Mediocre at everything he is rated on, outstanding at what he is not.
        row = {f"sub_{name}": 0.5 for name in rated_on}
        row |= {f"sub_{name}": 0.99 for name in zeroed}
        frame = pd.DataFrame([row | {"role": role, "Season": "2526"}])
        peak = rating._peak_over_rated_sub_ratings(frame, config)
        assert peak.iloc[0] == pytest.approx(0.5), (
            f"{role}: {zeroed} reached the peak despite being weighted zero"
        )


def test_the_peak_is_the_best_of_what_the_role_is_rated_on() -> None:
    """For every role, including those with nothing zero-weighted.

    The narrowing must not quietly drop a group that carries real weight --
    that would cost a specialist the credit `peak_weight` exists to give.
    """
    config = copy.deepcopy(load_config())
    config["peak_weight"] = 0.15
    for role, spec in config["positions"].items():
        rated_on = [n for n, w in spec["sub_rating_weights"].items() if w != 0]
        for standout in rated_on:
            row = {f"sub_{name}": 0.3 for name in rated_on}
            row[f"sub_{standout}"] = 0.9
            row |= {
                f"sub_{n}": 0.0 for n, w in spec["sub_rating_weights"].items() if w == 0
            }
            frame = pd.DataFrame([row | {"role": role, "Season": "2526"}])
            peak = rating._peak_over_rated_sub_ratings(frame, config)
            assert peak.iloc[0] == pytest.approx(0.9), f"{role}/{standout}"


def test_a_role_the_config_does_not_know_gets_no_peak() -> None:
    """Rather than a peak taken over columns that mean nothing for it."""
    config = copy.deepcopy(load_config())
    frame = pd.DataFrame([
        {"role": "SWEEPER", "Season": "2526", "sub_defending": 0.9},
        {"role": "CB", "Season": "2526", "sub_defending": 0.9},
    ])
    peak = rating._peak_over_rated_sub_ratings(frame, config)
    assert pd.isna(peak.iloc[0])
    assert peak.iloc[1] == pytest.approx(0.9)


def test_the_peak_survives_a_sub_rating_the_frame_never_got() -> None:
    """A season where a whole group is absent must not blank the peak.

    FBref served no `keeper` table for 2015/16 and no `misc` table for
    2014/15, so a frame legitimately arrives missing columns the config names.
    The peak should be taken over what is there.
    """
    config = copy.deepcopy(load_config())
    frame = pd.DataFrame([{"role": "CB", "Season": "1415", "sub_involvement": 0.7}])
    peak = rating._peak_over_rated_sub_ratings(frame, config)
    assert peak.iloc[0] == pytest.approx(0.7)
