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
import pytest

from footballml.players import ingest
from footballml.players.fbref import _is_same_name, _tokens, match_players
from footballml.players.rating import MAX_MISSING_WEIGHT, _weighted, load_config
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


def test_the_shipped_config_weights_team_defence_at_zero() -> None:
    """Nothing from the `outcome` arm ships until the sweep is re-run.

    A first sweep chose 0.30, on a join that silently missed 26.6% of
    defenders. The definition stays so the measurement can be repeated; the
    weight does not.
    """
    config = load_config()
    for role in ("CB", "FB"):
        weights = config["positions"][role]["sub_rating_weights"]
        assert weights["team_defence"] == 0.0
        assert sum(weights.values()) == pytest.approx(1.0)
        # And the definition is there to be switched on.
        assert set(config["positions"][role]["sub_ratings"]["team_defence"]) == {
            "team_xga_per_match", "team_ga_per_match", "team_clean_sheet_pct",
        }


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
    assert moved["defending"] == pytest.approx(base["defending"] - 0.30)
    for group in ("involvement", "creation", "finishing"):
        assert moved[group] == pytest.approx(base[group])


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
