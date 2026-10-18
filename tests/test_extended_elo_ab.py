"""The extended-Elo A/B: its arms, its pre-declared bar, and its isolation.

Elo is one zero-sum pool, so changing what it learns from changes the `elo`
feature on every domestic match, not just the European ones. The experiment is
therefore a non-inferiority test on 20,074 domestic fixtures, and these pin the
parts of it that must not drift.
"""

from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from pipelines import extended_elo_ab as ab


def _ties(*competitions: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "League": list(competitions),
            "HomeTeam": [f"H{i}" for i in range(len(competitions))],
            "AwayTeam": [f"A{i}" for i in range(len(competitions))],
            "FTHG": [1] * len(competitions),
            "FTAG": [0] * len(competitions),
        }
    )


# --- the arms -----------------------------------------------------------------


def test_the_baseline_arm_is_exactly_what_production_feeds_elo() -> None:
    """Named `production`, and it must *be* production.

    If this arm were anything other than the frame `add_elo(extra=...)` already
    receives, every delta would be measured against a configuration nobody
    runs, and a "no harm" verdict would say nothing about the live model.
    """
    kept = _ties("UCL", "UEL")
    name, frame = ab.arms(_ties("UECL"), kept)[0]
    assert name == "production"
    pd.testing.assert_frame_equal(frame, kept)


def test_the_extended_arm_adds_every_dropped_tie() -> None:
    kept, dropped = _ties("UCL"), _ties("UEL", "UECL", "UCL")
    frames = dict(ab.arms(dropped, kept))
    assert len(frames["extended"]) == len(kept) + len(dropped)


def test_the_third_arm_excludes_the_weakest_competition_only() -> None:
    """Declared before the run, not sliced out afterwards if the headline
    disappoints.

    The Conference League is the newest and weakest of the three and supplies
    723 of the dropped ties, so it is the likeliest source of noise. Naming it
    in advance is what separates a diagnosis from a fishing expedition.
    """
    kept, dropped = _ties("UCL"), _ties("UEL", "UECL", "UCL")
    frames = dict(ab.arms(dropped, kept))
    arm = frames["extended-no-uecl"]
    assert set(arm["League"]) == {"UCL", "UEL"}
    assert len(arm) == len(frames["extended"]) - 1


def test_the_weakest_competition_is_a_constant_not_a_result() -> None:
    """It has to be readable in the source before the numbers exist."""
    assert ab.WEAKEST == ("UECL",)
    assert isinstance(ab.WEAKEST, tuple)


def test_every_arm_keeps_the_kept_ties() -> None:
    """No arm may *remove* what production already learns from: this experiment
    only ever adds, so a regression cannot be blamed on lost data."""
    kept, dropped = _ties("UCL", "UEL"), _ties("UECL", "UEL")
    for name, frame in ab.arms(dropped, kept):
        assert len(frame) >= len(kept), name


# --- the bar ------------------------------------------------------------------


def test_the_margin_is_the_measured_floor() -> None:
    """0.0005, because that is what this harness cannot resolve.

    The defending sweep spanned 0.00012 across nine arms and the squad-size
    sweep 0.00018, neither producing a significant result on 20,074 fixtures. A
    non-inferiority margin tighter than that would claim a precision the
    pipeline does not have.
    """
    assert ab.MARGIN == 0.0005


def test_the_movable_season_count_excludes_the_dead_fold() -> None:
    """The earliest test season trains with the strength features absent, so its
    delta is exactly zero for any arm and it cannot count toward the bar."""
    assert ab.MOVABLE_SEASONS == 11


# --- isolation from production ------------------------------------------------


def test_the_experiment_cannot_write_anything() -> None:
    """A killed run must leave nothing behind.

    `build_match_features` takes the European frame as an argument, so every
    arm lives in memory. The same guarantee as `defending_sweep`, for the same
    reason: a `finally` does not survive the process being killed, and one
    already left `team_strength.csv` holding an experiment's numbers with the
    API serving them.
    """
    source = "\n".join(
        line
        for line in inspect.getsource(ab).split("\n")
        if not line.lstrip().startswith("#")
    )
    for forbidden in (
        "to_csv", "to_pickle", "to_parquet", "to_json", "to_hdf", "shutil",
        "os.replace", "os.remove", "os.rename", "unlink(", "write_text",
        "write_bytes", "mkdir(", "savetxt", "np.save",
    ):
        assert forbidden not in source, f"the experiment must not call {forbidden}"
    for line in source.splitlines():
        if "open(" in line and "fileno" not in line:
            assert not any(m in line for m in ('"w"', "'w'", '"a"', "'a'", '"x"', "'x'")), line


def test_the_experiment_does_not_import_the_thing_that_writes_production() -> None:
    tree = ast.parse(inspect.getsource(ab))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("build_players" in name for name in imported), imported


def test_the_european_frame_is_passed_in_rather_than_loaded_per_arm() -> None:
    """`build_match_features` defaults to reading `european_matches.csv` from
    disk. An arm that let it do so would silently measure production three
    times."""
    source = inspect.getsource(ab.main)
    assert "build_match_features(tmh, european=european)" in source
    assert "european=AUTO" not in source


def test_the_arms_are_checked_to_have_scored_the_same_fixtures() -> None:
    """Pairing two different fixture sets would make every interval nonsense."""
    source = inspect.getsource(ab.main)
    assert "arms scored different fixture counts" in source


def test_the_verdict_is_computed_from_the_declared_constants() -> None:
    """Not from numbers typed in after seeing the table."""
    source = inspect.getsource(ab.main)
    assert "hi < MARGIN" in source
    assert "seasons_ok >= 6" in source


@pytest.mark.parametrize("name", ["production", "extended", "extended-no-uecl"])
def test_each_declared_arm_is_present(name: str) -> None:
    built = dict(ab.arms(_ties("UEL", "UECL"), _ties("UCL")))
    assert name in built


def test_a_group_no_arm_can_move_is_not_counted_as_a_pass() -> None:
    """The earliest test fold trains with the strength features absent, so every
    arm scores it identically.

    Counting that as "neutral-or-better" hands every arm a free pass against a
    denominator that already excluded it, which is exactly how a 5-of-11 season
    record printed as 6 and turned a failing arm into a passing one.
    """
    import numpy as np

    keys = pd.Series(["dead"] * 4 + ["live"] * 4)
    base = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # Identical on the dead group, worse on the live one.
    arm = np.array([0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3])
    counts = ab._breakdown("group", keys, base, {"production": base, "arm": arm}, minimum=1)
    assert counts["arm"] == 0, "the unmovable group was counted as a pass"
