"""Tests for the weekly refresh runner.

The failure that matters is a *partial* refresh: if one step fails and the rest
carry on, some files are this week's and some last week's, and the site reports
success either way. So the ordering and the stop-on-failure behaviour are what
these assert, not the pipelines themselves, which have their own tests.
"""

from __future__ import annotations

import io
import subprocess
from urllib.error import HTTPError, URLError

import pytest

from pipelines import weekly


def test_steps_run_in_dependency_order() -> None:
    """Ratings feed training, training feeds the forecast.

    `train` reads team_strength.csv, and `score_upcoming` loads the newest model
    artifact, so reordering these silently trains on last week's ratings or
    forecasts with last week's model.
    """
    order = [step.name for step in weekly.STEPS]
    assert order.index("build_dataset") < order.index("build_players")
    assert order.index("build_players") < order.index("train")
    assert order.index("train") < order.index("score_upcoming")


def test_the_cross_league_correction_is_refitted_after_the_retrain() -> None:
    """The offsets describe one model's residual error on European ties and do
    not transfer to another, so refitting before the retrain would leave the
    site correcting a model it is no longer serving.
    """
    order = [step.name for step in weekly.STEPS]
    assert order.index("train") < order.index("validate_european")
    assert order.index("fetch_european") < order.index("validate_european")
    assert order.index("validate_european") < order.index("fit_league_adjustment")


def test_every_step_names_a_real_pipeline() -> None:
    """`run_step` shells out to `python -m pipelines.<name>`, so a typo here is
    invisible until the scheduled run fails on a Monday morning."""
    import importlib.util

    missing = [
        step.name
        for step in weekly.STEPS
        if importlib.util.find_spec(f"pipelines.{step.name}") is None
    ]
    assert not missing, f"no such pipeline module: {missing}"


def test_a_failed_step_stops_the_run(monkeypatch, tmp_path) -> None:
    """A partial refresh must not be reported as a success."""
    attempted: list[str] = []

    def fake_run(step):
        attempted.append(step.name)
        if step.name == "build_players":
            raise subprocess.CalledProcessError(1, step.name, "", "boom")
        return 0.0

    monkeypatch.setattr(weekly, "run_step", fake_run)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["weekly"])

    assert weekly.main() == 1
    assert attempted == ["build_dataset", "build_players"], "should stop at the failure"


def test_a_stopped_api_is_not_a_failure(monkeypatch, tmp_path) -> None:
    """The files are on disk; the next start picks them up."""
    monkeypatch.setattr(weekly, "run_step", lambda step: 0.0)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)

    def refuse(*_args, **_kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr(weekly, "urlopen", refuse)
    monkeypatch.setattr("sys.argv", ["weekly"])

    assert weekly.main() == 0
    assert weekly.reload_api() is True, "not running is not the same as refusing"


def test_a_refused_reload_fails_the_run(monkeypatch, tmp_path) -> None:
    """An API that is alive and says no leaves the site on the old model.

    That is a partial refresh -- every file on disk is this week's, the site is
    serving last week's -- and reporting success would hide it. It happens when
    the server has been running since before the feature code changed, so it
    cannot build what the model just trained expects.
    """
    monkeypatch.setattr(weekly, "run_step", lambda step: 0.0)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)

    def refuse(*_args, **_kwargs):
        raise HTTPError(
            "http://127.0.0.1:8000/admin/reload", 409, "Conflict", {},
            io.BytesIO(b'{"reason": "model needs features this process cannot build"}'),
        )

    monkeypatch.setattr(weekly, "urlopen", refuse)
    monkeypatch.setattr("sys.argv", ["weekly"])

    assert weekly.main() == 1
    assert weekly.reload_api() is False


def test_only_runs_a_single_step(monkeypatch, tmp_path) -> None:
    """So one failure can be rerun without repeating the expensive steps."""
    attempted: list[str] = []
    monkeypatch.setattr(weekly, "run_step", lambda step: attempted.append(step.name))
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr(weekly, "reload_api", lambda *a, **k: True)
    monkeypatch.setattr("sys.argv", ["weekly", "--only", "score_upcoming"])

    assert weekly.main() == 0
    assert attempted == ["score_upcoming"]


def test_skip_leaves_the_rest(monkeypatch, tmp_path) -> None:
    attempted: list[str] = []
    monkeypatch.setattr(weekly, "run_step", lambda step: attempted.append(step.name))
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr(weekly, "reload_api", lambda *a, **k: True)
    monkeypatch.setattr("sys.argv", ["weekly", "--skip", "build_players"])

    assert weekly.main() == 0
    assert "build_players" not in attempted
    assert "train" in attempted


@pytest.mark.parametrize(
    ("seconds", "expected"), [(9, "9s"), (65, "1m 05s"), (600, "10m 00s")]
)
def test_duration_reads_naturally(seconds: int, expected: str) -> None:
    assert weekly._duration(seconds) == expected


def test_an_optional_step_failing_does_not_stop_the_refresh(monkeypatch, tmp_path) -> None:
    """The regression this guards against.

    `fetch_european` drives a headless browser against FBref -- the most
    fragile thing in the job -- and it sits before `reload`. If a scraper
    outage stopped the run, a week of freshly fetched, rebuilt and retrained
    domestic data would sit on disk while the site kept serving the old model.
    That is a worse failure than the staleness this job exists to prevent.
    """
    attempted: list[str] = []
    reloaded: list[bool] = []

    def fake_run(step):
        attempted.append(step.name)
        if step.name == "fetch_european":
            raise subprocess.CalledProcessError(1, step.name, "", "no browser")
        return 0.0

    monkeypatch.setattr(weekly, "run_step", fake_run)
    monkeypatch.setattr(weekly, "reload_api", lambda url: reloaded.append(True) or True)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["weekly"])

    exit_code = weekly.main()

    # Everything after the failure still ran, and the site was told about it.
    assert "validate_european" in attempted
    assert "fit_league_adjustment" in attempted
    assert reloaded == [True], "the site must still be reloaded with fresh domestic data"
    # ... but the run is not a success. A job that exits 0 after an error is a
    # job nobody looks at again.
    assert exit_code == 1


def test_only_the_reporting_and_european_steps_are_optional() -> None:
    """Required steps build the dataset and the model the site serves. Optional
    ones refine a correction that applies to European ties, or re-score the
    walk-forward history -- neither of which the site needs in order to be
    correct, and neither of which may cost the reload when it fails."""
    optional = {s.name for s in weekly.STEPS if not s.required}
    assert optional == {
        "backtest", "fetch_european", "validate_european", "fit_league_adjustment",
    }


def test_the_backtest_runs_after_the_publication_path_and_before_european() -> None:
    """Order, stated as the two constraints that produced it.

    It needs `build_players` (it reads `team_strength.csv`) and must precede
    `validate_european`, which prints the walk-forward figure as its domestic
    reference. It does **not** need `train`: every fold fits its own model, so
    placing it after the model the site is waiting for costs the site nothing.
    """
    order = [s.name for s in weekly.STEPS]
    assert order.index("build_players") < order.index("backtest")
    assert order.index("project_season") < order.index("backtest"), (
        "the backtest delays publication"
    )
    assert order.index("backtest") < order.index("validate_european")


def test_the_backtest_is_asked_for_the_five_league_predictions() -> None:
    """Without both flags it scores one league, or scores and saves nothing --
    and the accuracy page reads the file it would not have written."""
    step = next(s for s in weekly.STEPS if s.name == "backtest")
    assert set(step.args) == {"--all-leagues", "--save-predictions"}


def test_a_required_step_failing_still_skips_the_reload(monkeypatch, tmp_path) -> None:
    """The other half: a half-built dataset must never reach the site."""
    reloaded: list[bool] = []

    def fake_run(step):
        if step.name == "train":
            raise subprocess.CalledProcessError(1, step.name, "", "boom")
        return 0.0

    monkeypatch.setattr(weekly, "run_step", fake_run)
    monkeypatch.setattr(weekly, "reload_api", lambda url: reloaded.append(True) or True)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["weekly"])

    assert weekly.main() == 1
    assert reloaded == [], "a failed required step must not publish anything"


# --- what a successful step is allowed to hide --------------------------------
#
# On 2026-10-19 `build_dataset` dropped 17 played matches -- every Bundesliga
# and Serie A fixture of that round -- because their shooting rows came back
# without the columns the model reads. It logged a WARNING naming the leagues
# and the count. The run took 94 minutes and ended with a coverage table, so
# the eight-line tail kept the table and discarded the only sentence that
# explained what had happened. The weekly log shows no trace of it.


def _output(lines: list[str]) -> str:
    return "\n".join(lines)


def test_a_warning_far_from_the_end_is_kept() -> None:
    """The case this exists for: 200 lines of noise after the only line that
    mattered."""
    warning = "WARNING Dropping 34 of 88 team-match row(s) (league(s) ['D1', 'I1'])"
    output = _output([warning, *[f"line {i}" for i in range(200)]])
    summary = weekly._summary_lines(output)
    assert warning in summary
    assert summary[0] == warning, "flagged lines come first"


def test_an_error_far_from_the_end_is_kept() -> None:
    error = "ERROR could not read the shooting table"
    output = _output([*[f"line {i}" for i in range(50)], error, *[f"tail {i}" for i in range(50)]])
    assert error in weekly._summary_lines(output)


def test_the_tail_is_still_the_tail() -> None:
    """A clean run must produce exactly what it produced before."""
    lines = [f"line {i}" for i in range(50)]
    assert weekly._summary_lines(_output(lines)) == lines[-weekly.SUMMARY_LINES:]


def test_a_flagged_line_inside_the_tail_is_not_repeated() -> None:
    """It is already going to be shown; hoisting it would print it twice."""
    warning = "WARNING something went wrong"
    lines = [*[f"line {i}" for i in range(50)], warning, "after"]
    summary = weekly._summary_lines(_output(lines))
    assert summary.count(warning) == 1
    assert summary == lines[-weekly.SUMMARY_LINES:], "the tail is unchanged"


def test_an_earlier_copy_of_a_tail_line_is_not_hoisted() -> None:
    """De-duplication is across the two groups, by text."""
    warning = "WARNING the same thing twice"
    lines = [warning, *[f"line {i}" for i in range(50)], warning, "after"]
    summary = weekly._summary_lines(_output(lines))
    assert summary.count(warning) == 1


def test_flagged_lines_keep_their_original_order() -> None:
    first = "WARNING first problem"
    second = "ERROR second problem"
    third = "WARNING third problem"
    lines = [first, "noise", second, "noise", third, *[f"line {i}" for i in range(30)]]
    summary = weekly._summary_lines(_output(lines))
    assert [line for line in summary if line in {first, second, third}] == [
        first, second, third,
    ]


def test_repeated_tail_lines_are_preserved() -> None:
    """Two identical lines in the tail are two things that happened, not one.

    De-duplicating the tail against itself would quietly rewrite a step's log.
    """
    lines = ["same", "same", "same", "same", "same", "same", "same", "same"]
    assert weekly._summary_lines(_output(lines)) == lines


def test_empty_and_blank_output() -> None:
    assert weekly._summary_lines(None) == []
    assert weekly._summary_lines("") == []
    assert weekly._summary_lines("\n   \n\n") == []


def test_output_shorter_than_the_tail_is_returned_whole() -> None:
    lines = ["WARNING something", "done"]
    assert weekly._summary_lines(_output(lines)) == lines


def test_a_word_that_merely_contains_a_level_is_not_flagged() -> None:
    """"no errors" is the opposite of an error, and `ERRORS` is not a level."""
    noise = "checked for errors and found none"
    lines = [noise, *[f"line {i}" for i in range(30)]]
    assert noise not in weekly._summary_lines(_output(lines))


def test_lines_are_stripped_as_before() -> None:
    noise = "\n".join(f"l{i}" for i in range(30))
    assert "WARNING padded" in weekly._summary_lines(f"   WARNING padded   \n{noise}")


# --- a failed backtest costs the accuracy page, never the reload --------------


def test_a_failed_backtest_still_reloads_the_site(monkeypatch, tmp_path) -> None:
    """It is reporting. The model, the projections and the fixtures are all
    published by steps that ran before it."""
    reloaded: list[bool] = []

    def fake_run(step):
        if step.name == "backtest":
            raise subprocess.CalledProcessError(1, step.name, "", "boom")
        return 0.0

    monkeypatch.setattr(weekly, "run_step", fake_run)
    monkeypatch.setattr(weekly, "reload_api", lambda url: reloaded.append(True) or True)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["weekly"])

    code = weekly.main()

    assert reloaded == [True], "a failed backtest blocked the reload"
    assert code != 0, "the run reported success despite a failed step"


def test_a_failed_backtest_is_named_in_the_summary(monkeypatch, tmp_path) -> None:
    """Silence here is how the figures went stale through three retrains in the
    first place.

    Asserted against the log file the job writes rather than `caplog`, because
    that file is what somebody actually reads on a Monday -- and the job
    installs its own handlers, so captured records prove nothing about it.
    """
    def fake_run(step):
        if step.name == "backtest":
            raise subprocess.CalledProcessError(1, step.name, "", "boom")
        return 0.0

    monkeypatch.setattr(weekly, "run_step", fake_run)
    monkeypatch.setattr(weekly, "reload_api", lambda url: True)
    monkeypatch.setattr(weekly, "LOG_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["weekly"])

    weekly.main()

    written = [p.read_text(encoding="utf-8") for p in tmp_path.glob("weekly-*.log")]
    assert written, "the run left no log behind"
    summary = written[0]
    assert "backtest" in summary, summary
    assert "optional step(s) failed" in summary, summary
