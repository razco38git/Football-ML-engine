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


def test_the_european_steps_are_the_optional_ones() -> None:
    """Required steps build the dataset the site serves; optional ones refine a
    correction that applies to European ties alone."""
    optional = {s.name for s in weekly.STEPS if not s.required}
    assert optional == {"fetch_european", "validate_european", "fit_league_adjustment"}


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
