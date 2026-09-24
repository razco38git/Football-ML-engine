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
