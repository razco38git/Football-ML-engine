"""The daily results refresh.

Its whole value is deciding *whether* to run. A match played on a Tuesday used
to wait until Monday, so this runs every morning -- but most mornings
football-data has published nothing new, and rebuilding an identical table and
bouncing the API every day would be worse than the problem it solves.

So the behaviour worth pinning is the short-circuit, the fingerprint it turns
on, and that a failure anywhere leaves the site serving the complete older copy
rather than a half-refreshed one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines import daily


@pytest.fixture(autouse=True)
def quiet_logging(monkeypatch, tmp_path):
    """Keep the tests out of the production log.

    `daily.main` configures logging to `logs/daily-<today>.log`, the same file
    the scheduled job writes. Running these tests put 43 lines of invented
    failures into it -- "Settling failed", "build_dataset failed" -- stamped
    with a real timestamp, in the file somebody reads to find out what the
    morning's run actually did.
    """
    monkeypatch.setattr(daily, "_configure_logging", lambda *a, **k: tmp_path / "daily.log")


@pytest.fixture
def runs(monkeypatch) -> list[str]:
    """Record which pipelines were invoked, running none of them."""
    called: list[str] = []
    monkeypatch.setattr(daily, "run_step", lambda step: called.append(step.name) or 0.0)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)
    return called


def _args(monkeypatch, **over) -> None:
    argv = ["daily"]
    if over.get("force"):
        argv.append("--force")
    monkeypatch.setattr(sys, "argv", argv)


# --- the fingerprint ----------------------------------------------------------

def test_fingerprint_is_rows_and_latest_date(tmp_path, monkeypatch) -> None:
    path = tmp_path / "matches_all.csv"
    pd.DataFrame({"Date": ["2026-09-19", "2026-09-20"]}).to_csv(path, index=False)
    monkeypatch.setattr(daily, "MATCHES", path)
    assert daily._fingerprint() == (2, "2026-09-20")


def test_fingerprint_notices_a_correction_that_adds_no_date(tmp_path, monkeypatch) -> None:
    """A re-published old match moves the count and not the latest date.

    Keying on the date alone would call that "nothing new" and leave the
    correction unpublished until the next Monday.
    """
    path = tmp_path / "matches_all.csv"
    monkeypatch.setattr(daily, "MATCHES", path)
    pd.DataFrame({"Date": ["2026-09-20"]}).to_csv(path, index=False)
    before = daily._fingerprint()
    pd.DataFrame({"Date": ["2026-09-19", "2026-09-20"]}).to_csv(path, index=False)
    assert daily._fingerprint() != before


def test_fingerprint_survives_a_missing_or_empty_file(tmp_path, monkeypatch) -> None:
    """First ever run, or a wiped data directory: not a crash."""
    missing = tmp_path / "nope.csv"
    monkeypatch.setattr(daily, "MATCHES", missing)
    assert daily._fingerprint() == (0, "")

    empty = tmp_path / "empty.csv"
    pd.DataFrame({"Date": []}).to_csv(empty, index=False)
    monkeypatch.setattr(daily, "MATCHES", empty)
    assert daily._fingerprint() == (0, "")


# --- the short-circuit --------------------------------------------------------

def test_settling_happens_before_the_fetch(monkeypatch, runs) -> None:
    """The ordering is the whole fix, so it is pinned rather than assumed.

    `build_dataset` grew the FBref top-up, which scrapes a page per team at a
    seven-second rate limit whenever football-data is behind -- every morning,
    since 2026-09-20. With settling second, the previous day's results waited
    twenty-five minutes behind a scrape they do not need.
    """
    _args(monkeypatch)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (29143, "2026-09-20"))

    assert daily.main() == 0
    assert runs.index("score_upcoming") < runs.index("build_dataset")


def test_the_first_reload_comes_before_the_fetch(monkeypatch) -> None:
    """Publishing early is the point: the tabs are correct a minute in."""
    order: list[str] = []
    monkeypatch.setattr(daily, "run_step", lambda s: order.append(s.name) or 0.0)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: order.append("reload") or True)
    _args(monkeypatch)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (29143, "2026-09-20"))

    daily.main()
    assert order.index("reload") < order.index("build_dataset")


def test_a_failed_fetch_keeps_the_settled_results_published(monkeypatch) -> None:
    """Settling already succeeded and published; that is a useful run, not a failed one."""
    _args(monkeypatch)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (1, "2026-09-20"))
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)

    def run(step):
        if step.name == "build_dataset":
            raise subprocess.CalledProcessError(1, step.name)
        return 0.0

    monkeypatch.setattr(daily, "run_step", run)
    assert daily.main() == 0


def test_a_failed_settle_does_not_stop_the_fetch(monkeypatch, runs) -> None:
    """The fetch may still bring results worth projecting."""
    _args(monkeypatch)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (29143, "2026-09-20"))
    called: list[str] = []

    def run(step):
        called.append(step.name)
        if step.args == ("--settle-only",):
            raise subprocess.CalledProcessError(1, step.name)
        return 0.0

    monkeypatch.setattr(daily, "run_step", run)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)
    daily.main()
    assert "build_dataset" in called


def test_nothing_new_still_settles(monkeypatch, runs) -> None:
    """The bug this guards: settling gated on the wrong source.

    On 2026-10-12, 51 predictions for the previous day sat unsettled with all
    seventeen results already on the schedule pages, because football-data had
    published nothing and the job stopped after the fetch. Settlement reads both
    sources, so it has work to do whatever football-data did.
    """
    _args(monkeypatch)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (29143, "2026-09-20"))

    assert daily.main() == 0
    assert runs == ["score_upcoming", "build_dataset"]


def test_nothing_new_does_not_reproject(monkeypatch, runs) -> None:
    """`project_season` simulates from the results table and nothing else.

    With no new rows it would spend four minutes rewriting the file it already
    wrote, so it is the one step the short-circuit still skips.
    """
    _args(monkeypatch)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (29143, "2026-09-20"))

    assert daily.main() == 0
    assert "project_season" not in runs


def test_new_results_run_the_whole_chain(monkeypatch, runs) -> None:
    _args(monkeypatch)
    seen = iter([(29143, "2026-09-20"), (29170, "2026-10-10")])
    monkeypatch.setattr(daily, "_fingerprint", lambda: next(seen))

    assert daily.main() == 0
    assert runs == ["score_upcoming", "build_dataset", "score_upcoming", "project_season"]


def test_force_reprojects_without_new_results(monkeypatch, runs) -> None:
    """`--force` is for re-simulating; settling needs no forcing."""
    _args(monkeypatch, force=True)
    monkeypatch.setattr(daily, "_fingerprint", lambda: (29143, "2026-09-20"))

    assert daily.main() == 0
    assert runs == ["score_upcoming", "build_dataset", "score_upcoming", "project_season"]


# --- failures leave the site complete rather than half-refreshed --------------

def test_a_failed_fetch_touches_nothing(monkeypatch, runs) -> None:
    """football-data being unreachable is normal and must cost nothing."""
    _args(monkeypatch)

    def boom(step):
        raise subprocess.CalledProcessError(1, step.name)

    monkeypatch.setattr(daily, "run_step", boom)
    reloaded: list[bool] = []
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: reloaded.append(True) or True)

    assert daily.main() == 1
    assert reloaded == []


def test_a_failed_projection_still_reports_failure(monkeypatch) -> None:
    """The settled results stay published; the run still exits non-zero.

    This used to assert that nothing was published until every step had
    succeeded. That is no longer the intent: settling publishes deliberately
    and early, because the three tabs reading it should not wait on a
    twenty-minute scrape. What must not happen is a *projection* failure
    passing silently -- a half-rebuilt table says nothing about which half it
    is.
    """
    _args(monkeypatch)
    seen = iter([(1, "2026-09-20"), (2, "2026-10-10")])
    monkeypatch.setattr(daily, "_fingerprint", lambda: next(seen))

    called: list[str] = []

    def run(step):
        called.append(step.name)
        if step.name == "project_season":
            raise subprocess.CalledProcessError(1, step.name)
        return 0.0

    monkeypatch.setattr(daily, "run_step", run)
    reloaded: list[bool] = []
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: reloaded.append(True) or True)

    assert daily.main() == 1
    assert called == ["score_upcoming", "build_dataset", "score_upcoming", "project_season"]
    # Settling published before the fetch, and that stands.
    assert reloaded == [True]


def test_a_dead_api_is_reported_not_swallowed(monkeypatch, runs) -> None:
    """The data is on disk; the run must still say it is not published."""
    _args(monkeypatch)
    seen = iter([(1, "2026-09-20"), (2, "2026-10-10")])
    monkeypatch.setattr(daily, "_fingerprint", lambda: next(seen))
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: False)

    assert daily.main() == 1


# --- the chain itself ---------------------------------------------------------

def test_the_expensive_weekly_steps_are_not_in_the_daily_chain() -> None:
    """Skipping these is the whole point; including one would be a regression.

    `build_players` is 7m21s of whole-season numbers that barely move in a day,
    and `train` would mint a model version every morning, scattering the live
    track record across seven versions a week.
    """
    names = {daily.FETCH.name, daily.SETTLE.name, daily.PROJECT.name}
    assert "build_players" not in names
    assert "train" not in names
    assert names == {"build_dataset", "score_upcoming", "project_season"}


def test_settling_comes_before_projecting() -> None:
    """`project_season` reads what `score_upcoming` has settled."""
    order = [s.name for s in daily.STEPS]
    assert order.index("score_upcoming") < order.index("project_season")
