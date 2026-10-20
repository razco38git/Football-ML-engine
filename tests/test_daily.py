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
    # `_report_stranded_european` reads the prediction store, and the real one
    # has stranded rows: without this the tests assert against live data.
    monkeypatch.setattr(daily, "PREDICTIONS", tmp_path / "predictions.csv")
    monkeypatch.setattr(daily, "EUROPEAN", tmp_path / "european_matches.csv")


@pytest.fixture
def runs(monkeypatch) -> list[str]:
    """Record which pipelines were invoked, running none of them."""
    called: list[str] = []
    monkeypatch.setattr(daily, "run_step", lambda step: called.append(step.name) or 0.0)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)
    return called


def _fingerprints(monkeypatch, matches, european=(0, "")) -> None:
    """Patch `_fingerprint` so the two results tables can move independently.

    Called with no path it answers for the match *history* -- the gate's
    default since `f3c11e1` -- and with one for `european_matches.csv`, which is
    how the run distinguishes them. A list
    for `matches` is consumed one call at a time, so a test can say "the fetch
    moved the table".
    """
    sequence = iter(matches) if isinstance(matches, list) else None

    def fingerprint(path=None):
        if path is not None:
            return european
        return next(sequence) if sequence is not None else matches

    monkeypatch.setattr(daily, "_fingerprint", fingerprint)


def _args(monkeypatch, **over) -> None:
    argv = ["daily"]
    if over.get("force"):
        argv.append("--force")
    monkeypatch.setattr(sys, "argv", argv)


# --- the fingerprint ----------------------------------------------------------

def test_fingerprint_is_rows_and_latest_date(tmp_path, monkeypatch) -> None:
    path = tmp_path / "matches_all.csv"
    pd.DataFrame({"Date": ["2026-09-19", "2026-09-20"]}).to_csv(path, index=False)
    monkeypatch.setattr(daily, "HISTORY", path)
    assert daily._fingerprint() == (2, "2026-09-20")


def test_fingerprint_notices_a_correction_that_adds_no_date(tmp_path, monkeypatch) -> None:
    """A re-published old match moves the count and not the latest date.

    Keying on the date alone would call that "nothing new" and leave the
    correction unpublished until the next Monday.
    """
    path = tmp_path / "matches_all.csv"
    monkeypatch.setattr(daily, "HISTORY", path)
    pd.DataFrame({"Date": ["2026-09-20"]}).to_csv(path, index=False)
    before = daily._fingerprint()
    pd.DataFrame({"Date": ["2026-09-19", "2026-09-20"]}).to_csv(path, index=False)
    assert daily._fingerprint() != before


def test_fingerprint_survives_a_missing_or_empty_file(tmp_path, monkeypatch) -> None:
    """First ever run, or a wiped data directory: not a crash."""
    missing = tmp_path / "nope.csv"
    monkeypatch.setattr(daily, "HISTORY", missing)
    assert daily._fingerprint() == (0, "")

    empty = tmp_path / "empty.csv"
    pd.DataFrame({"Date": []}).to_csv(empty, index=False)
    monkeypatch.setattr(daily, "HISTORY", empty)
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
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert runs.index("score_upcoming") < runs.index("build_dataset")


def test_the_first_reload_comes_before_the_fetch(monkeypatch) -> None:
    """Publishing early is the point: the tabs are correct a minute in."""
    order: list[str] = []
    monkeypatch.setattr(daily, "run_step", lambda s: order.append(s.name) or 0.0)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: order.append("reload") or True)
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    daily.main()
    assert order.index("reload") < order.index("build_dataset")


def test_a_failed_fetch_keeps_the_settled_results_published(monkeypatch) -> None:
    """Settling already succeeded and published; that is a useful run, not a failed one."""
    _args(monkeypatch)
    _fingerprints(monkeypatch, (1, "2026-09-20"))
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
    _fingerprints(monkeypatch, (29143, "2026-09-20"))
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
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert runs == ["score_upcoming", "fetch_european", "build_dataset"]


def test_nothing_new_does_not_reproject(monkeypatch, runs) -> None:
    """`project_season` simulates from the results table and nothing else.

    With no new rows it would spend four minutes rewriting the file it already
    wrote, so it is the one step the short-circuit still skips.
    """
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert "project_season" not in runs


def test_new_results_run_the_whole_chain(monkeypatch, runs) -> None:
    _args(monkeypatch)
    _fingerprints(monkeypatch, [(29143, "2026-09-20"), (29170, "2026-10-10")])

    assert daily.main() == 0
    assert runs == [
        "score_upcoming", "fetch_european", "build_dataset",
        "score_upcoming", "project_season",
    ]


def test_force_reprojects_without_new_results(monkeypatch, runs) -> None:
    """`--force` is for re-simulating; settling needs no forcing."""
    _args(monkeypatch, force=True)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert runs == [
        "score_upcoming", "fetch_european", "build_dataset",
        "score_upcoming", "project_season",
    ]


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
    _fingerprints(monkeypatch, [(1, "2026-09-20"), (2, "2026-10-10")])

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
    assert called == [
        "score_upcoming", "fetch_european", "build_dataset",
        "score_upcoming", "project_season",
    ]
    # Settling published before the fetch, and that stands.
    assert reloaded == [True]


def test_a_dead_api_is_reported_not_swallowed(monkeypatch, runs) -> None:
    """The data is on disk; the run must still say it is not published."""
    _args(monkeypatch)
    _fingerprints(monkeypatch, [(1, "2026-09-20"), (2, "2026-10-10")])
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


# --- European results ---------------------------------------------------------
#
# `score_upcoming` settles a UEFA tie from `european_matches.csv` and nothing
# else, and that file was only ever written by the Monday job, where the step is
# optional and browser-driven. It had been failing: on 2026-10-14 its newest
# result was 2026-09-17, so four weeks of European predictions sat unsettled and
# the Champions League was absent from Recent results and Prediction Accuracy
# with nothing anywhere saying so.

def test_european_results_are_fetched_daily(monkeypatch, runs) -> None:
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert "fetch_european" in runs


def test_the_european_fetch_comes_after_the_first_publish(monkeypatch) -> None:
    """It drives a browser over eleven seasons, so it must not delay settling.

    This is the same mistake the domestic settle was moved in front of: a
    fragile scrape ahead of the cheap step that the three tabs are waiting on.
    """
    order: list[str] = []
    monkeypatch.setattr(daily, "run_step", lambda s: order.append(s.name) or 0.0)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: order.append("reload") or True)
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    daily.main()
    assert order.index("reload") < order.index("fetch_european")


def test_new_european_results_are_settled_at_once(monkeypatch) -> None:
    """Otherwise a Wednesday tie waits for the next day's run to show up."""
    called: list[str] = []
    monkeypatch.setattr(daily, "run_step", lambda s: called.append(s.name) or 0.0)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)
    _args(monkeypatch)
    # The European table moves; the domestic one does not.
    sequence = iter([(822, "2026-09-17"), (825, "2026-10-13"), (825, "2026-10-13")])

    def fingerprint(path=None):
        return next(sequence) if path is not None else (29143, "2026-09-20")

    monkeypatch.setattr(daily, "_fingerprint", fingerprint)

    daily.main()
    # Settled once before the European fetch and again straight after it.
    assert called.count("score_upcoming") == 2
    assert called.index("fetch_european") < called.index("score_upcoming", 1)


def test_unchanged_european_results_do_not_resettle(monkeypatch, runs) -> None:
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"), european=(822, "2026-09-17"))

    daily.main()
    assert runs.count("score_upcoming") == 1


def test_the_european_fetch_runs_even_when_football_data_is_down(monkeypatch) -> None:
    """A UEFA tie does not need football-data, so an outage must not cost it.

    football-data was unreachable all of 2026-10-14 -- every connection reset --
    and the Champions League has nothing to do with it.
    """
    called: list[str] = []

    def run(step):
        called.append(step.name)
        if step.name == "build_dataset":
            raise subprocess.CalledProcessError(1, step.name)
        return 0.0

    monkeypatch.setattr(daily, "run_step", run)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert called.index("fetch_european") < called.index("build_dataset")


def test_a_failed_european_fetch_does_not_stop_the_run(monkeypatch) -> None:
    """Reported, not fatal -- but see the next test: not silent either."""
    called: list[str] = []

    def run(step):
        called.append(step.name)
        if step.name == "fetch_european":
            raise subprocess.CalledProcessError(1, step.name)
        return 0.0

    monkeypatch.setattr(daily, "run_step", run)
    monkeypatch.setattr(daily, "reload_api", lambda *_a, **_k: True)
    _args(monkeypatch)
    _fingerprints(monkeypatch, (29143, "2026-09-20"))

    assert daily.main() == 0
    assert "build_dataset" in called


# --- the staleness report -----------------------------------------------------

def _store(tmp_path, rows: list[tuple[str, str, str, str, object]]) -> None:
    """Write a prediction store where `actual_result` may be missing."""
    pd.DataFrame({
        "League": ["E0"] * len(rows),
        "competition": [r[0] for r in rows],
        "Date": [r[1] for r in rows],
        "HomeTeam": [r[2] for r in rows],
        "AwayTeam": [r[3] for r in rows],
        "actual_result": [r[4] for r in rows],
    }).to_csv(tmp_path / "predictions.csv", index=False)


def test_a_stranded_european_tie_is_reported(tmp_path, caplog) -> None:
    """The quietest failure here, so it is made the loudest.

    A tie committed to before kickoff whose result never arrives shows up
    nowhere: not as a wrong prediction, not as a gap, just as a competition
    that stopped appearing.
    """
    old = (pd.Timestamp.today().normalize() - pd.Timedelta(days=5)).date().isoformat()
    _store(tmp_path, [("UCL", old, "Arsenal", "Lille", None)])

    with caplog.at_level("ERROR"):
        daily._report_stranded_european()
    assert "Arsenal v Lille" in caplog.text
    assert "no result" in caplog.text


def test_a_settled_tie_is_not_reported(tmp_path, caplog) -> None:
    old = (pd.Timestamp.today().normalize() - pd.Timedelta(days=5)).date().isoformat()
    _store(tmp_path, [("UCL", old, "Arsenal", "Lille", "H")])

    with caplog.at_level("ERROR"):
        daily._report_stranded_european()
    assert caplog.text == ""


def test_last_night_s_tie_is_not_reported_yet(tmp_path, caplog) -> None:
    """A Wednesday tie is settled by Thursday morning; complaining then cries wolf."""
    last_night = (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).date().isoformat()
    _store(tmp_path, [("UCL", last_night, "Arsenal", "Lille", None)])

    with caplog.at_level("ERROR"):
        daily._report_stranded_european()
    assert caplog.text == ""


def test_an_unsettled_domestic_match_is_not_a_european_problem(tmp_path, caplog) -> None:
    """Domestic rows settle from a different source and have their own reporting."""
    old = (pd.Timestamp.today().normalize() - pd.Timedelta(days=5)).date().isoformat()
    _store(tmp_path, [("domestic", old, "Arsenal", "Leeds", None)])

    with caplog.at_level("ERROR"):
        daily._report_stranded_european()
    assert caplog.text == ""


def test_duplicate_store_rows_are_counted_once(tmp_path, caplog) -> None:
    """`score_upcoming` stores a fixture once per model version.

    The real store held four rows for each of the three stranded ties, and
    reporting twelve would misstate how much football is actually missing.
    """
    old = (pd.Timestamp.today().normalize() - pd.Timedelta(days=5)).date().isoformat()
    _store(tmp_path, [("UCL", old, "Arsenal", "Lille", None)] * 4)

    assert len(daily._stranded_european()) == 1


def test_a_missing_store_is_not_an_error(tmp_path) -> None:
    """First ever run: nothing is stranded because nothing was predicted."""
    assert daily._stranded_european().empty


def test_an_unreadable_store_is_not_an_error(tmp_path) -> None:
    (tmp_path / "predictions.csv").write_text("not,a,store\nrubbish\n", encoding="utf-8")
    assert daily._stranded_european().empty


# --- the chain ----------------------------------------------------------------

def test_the_european_fetch_is_in_the_daily_chain_and_the_rebuilds_are_not() -> None:
    """The split that keeps this job short: results yes, rebuilds no."""
    names = {
        daily.FETCH.name, daily.FETCH_EUROPEAN.name,
        daily.SETTLE.name, daily.PROJECT.name,
    }
    assert "fetch_european" in names
    assert "build_players" not in names
    assert "train" not in names
    # `validate_european` and `fit_league_adjustment` stay weekly: they refine
    # the cross-league correction against a retrained model, which this job
    # does not produce.
    assert "validate_european" not in names
    assert "fit_league_adjustment" not in names


# --- the gate watches what the gated steps read -------------------------------
#
# It used to fingerprint `matches_all.csv`, which football-data writes and
# nothing downstream reads. The two normally move together, so the mistake was
# invisible until 2026-10-19: football-data was unreachable, `build_dataset`
# fell back to its cached copies, and the FBref top-up brought 27 played
# matches into the history. `matches_all.csv` stayed at 12 October while the
# history reached the 18th.


def test_the_gate_reads_the_history_not_the_file_football_data_writes(
    tmp_path, monkeypatch
) -> None:
    """The regression itself: with the two files disagreeing, the gate must
    answer for the one `score_upcoming` and `project_season` actually load.

    Both files are present and different, so a default pointing at either one
    gives a different answer -- which is what makes this a real test rather
    than an assertion about a name.

    `matches_all.csv` is written but never pointed at: the `MATCHES` constant
    that used to hold its path is gone, so there is nothing left to patch. It
    stays on disk because a regression that re-pointed the default at that
    name, by any route, would then read the real file and answer with neither
    of these numbers.
    """
    history = tmp_path / "team_match_history_all.csv"
    history.write_text(
        "Date\n2026-10-17\n2026-10-18\n2026-10-18\n", encoding="utf-8"
    )
    matches = tmp_path / "matches_all.csv"
    matches.write_text("Date\n2026-10-12\n", encoding="utf-8")

    monkeypatch.setattr(daily, "HISTORY", history)
    assert not hasattr(daily, "MATCHES"), (
        "the dead constant is back; the gate must have exactly one source"
    )

    assert daily._fingerprint() == (3, "2026-10-18"), (
        "the gate is reading matches_all.csv, which nothing downstream reads"
    )


def test_projection_runs_when_the_history_advances_without_football_data(
    monkeypatch, runs
) -> None:
    """The case that prompted this: the top-up moved the history, the fetch
    brought nothing, and the projection must still run."""
    _args(monkeypatch)
    _fingerprints(monkeypatch, [(58380, "2026-10-12"), (58434, "2026-10-18")])
    daily.main()
    assert "project_season" in runs


def test_settling_runs_even_when_nothing_moved(monkeypatch, runs) -> None:
    """Settling is not behind the gate at all, and must not be put there.

    It reads the schedule pages, which carry a score days before football-data
    publishes one, so it has work to do on a morning when the fetch brought
    nothing.
    """
    _args(monkeypatch)
    _fingerprints(monkeypatch, (58434, "2026-10-18"))
    daily.main()
    assert "score_upcoming" in runs, "settling must not depend on the gate"


def test_projection_is_skipped_when_the_history_did_not_move(monkeypatch, runs) -> None:
    _args(monkeypatch)
    _fingerprints(monkeypatch, (58434, "2026-10-18"))
    daily.main()
    assert "project_season" not in runs


def test_force_still_projects_with_no_new_results(monkeypatch, runs) -> None:
    _args(monkeypatch, force=True)
    _fingerprints(monkeypatch, (58434, "2026-10-18"))
    daily.main()
    assert "project_season" in runs


def test_a_correction_that_adds_no_date_still_opens_the_gate(monkeypatch, runs) -> None:
    """A re-scored match changes the row count and not the latest date."""
    _args(monkeypatch)
    _fingerprints(monkeypatch, [(58434, "2026-10-18"), (58436, "2026-10-18")])
    daily.main()
    assert "project_season" in runs


def test_the_european_fingerprint_is_unaffected(tmp_path, monkeypatch) -> None:
    """Passing a path explicitly still answers for that file, so the European
    check keeps reading `european_matches.csv`."""
    path = tmp_path / "european_matches.csv"
    path.write_text("Date\n2026-10-01\n2026-10-02\n", encoding="utf-8")
    assert daily._fingerprint(path) == (2, "2026-10-02")
