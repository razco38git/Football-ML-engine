"""One refresh job at a time.

The weekly job runs at 07:00 and the daily one at 07:30, which keeps them apart
on an ordinary day. On 2026-10-12 the machine was off all morning, Windows
caught up both missed triggers when it woke, and the two started four seconds
apart -- both running `build_dataset` against the same files.

It did no harm that time. Two processes rewriting `data/processed` at once could
leave a half-written table that every later stage reads as real, which is the
failure this pipeline is most careful about everywhere else.

The two jobs want opposite things from a contended lock, and that asymmetry is
the part worth pinning: the daily run gives up, because the weekly does
everything it would have; the weekly run waits, because it is the one that must
not be skipped.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pipelines import _lock


@pytest.fixture(autouse=True)
def lock_in_tmp(tmp_path, monkeypatch) -> Path:
    path = tmp_path / ".refresh.lock"
    monkeypatch.setattr(_lock, "LOCK_PATH", path)
    return path


def test_an_uncontended_lock_is_taken(lock_in_tmp) -> None:
    with _lock.refresh_lock("weekly") as got:
        assert got
        assert lock_in_tmp.exists()
    assert not lock_in_tmp.exists()


def test_the_holder_is_recorded(lock_in_tmp) -> None:
    """A blocked run should be able to say who has it, not just that someone does."""
    with _lock.refresh_lock("weekly"):
        text = lock_in_tmp.read_text(encoding="utf-8")
        assert "weekly" in text
        assert f"pid={os.getpid()}" in text
        assert _lock.holder().startswith("weekly")


def test_a_second_holder_is_refused(lock_in_tmp) -> None:
    with _lock.refresh_lock("weekly") as first:
        assert first
        with _lock.refresh_lock("daily") as second:
            assert second is False


def test_refusal_does_not_release_the_other_run_s_lock(lock_in_tmp) -> None:
    """The failing acquirer must not delete a lock it never held."""
    with _lock.refresh_lock("weekly"):
        with _lock.refresh_lock("daily") as second:
            assert second is False
        # Still held by the outer block.
        assert lock_in_tmp.exists()
        assert _lock.holder().startswith("weekly")


def test_the_lock_is_released_after_an_exception(lock_in_tmp) -> None:
    """A crashed run must not block tomorrow's."""
    with pytest.raises(RuntimeError), _lock.refresh_lock("weekly") as got:
        assert got
        raise RuntimeError("build_dataset exploded")
    assert not lock_in_tmp.exists()


def test_a_stale_lock_is_broken(lock_in_tmp, monkeypatch) -> None:
    """A reboot leaves a file behind; it must not block every future run."""
    lock_in_tmp.write_text("weekly pid=1 since=long-ago", encoding="utf-8")
    old = time.time() - (_lock.STALE_AFTER_SECONDS + 60)
    os.utime(lock_in_tmp, (old, old))

    with _lock.refresh_lock("daily") as got:
        assert got
        assert "daily" in lock_in_tmp.read_text(encoding="utf-8")


def test_a_fresh_lock_is_not_broken(lock_in_tmp) -> None:
    """The stale check must not fire on a run that is merely slow.

    The holder is this process because "held" now means *a live owner*, not a
    recent timestamp: a lock naming a dead pid is released at once rather than
    an hour later, which is the point of the change.
    """
    lock_in_tmp.write_text(f"weekly pid={os.getpid()} since=now", encoding="utf-8")
    with _lock.refresh_lock("daily") as got:
        assert got is False
    # Untouched.
    assert "weekly" in lock_in_tmp.read_text(encoding="utf-8")


def test_waiting_gives_up_at_the_deadline(lock_in_tmp) -> None:
    """The weekly run waits, but not forever."""
    lock_in_tmp.write_text(f"daily pid={os.getpid()} since=now", encoding="utf-8")
    started = time.monotonic()
    with _lock.refresh_lock("weekly", wait_seconds=0.1) as got:
        assert got is False
    assert time.monotonic() - started < 30


def test_holder_is_empty_when_free(lock_in_tmp) -> None:
    assert _lock.holder() == ""


def test_the_two_jobs_disagree_on_purpose() -> None:
    """Pins the asymmetry: daily gives up at once, weekly waits.

    If these ever match, one of two things has broken -- either the daily job
    now blocks a visitor-facing refresh behind a twenty-minute rebuild, or the
    weekly job silently skips a week because a six-minute job was running.
    """
    import inspect

    from pipelines import daily, weekly

    daily_src = inspect.getsource(daily.main)
    assert 'refresh_lock("daily")' in daily_src
    assert "wait_seconds" not in daily_src

    weekly_src = inspect.getsource(weekly.main)
    assert 'refresh_lock("weekly"' in weekly_src
    assert "wait_seconds=LOCK_WAIT_SECONDS" in weekly_src


# --- a lock is abandoned when its owner is gone, not when it is old ----------
#
# The original rule was age alone, on the stated assumption that "the weekly
# job's longest run on record is 19m 37s". Three of the four runs before
# 2026-10-19 exceeded the hour; that day's run took 356 minutes, and for most
# of it a live lock looked abandoned to any daily run that fired.


def _lock_text(pid: int, since: datetime, owner: str = "weekly") -> str:
    return f"{owner} pid={pid} since={since.isoformat()}"


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_a_live_owner_holds_the_lock_however_old_it_is(lock_in_tmp) -> None:
    """The bug, in one test. A four-hour run is now ordinary."""
    lock_in_tmp.write_text(_lock_text(os.getpid(), datetime.now(UTC)), encoding="utf-8")
    _age(lock_in_tmp, 4 * 3600)
    assert not _lock._is_stale(lock_in_tmp)


def test_a_dead_owner_releases_the_lock_however_fresh_it_is(lock_in_tmp) -> None:
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    time.sleep(0.5)
    lock_in_tmp.write_text(_lock_text(done.pid, datetime.now(UTC)), encoding="utf-8")
    assert _lock._is_stale(lock_in_tmp)


def test_a_reused_pid_is_not_the_owner(lock_in_tmp) -> None:
    """Liveness alone does not prove ownership: the id can be recycled.

    This process is alive and holds the recorded pid, but it started long after
    a lock claiming to be two days old, so it cannot be that lock's owner.
    """
    taken = datetime.now(UTC) - timedelta(days=2)
    lock_in_tmp.write_text(_lock_text(os.getpid(), taken), encoding="utf-8")
    assert _lock._is_stale(lock_in_tmp)


def test_unknown_liveness_keeps_the_lock_and_never_ages_out(
    lock_in_tmp, monkeypatch, caplog
) -> None:
    """The asymmetry that decides every ambiguous case.

    Wrongly breaking a lock lets two jobs write the same non-atomic files;
    wrongly keeping one skips a run the next trigger retries.
    """
    monkeypatch.setattr(_lock, "_process", lambda pid: (None, None))
    lock_in_tmp.write_text(_lock_text(4242, datetime.now(UTC)), encoding="utf-8")
    _age(lock_in_tmp, 10 * 3600)
    with caplog.at_level("WARNING"):
        assert not _lock._is_stale(lock_in_tmp)
    assert any("Cannot determine" in r.message for r in caplog.records)


def test_access_denied_counts_as_running(lock_in_tmp, monkeypatch) -> None:
    """A process we may not query is still a process."""
    monkeypatch.setattr(_lock, "_process", lambda pid: (True, None))
    lock_in_tmp.write_text(_lock_text(4, datetime.now(UTC)), encoding="utf-8")
    _age(lock_in_tmp, 6 * 3600)
    assert not _lock._is_stale(lock_in_tmp)


def test_an_unidentifiable_lock_is_the_only_age_based_case(lock_in_tmp) -> None:
    lock_in_tmp.write_text("weekly since=whenever", encoding="utf-8")
    _age(lock_in_tmp, 2 * 3600)
    assert _lock._is_stale(lock_in_tmp)
    lock_in_tmp.write_text("weekly", encoding="utf-8")
    assert not _lock._is_stale(lock_in_tmp)


def test_a_naive_since_is_not_trusted_as_identity(lock_in_tmp) -> None:
    """It would be read as local time and compared against a UTC creation
    time -- the silently-wrong comparison this check exists to avoid."""
    text = f"weekly pid={os.getpid()} since={datetime.now().isoformat()}"
    lock_in_tmp.write_text(text, encoding="utf-8")
    assert _lock._recorded(text)[1] is None
    assert not _lock._is_stale(lock_in_tmp)


# --- breaking a lock has to be exclusive -------------------------------------


def test_recovery_does_not_unlink_a_live_lock(lock_in_tmp, monkeypatch) -> None:
    """The race the recovery lock exists for.

    Two runs could both judge a lock abandoned, the first unlink it and
    acquire, and the second then unlink *the first's live lock* and acquire as
    well -- both writing `data/processed` at once.
    """
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    lock_in_tmp.write_text(_lock_text(os.getpid(), datetime.now(UTC)), encoding="utf-8")
    _lock._recover()
    assert lock_in_tmp.exists(), "recovery removed a lock whose owner is alive"
    assert not _lock.BREAK_PATH.exists(), "recovery lock was not released"


def test_recovery_removes_a_genuinely_abandoned_lock(lock_in_tmp, monkeypatch) -> None:
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    time.sleep(0.5)
    lock_in_tmp.write_text(_lock_text(done.pid, datetime.now(UTC)), encoding="utf-8")
    _lock._recover()
    assert not lock_in_tmp.exists()


def test_a_recovery_lock_owned_by_someone_else_is_not_stolen(lock_in_tmp, monkeypatch) -> None:
    """A *live* owner, not merely a recent one.

    This used to write `pid=1` and rely on the file being young, which made the
    guarantee probabilistic: a process suspended past the 30-second bound inside
    the critical section would have had its lock taken, and two processes could
    then both break the main lock. The recovery lock is now decided by its
    owner's liveness, so the fixture has to supply a real one.
    """
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    lock_in_tmp.write_text(_lock_text(done.pid, datetime.now(UTC)), encoding="utf-8")

    owner = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        _lock.BREAK_PATH.write_text(f"pid={owner.pid}", encoding="utf-8")
        # Old enough that the age rule alone would have cleared it.
        _age(_lock.BREAK_PATH, _lock.BREAK_STALE_AFTER_SECONDS + 60)

        _lock._recover()

        assert lock_in_tmp.exists(), "broke the lock while another run held recovery"
        assert _lock.BREAK_PATH.exists(), "stole a recovery lock somebody else owned"
    finally:
        owner.kill()
        owner.wait()


def test_a_recovery_lock_whose_owner_exited_is_cleared(lock_in_tmp, monkeypatch) -> None:
    """The other half: liveness must not wedge recovery when the owner is gone,
    and it must not wait out the age bound to notice."""
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    lock_in_tmp.write_text(_lock_text(done.pid, datetime.now(UTC)), encoding="utf-8")

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    _lock.BREAK_PATH.write_text(f"pid={gone.pid}", encoding="utf-8")

    _lock._recover()
    assert not _lock.BREAK_PATH.exists(), "a dead owner's recovery lock survived"


def test_a_recovery_lock_with_no_pid_yet_falls_back_to_age(lock_in_tmp, monkeypatch) -> None:
    """`_recover` creates the file with O_EXCL and writes the pid after, so an
    empty one is a real state -- and it must read as held, not as abandoned."""
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    _lock.BREAK_PATH.write_text("", encoding="utf-8")
    assert not _lock._break_is_stale(_lock.BREAK_PATH)

    _age(_lock.BREAK_PATH, _lock.BREAK_STALE_AFTER_SECONDS + 60)
    assert _lock._break_is_stale(_lock.BREAK_PATH)


def test_an_abandoned_recovery_lock_cannot_wedge_recovery(lock_in_tmp, monkeypatch) -> None:
    """A process dying inside the critical section must not block it forever."""
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    _lock.BREAK_PATH.write_text("pid=1", encoding="utf-8")
    _age(_lock.BREAK_PATH, _lock.BREAK_STALE_AFTER_SECONDS + 60)
    _lock._recover()
    assert not _lock.BREAK_PATH.exists()


def test_a_run_recovers_an_abandoned_lock_and_acquires(lock_in_tmp, monkeypatch) -> None:
    """Recovery after an unexpected exit, end to end."""
    monkeypatch.setattr(_lock, "BREAK_PATH", lock_in_tmp.with_name(".refresh.break"))
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait()
    time.sleep(0.5)
    lock_in_tmp.write_text(_lock_text(done.pid, datetime.now(UTC)), encoding="utf-8")

    with _lock.refresh_lock("weekly") as got:
        assert got
        assert f"pid={os.getpid()}" in lock_in_tmp.read_text(encoding="utf-8")
    assert not lock_in_tmp.exists()


def test_the_probe_never_terminates_what_it_inspects() -> None:
    """`os.kill(pid, 0)` is not used, and must not be reintroduced.

    CPython routes any Windows signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT
    to TerminateProcess, so the obvious liveness check could kill the very job
    it is asked about.
    """
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
    try:
        time.sleep(1.0)
        running, _ = _lock._process(child.pid)
        assert running is True
        time.sleep(1.0)
        assert child.poll() is None, "the liveness probe killed the process"
    finally:
        child.kill()
        child.wait()


# --- a hung owner is kept, but not kept quietly ------------------------------
#
# Deciding staleness by liveness removed the risk of killing a long run and
# left the opposite one: a wedged job holds the lock for as long as its process
# survives, every later run declines, and the only symptom is a site that stops
# updating. Nothing is broken at this age -- breaking it is the dangerous move
# -- so the whole fix is that it says something.


def test_a_live_owner_past_every_recorded_runtime_is_reported(lock_in_tmp, caplog) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock_in_tmp.write_text(_lock_text(child.pid, datetime.now(UTC)), encoding="utf-8")
        _age(lock_in_tmp, _lock.REPORT_AFTER_SECONDS + 60)

        with caplog.at_level(logging.WARNING, logger="pipelines._lock"):
            stale = _lock._is_stale(lock_in_tmp)

        assert stale is False, "a live owner's lock was broken on age"
        assert "wedged" in caplog.text
        assert str(child.pid) in caplog.text
    finally:
        child.kill()
        child.wait()


def test_a_live_owner_inside_the_window_is_not_reported(lock_in_tmp, caplog) -> None:
    """A 229-minute weekly is a real run, not a fault. Warning about it would
    train whoever reads these logs to ignore them."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock_in_tmp.write_text(_lock_text(child.pid, datetime.now(UTC)), encoding="utf-8")
        _age(lock_in_tmp, 229 * 60)

        with caplog.at_level(logging.WARNING, logger="pipelines._lock"):
            assert _lock._is_stale(lock_in_tmp) is False

        assert caplog.text == ""
    finally:
        child.kill()
        child.wait()


def test_the_report_threshold_is_past_the_longest_run_on_record() -> None:
    """The constant and its justification cannot drift apart: 5h 56m on
    2026-10-19 is the longest weekly recorded, and it must not warn."""
    assert _lock.REPORT_AFTER_SECONDS > 356 * 60
