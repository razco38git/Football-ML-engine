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

import os
import time
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
    """The stale check must not fire on a run that is merely slow."""
    lock_in_tmp.write_text("weekly pid=1 since=now", encoding="utf-8")
    with _lock.refresh_lock("daily") as got:
        assert got is False
    # Untouched.
    assert "weekly" in lock_in_tmp.read_text(encoding="utf-8")


def test_waiting_gives_up_at_the_deadline(lock_in_tmp) -> None:
    """The weekly run waits, but not forever."""
    lock_in_tmp.write_text("daily pid=1 since=now", encoding="utf-8")
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
