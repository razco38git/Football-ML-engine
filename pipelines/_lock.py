"""One writer at a time across the refresh jobs.

The weekly job runs Mondays at 07:00 and the daily one at 07:30, half an hour
apart so they cannot overlap. That holds on an ordinary day and fails on the
first unusual one: on 2026-10-12 the machine was off all morning, Windows
caught up both missed triggers when it woke, and **both started within four
seconds of each other** and ran `build_dataset` against the same files
concurrently.

It did no harm that time -- both fetched the same rows and the daily job's
"nothing new" check stopped it -- but two processes rewriting
`data/processed/*.csv` at once is the one way this pipeline could leave a
half-written table that every later stage would read as real. A schedule offset
is not a lock, and this is.

Deliberately a file in `data/processed/` rather than a library: it needs no
dependency, it is visible to anyone wondering why a run exited early, and it
survives nothing -- a reboot leaves a stale file, which `_is_stale` is for.

**Whether a lock is abandoned is decided by its owner, not its age.** The
original rule was age alone and it was wrong: see `_is_stale` for the table and
`_recover` for why breaking one has to be exclusive.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "data" / "processed" / ".refresh.lock"

#: Held for microseconds while one process decides whether to break the main
#: lock. Its own staleness bound is short because nothing legitimate holds it
#: for long -- see `refresh_lock`.
BREAK_PATH = ROOT / "data" / "processed" / ".refresh.break"
BREAK_STALE_AFTER_SECONDS = 30.0

#: Only reached for a lock with **no parseable pid** -- a legacy or corrupt
#: file we cannot identify. Every identifiable lock is decided by its owner,
#: not its age.
#:
#: The old rule was age alone, on the stated assumption that "the weekly job's
#: longest run on record is 19m 37s". Three of the four runs before
#: 2026-10-19 exceeded the hour, and that day a 244-minute-old lock was alive
#: and writing while any daily run would have deleted it and written the same
#: non-atomic files concurrently.
STALE_AFTER_SECONDS = 3600.0

#: When a lock held by a *live* owner starts getting reported as suspicious.
#:
#: Not a staleness bound and deliberately not one: nothing is broken at this
#: age. Deciding by liveness removed the risk of killing a long run, and left
#: the opposite problem -- a genuinely hung job holds the lock until somebody
#: notices, and nothing was telling anybody. Eight hours is comfortably past
#: the longest run on record (5h 56m on 2026-10-19) and short enough that one
#: missed weekly surfaces before the next.
REPORT_AFTER_SECONDS = 8 * 3600.0

_STILL_ACTIVE = 259
_ERROR_INVALID_PARAMETER = 87
_ERROR_ACCESS_DENIED = 5
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_FILETIME_EPOCH_OFFSET = 11_644_473_600.0

#: Clock granularity between `datetime.now(UTC)` and a process creation time.
_IDENTITY_TOLERANCE_SECONDS = 1.0


def _recorded(text: str) -> tuple[int | None, float | None]:
    """The pid and lock-creation epoch a lock file records.

    A naive ``since=`` is returned as ``None`` rather than guessed at: it would
    be interpreted as local time and compared against a UTC process creation
    time, which is exactly the kind of silently-wrong comparison this check
    exists to avoid.
    """
    pid = since = None
    for field in text.split():
        if field.startswith("pid="):
            with contextlib.suppress(ValueError):
                pid = int(field[4:])
        elif field.startswith("since="):
            with contextlib.suppress(ValueError):
                stamp = datetime.fromisoformat(field[len("since="):])
                if stamp.tzinfo is not None:
                    since = stamp.timestamp()
    return pid, since


def _windows_process(pid: int) -> tuple[bool | None, float | None]:
    """``(running, created_at)``; ``None`` wherever it cannot be determined.

    Deliberately not ``os.kill(pid, 0)``. CPython routes any Windows signal
    other than CTRL_C_EVENT/CTRL_BREAK_EVENT to ``TerminateProcess``, so it
    could kill the very job it is asked about -- and measured on Python 3.13 it
    does not raise for a dead pid either, so it cannot tell the two apart.
    ``OpenProcess`` is read-only and does distinguish them.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        err = ctypes.get_last_error()
        if err == _ERROR_INVALID_PARAMETER:
            return False, None          # no process has this id
        if err == _ERROR_ACCESS_DENIED:
            return True, None           # it exists; we may not query it
        return None, None               # indeterminate: assume nothing
    try:
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None, None
        if code.value != _STILL_ACTIVE:
            return False, None
        creation, exited, kern, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation), ctypes.byref(exited),
            ctypes.byref(kern), ctypes.byref(user),
        ):
            return True, None
        ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
        return True, ticks / 1e7 - _FILETIME_EPOCH_OFFSET
    finally:
        # Closed even when GetExitCodeProcess or GetProcessTimes fails.
        kernel32.CloseHandle(handle)


def _process(pid: int | None) -> tuple[bool | None, float | None]:
    """Liveness and, where the platform can say, process creation time."""
    if pid is None or pid <= 0:
        return None, None
    if sys.platform == "win32":
        try:
            return _windows_process(pid)
        except Exception:  # pragma: no cover - ctypes unavailable or refused
            return None, None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False, None
    except PermissionError:
        return True, None               # exists, owned by another user
    except OSError:  # pragma: no cover - platform-specific
        return None, None
    return True, None                   # POSIX: liveness without identity


def _is_stale(path: Path) -> bool:
    """True if the lock is abandoned rather than held.

    =============================== ==========================================
    lock state                      decision
    =============================== ==========================================
    no parseable ``pid=``           age (the only age-based case)
    pid confirmed exited            abandoned
    pid running, started *after*    abandoned -- the id was reused
    pid running, started *before*   **held, at any age** (reported past 8h)
    liveness or start time unknown  **held**, and logged
    =============================== ==========================================

    When in doubt the lock is kept. Wrongly breaking one lets two jobs write
    the same non-atomic files; wrongly keeping one skips a run that the next
    trigger retries. The costs are not symmetric.
    """
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return True

    pid, since = _recorded(_describe(path))
    if pid is None:
        return age > STALE_AFTER_SECONDS

    running, created = _process(pid)
    if running is False:
        return True
    if running is None:
        logger.warning(
            "Cannot determine whether %s is still running; keeping the lock. "
            "Delete %s by hand if that run is known to be gone.",
            _describe(path), path,
        )
        return False
    if (
        created is not None
        and since is not None
        and created > since + _IDENTITY_TOLERANCE_SECONDS
    ):
        logger.warning(
            "The refresh lock names pid %d, but that pid started after the "
            "lock was taken, so it is a different process. Treating the lock "
            "as abandoned.", pid,
        )
        return True

    # Held, legitimately. But "held by a live pid" is also what a *hung* job
    # looks like, and since the age test was removed nothing breaks that lock
    # -- correctly, because breaking it is the dangerous move. The failure mode
    # it leaves is silence: every later run declines, forever, and the only
    # symptom is a site that stops updating. So say so.
    if age > REPORT_AFTER_SECONDS:
        logger.warning(
            "The refresh lock has been held by %s for %.1f hours, which is "
            "past any run on record (the longest is 5h 56m, 2026-10-19). The "
            "owner is still alive, so it is being left alone. If that process "
            "is wedged, stop pid %d and delete %s.",
            _describe(LOCK_PATH), age / 3600.0, pid, path,
        )
    return False


def _break_is_stale(path: Path) -> bool:
    """True if the recovery lock was abandoned mid-recovery.

    Age alone used to decide this, on the reasoning that nothing legitimate
    holds the recovery lock for more than microseconds. That is true and still
    is, but it made the rule probabilistic: a process suspended inside the
    critical section for 30 seconds -- a stop-the-world GC pause, a laptop
    sleeping, a VM migrating -- would have its lock taken from under it, and
    then two processes could both break the main lock, which is the one thing
    this module exists to prevent.

    `_recover` writes its pid, so the same liveness test the main lock uses
    applies here. Age is kept only as the backstop for a file with no parseable
    pid, which includes the microsecond window between `O_EXCL` creating it and
    the write landing.
    """
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return True

    pid, _ = _recorded(_describe(path))
    if pid is None:
        # Either written before this rule existed, or created-but-not-yet
        # written. Both resolve in well under the bound.
        return age > BREAK_STALE_AFTER_SECONDS

    running, _ = _process(pid)
    if running is False:
        return True
    if running is None:
        # Indeterminate. Age is the only evidence left, and the asymmetry is
        # the same as the main lock's: waiting costs a skipped run that the
        # next trigger retries.
        return age > BREAK_STALE_AFTER_SECONDS
    return False


def _describe(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


@contextlib.contextmanager
def refresh_lock(owner: str, wait_seconds: float = 0.0) -> Iterator[bool]:
    """Hold the shared refresh lock for the duration of the block.

    Yields ``True`` if the lock was taken and ``False`` if somebody else holds
    it, so the caller decides what that means -- the two jobs want opposite
    things. The daily run gives up immediately, because the weekly job does
    everything it would have done and more. The weekly run waits, because it is
    the one that must not be skipped.

    Args:
        owner: Name recorded in the lock file, so a blocked run can say who has
            it rather than just that something does.
        wait_seconds: How long to keep trying before giving up.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    held = False

    while True:
        try:
            # O_EXCL is the whole mechanism: creating the file *is* taking the
            # lock, and two processes cannot both succeed.
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if _is_stale(LOCK_PATH):
                _recover()
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(2.0)
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{owner} pid={os.getpid()} since={datetime.now(UTC).isoformat()}")
        held = True
        break

    try:
        yield held
    finally:
        if held:
            with contextlib.suppress(OSError):
                LOCK_PATH.unlink()


def _recover() -> None:
    """Break an abandoned lock, but only one process at a time.

    The unlink used to follow the staleness check directly, and nothing tied
    the two together: two runs could both judge a lock abandoned, the first
    unlink it and acquire, and the second then unlink *the first's live lock*
    and acquire as well. Both would then write `data/processed` at once, which
    is the single thing this module exists to prevent.

    The recovery lock makes breaking exclusive, and the **re-check inside it**
    is what actually closes the race: by the time the second process gets here,
    the first's lock is live and no longer stale, so it is left alone. If the
    first has not created it yet the file is simply absent, the unlink is a
    no-op, and both fall back to racing on ``O_EXCL`` -- which has exactly one
    winner by construction.
    """
    try:
        fd = os.open(BREAK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # Somebody else is recovering. Only clear it if it was abandoned
        # mid-recovery; otherwise wait for them rather than racing.
        if _break_is_stale(BREAK_PATH):
            logger.warning("Clearing an abandoned recovery lock at %s", BREAK_PATH)
            with contextlib.suppress(OSError):
                BREAK_PATH.unlink()
        else:
            time.sleep(0.2)
        return

    try:
        os.write(fd, f"pid={os.getpid()}".encode())
        if _is_stale(LOCK_PATH):
            logger.warning(
                "Removing an abandoned refresh lock held by %s.",
                _describe(LOCK_PATH),
            )
            with contextlib.suppress(OSError):
                LOCK_PATH.unlink()
    finally:
        os.close(fd)
        with contextlib.suppress(OSError):
            BREAK_PATH.unlink()


def holder() -> str:
    """Who currently holds the lock, for a log line. ``""`` if nobody does."""
    return _describe(LOCK_PATH) if LOCK_PATH.exists() else ""
