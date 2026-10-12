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
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "data" / "processed" / ".refresh.lock"

#: A lock older than this is assumed to belong to a process that died.
#:
#: The weekly job's longest run on record is 19m 37s. An hour is comfortably
#: past that and still short enough that a crashed run does not block the next
#: morning's.
STALE_AFTER_SECONDS = 3600.0


def _is_stale(path: Path) -> bool:
    """True if the lock looks abandoned rather than held."""
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return True
    return age > STALE_AFTER_SECONDS


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
    the one that must not be skipped, and the daily job it is waiting for
    finishes in at most a few minutes.

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
                logger.warning(
                    "Removing a stale refresh lock held by %s; it is over %.0f "
                    "minutes old, so that run is gone.",
                    _describe(LOCK_PATH), STALE_AFTER_SECONDS / 60,
                )
                with contextlib.suppress(OSError):
                    LOCK_PATH.unlink()
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


def holder() -> str:
    """Who currently holds the lock, for a log line. ``""`` if nobody does."""
    return _describe(LOCK_PATH) if LOCK_PATH.exists() else ""
