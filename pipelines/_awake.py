"""Keep the machine from idle-sleeping while a refresh runs.

Both scheduled jobs are long -- the weekly's last clean run took 92 minutes,
and on 2026-11-03 it took longer still because five Understat connect timeouts
cost 18 minutes in `build_dataset` alone. They run unattended on a desktop
someone also uses, and Windows' idle timer does not care that a background
process is mid-rebuild.

What that costs is not a slow run but a *stalled* one. On 2026-11-03 the
machine slept at about 11:36 with `validate_european` in flight and woke at
14:53: the run survived, but it sat suspended for three and a quarter hours
holding the refresh lock, so the scheduled weekly that fired at 11:26 found
the lock held and declined to start. One sleep cost a run and blocked its
replacement.

So each job asks Windows to keep the system up for as long as it holds the
lock, the way a video player or an installer does. The request is per
*thread* and is released on exit, so nothing is left behind if the job
crashes -- and because it is scoped to the job rather than set in the power
plan, a machine that is doing nothing still sleeps normally.

**What this does not do, stated because the gap matters.** It prevents *idle*
sleep only. Closing the lid still sleeps, and so does choosing Sleep from the
menu; `ES_AWAYMODE_REQUIRED` would cover some of that and is deliberately not
used, because on laptops it is unreliable and it leaves the machine in a state
that looks asleep while it is not. The display is left alone too -- a 07:00
job has no business lighting the screen -- which is why only
``ES_SYSTEM_REQUIRED`` is set.

Every other platform gets a no-op, so the pipelines import and run unchanged
on the Linux box CI uses.

It lives beside `_lock.py` rather than in `footballml` because it is
orchestration and not domain logic: `weekly.py` is a pure subprocess driver
that puts nothing from `src` on its path, and giving it one so it could import
a dozen lines of `ctypes` would be the wrong trade.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from collections.abc import Iterator

logger = logging.getLogger(__name__)

#: `SetThreadExecutionState` flags. See the Windows API docs; the values are
#: fixed and small enough that naming them beats a dependency.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001


def _set_state(flags: int) -> int | None:
    """Call `SetThreadExecutionState`, or return `None` where it does not exist.

    Returns the previous state on success, `None` on any failure -- including a
    non-Windows platform, a missing `kernel32`, or the call itself refusing,
    which it signals by returning 0.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        previous = ctypes.windll.kernel32.SetThreadExecutionState(flags)  # type: ignore[attr-defined]
    except (AttributeError, OSError) as exc:
        logger.debug("SetThreadExecutionState unavailable: %s", exc)
        return None
    return int(previous) or None


@contextlib.contextmanager
def keep_awake(what: str) -> Iterator[bool]:
    """Hold off idle sleep for the duration of the block.

    Args:
        what: The job's name, for the log line.

    Yields:
        Whether the request was granted. A refusal is logged and the block
        runs anyway: a job that will not start because the machine might
        sleep is worse than one that might be interrupted.
    """
    held = _set_state(_ES_CONTINUOUS | _ES_SYSTEM_REQUIRED) is not None
    if held:
        logger.info("Holding off idle sleep for the %s run", what)
    else:
        logger.info(
            "Could not ask the system to stay awake (%s); the %s run may be "
            "suspended if the machine idles",
            sys.platform,
            what,
        )
    try:
        yield held
    finally:
        if held:
            # `ES_CONTINUOUS` alone clears the requirement without asserting a
            # new one, which is how the API says to let go.
            _set_state(_ES_CONTINUOUS)
            logger.info("Released the sleep hold after the %s run", what)
