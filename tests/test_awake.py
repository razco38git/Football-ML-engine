"""The sleep hold the refresh jobs take while they run.

A wake lock is easy to get wrong in one direction that matters: leaving it
asserted. A job that crashes without releasing it would keep a desktop awake
until the next reboot, which is worse than the stall it exists to prevent.
"""

from __future__ import annotations

import pytest

from pipelines import _awake
from pipelines._awake import _ES_CONTINUOUS, _ES_SYSTEM_REQUIRED, keep_awake


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record every flag set, standing in for the Windows API."""
    seen: list[int] = []

    def record(flags: int) -> int | None:
        seen.append(flags)
        return 0x1

    monkeypatch.setattr(_awake, "_set_state", record)
    return seen


def test_it_asks_to_keep_the_system_up_and_leaves_the_display_alone(
    calls: list[int],
) -> None:
    """A 07:00 job has no business lighting the screen."""
    with keep_awake("weekly") as held:
        assert held

    assert calls[0] == _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    assert not calls[0] & 0x00000002, "ES_DISPLAY_REQUIRED must not be set"


def test_the_hold_is_released_on_the_way_out(calls: list[int]) -> None:
    with keep_awake("daily"):
        pass

    assert calls[-1] == _ES_CONTINUOUS, "ES_CONTINUOUS alone clears the requirement"


def test_a_crash_still_releases_it(calls: list[int]) -> None:
    """The failure that would outlive the job and keep a desktop awake."""
    with pytest.raises(RuntimeError), keep_awake("weekly"):
        raise RuntimeError("build_dataset exploded")

    assert calls[-1] == _ES_CONTINUOUS


def test_a_refusal_does_not_stop_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job that will not run because the machine might sleep is worse."""
    monkeypatch.setattr(_awake, "_set_state", lambda flags: None)

    ran = False
    with keep_awake("weekly") as held:
        ran = True

    assert ran
    assert held is False


def test_nothing_is_released_that_was_never_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clearing a hold we never took could cancel somebody else's."""
    seen: list[int] = []

    def refuse(flags: int) -> int | None:
        seen.append(flags)
        return None

    monkeypatch.setattr(_awake, "_set_state", refuse)

    with keep_awake("daily"):
        pass

    assert len(seen) == 1, "only the failed request, no release"


def test_it_is_a_no_op_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """CI runs on Linux and must import and run this unchanged."""
    monkeypatch.setattr(_awake.sys, "platform", "linux")

    assert _awake._set_state(_ES_CONTINUOUS) is None


def test_a_missing_kernel32_is_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wine, a stripped image, a future Python without `windll`."""
    monkeypatch.setattr(_awake.sys, "platform", "win32")

    class NoWindll:
        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    monkeypatch.setitem(__import__("sys").modules, "ctypes", NoWindll())

    assert _awake._set_state(_ES_CONTINUOUS) is None
