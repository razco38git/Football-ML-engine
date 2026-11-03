"""The retry watchdog must test reachability, not resolution.

`scripts/weekly_retry.ps1` re-runs a failed weekly once the sources are back,
and for one outage it worked. The 2026-11-02 failure was a DNS failure, so
checking `getaddrinfo` matched exactly what the pipeline saw.

The next day the same outage changed shape. understat.com resolved to
45.90.32.72 instantly and then refused every connection. The guard saw a clean
lookup, logged "every source resolves", and launched all six permitted retries
-- each spending about eighteen minutes in `build_dataset` to reach the same
dead socket -- before concluding the outage was permanent and standing down
until the following Monday. Measured afterwards: resolution 0ms and succeeding,
TCP connect timing out at 5s.

These are text assertions because the script is PowerShell and CI is Linux, so
they pin the intent rather than the behaviour -- enough to catch a revert to a
resolution-only check, which is the specific mistake that cost six runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "weekly_retry.ps1"


@pytest.fixture(scope="module")
def source() -> str:
    if not SCRIPT.exists():  # pragma: no cover - the script is part of the repo
        pytest.skip(f"{SCRIPT.name} is not present")
    return SCRIPT.read_text(encoding="utf-8-sig")


def test_the_guard_opens_a_connection(source: str) -> None:
    """A resolvable host that answers nothing is the case that broke it."""
    assert "TcpClient" in source
    assert "BeginConnect" in source


def test_the_connect_attempt_is_bounded(source: str) -> None:
    """An unbounded connect would hang the hourly task on a silent host."""
    assert "WaitOne" in source
    assert "TimeoutMs" in source


def test_a_refused_connection_is_not_counted_as_success(source: str) -> None:
    """`WaitOne` returning true only means the attempt finished."""
    assert "EndConnect" in source


def test_resolution_alone_no_longer_decides(source: str) -> None:
    assert "GetHostEntry" not in source


def test_the_log_does_not_claim_more_than_it_checked(source: str) -> None:
    """It said "every source resolves" while nothing could be reached."""
    assert "every source resolves" not in source
    assert "still unresolvable" not in source
