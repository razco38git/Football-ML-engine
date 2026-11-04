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

Then it was wrong a second time, one layer further in. On 2026-11-04
understat.com accepted TCP on 443 and returned 200 for a 26,719-byte page whose
entire visible text was "We will come back soon" -- no scripts, no `/league/`
links, no data. The connect test passed, the watchdog launched a weekly, and it
spent 28 minutes in `build_dataset` before falling back to carried-forward xG
while `build_players` died on a JSONDecodeError parsing HTML as JSON. Answering
is not serving, so a host in `$ServingMarkers` must now also return a page
carrying a marker the live site has.

The marker is deliberately a *positive* feature of a working site rather than a
match on the maintenance wording: wording can be changed at any time and would
then read as healthy, whereas a missing `/league/` cannot be faked into looking
fine. Matching what a working site has fails safe; matching what a broken one
says fails open.

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


def test_a_host_must_prove_it_serves_data(source: str) -> None:
    """A 200 from a maintenance page is the case that broke it the second time."""
    assert "Test-HostServing" in source
    assert "ServingMarkers" in source


def test_understat_is_the_host_that_must_prove_it(source: str) -> None:
    """It is the one source with no fallback, so it is the one worth checking."""
    assert "understat.com" in source
    assert "/league/" in source, "the live-site marker is gone"


def test_the_outage_wording_is_documented_but_never_matched_on(source: str) -> None:
    """The incident is worth recording; matching on it is not.

    Reworded by the operator, a wording match reads as healthy and the guard
    fails open. So the phrase may appear in a comment -- it is why this check
    exists -- but never as the operand of a comparison.
    """
    phrase = "We will come back soon"
    compared = [
        line
        for line in source.splitlines()
        if phrase in line and any(op in line for op in ("-like", "-match", "-contains", "-eq"))
    ]
    assert not compared, f"outage wording used as a test: {compared}"

    markers = [line for line in source.splitlines() if "ServingMarkers = @{" in line]
    assert markers, "the marker table is gone"
    assert "/league/" in markers[0], f"marker is not a live-site feature: {markers[0]}"


def test_not_serving_is_reported_apart_from_unreachable(source: str) -> None:
    """They need different responses: one is somebody else's outage, the other
    may mean this guard's own marker has gone stale and is holding the weekly
    back while the source is healthy."""
    assert "answering but not serving data" in source
    assert "still unreachable" in source


def test_a_failed_fetch_is_not_counted_as_serving(source: str) -> None:
    """A timeout or TLS error mid-check must not read as a healthy source."""
    assert "TimeoutSec" in source
    assert "return $false" in source
