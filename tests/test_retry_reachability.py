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

Then a third time, in the other direction. On 2026-11-22 fbref.com accepted TCP
on 443 and answered **403** to every request -- bot protection, not an outage.
A host with no entry in `$ServingMarkers` skipped the serving check entirely,
so that 403 read as perfectly healthy. Every host now has to return 200, and a
marker is an additional test rather than the only one.

The same day added the second decision these also cover: nudging the *daily*
when results alone are behind. Three things about it are worth pinning, because
each was got wrong first.

football-data.co.uk is deliberately **not** in `$RequiredHosts`. That list
gates the weekly, whose blocked step is `build_players`, which needs Understat
and never touches football-data -- so gating the weekly on it would hold a
rebuild back for an outage that cannot affect it.

The daily's trigger is the *data*, not an exit code. The daily's steps are
optional by design, so it returns 0 having published nothing new: on 2026-11-22
it exited 0 with Friday's three results and none of Saturday's twenty-six. Only
an unsettled fixture whose date has passed says so.

And that count is windowed. The store held 39 unsettled played fixtures and 13
were abandoned or rescheduled matches from September and October, which never
settle. Counting those, the total can never reach zero, the nudge counter never
clears, and three launches switch the nudge off for good.

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


def test_every_host_must_answer_200_not_merely_connect(source: str) -> None:
    """fbref.com's 403 accepted TCP and served nothing.

    The serving check must run for every host, not only those carrying a
    marker, because the status code is what catches bot protection.
    """
    assert "StatusCode -ne 200" in source
    assert "if ($marker -and" not in source, "a host without a marker skips the status check"


def test_a_results_outage_does_not_gate_the_weekly(source: str) -> None:
    """`build_players` is what blocks the weekly, and it needs Understat."""
    required = [line for line in source.splitlines() if "$RequiredHosts = @(" in line]
    assert required, "the weekly's host list is gone"
    assert "football-data" not in required[0], f"results host gates the weekly: {required[0]}"


def test_the_results_sources_are_watched_in_their_own_list(source: str) -> None:
    result_hosts = [line for line in source.splitlines() if "$ResultHosts = @(" in line]
    assert result_hosts, "nothing watches the results sources"
    assert "football-data.co.uk" in result_hosts[0]


def test_one_results_source_is_enough(source: str) -> None:
    """Either source can supply a result.

    Requiring both would let an FBref outage block a refresh football-data
    could serve -- the same mistake as gating the weekly on football-data.
    """
    assert "$usable" in source
    assert "$usable.Count -eq 0" in source, "the daily waits for every results host, not any"


def test_the_daily_trigger_is_the_data_not_an_exit_code(source: str) -> None:
    """The daily exits 0 while publishing stale results, by design."""
    assert "Get-UnsettledCount" in source
    assert "actual_home_goals" in source


def test_being_unable_to_tell_is_not_the_same_as_nothing_to_do(source: str) -> None:
    """A missing store, or a renamed Date column, must not read as "current"."""
    assert "return -1" in source
    assert "$unsettled -lt 0" in source, "a negative count is treated as a real count"


def test_the_unsettled_count_is_windowed(source: str) -> None:
    """13 fixtures from September and October never settle; unbounded, the
    counter can never clear and the nudge dies after its cap."""
    assert "$ResultWindowDays" in source
    window = [line for line in source.splitlines() if "$ResultWindowDays = " in line]
    assert window, "the window has no default"


def test_the_daily_nudge_is_capped_separately(source: str) -> None:
    """Its trigger is softer than the weekly's: a late result can mean the
    source has not published yet rather than that we failed to ask."""
    assert "$MaxDailyAttempts" in source


def test_nothing_starts_while_a_job_is_already_writing(source: str) -> None:
    """Two processes in `data/processed` is the one way this could corrupt
    something, so the check has to precede both decisions rather than sit
    inside one of them."""
    running = source.index("is already running")
    assert running < source.index("decision 1"), "the weekly decision runs before the check"
    assert running < source.index("decision 2"), "the daily decision runs before the check"


def test_the_daily_decision_is_reachable_while_the_weekly_is_owed(source: str) -> None:
    """The case that prompted it: on 2026-11-22 the weekly was owed and blocked
    on Understat, so a daily nudge nested inside "the weekly succeeded" would
    never have run at all. It belongs after the weekly declines to start."""
    assert source.index("$startWeekly) {") < source.index("decision 2")


def test_fbref_is_judged_on_a_connect_alone(source: str) -> None:
    """A plain GET to fbref.com is always 403, and the pipeline never makes one.

    soccerdata reads it through a browser stack and got real pages on the same
    mornings the 403 was seen. Held to the 200 test it could never pass, which
    disabled the weekly retry -- FBref is in `$RequiredHosts` -- and removed the
    one results source that was working.
    """
    exempt = [line for line in source.splitlines() if "$ReachabilityOnly = @(" in line]
    assert exempt, "the exemption list is gone"
    assert "fbref.com" in exempt[0]
    assert "$SkipStatus -contains $hostname" in source, "the exemption is declared but not applied"


def test_only_browser_read_hosts_may_skip_the_status_check(source: str) -> None:
    """The exemption exists for a host a plain GET cannot judge. Understat and
    football-data are read over plain HTTP, so a 403 from either *is* what a
    fetch would get, and exempting them would reopen the case it was added for."""
    exempt = [line for line in source.splitlines() if "$ReachabilityOnly = @(" in line][0]
    assert "understat.com" not in exempt
    assert "football-data.co.uk" not in exempt


def test_both_decisions_apply_the_exemption(source: str) -> None:
    """Exempting FBref from one decision and not the other would leave it
    blocking whichever was missed."""
    assert source.count("-SkipStatus $ReachabilityOnly") == 2
