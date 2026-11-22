<#
.SYNOPSIS
    Re-run a refresh whose sources were down, once they are back.

.DESCRIPTION
    Two decisions, in priority order, because two different jobs can be owed
    and only one may touch `data/processed` at a time.

    **The weekly**, when its last run failed. The weekly job stops the whole
    chain when a step fails, on purpose: a partial refresh reads as a complete
    one, and the site would serve a mix of old and new. The cost of that
    correctness is a week of staleness, because the next attempt is the next
    Monday.

    On 2026-11-02 that bill came due. `build_dataset` spent 27 minutes and then
    died on `lookup understat.com: getaddrinfow` -- a DNS failure, with
    fbref.com, github.com and football-data.co.uk all resolving normally. Every
    later step was skipped, so `build_players` never ran and the API was never
    reloaded. A correct ingest fix sat merged and invisible, and would have
    until 2026-11-09.

    **The daily**, when results alone are behind. Added 2026-11-22, for the
    failure the weekly decision cannot see. The daily job is deliberately built
    to carry on when a source fails -- optional steps log and continue -- so it
    exits 0 while publishing stale results, and `LastTaskResult` says nothing.
    That day it exited 0 having published Friday's three matches and none of
    Saturday's twenty-six: football-data.co.uk refused every connection, its
    cached copy predated the weekend, and the FBref top-up that exists to cover
    exactly this was returning 403. The site then waited until the next 10:30
    run, because nothing watched the one host that carries results.

    So the trigger here is not an exit code but the data: a fixture whose date
    has passed and whose result is still missing. That is the only honest
    signal, and it clears itself the moment the results land.

    Note what is deliberately *not* done: football-data.co.uk is not added to
    `$RequiredHosts`. That list gates the weekly, and the step that blocks the
    weekly is `build_players`, which needs Understat and never touches
    football-data. Gating the weekly on it would hold a rebuild back for an
    outage that cannot affect it.

    Neither decision is a second copy of a job: each starts the existing
    scheduled task, so there is exactly one definition of what a run is.

.NOTES
    Five guards, each for a failure this would otherwise cause:

    - **Already running.** Starting a job while one is in flight would put two
      processes in `data/processed` at once, which is the one way this could
      corrupt something. Checked first, before either decision, and it covers
      the weekly and the daily because both write there.
    - **Nothing to retry.** `LastTaskResult` of 0 means the last weekly run
      succeeded, so the weekly decision stands down and the state file is
      cleared. This is what makes an hourly schedule safe: once the rebuild
      lands, the task fires, finds nothing to do in milliseconds, and stops.
    - **Sources still down.** Checked in three stages, because this guard has
      been wrong in three different ways. First a TCP connection on 443, not a
      name lookup: 2026-11-02 was a DNS failure, 2026-11-03 was a host that
      resolved and then answered nothing, and only the connect test catches
      both. Then a 200 from the homepage: on 2026-11-22 fbref.com accepted TCP
      and answered 403 to everything, which a connect test alone reads as
      healthy. Then, for hosts in `$ServingMarkers`, that page must contain a
      marker only the live site has: 2026-11-04 was a 200 OK maintenance page
      reading "We will come back soon", which passed the connect test and
      launched a weekly that could not fetch a single row. A failed check is
      the normal case during an outage and costs at most five seconds per host
      to connect, plus one page fetch.
    - **Results are not actually behind.** The daily decision reads the
      prediction store rather than a task result, and if it cannot read it at
      all it says so and does nothing. A missing file is not evidence that
      results are current.
    - **A failure that is not transient.** If a job is broken for a reason no
      amount of waiting fixes, retrying hourly forever would burn 27 minutes an
      hour indefinitely. `MaxAttempts` caps the launches per failure, after
      which the next scheduled run is the right remedy rather than more
      hammering. Each counter resets on the thing it is trying to achieve --
      the weekly's on a successful weekly, the daily's on results arriving --
      not on each run: starting a task moves `LastRunTime`, so anchoring on
      that would reset every attempt and never cap anything.

    Register it with `scripts/register_weekly_retry.ps1`.

    Watch what it decides:
      Get-Content logs/weekly-retry.log -Tail 20
#>
[CmdletBinding()]
param(
    [string] $WeeklyTaskName = "FootballML Weekly Refresh",
    [string] $DailyTaskName = "FootballML Daily Refresh",
    # Hosts the *weekly* waits for. Understat is the source with no fallback:
    # football-data has the results, but xG and every player season come from
    # here alone, and `build_players` is the step that fails without it.
    [string[]] $RequiredHosts = @("understat.com", "fbref.com"),
    # Hosts the *daily* waits for, when results are behind. Only the results
    # sources belong here, and they are checked separately from the weekly's so
    # that one source's outage cannot hold back a job it cannot affect.
    [string[]] $ResultHosts = @("football-data.co.uk", "fbref.com"),
    # Hosts that have to prove they are *serving* rather than merely answering,
    # and the marker that proves it. See guard 3, and the note below on why
    # football-data.co.uk is not in here yet.
    [hashtable] $ServingMarkers = @{ "understat.com" = "/league/" },
    # Six launches, not unlimited: see the last guard above.
    [int] $MaxAttempts = 6,
    # Fewer for the daily, because its trigger is softer. A stale result can
    # mean the source has not published yet rather than that we failed to ask,
    # and three hourly attempts is enough to catch a recovery without spending
    # the afternoon re-asking a source that has nothing new.
    [int] $MaxDailyAttempts = 3,
    # How far back an unsettled fixture still counts as news.
    #
    # Not unbounded, and this is the difference between a trigger that heals
    # itself and one that jams. Some fixtures never settle: on 2026-11-22 the
    # store held 39 unsettled played fixtures, and 13 of them were from
    # September and October -- abandoned or rescheduled matches whose original
    # row stays unsettled for good. Counting those, the total can never reach
    # zero, so the nudge counter never clears and three launches switch it off
    # permanently.
    #
    # A week also matches what this is for: closing a gap of hours. Beyond
    # that the daily's own 10:30 schedule has had seven goes and a nudge is not
    # the remedy -- and when a source does return, the next ordinary daily
    # backfills the whole history regardless of this window.
    [int] $ResultWindowDays = 7,
    # The prediction store the daily decision reads. A parameter so the
    # "cannot tell" branch can be exercised against a path that does not
    # exist, rather than asserted and hoped for. Defaults below, once $root is
    # known.
    [string] $PredictionsPath
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "logs"
$log = Join-Path $logDir "weekly-retry.log"
$state = Join-Path $logDir "weekly-retry.state"
$dailyState = Join-Path $logDir "daily-nudge.state"
if (-not $PredictionsPath) {
    $PredictionsPath = Join-Path $root "data\processed\predictions.csv"
}

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Write-Line([string] $message) {
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -Path $log -Value "$stamp  $message"
}

function Get-TaskOrNull([string] $name) {
    try { Get-ScheduledTask -TaskName $name -ErrorAction Stop } catch { $null }
}

# --- source health ----------------------------------------------------------
# Resolution is not reachability, and this guard learned the difference the
# expensive way. The 2026-11-02 outage began as a DNS failure, so testing
# `getaddrinfo` matched exactly what the pipeline saw. By 2026-11-03 it had
# changed character: understat.com resolved to 45.90.32.72 immediately and
# then refused every connection. This guard saw a clean lookup, logged "every
# source resolves", and waved all six retries through -- each spending ~18
# minutes in build_dataset to reach the same dead socket -- before declaring a
# live outage permanent and standing down until the next Monday.
#
# A TCP connect is what the fetch actually needs, so it is what gets tested.
# It subsumes the DNS check rather than replacing it: an unresolvable name
# throws from BeginConnect just as a refused port does, so one test covers
# both shapes of outage and the normal case still costs well under a second.
function Test-HostReachable {
    param(
        [string] $Hostname,
        [int] $Port = 443,
        [int] $TimeoutMs = 5000
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect($Hostname, $Port, $null, $null)
        # A timeout is the failure mode that matters here: the host that
        # answers nothing looks identical to a healthy one until you wait.
        if (-not $pending.AsyncWaitHandle.WaitOne($TimeoutMs, $false)) {
            return $false
        }
        # Throws on a refused or reset connection, which WaitOne does not.
        $client.EndConnect($pending)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

# Answering is not the same as serving, which is the second way this guard has
# been wrong. On 2026-11-04 understat.com accepted TCP on 443 and returned 200
# for a 26,719-byte page whose entire visible text was "We will come back
# soon": no scripts, no /league/ links, no data. The connect test passed, the
# weekly launched, spent 28 minutes in build_dataset, fell back to carried-
# forward xG, and build_players died on a JSONDecodeError parsing HTML as JSON.
# Six such retries is three hours of republishing data we already had.
#
# So every host must return **200** from its homepage, and a host in
# $ServingMarkers must additionally return a page containing its marker.
#
# The 200 applies to everyone because of a third failure seen on 2026-11-22:
# fbref.com accepted TCP on 443 and answered **403** to every request, bot
# protection rather than an outage. A host with no marker used to skip this
# function entirely, so that 403 read as perfectly healthy -- and since FBref
# is the results fallback, it would have launched a 42-minute daily that could
# not fetch a row. A status check costs one page fetch and catches it.
#
# `/league/` is chosen as a *positive* feature of the live site -- its homepage
# is a set of league links -- rather than a match on the maintenance wording,
# which the operator can reword at any time and which would then read as
# healthy. Matching what a working site has fails safe; matching what a broken
# one says fails open.
#
# football-data.co.uk has no marker yet, deliberately. Its outage began before
# this was written and its homepage cannot be inspected while it is refusing
# connections, so any marker would be a guess -- and a wrong marker fails in
# the dangerous direction, holding a healthy source back indefinitely while
# logging that it is "answering but not serving". The 200 check covers the
# maintenance-page case for now. When the site answers again, fetch its
# homepage, confirm a stable navigation element, and add it here.
function Test-HostServing {
    param(
        [string] $Hostname,
        [string] $Marker,
        [int] $TimeoutSec = 20
    )
    try {
        $response = Invoke-WebRequest -Uri "https://$Hostname/" -TimeoutSec $TimeoutSec `
            -MaximumRedirection 3 -UseBasicParsing -ErrorAction Stop
    } catch {
        # A non-2xx status throws here, which is the 403 case, as do timeouts
        # and TLS errors. None of them is a source that can serve data.
        return $false
    }
    if ($response.StatusCode -ne 200) { return $false }
    if (-not $Marker) { return $true }
    return ($response.Content -like "*$Marker*")
}

# Splits a host list into the ones that cannot be reached and the ones that
# answer without serving. Kept apart rather than merged into "down" because the
# two need different responses: see where they are logged.
function Test-Sources {
    param([string[]] $Hostnames, [hashtable] $Markers)
    $down = @()
    $notServing = @()
    foreach ($hostname in $Hostnames) {
        if (-not (Test-HostReachable -Hostname $hostname)) {
            $down += $hostname
            continue
        }
        # Called for every host, not only the ones with a marker: the status
        # check is what catches a 403, and skipping it was how FBref's bot
        # protection read as healthy.
        if (-not (Test-HostServing -Hostname $hostname -Marker $Markers[$hostname])) {
            $notServing += $hostname
        }
    }
    return @{ Down = $down; NotServing = $notServing }
}

# --- how far behind the results are ----------------------------------------
# The daily's exit code cannot answer this: its steps are optional by design,
# so it returns 0 having published nothing new. The prediction store can -- a
# fixture whose date has passed with no `actual_home_goals` is a match that was
# played and never settled.
#
# Returns -1 for "cannot tell", which is not the same as zero and must not be
# treated as it. A missing file, or a Date column that has changed shape, would
# otherwise read as "results are current" and silently switch this off forever.
function Get-UnsettledCount {
    param([string] $Path, [int] $WindowDays)
    if (-not (Test-Path $Path)) { return -1 }
    $today = (Get-Date).Date
    $oldest = $today.AddDays(-$WindowDays)
    $parsed = 0
    $unsettled = 0
    foreach ($row in (Import-Csv -Path $Path)) {
        $when = [datetime]::MinValue
        $ok = [datetime]::TryParseExact(
            $row.Date, "yyyy-MM-dd",
            [System.Globalization.CultureInfo]::InvariantCulture,
            [System.Globalization.DateTimeStyles]::None, [ref] $when
        )
        if (-not $ok) { continue }
        $parsed += 1
        # Today's fixtures are not late, and anything before the window is an
        # orphan this cannot fix: see $ResultWindowDays.
        if ($when.Date -ge $today -or $when.Date -lt $oldest) { continue }
        if ([string]::IsNullOrWhiteSpace($row.actual_home_goals)) { $unsettled += 1 }
    }
    if ($parsed -eq 0) { return -1 }
    return $unsettled
}

$weekly = Get-TaskOrNull $WeeklyTaskName
if (-not $weekly) {
    Write-Line "no '$WeeklyTaskName' task registered; nothing to retry"
    exit 0
}

# --- guard: is anything already writing data/processed? ---------------------
foreach ($name in @($WeeklyTaskName, $DailyTaskName)) {
    $task = Get-TaskOrNull $name
    if ($task -and $task.State -eq "Running") {
        Write-Line "'$name' is already running; leaving it alone"
        exit 0
    }
}

# --- decision 1: the weekly, when its last run failed -----------------------
$startWeekly = $false
$info = $weekly | Get-ScheduledTaskInfo
if ($info.LastTaskResult -eq 0) {
    if (Test-Path $state) {
        Remove-Item $state -Force
        Write-Line "weekly last succeeded; retry counter cleared"
    }
} else {
    $health = Test-Sources -Hostnames $RequiredHosts -Markers $ServingMarkers
    if ($health.Down.Count -gt 0) {
        Write-Line ("still unreachable: " + ($health.Down -join ", ") + " -- not retrying")
    }
    # Logged apart from "unreachable" on purpose: the two need different
    # responses. Unreachable is somebody else's outage and waiting is all there
    # is to do. A host that answers without serving may instead mean this
    # marker is wrong, and that failure is the dangerous one -- it would hold
    # the weekly back while the source is healthy -- so it has to be
    # distinguishable in the log at a glance.
    elseif ($health.NotServing.Count -gt 0) {
        Write-Line (
            "answering but not serving data: " + ($health.NotServing -join ", ") +
            " -- not retrying. If the site is in fact up, this guard's marker is " +
            "stale: see `$ServingMarkers in scripts/weekly_retry.ps1."
        )
    } else {
        $attempts = 0
        if (Test-Path $state) {
            $raw = (Get-Content $state -Raw).Trim()
            [void][int]::TryParse($raw, [ref] $attempts)
        }
        if ($attempts -ge $MaxAttempts) {
            Write-Line (
                "started the weekly $attempts time(s) with every source reachable " +
                "and it failed each time, so this is not a transient outage; " +
                "waiting for the scheduled run rather than retrying. " +
                "Investigate logs/weekly-*.log."
            )
        } else {
            $attempts += 1
            Set-Content -Path $state -Value $attempts
            Write-Line (
                "last weekly exited $($info.LastTaskResult) and every source is reachable; " +
                "starting '$WeeklyTaskName' (attempt $attempts of $MaxAttempts)"
            )
            $startWeekly = $true
        }
    }
}

if ($startWeekly) {
    Start-ScheduledTask -TaskName $WeeklyTaskName
    exit 0
}

# --- decision 2: the daily, when results alone are behind -------------------
# Only reached when the weekly is not being started, because the weekly runs
# `build_dataset` itself and would fetch the same results. This is the branch
# that matters during a partial outage: on 2026-11-22 the weekly was owed and
# blocked on Understat, so a decision nested inside "the weekly succeeded"
# would never have run at all.
$daily = Get-TaskOrNull $DailyTaskName
if (-not $daily) {
    Write-Line "no '$DailyTaskName' task registered; cannot nudge the daily"
    exit 0
}

$unsettled = Get-UnsettledCount -Path $PredictionsPath -WindowDays $ResultWindowDays
if ($unsettled -lt 0) {
    Write-Line (
        "cannot tell whether results are behind: $PredictionsPath is missing, or its " +
        "Date column is no longer yyyy-MM-dd -- not nudging the daily"
    )
    exit 0
}
if ($unsettled -eq 0) {
    if (Test-Path $dailyState) {
        Remove-Item $dailyState -Force
        Write-Line (
            "every fixture played in the last $ResultWindowDays day(s) is settled; " +
            "daily nudge counter cleared"
        )
    }
    exit 0
}

# **Any** results host will do, where the weekly needs all of its. Either
# source can supply a result -- football-data is what `build_dataset` fetches
# and FBref is the top-up that covered Friday's three matches when it could not
# -- so requiring both would let an FBref outage block a refresh football-data
# could serve. That is the same mistake as gating the weekly on football-data,
# one layer down.
$health = Test-Sources -Hostnames $ResultHosts -Markers $ServingMarkers
$usable = @(
    $ResultHosts | Where-Object {
        @($health.Down) -notcontains $_ -and @($health.NotServing) -notcontains $_
    }
)
if ($usable.Count -eq 0) {
    $why = @()
    if (@($health.Down).Count -gt 0) {
        $why += "still unreachable: " + (@($health.Down) -join ", ")
    }
    # Said apart from "unreachable" for the reason given in decision 1: a host
    # that answers without serving may mean this guard's own marker is stale,
    # and that failure holds a healthy source back.
    if (@($health.NotServing).Count -gt 0) {
        $why += (
            "answering but not serving data: " + (@($health.NotServing) -join ", ") +
            " (if a site is in fact up, this guard's marker is stale: see " +
            "`$ServingMarkers in scripts/weekly_retry.ps1)"
        )
    }
    Write-Line (
        "$unsettled fixture(s) played in the last $ResultWindowDays day(s) are " +
        "still unsettled, but no results source is " +
        "usable -- " + ($why -join "; ") + " -- not nudging the daily"
    )
    exit 0
}

$nudges = 0
if (Test-Path $dailyState) {
    $raw = (Get-Content $dailyState -Raw).Trim()
    [void][int]::TryParse($raw, [ref] $nudges)
}
if ($nudges -ge $MaxDailyAttempts) {
    Write-Line (
        "started the daily $nudges time(s) with the results sources reachable and " +
        "$unsettled fixture(s) are still unsettled, so the gap is not ours to " +
        "close; waiting for the scheduled run. The source may simply not have " +
        "published them yet -- check logs/daily-*.log."
    )
    exit 0
}

$nudges += 1
Set-Content -Path $dailyState -Value $nudges
Write-Line (
    "$unsettled fixture(s) played in the last $ResultWindowDays day(s) are " +
    "unsettled and " + ($usable -join ", ") +
    " is serving; starting '$DailyTaskName' (nudge $nudges of $MaxDailyAttempts)"
)
Start-ScheduledTask -TaskName $DailyTaskName
exit 0
