<#
.SYNOPSIS
    Re-run the weekly refresh if its last run failed and the sources are back.

.DESCRIPTION
    The weekly job stops the whole chain when a step fails, on purpose: a
    partial refresh reads as a complete one, and the site would serve a mix of
    old and new. The cost of that correctness is a week of staleness, because
    the next attempt is the next Monday.

    On 2026-11-02 that bill came due. `build_dataset` spent 27 minutes and then
    died on `lookup understat.com: getaddrinfow` -- a DNS failure, with
    fbref.com, github.com and football-data.co.uk all resolving normally. Every
    later step was skipped, so `build_players` never ran and the API was never
    reloaded. A correct ingest fix sat merged and invisible, and would have
    until 2026-11-09.

    This closes that window. It is not a second copy of the weekly job: it
    starts the existing "FootballML Weekly Refresh" task, so there is exactly
    one definition of what the weekly run is, and Task Scheduler's own
    `LastTaskResult` is what tells us whether a retry is still wanted.

.NOTES
    Four guards, each for a failure this would otherwise cause:

    - **Nothing to retry.** `LastTaskResult` of 0 means the last weekly run
      succeeded, so this exits silently and the state file is cleared. This is
      what makes an hourly schedule safe: once the rebuild lands, the task
      fires, finds nothing to do in milliseconds, and stops.
    - **Already running.** Starting a second weekly while one is in flight
      would put two processes in `data/processed` at once, which is the one way
      this could corrupt something. The daily job writes there too, so it is
      checked as well.
    - **Sources still down.** Checked in two stages, because this guard has
      been wrong in two different ways. First a TCP connection on 443, not a
      name lookup: 2026-11-02 was a DNS failure, 2026-11-03 was a host that
      resolved and then answered nothing, and only the connect test catches
      both. Then, for hosts in `$ServingMarkers`, the homepage must contain a
      marker that only the live site has: 2026-11-04 was a 200 OK maintenance
      page reading "We will come back soon", which passed the connect test and
      launched a weekly that could not fetch a single row. A failed check is
      the normal case during an outage and costs at most five seconds per host
      to connect, plus one page fetch.
    - **A failure that is not transient.** If the weekly is broken for a reason
      no amount of waiting fixes, retrying hourly forever would burn 27 minutes
      an hour indefinitely. `MaxAttempts` caps the launches per failure, after
      which the next scheduled Monday run is the right remedy rather than more
      hammering. The counter resets on success, not on each run -- starting the
      task moves `LastRunTime`, so anchoring on that would reset every attempt
      and never cap anything.

    Register it with `scripts/register_weekly_retry.ps1`.

    Watch what it decides:
      Get-Content logs/weekly-retry.log -Tail 20
#>
[CmdletBinding()]
param(
    [string] $WeeklyTaskName = "FootballML Weekly Refresh",
    [string] $DailyTaskName = "FootballML Daily Refresh",
    # The host whose outage this exists for. Understat is the only source with
    # no fallback: football-data has the results, but xG and every player
    # season come from here alone.
    [string[]] $RequiredHosts = @("understat.com", "fbref.com"),
    # Hosts that have to prove they are *serving* rather than merely answering,
    # and the marker that proves it. See guard 3.
    [hashtable] $ServingMarkers = @{ "understat.com" = "/league/" },
    # Six launches, not unlimited: see the fourth guard above.
    [int] $MaxAttempts = 6
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $root "logs"
$log = Join-Path $logDir "weekly-retry.log"
$state = Join-Path $logDir "weekly-retry.state"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

function Write-Line([string] $message) {
    $stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Add-Content -Path $log -Value "$stamp  $message"
}

function Get-TaskOrNull([string] $name) {
    try { Get-ScheduledTask -TaskName $name -ErrorAction Stop } catch { $null }
}

$weekly = Get-TaskOrNull $WeeklyTaskName
if (-not $weekly) {
    Write-Line "no '$WeeklyTaskName' task registered; nothing to retry"
    exit 0
}

# --- guard 1: is a retry wanted at all? -------------------------------------
$info = $weekly | Get-ScheduledTaskInfo
if ($info.LastTaskResult -eq 0) {
    if (Test-Path $state) {
        Remove-Item $state -Force
        Write-Line "weekly last succeeded; retry counter cleared"
    }
    exit 0
}

# --- guard 2: is anything already writing data/processed? -------------------
foreach ($name in @($WeeklyTaskName, $DailyTaskName)) {
    $task = Get-TaskOrNull $name
    if ($task -and $task.State -eq "Running") {
        Write-Line "'$name' is already running; leaving it alone"
        exit 0
    }
}

# --- guard 3: have the sources come back? -----------------------------------
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
# So a host in $ServingMarkers must also return a page containing its marker.
# `/league/` is chosen as a *positive* feature of the live site -- its homepage
# is a set of league links -- rather than a match on the maintenance wording,
# which the operator can reword at any time and which would then read as
# healthy. Matching what a working site has fails safe; matching what a broken
# one says fails open.
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
        return $false
    }
    if ($response.StatusCode -ne 200) { return $false }
    return ($response.Content -like "*$Marker*")
}

$down = @()
$notServing = @()
foreach ($hostname in $RequiredHosts) {
    if (-not (Test-HostReachable -Hostname $hostname)) {
        $down += $hostname
        continue
    }
    $marker = $ServingMarkers[$hostname]
    if ($marker -and -not (Test-HostServing -Hostname $hostname -Marker $marker)) {
        $notServing += $hostname
    }
}
if ($down.Count -gt 0) {
    Write-Line ("still unreachable: " + ($down -join ", ") + " -- not retrying")
    exit 0
}
# Logged apart from "unreachable" on purpose: the two need different responses.
# Unreachable is somebody else's outage and waiting is all there is to do. A
# host that answers without serving may instead mean this marker is wrong, and
# that failure is the dangerous one -- it would hold the weekly back while the
# source is healthy -- so it has to be distinguishable in the log at a glance.
if ($notServing.Count -gt 0) {
    Write-Line (
        "answering but not serving data: " + ($notServing -join ", ") +
        " -- not retrying. If the site is in fact up, this guard's marker is " +
        "stale: see `$ServingMarkers in scripts/weekly_retry.ps1."
    )
    exit 0
}

# --- guard 4: have we tried enough times? -----------------------------------
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
    exit 0
}

$attempts += 1
Set-Content -Path $state -Value $attempts

Write-Line (
    "last weekly exited $($info.LastTaskResult) and every source is reachable; " +
    "starting '$WeeklyTaskName' (attempt $attempts of $MaxAttempts)"
)
Start-ScheduledTask -TaskName $WeeklyTaskName
exit 0
