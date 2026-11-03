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
    - **Sources still down.** Checked by opening a TCP connection on 443, not
      by resolving the name. An outage can present either way -- 2026-11-02
      was a DNS failure, 2026-11-03 was a host that resolved and then answered
      nothing -- and only the connect test catches both. A failed check is the
      normal case during an outage and costs at most five seconds per host.
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

$down = @()
foreach ($hostname in $RequiredHosts) {
    if (-not (Test-HostReachable -Hostname $hostname)) {
        $down += $hostname
    }
}
if ($down.Count -gt 0) {
    Write-Line ("still unreachable: " + ($down -join ", ") + " -- not retrying")
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
