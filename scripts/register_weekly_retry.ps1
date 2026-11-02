<#
.SYNOPSIS
    Register (or re-register) the hourly retry of a failed weekly refresh.

.DESCRIPTION
    The weekly job aborts the whole chain on a failed step, so one bad source
    costs a week of staleness: the next attempt is the next Monday. This task
    runs `scripts/weekly_retry.ps1` every hour, which starts the weekly job
    again if -- and only if -- its last run failed and every source resolves.

    It is cheap to leave registered. Once a weekly run succeeds, the retry
    finds `LastTaskResult` of 0 and exits in milliseconds, so the hourly
    schedule costs nothing in the normal case and the task never needs
    removing after an incident.

    Safe to run twice: the task is replaced, not duplicated.

.NOTES
    Hourly rather than every few minutes. The thing being waited on is someone
    else's DNS coming back, which is not worth polling hard, and a weekly run
    takes roughly half an hour -- `weekly_retry.ps1` refuses to start a second
    one, but there is no reason to keep asking.

    Deliberately not aligned to :00. The daily task runs at 10:30 and the
    weekly at 10:00 on Mondays, and `weekly_retry.ps1` already refuses to run
    while either is in flight; starting at :20 keeps it clear of both anyway
    rather than relying only on that check.

    Remove it with:
      Unregister-ScheduledTask -TaskName "FootballML Weekly Retry" -Confirm:$false

    See what it has decided, and why:
      Get-Content logs/weekly-retry.log -Tail 20

    Check the task itself:
      Get-ScheduledTask "FootballML Weekly Retry" | Get-ScheduledTaskInfo
#>
[CmdletBinding()]
param(
    [string] $TaskName = "FootballML Weekly Retry",
    [string] $At = "00:20"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "weekly_retry.ps1"

if (-not (Test-Path $script)) {
    throw "No retry script at $script."
}

# -NoProfile and -NonInteractive: under Task Scheduler there is no console and
# no reason to load a user profile. -ExecutionPolicy Bypass because the script
# is local and unsigned, and the task would otherwise fail on a default policy.
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument (
    "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$script`""
)

# -RepetitionInterval on a daily trigger, rather than 24 one-off triggers.
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$trigger.Repetition = (New-ScheduledTaskTrigger `
    -Once -At $At `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Hours 24)).Repetition

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description (
        "Hourly: if the last weekly refresh failed and every data source " +
        "resolves again, start it again. Does nothing once a weekly run has " +
        "succeeded. Without this, one transient source outage costs a week " +
        "of stale ratings."
    ) `
    -Force | Out-Null

$info = Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
Write-Host "Registered '$TaskName'"
Write-Host "  runs hourly from $At"
Write-Host "  next run: $($info.NextRunTime)"
Write-Host ""
Write-Host "Try it now with:  powershell -NoProfile -File scripts\weekly_retry.ps1"
Write-Host "Then read:        Get-Content logs\weekly-retry.log -Tail 5"
Write-Host "Remove it with:   Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false"
