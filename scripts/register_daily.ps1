<#
.SYNOPSIS
    Register (or re-register) the daily results refresh as a scheduled task.

.DESCRIPTION
    The weekly job rebuilds everything on a Monday. This one runs every morning
    and does the smallest correct thing: fetch results, and if football-data
    published a match we have not seen, settle it, re-simulate the tables and
    reload the API. On a day with nothing new it stops after the fetch.

    Without it, a match played on a Tuesday waits until Monday to appear.

    Safe to run twice: the task is replaced, not duplicated.

.NOTES
    Mirrors the existing "FootballML Weekly Refresh" task, which runs Mondays at
    07:00. This one is at 07:30 so the two never overlap -- the weekly run takes
    about twenty minutes, and both start by writing the same data files.

    Remove it with:
      Unregister-ScheduledTask -TaskName "FootballML Daily Refresh" -Confirm:$false

    Check when it last ran, and whether it worked:
      Get-ScheduledTask "FootballML Daily Refresh" | Get-ScheduledTaskInfo
#>
[CmdletBinding()]
param(
    [string] $TaskName = "FootballML Daily Refresh",
    # 07:30, not 07:00: the weekly task starts then on a Monday and runs for
    # roughly twenty minutes. Two jobs writing data/processed at once is the
    # one way this could corrupt something.
    [string] $At = "07:30"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "No virtualenv at $python. Create it before registering the task."
}

# `cmd /c cd /d ... && python -m` rather than invoking python directly, to match
# the weekly task and because the pipelines resolve paths from the working
# directory.
$action = New-ScheduledTaskAction -Execute "cmd" -Argument (
    "/c cd /d `"$root`" && `".venv\Scripts\python.exe`" -m pipelines.daily"
)
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Fetch new match results; if any arrived, settle them, re-simulate the projected tables and reload the API. Skips the expensive weekly rebuild." `
    -Force | Out-Null

$info = Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo
Write-Host "Registered '$TaskName'"
Write-Host "  runs daily at $At"
Write-Host "  next run: $($info.NextRunTime)"
Write-Host ""
Write-Host "Run it now with:  python -m pipelines.daily"
Write-Host "Remove it with:   Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false"
