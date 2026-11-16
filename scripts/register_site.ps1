<#
.SYNOPSIS
    Bring the API and the web server back up after a reboot.

.DESCRIPTION
    The scheduled pipelines survive a restart because they are registered with
    Task Scheduler. The two servers did not: `start_site.ps1` launches them
    detached with `Start-Process`, which outlives the shell that ran it but not
    the machine.

    So a reboot takes the site offline silently, and nothing says so -- the
    pipelines keep running, keep fetching results and keep publishing to disk,
    which is exactly what makes it hard to notice. On 2026-11-11 the machine
    restarted at 03:49 and the site stayed down for five days while the data
    behind it went on updating every morning.

    This registers `start_site.ps1` to run at logon, which closes that gap.

    Safe to run twice: the task is replaced, not duplicated. Safe to fire when
    the site is already up: `start_site.ps1` leaves a healthy listener alone
    and only replaces an API older than the code it is serving.

.NOTES
    **At logon, not at startup.** A startup trigger runs as SYSTEM before
    anyone signs in, and the web server is a Vite dev server that wants the
    user's PATH, node install and profile. The site is also only worth serving
    when somebody is at the machine, so the earlier trigger buys nothing.

    The delay is for the same reason a person would wait: at logon the network
    adapter is often not up yet, and the API binds a port.

    Battery flags matter here -- this runs on a laptop, and without them a task
    will not start, or will be stopped, when it is unplugged.

    Remove it with:
      Unregister-ScheduledTask -TaskName "FootballML Site" -Confirm:$false

    Check it:
      Get-ScheduledTask "FootballML Site" | Get-ScheduledTaskInfo
#>
[CmdletBinding()]
param(
    [string] $TaskName = "FootballML Site",
    # Long enough for the network to come up, short enough that the site is
    # there by the time anyone opens a browser.
    [string] $Delay = "PT1M"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot "start_site.ps1"

if (-not (Test-Path $script)) {
    throw "No start_site.ps1 at $script."
}

# -NoProfile so a slow or broken user profile cannot hang logon, and
# -ExecutionPolicy Bypass because this runs unattended and must not stop on a
# policy prompt.
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument (
    "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
) -WorkingDirectory $root

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = $Delay

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Start the FootballML API and web server at logon. The pipelines survive a reboot because they are scheduled tasks; the servers are detached processes and do not." `
    -Force | Out-Null

Write-Host "Registered '$TaskName'"
Write-Host "  runs at logon for $env:USERNAME, after $Delay"
Write-Host "  action: $script"
Write-Host ""
Write-Host "Test it without rebooting:  Start-ScheduledTask -TaskName `"$TaskName`""
Write-Host "Remove it with:             Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false"
