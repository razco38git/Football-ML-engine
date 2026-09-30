<#
.SYNOPSIS
    Start the API and the web front end, unless they are already running.

.DESCRIPTION
    The site is two processes and needs both: the page renders from the Vite dev
    server on 8443, but every tab on it reads the API on 8000, so a site started
    without the API shows "Cannot reach the API" everywhere.

    Neither survives a reboot, and the weekly scheduled task refreshes the *data*,
    not the servers -- so without this the site is simply down until someone
    remembers to start it by hand.

    Safe to run repeatedly. It starts only what is not already listening, so a
    second run cannot leave two servers fighting over a port. That matters
    because it is registered to run at login and may also be run by hand.

.NOTES
    Both bind to localhost only. To open the site from another device you would
    need uvicorn on --host 0.0.0.0 and VITE_API_URL pointing at this machine --
    deliberately not the default.

    Registered at login via a shortcut in:
      %APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
    Remove it with:
      Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\FootballML.lnk"
#>
[CmdletBinding()]
param(
    [int] $ApiPort = 8000,
    # 8443, not 5173: web/vite.config.ts reads `process.env.PORT || '8443'`.
    # `.claude/launch.json` asks for 5173, which is why the dev preview lands
    # there -- but a plain `npm run dev` serves 8443, and that is what this
    # script starts.
    [int] $WebPort = 8443,
    # Where to keep the servers' own output. A start script that discards it
    # leaves nothing to read when the site is up but wrong.
    [string] $LogDir = (Join-Path $PSScriptRoot "..\logs")
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$null = New-Item -ItemType Directory -Force -Path $LogDir

function Test-PortListening {
    param([int] $Port)
    $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Start-Background {
    param([string] $Name, [string] $Exe, [string[]] $Arguments, [int] $Port)

    if (Test-PortListening -Port $Port) {
        Write-Host "  $Name already listening on $Port -- leaving it alone"
        return
    }
    $log = Join-Path $LogDir "$Name.log"
    $proc = Start-Process -FilePath $Exe -ArgumentList $Arguments `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    Write-Host "  started $Name (PID $($proc.Id)) -> $log"
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No virtualenv at $python -- run the project setup first."
}

Write-Host "FootballML:"
Start-Background -Name "api" -Exe $python -Port $ApiPort -Arguments @(
    "-m", "uvicorn", "footballml.api.app:app", "--host", "127.0.0.1", "--port", "$ApiPort"
)
# npm is a shell shim on Windows, so it cannot be launched directly.
Start-Background -Name "web" -Exe "cmd.exe" -Port $WebPort -Arguments @(
    "/c", "npm", "run", "dev", "--prefix", "web"
)

# The API rebuilds features over the full history at startup, which takes a
# while; reporting "ready" before it answers would be worse than saying nothing.
$deadline = (Get-Date).AddSeconds(120)
while ((Get-Date) -lt $deadline) {
    if ((Test-PortListening -Port $ApiPort) -and (Test-PortListening -Port $WebPort)) { break }
    Start-Sleep -Seconds 2
}

$apiUp = Test-PortListening -Port $ApiPort
$webUp = Test-PortListening -Port $WebPort
Write-Host ""
Write-Host "  API  $(if ($apiUp) { 'up' } else { 'NOT UP' })  http://127.0.0.1:$ApiPort"
Write-Host "  site $(if ($webUp) { 'up' } else { 'NOT UP' })  http://localhost:$WebPort"
if (-not ($apiUp -and $webUp)) {
    Write-Warning "Something did not start. See $LogDir\api.log.err and $LogDir\web.log.err"
    exit 1
}
Write-Host ""
Write-Host "Open http://localhost:$WebPort"
