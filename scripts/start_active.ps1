param([switch]$WaitForStart)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$status = Join-Path $root 'log/active-status.json'
if (Test-Path -LiteralPath $status) {
    $previous = Get-Content -LiteralPath $status -Raw | ConvertFrom-Json
    $recent = ([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$previous.updated) -lt 6
    $existing = Get-Process -Id $previous.pid -ErrorAction SilentlyContinue
    if ($previous.running -and $recent -and $existing -and $existing.ProcessName -eq 'pythonw') {
        Write-Host 'Maplebot is already running.'
        exit 0
    }
}
$stopFile = Join-Path $root 'log/stop-active'
if (Test-Path -LiteralPath $stopFile) { Remove-Item -LiteralPath $stopFile }
$armFile = Join-Path $root 'log/arm-active'
if (Test-Path -LiteralPath $armFile) { Remove-Item -LiteralPath $armFile }
foreach ($signal in @('pause-active', 'resume-active', 'reload-active', 'return-to-start-active', 'hold-farming-active')) {
    $signalPath = Join-Path $root ('log/' + $signal)
    if (Test-Path -LiteralPath $signalPath) { Remove-Item -LiteralPath $signalPath }
}
$python = Join-Path $env:USERPROFILE '.venvs/maplebot/Scripts/pythonw.exe'
$arguments = @('-u', 'scripts/run_active.py')
if ($WaitForStart) { $arguments += '--wait-for-start' }
Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -Verb RunAs -WindowStyle Hidden
