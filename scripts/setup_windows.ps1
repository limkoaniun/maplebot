# One-time setup on Windows. Run from PowerShell:  .\scripts\setup_windows.ps1
$ErrorActionPreference = "Stop"
$py = "C:\Python313\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$venv = "$env:USERPROFILE\.venvs\maplebot"
if (-not (Test-Path "$venv\Scripts\python.exe")) {
    & $py -m venv $venv
}
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install numpy pyyaml opencv-python mss pynput pywin32 pytest
# Editable install is a convenience only; running from the repo root works without it
# (pip cannot install from a \\wsl.localhost path).
try { & "$venv\Scripts\python.exe" -m pip install -e (Join-Path $PSScriptRoot "..") } catch { Write-Warning "editable install skipped: $_" }
Write-Host "Done. Run the bot with:  .\scripts\run.ps1"
