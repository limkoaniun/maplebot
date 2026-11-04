# Start the bot with config\user.yaml. Extra arguments are passed through, e.g.
#   .\scripts\run.ps1 --map north_forest_training_ground_2 --monsters green_mushroom
$venv = "$env:USERPROFILE\.venvs\maplebot"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$cfg = Join-Path $root "config\user.yaml"
if (Test-Path $cfg) {
    & "$venv\Scripts\python.exe" -m maplebot run --cfg $cfg @args
} else {
    & "$venv\Scripts\python.exe" -m maplebot run @args
}
