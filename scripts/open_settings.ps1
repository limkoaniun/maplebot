$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $env:USERPROFILE '.venvs/maplebot/Scripts/pythonw.exe'
Start-Process -FilePath $python -ArgumentList @('-m', 'maplebot.gui') -WorkingDirectory $root -WindowStyle Normal
