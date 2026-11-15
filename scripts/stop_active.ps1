$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Path (Join-Path $root 'log') -Force | Out-Null
New-Item -ItemType File -Path (Join-Path $root 'log/stop-active') -Force | Out-Null
