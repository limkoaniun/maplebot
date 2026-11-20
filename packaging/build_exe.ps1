# Developer tool: build the standalone Maplebot folder and zip in a SEPARATE build venv.
# End users never need this. Running it downloads packages from PyPI into $BuildRoot (never the
# live venv) and is the developer's decision. All build logic lives in packaging\build.py.
param(
    [switch]$Clean,
    [string]$Python = 'C:\Python313\python.exe',
    [string]$BuildRoot = "$env:LOCALAPPDATA\maplebot-build",
    [switch]$KeepFfmpeg,
    [switch]$AllowDirty,
    [switch]$NoExample,
    [switch]$Public,
    [string[]]$DenyToken = @(),
    [switch]$SkipZip,
    [string]$SignThumbprint,
    [switch]$RefreshLock
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$liveVenv = [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE '.venvs\maplebot'))
$BuildRoot = [IO.Path]::GetFullPath($BuildRoot)
$venv = Join-Path $BuildRoot 'venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
$lock = Join-Path $PSScriptRoot 'requirements-build.lock'
$pins = Join-Path $PSScriptRoot 'requirements-build.txt'

function Test-Inside([string]$Child, [string]$Parent) {
    $c = [IO.Path]::GetFullPath($Child).TrimEnd('\') + '\'
    $p = [IO.Path]::GetFullPath($Parent).TrimEnd('\') + '\'
    return $c.StartsWith($p, [StringComparison]::OrdinalIgnoreCase)
}

if (Test-Inside $Python $liveVenv) { throw "Refusing to build from the live venv ($liveVenv). Pass a base Python with -Python." }
if (Test-Inside $venv $liveVenv) { throw "The build venv must not be the live venv." }
if (Test-Inside $BuildRoot $repo) { throw "The build root must be outside the repository ($repo)." }

if ($Clean) {
    foreach ($name in @('venv', 'stage', 'work', 'dist')) {
        $path = Join-Path $BuildRoot $name
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Recurse -Force }
    }
}
New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
if (-not (Test-Path -LiteralPath $venvPython)) { & $Python -m venv $venv; if ($LASTEXITCODE) { throw 'venv creation failed' } }

if ($RefreshLock) {
    $wheels = Join-Path $BuildRoot 'wheels'
    if (Test-Path -LiteralPath $wheels) { Remove-Item -LiteralPath $wheels -Recurse -Force }
    & $venvPython -m pip download --only-binary=:all: -r $pins -d $wheels
    if ($LASTEXITCODE) { throw 'pip download failed' }
    & $venvPython -I (Join-Path $PSScriptRoot 'lock_requirements.py') $wheels $lock
    if ($LASTEXITCODE) { throw 'writing the lock failed' }
    Write-Host "Review and commit $lock, then run the build again without -RefreshLock."
    exit 0
}

if (-not (Select-String -LiteralPath $lock -Pattern '--hash=sha256:' -SimpleMatch -Quiet)) {
    throw "requirements-build.lock has no hashes yet. Run: packaging\build_exe.ps1 -RefreshLock"
}
& $venvPython -m pip install --only-binary=:all: --require-hashes -r $lock
if ($LASTEXITCODE) { throw 'pip install from the lock failed' }

$arguments = @((Join-Path $PSScriptRoot 'build.py'), '--repo', $repo, '--build-root', $BuildRoot)
if ($AllowDirty) { $arguments += '--allow-dirty' }
if ($KeepFfmpeg) { $arguments += '--keep-ffmpeg' }
if ($NoExample) { $arguments += '--no-example' }
if ($Public) { $arguments += '--public' }
foreach ($token in $DenyToken) { $arguments += @('--deny-token', $token) }
if ($SkipZip) { $arguments += '--skip-zip' }
if ($SignThumbprint) { $arguments += @('--sign-thumbprint', $SignThumbprint) }
& $venvPython @arguments
exit $LASTEXITCODE
