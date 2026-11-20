# Standalone Windows program

Maplebot can be built into a portable Windows folder that needs no Python, PowerShell or Claude:
the settings app and map editor, the elevated controller (`scripts/run_active.py`), the
read-only relay (`scripts/relay_client.py`) and the optional supervisor
(`scripts/supervise_farming.py`). The controller is the same code, executed from its bundled
source, so every safety behaviour in [DEVELOPMENT_HANDOFF.md](DEVELOPMENT_HANDOFF.md) is
unchanged. Consent at the Windows administrator prompt stays with the user. Private settings,
map profiles and logs are never bundled.

The user-facing instructions are in [packaging/README-FIRST.txt](../packaging/README-FIRST.txt),
which ships beside the exes.

## Layout

The zip `Maplebot-<version>-<commit>[+dirty][-personal]-win64.zip` holds one top-level
`Maplebot\` folder (extract it into `%LOCALAPPDATA%\Programs\`, or `C:\` when the Windows user
name is not ASCII):

| Path | Owner | Notes |
| --- | --- | --- |
| `Maplebot.exe` | program | windowed, asInvoker, no DPI declaration: `gui` (default), `controller [--wait-for-start]`, `relay` |
| `maplebot-cli.exe` | program | console: `start`, `stop`, `status`, `paths [--probe]`, `selftest`, `supervise`, `restore-example`, `import-settings`, `version` |
| `_internal\` | program | PyInstaller runtime; `_internal\scripts\` holds the three scripts as data |
| `config\default.yaml`, `config\user.example.yaml`, `monster\`, `assets\` | program | listed with hashes in `maplebot-install.json` |
| `examples\construction-site\` | program | only when the build includes the example |
| `config\user.yaml`, `config\map_profiles\`, `log\` | user | created at run time, never shipped, kept on upgrade |

The root is the folder that holds `Maplebot.exe` (`maplebot/paths.py`). There is no home
override. The root must not contain `*.py`, `*.pyc`, `*.pyd`, `*.pth`, a Python package folder
or a `scripts\` folder: `run_active.py` puts the root first on `sys.path`, so the frozen
controller, relay and supervisor refuse to start when such files exist (exit 5).

## How the modes run

- `maplebot/app.py` dispatches the modes. `run_script` compiles a bundled script with a virtual
  `__file__ = <root>\scripts\<name>` and executes it as `__main__`, so `ROOT`, `chdir`, the log
  paths and the per-root mutex resolve to the install folder.
- **Start active** (settings app) and `maplebot-cli start` use `maplebot/launcher.py`:
  1. refuse (183, no signal file cleared) when the global mutex `Local\MaplebotController`,
     this folder's `Local\MaplebotActive_<hash>` mutex, a fresh running status, or a
     source-checkout controller process is found. A Python process whose command line names
     `run_active.py`, a live `verify_*`/`watch_*` tool or `maplebot run`, or whose command
     line cannot be read (an elevated controller), counts. This fails closed;
  2. clear the same seven start signals as `start_active.ps1`;
  3. `ShellExecuteExW("runas", Maplebot.exe, "controller")` on a worker thread
     (`SEE_MASK_NOASYNC` only, so Windows dialogs stay visible).
- The elevated `Maplebot.exe controller` takes the global mutex first (a duplicate exits 0
  before touching any user file), refuses when not elevated (4), repeats the foreign-process
  scan with full rights, applies the shadow guard (5), creates the user layout and runs
  `run_active.py`, which takes its own per-root mutex exactly as before.
- Stop, Pause, Hold, Resume and Reload use the same signal files under `<root>\log`.

### Source checkout differences

`maplebot/gui.py` now starts the controller through `launcher.py` in source mode too. The
program and arguments are the same (`%USERPROFILE%\.venvs\maplebot\Scripts\pythonw.exe -u
<abs root>\scripts\run_active.py`), but:

- Start refuses more often: besides the fresh-status rule it checks the per-root mutex, the
  global mutex and the foreign-process scan. An unrelated elevated `python.exe` therefore
  blocks Start, with its pid in the message.
- With a stale status but a controller still holding its lock, Start refuses (183) and leaves
  that controller's pending signals alone. The script used to clear them and launch a duplicate
  that then exited.
- Elevation uses `ShellExecuteExW("runas")` instead of `Start-Process -Verb RunAs`; a second
  click while the prompt is open is ignored.
- The relay command is unchanged. The change applies only after the settings app restarts.

`scripts/start_active.ps1` remains a developer tool. It does **not** probe the global mutex or
foreign processes. Do not run it, or `python scripts/run_active.py`, while an exe controller
runs; `tests/test_launcher.py` pins its signal list against `launcher.START_SIGNALS`.

## Moving settings into an install (this PC)

`maplebot-cli import-settings <other Maplebot folder> [--force]` (or **Import settings...** in
the settings app) copies `<folder>\config\user.yaml` and `<folder>\config\map_profiles\*` into
the install's `config\`. It:

- refuses while any controller runs: this install's checks (both mutexes, status, foreign
  scan) and the source folder's mutex and status;
- refuses unless the install's `user.yaml` is still the empty seed and `map_profiles\` is
  empty, unless `--force` is given (the settings app never forces);
- parses the YAML and every profile JSON before writing anything;
- writes every file through `<name>.import.tmp` and `os.replace`, profiles first;
- never deletes: with `--force`, a different existing file is first copied to
  `<name>.before-import-<time>` (which is not loaded as a profile);
- warns when the imported nametag or appearance image is not present under the install.

Close the settings app before the CLI import (or use its button). If it stays open, its next
**Save & apply** or **Start active** detects that a form-managed key (keys, HP thresholds,
nametag, monsters, circuit) changed in `user.yaml` since the form was loaded, reloads the form
and refuses that save, so the stale form never overwrites the imported settings.

An `-AllowDirty` build stages untracked files only from `maplebot/` and `packaging/`; files
under `assets/`, `monster/` and `examples/` ship only when tracked, so a nametag crop that
**Browse...** copied into a checkout's `assets\` is never bundled. `-Public` refuses a dirty
tree.

The import is a run-time action on the user's PC. Private files never enter a build:
`packaging/stage.py` denies `config/user.yaml*`, `config/map_profiles/**`, `log/**` and backups
on top of its allowlist, and `audit_dist.py` and `selftest --strict` fail when they appear in
the dist folder.

## Building (developer)

Prerequisites: `C:\Python313\python.exe` (CPython 3.13.1, the live venv's base), git, and
network access for the first package download. Nothing is installed into the live venv.

1. Commit the packaging work. A dirty tree is refused unless `-AllowDirty` (the build is then
   stamped `+dirty` and staged from the working tree instead of `git archive HEAD`).
2. Generate the hashed lock once (downloads wheels into the build root):
   `powershell -NoProfile -File packaging\build_exe.ps1 -RefreshLock`, review and commit
   `packaging\requirements-build.lock`. Verified 2026-11-18: PyInstaller 6.16.0
   (hooks-contrib 2026.10) on CPython 3.13.1 with numpy 2.5.4 and opencv-python 5.0.0.93 builds,
   passes the audit and `selftest --strict`, and passes the relocation check (V5). The lock
   lists 15 wheels.
3. `powershell -NoProfile -File packaging\build_exe.ps1 -Clean`. Use `-BuildRoot C:\mbuild`
   for builds that leave the PC (shorter, user-neutral paths). `-NoExample` drops the
   character-specific example and assets. `-Public` additionally drops `assets\README.md` and
   `monster\*\SOURCE.md` and fails on private tokens; it fails today by design (see phase 2).
4. Output in the build root: `dist\Maplebot\`, the zip, `SHA256SUMS.txt`, `selftest.json` and
   `build-requirements.txt`. Distribute only the zip, never a used install folder.

`packaging/build.py` fails the build when: the build interpreter is the live venv or can
import `maplebot`; the repository changes during the build (git status, tracked config
hashes, new `log/maplebot_*.log`); untracked config changes ("rerun while the session is
idle"); the audit fails (forbidden or shadowing files, scripts, manifest hashes, PE subsystems,
an RT_MANIFEST that is not asInvoker + longPathAware or that declares DPI awareness, 220 MB
budget, private tokens when public); the strict selftest fails from a temporary folder with a
stripped environment; the dist tree changes during the selftest; or `version` lacks the commit.
PyInstaller runs as `python -I -m PyInstaller` with `PYTHONPATH`, `PYTHONHOME`,
`PYTHONSTARTUP` and `PYTHONUSERBASE` removed.

## Verification

Offline, safe while a live controller runs (they create nothing in the repository's `config\`
or `log\`):

```powershell
$python = Join-Path $env:USERPROFILE '.venvs/maplebot/Scripts/python.exe'
& $python -m pytest -q tests/test_paths.py tests/test_first_run.py tests/test_launcher.py `
    tests/test_app_entrypoints.py tests/test_gui_launch.py tests/test_packaging_stage.py `
    tests/test_selftest.py tests/test_import_settings.py --basetemp (Join-Path .pytest_cache ('claude-' + [guid]::NewGuid().ToString('N')))
& $python -m maplebot.app version
& $python -m maplebot.app paths        # read-only; --probe writes and removes probe files
& $python -m maplebot.app selftest     # runs in a temporary folder
& $python -m maplebot.app status       # reports the live controller, if any
```

Never run `gui`, `start`, `controller`, `relay`, `supervise`, `restore-example` or
`import-settings` against the repository or a real install from an automated session.

After a build, check a copy in `%TEMP%\maplebot-relocate-<guid>`: `paths --probe` (writable,
probe files removed), `status` (the foreign scan reports a running source controller, which
is correct), `stop` (only `log\stop-active` appears), `supervise --help`; then delete the copy.

End-to-end (the user's job): optionally in Windows Sandbox, then on the host only after the
source controller has stopped (`active-status.json` shows `running: false` and its `pythonw`
process has exited). Approving the administrator prompt is the user's action.

## Upgrade

Stop the controller and any supervisor, close the settings app, make sure no `Maplebot.exe`
or `maplebot-cli.exe` runs (their `_internal` files are locked), delete the two exes and
`_internal\`, then extract the new zip into the same parent folder. `config\user.yaml`,
`config\map_profiles\` and `log\` are kept; edits to shipped files are replaced (the settings
app logs which ones to `log\gui.log`).

## Known risks

- **Two controllers.** Covered by the per-root mutex, the global mutex and the fail-closed
  foreign scan (unelevated at Start, elevated inside the controller). Remaining gap: starting
  `start_active.ps1` or `run_active.py` by hand while an exe controller runs.
- **Virtual `__file__`.** If `run_active.py` gains another `__file__`-relative path it would
  point at the missing `<root>\scripts`; the selftest text pins fail the build.
- **User-writable root on `sys.path` in an elevated process.** Shadow guard, audit and
  README warning; phase 2 drops the insert when frozen.
- **Non-ASCII install paths.** OpenCV opens narrow paths: the settings app warns, the selftest
  decodes images by absolute path, the character-image picker refuses non-ASCII names.
- **DPI and long paths.** The audit keeps the exe manifest equal to `pythonw.exe` (no DPI
  entry, longPathAware), because `capture/window.py` and `gui.py` set DPI awareness at run time.
- **Mark of the Web.** Unblock the zip; the settings app and selftest warn when the exe still
  carries `Zone.Identifier`.
- **Privacy.** Default builds are personal. `-Public` is blocked until phase 2 neutralises the
  greeting in `run_active.py`.
- **Antivirus.** Onedir, no UPX, a version resource, optional Authenticode signing and
  `SHA256SUMS.txt`. Never add broad exclusions.

## Phase 2 (after coordination)

- `run_active.py`: replace the character-specific greeting (unblocks `-Public`).
- `run_active.py` `__main__`: also take `Local\MaplebotController`, guarded so the frozen
  controller does not collide with `app.py`'s copy (or move the lock there), keeping the
  duplicate-before-user-files order.
- Optionally `ROOT = paths.app_root()` in `run_active.py` and `relay_client.py` without the
  `sys.path` insert when frozen; update the selftest pins.
- README.md and DEVELOPMENT_HANDOFF.md: link this page.
