# Maplebot: Claude Code handoff

Read [docs/DEVELOPMENT_HANDOFF.md](docs/DEVELOPMENT_HANDOFF.md) before changing runtime behavior. It records the current design, live verification, limitations, and next priorities. [README.md](README.md) explains the settings app and map editor.

## Working directory and setup

- The working Windows checkout is `%USERPROFILE%\source\repos\maplebot`. The WSL checkout at `~/Dev/maplebot` is a separate checkout; do not assume its files or running processes are synchronized.
- The Windows interpreter is `%USERPROFILE%\.venvs\maplebot\Scripts\python.exe`. Run commands from the repository root. Live capture/input must run on Windows.
- Reference implementations are available locally at `~/Dev/maplestory-auto-ds` and `~/Dev/MapleStoryAutoLevelUp`. They are research references, not runtime dependencies.
- Install with `scripts/setup_windows.ps1`, then use `scripts/open_settings.ps1`. `scripts/start_active.ps1 -WaitForStart` starts the controller paused. Windows consent, if shown, must be handled by the user.
- Use `scripts/run_active.py` for live operation. The legacy `python -m maplebot run` command bypasses the settings app's startup-return coordinator.
- Keep local `config/user.yaml`, `config/map_profiles/`, logs, and recordings out of commits. The versioned [Construction Site example](examples/construction-site/README.md) preserves this handoff's calibration without overwriting live settings.

## Check the running session before touching it

The controller was left farming at the end of the last supervised run. This is historical context, not a current process guarantee. Inspect `log/active-status.json`, its timestamp, and the actual process before taking action. Do not import `scripts/run_active.py` for inspection: it has runtime side effects.

- **Hold farming** stops movement/combat and retains guarded HP protection. Use the settings app or `EditorBridge.hold_farming()` when work requires holding the character.
- **Pause all**, **Stop**, opening the editor, and temporary connection tests do not provide the same HP protection. Do not leave the character exposed assuming a full pause heals it.
- Input and HP protection require fresh capture, a recognized map, and the game in front. Do not remove these checks to make tests progress.
- Preserve explicit user Pause/Stop. Never run two live controllers or a standalone potion tester alongside the active controller.
- Prefer existing bridge/runtime controls over ad hoc key injection. Do not restart the controller merely to edit documentation or run offline tests.

## Core invariants

1. New frame sequence and capture timestamp gate every input grant. Duplicate frames do not advance navigation or renew a lease. The keyboard worker independently expires commands and short movement pulses.
2. Navigate in the saved reference minimap coordinates: canonical position = raw position + validated alignment offset. Keep profile identity, reference hash, capture signature, and coordinate metadata consistent across editor, tests, and runtime.
3. A route index is not landing proof. Confirm actual supported destination position over fresh observations; retain the existing departure/descent, dwell, and walk-away checks where applicable.
4. Resume returns physically to point 1 through saved connections. Block recovery is bounded; failed recovery retains guarded HP Hold. Never fake recovery by resetting the waypoint index alone.
5. HP protection counts actual sent taps and checks subsequent readings. A queued key is not proof of healing. Keep persistent failure latches and the distinction between farming Hold and full pause.
6. Attack only on fresh valid detections. Loot uses a bounded positional heuristic; X key counts are not item counts. Do not report kills, loot, or potion success from request counts alone.
7. Preserve strict platform support, transition priority, and cancellation on changed map/settings/focus. Do not loosen matching or landing tolerances to hide a bad route.

## Main code

| Area | Files |
| --- | --- |
| Desktop settings, relay, editor | `maplebot/gui.py`, `maplebot/map_editor.py`, `maplebot/editor_bridge.py` |
| Live lifecycle and tests | `scripts/run_active.py`, `maplebot/editor_test_runtime.py` |
| Frame processing and input | `maplebot/bot.py`, `maplebot/capture/window.py`, `maplebot/control/keyboard.py`, `maplebot/types.py` |
| Map identity and coordinates | `maplebot/map_profiles.py` |
| Route compilation and movement | `maplebot/nav/editor.py`, `maplebot/nav/circuit.py` |
| Startup and blocked-route recovery | `maplebot/recovery_runtime.py`, `maplebot/nav/recovery.py` |
| HP protection | `maplebot/health_protection.py`, `maplebot/health.py` |
| Combat and collection | `maplebot/brain/fsm.py`, `maplebot/vision/monsters.py`, `maplebot/vision/player.py`, `maplebot/nav/loot.py` |

## Offline verification

```powershell
$python = Join-Path $env:USERPROFILE '.venvs/maplebot/Scripts/python.exe'
New-Item -ItemType Directory -Path '.pytest_cache' -Force | Out-Null
$testBase = Join-Path '.pytest_cache' ('claude-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q --tb=short --basetemp $testBase
```

Use a unique test directory on Windows to avoid old file locks. Run focused regressions while developing, then the suite before delivering behavior changes. Tests must remain offline; never make pytest start live input. Keep small regression crops under `tests/fixtures/`, with their provenance, rather than full game/desktop recordings.

Live checks are separate: `verify_potions.py`, `verify_route.py`, `verify_lap.py`, `watch_full_circuit.py`, and `watch_circuits.py` under `scripts/`. Read their preconditions and the handoff before invoking them. The two watchers currently assume the specific Construction Site circuit; they are not generic map validators.

## Current priorities

Improve collection confirmation and target reachability on separate stairs/platforms, then generalize map-specific live verification. Preserve the established safety behavior while doing so. No outstanding live test should be inferred from old chronological notes under `log/`; use the verified results and limitations in the handoff.
