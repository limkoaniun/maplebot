# Development handoff — 15 November 2026

This document describes the development state handed to Claude Code. It supersedes older chronological session notes, but is not a live status feed. The Windows controller was left farming after the last supervised check; inspect current status before interacting with it.

## Product and user intent

Maplebot observes screen pixels and sends ordinary keyboard input. The user wants effective farming with actual kills and item collection, HP above 50%, configurable HP/MP controls, and recovery from failed movement. The desktop app includes settings, game/detection relays, and a clickable minimap that opens a larger route editor. Users draw platform boundaries, slopes, ropes, jumps, drops, and route stops themselves.

The minimap is used for position, map recognition, and following the saved graph. The program does not automatically discover an optimal farming route for any map. Different maps need their own profiles, reachable connections, and supervised validation.

The active development checkout is `%USERPROFILE%\source\repos\maplebot`; the upstream is the private repository `limkoaniun/maplebot`, branch `master`. The WSL checkout is separate. Do not overwrite the Windows checkout from it without comparing changes.

## What is included and what stays local

The repository includes source, launchers, offline regression tests, small game-image fixtures, character templates, monster sprites, and a saved Construction Site example. The example is an explicit snapshot, not automatically loaded configuration.

Live `config/user.yaml`, `config/map_profiles/`, `log/`, `screenshot/`, and `video/` remain ignored. They contain mutable settings, evidence, signals, process status, and recordings. No credentials or raw desktop/game recordings are needed to develop the application.

On this PC, existing local files remain the source of truth for the running character. On a fresh clone, read [examples/construction-site/README.md](../examples/construction-site/README.md) to restore the saved setup deliberately. Do not copy it over an existing configuration as part of normal startup.

The historical `scripts/calibrate_player.py` helper is local-only: it depends on an old capture in `log/` and overwrites a character asset. It is not a supported general calibration command.

## Setup and entry points

1. Use Windows where the game runs, with Python 3.10 or newer and Tk available. The tested environment uses Python 3.13.
2. Run `scripts/setup_windows.ps1`. It creates `%USERPROFILE%\.venvs\maplebot` and installs the project. If editable installation fails, explicitly install from a normal Windows checkout with that interpreter: `python -m pip install -e .` and install `pytest` for development. `windows-capture` is declared in `pyproject.toml` and is needed for the current graphics backend.
3. Open `scripts/open_settings.ps1`. Review keys, health settings, character tracking, capture size, and map profile before Start active.
4. To start a controller without immediately farming, use `scripts/start_active.ps1 -WaitForStart`. It uses Windows elevation because the game requires matching privileges. The user handles any Windows consent prompt.
5. Keep MapleStory in front for input. F1 pauses/resumes; F12 stops. Closing settings does not stop the separate controller.

The `python -m maplebot run` CLI remains useful for offline replay and diagnostics, but does not run the full startup-return coordinator. Use the supported settings/controller path for live circuit testing.

The controller has a per-checkout Windows mutex. `log/active-status.json` contains its PID, timestamp, pause state, map identity, capture metadata, route position, health, recovery state, and actual key counters. `log/active.log` records errors. A stale status file does not establish that a process is still running.

`EditorBridge` coordinates profile changes and controlled requests. Runtime signals include `hold-farming-active`, `resume-active`, `pause-active`, `stop-active`, `reload-active`, and `return-to-start-active`. Use the GUI or bridge where available; do not create competing controllers or inject keys around its guards.

## Architecture and behavior to preserve

### Capture, map identity, and input leases

`capture/window.py` supplies fresh capture metadata, including a sequence and monotonic source timestamp. `bot.py` rejects stale or duplicate observations for control. `control/keyboard.py` independently expires command permissions; its worker cannot extend a short pulse just because the next vision frame is late. `Command.valid_for_s` carries bounded movement leases. The default frame/command age limits are 0.75 seconds.

`map_profiles.py` recognizes terrain and estimates translation against the saved reference. Route coordinates are canonical reference pixels, not raw scrolling minimap pixels. The conversion is `canonical = raw + offset`. Profile key, reference hash, crop/capture signature, and coordinate metadata must stay consistent through the bot, editor, overlay, and test bridge.

Recognition uses terrain information with marker masking and rejects ambiguous matches. Thresholds were not weakened to accommodate scrolling. Blank images, wrong maps, mirrored terrain, missing tower, ambiguous fragments, and excessive translations have regressions. Brief recognition loss releases all keys immediately. Resuming the same activity requires fresh confirmation of the same map within the bounded two-second reacquisition window; longer or incompatible loss holds operation.

### Route graph, transitions, and physical recovery

`nav/editor.py` compiles user-drawn platforms and connections into waypoints. `nav/circuit.py` runs walks, climbs, jumps, drops, and ramps. A platform may be horizontal, sloped, or piecewise sloped. Arrival must be supported by actual fresh position observations, with stable height and dwell; an incremented index alone is insufficient proof.

Recent fixes address three observed failures:

- Short alignment pulses now expire independently of capture latency. Construction Site uses 0.04-second movement and 0.45-second release intervals, with 0.4-pixel takeoff alignment tolerance.
- A jump destination is latched only after the existing legitimate landing predicate succeeds. Later horizontal drift on the same supported floor no longer steers the character backward into the gap. Unsupported positions, missing tracking, suspension, and retry clear the proof.
- Ramp arrival tolerates horizontal knockback within the destination's strict support. Small backward knockback on the source stair gets a bounded wait and one local realignment retry after three fresh stable samples. Falling below the surface does not qualify. Original transition deadlines still apply.

`recovery_runtime.py` owns normal Start/Resume return behavior. `nav/recovery.py` compiles a physical path from a measured supported position through saved connections to point 1. Startup does not consume the ordinary blocked-route recovery allowance. It does not pretend to return by setting the index to zero.

Blocked-route recovery has a budget of one return per completed lap and a bounded active return duration. A successful full original-route wrap replenishes the allowance. Failure or exhausted recovery leaves guarded HP Hold. Known lower-floor falls may rejoin the saved route; these interrupted passes are not counted as clean full laps by the watcher. Unknown/ambiguous support or missing graph connections must not trigger invented movement.

### Health and interruption semantics

**Hold farming** releases movement/combat and can continue HP-only protection on fresh, recognized, focused frames. **Pause all**, **Stop**, and opening the map editor release all input, including potions. Temporary connection tests also disable normal potion automation. These states must remain distinct in both code and UI.

`health_protection.py` observes actual HP taps and subsequent fresh bar readings. Ineffective taps are bounded; three without improvement produce a persistent attention state. Missing/unreadable HP, no usable key, and zero HP do not count as success. Recovery requires restored HP and explicit Resume after a terminal health failure. The control lock/generation guard prevents stale movement grants from racing Pause, Stop, tests, and HP Hold.

Current character settings trigger HP refill below 70%, target 75%, and require at least 50% for farming. The HP protection retry interval is one second, with at most three ineffective actual taps. MP triggers at 20%; its ordinary retry interval is three seconds and retry count is two. Movement may suspend while HP refills. No protection works when focus or freshness guards disallow input.

### Combat and loot

Monster detection uses the existing template library with shape and color validation. Padding the search crop by the template size fixes clipped monsters near the search boundary, while accepted centers remain inside the original search area. The binary-mask grayscale optimization preserves the intended matching formula; a fallback remains available for other masks.

The brain retains fresh engaged targets, waits briefly for reacquisition, and may pursue a nearby target for a bounded period during ordinary walking. Transitions and recovery take priority. Attacks require a current valid target in melee range; no stale-target attack is justified by a remembered detection.

`nav/loot.py` is a positional post-combat heuristic, not a general ground-item detector. It estimates a target from a recently disappeared monster, waits for drops, approaches briefly, and dwells while pressing X. A live target can take priority again. The pickup counter only counts input, not collected items.

Committed pickup work can survive a short character-visibility gap of at most 0.6 seconds, with all input stopped during the gap. Resumption requires the same fresh profile/reference/floor/route context. Changing focus, map, capture, floor, transition, recovery, or explicit hold cancels it.

The final pickup fix stores original platform bounds and brakes before approaching close to a goal or outward edge (five minimap pixels), or reversing direction. It uses 0.45-second release and 0.04-second leased pulses. Leaving the original platform cancels the attempt even if a different overlapping stair also contains the target. The local setup permits a four-second approach and 1.2-second settling time.

## Construction Site calibration

The saved profile is `kerning-city-construction-site-7bcaa4fc6a`. Its reference SHA-256 is `b131badde8048e46b7f79abe17f4be1b6e966dc8295619fca0834e38aa964427`.

The prepared frame is 1366 × 768; minimap crop is `[20, 82, 210, 128]`. The 44-waypoint loop has 17 platform transitions. Point 1 is `(49, 105)` on the ground. The main floor heights are 105, 88, 71, 55, 38, and 22. Three gap jumps take off at X=91 and land at X=85. Piecewise stairs, a barricade, entry pipes, right-side drops, and bottom brick stairs provide explicit traversal/recovery connections. Use the actual profile graph instead of reconstructing it from this summary.

| Control | Binding |
| --- | --- |
| Double Stab | Z |
| Normal attack | Alt |
| Jump | C |
| HP / MP potion | Q / A |
| Pickup | X |

Current monster names are blue snail, octopus, orange mushroom, stump, and green mushroom. Character tracking uses the included nametag/head templates. These images and offsets are character-specific. The example must be recalibrated for another character, display setup, or map; it is not a universal preset.

## Verification completed

The previous production code passed **1,055 offline tests**. The handoff also promotes the previously local multi-lap observer and its regressions into `scripts/watch_circuits.py` and `tests/test_watch_circuits.py`. The complete handoff source passed **1,069 tests in 19.79 seconds** from a clean export of the Git index, without local configuration, profiles, or log artifacts. The clean-folder check also verified required assets, example reference identity, and handoff links. Test instructions below explicitly create the parent test-cache directory, which a fresh checkout does not contain.

Supervised game evidence is local, not uploaded as recordings:

| Run | Verified full laps | Result |
| --- | --- | --- |
| `log/three-circuits-v3.summary.json` | 2 | All 44 points and 17 landings per completed lap; minimum recorded HP 56.73%; 20 HP and 7 MP taps followed by visible increases. Interrupted passes were excluded. |
| `log/final-circuit-v4.summary.json` | 1 | After the final loot-braking change: all 44 points, all 17 landings, actual last-to-first wrap, and stable point 1. About 363 seconds including startup return; minimum recorded HP 63.46%; 3 HP and 2 MP taps with positive responses. No fall/rejoin/recovery/map gap during this completed lap. |

These are **three complete laps across runs, not three consecutive laps on the final version**. The final observer left farming active on success. Do not treat this historical result as proof that an indefinitely running session remains healthy.

Actual kills were visually confirmed. Actual pickup evidence included item/mesos gain messages; the final run specifically showed Tree Branch +1 and mesos gains. At a separate near-edge pickup, braking prevented overshoot, but the branch landed on a higher stair and its collection was not confirmed. Do not combine those observations into a claim that every drop was picked up.

Standalone potion checks previously measured Q restoring 495/532 HP to 532/532 and A restoring 289/436 MP to 436/436. These recorded results are not interchangeable with current evidence after changing keys, profile, reference, or capture calibration.

Detailed local evidence includes `log/final-circuit-review-v4/findings.json`, `log/final-circuit-review-v4/potion-observations.json`, and the adjacent traces/images. Older `log/current-session-handoff.md` contains superseded plans and process IDs; read its latest completion section before using any older entry. Avoid interpreting key counts as kill/item counts or ambiguous tiny EXP text as reliable numeric gains.

## Running tests and supervising a future live change

From the repository root:

```powershell
$python = Join-Path $env:USERPROFILE '.venvs/maplebot/Scripts/python.exe'
New-Item -ItemType Directory -Path '.pytest_cache' -Force | Out-Null
$testBase = Join-Path '.pytest_cache' ('handoff-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q --tb=short --basetemp $testBase
```

The offline suite covers stale capture, command leases, pause/health races, map recognition and canonical coordinates, profile/editor serialization, startup return, blocked-route recovery, jumps/ramps/drops, target boundaries, loot cancellation/braking, and evidence-watch continuity. Use recorded or synthetic regressions for observed failures before repeating a live circuit.

| Live helper | Preconditions and behavior |
| --- | --- |
| `scripts/verify_potions.py` | Exclusive controller ownership; uses the same mutex and rejects an already running controller. Requires measurable HP/MP deficit. Sends controlled taps and records fresh responses. Optional `--start-controller-paused` starts the normal controller only after success. |
| `scripts/verify_route.py` | A paused controller, a matching saved map, supported source position, connection IDs, and sufficient HP. Uses bounded editor tests and destination/walk-away evidence; normal farming and potions are off during tests. |
| `scripts/verify_lap.py` | Checks route chunks with the connection-test machinery. Resuming a later chunk is explicitly a partial result. |
| `scripts/watch_full_circuit.py` | Watches normal farming with potion automation. Requires matching prior potion evidence and the specific 44-point/17-transition Construction Site route. Requests guarded Hold when finished. |
| `scripts/watch_circuits.py` | Preserves consecutive wrap boundaries and validates 1–3 complete laps. `--start` requests one normal Resume after preflight. `--timeout` is bounded to 1,800 seconds. Success leaves farming active; failure requests guarded HP Hold and preserves explicit full pause. |

Example for a future authorized, supervised test after reading the preflight requirements:

```powershell
& $python scripts/watch_circuits.py --start --laps 3 --timeout 1800 --output log/claude-circuits.jsonl
```

The watchers currently hardcode route size, transition count, and starting coordinates. Do not apply them unchanged to another map or weaken their checks to make a new route pass. Freshness, profile/reference/capture identity, ordered route progression, landing proof, and stable ground return are part of the result.

## Known limitations and next development priorities

1. **Prove collection and improve drop targeting.** There is no general item detector or verified item-count metric. Drops on nearby stairs may remain uncollected; late-spawning items can outlast a bounded attempt. Add small labeled recordings and explicit collection evidence before expanding movement toward guessed drops.
2. **Model target reachability across surfaces.** A target on a lower stair near the barricade could still attract an ineffective attack. Refine vertical/platform eligibility without allowing stale attacks or expanding the current melee range indiscriminately.
3. **Generalize live verification and map onboarding.** Derive expected transitions/start from a validated profile, retain per-transition proof, and clearly separate partial runs, rejoined falls, blocked returns, and clean full laps. Unknown maps must still open an empty editor and require calibration.
4. **Expose useful farming measures.** Count verified events separately from key presses; record time spent fighting, looting, traversing, and recovering. Current success proves observed behavior, not optimal EXP or items per hour.
5. **Keep settings and recovery maintainable.** Several features provide defaults in code rather than all being listed in `config/default.yaml`. Consolidate/document configuration deliberately, with round-trip tests, without changing existing saved profiles silently. Expand coverage of return paths from mapped starting positions rather than promising recovery from arbitrary terrain.

Do not solve these issues by disabling focus/freshness/map guards, increasing tolerances until a drawing appears to pass, repeatedly resetting to point 1 without moving, or looping recovery without a budget.

## Reference implementations and local geometry research

The original rewrite was informed by `MapleStoryAutoLevelUp` and `maplestory-auto-ds`. Local reference checkouts are `~/Dev/MapleStoryAutoLevelUp` and `~/Dev/maplestory-auto-ds`; consult their actual code when borrowing an approach. Their existence does not make them dependencies or establish that their behavior is safe for this game's calibration.

Monster/template provenance is recorded in fixture documentation and `monster/octopus/SOURCE.md`. Death poses are kept below `_excluded_death/`, outside the library's direct `*.png` scan, so they are not live targets.

Local Classic map foothold research informed the piecewise Construction Site stairs. Useful local files are `log/construction-classic-calibrated-geometry.json`, `log/bottom-brick-stairs-geometry.json`, and `log/public-osms-cbt2-map10003010-decoded.json`. They are research artifacts, not runtime requirements. The old cached GMS foothold ground was incomplete; do not replace the measured graph from that cache without validation. Large downloaded map/canvas blobs and full recordings stay out of Git.
