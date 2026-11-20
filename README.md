# maplebot

Continuing development with Claude Code? Start with [CLAUDE.md](CLAUDE.md) and the
[development handoff](docs/DEVELOPMENT_HANDOFF.md). The calibrated
[Construction Site example](examples/construction-site/README.md) preserves the
current route and character settings for a fresh checkout.

A personal MapleStory auto-leveling bot that works purely from screen pixels and simulated
key presses. No memory reading, no injection. It follows a recorded route (or sweeps left
and right), detects monsters near your character by template matching, faces and attacks
them, drinks HP/MP potions, and walks itself out when stuck.

Design notes: `docs/superpowers/specs/2026-11-04-maplebot-design.md`.
It is a fresh rewrite informed by
[MapleStoryAutoLevelUp](https://github.com/KenYu910645/MapleStoryAutoLevelUp) and
`maplestory-auto-ds`, and it reuses their asset formats so their `minimaps/` and `monster/`
folders drop straight in.

## Setup (Windows, where the game runs)

```powershell
git clone <this repo> maplebot
cd maplebot
.\scripts\setup_windows.ps1        # creates %USERPROFILE%\.venvs\maplebot and installs deps
copy config\user.example.yaml config\user.yaml
```

Edit `config/user.yaml`. Everything you do not set falls back to `config/default.yaml`,
which documents every key.

## In-game preparation

1. Windowed mode. Any resolution works, but keep it fixed once routes are recorded.
2. Minimap visible (top-left) with its white frame.
3. For the default `player.locator: party_bar`, be in a party so the small red HP bar
   shows above your character. Alternatives: `nametag` (a PNG of your nametag on a pure
   green background) or `appearance` (a crop of your character).
4. Monster templates in `monster/<name>/*.png` (green or transparent background).
   Download them with `python -m maplebot fetch-mob "Orange Mushroom"`.
5. A route in `minimaps/<map>/map.png` + `route1.png`, `route2.png`, ... (optional; see
   below). Without a route the bot sweeps left and right inside the minimap bounds.

## Run

For the settings app, saved map profiles, and automatic return on Resume, use:

```powershell
.\scripts\open_settings.ps1
```

The original standalone runner below is for debugging. It does not use the
settings app's startup-return coordinator; use the settings app for live circuits.

```powershell
.\scripts\run.ps1 --map my_map --monsters green_mushroom,spike_mushroom
# or
%USERPROFILE%\.venvs\maplebot\Scripts\python.exe -m maplebot run --cfg config\user.yaml
```

Hotkeys: **F1** pause/resume, **F2** screenshot to `screenshot/`, **F12** quit. The debug
window shows the player fix, minimap dot, attack and search boxes, detections, HP/MP/EXP,
the current command, and a zoomed route preview.

Useful flags: `--no-input` (watch only, press nothing), `--no-viz`, `--source shot.png`
or `--source clip.mp4` to replay a capture offline, `--frames N` to stop after N frames.

## Local settings window

Open **Maplebot Settings** on the desktop, or run `scripts/open_settings.ps1`.
Save changes before starting. Start active launches the controller with the same
Windows privileges as this game; approve the Windows prompt yourself if it appears.
Keep MapleStory in front while running. F1 pauses/resumes and F12 stops it.

**Bot view** shows the actual detected character, monsters, minimap position, and
HP/MP/EXP readings. **Game view** is a separate live window capture and also works
while the controller is stopped. Closing settings closes its preview; use Stop or
F12 to stop the controller. The pickup counter counts key presses, not collected items.

The local example profile uses Z for Double Stab, Alt for normal attack, C to jump,
Q/A for HP/MP potions, and X for pickup. Its monster matches require both shape and
color agreement. Character tracking uses the nametag with a short, local head-image
fallback; missing capture or prolonged missing character/minimap detection stops input.
An already-started climb or drop can finish for up to two seconds using the current
minimap dot while the character is obscured; attacks remain disabled during this interval.

An optional `route.circuit` follows explicit minimap waypoints with walk, climb, jump,
drop, and ramp transitions. This is a configured lap, not automatic route optimization.
Its coordinates depend on the current minimap crop.
Combat and a bounded post-combat collection pass interrupt the lap and then resume it.
If a transition repeatedly fails, the circuit stops and reports the blocked point.

Enable **Circuit → Return to point 1 if route blocks** to recover through the
saved map connections. The controller confirms a stable starting platform, follows
a temporary return route, then confirms arrival at point 1 before restarting the
lap. **Return to point 1** also requests this action manually. Recovery suppresses
combat, pickup, and buffs. Health protection can suspend a return to refill HP
using only the HP key, then continue after a confirmed healthy reading. It requires
a recognized map, fresh captures, a visible minimap dot, and game focus.
Loss of focus or position suspends movement; a changed map, low/unreadable HP,
missing return path, or failed return stops it. The default limit is one return
per completed lap and 60 seconds of active return time. Exhausted or failed
recovery stays paused until the circuit is checked and reapplied. These settings
are saved per map; temporary connection tests never recover automatically.

**Start active** and **Resume farming** first return the character to point 1
through the saved map connections. Farming begins only after arrival is confirmed.
This also works from the mapped left entry pipes in Construction Site. If the
current position has no saved connection to the route, farming stays held and
shows the missing-path reason. A successful startup return does not consume the
lap's blocked-route recovery allowance.

If the character is not standing on a drawn platform when a return starts (for
example on undrawn pipes, hanging on a saved rope, just past a drawn end, or in
the air), the controller first tries to regain support: it waits until the
position is still, then holds Down on a saved rope, or sends short Left/Right
pulses toward the drawn floor directly underneath or the drawn platform beside
it. It never presses Up or Jump, walks at most `regain_max_walk_x` (16) minimap
pixels, and gives up after `regain_timeout_s` (20 s). The normal return starts
only once the strict platform check accepts the position. If the Left and
Right pulses both move nothing (the character is evidently not standing on a
walkable surface, for example hanging on a rope or ladder that the map does not
draw) and exactly one drawn floor lies underneath, it then holds Down alone in
short leases: Down climbs down a rope and lets go at its lower end, only
crouches on solid ground, and does not enter portals. That attempt ends after
1.5 s without descent. Next to the end of the drawn floor, where one direction
has under 2 pixels of room, that direction is pulsed for its full 6 s before it
counts as not moving, so the descent starts later there. If the character has
moved since the pulses, Left and Right are tried again at the new spot before
Down. This does not work from everywhere: a wide undrawn
surface, a spot with no drawn floor below or beside, or an undrawn spot where
neither Left, Right nor Down moves the character still ends in Hold farming
with a "Could not reach a drawn platform" reason; move the character or draw
that surface. Set `route.circuit.return_regain_support: false` to turn the
stage off. The saved-rope and the undrawn-rope descent were each seen once in a
supervised run; the walk-off is covered by offline tests only. None of this is
a general verification in the live game.

**Hold farming** releases movement and combat while keeping guarded HP protection
on. **Resume farming** starts the return described above. **Pause all** and **Stop**
also disable potion input.

The **Potions** tab has separate HP and MP thresholds. HP defaults to a 60%
trigger with a 50% minimum farming level; a refill pauses farming until HP reaches
the trigger plus the recovery margin (65% by default). MP has its own trigger and
uses the MP key from Controls. **Keep HP protection on after a route failure**
is enabled by default: an automatic route stop continues watching HP and may send
only the HP key. It never resumes a failed route by itself. **Pause all**, **Stop**,
opening the map editor, loss of map identity, and stale/unfocused input release keys.

HP protection counts actual key presses and checks subsequent fresh HP readings.
Three presses without observed improvement, an unreadable HP bar, a missing key,
or zero HP produce a persistent alert and keep farming stopped. No healing success
is inferred from a queued key request. A failed health check requires HP restored
to the refill target and an explicit Resume before farming can continue. These
checks cannot guarantee survival if potions are unavailable or input fails; check
the visible/audible alert promptly. HP protection is off during manual full pause
and temporary connection tests.

The current character profile uses a 70% HP trigger and 75% refill target, with
the minimum farming level at 50%. Its MP trigger is 20%. Nearby fresh monster
detections can briefly interrupt ordinary walking for pursuit; attacks still
require a current detection in melee range. Missing detections allow a short
stationary reacquisition period, and climbing/jumping transitions take priority.

## Drawing a route on the minimap

Click the minimap in **Bot view**, or choose **Circuit → Open map editor**. Opening
the editor pauses the controller. The enlarged map is a frozen snapshot so it stays
still while you draw. An unknown map opens an empty drawing instead of importing
the previous map's route. Use **Minimap crop** to select its minimap in the full
game image, then **Map settings** to select monsters and navigation calibration.

1. Use **Platform / boundary** to draw each walkable section from its left limit to
   its right limit. Set the line at the yellow player's walking height, not the
   grass edge in the map artwork. Use **Slope boundary** for a section whose ends
   have different heights. Separate ledges need separate platform lines.
2. Drag **Rope**, **Jump**, **Drop**, or **Ramp** between platform lines. Ropes and ramps work both ways;
   jumps and drops follow the direction you draw. A jump has a takeoff point and a
   landing point. Ropes and drops use the same horizontal position on both floors.
   A ramp uses horizontal movement along a drawn slope and waits for a stable
   destination height; it never substitutes a jump when the slope cannot be reached.
3. Click **Route stops** in the order you want to visit them. Enable **Repeat route**
   for repeated laps. The compiler connects stops only through the connections you
   drew and reports missing connections during **Preview**.
4. Use **Select / edit** to drag lines, endpoints, or route stops. Undo and redo are
   available. **Save draft** stores a named map profile without changing the route;
   **Use route** validates and applies it. Both leave the controller paused.

Calibrated stair boundaries can contain several joined sections, including flat
shelves. Select one to show its interior handles; drag a handle to adjust that
part of the surface, or drag the line to move the whole boundary. Handles stay in
left-to-right order. After moving a boundary, check its attached stops and
connections with **Preview** before applying it.

To check a connection, stand on its starting platform, select it in the editor, and
choose **Test selected connection**. Switch to MapleStory to let the test move.
Combat, pickup, and potions are disabled during the test. It runs once, confirms the
destination by walking a short distance along the floor and waiting, then restores
the normal configuration paused. Platforms too narrow for this check are rejected.
**Stop test** cancels;
**F12** stops the controller. Drawn jump distances still need to be achievable by
your character—preview checks connectivity, not the game's jump physics.

Profiles and their reference images are stored locally in `config/map_profiles/`.
Each profile owns its minimap crop, monster settings, navigation calibration, and
route geometry. Character keys, skill range, and health settings remain shared.
Live operation checks map identity on every fresh frame using each saved map's
own crop. A known different map loads its complete settings paused; an unknown or
ambiguous map releases input and requires calibration. Older geometry-only profiles
remain available but need explicit map settings before they can run. Keep minimap
size and game resolution consistent with calibration.

The controller aligns scrolling minimap terrain to the saved reference before
reading route positions. Navigation, the enlarged editor, and the minimap preview
share those reference coordinates. Ambiguous alignment still stops input; a moving
background behind the translucent minimap is not a route landmark.
After a brief obstruction, the same saved map can regain control only through
fresh confirmed readings within two seconds. No keys are sent during that gap.
A confirmed different map, changed settings, or longer loss keeps the bot paused.

Captured frames carry a sequence number and monotonic timestamp. Re-reading a
buffered frame does not advance the route or refresh input permission. The keyboard
worker independently releases keys and cancels queued actions when its command or
source frame expires (`safety.command_timeout_s` and `safety.max_frame_age_s`, both
0.75 seconds by default). Resume waits for a fresh observation before sending input.
Short positioning taps carry their own shorter deadline, so a slow vision frame
cannot turn a small correction into a long walk off a ledge.
On a recognized map, a current player dot can carry the drawn route through beams
that obscure the character. This mode permits movement only: attacks, pickup,
buffs, and potion requests stay disabled until the character is visible again.
Losing the current dot, map match, game focus, or fresh capture releases input.

For supervised route verification, `scripts/verify_route.py` accepts saved connection
IDs, and `scripts/verify_lap.py` checks the saved circuit in bounded chunks. Both
require a matching map and a paused controller, monitor HP, retain traces and game
images under `log/`, and leave the controller paused. These checks do not enable
combat or potion use; restore HP before testing. A drawing alone does not establish
that a jump is physically reachable.
After recovering to a chunk's starting platform, `verify_lap.py --start-chunk N`
checks the remaining sections. Its report explicitly identifies a partial lap;
earlier failed sections are never marked as passed by resuming.

## Recording a route

```powershell
python -m maplebot record-route my_map
```

Walk the whole map first so `map.png` is complete (press **F4** to save it), then press
**F1** to start painting, walk the route, and press **F3** at the end point: that writes
`route1.png` and starts `route2.png`. The colours are the same codes as the reference
project (`route.color_code` in `config/default.yaml`): red = walk left, blue = right,
orange/cyan = jump left/right, lime = jump down, magenta = jump, grey = climb up,
light-yellow = climb down, yellow = goal (switch to the next route). Touch up the PNGs in
any paint program if needed; the bot reads exact colours.

## Tuning

- `route.circuit.precision_walk` enables release-and-pulse approaches near walking
  stops. `jump_align_tolerance_x` and `drop_align_tolerance_x` can allow broader
  takeoff alignment while keeping `align_tolerance_x` precise for ropes.
  `jump_landing_overshoot_x` permits up to three extra minimap pixels in the travel
  direction, only with confirmed support and a stable landing on the destination.
- `attack.range_x/range_y` and `attack.deadzone_px` control when the bot attacks and when
  it stops walking toward a target. `attack.mode: aoe` uses a centred box instead.
- `attack.pre_transition_s` (default 0 = off, at most 10) lets the bot turn and attack a
  monster that is in range on its own platform while it is still lining up for a jump,
  ramp or drop, for that many seconds per waypoint. It never walks toward the target
  there, and nothing interrupts a transition once its takeoff key was sent. When the
  monster is not detected for a moment the bot waits without keys for up to 0.4 s
  before lining up again; that wait counts toward the same seconds. Not yet verified
  in the game.
- `attack.surface.enabled` requires a target's sprite to rest on the character's own
  platform surface, following stairs, so a monster on the stair or floor underneath is
  ignored. `loot.walk_speed_x` sizes pickup approach pulses from the distance and
  `loot.max_pending` remembers kill sites displaced by another fight. The
  `route.circuit.approach_*` keys speed up walking to route stops. Speed keys must be at
  least the real held walking speed in minimap pixels per second. `evidence.enabled`
  publishes on-screen EXP and gain-line observations in `log/active-status.json`. All are
  off by default and documented in `config/default.yaml`.
- `route.circuit.commit_budget`, `route.circuit.takeoff_far_approach` and
  `route.circuit.recovery_far_approach` (all off by default) change how long route
  transitions may take. With `commit_budget`, a ramp, jump, drop or rope climb that has
  already started is no longer stopped by the route point's general time limit; it gets
  its own limit from the moment it starts. A drop or rope climb is also stopped after
  two seconds without progress. A ramp is not, and by default it gets the route point's
  whole time limit again from its start. The time allowed for lining up before the
  start is unchanged. `takeoff_far_approach` lets the bot walk to a jump, ramp, drop or
  rope start with the same longer, distance-sized steps used for route stops. Each step
  is short enough to stop the configured margin before the end of the platform even if
  the character is knocked onto the start point while the key is down, so a start
  within about two pixels of a platform end (the gap jumps) keeps the short taps when
  it is approached toward that end. The last pixel and the short wait before the start
  are as before. `recovery_far_approach` allows the same steps on the way back to
  point 1 and when rejoining a platform. The last two need the `approach_*` keys. A
  saved map profile needs the keys in its `circuit_settings` too. Tested offline only;
  not yet verified in the game.
- `route.circuit.takeoff_zones` (off by default) lets the first try of a jump, ramp or
  drop start from a short stretch of the platform instead of from one exact spot, so a
  character that was pushed a few pixels aside does not have to creep back first. The
  bot works that stretch out from the drawn map: for a drop, the part of the floor where
  nothing else is drawn in or next to the way down; for a ramp, the part of the stair
  behind the saved start; for a jump over a small level gap, the last pixel or two
  before the gap. It still stands still for the usual moment before it starts, every
  landing is checked exactly as before, and a second try goes back to the saved spot.
  `takeoff_zone_kinds` (all of `[drop, jump, ramp]`) picks which kinds get a stretch;
  `[drop, jump]` keeps every ramp on its saved spot, which is the suggested setting for a
  first supervised run, because a ramp started further back is committed to the stair
  earlier and in offline tests that helped or hurt depending on where hits landed.
  `takeoff_zone_max_x` (8) limits how far the stretch may reach from the saved spot and
  `takeoff_zone_margin_x` (1.5) how close it may come to the end of a platform. Jumps
  onto stairs or other heights, ropes, and connection tests in the map editor always
  use the saved spot. A saved map profile needs the keys in its `circuit_settings`
  too. Tested offline only; not yet verified in the game.
- `route.circuit.arrive_on_destination` and `route.circuit.platform_rejoin` (both off by
  default) handle a character that was knocked onto a drawn platform the current route
  point does not expect. The first accepts a jump point when the character already stands
  on that jump's saved landing platform: it never presses Jump, waits for three fresh
  frames over half a second, walks to the landing spot and then needs the usual stable
  landing; if the knock happens while still walking to that jump's takeoff, the bot
  goes straight to this check. The second walks the saved connections back to the point's own platform (or
  to a jump or ramp takeoff) without changing the route point and without using the
  lap's return to point 1; it is limited by `max_platform_rejoins_per_lap` (3),
  `platform_rejoin_timeout_s` (20) and `platform_rejoin_max_transitions` (1). The bot
  does not attack or collect during such a rejoin. If it fails, the point times out as
  before. A saved map profile needs the keys in its `circuit_settings` too. Tested
  offline only; not yet verified in the game.
- `monsters.diff_thres` (lower = stricter), `monsters.scales` if your sprites are not 1:1,
  `monsters.min_color_spread` filters grey false positives (clouds), and
  `monsters.per_monster` overrides any of those per folder.
- `health.*` sets potion thresholds; `health.ui_y_start` is where the bottom UI begins,
  or give `health.fixed_rects` for the HP/MP bars if auto detection misses them.
- `minimap.player_color_bgr` and `minimap.fixed_rect` if your client draws the minimap
  differently; `minimap.width_bound_pct` keeps the sweep inside the middle of the map.
- `stuck.*` and `search.*` for recovery and no-route sweeping.

## Development (WSL / Linux, no game needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -e . pytest opencv-python-headless numpy pyyaml
.venv/bin/pytest -q
.venv/bin/python -m maplebot run --source tests/fixtures/frame_synth.png --no-input --no-viz --frames 30
```

All vision, navigation and decision code is pure functions over numpy arrays and is
covered by synthetic-image tests; `mss`, `pynput` and `win32gui` are only imported on
Windows at runtime.

## Layout

```
maplebot/
  capture/   frame sources: live window (mss + win32gui), PNG, video
  vision/    minimap, player, monsters, status bars, template matching
  nav/       route maps, global localisation, stuck detection
  brain/     combat targeting, sweep search, decision FSM
  control/   keyboard controller (pynput backend, fake backend for tests), hotkeys
  tools/     route recorder, mob sprite fetcher
  bot.py     per-frame pipeline and main loop;  overlay.py debug drawing
config/      default.yaml (documented) + user.example.yaml
tests/       pytest suite with synthetic frames and a fixture map
```

Use at your own risk; botting violates most servers' terms of service.
