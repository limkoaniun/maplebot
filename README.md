# maplebot

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

- `attack.range_x/range_y` and `attack.deadzone_px` control when the bot attacks and when
  it stops walking toward a target. `attack.mode: aoe` uses a centred box instead.
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
