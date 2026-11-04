# maplebot — design

## Intent

A personal MapleStory auto-leveling bot that works purely from screen pixels
and simulated keyboard input. It is a fresh, smaller, test-driven rewrite
informed by two references in `~/Dev`:

- `MapleStoryAutoLevelUp` (Artale, Ken Yu, MIT): minimap-based global
  localization, colour-coded route PNGs, party-red-bar player tracking,
  template-matched monsters, FSM states, health monitor thread.
- `maplestory-auto-ds` (079/083 private servers): appearance/title-template
  player tracking, per-monster thresholds and scales, colour-purity guard,
  no-route left/right sweep search, stuck recovery, fixed-rect HP/MP bars.

The user gave no further requirements ("dig into the repos and start"), so
every choice below is the builder's call and is adjustable through config.

## Goals

1. Hunt on a map: follow a recorded route, detect monsters near the player,
   face them and attack, drink HP/MP potions, recover when stuck.
2. Hunt without a route: sweep left/right inside minimap bounds, attack
   whatever comes in range.
3. Everything in `vision/`, `nav/`, `brain/` is a pure function or a small
   state object that runs headless on a numpy frame, so the whole decision
   pipeline is unit-tested on synthetic images in WSL without the game.
4. Reuse the reference asset formats unchanged: `minimaps/<map>/map.png` +
   `route*.png` with the same RGB colour codes, `monster/<name>/*.png` with
   a green (0,255,0) or alpha background. Existing assets copy straight in.
5. Runs on Windows with the game in windowed mode, any resolution (positions
   are detected, not hard-coded; UI rectangles that must be fixed live in
   config).

## Non-goals (v1)

- Rune solving, auto channel change, auto login, email alerts, GUI.
- macOS/Linux capture. Capture and input are Windows-only; the rest of the
  code is platform-neutral.
- Game-memory access of any kind.

## Anti-goals

- A frame loop that blocks on key presses or on slow detection.
- Hard-coded pixel coordinates for a single resolution buried in code.
- Python double loops over pixels (route search, bar percent) — numpy only.

## Architecture

```
capture.FrameSource ──frame(BGR)──▶ Bot.step()
                                      │
      vision.minimap ─▶ minimap box, player dot, other-player dots
      vision.player  ─▶ player (x,y) on screen   [party_bar | nametag | appearance]
      vision.monsters─▶ [Detection]              [templates ± hp bars]
      vision.bars    ─▶ hp%, mp%, exp%
      nav.route      ─▶ global (x,y), nearest route Command
      brain.fsm      ─▶ Command(move_x, move_y, action)
      health         ─▶ potion requests
                                      │
                            control.KeyboardController (thread)
                            overlay.draw_debug (cv2.imshow)
```

- One main loop at `system.fps_main` (default 12). Capture runs in its own
  thread (mss + win32gui) and hands over the latest frame. The keyboard
  controller runs in its own thread at 30 Hz and applies the *difference*
  between the current `Command` and held keys (hold left/right/up/down, tap
  jump/attack/potions with cooldowns), so the vision loop never sleeps on
  input.
- `Command` is a frozen dataclass `(move_x, move_y, action)` with
  `move_x ∈ {none,left,right,stop}`, `move_y ∈ {none,up,down,stop}`,
  `action ∈ {none,jump,attack,aoe,teleport,goal}`. Route colour codes decode
  to the same type.
- Config: `config/default.yaml` deep-merged with a user YAML
  (`--cfg path`). Dicts all the way down; `maplebot.config.load_config`.
- Player localisation strategies (config `player.locator`):
  `party_bar` (HSV red bar + offset), `nametag` (grayscale template split
  into vertical strips, masked), `appearance` (colour template of the
  character, flipped variant, ellipse mask, CCORR). Each returns
  `PlayerFix(x, y, score)` or `None`; the bot keeps the last fix on a miss.
- Minimap: auto-detect the white 1-px bordered rectangle ≥100×100, or use
  `minimap.fixed_rect`. Player dot: colour tolerance tiers, compact
  connected components (2–50 px), prefer nearest to last fix, reject jumps
  > `max_jump_px`.
- Monsters: per-monster `diff_thres`, `scales`, `min_color_spread`; modes
  `color` and `grayscale` (SQDIFF_NORMED, masked); NMS IoU 0.4; optional
  HP-bar colour blobs as extra detections. Search box = attack range +
  margin around the player; player's own box is ignored.
- Navigation: locate minimap crop on `map.png` by masked SQDIFF with local
  search around last position; `global = minimap_loc + player_dot`.
  Nearest route pixel within `search_range` via numpy distance over a
  colour-index image (route PNG pre-converted to an int8 code map once).
  Ladder complement logic from the references (horizontal and up/down codes
  complement each other only while on a ladder). `goal` advances to the
  next route; `miss_grace_frames` keeps the last move across small gaps.
- Combat: left/right attack boxes (directional) or centred box (aoe).
  Target = nearest overlapping detection whose centre is on the correct
  side; tie within 50 px → no attack (avoid flip-flop). Attack cooldown per
  mode. When ≥ `aoe_monster_count` monsters are visible and an AoE key is
  set, use `aoe`.
- FSM states: `Hunting` (route or sweep + combat), `Recovering` (stuck:
  walk the opposite way of the last movement for N s, jump), with
  `StuckDetector` on global/minimap position. `Searching` is a sub-behaviour
  of Hunting when no route is loaded (`SweepSearcher`: flip on timer, on
  stall, on max offset, on minimap width bounds).
- Health: bars found as white-bordered contours in the UI strip
  (aspect 4–10, area window) sorted by x → HP, MP, EXP; or `fixed_rects`.
  Fill % = non-grey pixels on the bar's centre line. Policy: drink when
  below threshold, re-arm when back above `rearm`, retry after timeout,
  warm-up reads ignored. Buffs: keys with cooldowns, cast when idle.
- Hotkeys (pynput): F1 pause/resume, F2 screenshot, F12 quit.
- Tools: `maplebot record-route` (stitch minimap into map.png, draw route
  from pressed keys, F3 saves route, F4 saves map), `maplebot fetch-mob
  <name>` (maplestory.io sprites → `monster/<name>/`).

## Constraints

- Python ≥ 3.11 (runtime: `C:\Python313`), OpenCV ≥ 4.8, numpy ≥ 1.26,
  PyYAML, mss, pynput, pywin32 (Windows only), pytest. No PySide/GUI.
- Tests run in WSL with `opencv-python-headless`; nothing in `tests/`
  imports `mss`, `pynput`, or `win32gui`.
- Project root `~/Dev/maplebot`, package `maplebot/`, CLI `python -m
  maplebot`.

## Left to the builder

Exact synthetic fixtures, overlay layout, log format, CLI flag names beyond
`run --cfg --source --no-input --no-viz`, and the Windows setup script.
