import os

import cv2
import numpy as np
import pytest

from maplebot.bot import Bot, BotSnapshot, build_bot
from maplebot.capture.window import WindowCapture
from maplebot.config import deep_merge, load_config, parse_color_codes
from maplebot.control.backend import NullBackend, RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.nav.route import RouteMap
from maplebot.overlay import draw_debug
from maplebot.types import Command
from maplebot.vision.monsters import MonsterLibrary
from maplebot.vision.player import to_cv_hsv
from maplebot.vision.template import make_mask
from tests.synth import make_frame, make_sprite, paste_minimap, paste_sprite

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "minimaps", "nf2")
YELLOW = (136, 255, 255)


def _red():
    hsv = np.uint8([[to_cv_hsv((0, 80, 80))]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _cfg(**over):
    return deep_merge(load_config(), over)


def _library(tmp_path):
    d = tmp_path / "monster" / "mob"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "mob.png"), make_sprite())
    return MonsterLibrary.load(str(tmp_path / "monster"), [], {}, [1.0])


class _Src:
    title = "synthetic"

    def __init__(self, frame):
        self.frame = frame

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self):
        return None if self.frame is None else self.frame.copy()


def _full_frame(sprite_x=741):
    f = make_frame()
    paste_minimap(f, 10, 60, 150, 120, dot=(40, 30))
    f[280:286, 621:651] = _red()                      # party bar -> player (641, 346)
    bgr, mask = make_mask(make_sprite())
    paste_sprite(f, bgr, mask, sprite_x, 331)         # monster centred at y=346
    return f


def _bot(frame, cfg, library, route_map=None):
    be = RecordingBackend()
    kb = KeyboardController(cfg, be, sleep=lambda s: None)
    return Bot(cfg, _Src(frame), kb, library, route_map), be


def test_step_without_minimap_does_not_raise(tmp_path):
    bot, be = _bot(make_frame(), _cfg(), MonsterLibrary())
    snap = bot.step(now=0.0)
    assert isinstance(snap, BotSnapshot)
    assert snap.minimap_box is None and snap.player is None
    assert snap.command == Command.NONE
    assert bot.keyboard.cmd == Command.NONE


def test_step_full_synthetic_frame_attacks(tmp_path):
    bot, be = _bot(_full_frame(), _cfg(), _library(tmp_path))
    snap = bot.step(now=0.0)
    assert snap.player == (641, 346)
    assert snap.minimap_pos == (40, 30)
    assert len(snap.detections) == 1
    assert snap.command.action == "attack" and snap.command.move_x == "right"
    assert bot.keyboard.cmd == snap.command


def test_unfocused_game_suspends_navigation_and_commands(tmp_path):
    bot, _ = _bot(_full_frame(), _cfg(), _library(tmp_path))
    suspended = []
    bot.brain.suspend = suspended.append
    bot.keyboard.is_active = lambda: False
    snap = bot.step(now=42.0)
    assert snap.command == Command.NONE
    assert bot.keyboard.cmd == Command.NONE
    assert suspended == [42.0]


def test_step_keeps_last_player_fix_on_miss(tmp_path):
    f = _full_frame()
    bot, be = _bot(f, _cfg(), _library(tmp_path))
    bot.step(now=0.0)
    bot.source.frame = make_frame()
    snap = bot.step(now=0.1)
    assert snap.player == (641, 346)


def test_step_global_position_with_route():
    cfg = _cfg(route={"map": "nf2"})
    codes = parse_color_codes(cfg["route"]["color_code"]), parse_color_codes(cfg["route"]["color_code_up_down"])
    rm = RouteMap.load(FIX, *codes)
    f = make_frame()
    paste_minimap(f, 10, 60, 122, 102)
    f[61:161, 11:131] = rm.map_bgr[60:160, 40:160]    # minimap shows the map crop at (40, 60)
    f[90:93, 40:43] = YELLOW                          # dot at minimap (30, 30)
    bot, be = _bot(f, cfg, MonsterLibrary(), rm)
    snap = bot.step(now=0.0)
    assert snap.minimap_pos == (30, 30)
    assert snap.global_pos == (70, 90)
    assert snap.route_idx == 0


def test_step_reads_bars_and_requests_potion(tmp_path):
    cfg = _cfg(health={"warmup_reads": 0, "hp_percent": 50, "ui_y_start": 610})
    f = _full_frame(sprite_x=1100)
    bar = np.full((22, 165, 3), 255, np.uint8)
    bar[1:21, 1:50] = (60, 60, 220)
    bar[1:21, 50:164] = (90, 90, 90)
    for x in (100, 400, 700):
        f[640:662, x:x + 165] = bar
    bot, be = _bot(f, cfg, MonsterLibrary())
    snap = bot.step(now=0.0)
    assert snap.hp is not None and snap.hp < 50
    bot.keyboard.apply_once(0.0)
    assert ("down", "1") in be.events


def test_ladder_flag_from_vertical_motion(tmp_path):
    f = _full_frame(sprite_x=1100)
    bot, be = _bot(f, _cfg(), MonsterLibrary())
    bot.step(now=0.0)
    assert bot.on_ladder is False
    f2 = f.copy()
    f2[280:286, 621:651] = (40, 40, 40)
    f2[270:276, 621:651] = _red()      # moved straight up 10px
    bot.source.frame = f2
    bot.step(now=0.1)
    assert bot.on_ladder is True
    f3 = f2.copy()
    f3[270:276, 621:651] = (40, 40, 40)
    f3[270:276, 641:671] = _red()      # moved 20px sideways
    bot.source.frame = f3
    bot.step(now=0.2)
    assert bot.on_ladder is False


def test_pause_toggle_releases_and_blocks_commands(tmp_path):
    bot, be = _bot(_full_frame(), _cfg(), _library(tmp_path))
    bot.step(now=0.0)
    bot.toggle_pause()
    assert bot.paused is True
    snap = bot.step(now=0.1)
    assert bot.keyboard.cmd == Command.NONE and bot.keyboard.enabled is False
    bot.toggle_pause()
    bot.step(now=0.2)
    assert bot.keyboard.enabled is True and bot.keyboard.cmd.action == "attack"


def test_overlay_draws_without_mutating(tmp_path):
    cfg = _cfg()
    bot, be = _bot(_full_frame(), cfg, _library(tmp_path))
    snap = bot.step(now=0.0)
    before = snap.frame.copy()
    img = draw_debug(snap.frame, snap, cfg)
    assert img.shape == snap.frame.shape
    assert np.array_equal(before, snap.frame)
    assert not np.array_equal(img, snap.frame)


def test_build_bot_with_image_source_no_input(tmp_path):
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), _full_frame())
    _library(tmp_path)
    cfg = _cfg(monsters={"dir": str(tmp_path / "monster")})
    bot = build_bot(cfg, source_path=str(p), no_input=True)
    assert isinstance(bot.keyboard.backend, NullBackend)
    assert bot.library.names == ["mob"]
    snap = bot.step(now=0.0)
    assert snap.command.action == "attack"
    bot.close()


def test_build_bot_missing_monster_dir_gives_empty_library(tmp_path):
    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), make_frame())
    cfg = _cfg(monsters={"dir": str(tmp_path / "nope")})
    bot = build_bot(cfg, source_path=str(p), no_input=True)
    assert bot.library.templates == []
    bot.close()


def test_cli_run_frames(tmp_path, capsys):
    from maplebot.__main__ import main

    p = tmp_path / "frame.png"
    cv2.imwrite(str(p), _full_frame())
    _library(tmp_path)
    user = tmp_path / "u.yaml"
    user.write_text(f"monsters:\n  dir: {tmp_path / 'monster'}\n")
    rc = main(["run", "--cfg", str(user), "--source", str(p), "--no-input", "--no-viz", "--frames", "3"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "attack" in out


def test_cli_parser_has_tool_subcommands():
    from maplebot.__main__ import build_parser

    p = build_parser()
    a = p.parse_args(["record-route", "my_map", "--map", "x.png"])
    assert a.name == "my_map" and a.map == "x.png"
    b = p.parse_args(["fetch-mob", "Orange Mushroom", "--version", "65"])
    assert b.name == "Orange Mushroom" and b.version == "65"


# --- review fix pass -------------------------------------------------------

def test_lost_minimap_stops_navigation_after_grace():
    cfg = _cfg(route={"map": "nf2", "miss_grace_frames": 2})
    codes = parse_color_codes(cfg["route"]["color_code"]), parse_color_codes(cfg["route"]["color_code_up_down"])
    rm = RouteMap.load(FIX, *codes)
    f = make_frame()
    paste_minimap(f, 10, 60, 122, 102)
    f[61:161, 11:131] = rm.map_bgr[60:160, 40:160]
    f[90:93, 40:43] = YELLOW
    bot, be = _bot(f, cfg, MonsterLibrary(), rm)
    bot.step(now=0.0)
    loc = bot._minimap_loc
    assert loc == (40, 60)
    bot.source.frame = make_frame()           # minimap gone (loading screen / occluded)
    snaps = [bot.step(now=0.1 * i) for i in range(1, 5)]
    assert all(s.global_pos is None for s in snaps)
    assert bot._minimap_loc == loc
    assert snaps[-1].command == Command.NONE


def test_none_frame_sends_none_command(tmp_path):
    bot, be = _bot(_full_frame(), _cfg(), _library(tmp_path))
    bot.step(now=0.0)
    assert bot.keyboard.cmd.action == "attack"
    bot.source.frame = None
    assert bot.step(now=0.1) is None
    assert bot.keyboard.cmd == Command.NONE


def test_auto_detected_bars_require_three():
    from maplebot.vision.bars import read_bars

    ui = make_frame(1282, 83)
    bar = np.full((22, 165, 3), 255, np.uint8)
    bar[1:21, 1:50] = (60, 60, 220)
    bar[1:21, 50:164] = (90, 90, 90)
    for x in (100, 400):
        ui[30:52, x:x + 165] = bar
    assert read_bars(ui, {"fixed_rects": None, "bar_white_thresh": 240}) == (None, None, None)


class _LiveSrc(_Src, WindowCapture):
    """Exercise live-source gating without opening a window or sending real input."""
    get_sample = None  # This synthetic source deliberately has no asynchronous capture buffer.


def _live_bot(frame, cfg, library=None):
    be = RecordingBackend()
    kb = KeyboardController(cfg, be, sleep=lambda s: None)
    return Bot(cfg, _LiveSrc(frame), kb, library or MonsterLibrary(), None), be


def test_live_input_waits_for_initial_visual_lock():
    bot, be = _live_bot(make_frame(), _cfg(search={"trigger_frames": 0}))
    for i in range(6):
        snap = bot.step(now=i * 0.1)
        bot.keyboard.apply_once(i * 0.1)
        assert snap.command == Command.NONE
    assert not bot.keyboard.enabled and not be.events
    bot.source.frame = _full_frame()
    snap = bot.step(now=1.0)
    bot.keyboard.apply_once(1.0)
    assert bot.keyboard.enabled and snap.command.move_x == "left"
    assert be.held == {"left"}


@pytest.mark.parametrize("missing", ["player", "minimap"])
def test_live_lost_lock_blocks_movement_and_potions_until_reacquired(tmp_path, monkeypatch, missing):
    cfg = _cfg(safety={"max_missing_frames": 3}, health={"warmup_reads": 0})
    bot, be = _live_bot(_full_frame(), cfg, _library(tmp_path))
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (100, 100, None))
    bot.step(now=0.0)
    bot.keyboard.apply_once(0.0)
    assert "right" in be.held
    bad = _full_frame()
    if missing == "player":
        bad[280:286, 621:651] = (40, 40, 40)
    else:
        bad[60:182, 10:162] = (40, 40, 40)
    bot.source.frame = bad
    bot.step(now=0.1)
    bot.step(now=0.2)
    assert bot.keyboard.enabled  # short detection misses retain the current command
    bot.keyboard.request("hp_potion")
    snap = bot.step(now=0.3)
    assert snap.command == Command.NONE and not bot.keyboard.enabled
    assert not be.held
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (10, 10, None))
    n = len(be.events)
    for i in range(4, 8):
        bot.step(now=i * 0.1)
        bot.keyboard.apply_once(i * 0.1)
    assert len(be.events) == n
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (100, 100, None))
    bot.source.frame = _full_frame()
    snap = bot.step(now=1.0)
    bot.keyboard.apply_once(1.0)
    assert bot.keyboard.enabled and snap.command.action == "attack"
    assert ("down", "1") not in be.events and ("down", "2") not in be.events


def test_live_reacquiring_lock_preserves_manual_pause():
    bot, be = _live_bot(make_frame(), _cfg(search={"trigger_frames": 0}))
    bot.toggle_pause()
    bot.toggle_pause()
    assert not bot.paused and not bot.keyboard.enabled
    bot.toggle_pause()
    bot.source.frame = _full_frame()
    snap = bot.step(now=1.0)
    bot.keyboard.apply_once(1.0)
    assert bot.paused and not bot.keyboard.enabled and snap.command == Command.NONE
    assert not be.events
    bot.toggle_pause()
    assert not bot.keyboard.enabled  # resuming waits for the next observed frame
    bot.step(now=1.1)
    assert bot.keyboard.enabled


def test_live_no_frame_blocks_immediately_and_waits_for_current_lock():
    bot, be = _live_bot(_full_frame(), _cfg(search={"trigger_frames": 0}))
    bot.step(now=0.0)
    bot.keyboard.apply_once(0.0)
    assert be.held
    bot.keyboard.request("hp_potion")
    bot.source.frame = None
    assert bot.step(now=0.1) is None
    assert not bot.keyboard.enabled and not be.held
    bot.source.frame = make_frame()
    assert bot.step(now=0.2).command == Command.NONE
    assert not bot.keyboard.enabled


@pytest.mark.parametrize("blocked_by", ["pause", "focus"])
def test_unsent_potions_do_not_consume_health_retries(monkeypatch, blocked_by):
    cfg = _cfg(health={"warmup_reads": 0, "retry_s": 3, "max_retries": 2, "backoff_s": 30})
    bot, be = _bot(_full_frame(), cfg, MonsterLibrary())
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (10, 100, None))
    active = {"on": blocked_by != "focus"}
    bot.keyboard.is_active = lambda: active["on"]
    if blocked_by == "pause":
        bot.toggle_pause()
    for now in (0.0, 3.0, 6.0):
        bot.step(now)
        bot.keyboard.apply_once(now)
    assert ("down", "1") not in be.events
    if blocked_by == "pause":
        bot.toggle_pause()
    active["on"] = True
    bot.step(7.0)
    bot.keyboard.apply_once(7.0)
    assert ("down", "1") in be.events  # no 30-second backoff for potions never sent


@pytest.mark.parametrize("mode, skill_action", [("directional", "attack"), ("aoe", "aoe")])
def test_low_mp_uses_normal_attack_and_restores_skill(tmp_path, monkeypatch, mode, skill_action):
    cfg = _cfg(keys={"normal_attack": "alt"}, attack={"mp_fallback_percent": 5, "mode": mode})
    bot, _ = _bot(_full_frame(), cfg, _library(tmp_path))
    reading = {"mp": 0}
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (100, reading["mp"], None))
    assert bot.step(0.0).command.action == "normal_attack"
    reading["mp"] = 20
    assert bot.step(0.1).command.action == skill_action
    reading["mp"] = None
    assert bot.step(0.2).command.action == skill_action
    reading["mp"] = 0
    bot.toggle_pause()
    assert bot.step(0.3).command == Command.NONE
    bot.toggle_pause()
    cfg["keys"]["normal_attack"] = ""
    assert bot.step(0.4).command.action == skill_action
    cfg["keys"]["normal_attack"] = "alt"
    del cfg["attack"]["mp_fallback_percent"]
    assert bot.step(0.5).command.action == skill_action


def _hybrid_setup(tmp_path, max_frames=12, radius=70):
    rng = np.random.default_rng(23)
    nametag = rng.integers(20, 240, (14, 40, 3), dtype=np.uint8)
    head = rng.integers(20, 240, (20, 24, 4), dtype=np.uint8)
    head[:, :, 3] = 255
    head[[0, -1], :, 3] = 0
    head[:, [0, -1], 3] = 0
    tag_path, head_path = tmp_path / "nametag.png", tmp_path / "head.png"
    cv2.imwrite(str(tag_path), nametag)
    cv2.imwrite(str(head_path), head)
    cfg = _cfg(player={"locator": "hybrid",
                       "nametag": {"path": str(tag_path), "offset": [0, -30],
                                   "diff_thres": 0.001, "split_width": 0},
                       "appearance": {"path": str(head_path), "feet_offset": [0, 12],
                                      "match_thres": 0.99, "search_radius": radius},
                       "hybrid": {"max_fallback_frames": max_frames}},
               safety={"max_missing_frames": 1})
    head_bgr, head_mask = make_mask(head)

    def frame(primary=True, x=300, head_x=None):
        image = make_frame()
        paste_minimap(image, 10, 60, 150, 120, dot=(40, 30))
        if primary:
            image[316:330, x - 20:x + 20] = nametag
        paste_sprite(image, head_bgr, head_mask, (x if head_x is None else head_x) - 12, 268)
        return image

    bot, _ = _live_bot(frame(primary=False), cfg)
    return bot, frame


def test_hybrid_requires_primary_lock_then_uses_independent_local_appearance(tmp_path):
    bot, frame = _hybrid_setup(tmp_path)
    bot.step(0.0)
    assert bot.player is None and bot._input_blocked
    bot.source.frame = frame()
    bot.step(0.1)
    assert bot.player.source == "nametag" and bot.player.pos == (300, 300)
    bot.source.frame = frame(primary=False, x=320)
    bot.step(0.2)
    assert bot.player.source == "appearance_fallback" and bot.player.pos == (320, 300)
    assert not bot._input_blocked


def test_hybrid_never_uses_global_appearance_or_exceeds_radius_cap(tmp_path):
    bot, frame = _hybrid_setup(tmp_path, radius=1000)
    bot.source.frame = frame()
    bot.step(0.0)
    bot.source.frame = frame(primary=False, x=450)
    snap = bot.step(0.1)
    assert bot._input_blocked and snap.command == Command.NONE
    assert bot.player.pos == (300, 300)  # distant perfect appearance match is not an identity lock
    bot.source.frame = frame(primary=False, x=320)
    bot.step(0.2)
    assert bot.player.source == "appearance_fallback" and bot.player.pos == (320, 300)


@pytest.mark.parametrize("configured,allowed", [(2, 2), (999, 12)])
def test_hybrid_fallback_expires_and_primary_reacquisition_resets_it(tmp_path, configured, allowed):
    bot, frame = _hybrid_setup(tmp_path, max_frames=configured)
    bot.source.frame = frame()
    bot.step(0.0)
    bot.source.frame = frame(primary=False, x=320)
    for i in range(allowed):
        bot.step((i + 1) * 0.1)
        assert bot.player.source == "appearance_fallback" and not bot._input_blocked
    snap = bot.step(2.0)
    assert bot._input_blocked and snap.command == Command.NONE
    assert bot.step(2.1).command == Command.NONE  # a perfect head match cannot renew the allowance
    bot.source.frame = frame(x=390, head_x=320)
    bot.step(2.2)
    assert bot.player.source == "nametag" and bot.player.pos == (390, 300)
    assert not bot._input_blocked  # the primary wins even with an old appearance match nearby
    bot.source.frame = frame(primary=False, x=410)
    bot.step(2.3)
    assert bot.player.source == "appearance_fallback" and bot.player.pos == (410, 300)
    bot.source.frame = None
    bot.step(2.4)
    bot.source.frame = frame(primary=False, x=520)
    assert bot.step(2.5).command == Command.NONE and bot._input_blocked


def test_hybrid_primary_stays_local_until_capture_loss_requires_new_identity(tmp_path):
    bot, frame = _hybrid_setup(tmp_path)
    bot.source.frame = frame()
    bot.step(0)
    bot.source.frame = frame(x=550)
    assert bot.step(.1).command == Command.NONE
    assert bot.player.pos == (300, 300) and bot._input_blocked
    bot.source.frame = None
    bot.step(.2)
    bot.source.frame = frame(x=550)
    bot.step(.3)
    assert bot.player.pos == (550, 300) and not bot._input_blocked


def _vertical_frame(dot=(40, 30), player=True):
    frame = _full_frame()
    paste_minimap(frame, 10, 60, 150, 120, dot=dot)
    if not player:
        frame[280:286, 621:651] = (40, 40, 40)
    return frame


def _started_vertical_bot(monkeypatch, **safety):
    cfg = _cfg(keys={"pickup": ""}, safety={"max_missing_frames": 1, **safety},
               route={"circuit": {"enabled": True, "start_idx": 0,
                       "waypoints": [{"x": 40, "y": 15, "action": "climb", "dwell_s": .1},
                                     {"x": 80, "y": 15}]}})
    bot, backend = _live_bot(_vertical_frame(), cfg)
    bot.step(0)
    assert bot.step(.3).command == Command("none", "up", "jump")
    bot._hybrid_primary_pos = bot.player.pos
    monkeypatch.setattr(bot.brain, "decide", lambda p: pytest.fail("ordinary brain used without a current character fix"))
    bot.source.frame = _vertical_frame((40, 28), player=False)
    return bot, backend


def test_minimap_vertical_continuation_never_attacks_or_moves_after_waypoint_advance(monkeypatch):
    bot, backend = _started_vertical_bot(monkeypatch)
    assert bot.step(.4).command == Command("none", "up")
    assert bot._vertical_minimap_active and bot._input_blocked and bot.keyboard.enabled
    bot.keyboard.apply_once(.4)
    assert backend.held == {"up"}
    bot.source.frame = _vertical_frame((40, 15), player=False)
    assert bot.step(.5).command == Command("none", "up")
    assert bot.step(.7).command == Command.NONE
    assert bot.brain.circuit.idx == 1 and not bot._vertical_minimap_active
    assert not bot.keyboard.enabled and not backend.held
    assert bot.step(.8).command == Command.NONE  # new walk/combat leg requires a fresh fix


@pytest.mark.parametrize("reason", ["minimap", "alignment", "identity", "focus", "pause", "capture"])
def test_minimap_vertical_continuation_rejects_lost_observation_and_stays_blocked(monkeypatch, reason):
    bot, backend = _started_vertical_bot(monkeypatch)
    bot.step(.4)
    bot.keyboard.apply_once(.4)
    if reason == "minimap":
        bot.source.frame[60:182, 10:162] = (40, 40, 40)
    elif reason == "alignment":
        bot.source.frame = _vertical_frame((44, 28), player=False)
    elif reason == "identity":
        bot._hybrid_primary_pos = None
    elif reason == "focus":
        monkeypatch.setattr(bot.keyboard, "is_active", lambda: False)
    elif reason == "pause":
        bot.toggle_pause()
    else:
        bot.source.frame = None
    snap = bot.step(.5)
    assert snap is None or snap.command == Command.NONE
    assert not bot._vertical_minimap_active and not bot.keyboard.enabled and not backend.held
    bot.source.frame = _vertical_frame((40, 27), player=False)
    bot.paused = False
    monkeypatch.setattr(bot.keyboard, "is_active", lambda: True)
    assert bot.step(.6).command == Command.NONE


def test_minimap_vertical_continuation_caps_configured_grace_at_two_seconds(monkeypatch):
    bot, _ = _started_vertical_bot(monkeypatch, vertical_minimap_grace_s=999)
    assert bot.step(.4).command.move_y == "up"
    assert bot.step(1.3).command.move_y == "up"
    assert bot.step(2.3).command.move_y == "up"
    assert bot.step(2.5).command == Command.NONE
    assert not bot._vertical_minimap_active and not bot.keyboard.enabled
