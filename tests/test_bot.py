import os

import cv2
import numpy as np
import pytest

from maplebot.bot import Bot, BotSnapshot, build_bot
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
        return self.frame.copy()


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
