from copy import deepcopy
import hashlib

import cv2
import numpy as np

from maplebot.bot import Bot
from maplebot.config import load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.editor_bridge import capture_signature
from maplebot.map_profiles import MapProfile, MapProfileRecognizer, apply_profile, map_settings_from_config
from maplebot.types import Command
from maplebot.vision.monsters import MonsterLibrary


class Source:
    def __init__(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame.copy()


def profile(name, rect, seed):
    cfg = load_config()
    cfg['game_window'].update(title='MapleStory', working_size=[160, 120], title_bar_px=0)
    cfg['minimap']['fixed_rect'] = rect
    cfg['monsters']['names'] = ['blue_snail']
    plan = {'version': 1, 'name': name, 'image_size': rect[2:], 'loop': False,
            'capture': capture_signature(cfg),
            'platforms': [{'id': 'ground', 'left': 3, 'right': rect[2]-4, 'y': rect[3]-6}],
            'connections': [], 'route': [{'platform_id': 'ground', 'x': 4},
                                        {'platform_id': 'ground', 'x': rect[2]-5}]}
    image = np.random.default_rng(seed).integers(0, 255, (rect[3], rect[2], 3), dtype=np.uint8)
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return MapProfile(name, name, plan, image, map_settings_from_config(cfg), plan['capture'], digest), cfg


def scene(p):
    frame = np.zeros((120, 160, 3), np.uint8)
    x, y, w, h = p.capture['minimap_rect']
    frame[y:y+h, x:x+w] = p.reference_bgr
    return frame


def bot_for(p, base, profiles):
    cfg = apply_profile(base, p)
    backend = RecordingBackend()
    kb = KeyboardController(cfg, backend, clock=lambda: 0., sleep=lambda _: None)
    bot = Bot(cfg, Source(scene(p)), kb, MonsterLibrary(), None)
    bot.map_recognizer = MapProfileRecognizer(profiles)
    bot.map_ready = False
    bot.paused = True
    return bot, backend


def test_known_map_requires_three_distinct_observations_and_matching_settings():
    first, cfg = profile('first', [5, 5, 50, 40], 10)
    bot, _ = bot_for(first, cfg, [first])
    assert not bot._check_map(scene(first), 0)
    assert not bot._check_map(scene(first), .1)
    assert bot._check_map(scene(first), .2)
    assert bot.paused  # recognition does not automatically resume farming
    bot.cfg['minimap']['fixed_rect'] = [6, 5, 50, 40]
    assert not bot._check_map(scene(first), .3)
    assert bot.map_match.state == 'settings_mismatch'


def test_portal_to_map_with_different_crop_releases_input_before_profile_switch():
    first, cfg = profile('first', [5, 5, 50, 40], 10)
    second, _ = profile('second', [20, 10, 100, 80], 11)
    bot, backend = bot_for(first, cfg, [first, second])
    for t in (0, .1, .2):
        bot._check_map(scene(first), t)
    bot.paused = False
    bot.keyboard.set_command(Command('right'))
    bot.keyboard.set_enabled(True)
    bot.keyboard.apply_once(0)
    assert 'right' in backend.held
    assert not bot._check_map(scene(second), .3)
    assert bot.map_reacquiring and not backend.held and not bot.keyboard.enabled
    assert bot.pending_map_profile is None
    bot._check_map(scene(second), .4)
    bot._check_map(scene(second), .5)
    assert bot.paused and bot.full_pause  # a confirmed different profile is a hard stop
    assert bot.pending_map_profile is second
    assert bot.cfg['minimap']['fixed_rect'] == first.capture['minimap_rect']


def test_unknown_map_clears_character_rope_and_minimap_state():
    first, cfg = profile('first', [5, 5, 50, 40], 10)
    bot, _ = bot_for(first, cfg, [first])
    bot.map_ready = True
    bot.paused = False
    bot.minimap_pos = (15, 20)
    bot._hybrid_primary_pos = (70, 40)
    bot.on_ladder = True
    bot._vertical_minimap_active = True
    snap = bot.step(1)
    # Initial confirmation already disallows input; then a truly unknown frame must erase identity.
    bot.source.frame[:] = 0
    snap = bot.step(1.1)
    assert snap.state == 'map_unknown' and snap.command == Command.NONE
    assert not bot.map_ready and not bot.keyboard.enabled
    assert bot.minimap_pos is None and bot._hybrid_primary_pos is None
    assert not bot.on_ladder and not bot._vertical_minimap_active


def test_temporary_movement_test_keeps_map_guard_without_replacing_test_route():
    first, cfg = profile('first', [5, 5, 50, 40], 10)
    bot, _ = bot_for(first, cfg, [first])
    test_route = deepcopy(bot.cfg['route']['circuit'])
    test_route.update(movement_test=True, waypoints=[{'x': 6, 'y': 34, 'action': 'walk'},
                                                  {'x': 8, 'y': 34, 'action': 'walk'}])
    bot.cfg['route']['circuit'] = test_route
    for t in (0, .1, .2):
        ready = bot._check_map(scene(first), t)
    assert ready
    assert bot.cfg['route']['circuit'] is test_route
