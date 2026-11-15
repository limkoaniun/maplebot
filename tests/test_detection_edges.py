"""Full sprite matching must retain the configured search and exclusion bounds."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from maplebot.bot import Bot
from maplebot.brain.combat import search_box
from maplebot.types import Box, PlayerFix
from maplebot.vision.monsters import MonsterLibrary, MonsterTemplate, detect_monsters


FIXTURES = Path(__file__).parent / "fixtures" / "detection_edges"


def detector(library, player=(150, 150), *, range_x=50, range_y=100):
    bot = Bot.__new__(Bot)  # exercise detection without any capture/input backend
    bot.player = PlayerFix(*player)
    bot.library = library
    bot.cfg = {
        "attack": {"range_x": range_x, "range_y": range_y, "search_margin": 0},
        "player": {"box_w": 40, "box_h": 60},
        "monsters": {"mode": "color", "diff_thres": .001,
                     "max_matches_per_template": 5, "hp_bar": {"enable": False}},
    }
    return bot


@pytest.fixture
def sprite_library():
    image = np.random.default_rng(14).integers(15, 240, (21, 31, 3), dtype=np.uint8)
    mask = np.full(image.shape[:2], 255, np.uint8)
    return MonsterLibrary([MonsterTemplate("mob", image, mask, 1)], ["mob"])


@pytest.mark.parametrize("center", [(101, 150), (199, 150), (150, 101), (150, 199)])
def test_complete_sprite_straddling_search_boundary_is_detected(sprite_library, center):
    bot = detector(sprite_library)
    frame = np.full((300, 300, 3), 35, np.uint8)
    image = sprite_library.templates[0].img_bgr
    x, y = center[0] - 15, center[1] - 10
    frame[y:y+21, x:x+31] = image
    before = frame.copy()
    detections, reported = bot._detect(frame)
    assert reported == Box(100, 100, 100, 100)
    assert len(detections) == 1 and detections[0].box.center == center
    np.testing.assert_array_equal(frame, before)


@pytest.mark.parametrize("center", [(99, 150), (200, 150), (150, 99), (150, 200)])
def test_padding_does_not_expand_logical_search_boundary(sprite_library, center):
    bot = detector(sprite_library)
    frame = np.full((300, 300, 3), 35, np.uint8)
    x, y = center[0] - 15, center[1] - 10
    frame[y:y+21, x:x+31] = sprite_library.templates[0].img_bgr
    detections, reported = bot._detect(frame)
    assert reported == Box(100, 100, 100, 100)
    assert detections == []


def test_padding_keeps_player_exclusion(sprite_library):
    bot = detector(sprite_library)
    frame = np.full((300, 300, 3), 35, np.uint8)
    frame[140:161, 135:166] = sprite_library.templates[0].img_bgr
    assert bot._detect(frame)[0] == []


def test_padding_clamps_at_frame_boundary(sprite_library):
    bot = detector(sprite_library, player=(30, 30))
    frame = np.full((100, 100, 3), 35, np.uint8)
    frame[65:86, 5:36] = sprite_library.templates[0].img_bgr
    detections, reported = bot._detect(frame)
    assert reported == Box(0, 0, 80, 80)
    assert len(detections) == 1 and detections[0].box.center == (20, 75)


@pytest.mark.parametrize("filename,player,names,expected", [
    ("green_mushroom_near_edge.png", (158, 78), ["green_mushroom"],
     {"green_mushroom": (64, 64)}),
    ("ground_monsters_near_edges.png", (207, 113), ["stump", "blue_snail"],
     {"stump": (114, 98), "blue_snail": (308, 108)}),
])
def test_recorded_monsters_require_complete_pixels_without_looser_thresholds(
        filename, player, names, expected):
    frame = cv2.imread(str(FIXTURES / filename))
    library = MonsterLibrary.load(str(FIXTURES / "templates"), names, {}, [.975])
    bot = detector(library, player, range_x=70, range_y=48)
    bot.cfg["attack"]["search_margin"] = 45
    bot.cfg["monsters"].update(
        mode="grayscale", diff_thres=.12, color_diff_thres=.18, min_color_spread=25,
        per_monster={"stump": {"diff_thres": .24, "color_diff_thres": .26},
                     "blue_snail": {"diff_thres": .17, "color_diff_thres": .19}})
    h, w = frame.shape[:2]
    logical = search_box(player, bot.cfg["attack"], w, h)
    exclusion = Box(player[0]-20, player[1]-30, 40, 60)
    old = detect_monsters(frame[logical.y:logical.y2, logical.x:logical.x2],
                          library, bot.cfg["monsters"], exclusion, (logical.x, logical.y))
    detections, reported = bot._detect(frame)
    assert reported == logical
    assert {d.name for d in detections} == set(expected)
    assert len(old) < len(detections)
    for detection in detections:
        target = expected[detection.name]
        assert abs(detection.box.center[0] - target[0]) <= 1
        assert abs(detection.box.center[1] - target[1]) <= 1
