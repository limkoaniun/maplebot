import numpy as np

from maplebot.types import Box
from maplebot.vision.minimap import find_minimap, find_other_players, find_player_dot
from tests.synth import make_frame, paste_minimap

YELLOW = (136, 255, 255)


def test_find_minimap_returns_inner_box():
    f = make_frame()
    paste_minimap(f, 10, 60, 150, 120)
    box = find_minimap(f)
    assert box == Box(11, 61, 148, 118)


def test_find_minimap_returns_none_without_border():
    assert find_minimap(make_frame()) is None


def test_find_minimap_ignores_small_white_boxes():
    f = make_frame()
    paste_minimap(f, 10, 60, 50, 50)
    assert find_minimap(f) is None


def _minimap(dots, w=150, h=120):
    f = make_frame()
    paste_minimap(f, 10, 60, w, h, extra_dots=dots)
    return f[61:61 + h - 2, 11:11 + w - 2]


def test_find_player_dot_center():
    mm = _minimap([((40, 30), YELLOW)])
    assert find_player_dot(mm, YELLOW) == (40, 30)


def test_find_player_dot_prefers_nearest_to_last():
    mm = _minimap([((40, 30), YELLOW), ((100, 80), YELLOW)])
    assert find_player_dot(mm, YELLOW, last=(95, 78)) == (100, 80)


def test_find_player_dot_rejects_big_jump():
    mm = _minimap([((100, 100), YELLOW)])
    assert find_player_dot(mm, YELLOW, last=(10, 10), max_jump_px=30) is None


def test_find_player_dot_tolerance_tiers():
    mm = _minimap([((40, 30), (124, 243, 243))])
    assert find_player_dot(mm, YELLOW, tolerance=8) == (40, 30)


def test_find_player_dot_none_on_empty():
    assert find_player_dot(_minimap([]), YELLOW) is None


def test_find_other_players():
    mm = _minimap([((20, 20), (0, 0, 255)), ((90, 70), (0, 0, 255))])
    pts = find_other_players(mm, (0, 0, 255))
    assert sorted(pts) == [(20, 20), (90, 70)]
