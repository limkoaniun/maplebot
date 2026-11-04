import os

import cv2
import numpy as np
import pytest

from maplebot.config import load_config, parse_color_codes
from maplebot.nav.route import RouteFollower, RouteMap, locate_minimap_on_map, nearest_command
from maplebot.types import Command

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "minimaps", "nf2")


@pytest.fixture(scope="module")
def codes():
    cfg = load_config()["route"]
    return parse_color_codes(cfg["color_code"]), parse_color_codes(cfg["color_code_up_down"])


@pytest.fixture(scope="module")
def nf2(codes):
    return RouteMap.load(FIX, *codes)


def _rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def _synthetic_route(tmp_path, codes, pixels, map_pixels=()):
    """Write a 50x50 map/route pair. pixels: list of ((x, y), rgb) drawn on route1.png."""
    d = tmp_path / "synth"
    d.mkdir()
    m = np.zeros((50, 50, 3), np.uint8)
    for (x, y), rgb in map_pixels:
        m[y, x] = _rgb_to_bgr(rgb)
    r = m.copy()
    for (x, y), rgb in pixels:
        r[y, x] = _rgb_to_bgr(rgb)
    cv2.imwrite(str(d / "map.png"), m)
    cv2.imwrite(str(d / "route1.png"), r)
    return RouteMap.load(str(d), *codes)


def test_route_load_counts_and_codes(nf2):
    assert len(nf2.routes) == 2  # route_rest.png excluded
    assert nf2.map_bgr.shape[:2] == nf2.routes[0].shape
    assert nf2.map_bgr.shape[2] == 3
    assert "left none none" in nf2.codes
    assert (nf2.routes[0] >= 0).any()


def test_route_load_blanks_map_coloured_pixels(tmp_path, codes):
    rm = _synthetic_route(tmp_path, codes, pixels=[((5, 5), (255, 0, 0)), ((6, 6), (255, 0, 0))],
                          map_pixels=[((5, 5), (255, 0, 0))])
    assert rm.routes[0][5, 5] == -1
    assert rm.codes[rm.routes[0][6, 6]] == "left none none"


def test_nearest_command_picks_closest(tmp_path, codes):
    rm = _synthetic_route(tmp_path, codes, pixels=[((10, 10), (255, 0, 0)), ((20, 10), (0, 0, 255))])
    hit, hit_ud = nearest_command(rm, 0, (12, 10), radius=10)
    assert hit.command == Command("left")
    assert hit.pixel == (10, 10) and hit.distance == 2
    assert hit_ud is None


def test_nearest_command_separates_up_down(tmp_path, codes):
    rm = _synthetic_route(tmp_path, codes, pixels=[((10, 10), (255, 0, 0)), ((11, 10), (127, 127, 127))])
    hit, hit_ud = nearest_command(rm, 0, (12, 10), radius=5)
    assert hit.command == Command("left") and hit.distance == 2
    assert hit_ud.command == Command("none", "up") and hit_ud.distance == 1


def test_nearest_command_none_outside_radius(tmp_path, codes):
    rm = _synthetic_route(tmp_path, codes, pixels=[((10, 10), (255, 0, 0))])
    assert nearest_command(rm, 0, (40, 40), radius=10) == (None, None)


def test_nearest_command_clamps_at_edges(tmp_path, codes):
    rm = _synthetic_route(tmp_path, codes, pixels=[((0, 0), (255, 0, 0))])
    hit, _ = nearest_command(rm, 0, (1, 1), radius=10)
    assert hit.pixel == (0, 0)


def test_locate_minimap_on_map_exact_crop(nf2):
    crop = nf2.map_bgr[60:160, 40:160]
    res = locate_minimap_on_map(nf2.map_bgr, crop, last=None)
    assert res.loc == (40, 60)
    res2 = locate_minimap_on_map(nf2.map_bgr, crop, last=(38, 58))
    assert res2.loc == (40, 60) and res2.local


def _follower(tmp_path, codes, pixels, **over):
    rm = _synthetic_route(tmp_path, codes, pixels)
    cfg = {"search_range": 5, "miss_grace_frames": 3, "ladder_complement_dist": 3}
    cfg.update(over)
    return RouteFollower(rm, cfg)


def test_follower_plain_move(tmp_path, codes):
    f = _follower(tmp_path, codes, [((10, 10), (0, 0, 255))])
    assert f.step((11, 10), on_ladder=False) == Command("right")


def test_follower_ladder_complement(tmp_path, codes):
    # grey (up) at distance 1, red (left) at distance 2; on a ladder both apply
    f = _follower(tmp_path, codes, [((11, 10), (127, 127, 127)), ((10, 10), (255, 0, 0))])
    assert f.step((12, 10), on_ladder=True) == Command("left", "up")
    assert f.step((12, 10), on_ladder=False) == Command("none", "up")


def test_follower_jump_near_ladder_holds_up(tmp_path, codes):
    # orange (left jump) nearest, grey (up) adjacent: jump + up to grab the rope
    f = _follower(tmp_path, codes, [((10, 10), (255, 127, 0)), ((11, 10), (127, 127, 127))])
    assert f.step((10, 10), on_ladder=False) == Command("left", "up", "jump")


def test_follower_goal_advances_route(tmp_path, codes):
    d = tmp_path / "two"
    d.mkdir()
    m = np.zeros((50, 50, 3), np.uint8)
    r1 = m.copy(); r1[10, 10] = _rgb_to_bgr((255, 255, 0))
    r2 = m.copy(); r2[10, 10] = _rgb_to_bgr((255, 0, 0))
    for name, img in (("map.png", m), ("route1.png", r1), ("route2.png", r2)):
        cv2.imwrite(str(d / name), img)
    f = RouteFollower(RouteMap.load(str(d), *codes), {"search_range": 5, "miss_grace_frames": 3, "ladder_complement_dist": 3})
    assert f.idx == 0
    cmd = f.step((10, 10), on_ladder=False)
    assert cmd.action == "goal" and f.idx == 1
    assert f.step((10, 10), on_ladder=False) == Command("left")
    f.step((10, 10), on_ladder=False)
    assert f.idx == 1  # no goal on route 2, stays
    

def test_follower_miss_grace_then_stop(tmp_path, codes):
    f = _follower(tmp_path, codes, [((10, 10), (255, 0, 0))], miss_grace_frames=2)
    assert f.step((10, 10), on_ladder=False) == Command("left")
    assert f.step((40, 40), on_ladder=False) == Command("left")
    assert f.step((40, 40), on_ladder=False) == Command("left")
    assert f.step((40, 40), on_ladder=False) == Command.NONE


def test_follower_none_position(tmp_path, codes):
    f = _follower(tmp_path, codes, [((10, 10), (255, 0, 0))])
    assert f.step(None, on_ladder=False) == Command.NONE
