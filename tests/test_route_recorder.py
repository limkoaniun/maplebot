import cv2
import numpy as np

from maplebot.config import load_config, parse_color_codes
from maplebot.tools.route_recorder import MapStitcher, RouteCanvas, command_from_keys
from maplebot.types import Command


def _world(seed=1, w=400, h=300):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    img[img.sum(axis=2) < 60] = 1  # avoid pure black (treated as "unknown")
    return img


def test_stitcher_grows_canvas_and_places_second_crop():
    world = _world()
    st = MapStitcher(padding=10)
    a = world[50:150, 60:180]
    pos_a = st.add(a, player_dot=(10, 10))
    assert pos_a == (10, 10)
    assert st.map.shape[:2] == (120, 140)
    b = world[80:180, 120:240]   # shifted +60 x, +30 y from a
    pos_b = st.add(b, player_dot=(5, 5))
    assert pos_b == (70, 40)
    assert st.map.shape[:2] == (160, 210)   # grew right/bottom with padding
    # the stitched map now contains b's pixels at the right place
    assert np.array_equal(st.map[40:140, 70:190], b) or np.array_equal(st.map[41:140, 71:190], b[1:, 1:])


def test_stitcher_blanks_player_dot():
    world = _world(2)
    st = MapStitcher(padding=5, dot_radius=2)
    a = world[0:100, 0:100].copy()
    a[48:53, 48:53] = (136, 255, 255)
    st.add(a, player_dot=(50, 50))
    patch = st.map[5 + 48:5 + 53, 5 + 48:5 + 53]
    assert not np.any(np.all(patch == (136, 255, 255), axis=2))


def test_route_canvas_marks_line_and_blob_colours():
    cfg = load_config()["route"]
    codes = parse_color_codes(cfg["color_code"])
    codes.update(parse_color_codes(cfg["color_code_up_down"]))
    canvas = RouteCanvas(np.zeros((50, 50, 3), np.uint8), codes)
    canvas.mark((10, 10), Command("left"), blob=False)
    canvas.mark((20, 10), Command("left"), blob=False)
    assert tuple(canvas.img[10, 15]) == (0, 0, 255)         # red in BGR along the line
    canvas.mark((30, 30), Command("none", "none", "jump"), blob=True)
    assert tuple(canvas.img[30, 30]) == (255, 0, 255)       # magenta blob
    canvas.mark((40, 40), Command("none", "none", "goal"), blob=True)
    assert tuple(canvas.img[40, 40]) == (0, 255, 255)       # yellow goal


def test_route_canvas_save_and_reset(tmp_path):
    canvas = RouteCanvas(np.zeros((20, 20, 3), np.uint8), {(255, 0, 0): "left none none"})
    canvas.mark((5, 5), Command("left"), blob=True)
    p = tmp_path / "route1.png"
    canvas.save(str(p))
    assert cv2.imread(str(p)) is not None
    canvas.reset(np.zeros((20, 20, 3), np.uint8))
    assert not canvas.img.any()


def test_command_from_keys():
    assert command_from_keys({"left"}, jump_key="space", teleport_key="e") == Command("left")
    assert command_from_keys({"left", "space"}, "space", "e") == Command("left", "none", "jump")
    assert command_from_keys({"down", "space"}, "space", "e") == Command("none", "down", "jump")
    assert command_from_keys({"up", "e"}, "space", "e") == Command("none", "up", "teleport")
    assert command_from_keys({"up"}, "space", "e") == Command("none", "up")
    assert command_from_keys(set(), "space", "e") is None
