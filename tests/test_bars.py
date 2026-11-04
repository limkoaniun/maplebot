import numpy as np

from maplebot.types import Box
from maplebot.vision.bars import bar_fill_percent, find_status_bars, read_bars
from tests.synth import make_frame


def _bar(w=100, h=10, fill_frac=0.5, fill_color=(60, 60, 220), empty_color=(90, 90, 90)):
    """A status bar: 1px white border, left part filled with colour, rest grey."""
    img = np.full((h, w, 3), 255, np.uint8)
    inner_w = w - 2
    n_fill = int(round(inner_w * fill_frac))
    img[1:h - 1, 1:1 + n_fill] = fill_color
    img[1:h - 1, 1 + n_fill:w - 1] = empty_color
    return img


def test_bar_fill_percent_half():
    assert abs(bar_fill_percent(_bar(fill_frac=0.5)) - 50) <= 2


def test_bar_fill_percent_full_and_empty():
    assert bar_fill_percent(_bar(fill_frac=1.0)) >= 98
    assert bar_fill_percent(_bar(fill_frac=0.0)) <= 2


def test_find_status_bars_three_sorted():
    ui = make_frame(1282, 83)
    for i, x in enumerate((700, 400, 100)):
        ui[30:52, x:x + 165] = _bar(165, 22, fill_frac=0.3 * (i + 1))
    boxes = find_status_bars(ui)
    assert [b.x for b in boxes] == [100, 400, 700]
    assert all(b.h == 22 and b.w == 165 for b in boxes)


def test_find_status_bars_rejects_wrong_aspect():
    ui = make_frame(1282, 83)
    ui[10:70, 100:160] = 255  # a 60x60 white square
    assert find_status_bars(ui) == []


def test_read_bars_detected():
    ui = make_frame(1282, 83)
    for x, frac in ((100, 0.5), (400, 0.25), (700, 0.75)):
        ui[30:52, x:x + 165] = _bar(165, 22, fill_frac=frac)
    hp, mp, exp = read_bars(ui, {"fixed_rects": None, "bar_white_thresh": 240})
    assert abs(hp - 50) <= 2 and abs(mp - 25) <= 2 and abs(exp - 75) <= 2


def test_read_bars_two_bars_without_exp():
    ui = make_frame(1282, 83)
    for x, frac in ((100, 0.5), (400, 0.25)):
        ui[30:52, x:x + 165] = _bar(165, 22, fill_frac=frac)
    hp, mp, exp = read_bars(ui, {"fixed_rects": None, "bar_white_thresh": 240})
    assert abs(hp - 50) <= 2 and abs(mp - 25) <= 2 and exp is None


def test_read_bars_none_when_count_wrong():
    ui = make_frame(1282, 83)
    ui[30:52, 100:265] = _bar(165, 22)
    assert read_bars(ui, {"fixed_rects": None, "bar_white_thresh": 240}) == (None, None, None)


def test_read_bars_fixed_rects():
    ui = make_frame(1282, 83)
    ui[30:40, 100:200] = _bar(100, 10, fill_frac=0.5)
    ui[30:40, 300:400] = _bar(100, 10, fill_frac=0.1)
    hp, mp, exp = read_bars(ui, {"fixed_rects": [[100, 30, 100, 10], [300, 30, 100, 10]]})
    assert abs(hp - 50) <= 2 and abs(mp - 10) <= 2 and exp is None
