import cv2
import numpy as np
import pytest

from maplebot.types import Box, Command, Detection
from maplebot.vision.template import make_mask, match_template, nms


def test_command_parse_roundtrip():
    assert str(Command.parse("left none jump")) == "left none jump"


def test_command_rejects_bad_token():
    with pytest.raises(ValueError):
        Command.parse("sideways none none")


def test_command_none_constant():
    assert Command.NONE == Command()
    assert str(Command.NONE) == "none none none"


def test_box_iou_half_overlap():
    assert Box(0, 0, 10, 10).iou(Box(5, 0, 10, 10)) == pytest.approx(1 / 3)
    assert Box(0, 0, 10, 10).center == (5, 5)
    assert Box(0, 0, 10, 10).intersection_area(Box(20, 20, 5, 5)) == 0


def test_make_mask_from_green_key():
    img = np.full((4, 4, 3), 200, np.uint8)
    img[1, 2] = (0, 255, 0)
    bgr, mask = make_mask(img)
    assert bgr.shape == (4, 4, 3)
    assert mask[1, 2] == 0 and mask.sum() == 255 * 15


def test_make_mask_from_alpha():
    img = np.full((4, 4, 4), 200, np.uint8)
    img[0, 0, 3] = 0
    bgr, mask = make_mask(img)
    assert bgr.shape == (4, 4, 3)
    assert mask[0, 0] == 0 and mask[3, 3] == 255


def _noise_with_pattern(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    pat = rng.integers(0, 256, (10, 10, 3), dtype=np.uint8)
    img[40:50, 30:40] = pat
    return img, pat


def test_match_finds_pattern_exact():
    img, pat = _noise_with_pattern()
    res = match_template(img, pat)
    assert res.loc == (30, 40)
    assert res.score < 0.01
    assert res.local is False


def test_match_local_search_hits():
    img, pat = _noise_with_pattern()
    res = match_template(img, pat, last_loc=(28, 38), local_radius=10)
    assert res.loc == (30, 40)
    assert res.local is True


def test_match_pads_small_image():
    img = np.zeros((5, 5, 3), np.uint8)
    pat = np.zeros((10, 10, 3), np.uint8)
    res = match_template(img, pat)
    assert res.loc == (0, 0)


def test_match_ccorr_higher_is_better():
    img, pat = _noise_with_pattern(1)
    res = match_template(img, pat, method=cv2.TM_CCORR_NORMED)
    assert res.loc == (30, 40)
    assert res.score > 0.99


def test_nms_suppresses_overlap():
    a = Detection("m", Box(0, 0, 10, 10), 0.1)
    b = Detection("m", Box(1, 0, 10, 10), 0.3)
    c = Detection("m", Box(50, 50, 10, 10), 0.2)
    kept = nms([b, a, c], iou_thres=0.4)
    assert kept == [a, c]


def test_nms_higher_is_better():
    a = Detection("m", Box(0, 0, 10, 10), 0.9)
    b = Detection("m", Box(1, 0, 10, 10), 0.5)
    assert nms([b, a], iou_thres=0.4, higher_is_better=True) == [a]
