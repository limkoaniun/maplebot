import cv2
import numpy as np

from maplebot.types import Box
from maplebot.vision.player import locate_by_party_bar, locate_by_template, to_cv_hsv
from maplebot.vision.template import make_mask
from tests.synth import make_frame, paste_sprite

PARTY_CFG = {
    "lower_hsv": [0, 60, 60],
    "upper_hsv": [0, 100, 100],
    "offset": [20, 66],
    "min_h": 5,
    "max_h": 7,
    "max_w": 50,
}


def _red_bgr():
    hsv = np.uint8([[to_cv_hsv((0, 80, 80))]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def test_to_cv_hsv_scales():
    assert tuple(to_cv_hsv((360, 100, 100))) == (179, 255, 255)


def test_party_bar_found_with_offset():
    f = make_frame()
    f[300:306, 500:530] = _red_bgr()
    fix = locate_by_party_bar(f, PARTY_CFG)
    assert fix is not None
    assert (fix.x, fix.y) == (520, 366)
    assert fix.source == "party_bar"


def test_party_bar_ignores_excluded_region():
    f = make_frame()
    f[300:306, 500:530] = _red_bgr()
    assert locate_by_party_bar(f, PARTY_CFG, exclude=Box(490, 290, 60, 30)) is None


def test_party_bar_none_on_plain_frame():
    assert locate_by_party_bar(make_frame(), PARTY_CFG) is None


def test_party_bar_picks_largest():
    f = make_frame()
    f[300:306, 500:510] = _red_bgr()
    f[400:406, 700:740] = _red_bgr()
    fix = locate_by_party_bar(f, PARTY_CFG)
    assert (fix.x, fix.y) == (720, 466)


def _tag(seed=5):
    rng = np.random.default_rng(seed)
    tag = np.empty((14, 60, 3), np.uint8)
    tag[:] = (0, 255, 0)
    tag[2:12, 2:58] = rng.integers(0, 256, (10, 56, 3), dtype=np.uint8)
    tag[2:12, 2:58, 1] = np.minimum(tag[2:12, 2:58, 1], 200)
    return tag


def test_template_locates_nametag_with_split():
    f = make_frame()
    tag = _tag()
    bgr, mask = make_mask(tag)
    paste_sprite(f, bgr, mask, 600, 350)
    fix = locate_by_template(f, bgr, mask, last=None, offset=(0, -30), diff_thres=0.2, split_width=30)
    assert fix is not None
    # anchor is the template's bottom-centre: (600+30, 350+14) + offset
    assert abs(fix.x - 630) <= 2 and abs(fix.y - 334) <= 2


def test_template_rejects_high_diff():
    f = make_frame()
    bgr, mask = make_mask(_tag())
    assert locate_by_template(f, bgr, mask, last=None, offset=(0, 0), diff_thres=0.2) is None


def test_template_flip_matches_mirrored_sprite():
    f = make_frame()
    bgr, mask = make_mask(_tag())
    paste_sprite(f, cv2.flip(bgr, 1), cv2.flip(mask, 1), 200, 100)
    assert locate_by_template(f, bgr, mask, last=None, offset=(0, 0), diff_thres=0.2) is None
    fix = locate_by_template(f, bgr, mask, last=None, offset=(0, 0), diff_thres=0.2, flip=True)
    assert fix is not None and abs(fix.x - 230) <= 2


def test_template_uses_last_for_local_search():
    f = make_frame()
    bgr, mask = make_mask(_tag())
    paste_sprite(f, bgr, mask, 600, 350)
    fix = locate_by_template(f, bgr, mask, last=(625, 330), offset=(0, -30), diff_thres=0.2)
    assert fix is not None and abs(fix.x - 630) <= 2
