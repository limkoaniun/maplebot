import os

import cv2
import numpy as np
import pytest

from maplebot.config import PACKAGE_ROOT, load_config
from maplebot.vision.gains import (changed_pixel_count, classify_gain_line, crop_rect, exp_text_changed,
                                   exp_text_plausible, gain_line_arrivals, gain_line_pixel_count,
                                   gain_line_scores, line_similarity, load_gain_templates)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gains")
PITCH, SLOT_H = 14, 16


def _img(name):
    img = cv2.imread(os.path.join(FIX, name + ".png"))
    assert img is not None, name
    return img


def _slots(name, n=3):
    """Fixture stacks are [1150, 576, 214, 44]: slot k is 16 rows ending 14*k rows above the bottom."""
    img = _img(name)
    return [img[img.shape[0] - SLOT_H - PITCH * k:img.shape[0] - PITCH * k] for k in range(n)]


@pytest.fixture(scope="module")
def templates():
    paths = load_config()["evidence"]["templates"]
    loaded = load_gain_templates({k: os.path.join(PACKAGE_ROOT, v) for k, v in paths.items()})
    assert set(loaded) == {"exp", "meso"}
    return loaded


def test_exp_text_same_value_is_pixel_identical_across_captures():
    a, b = _img("exp-text-4975-a"), _img("exp-text-4975-b")
    assert changed_pixel_count(a, b) == 0
    assert exp_text_changed(a, b) is False


def test_exp_text_changed_value_is_detected():
    a, b, c = _img("exp-text-4975-a"), _img("exp-text-4990"), _img("exp-text-next")
    assert changed_pixel_count(a, b) >= 27  # smallest recorded change was 27 pixels
    assert exp_text_changed(a, b) is True and exp_text_changed(b, c) is True


def test_exp_text_not_comparable_is_none_not_false():
    a = _img("exp-text-4975-a")
    assert exp_text_changed(a, a[:, :50]) is None
    assert exp_text_changed(None, a) is None


def test_exp_text_plausibility_rejects_scenery_and_blank():
    assert all(exp_text_plausible(_img(n)) for n in ("exp-text-4975-a", "exp-text-4990", "exp-text-next"))
    assert not exp_text_plausible(np.zeros((14, 107, 3), np.uint8))
    assert not exp_text_plausible(np.full((14, 107, 3), 255, np.uint8))
    assert not exp_text_plausible(_img("lines-exp-bright-wall")[:14, :107])


def test_crop_rect_requires_the_whole_rect_inside_the_frame():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert crop_rect(frame, (10, 10, 50, 20)).shape == (20, 50, 3)
    for rect in ((180, 10, 50, 20), (10, 90, 50, 20), (-1, 0, 5, 5), (0, 0, 0, 5), None, "bad", (1, 2, 3)):
        assert crop_rect(frame, rect) is None
    assert crop_rect(None, (0, 0, 5, 5)) is None
    assert crop_rect(np.zeros((100, 200), np.uint8), (0, 0, 5, 5)) is None


def test_gain_line_pixel_count_separates_empty_from_fresh_and_decays():
    assert gain_line_pixel_count(_slots("lines-empty")[0]) <= 8
    assert gain_line_pixel_count(_slots("lines-exp")[0]) >= 100
    assert gain_line_pixel_count(_slots("lines-meso-over-exp")[0]) >= 150
    assert gain_line_pixel_count(_slots("lines-item-over-meso")[0]) >= 80
    decay = [gain_line_pixel_count(_slots(f"lines-decay-{i}")[0]) for i in range(5)]
    assert decay == sorted(decay, reverse=True) and decay[0] >= 40 and decay[-1] < 15
    assert gain_line_pixel_count(np.zeros((0, 0, 3), np.uint8)) == 0


def test_classify_exp_meso_and_item_lines(templates):
    assert classify_gain_line(_slots("lines-exp")[0], templates) == "exp"
    assert classify_gain_line(_slots("lines-meso-over-exp")[0], templates) == "meso"
    assert classify_gain_line(_slots("lines-item-over-meso")[0], templates) == "other"
    assert classify_gain_line(_slots("lines-meso-exp-item")[0], templates) == "meso"
    assert classify_gain_line(_slots("lines-exp-bright-wall")[0], templates) == "exp"
    assert classify_gain_line(_slots("lines-decay-0")[0], templates) == "exp"  # a different capture than the template


def test_classification_scores_keep_a_margin(templates):
    exp = gain_line_scores(_slots("lines-decay-0")[0], templates)
    meso = gain_line_scores(_slots("lines-meso-exp-item")[0], templates)
    item = gain_line_scores(_slots("lines-item-over-meso")[0], templates)
    assert exp["exp"] > 0.7 and exp["meso"] < 0.4
    assert meso["meso"] > 0.7 and meso["exp"] < 0.45
    assert max(item.values()) < 0.4


def test_classify_degrades_to_other(templates):
    assert classify_gain_line(_slots("lines-empty")[0], templates) == "other"
    assert classify_gain_line(_slots("lines-exp")[0], {}) == "other"
    assert classify_gain_line(np.zeros((4, 10, 3), np.uint8), templates) == "other"  # smaller than a template
    assert classify_gain_line(np.zeros((16, 214, 3), np.uint8), templates) == "other"  # flat image
    assert load_gain_templates({"exp": os.path.join(FIX, "missing.png"), "meso": ""}) == {}


def test_one_line_is_one_arrival_while_it_fades():
    frames = [_slots("lines-empty")] + [_slots(f"lines-decay-{i}") for i in range(5)] + [_slots("lines-empty")]
    arrivals = [gain_line_arrivals(a, b) for a, b in zip(frames, frames[1:])]
    assert arrivals == [1, 0, 0, 0, 0, 0]


def test_identical_frames_never_arrive():
    for name in ("lines-exp", "lines-meso-over-exp", "lines-meso-exp-item", "lines-repeat-after", "lines-empty"):
        assert gain_line_arrivals(_slots(name), _slots(name)) == 0


def test_new_line_pushing_the_old_one_up_is_one_arrival(templates):
    before, after = _slots("lines-shift-before"), _slots("lines-shift-after")
    assert gain_line_pixel_count(before[0]) >= 40  # the rising-edge rule alone would miss this
    assert line_similarity(before[0], after[1]) > 0.7 > line_similarity(before[0], before[1]) + 0.15
    assert gain_line_arrivals(before, after) == 1
    assert classify_gain_line(before[0], templates) == "meso" and classify_gain_line(after[0], templates) == "exp"


def test_same_text_repeated_is_found_through_the_second_slot(templates):
    before, after = _slots("lines-repeat-before"), _slots("lines-repeat-after")
    assert classify_gain_line(before[0], templates) == classify_gain_line(after[0], templates) == "exp"
    assert gain_line_arrivals(before, after) == 1


def test_two_lines_within_one_interval_are_two_arrivals():
    assert gain_line_arrivals(_slots("lines-double-before"), _slots("lines-double-after")) == 2
    # With only two slots watched the second arrival cannot be proven, so nothing is counted.
    assert gain_line_arrivals(_slots("lines-double-before", 2), _slots("lines-double-after", 2)) == 0
    # Both slots filling from empty are two arrivals as well; a third, faded line above is not.
    assert gain_line_arrivals(_slots("lines-empty"), _slots("lines-meso-exp-item")) == 2


def test_unexplained_changes_are_not_counted():
    # A different line replaced the old one, but the old one is not found above it.
    assert gain_line_arrivals(_slots("lines-exp"), _slots("lines-item-over-meso")) == 0
    assert gain_line_arrivals(_slots("lines-exp"), _slots("lines-empty")) == 0
    assert gain_line_arrivals(_slots("lines-exp"), None) == 0
    assert gain_line_arrivals(_slots("lines-exp"), _slots("lines-exp", 2)) == 0
    assert line_similarity(_slots("lines-empty")[0], _slots("lines-exp")[0]) < 0.3
    assert line_similarity(np.zeros((16, 214, 3), np.uint8), _slots("lines-exp")[0]) == 0.0


def test_line_shape_gate_rejects_scenery_like_masks():
    from maplebot.vision.gains import gain_line_like, gain_line_mask
    assert all(gain_line_like(gain_line_mask(_slots(n)[0])) for n in (
        "lines-exp", "lines-meso-over-exp", "lines-item-over-meso", "lines-exp-bright-wall", "lines-decay-0"))
    left = np.zeros((16, 214), bool)
    left[5:11, 10:60] = True          # plenty of pixels, but not right-aligned
    edges = np.zeros((16, 214), bool)
    edges[0:2, 150:210] = True        # touches the slot's top rows
    assert not gain_line_like(left) and not gain_line_like(edges) and not gain_line_like(np.zeros((16, 214), bool))
    # Outlined-white scenery that is not right-aligned never becomes an arrival.
    scene = np.zeros((16, 214, 3), np.uint8)
    scene[4:12:2, 10:120] = 255
    empty = np.zeros((16, 214, 3), np.uint8)
    assert gain_line_pixel_count(scene) >= 40
    assert gain_line_arrivals([empty] * 3, [scene, empty, empty]) == 0


# -- a fading line over changing scenery ---------------------------------------
def _faded(line, background, alpha):
    """The line's own text and outline pixels alpha-blended over a flat grey background.

    A model of the fade (the notification blends toward the scenery; lines-decay-0..4 show
    it on a real background), built from the real "You received EXP (+15)" crop.
    """
    from maplebot.vision.gains import gain_line_mask
    grey = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    text = gain_line_mask(line)
    outline = (grey < 70) & (cv2.dilate(text.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    out = np.full(line.shape, float(background), np.float32)
    out[text] = alpha * 255 + (1 - alpha) * background
    out[outline] = (1 - alpha) * background
    return out.astype(np.uint8)


def _flat(background):
    return np.full((SLOT_H, 214, 3), background, np.uint8)


@pytest.mark.parametrize("alpha", [0.74, 0.70, 0.64])
def test_fading_line_over_darker_then_brighter_scenery_is_not_a_new_arrival(alpha):
    line = _slots("lines-exp")[0]
    dark = [_faded(line, 30, alpha), _flat(30), _flat(30)]
    mid = [_faded(line, 110, alpha), _flat(110), _flat(110)]
    # Same line, same slot: the count says "empty" over dark scenery and "fresh" over mid-tone scenery.
    assert gain_line_pixel_count(dark[0]) < 15 and gain_line_pixel_count(mid[0]) >= 100
    assert line_similarity(mid[0], dark[0]) > 0.8
    assert gain_line_arrivals(dark, mid) == 0 and gain_line_arrivals(mid, dark) == 0
    # A second, older copy of the same text above it changes nothing.
    dark[1], mid[1] = _faded(line, 30, alpha - 0.2), _faded(line, 110, alpha - 0.2)
    assert gain_line_arrivals(dark, mid) == 0


def test_genuine_arrivals_into_an_empty_slot_stay_far_below_the_similarity_guard():
    empty = _slots("lines-empty")
    for name in ("lines-exp", "lines-exp-bright-wall", "lines-meso-over-exp", "lines-item-over-meso",
                 "lines-decay-0", "lines-repeat-before"):
        assert line_similarity(_slots(name)[0], empty[0]) < 0.2
        assert gain_line_arrivals(empty, _slots(name)) == 1


def test_same_text_below_a_faded_line_counts_only_when_the_old_line_moved_up():
    # Real crops rearranged: the faded "EXP (+15)" line (8 pixels, "empty" by count) is the old
    # bottom line; a fresh line with the same text then appears in the bottom slot.
    old, fresh, empty = _slots("lines-decay-4")[0], _slots("lines-exp")[0], _slots("lines-empty")
    assert gain_line_pixel_count(old) < 15 and line_similarity(fresh, old) >= 0.45
    before = [old, empty[1], empty[2]]
    assert gain_line_arrivals(before, [fresh, old, empty[2]]) == 1       # old line now one slot up
    assert gain_line_arrivals(before, [fresh, empty[1], empty[2]]) == 0  # nothing shows it moved
    # A different text under a faded line was never affected.
    assert gain_line_arrivals(before, _slots("lines-meso-over-exp")) == 1


def test_upper_slot_counts_only_when_it_was_empty_and_shows_something_new():
    empty = _slots("lines-empty")
    meso, exp, _ = _slots("lines-meso-exp-item")
    item = _slots("lines-item-over-meso")[0]
    assert gain_line_arrivals(empty, [meso, exp, empty[2]]) == 2
    # The slot above already held a (different) line: only the bottom arrival is proven.
    assert gain_line_pixel_count(item) >= 15 and line_similarity(exp, item) < 0.45
    assert gain_line_arrivals([empty[0], item, empty[2]], [meso, exp, empty[2]]) == 1
    # The slot above held the same text, faded below the count: not a second arrival either.
    old = _slots("lines-decay-4")[0]
    new = _slots("lines-decay-0")[0]
    assert gain_line_pixel_count(old) < 15 and gain_line_pixel_count(new) >= 40
    assert gain_line_arrivals([empty[0], old, empty[2]], [meso, new, empty[2]]) == 1


def test_three_fresh_lines_from_empty_prove_two_arrivals_at_most():
    empty = _slots("lines-empty")
    full = [_slots("lines-exp")[0], _slots("lines-meso-over-exp")[0], _slots("lines-item-over-meso")[0]]
    assert all(gain_line_pixel_count(slot) >= 40 for slot in full)
    assert gain_line_arrivals(empty, full) == 2  # the top watched slot cannot be told from an older line


def test_shifted_line_needs_fresh_lines_in_every_slot_below_it():
    before, after = _slots("lines-shift-before"), _slots("lines-shift-after")
    empty = _slots("lines-empty")
    # The old bottom line reappears two slots up, but the slot between holds no fresh line.
    jumped = [after[0], empty[1], before[0]]
    assert line_similarity(before[0], jumped[2]) > 0.9
    assert gain_line_arrivals(before, jumped) == 0
    assert gain_line_arrivals(before, [after[0], after[0], before[0]]) == 2


def test_coloured_bright_text_is_not_a_gain_line():
    from maplebot.vision.gains import gain_line_like, gain_line_mask
    line = _slots("lines-exp")[0]
    coloured = line.copy()
    coloured[gain_line_mask(line)] = (120, 255, 255)  # bright yellow text, same dark outline
    loose = gain_line_mask(coloured, max_spread=256)
    assert int(loose.sum()) >= 100 and gain_line_like(loose)  # only the saturation test rejects it
    assert gain_line_pixel_count(coloured) == 0
    empty = _slots("lines-empty")
    assert gain_line_arrivals(empty, [coloured, empty[1], empty[2]]) == 0
    assert gain_line_arrivals(empty, [line, empty[1], empty[2]]) == 1
