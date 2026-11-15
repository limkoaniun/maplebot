"""Actual minimap-only crops: no game chat, character names, or full screenshots."""
from copy import deepcopy
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from maplebot.config import load_config
from maplebot.editor_bridge import capture_signature
from maplebot.map_profiles import (MapProfile, MapProfileRecognizer, map_settings_from_config,
                                   profile_key, stable_map_similarity)


FIXTURES = Path(__file__).parent / "fixtures/minimaps"
RECTS = {"henesys": [56, 81, 111, 98], "construction": [20, 82, 210, 128]}
MARKERS = {"player_color_bgr": [43, 223, 255], "other_color_bgr": [0, 0, 255],
           "player_tolerance": 40}


def recorded(name):
    image = cv2.imread(str(FIXTURES / f"profile_{name}.png"))
    assert image is not None
    return image


def make_profile(name):
    image = recorded(name + "_reference")
    cfg = load_config()
    cfg["game_window"].update(title="MapleStory", working_size=[1366, 768], title_bar_px=0)
    cfg["minimap"].update(deepcopy(MARKERS), fixed_rect=RECTS[name])
    cfg["monsters"]["names"] = ["blue_snail"]
    width, height = image.shape[1::-1]
    plan = {"version": 1, "name": name, "image_size": [width, height], "loop": False,
            "capture": capture_signature(cfg),
            "platforms": [{"id": "test-floor", "left": 5, "right": width-6, "y": height-10}],
            "connections": [], "route": [{"platform_id": "test-floor", "x": 6},
                                          {"platform_id": "test-floor", "x": width-7}]}
    digest = hashlib.sha256((FIXTURES / f"profile_{name}_reference.png").read_bytes()).hexdigest()
    profile = MapProfile(profile_key(name), name, plan, image, map_settings_from_config(cfg),
                         plan["capture"], digest)
    assert profile.ready
    return profile, cfg


def frame_with(name, image=None):
    # Only the minimap is recorded; the remainder is a synthetic blank canvas.
    frame = np.zeros((768, 1366, 3), np.uint8)
    x, y, w, h = RECTS[name]
    frame[y:y+h, x:x+w] = recorded(name + "_reference") if image is None else image
    return frame


@pytest.mark.parametrize("name", ["henesys", "construction"])
def test_actual_profile_is_not_accepted_on_other_maps_live_crop(name):
    profile, cfg = make_profile(name)
    other = "construction" if name == "henesys" else "henesys"
    image = recorded("construction_live") if other == "construction" else None
    recognizer = MapProfileRecognizer([profile])
    for _ in range(3):
        match = recognizer.observe(frame_with(other, image), cfg)
    assert match.state == "unknown" and match.score < .78


@pytest.mark.parametrize("name", ["henesys", "construction"])
def test_both_real_profiles_choose_own_crop_even_when_active_crop_is_other_map(name):
    hhg, cfg = make_profile("henesys")
    construction, _ = make_profile("construction")
    cfg["minimap"]["fixed_rect"] = RECTS["construction" if name == "henesys" else "henesys"]
    recognizer = MapProfileRecognizer([hhg, construction])
    image = recorded("construction_live") if name == "construction" else None
    results = [recognizer.observe(frame_with(name, image), cfg) for _ in range(3)]
    assert [r.state for r in results] == ["candidate", "candidate", "matched"]
    assert results[-1].profile_name == name and results[-1].margin >= .10
    assert results[-1].crop == tuple(RECTS[name])


def test_recorded_second_frame_matches_despite_changed_clouds_and_other_player_markers():
    reference = recorded("construction_reference")
    later = recorded("construction_live")
    assert not np.array_equal(reference, later)
    assert stable_map_similarity(reference, later, MARKERS) > .90


@pytest.mark.parametrize("name", ["henesys", "construction"])
def test_blank_or_uniform_crop_never_matches_recorded_geometry(name):
    image = recorded(name + "_reference")
    for value in (0, 64, 128, 255):
        blank = np.full_like(image, value)
        assert stable_map_similarity(image, blank, MARKERS) == 0
        assert stable_map_similarity(blank, blank, MARKERS) == 0


def test_different_maps_stay_distinct_even_if_resized_to_same_dimensions():
    hhg = recorded("henesys_reference")
    construction = recorded("construction_reference")
    assert stable_map_similarity(hhg, cv2.resize(construction, hhg.shape[1::-1]), MARKERS) < .78
    assert stable_map_similarity(construction, cv2.resize(hhg, construction.shape[1::-1]), MARKERS) < .78


def test_moving_colored_player_markers_are_ignored_without_erasing_map_structure():
    base = recorded("construction_reference")
    first, second = base.copy(), base.copy()
    for pos in ((22, 20), (100, 25), (145, 40)):
        cv2.circle(first, pos, 3, MARKERS["player_color_bgr"], -1)
    for pos in ((42, 15), (120, 30), (165, 30)):
        cv2.circle(second, pos, 3, MARKERS["other_color_bgr"], -1)
    masked = stable_map_similarity(first, second, MARKERS)
    unmasked = stable_map_similarity(first, second)
    assert masked > .98 and masked > unmasked


def test_moderate_brightness_change_preserves_real_map_identity():
    image = recorded("construction_live")
    changed = cv2.convertScaleAbs(image, alpha=.8, beta=12)
    assert stable_map_similarity(image, changed, MARKERS) > .95


def test_real_map_change_resets_required_stability_before_recognizing_new_map():
    hhg, cfg = make_profile("henesys")
    construction, _ = make_profile("construction")
    recognizer = MapProfileRecognizer([hhg, construction])
    for _ in range(3):
        previous = recognizer.observe(frame_with("henesys"), cfg)
    assert previous.state == "matched"
    first = recognizer.observe(frame_with("construction", recorded("construction_live")), cfg)
    assert first.state == "candidate" and first.stable_count == 1
    assert recognizer.observe(np.zeros((768, 1366, 3), np.uint8), cfg).state == "unknown"
    again = recognizer.observe(frame_with("construction", recorded("construction_live")), cfg)
    assert again.state == "candidate" and again.stable_count == 1
