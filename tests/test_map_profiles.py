from copy import deepcopy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from maplebot.config import load_config
from maplebot.editor_bridge import capture_signature
from maplebot.map_profiles import (MapProfile, MapProfileRecognizer, apply_profile,
    align_map, load_profiles, map_settings_from_config, profile_config_matches, profile_key,
    read_profile, stable_map_similarity)


def example(name="Map A", rect=(20, 30, 90, 60), seed=3):
    cfg = load_config()
    cfg["game_window"].update(title="Test game", working_size=[400, 240])
    cfg["minimap"].update(fixed_rect=list(rect), player_color_bgr=[0, 255, 255], player_tolerance=15)
    cfg["monsters"].update(names=["slime"])
    image = np.random.default_rng(seed).integers(0, 180, (rect[3], rect[2], 3), dtype=np.uint8)
    plan = {"version": 1, "name": name, "image_size": list(rect[2:]), "loop": True,
            "platforms": [{"id": "floor", "left": 3, "right": rect[2]-4, "y": rect[3]-10}],
            "connections": [], "route": [{"platform_id": "floor", "x": 10}, {"platform_id": "floor", "x": rect[2]-10}],
            "capture": capture_signature(cfg)}
    _, encoded = cv2.imencode(".png", image)
    digest = hashlib.sha256(encoded.tobytes()).hexdigest()
    profile = MapProfile(profile_key(name), name, plan, image, map_settings_from_config(cfg), plan["capture"], digest)
    frame = np.zeros((240, 400, 3), np.uint8)
    x, y, w, h = rect
    frame[y:y+h, x:x+w] = image
    return cfg, profile, frame


def test_candidates_use_their_own_crops_and_need_stability():
    cfg, first, _ = example()
    _, second, frame = example("Map B", (130, 35, 180, 100), 7)
    recognizer = MapProfileRecognizer([first, second])
    assert recognizer.observe(frame, cfg).state == "candidate"
    assert recognizer.observe(frame, cfg).stable_count == 2
    match = recognizer.observe(frame, cfg)
    assert match.state == "matched" and match.profile is second
    assert match.crop == (130, 35, 180, 100)
    assert match.score > .99


def test_unknown_frame_resets_stability_and_returns_no_applicable_profile():
    cfg, profile, frame = example()
    recognizer = MapProfileRecognizer([profile])
    recognizer.observe(frame, cfg)
    match = recognizer.observe(np.zeros_like(frame), cfg)
    assert match.state == "unknown" and match.profile is None
    assert recognizer.observe(frame, cfg).stable_count == 1
    recognizer.reset()
    assert recognizer.observe(frame, cfg).stable_count == 1


def test_identical_maps_are_ambiguous_even_after_many_frames():
    cfg, first, frame = example()
    second = deepcopy(first)
    second.key, second.name = profile_key("Duplicate"), "Duplicate"
    recognizer = MapProfileRecognizer([first, second], stable_frames=1)
    for _ in range(4):
        match = recognizer.observe(frame, cfg)
        assert match.state == "ambiguous" and match.profile is None


@pytest.mark.parametrize("field,value", [("title", "Other game"), ("working_size", [800, 480]), ("title_bar_px", 5)])
def test_incompatible_capture_never_matches(field, value):
    cfg, profile, frame = example()
    cfg["game_window"][field] = value
    assert MapProfileRecognizer([profile], stable_frames=1).observe(frame, cfg).state == "unknown"


def test_blank_maps_and_changed_structure_never_identify_a_map():
    blank = np.zeros((60, 90, 3), np.uint8)
    assert stable_map_similarity(blank, blank) == 0
    _, profile, _ = example()
    other = np.random.default_rng(99).integers(0, 180, profile.reference.shape, dtype=np.uint8)
    assert stable_map_similarity(profile.reference, other) < .78


def test_moving_player_dots_are_masked_without_erasing_the_map():
    _, profile, _ = example()
    a, b = profile.reference.copy(), profile.reference.copy()
    cv2.circle(a, (20, 20), 2, (0, 255, 255), -1)
    cv2.circle(b, (65, 40), 2, (0, 255, 255), -1)
    assert stable_map_similarity(a, b, profile.map_settings["minimap"]) > .98


def _recorded_map(name):
    return cv2.imread(str(Path(__file__).parent / "fixtures/map_recognition" / name))


def _recorded_settings():
    return {"player_color_bgr": [43, 223, 255], "other_color_bgr": [0, 0, 255], "player_tolerance": 40}


@pytest.mark.parametrize("name,offset", [("construction-reference.png", (0, 0)),
    ("construction-crowded.png", (0, 0)), ("construction-panned.png", (23, 0))])
def test_recorded_translucent_crowded_and_panned_minimap_aligns_unique_terrain(name, offset):
    reference = _recorded_map("construction-reference.png")
    current = _recorded_map(name)
    score, measured, margin = align_map(reference, current, _recorded_settings())
    assert score > .9 and measured == offset and margin >= .10
    if offset != (0, 0):
        assert stable_map_similarity(reference, current, _recorded_settings()) < .78


@pytest.mark.parametrize("kind", ["blank", "brown", "other_map", "flipped", "missing_tower", "small_fragment", "too_far"])
def test_terrain_alignment_rejects_cross_map_and_insufficient_structural_evidence(kind):
    reference = _recorded_map("construction-reference.png")
    if kind == "blank":
        current = np.zeros_like(reference)
    elif kind == "brown":
        current = np.full_like(reference, (30, 90, 150))
    elif kind == "other_map":
        current = cv2.resize(_recorded_map("hhg-reference.png"), reference.shape[1::-1])
    elif kind == "flipped":
        current = reference[:, ::-1].copy()
    elif kind == "missing_tower":
        current = reference.copy()
        current[:85] = 90  # identical brick strip cannot identify a different upper map
    elif kind == "small_fragment":
        current = np.zeros_like(reference)
        current[95:] = reference[95:]
    else:
        current = cv2.warpAffine(reference, np.float32([[1, 0, -90], [0, 1, 0]]), reference.shape[1::-1])
    assert align_map(reference, current, _recorded_settings())[0] < .78


def test_repeated_terrain_with_multiple_valid_offsets_is_rejected():
    tile = np.random.default_rng(16).integers(30, 230, (20, 30, 3), dtype=np.uint8)
    repeated = np.tile(tile, (6, 7, 1))
    score, _, margin = align_map(repeated, repeated)
    assert score == 0 and margin < .10


def test_matched_profile_keeps_identity_when_unique_minimap_offset_changes():
    cfg, profile, frame = example(rect=(20, 30, 210, 128))
    profile.reference_bgr = _recorded_map("construction-reference.png")
    profile.map_settings["minimap"].update(_recorded_settings())
    recognizer = MapProfileRecognizer([profile])
    frame[30:158, 20:230] = profile.reference_bgr
    for _ in range(3):
        match = recognizer.observe(frame, cfg)
    assert match.state == "matched" and match.offset == (0, 0)
    frame[30:158, 20:230] = _recorded_map("construction-panned.png")
    match = recognizer.observe(frame, cfg)
    assert match.state == "matched" and match.offset == (23, 0)
    assert tuple(raw + delta for raw, delta in zip((104, 84), match.offset)) == (127, 84)


def test_profile_replaces_all_map_settings_and_preserves_character_config():
    cfg, first, _ = example()
    _, second, _ = example("Map B", (130, 35, 180, 100), 7)
    cfg["keys"]["jump"] = "c"
    cfg["health"]["hp_percent"] = 62
    cfg["player"]["custom_identity"] = "keep"
    cfg["route"]["circuit"] = {"movement_test": False, "stale_setting": 123, "waypoints": [{"x": 999}]}
    second.map_settings["monsters"]["names"] = ["stump"]
    result = apply_profile(cfg, second)
    assert result["keys"] == cfg["keys"] and result["health"] == cfg["health"] and result["player"] == cfg["player"]
    assert result["minimap"]["fixed_rect"] == [130, 35, 180, 100]
    assert result["monsters"]["names"] == ["stump"]
    assert "stale_setting" not in result["route"]["circuit"]
    assert profile_config_matches(result, second)
    assert cfg["minimap"]["fixed_rect"] == [20, 30, 90, 60]


@pytest.mark.parametrize("change", ["disabled", "waypoints", "platforms", "monsters", "crop", "digest"])
def test_stale_or_modified_active_configuration_cannot_keep_identity(change):
    cfg, profile, _ = example()
    cfg = apply_profile(cfg, profile)
    if change == "disabled": cfg["route"]["circuit"]["enabled"] = False
    if change == "waypoints": cfg["route"]["circuit"]["waypoints"][0]["x"] += 1
    if change == "platforms": cfg["route"]["circuit"]["platforms"][0]["right"] -= 1
    if change == "monsters": cfg["monsters"]["names"] = ["wrong"]
    if change == "crop": cfg["minimap"]["fixed_rect"][0] += 1
    if change == "digest": cfg["map_profile"]["reference_sha256"] = "wrong"
    assert not profile_config_matches(cfg, profile)


def test_temporary_movement_test_preserves_map_identity_but_not_wrong_crop():
    cfg, profile, _ = example()
    cfg = apply_profile(cfg, profile)
    cfg["route"]["circuit"] = {"movement_test": True, "waypoints": [{"test": "temporary"}]}
    assert profile_config_matches(cfg, profile)
    assert apply_profile(cfg, profile, preserve_test=True)["route"]["circuit"] == cfg["route"]["circuit"]
    cfg["minimap"]["fixed_rect"][0] += 1
    assert not profile_config_matches(cfg, profile)


def test_legacy_profile_is_readable_but_requires_explicit_settings(tmp_path):
    cfg, profile, frame = example()
    folder = tmp_path / "config/map_profiles"
    folder.mkdir(parents=True)
    path = folder / f"{profile.key}.json"
    path.write_text(json.dumps({"version": 1, "plan": profile.plan}), encoding="utf-8")
    cv2.imwrite(str(path.with_suffix(".png")), profile.reference)
    legacy = load_profiles(tmp_path)[0]
    assert legacy.map_settings is None and not legacy.ready
    assert MapProfileRecognizer([legacy], stable_frames=1).observe(frame, cfg).state == "incomplete"
    with pytest.raises(ValueError, match="older profile"):
        apply_profile(cfg, legacy)


def test_preflight_reports_a_different_known_map_without_applying_it():
    cfg, first, _ = example()
    _, second, frame = example("Map B", (130, 35, 180, 100), 7)
    cfg = apply_profile(cfg, first)
    match = MapProfileRecognizer([first, second], stable_frames=1).preflight(frame, cfg)
    assert match.state == "wrong_profile" and match.profile is None
    assert cfg["map_profile"]["active"] == first.key
