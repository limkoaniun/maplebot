import cv2
import numpy as np
import pytest

from maplebot.types import Box
from maplebot.vision.monsters import MonsterLibrary, MonsterTemplate, color_spread, detect_hp_bars, detect_monsters
from maplebot.vision.template import make_mask
from tests.synth import make_frame, make_sprite, paste_sprite

CFG = {
    "mode": "color",
    "diff_thres": 0.25,
    "scales": [1.0],
    "min_color_spread": 30,
    "max_matches_per_template": 50,
    "hp_bar": {"enable": False, "color_bgr": [71, 204, 64], "tolerance": 10},
    "per_monster": {},
}


@pytest.fixture
def lib(tmp_path):
    d = tmp_path / "monster" / "testmob"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "testmob_0.png"), make_sprite())
    return MonsterLibrary.load(str(tmp_path / "monster"), ["testmob"], {}, [1.0])


def _roi_with_sprite(x=200, y=150, w=600, h=400, sprite=None):
    roi = make_frame(w, h)
    sprite = make_sprite() if sprite is None else sprite
    bgr, mask = make_mask(sprite)
    paste_sprite(roi, bgr, mask, x, y)
    return roi


def test_library_loads_and_flips(tmp_path):
    d = tmp_path / "monster" / "mob"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "mob_0.png"), make_sprite())
    lib = MonsterLibrary.load(str(tmp_path / "monster"), ["mob"], {}, [1.0, 1.2])
    assert len(lib.templates) == 4
    assert {t.scale for t in lib.templates} == {1.0, 1.2}
    assert lib.min_area == 40 * 30
    assert lib.names == ["mob"]


def test_library_loads_all_when_names_empty(tmp_path):
    for n in ("a", "b"):
        d = tmp_path / "monster" / n
        d.mkdir(parents=True)
        cv2.imwrite(str(d / f"{n}.png"), make_sprite())
    lib = MonsterLibrary.load(str(tmp_path / "monster"), [], {}, [1.0])
    assert lib.names == ["a", "b"]


def test_library_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MonsterLibrary.load(str(tmp_path), ["nope"], {}, [1.0])


def test_detect_finds_pasted_sprite(lib):
    dets = detect_monsters(_roi_with_sprite(), lib, CFG, exclude=None)
    assert len(dets) == 1
    d = dets[0]
    assert d.name == "testmob"
    assert abs(d.box.x - 200) <= 1 and abs(d.box.y - 150) <= 1
    assert d.box.w == 40 and d.box.h == 30
    assert d.score < 0.05


def test_detect_ignores_player_box(lib):
    dets = detect_monsters(_roi_with_sprite(), lib, CFG, exclude=Box(180, 130, 100, 100))
    assert dets == []


def test_detect_rejects_grey_patch_by_color_spread(lib):
    grey = make_sprite()
    bgr, mask = make_mask(grey)
    g = cv2.cvtColor(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    g[mask == 0] = (0, 255, 0)
    roi = _roi_with_sprite(sprite=g)
    cfg = dict(CFG, diff_thres=0.9)
    assert detect_monsters(roi, lib, cfg, exclude=None) == []
    cfg["min_color_spread"] = 0
    assert len(detect_monsters(roi, lib, cfg, exclude=None)) >= 1


def test_detect_grayscale_mode(lib):
    cfg = dict(CFG, mode="grayscale")
    dets = detect_monsters(_roi_with_sprite(), lib, cfg, exclude=None)
    assert len(dets) == 1 and abs(dets[0].box.x - 200) <= 1


def test_color_verification_rejects_same_brightness_wrong_colour():
    sprite = np.full((20, 20, 3), (0, 0, 255), np.uint8)
    mask = np.full((20, 20), 255, np.uint8)
    library = MonsterLibrary([MonsterTemplate("red_mob", sprite, mask, 1.0)], ["red_mob"])
    roi = np.zeros((40, 90, 3), np.uint8)
    roi[10:30, 10:30] = sprite
    roi[10:30, 55:75] = (0, 130, 0)  # Same grey value as red; different hue.
    config = dict(CFG, mode="grayscale", diff_thres=0.01, min_color_spread=0)
    assert any(d.box.x >= 50 for d in detect_monsters(roi, library, config, None))
    config["color_diff_thres"] = 0.2
    detected = detect_monsters(roi, library, config, None)
    assert len(detected) == 1 and detected[0].box.x == 10


def test_detect_per_monster_threshold_override(lib):
    sprite = make_sprite()
    sprite[10:14, 10:14] = (10, 10, 10)  # perturb so the match score is strictly positive
    roi = _roi_with_sprite(sprite=sprite)
    dets = detect_monsters(roi, lib, CFG, exclude=None)
    assert len(dets) == 1 and dets[0].score > 0
    cfg = dict(CFG, per_monster={"testmob": {"diff_thres": dets[0].score / 2}})
    assert detect_monsters(roi, lib, cfg, exclude=None) == []


def test_detect_offsets_by_roi_origin(lib):
    dets = detect_monsters(_roi_with_sprite(), lib, CFG, exclude=None, roi_origin=(100, 50))
    assert abs(dets[0].box.x - 300) <= 1 and abs(dets[0].box.y - 200) <= 1


def test_detect_hp_bars():
    roi = make_frame(300, 200)
    roi[50:53, 100:120] = (71, 204, 64)
    dets = detect_hp_bars(roi, (71, 204, 64), 10, guess_w=70, guess_h=50, roi_origin=(10, 20))
    assert len(dets) == 1
    assert dets[0].name == "hp_bar"
    assert dets[0].box == Box(110, 80, 70, 50)


def test_color_spread_grey_vs_colour():
    grey = np.full((5, 5, 3), 120, np.uint8)
    colour = np.zeros((5, 5, 3), np.uint8)
    colour[:, :, 2] = 200
    mask = np.full((5, 5), 255, np.uint8)
    assert color_spread(grey, mask) == 0
    assert color_spread(colour, mask) == 200


@pytest.mark.parametrize("mask_value", [1, 255])
def test_fast_grayscale_difference_matches_opencv_binary_mask(mask_value):
    rng = np.random.default_rng(113)
    roi = rng.integers(5, 250, (83, 137), dtype=np.uint8).astype(np.float32)
    sprite = rng.integers(5, 250, (25, 31, 3), dtype=np.uint8)
    mask = (rng.random((25, 31)) > .25).astype(np.uint8) * mask_value
    template = MonsterTemplate("mob", sprite, mask, 1)
    library = MonsterLibrary([template], ["mob"])
    gray = template.gray.astype(np.float32)
    roi[21:46, 47:78][mask > 0] = gray[mask > 0]
    expected = cv2.matchTemplate(roi, gray, cv2.TM_SQDIFF_NORMED, mask=mask)
    actual = library.gray_difference(roi, roi * roi, template)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    assert cv2.minMaxLoc(actual)[2] == (47, 21)


@pytest.mark.parametrize("image_value,template_value,mask_value", [
    (0, 0, 255), (0, 120, 255), (120, 0, 255), (120, 120, 0),
])
def test_fast_grayscale_difference_rejects_undefined_black_scores(
        image_value, template_value, mask_value):
    roi = np.full((40, 50), image_value, np.float32)
    template = MonsterTemplate("black", np.full((12, 15, 3), template_value, np.uint8),
                               np.full((12, 15), mask_value, np.uint8), 1)
    library = MonsterLibrary([template], ["black"])
    scores = library.gray_difference(roi, roi * roi, template)
    assert np.all(np.isfinite(scores)) and np.all(scores == 1)


def test_grayscale_native_backend_remains_available(lib):
    cfg = dict(CFG, mode="grayscale")
    fast = detect_monsters(_roi_with_sprite(), lib, cfg, None)
    native = detect_monsters(_roi_with_sprite(), lib, dict(cfg, fast_grayscale=False), None)
    assert [(d.name, d.box) for d in fast] == [(d.name, d.box) for d in native]
    assert [d.score for d in fast] == pytest.approx([d.score for d in native], abs=1e-5)


def test_grayscale_difference_preserves_soft_mask_semantics():
    rng = np.random.default_rng(73)
    roi = rng.integers(5, 250, (40, 50), dtype=np.uint8).astype(np.float32)
    sprite = rng.integers(5, 250, (12, 15, 3), dtype=np.uint8)
    mask = rng.random((12, 15), dtype=np.float32)
    template = MonsterTemplate("soft", sprite, mask, 1)
    library = MonsterLibrary([template], ["soft"])
    expected = cv2.matchTemplate(roi, template.gray.astype(np.float32),
                                cv2.TM_SQDIFF_NORMED, mask=mask)
    np.testing.assert_array_equal(library.gray_difference(roi, roi * roi, template), expected)
