import cv2
import numpy as np
import pytest

from maplebot.types import Box
from maplebot.vision.monsters import MonsterLibrary, color_spread, detect_hp_bars, detect_monsters
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
