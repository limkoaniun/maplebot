"""Recorded glyph/ladder crops: identity acquisition must survive occlusion safely.

Fixtures contain only the player's local tag/nearby scenery or the known HUD
lookalike. They deliberately exclude chat and the rest of the desktop.
"""
from pathlib import Path
import json

import cv2
import numpy as np
import pytest

from maplebot.bot import Bot
from maplebot.config import deep_merge, load_config
from maplebot.vision.monsters import MonsterLibrary
from maplebot.vision.player import locate_by_template


ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "nametags"
EXPECTED = json.loads((FIX / "expected.json").read_text())
NAMETAG = ROOT / "assets" / "player-nametag.png"
HEAD = ROOT / "assets" / "player-head.png"

# The template and the recorded frames show the character's in-game name, so they are
# git-ignored local files (the repository is public). Without them these regressions skip.
_LOCAL = [NAMETAG, HEAD] + [FIX / f"{name}.png" for name in EXPECTED]
_MISSING = [p.relative_to(ROOT).as_posix() for p in _LOCAL if not p.is_file()]
pytestmark = pytest.mark.skipif(bool(_MISSING), reason="local private nametag files absent: " + ", ".join(_MISSING))


def frame(name):
    image = cv2.imread(str(FIX / f"{name}.png"))
    assert image is not None
    return image


def bot(center=9, threshold=.12, center_threshold=.11):
    cfg = deep_merge(load_config(), {"player": {
        "locator": "hybrid",
        "nametag": {
            "path": str(NAMETAG),
            "offset": [0, -32], "split_width": 20, "diff_thres": threshold,
            "center_occlusion_px": center,
            "center_occlusion_diff_thres": center_threshold,
        },
        "appearance": {"path": str(HEAD)},
        "hybrid": {"max_fallback_frames": 0},
    }})
    return Bot(cfg, None, None, MonsterLibrary(), None)


def test_recorded_ladder_explains_original_failure_and_recovers_identity():
    b = bot()
    image = frame("construction_ladder_occlusion")
    old = locate_by_template(image, b._tmpl, b._tmpl_mask, None, (0, -32), .12,
                             split_width=20)
    assert old is None
    fix = b._locate_player(image)
    assert fix is not None
    assert fix.pos == tuple(EXPECTED["construction_ladder_occlusion"]["player"])
    assert fix.source == "nametag_occlusion"
    assert fix.score == pytest.approx(.09545755, abs=.002)
    assert b._hybrid_primary_pos == fix.pos


def test_occlusion_mask_keeps_original_mask_and_both_sides():
    b = bot()
    original = cv2.imread(str(NAMETAG), -1)[:, :, 3]
    assert np.array_equal(b._tmpl_mask, original)
    assert np.array_equal(b._nametag_occlusion_mask[:, :21], original[:, :21])
    assert np.array_equal(b._nametag_occlusion_mask[:, 30:], original[:, 30:])
    assert np.count_nonzero(b._nametag_occlusion_mask[:, 21:30]) == 0
    assert np.count_nonzero(b._nametag_occlusion_mask) == 159


@pytest.mark.parametrize("name", ["construction_visible", "construction_hanging", "henesys_rope"])
def test_recorded_visible_names_still_use_ordinary_primary(name):
    fix = bot()._locate_player(frame(name))
    assert fix is not None
    assert fix.pos == tuple(EXPECTED[name]["player"])
    assert fix.source == "nametag"
    assert fix.score < .12


def test_recorded_hud_lookalike_and_blank_do_not_acquire_identity():
    b = bot()
    assert b._locate_player(frame("hud_distractor")) is None
    assert b._hybrid_primary_pos is None
    assert b._locate_player(np.full((100, 150, 3), 180, np.uint8)) is None
    assert b._hybrid_primary_pos is None


def test_occlusion_fallback_does_not_reacquire_distant_name_after_lock():
    b = bot()
    small = frame("construction_ladder_occlusion")
    initial = b._locate_player(small)
    assert initial is not None
    b.player = initial
    distant = np.full((100, 450, 3), 180, np.uint8)
    distant[:small.shape[0], 300:300 + small.shape[1]] = small
    assert b._locate_player(distant) is None
    assert b._hybrid_primary_pos == initial.pos


def test_occlusion_fallback_is_opt_in_and_never_loosens_primary_threshold():
    image = frame("construction_ladder_occlusion")
    assert bot(center=0)._locate_player(image) is None
    assert bot(threshold=.09, center_threshold=.5)._locate_player(image) is None
