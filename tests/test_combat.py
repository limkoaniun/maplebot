from maplebot.brain.combat import AttackTimer, attack_boxes, choose_target, search_box
from maplebot.types import Box, Detection

CFG = {
    "mode": "directional",
    "range_x": 300,
    "range_y": 70,
    "aoe_range_x": 400,
    "aoe_range_y": 170,
    "cooldown_s": 0.9,
    "aoe_cooldown_s": 1.0,
    "aoe_monster_count": 3,
    "search_margin": 50,
    "min_overlap_px": 1500,
    "tie_margin_px": 50,
}
W, H = 1282, 693
P = (641, 346)


def _det(x, y, w=40, h=30, name="m"):
    return Detection(name, Box(x, y, w, h), 0.1)


def test_attack_boxes_directional():
    boxes = attack_boxes(P, CFG, W, H)
    assert boxes["left"] == Box(341, 311, 300, 70)
    assert boxes["right"] == Box(641, 311, 300, 70)


def test_attack_boxes_directional_clamped_at_left_edge():
    boxes = attack_boxes((100, 346), CFG, W, H)
    assert boxes["left"] == Box(0, 311, 100, 70)


def test_attack_boxes_aoe_centered():
    boxes = attack_boxes(P, dict(CFG, mode="aoe"), W, H)
    assert list(boxes) == ["aoe"]
    assert boxes["aoe"] == Box(441, 261, 400, 170)


def test_search_box_union_plus_margin():
    assert search_box(P, CFG, W, H) == Box(291, 261, 700, 170)


def test_choose_target_left_only():
    assert choose_target([_det(500, 330)], P, CFG, 1200, W, H) == "left"


def test_choose_target_right_only():
    assert choose_target([_det(700, 330)], P, CFG, 1200, W, H) == "right"


def test_choose_target_none_outside_range():
    assert choose_target([_det(100, 330)], P, CFG, 1200, W, H) is None
    assert choose_target([_det(700, 100)], P, CFG, 1200, W, H) is None


def test_choose_target_small_overlap_ignored():
    # only a 5x30 sliver of the monster is inside the left box
    assert choose_target([_det(306, 330)], P, CFG, 1200, W, H) is None


def test_choose_target_wrong_side_ignored():
    # centre at x=660 (right of the player) although part of the box overlaps the left range
    assert choose_target([_det(630, 330, 60, 30)], P, CFG, 1200, W, H) == "right"


def test_choose_target_tie_returns_none():
    assert choose_target([_det(521, 330), _det(721, 330)], P, CFG, 1200, W, H) is None


def test_choose_target_prefers_nearer_by_margin():
    assert choose_target([_det(421, 330), _det(721, 330)], P, CFG, 1200, W, H) == "right"


def test_choose_target_aoe_mode():
    cfg = dict(CFG, mode="aoe")
    assert choose_target([_det(500, 330)], P, cfg, 1200, W, H) == "aoe"
    assert choose_target([_det(100, 330)], P, cfg, 1200, W, H) is None


def test_choose_target_empty():
    assert choose_target([], P, CFG, 1200, W, H) is None


def test_attack_timer_cooldown():
    t = AttackTimer(CFG)
    assert t.ready("attack", 0.0)
    t.mark("attack", 0.0)
    assert not t.ready("attack", 0.5)
    assert t.ready("attack", 0.95)
    assert t.ready("aoe", 0.5)
    t.mark("aoe", 0.5)
    assert not t.ready("aoe", 1.4)
    assert t.ready("aoe", 1.6)
