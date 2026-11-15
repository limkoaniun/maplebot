"""Targets must stand on the player's own surface, not merely overlap the attack band.

Geometry is the saved Construction Site profile (floors 17 minimap px apart,
16 screen px per minimap px, player fix about 12 px above the feet). Same-floor
sprite bottoms were recorded at player y +5..+20 in `log/final-circuit-v4.jsonl`.
"""
import pytest

from maplebot.brain.fsm import Brain, Perception
from maplebot.config import deep_merge, load_config
from maplebot.types import Box, Detection

W, H, P = 1366, 768, (641, 346)
FLOOR1 = {"id": "floor-1", "left": 42, "right": 132.35, "y": 88}
FLOOR2 = {"id": "floor-2", "left": 42, "right": 88.6, "y": 71}
FLOOR4 = {"id": "floor-4", "left": 42, "right": 88.537, "y": 38}
BARRICADE = {"id": "barricade", "left": 60.2875, "right": 67.5375, "y": 34}
STAIR1 = {"id": "stair-1", "left": 66.79, "y": 87.71, "right": 89.6, "right_y": 71.33,
          "knots": [[70.6, 84.58], [72.41, 81.9], [80.35, 81.96], [84.29, 77.02], [86.6, 75.4]]}
STAIR2 = {"id": "stair-2", "left": 66.787, "y": 71.037, "right": 89.6, "right_y": 54.663,
          "knots": [[70.6, 67.912], [72.412, 65.225], [80.35, 65.287], [84.287, 60.35], [86.6, 58.725]]}
PLATFORMS = [FLOOR1, FLOOR2, FLOOR4, BARRICADE, STAIR1, STAIR2]


def brain(surface=True, waypoint=(50, 71), **over):
    cfg = deep_merge(load_config(), {
        "keys": {"pickup": "x"},
        "attack": {"range_x": 70, "range_y": 48, "search_margin": 45, "min_overlap_px": 300,
                   "deadzone_px": 28, "tie_margin_px": 12, "surface": {"enabled": surface}},
        "loot": {"enabled": True, "settle_s": 1.2, "max_approach_s": 4.0, "tolerance_x": 1, "disappear_s": .25},
        "route": {"circuit": {"enabled": True, "start_idx": 0, "timeout_s": 15, "pixels_per_minimap_x": 16,
                  "tolerance_x": 1.0, "tolerance_y": 1.0, "platform_tolerance_y": 5, "platforms": PLATFORMS,
                  "waypoints": [{"x": waypoint[0], "y": waypoint[1]}, {"x": waypoint[0] + 4, "y": waypoint[1]}]}}})
    return Brain(deep_merge(cfg, over), None, 3111)


def see(dets, mm, now=0.0, player=P):
    return Perception(W, H, player, None, mm, 190, False, list(dets), now, 0)


def mob(dx, bottom_dy, w=62, h=64, name="orange_mushroom", player=P):
    """Sprite centred `dx` px from the player whose bottom is `bottom_dy` px below the player fix."""
    return Detection(name, Box(player[0] + dx - w // 2, player[1] + bottom_dy - h, w, h), 0.05)


# (player minimap position, detection, description)
OTHER_SURFACE = [
    ((85, 71), mob(48, 12 + 40), "floor-2 player, mushroom 40 px lower on the top of stair-1"),
    ((69, 86), mob(32, 12 + 34), "stair-1 base player, mushroom on floor-1 underneath"),
    ((63, 34), mob(32, 12 + 64), "barricade player, mushroom on floor-4 below"),
]


@pytest.mark.parametrize("mm,det,_why", OTHER_SURFACE, ids=[c[2] for c in OTHER_SURFACE])
def test_band_overlap_alone_attacks_a_monster_on_another_surface(mm, det, _why):
    # Reproduces the handoff limitation: the horizontal platform filter and the
    # 48 px band both accept these sprites although they stand elsewhere.
    b = brain(surface=False, waypoint=(50, mm[1]))
    assert b.decide(see([det], mm)).action == "attack"


@pytest.mark.parametrize("mm,det,_why", OTHER_SURFACE, ids=[c[2] for c in OTHER_SURFACE])
def test_surface_test_rejects_monster_on_another_surface_without_attack_pursuit_or_loot_goal(mm, det, _why):
    b = brain(waypoint=(50, mm[1]))
    for now in (0.0, .15, .3):
        cmd = b.decide(see([det], mm, now))
        assert cmd.action == "none" and cmd.face_x == "none"
        assert b.target is None and b.loot.goal is None
        assert b.state not in {"hunting", "pursuing_target", "reacquiring_target"}


@pytest.mark.parametrize("bottom_dy", [5, 12, 20])
def test_same_floor_monster_within_recorded_foot_band_is_still_attacked(bottom_dy):
    b = brain()
    cmd = b.decide(see([mob(48, bottom_dy)], (75, 71)))
    assert cmd.action == "attack" and b.target == "right" and b.loot.goal == (78.0, 71.0)


def test_recorded_uphill_snail_on_the_players_own_stair_remains_a_target():
    # log/combat-frame-04 (15 Nov 2026, capture_sequence 20469, trace sequence
    # 185): player (726,467) at minimap (71,67) on stair-2, blue_snail
    # [753,420,34,33] on a higher tread of the same stair. It stays eligible
    # with the test on or off; the slope cases below pin the slope term itself.
    player = (726, 467)
    snail = Detection("blue_snail", Box(753, 420, 34, 33), 0.17)
    below = Detection("stump", Box(633, 503, 60, 50), 0.2)  # floor underneath, never eligible
    b = brain(waypoint=(72, 65.225))
    cmd = b.decide(see([snail, below], (71, 67), player=player))
    assert cmd.action == "attack" and b.target == "right"
    assert b.loot.goal[0] == pytest.approx(73.75) and b.loot.goal[1] == pytest.approx(65.235, abs=.01)


def test_recorded_barricade_swing_at_a_mushroom_on_the_floor_below_is_rejected():
    # log/three-circuits-v3.jsonl (15 Nov 2026): eight attack samples from the
    # barricade (minimap 63-64,34) at an orange_mushroom standing on floor-4,
    # sprite bottom 73 px below the player fix (61 px below the barricade top).
    player, mushroom = (684, 477), Detection("orange_mushroom", Box(697, 495, 63, 55), 0.06)
    assert brain(surface=False, waypoint=(62, 34)).decide(see([mushroom], (63, 34), player=player)).action == "attack"
    b = brain(waypoint=(62, 34))
    assert b.decide(see([mushroom], (63, 34), player=player)).action == "none" and b.loot.goal is None


@pytest.mark.parametrize("player,mm,box", [
    ((660, 523), (47, 104), (696, 500, 64, 53)),   # player one minimap px above the ground mid-knockback
    ((677, 485), (57, 37), (583, 449, 60, 67)),
    ((732, 461), (100, 71), (783, 442, 60, 50)),
])
def test_recorded_same_floor_targets_seen_while_the_player_is_knocked_up_stay_eligible(player, mm, box):
    # Recorded sprite bottoms reach 30-32 px below the player fix while the
    # player is briefly lifted; they are on the player's floor and remain targets.
    b = brain(waypoint=(50, round(mm[1] / 17) * 17 + 3),
              route={"circuit": {"platforms": PLATFORMS + [{"id": "ground", "left": 0, "right": 101, "y": 105},
                                                             {"id": "floor-2-right", "left": 89, "right": 111, "y": 71}]}})
    assert b.decide(see([Detection("m", Box(*box), 0.1)], mm, player=player)).action == "attack"


def test_monster_hopping_on_the_players_floor_stays_a_target():
    b = brain()
    for now, lift in ((0.0, 0), (.15, 15), (.3, 30), (.45, 15), (.6, 0)):
        assert b.decide(see([mob(48, 12 - lift)], (75, 71), now)).action == "attack"


def test_monster_far_above_the_surface_is_not_attacked_and_never_counted_as_a_kill():
    b = brain()
    assert b.decide(see([mob(48, 12)], (75, 71), 0.0)).action == "attack"
    for now in (.15, .3, .45, .6, .75, .9):
        cmd = b.decide(see([mob(48, 12 - 50)], (75, 71), now))  # 50 px above the floor
        assert cmd.action == "none" and not b.state.startswith("looting_")
    assert b.decide(see([mob(48, 12)], (75, 71), 1.05)).action == "attack"


@pytest.mark.parametrize("surface,collects", [(False, False), (True, True)])
def test_monster_on_lower_stair_does_not_postpone_collecting_a_real_kill(surface, collects):
    lower = mob(48, 12 + 40)
    b = brain(surface=surface)
    assert b.decide(see([mob(-40, 12, 34, 33, "blue_snail"), lower], (85, 71), 0.0)).action == "attack"
    states = []
    for now in (.15, .3, .45, .6, .75, .9, 1.05):
        b.decide(see([lower], (85, 71), now))  # the snail is gone; only the unreachable one remains
        states.append(b.state)
    # Without the surface test the bot turns to swing at the unreachable monster instead.
    assert any(s.startswith("looting_") for s in states) is collects


@pytest.mark.parametrize("bad", [{"feet_offset_px": float("nan")}, {"above_px": -1}, {"below_px": 500}])
def test_surface_settings_are_validated(bad):
    with pytest.raises(ValueError, match="attack.surface"):
        brain(attack={"surface": bad})


def test_shipped_default_keeps_the_surface_test_off():
    assert load_config()["attack"]["surface"]["enabled"] is False
    mm, det, _ = OTHER_SURFACE[0]
    cfg = deep_merge(load_config(), {"attack": {"range_x": 70, "range_y": 48, "min_overlap_px": 300},
        "route": {"circuit": {"enabled": True, "pixels_per_minimap_x": 16, "platform_tolerance_y": 5,
                  "platforms": PLATFORMS, "waypoints": [{"x": 50, "y": 71}, {"x": 54, "y": 71}]}}})
    assert Brain(cfg, None, 3111).decide(see([det], mm)).action == "attack"


SNAIL = dict(w=34, h=33, name="blue_snail")


def test_downhill_monster_on_the_players_own_stair_needs_the_slope_term():
    # Player at (70,85) on stair-1; a snail 40 px left stands 2.06 minimap px
    # (33 screen px) lower on the same stair. A flat expectation would call it
    # another surface (+33 > 22); following the stair accepts it.
    b = brain(waypoint=(72, 82))
    assert b.decide(see([mob(-40, 12 + 33, **SNAIL)], (70, 85))).action == "attack"


def test_monster_level_with_the_player_above_a_falling_stair_is_not_on_it():
    # Same place, but the sprite is level with the player although the stair
    # under it is 39 px lower: it stands on something else.
    level = mob(-48, 12, **SNAIL)
    assert brain(surface=False, waypoint=(72, 82)).decide(see([level], (70, 85))).action == "attack"
    b = brain(waypoint=(72, 82))
    assert b.decide(see([level], (70, 85))).action == "none" and b.loot.goal is None


def test_vertical_minimap_scale_is_configurable():
    downhill = mob(-40, 12 + 33, **SNAIL)
    b = brain(waypoint=(72, 82), route={"circuit": {"pixels_per_minimap_y": 40}})
    assert b.decide(see([downhill], (70, 85))).action == "none"  # 82 px expected, 33 seen


@pytest.mark.parametrize("lift,eligible", [(30, True), (32, True), (33, False), (45, False)])
def test_above_px_bounds_how_far_over_the_surface_a_target_may_float(lift, eligible):
    # A taller attack band than the live 48 px lets the band itself admit these.
    tall = {"attack": {"range_y": 80}}
    assert brain(surface=False, **tall).decide(see([mob(48, 12 - lift)], (75, 71))).action == "attack"
    assert (brain(**tall).decide(see([mob(48, 12 - lift)], (75, 71))).action == "attack") is eligible


def test_living_target_that_crossed_the_floor_seam_is_not_collected_as_a_kill():
    # floor-2 ends at 88.6 and floor-2-right starts at 89 at the same height.
    b = brain()
    assert b.decide(see([mob(64, 12)], (84, 71), 0.0)).action == "attack"
    for now in (.15, .3, .45, .6, .75, .9, 1.05, 1.2):
        b.decide(see([mob(128, 12)], (84, 71), now))  # alive at minimap x=92, outside the platform
        assert not b.state.startswith("looting_")


def test_brain_remembers_kill_sites_through_a_second_fight_and_pursuit():
    loot = {"loot": {"max_pending": 2, "walk_speed_x": 10}}
    a, b_mob = mob(-40, 12, **SNAIL), mob(48, 12)
    # First target vanishes while the second is fought: its site (72.5) is kept.
    brain_ = brain(**loot)
    assert brain_.decide(see([a, b_mob], (75, 71), 0.0)).move_x == "left"
    assert brain_.decide(see([b_mob], (75, 71), .15)).move_x == "right"
    assert brain_.loot.pending_count == 1 and brain_.loot.goal == (78.0, 71.0)
    # A raw detection standing on that site, although filtered from targets, removes it.
    brain_.decide(see([b_mob, mob(-40, 12 - 50, **SNAIL)], (75, 71), .3))
    assert brain_.loot.pending_count == 0
    # A begun visit interrupted by pursuit of a newly seen monster is resumed later.
    brain_ = brain(**loot)
    brain_.decide(see([a], (75, 71), 0.0))
    now = .15
    while brain_.state != "looting_approaching":
        brain_.decide(see([], (75, 71), now))
        now += .15
        assert now < 2
    assert brain_.decide(see([mob(100, 12)], (75, 71), now)).move_x == "right"
    assert brain_.state == "pursuing_target" and brain_.loot.pending_count == 1
