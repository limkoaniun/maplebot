import cv2
import numpy as np
import pytest

from maplebot.brain.fsm import Brain, Perception
from maplebot.config import deep_merge, load_config, parse_color_codes
from maplebot.nav.route import RouteFollower, RouteMap
from maplebot.types import Box, Command, Detection

W, H = 1282, 693
P = (641, 346)


class _Route:
    """Stand-in for RouteFollower with a fixed answer."""

    def __init__(self, cmd):
        self.cmd = cmd
        self.calls = []

    def step(self, pos, on_ladder):
        self.calls.append((pos, on_ladder))
        return self.cmd


def _cfg(**over):
    cfg = load_config()
    return deep_merge(cfg, over)


def _p(dets=(), now=0.0, frame_idx=0, player=P, global_pos=(50, 50), minimap_pos=(50, 50),
       minimap_w=150, on_ladder=False):
    return Perception(W, H, player, global_pos, minimap_pos, minimap_w, on_ladder, list(dets), now, frame_idx)


def _det(x, y, w=40, h=30):
    return Detection("m", Box(x, y, w, h), 0.1)


def test_brain_follows_route_without_monsters():
    route = _Route(Command("right"))
    b = Brain(_cfg(), route, 1200)
    assert b.decide(_p()) == Command("right")
    assert route.calls == [((50, 50), False)]
    assert b.state == "hunting"


def test_brain_attacks_left_when_monster_left():
    b = Brain(_cfg(), _Route(Command("right")), 1200)
    cmd = b.decide(_p([_det(500, 330)]))
    assert cmd == Command("left", "none", "attack")


def test_brain_stops_walking_inside_deadzone():
    b = Brain(_cfg(attack={"deadzone_px": 30}), _Route(Command("right")), 1200)
    cmd = b.decide(_p([_det(600, 330)]))  # centre 620, 21px from the player
    assert cmd == Command("none", "none", "attack", face_x="left")
    opposite = b.decide(_p([_det(642, 330)], now=0.1))
    assert opposite == Command("none", "none", "attack", face_x="right")
    assert opposite.replace(action="normal_attack").face_x == "right"


def test_brain_keeps_vertical_route_command_on_ladder():
    b = Brain(_cfg(), _Route(Command("none", "up")), 1200)
    cmd = b.decide(_p([_det(500, 330)], on_ladder=True))
    assert cmd.move_y == "up" and cmd.action == "attack"


def test_brain_uses_aoe_with_crowd():
    b = Brain(_cfg(attack={"aoe_monster_count": 2}), _Route(Command.NONE), 1200)
    cmd = b.decide(_p([_det(500, 330), _det(540, 330)]))
    assert cmd.action == "aoe"


def test_brain_no_aoe_without_aoe_key():
    b = Brain(_cfg(attack={"aoe_monster_count": 2}, keys={"aoe": ""}), _Route(Command.NONE), 1200)
    cmd = b.decide(_p([_det(500, 330), _det(540, 330)]))
    assert cmd.action == "attack"


def test_brain_no_attack_on_tie():
    b = Brain(_cfg(), _Route(Command("right")), 1200)
    assert b.decide(_p([_det(521, 330), _det(721, 330)])) == Command("right")


def test_brain_sweeps_without_route():
    b = Brain(_cfg(search={"trigger_frames": 1}), None, 1200)
    assert b.decide(_p(frame_idx=0)) == Command.NONE
    cmd = b.decide(_p(frame_idx=1, now=0.1))
    assert cmd.move_x in ("left", "right") and cmd.action == "none"
    b2 = Brain(_cfg(search={"trigger_frames": 1, "attack_while_moving": True}), None, 1200)
    b2.decide(_p(frame_idx=0))
    assert b2.decide(_p(frame_idx=1, now=0.1)).action == "attack"


def test_brain_enters_recovering_when_stuck_and_exits_after_timeout():
    b = Brain(_cfg(stuck={"timeout_s": 2.0, "recover_s": 1.0}), _Route(Command("left")), 1200)
    b.decide(_p(now=0.0))
    b.decide(_p(now=1.0))
    cmd = b.decide(_p(now=2.5))
    assert b.state == "recovering"
    assert cmd == Command("right", "none", "jump")       # opposite of the last move, jump once
    assert b.decide(_p(now=2.8)) == Command("right")
    assert b.decide(_p(now=3.6)) == Command("left")      # back to the route
    assert b.state == "hunting"


def test_brain_not_stuck_while_fighting():
    b = Brain(_cfg(stuck={"timeout_s": 1.0}), _Route(Command("left")), 1200)
    for t in (0.0, 0.6, 1.2, 1.8):
        b.decide(_p([_det(500, 330)], now=t))
    assert b.state == "hunting"


def test_brain_stuck_disabled():
    b = Brain(_cfg(stuck={"enable": False, "timeout_s": 1.0}), _Route(Command("left")), 1200)
    for t in (0.0, 0.6, 1.2, 1.8, 5.0):
        b.decide(_p(now=t))
    assert b.state == "hunting"


def test_brain_without_player_fix_still_moves():
    b = Brain(_cfg(), _Route(Command("left")), 1200)
    assert b.decide(_p([_det(500, 330)], player=None)) == Command("left")


def test_hunting_stops_after_grace(tmp_path):
    cfg = _cfg(route={"search_range": 5, "miss_grace_frames": 2})
    d = tmp_path / "r"
    d.mkdir()
    m = np.zeros((50, 50, 3), np.uint8)
    r = m.copy()
    r[10, 10] = (0, 0, 255)  # red in BGR = "left none none"
    cv2.imwrite(str(d / "map.png"), m)
    cv2.imwrite(str(d / "route1.png"), r)
    rm = RouteMap.load(str(d), parse_color_codes(cfg["route"]["color_code"]),
                       parse_color_codes(cfg["route"]["color_code_up_down"]))
    b = Brain(cfg, RouteFollower(rm, cfg["route"]), 1200)
    assert b.decide(_p(global_pos=(10, 10))) == Command("left")
    out = [b.decide(_p(global_pos=(40, 40), now=0.1 * i)) for i in range(1, 4)]
    assert out[0] == Command("left") and out[1] == Command("left") and out[2] == Command.NONE


# --- review fix pass -------------------------------------------------------

def test_brain_keeps_sweeping_when_detection_has_no_target():
    # a monster just outside the attack box (margin only) must not freeze the sweep
    b = Brain(_cfg(search={"trigger_frames": 1, "stall_frames": 0}), None, 1200)
    for i in range(3):
        cmd = b.decide(_p([_det(1000, 330)], frame_idx=i, now=0.1 * i, player=P))
    assert cmd.move_x in ("left", "right")


def test_brain_tie_keeps_previous_target():
    b = Brain(_cfg(), _Route(Command("right")), 1200)
    assert b.decide(_p([_det(500, 330)], now=0.0)).move_x == "left"
    cmd = b.decide(_p([_det(521, 330), _det(721, 330)], now=0.1))
    assert cmd.move_x == "left" and cmd.action == "attack"


def test_brain_stuck_fires_with_untargetable_detection():
    b = Brain(_cfg(stuck={"timeout_s": 1.0, "recover_s": 1.0}), _Route(Command("left")), 1200)
    for t in (0.0, 0.5, 1.2):
        b.decide(_p([_det(1000, 330)], now=t))
    assert b.state == "recovering"


def _circuit_cfg(**over):
    return deep_merge(_cfg(route={"circuit": {
        "enabled": True, "start_idx": 0, "timeout_s": .5, "pixels_per_minimap_x": 16,
        "waypoints": [{"x": 80, "y": 50}, {"x": 20, "y": 50}],
    }}), over)


def test_circuit_combat_suspends_timeout_and_resumes_same_waypoint():
    b = Brain(_circuit_cfg(), None, 1200)
    assert b.decide(_p(now=0)) == Command("right")
    for t in (.4, 1.0, 1.6):
        assert b.decide(_p([_det(500, 330)], now=t)).action == "attack"
    assert b.decide(_p(now=2.0)) == Command("right")
    assert b.circuit.idx == 0 and b.circuit.retries == 0


@pytest.mark.parametrize("action, destination, direction", [("climb", 35, "up"), ("drop", 65, "down")])
def test_circuit_combat_cannot_interrupt_started_vertical_transition(action, destination, direction):
    cfg = _circuit_cfg(route={"circuit": {"waypoints": [
        {"x": 50, "y": destination, "action": action}, {"x": 80, "y": destination}]}})
    b = Brain(cfg, None, 1200)
    first = b.decide(_p(now=0))
    assert first == (Command("none", "up") if action == "climb" else Command.NONE)
    assert b.decide(_p([_det(500, 330)], now=.3)) == Command("none", direction, "jump")
    cmd = b.decide(_p([_det(500, 330)], now=.4,
                      minimap_pos=(50, 52) if action == "drop" else (50, 50)))
    assert cmd == (Command.NONE if action == "drop" else Command("none", direction))
    assert b.target is None and b.circuit.idx == 0


def test_circuit_loot_visits_drop_and_resumes_without_consuming_waypoint_timeout():
    b = Brain(_circuit_cfg(keys={"pickup": "x"}, loot={
        "enabled": True, "disappear_s": .25, "settle_s": .3}), None, 1200)
    b.decide(_p(now=0))
    assert b.decide(_p([_det(500, 330)], now=.1)).action == "attack"
    assert b.loot.goal[0] == pytest.approx(50 + (520 - P[0]) / 16)
    assert b.decide(_p(now=.2)) == Command.NONE
    assert b.decide(_p(now=.5)) == Command("left")
    assert b.decide(_p(now=.6, minimap_pos=(42, 50))) == Command.NONE
    assert b.loot.state == "dwelling"
    assert b.decide(_p(now=1.0, minimap_pos=(42, 50))) == Command("right")
    assert b.loot.state == "idle" and b.circuit.idx == 0 and b.circuit.retries == 0


def test_disabled_pickup_prevents_loot_detour_and_suspend_cancels_pending_drop():
    cfg = _circuit_cfg(keys={"pickup": ""}, loot={"enabled": True})
    b = Brain(cfg, None, 1200)
    b.decide(_p([_det(500, 330)], now=0))
    assert b.decide(_p(now=.1)) == Command("right")
    assert b.loot.goal is None

    b = Brain(deep_merge(cfg, {"keys": {"pickup": "x"}}), None, 1200)
    b.decide(_p([_det(500, 330)], now=0))
    assert b.loot.goal is not None
    b.suspend(.1)
    assert b.decide(_p(now=100)) == Command("right")
    assert b.loot.goal is None and b.circuit.retries == 0


def test_platform_guard_ignores_edge_target_but_keeps_in_bounds_combat_and_loot():
    cfg = _circuit_cfg(keys={"pickup": "x"}, loot={"enabled": True},
                       route={"circuit": {"waypoints": [{"x": 20, "y": 50}, {"x": 80, "y": 50}],
                                          "platforms": [{"y": 50, "left": 20, "right": 80}]}})
    b = Brain(cfg, None, 1200)
    # Monster centre 761 is 120 screen px right: minimap 75 + 120/16 = 82.5, beyond edge.
    cmd = b.decide(_p([_det(741, 330)], minimap_pos=(75, 50)))
    assert cmd == Command("left") and b.target is None and b.loot.goal is None
    cmd = b.decide(_p([_det(500, 330)], minimap_pos=(75, 50), now=.1))
    assert cmd.action == "attack" and cmd.move_x == "left"
    assert 20 <= b.loot.goal[0] <= 80


def test_platform_limit_releases_outward_combat_motion_preserving_facing_attack():
    b = Brain(_circuit_cfg(route={"circuit": {"platforms": [{"y": 50, "left": 20, "right": 80}]}}), None, 1200)
    # The final command clamp is independent of target filtering, including a retained target.
    cmd = b._combat(_p([_det(741, 330)], minimap_pos=(80, 50)), Command.NONE, "right")
    assert cmd == Command("none", "none", "attack", face_x="right")


def test_platform_guard_does_not_clamp_explicit_waypoint_motion():
    b = Brain(_circuit_cfg(route={"circuit": {"waypoints": [{"x": 90, "y": 50}, {"x": 20, "y": 50}],
                                              "platforms": [{"y": 50, "left": 20, "right": 80}]}}), None, 1200)
    assert b.decide(_p(minimap_pos=(85, 50))) == Command("right")


def _farming_brain():
    return Brain(_circuit_cfg(attack={"range_x": 70, "range_y": 48, "min_overlap_px": 300,
                                     "deadzone_px": 28, "tie_margin_px": 12},
        route={"circuit": {"platforms": [{"left": 20, "right": 80, "y": 50}]}}), None, 1200)


def test_circuit_brief_target_loss_holds_without_attacking_then_resumes_same_point():
    b = _farming_brain()
    assert b.decide(_p([_det(571, 331)], now=0)).action == "attack"
    for now in (.1, .3, .39):
        assert b.decide(_p(now=now)) == Command.NONE
        assert b.state == "reacquiring_target" and b.target is None
    assert b.decide(_p(now=.41)) == Command("right")
    assert b.circuit.idx == 0 and b.circuit.retries == 0


def test_circuit_reacquires_same_fresh_monster_instead_of_turning_to_closer_other_side():
    b = _farming_brain()
    assert b.decide(_p([_det(571, 331)], now=0)).move_x == "left"
    assert b.decide(_p(now=.1)) == Command.NONE
    cmd = b.decide(_p([_det(574, 331), _det(641, 331)], now=.2))
    assert cmd == Command("left", "none", "attack")
    assert b.target == "left"  # the opposite close monster must not control this target's deadzone


def test_circuit_initial_direction_tie_selects_one_fresh_attackable_monster():
    b = _farming_brain()
    cmd = b.decide(_p([_det(671, 331), _det(571, 331)], now=0))
    assert cmd == Command("left", "none", "attack")
    assert b.circuit.idx == 0


def test_circuit_out_of_range_detection_never_turns_reacquisition_into_stale_attack():
    b = _farming_brain()
    b.decide(_p([_det(571, 331)], now=0))
    assert b.decide(_p([_det(780, 331)], now=.1)) == Command.NONE
    assert b.target is None and b.state == "reacquiring_target"


@pytest.mark.parametrize("lost_context", ["floor", "player", "bounds", "ladder", "suspend"])
def test_combat_continuity_clears_when_supported_context_or_permission_is_lost(lost_context):
    b = _farming_brain()
    b.decide(_p([_det(571, 331)], now=0))
    kwargs = {"now": .1}
    if lost_context == "floor":
        kwargs["minimap_pos"] = (50, 55)
    elif lost_context == "player":
        kwargs["player"] = None
    elif lost_context == "bounds":
        kwargs["minimap_pos"] = (81, 50)
    elif lost_context == "ladder":
        kwargs["on_ladder"] = True
    else:
        b.suspend(.05)
    b.decide(_p(**kwargs))
    assert b._combat_memory is None and b.state != "reacquiring_target"


def test_atomic_jump_keeps_priority_over_existing_combat_memory():
    cfg = _circuit_cfg(route={"circuit": {
        "platforms": [{"left": 20, "right": 80, "y": 50}, {"left": 20, "right": 80, "y": 35}],
        "waypoints": [{"x": 50, "y": 50, "action": "walk"},
                      {"x": 55, "y": 35, "action": "jump", "from_x": 50, "from_y": 50}]}})
    b = Brain(cfg, None, 1200)
    assert b.decide(_p([_det(571, 331)], now=0)).action == "attack"
    b.circuit.idx, b.circuit._selected = 1, True
    cmd = b.decide(_p([_det(571, 331)], now=.1))
    assert cmd.action != "attack" and b.state.startswith("circuit_")
    assert b._combat_memory is None and b.target is None


def test_circuit_approaches_fresh_search_margin_monster_before_walking_away():
    b = _farming_brain()
    cmd = b.decide(_p([_det(531, 331)], now=0))  # 90 px left, beyond the 70 px attack box
    assert cmd == Command("left") and b.target is None
    assert b.state == "pursuing_target" and b.circuit.idx == 0
    assert b.decide(_p([_det(571, 331)], now=.2)).action == "attack"


def test_pursuit_loss_stops_immediately_and_never_refreshes_its_waypoint_budget():
    b = _farming_brain()
    b.circuit.timeout_s = 10
    assert b.decide(_p([_det(531, 331)], now=0)) == Command("left")
    for now in (.1, .3, .39):
        assert b.decide(_p(now=now)) == Command.NONE
        assert b.target is None
    assert b.decide(_p(now=.41)) == Command("right")
    assert b.decide(_p([_det(531, 331)], now=.7)) == Command("left")
    for now in (1.21, 1.4, 1.8, 2.4):
        assert b.decide(_p([_det(531, 331)], now=now)) == Command("right")
        assert b.target is None and b.state != "pursuing_target"
    assert b.circuit.idx == 0


@pytest.mark.parametrize("context", ["other_floor", "edge", "outside", "unsupported", "ladder"])
def test_pursuit_requires_same_supported_floor_and_interior_target(context):
    b = _farming_brain()
    kwargs = {"now": 0}
    det = _det(531, 331)
    if context == "other_floor":
        det = _det(531, 280)
    elif context in ("edge", "outside"):
        det = _det(711, 331)
        kwargs["minimap_pos"] = (74 if context == "edge" else 75, 50)
    elif context == "unsupported":
        kwargs["minimap_pos"] = (50, 52)
    else:
        kwargs["on_ladder"] = True
    cmd = b.decide(_p([det], **kwargs))
    assert cmd.action == "none" and b.target is None
    assert b.state != "pursuing_target" and b._pursuit_started is None


def test_pursuit_never_overrides_jump_or_movement_verification():
    cfg = _circuit_cfg(route={"circuit": {
        "platforms": [{"left": 20, "right": 80, "y": 50}, {"left": 20, "right": 80, "y": 35}],
        "waypoints": [{"x": 50, "y": 50, "action": "walk"},
                      {"x": 55, "y": 35, "action": "jump", "from_x": 50, "from_y": 50}]}},
        attack={"range_x": 70, "range_y": 48, "min_overlap_px": 300})
    b = Brain(cfg, None, 1200)
    assert b.decide(_p([_det(531, 331)], now=0)) == Command("left")
    b.circuit.idx, b.circuit._selected = 1, True
    cmd = b.decide(_p([_det(531, 331)], now=.1))
    assert cmd.move_x == "none" and b.state.startswith("circuit_")
    test = Brain(deep_merge(cfg, {"route": {"circuit": {"movement_test": True}}}), None, 1200)
    test.decide(_p([_det(531, 331)], now=0))
    assert test.state.startswith("circuit_") and test._pursuit_started is None


def test_suspending_pursuit_clears_stale_hold_without_replenishing_budget():
    b = _farming_brain()
    b.decide(_p([_det(531, 331)], now=0))
    b.suspend(.1)
    assert b.decide(_p(now=.2)) == Command("right")
    assert b.decide(_p([_det(531, 331)], now=2)) == Command("right")
    assert b.state != "pursuing_target"
