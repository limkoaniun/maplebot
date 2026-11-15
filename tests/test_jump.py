import pytest

from maplebot.brain.fsm import Brain
from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command
from tests.test_fsm import _cfg, _det, _p


def jump_circuit(**over):
    return MinimapCircuit({
        "start_idx": 1, "loop": False,
        "platforms": [{"left": 0, "right": 15, "y": 50}, {"left": 25, "right": 40, "y": 50}],
        "waypoints": [{"x": 10, "y": 50, "dwell_s": 0},
                      {"x": 30, "y": 50, "action": "jump", "from_x": 10, "from_y": 50, "dwell_s": 0}],
        **over,
    })


def test_jump_takeoff_tolerance_can_differ_from_rope_alignment():
    nav = jump_circuit(align_tolerance_x=.4, jump_align_tolerance_x=1)
    assert nav.align_tolerance_x == .4
    assert nav.step((11, 50), 0) == Command.NONE
    assert nav.state == "jump_settling"
    assert nav.step((11, 50), .3).action == "jump"
    outside = jump_circuit(align_tolerance_x=.4, jump_align_tolerance_x=1)
    outside.step((12, 50), 0)
    assert outside.state == "jump_aligning"


def test_jump_aligns_launches_observes_flight_and_confirms_landing_before_completion():
    nav = jump_circuit()
    assert nav.step((5, 50), 0) == Command.NONE  # precision release before approach
    assert nav.step((5, 50), .21) == Command("right")
    assert nav.step((10, 50), .3) == Command.NONE
    assert nav.step((10, 50), .6) == Command("right", "none", "jump")
    assert nav.step((12, 48), .65) == Command("right", "none", "jump")
    assert nav.step((20, 40), .85) == Command("right")
    assert nav.state == "airborne" and not nav.completed
    assert nav.step((28, 45), 1.05) == Command.NONE
    assert nav.step((30, 50), 1.2) == Command.NONE and not nav.completed
    assert nav.state == "jump_landing"
    nav.step((30, 50), 1.35)
    assert nav.step((30, 50), 1.5) == Command.NONE and nav.completed
    assert nav.state == "completed" and nav.step((10, 50), 2) == Command.NONE


def test_jump_cannot_complete_from_horizontal_motion_without_observed_departure():
    nav = jump_circuit(max_retries=0, jump_flight_timeout_s=.5)
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((30, 50), .5)
    nav.step((30, 50), .7)
    assert not nav.completed
    assert nav.step((30, 50), .9) == Command.NONE and nav.state == "blocked"


def test_takeoff_waits_for_stable_source_floor_height():
    nav = jump_circuit()
    assert nav.step((10, 49), 0) == Command.NONE
    assert nav.step((10, 50), .2) == Command.NONE
    assert nav.step((10, 50), .3) == Command.NONE
    assert nav.step((10, 50), .5).action == "jump"


def test_jump_does_not_treat_upward_apex_as_a_landing():
    nav = jump_circuit(platforms=[{"left": 0, "right": 15, "y": 50}, {"left": 25, "right": 40, "y": 35}],
                       waypoints=[{"x": 10, "y": 50}, {"x": 30, "y": 35, "action": "jump",
                                   "from_x": 10, "from_y": 50, "dwell_s": 0}])
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((30, 35), .5)
    nav.step((30, 35), .8)
    assert not nav.completed and nav.state == "airborne"
    nav.step((30, 33), .9)
    nav.step((30, 35), 1.0)
    nav.step((30, 35), 1.15)
    nav.step((30, 35), 1.3)
    assert nav.completed


def test_jump_failed_launch_retries_once_then_blocks_without_endless_jumping():
    nav = jump_circuit(max_retries=1, jump_flight_timeout_s=.4, recovery_s=.2)
    nav.step((10, 50), 0)
    assert nav.step((10, 50), .3).action == "jump"
    assert nav.step((10, 50), .8) == Command.NONE and nav.state == "jump_retry"
    assert nav.step((10, 50), 1.1) == Command.NONE
    assert nav.step((10, 50), 1.4).action == "jump"
    assert nav.step((10, 50), 1.9) == Command.NONE and nav.state == "blocked"
    assert nav.retries == 1 and not nav.completed
    assert nav.step((10, 50), 2.2) == Command.NONE


def test_jump_landing_requires_target_platform_containment():
    nav = jump_circuit(max_retries=0, jump_flight_timeout_s=.7,
                       waypoints=[{"x": 10, "y": 50}, {"x": 25, "y": 50, "action": "jump",
                                   "from_x": 10, "from_y": 50, "dwell_s": 0}])
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((20, 40), .5)
    nav.step((24, 50), .7)  # within arrival tolerance, but outside destination platform
    nav.step((24, 50), 1.1)
    assert not nav.completed and nav.state == "blocked"


def test_jump_pause_does_not_reissue_launch_or_count_as_stable_landing():
    nav = jump_circuit()
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.suspend(.31)
    assert nav.step((12, 48), 10).action == "none"
    nav.step((20, 40), 10.2)
    nav.step((30, 50), 10.4)
    nav.suspend(10.5)
    nav.step((30, 50), 20)
    nav.step((30, 50), 20.2)
    assert not nav.completed
    nav.step((30, 50), 20.3)
    assert nav.completed


def test_separate_same_height_platforms_are_selected_by_horizontal_position():
    nav = jump_circuit()
    assert nav.platform_at((10, 50)).right == 15
    assert nav.platform_at((30, 50)).left == 25
    assert nav._containing_platform((20, 50)) is None
    # A route starting on the left platform must not choose a closer right-platform walk across a gap.
    nav = jump_circuit(start_idx=None, waypoints=[{"x": 0, "y": 50}, {"x": 25, "y": 50}])
    assert nav.step((15, 50), 0) == Command("left") and nav.idx == 0


def test_jump_rejects_missing_takeoff_and_endpoints_outside_platforms():
    with pytest.raises(ValueError, match="takeoff"):
        jump_circuit(waypoints=[{"x": 10, "y": 50}, {"x": 30, "y": 50, "action": "jump"}])
    with pytest.raises(ValueError, match="configured platforms"):
        jump_circuit(waypoints=[{"x": 10, "y": 50}, {"x": 20, "y": 50, "action": "jump",
                               "from_x": 10, "from_y": 50}])


def test_jump_and_movement_test_bypass_combat_and_loot():
    for movement_test in (False, True):
        cfg = _cfg(keys={"pickup": "x"}, loot={"enabled": True}, route={"circuit": {
            "enabled": True, "start_idx": 1, "movement_test": movement_test,
            "waypoints": [{"x": 50, "y": 50}, {"x": 70, "y": 50, "action": "jump",
                          "from_x": 50, "from_y": 50}]}})
        brain = Brain(cfg, None, 1200)
        assert brain.decide(_p([_det(500, 330)], now=0)) == Command.NONE
        assert brain.decide(_p([_det(500, 330)], now=.3)) == Command("right", "none", "jump")
        assert brain.target is None and brain.loot.goal is None


def test_one_pass_walk_test_stops_at_completion_even_with_monsters_present():
    cfg = _cfg(route={"circuit": {"enabled": True, "start_idx": 0, "loop": False, "movement_test": True,
        "waypoints": [{"x": 50, "y": 50, "dwell_s": 0}, {"x": 60, "y": 50, "dwell_s": 0}]}})
    brain = Brain(cfg, None, 1200)
    assert brain.decide(_p([_det(500, 330)], now=0)) == Command.NONE
    assert brain.decide(_p([_det(500, 330)], now=.1)) == Command("right")
    assert brain.decide(_p(now=.2, minimap_pos=(60, 50))) == Command.NONE
    assert brain.circuit.completed
    brain.movement_test = False
    assert brain.decide(_p([_det(500, 330)], now=.3)) == Command.NONE


def test_landing_requires_three_observations_even_when_two_span_full_dwell():
    nav = jump_circuit()
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((20, 40), .5)
    nav.step((30, 50), .7)
    nav.step((30, 50), 1.0)
    assert nav.state == "jump_landing" and not nav.completed
    assert nav._jump_landing_frames == 2
    nav.step((30, 50), 1.1)
    assert nav.completed


@pytest.mark.parametrize("reverse", [False, True])
def test_supported_forward_jump_overshoot_holds_still_for_three_landing_observations(reverse):
    takeoff, target, landing = (10, 50), (30, 50), (33, 50)
    if reverse:
        takeoff, target, landing = (30, 50), (10, 50), (7, 50)
    nav = jump_circuit(tolerance_x=1, tolerance_y=.75, jump_landing_overshoot_x=3,
        waypoints=[{"x": takeoff[0], "y": takeoff[1]},
                   {"x": target[0], "y": target[1], "action": "jump",
                    "from_x": takeoff[0], "from_y": takeoff[1], "dwell_s": 0}])
    nav.step(takeoff, 0)
    nav.step(takeoff, .3)
    nav.step((20, 40), .5)
    for now in (.7, 1.0):
        assert nav.step(landing, now) == Command.NONE  # do not steer back toward the gap
    assert nav._jump_landing_frames == 2 and not nav.completed
    assert nav.step(landing, 1.1) == Command.NONE and nav.completed


@pytest.mark.parametrize("position,allowance,right_edge", [
    ((33, 50), 0, 40),  # default retains the ordinary target tolerance
    ((27, 50), 3, 40),  # behind target, toward the source, is not overshoot
    ((34, 50), 3, 40),  # more than the configured allowance
    ((33, 48), 3, 40),  # not supported at destination height
    ((33, 50), 3, 31),  # beyond the actual destination platform
])
def test_jump_overshoot_never_relaxes_source_side_height_or_platform_bounds(position, allowance, right_edge):
    nav = jump_circuit(tolerance_x=1, tolerance_y=.75, jump_landing_overshoot_x=allowance,
        platforms=[{"left": 0, "right": 15, "y": 50}, {"left": 25, "right": right_edge, "y": 50}])
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((20, 40), .5)
    for now in (.7, 1.0, 1.1):
        nav.step(position, now)
    assert not nav.completed and nav._jump_landing_frames == 0


def test_brief_forward_overshoot_at_apex_cannot_confirm_a_landing():
    nav = jump_circuit(tolerance_x=1, jump_landing_overshoot_x=3,
        platforms=[{"left": 0, "right": 15, "y": 50}, {"left": 25, "right": 40, "y": 35}],
        waypoints=[{"x": 10, "y": 50}, {"x": 30, "y": 35, "action": "jump",
                   "from_x": 10, "from_y": 50, "dwell_s": 0}])
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    for now in (.5, .7, 1.0):
        nav.step((33, 35), now)
    assert not nav.completed and not nav._jump_descending and nav.state == "airborne"


def _short_upward_jump(**overrides):
    return jump_circuit(tolerance_x=1, tolerance_y=.75,
        platforms=[{"left": 40, "right": 60, "y": 38}, {"left": 62, "right": 70, "y": 34}],
        waypoints=[{"x": 59, "y": 38}, {"x": 63, "y": 34, "action": "jump",
                   "from_x": 59, "from_y": 38, "dwell_s": 0}], **overrides)


def test_short_upward_jump_accepts_long_stable_support_without_rounded_descent():
    nav = _short_upward_jump()
    nav.step((59, 38), 0)
    nav.step((59, 38), .3)
    for now in (.5, 1.0):
        assert nav.step((62, 34), now) == Command.NONE
    assert nav._jump_landing_frames == 2 and not nav.completed
    assert nav.step((62, 34), 1.2) == Command.NONE
    assert nav.completed and nav._jump_departed and not nav._jump_descending


def test_no_descent_landing_requires_minimum_flight_even_after_stable_dwell():
    nav = _short_upward_jump()
    nav.step((59, 38), 0)
    nav.step((59, 38), .3)
    for now in (.35, .7, 1.01):
        nav.step((62, 34), now)
    assert nav._jump_landing_frames == 3 and not nav.completed
    nav.step((62, 34), 1.11)
    assert nav.completed


@pytest.mark.parametrize("position", [(62, 32), (61, 34)])
def test_long_stable_reading_without_destination_support_does_not_land(position):
    nav = _short_upward_jump()
    nav.step((59, 38), 0)
    nav.step((59, 38), .3)
    for now in (.5, .8, 1.2, 1.4):
        nav.step(position, now)
    assert not nav.completed and nav._jump_landing_frames == 0


def test_no_descent_support_confirmation_cannot_outlive_flight_timeout():
    nav = _short_upward_jump(jump_flight_timeout_s=.7, max_retries=0)
    nav.step((59, 38), 0)
    nav.step((59, 38), .3)
    for now in (.5, .8, 1.1):
        nav.step((62, 34), now)
    assert not nav.completed and nav.state == "blocked"


@pytest.mark.parametrize("value", [-1, 3.1, float("nan"), float("inf")])
def test_jump_overshoot_allowance_is_bounded(value):
    with pytest.raises(ValueError, match="jump_landing_overshoot_x"):
        jump_circuit(jump_landing_overshoot_x=value)


def test_three_fast_observations_do_not_bypass_stable_dwell():
    nav = jump_circuit()
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((20, 40), .5)
    for now in (.7, .75, .8):
        nav.step((30, 50), now)
    assert nav._jump_landing_frames == 3 and not nav.completed
    nav.step((30, 50), 1.0)
    assert nav.completed


@pytest.mark.parametrize("interruption", ["suspend", "missing", "changed_y", "leave_landing"])
def test_landing_count_restarts_after_interruption_or_height_change(interruption):
    nav = jump_circuit()
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    nav.step((20, 40), .5)
    nav.step((30, 50), .7)
    nav.step((30, 50), .8)
    if interruption == "suspend":
        nav.suspend(.9)
    elif interruption == "missing":
        nav.step(None, .9)
    elif interruption == "leave_landing":
        nav.step((30, 47), .9)
    landing = (30, 49) if interruption == "changed_y" else (30, 50)
    nav.step(landing, 1.0)
    nav.step(landing, 1.3)
    assert not nav.completed and nav._jump_landing_frames == 2
    nav.step(landing, 1.4)
    assert nav.completed


@pytest.mark.parametrize("takeoff,target,gap_x,land_x,direction", [
    (66, 69, 67, 68, "right"), (77, 74, 76, 75, "left"),
])
def test_short_stair_jump_corrects_gap_position_inside_target_deadzone(
        takeoff, target, gap_x, land_x, direction):
    nav = jump_circuit(tolerance_y=.75, platforms=[{"left": 42, "right": 111, "y": 88},
                                                 {"left": 68, "right": 75, "y": 86}],
        waypoints=[{"x": takeoff, "y": 88},
                   {"x": target, "y": 86, "action": "jump", "from_x": takeoff,
                    "from_y": 88, "dwell_s": 0}])
    nav.step((takeoff, 88), 0)
    assert nav.step((takeoff, 88), .3) == Command(direction, "none", "jump")
    assert nav.step((gap_x, 86), .5) == Command(direction)
    assert nav.state == "airborne" and not nav.completed  # crossing target height while rising
    assert nav.step((gap_x, 82), .7) == Command(direction)
    assert nav.step((land_x, 86), .9) == Command.NONE
    nav.step((land_x, 86), 1.2)
    assert not nav.completed
    nav.step((land_x, 86), 1.3)
    assert nav.completed
