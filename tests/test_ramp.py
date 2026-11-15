import pytest

from maplebot.brain.fsm import Brain
from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command
from tests.test_fsm import _cfg, _det, _p


def ramp_circuit(**overrides):
    return MinimapCircuit({
        "start_idx": 1, "loop": False,
        "platforms": [{"left": 0, "right": 65, "y": 80},
                      {"left": 75, "right": 100, "y": 63}],
        "waypoints": [{"x": 63, "y": 80},
                      {"x": 80, "y": 63, "action": "ramp", "from_x": 63, "from_y": 80,
                       "dwell_s": 0}],
        **overrides,
    })


def test_ramp_walks_up_slope_and_requires_stable_landing():
    nav = ramp_circuit()
    assert nav.transition_active
    assert nav.step((63, 80), 0) == Command.NONE
    assert nav.step((63, 80), .3) == Command("right")
    for pos, now in [((67, 76), .5), ((72, 71), .7), ((77, 66), .9)]:
        assert nav.step(pos, now) == Command("right")
        assert nav.state == "ramping" and nav.transition_active
    assert nav.step((80, 63), 1.1) == Command.NONE
    assert nav.state == "ramp_landing" and not nav.completed
    nav.step((80, 63), 1.3)
    assert not nav.completed
    nav.step((80, 63), 1.5)
    assert nav.completed and not nav.transition_active
    assert nav.step((63, 80), 2) == Command.NONE


def test_ramp_descends_without_down_or_jump_and_stops_after_landing():
    nav = ramp_circuit(waypoints=[{"x": 80, "y": 63},
        {"x": 63, "y": 80, "action": "ramp", "from_x": 80, "from_y": 63, "dwell_s": 0}])
    nav.step((80, 63), 0)
    assert nav.step((80, 63), .3) == Command("left")
    assert nav.step((75, 68), .6) == Command("left")
    assert nav.step((67, 76), .9) == Command("left")
    assert nav.step((63, 80), 1.1) == Command.NONE
    nav.step((63, 80), 1.3)
    nav.step((63, 80), 1.5)
    assert nav.completed


@pytest.mark.parametrize("reverse", [False, True])
def test_small_destination_overshoot_requires_supported_stable_landing(reverse):
    points = [{"x": 63, "y": 80},
              {"x": 80, "y": 63, "action": "ramp", "from_x": 63, "from_y": 80, "dwell_s": 0}]
    start, end, overshoot = (63, 80), (80, 63), (83, 63)
    if reverse:
        points = [{"x": 80, "y": 63},
                  {"x": 63, "y": 80, "action": "ramp", "from_x": 80, "from_y": 63, "dwell_s": 0}]
        start, end, overshoot = (80, 63), (63, 80), (60, 80)
    nav = ramp_circuit(waypoints=points, tolerance_x=1, tolerance_y=.75)
    nav.step(start, 0)
    nav.step(start, .3)
    assert nav.step(end, .5) == Command.NONE
    assert nav.step(overshoot, .7) == Command.NONE
    assert nav.state == "ramp_landing" and not nav.completed
    nav.step(overshoot, .9)
    assert nav.completed  # horizontal knockback preserves fresh stable floor support


@pytest.mark.parametrize("reverse", [False, True])
def test_confirmed_destination_entry_tolerates_horizontal_knockback_within_floor(reverse):
    points = [{"x": 63, "y": 80},
              {"x": 80, "y": 63, "action": "ramp", "from_x": 63,
               "from_y": 80, "dwell_s": .6}]
    start, end, pushed = (63, 80), (80, 63), [(86, 63), (90, 63)]
    if reverse:
        points = [{"x": 80, "y": 63},
                  {"x": 63, "y": 80, "action": "ramp", "from_x": 80,
                   "from_y": 63, "dwell_s": .6}]
        start, end, pushed = (80, 63), (63, 80), [(57, 80), (53, 80)]
    nav = ramp_circuit(waypoints=points, tolerance_x=1, tolerance_y=.75)
    nav.step(start, 0)
    nav.step(start, .3)
    assert nav.step(end, .5) == Command.NONE
    assert nav.step(pushed[0], .9) == Command.NONE
    assert not nav.completed and nav.state == "ramp_landing"
    assert nav.step(pushed[1], 1.2) == Command.NONE
    assert nav.completed


@pytest.mark.parametrize("position", [(86, 65), (86, 61), (101, 63), (74, 63)])
def test_knockback_after_arrival_still_rejects_missing_destination_support(position):
    nav = ramp_circuit(tolerance_x=1, tolerance_y=.75)
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    nav.step((80, 63), .5)
    assert nav.step(position, .9) == Command.NONE
    assert nav.state == "blocked" and not nav.completed


def test_landing_confirmation_restarts_after_brief_airborne_height():
    nav = ramp_circuit(tolerance_x=1, tolerance_y=.75)
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    nav.step((80, 63), .5)
    nav.step((80, 61), .8)
    assert not nav.completed
    nav.step((80, 63), 1.1)
    nav.step((86, 63), 1.5)
    assert not nav.completed  # dwell alone does not replace three fresh observations
    nav.step((87, 63), 1.7)
    assert nav.completed


@pytest.mark.parametrize("position", [(82, 64), (84, 63), (82, 59)])
def test_destination_overshoot_rejects_wrong_height_or_more_than_three_pixels(position):
    nav = ramp_circuit(tolerance_x=1, tolerance_y=.75)
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    assert nav.step(position, .5) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "left ramp corridor"


def test_destination_overshoot_never_extends_platform_horizontal_bounds():
    nav = ramp_circuit(tolerance_x=1, tolerance_y=.75,
                       platforms=[{"left": 0, "right": 65, "y": 80},
                                  {"left": 75, "right": 81, "y": 63}])
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    assert nav.step((82, 63), .5) == Command.NONE
    assert nav.state == "blocked" and not nav.completed


@pytest.mark.parametrize("position", [(72, 80), (84, 63), (59, 80)])
def test_departing_ramp_corridor_blocks_without_random_recovery(position):
    nav = ramp_circuit(max_retries=5)
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    assert nav.step(position, .5) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "left ramp corridor"
    assert nav.step((63, 80), 1) == Command.NONE
    assert nav.step((63, 80), 1.5) == Command.NONE
    assert nav.retries == 0


def test_stalled_ramp_times_out_without_jumping_or_recovery():
    nav = ramp_circuit(timeout_s=1, max_retries=5)
    nav.step((63, 80), 0)
    assert nav.step((63, 80), .3) == Command("right")
    assert nav.step((63, 80), .6) == Command("right")
    assert nav.step((63, 80), 1.1) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "ramp timeout" and nav.retries == 0


def test_endpoint_x_without_target_floor_never_completes():
    nav = ramp_circuit(timeout_s=1.5)
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    assert nav.step((80, 66), .7) == Command.NONE
    nav.step((80, 66), 1)
    assert not nav.completed
    nav.step((80, 66), 1.6)
    assert nav.state == "blocked"


def test_pause_or_missing_position_does_not_count_as_stable_ramp_landing():
    nav = ramp_circuit()
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    nav.step((80, 63), .7)
    nav.suspend(.8)
    nav.step((80, 63), 10)
    nav.step((80, 63), 10.2)
    assert not nav.completed
    assert nav.step(None, 10.3) == Command.NONE
    nav.step((80, 63), 11)
    nav.step((80, 63), 11.2)
    assert not nav.completed
    nav.step((80, 63), 11.4)
    assert nav.completed


def test_ramp_does_not_accept_target_tolerance_outside_destination_platform():
    nav = ramp_circuit(waypoints=[{"x": 63, "y": 80},
        {"x": 75, "y": 63, "action": "ramp", "from_x": 63, "from_y": 80, "dwell_s": 0}])
    nav.step((63, 80), 0)
    nav.step((63, 80), .3)
    assert nav.step((74, 63), .7) == Command("right")
    nav.step((74, 63), 1.1)
    assert not nav.completed


def construction_stair_circuit():
    return ramp_circuit(tolerance_x=1, tolerance_y=.75,
        platforms=[{"left": 66.787, "y": 71.037, "right": 89.6, "right_y": 54.663,
                    "knots": [[70.6, 67.912], [72.412, 65.225], [80.35, 65.287],
                              [84.287, 60.35], [86.6, 58.725]]},
                   {"left": 89, "right": 111, "y": 55}],
        waypoints=[{"x": 82, "y": 63.21789916179833},
                   {"x": 92, "y": 55, "action": "ramp", "from_x": 82,
                    "from_y": 63.21789916179833, "dwell_s": .6}])


def test_piecewise_ramp_uses_drawn_surface_instead_of_endpoint_chord():
    nav = construction_stair_circuit()
    nav.step((82, 63), 0)
    assert nav.step((82, 63), .3) == Command("right")
    assert nav.step((87, 59), .5) == Command("right")
    # Live rounded ascent: the endpoint chord rejects this by >4px, but the
    # actual stair support interval remains inside the unchanged 4px corridor.
    assert nav.step((88, 54), .7) == Command("right")
    assert nav.state == "ramping" and not nav.completed
    assert nav.step((90, 55), .9) == Command("right")
    assert nav.step((92, 55), 1.1) == Command.NONE
    nav.step((92, 55), 1.5)
    assert not nav.completed
    nav.step((92, 55), 1.8)
    assert nav.completed


@pytest.mark.parametrize("position", [(88, 50), (88, 62), (90, 60), (96, 55)])
def test_piecewise_corridor_still_blocks_real_departure(position):
    nav = construction_stair_circuit()
    nav.step((82, 63), 0)
    nav.step((82, 63), .3)
    assert nav.step(position, .5) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "left ramp corridor"


def test_piecewise_corridor_does_not_turn_tolerated_height_into_a_landing():
    nav = construction_stair_circuit()
    nav.step((82, 63), 0)
    nav.step((82, 63), .3)
    for now in (.5, .8, 1.1):
        assert nav.step((92, 58), now) == Command.NONE
    assert not nav.completed and nav.state == "ramping"


def test_ramp_blocks_combat_and_loot_even_before_slope_entry():
    cfg = _cfg(keys={"pickup": "x"}, loot={"enabled": True}, route={"circuit": {
        "enabled": True, "start_idx": 1,
        "waypoints": [{"x": 50, "y": 50},
                      {"x": 70, "y": 30, "action": "ramp", "from_x": 50, "from_y": 50}]}})
    brain = Brain(cfg, None, 1200)
    assert brain.decide(_p([_det(500, 330)], now=0)) == Command.NONE
    assert brain.decide(_p([_det(500, 330)], now=.3)) == Command("right")
    assert brain.target is None and brain.loot.goal is None


@pytest.mark.parametrize("point", [
    {"x": 80, "y": 63, "action": "ramp"},
    {"x": 63, "y": 63, "action": "ramp", "from_x": 63, "from_y": 80},
    {"x": 80, "y": 80, "action": "ramp", "from_x": 63, "from_y": 80},
])
def test_invalid_ramp_endpoints_are_rejected(point):
    with pytest.raises(ValueError, match="ramp"):
        ramp_circuit(waypoints=[{"x": 63, "y": 80}, point])
