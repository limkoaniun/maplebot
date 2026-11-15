import pytest

from maplebot.nav.circuit import MinimapCircuit, Platform
from maplebot.types import Command


SLOPE = {"left": 67, "right": 84, "y": 88, "right_y": 71}
LOWER = {"left": 42, "right": 111, "y": 88}


def circuit(points, **overrides):
    return MinimapCircuit({"start_idx": 1, "loop": False, "tolerance_y": .75,
                          "platforms": [LOWER, SLOPE], "waypoints": points, **overrides})


def jump_to_slope(**overrides):
    return circuit([{"x": 66, "y": 88}, {"x": 69, "y": 86, "action": "jump",
                    "from_x": 66, "from_y": 88, "dwell_s": 0}], **overrides)


def begin_jump(nav):
    nav.step((66, 88), 0)
    assert nav.step((66, 88), .3).action == "jump"
    nav.step((70, 80), .5)


def test_platform_height_interpolates_without_extending_beyond_bounds():
    slope = Platform(88, 67, 84, 71)
    assert slope.y_at(69) == 86 and slope.y_at(70) == 85
    assert slope.y_at(60) == 88 and slope.y_at(90) == 71
    assert Platform(88, 42, 111).y_at(70) == 88
    nav = jump_to_slope()
    assert nav.platform_at((70, 85)).sloped
    assert nav._containing_platform((70, 85)).sloped
    assert nav._containing_platform((85, 70)) is None


def test_short_jump_accepts_actual_slope_height_at_observed_x_after_three_stable_frames():
    nav = jump_to_slope()
    begin_jump(nav)
    assert nav.step((70, 85), .7) == Command.NONE
    nav.step((70, 85), 1.0)
    assert not nav.completed and nav._jump_landing_frames == 2
    nav.step((70, 85), 1.1)
    assert nav.completed


def test_same_height_change_is_not_accepted_for_a_flat_platform():
    nav = jump_to_slope(platforms=[LOWER, {"left": 67, "right": 84, "y": 86}])
    begin_jump(nav)
    for now in (.7, 1.0, 1.1):
        nav.step((70, 85), now)
    assert not nav.completed and nav._jump_landing_frames == 0


@pytest.mark.parametrize("position", [(70, 83), (70, 88), (85, 70)])
def test_airborne_lower_floor_or_outside_slope_cannot_be_a_landing(position):
    nav = jump_to_slope()
    begin_jump(nav)
    for now in (.7, 1.0, 1.1):
        nav.step(position, now)
    assert not nav.completed and nav._jump_landing_frames == 0


def test_walk_up_slope_keeps_local_floor_and_arrives_at_offset_x():
    nav = circuit([{"x": 69, "y": 86}, {"x": 80, "y": 75, "dwell_s": 0}])
    assert nav.step((69, 86), 0) == Command("right")
    assert nav.step((74, 81), .2) == Command("right")
    # One pixel short horizontally means one pixel lower on this declared slope.
    assert nav.step((79, 76), .4) == Command.NONE and nav.completed


def test_walk_can_start_at_shared_flat_and_slope_endpoint():
    nav = circuit([{"x": 67, "y": 88}, {"x": 80, "y": 75, "dwell_s": 0}])
    assert nav.step((67, 88), 0) == Command("right")


def test_walk_does_not_accept_airborne_position_within_broad_platform_search_tolerance():
    nav = circuit([{"x": 69, "y": 86}, {"x": 80, "y": 75, "dwell_s": 0}])
    assert nav.step((79, 73), 0) == Command.NONE
    assert nav.state == "waiting_floor" and not nav.completed


def test_nearest_walk_selection_can_choose_a_stop_at_different_height_on_same_slope():
    nav = circuit([{"x": 68, "y": 87}, {"x": 80, "y": 75}], start_idx=None)
    assert nav.step((73, 82), 0) == Command("left")
    assert nav.idx == 0


def test_jump_takeoff_on_slope_uses_observed_floor_height():
    nav = circuit([{"x": 69, "y": 86}, {"x": 80, "y": 75, "action": "jump",
                    "from_x": 69, "from_y": 86, "dwell_s": 0}])
    assert nav.step((70, 85), 0) == Command.NONE
    assert nav.step((70, 85), .3).action == "jump"
    assert nav._jump_takeoff_y == 85
    nav.step((71, 84), .5)
    assert not nav._jump_departed  # one actual pixel, rather than two from nominal y86


def test_drop_and_ramp_landings_use_destination_height_at_observed_x():
    drop = circuit([{"x": 69, "y": 70}, {"x": 69, "y": 86, "action": "drop", "dwell_s": 0}],
                   platforms=[{"left": 60, "right": 90, "y": 70}, SLOPE])
    drop.step((69, 70), 0)
    drop.step((69, 70), .3)
    drop.step((70, 80), .5)
    assert drop.step((70, 85), .7) == Command.NONE
    drop.step((70, 85), 1.0)
    assert drop.completed
    ramp = circuit([{"x": 66, "y": 88}, {"x": 69, "y": 86, "action": "ramp",
                    "from_x": 66, "from_y": 88, "dwell_s": 0}])
    ramp.step((66, 88), 0)
    assert ramp.step((66, 88), .3) == Command("right")
    ramp.step((70, 85), .5)
    ramp.step((70, 85), .7)
    ramp.step((70, 85), .9)
    assert ramp.completed


def test_invalid_slope_endpoint_height_is_rejected():
    with pytest.raises(ValueError, match="right_y"):
        jump_to_slope(platforms=[LOWER, dict(SLOPE, right_y=float("nan"))])


MEASURED_SURFACE = {
    "left": 66.79, "y": 87.71, "right": 89.60, "right_y": 71.33,
    "knots": [[70.60, 84.58], [72.41, 81.90], [80.35, 81.96],
              [84.29, 77.02], [86.60, 75.40]],
}


def test_piecewise_surface_preserves_measured_shelf_and_each_endpoint():
    surface = Platform(87.71, 66.79, 89.60, 71.33, MEASURED_SURFACE["knots"])
    assert surface.sloped
    for x, y in MEASURED_SURFACE["knots"]:
        assert surface.y_at(x) == pytest.approx(y)
    assert surface.y_at(66.79) == 87.71 and surface.y_at(89.60) == 71.33
    assert 81.90 < surface.y_at(79) < 81.96  # the shelf is not the endpoints' straight diagonal
    assert surface.y_at(60) == 87.71 and surface.y_at(95) == 71.33
    # Normalized tuples keep the frozen Platform immutable.
    assert isinstance(surface.knots, tuple) and isinstance(surface.knots[0], tuple)


def test_interior_knots_work_when_outer_endpoints_have_equal_height():
    surface = Platform(20, 0, 10, knots=[[5, 15]])
    assert surface.sloped and surface.y_at(2.5) == 17.5
    assert surface.y_at(7.5) == 17.5 and surface.y_at(10) == 20


@pytest.mark.parametrize("knots", [None, [[70]], [[70, float("nan")]], [[70, -1]],
                                    [[70, 84], [70, 83]], [[72, 82], [71, 83]],
                                    [[66.79, 87]], [[89.60, 72]], [[True, 80]]])
def test_invalid_interior_knots_are_rejected(knots):
    with pytest.raises(ValueError, match="knots"):
        Platform(87.71, 66.79, 89.60, 71.33, knots)


def test_ramp_can_realign_from_same_connected_shelf_after_knockback():
    nav = circuit([{"x": 82, "y": 80}, {"x": 89, "y": 72, "action": "ramp",
                    "from_x": 82, "from_y": 80}],
                  platforms=[LOWER, MEASURED_SURFACE], align_tolerance_x=.4, tolerance_x=2)
    assert nav.step((79, 82), 0) == Command.NONE
    assert nav.state == "ramp_aligning"  # recognized shelf, rather than waiting for an absent floor
    assert nav.step((79, 82), .21) == Command("right")
    assert nav.step((80, 82), .3) == Command.NONE
    assert nav.state == "ramp_settling"  # walking tolerance suffices; no exact rope alignment
    assert nav.step((80, 82), .6) == Command("right")
    assert nav.state == "ramping"


def test_jump_landing_on_piecewise_shelf_uses_its_local_height():
    nav = circuit([{"x": 66, "y": 88}, {"x": 78, "y": 81.94, "action": "jump",
                    "from_x": 66, "from_y": 88, "dwell_s": 0}],
                  platforms=[LOWER, MEASURED_SURFACE])
    nav.step((66, 88), 0)
    nav.step((66, 88), .3)
    nav.step((76, 76), .5)
    for now in (.7, .9, 1.1):
        nav.step((79, 82), now)
    assert nav.completed
    assert nav._containing_platform((79, 78)) is None


def test_quantized_support_error_accepts_measured_slope_without_loosening_flat_shelf():
    surface = Platform(**MEASURED_SURFACE)
    assert surface.support_error((82, 79)) < .75
    assert surface.support_error((82, 77)) > .75
    assert surface.support_range(82) == pytest.approx((surface.y_at(82.5), surface.y_at(81.5)))
    shelf = Platform(88, 60, 90, 71, [[72, 82], [80, 82]])
    assert shelf.support_range(79) == (82, 82)
    assert shelf.support_error((79, 81)) == 1
    assert Platform(82, 60, 90).support_error((79, 81)) == 1


def test_support_range_includes_interior_extrema_and_clamps_endpoints():
    surface = Platform(20, 0, 10, knots=[[5, 15]])
    assert surface.support_range(5) == (15, 15.5)
    assert surface.support_error((5, 15)) == 0
    assert surface.support_range(0) == (19.5, 20)


@pytest.mark.parametrize("height,accepted", [(79, True), (77, False)])
def test_ramp_source_uses_local_quantization_support_only(height, accepted):
    nav = circuit([{"x": 82, "y": 80}, {"x": 89, "y": 72, "action": "ramp",
                    "from_x": 82, "from_y": 80}], platforms=[MEASURED_SURFACE])
    assert nav.step((82, height), 0) == Command.NONE
    command = nav.step((82, height), .3)
    assert command == (Command("right") if accepted else Command.NONE)
    assert nav.state == ("ramping" if accepted else "ramp_waiting_floor")


def test_quantized_observations_do_not_loosen_declared_waypoint_validation():
    with pytest.raises(ValueError, match="takeoff and landing"):
        circuit([{"x": 82, "y": 79}, {"x": 89, "y": 72, "action": "ramp",
                 "from_x": 82, "from_y": 79}], platforms=[MEASURED_SURFACE])


@pytest.mark.parametrize("height,accepted", [(79, True), (77, False)])
def test_walk_arrival_uses_same_local_support_error(height, accepted):
    nav = circuit([{"x": 73, "y": 81.9}, {"x": 82, "y": 80, "dwell_s": 0}],
                  platforms=[MEASURED_SURFACE])
    assert nav.step((82, height), 0) == Command.NONE
    assert nav.completed is accepted


@pytest.mark.parametrize("height,accepted", [(79, True), (77, False)])
def test_jump_requires_three_stable_quantized_slope_observations(height, accepted):
    nav = circuit([{"x": 66, "y": 88}, {"x": 82, "y": 80, "action": "jump",
                    "from_x": 66, "from_y": 88, "dwell_s": 0}],
                  platforms=[LOWER, MEASURED_SURFACE])
    nav.step((66, 88), 0)
    nav.step((66, 88), .3)
    nav.step((80, 74), .5)
    for now in (.7, .9):
        nav.step((82, height), now)
    assert not nav.completed
    nav.step((82, height), 1.1)
    assert nav.completed is accepted
