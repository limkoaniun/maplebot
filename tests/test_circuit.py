import pytest

from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command


def circuit(points, **options):
    return MinimapCircuit({"waypoints": points, **options})


def test_selects_nearest_walk_on_current_floor_and_loops():
    nav = circuit([{"x": 10, "y": 76, "dwell_s": 0},
                   {"x": 30, "y": 61, "dwell_s": 0},
                   {"x": 70, "y": 61, "dwell_s": 0}])
    assert nav.step((28, 61), 0) == Command.NONE and nav.idx == 2
    assert nav.step((30, 61), .1) == Command("right")
    assert nav.step((70, 61), .2) == Command.NONE and nav.idx == 0


def test_unknown_floor_waits_instead_of_approaching_wrong_climb():
    nav = circuit([{"x": 40, "y": 61, "action": "climb"}, {"x": 50, "y": 61}])
    assert nav.step((40, 76), 0) == Command.NONE
    assert nav.state == "waiting_floor"


@pytest.mark.parametrize("position", [(10, 40), (20, 40)])
def test_walk_does_not_cross_gap_to_different_platform_at_same_height(position):
    nav = circuit([{"x": 22, "y": 40, "dwell_s": 0}, {"x": 40, "y": 40}],
                  start_idx=0, platforms=[{"left": 1, "right": 20, "y": 40},
                                          {"left": 22, "right": 50, "y": 40}])
    assert nav.step(position, 0) == Command.NONE
    assert nav.state == "waiting_floor" and nav.idx == 0


def test_climb_aligns_jumps_up_and_finishes_on_floor_before_advancing():
    nav = circuit([{"x": 40, "y": 61, "action": "climb", "dwell_s": .3},
                   {"x": 80, "y": 61}], start_idx=0)
    assert nav.step((30, 76), 0) == Command("right")
    assert nav.step((40, 76), .1) == Command("none", "up")
    assert nav.step((40, 76), .4) == Command("none", "up", "jump")
    assert nav.step((40, 75), .5, on_ladder=True) == Command("none", "up")
    assert nav.step((40, 63), .6, on_ladder=True) == Command("none", "up")
    assert nav.idx == 0  # two minimap pixels early is still below the platform
    assert nav.step((40, 61), .7, on_ladder=True) == Command("none", "up")
    assert nav.idx == 0 and nav.transition_active
    nav.step((40, 61), 1.1, on_ladder=True)
    assert nav.idx == 1
    assert nav.step((40, 61), 1.2) == Command("right")


def test_drop_aligns_then_releases_controls_during_fall_and_waits_for_landing():
    nav = circuit([{"x": 78, "y": 76, "action": "drop", "dwell_s": 0},
                   {"x": 4, "y": 76}], start_idx=0, drop_interval_s=.5)
    assert nav.step((60, 61), 0) == Command("right")
    assert nav.step((78, 61), .1) == Command.NONE
    assert nav.step((78, 61), .4) == Command("none", "down", "jump")
    assert nav.step((78, 65), .5) == Command.NONE
    assert nav.step((78, 70), 1.0) == Command.NONE
    nav.step((78, 76), 1.1)
    assert nav.idx == 0
    nav.step((78, 76), 1.4)
    assert nav.idx == 1


def test_climb_retries_entry_despite_stale_ladder_flag_but_not_during_ascent():
    nav = circuit([{"x": 40, "y": 61, "action": "climb"}, {"x": 80, "y": 61}],
                  start_idx=0, drop_interval_s=.7)
    assert nav.step((40, 76), 0, on_ladder=True).action == "none"
    assert nav.step((40, 76), .3, on_ladder=True).action == "jump"
    assert nav.step((40, 76), 1.1, on_ladder=True).action == "jump"
    assert nav.step((40, 75), 1.8, on_ladder=False).action == "none"
    assert nav.step((40, 74), 2.3, on_ladder=False).action == "none"


def test_precision_approach_releases_between_pulses_and_restarts_settle_after_drift():
    nav = circuit([{"x": 40, "y": 61, "action": "climb"}, {"x": 80, "y": 61}],
                  start_idx=0, align_tolerance_x=.4)
    assert nav.step((30, 76), 0) == Command("right")
    assert nav.step((36, 76), .1) == Command.NONE  # entering precision range releases first
    assert nav.step((37, 76), .25) == Command.NONE
    assert nav.step((37, 76), .31) == Command("right")
    assert nav.step((39, 76), .42) == Command.NONE
    assert nav.step((40, 76), .63) == Command("none", "up")
    assert nav.step((41, 76), .8) == Command.NONE  # drift cancels stable alignment
    assert nav.step((40, 76), .9) == Command("none", "up")
    assert nav.step((40, 76), 1.1) == Command("none", "up")
    assert nav.step((40, 76), 1.2) == Command("none", "up", "jump")
    assert nav.step((41, 75), 1.3) == Command("none", "up")  # no horizontal rope-jitter correction
    assert nav.step((41, 70), 1.7) == Command("none", "up")


def test_precision_direction_change_and_pause_require_fresh_release_and_settle():
    nav = circuit([{"x": 40, "y": 61, "action": "climb"}, {"x": 80, "y": 61}],
                  start_idx=0, align_tolerance_x=.4)
    nav.step((43, 76), 0)
    assert nav.step((43, 76), .21) == Command("left")
    assert nav.step((39, 76), .22) == Command.NONE  # no immediate reversal while drifting
    assert nav.step((40, 76), .3).action == "none"
    nav.suspend(.4)
    assert nav.step((40, 76), 10).action == "none"
    assert nav.step((40, 76), 10.2).action == "none"
    assert nav.step((40, 76), 10.3).action == "jump"


def test_optional_precision_walk_releases_then_pulses_near_target_and_keeps_arrival():
    nav = circuit([{"x": 90, "y": 38, "dwell_s": .3}, {"x": 100, "y": 38}],
                  start_idx=0, precision_walk=True, approach_radius_x=6, tolerance_x=1,
                  platforms=[{"left": 89, "right": 110, "y": 38}])
    assert nav.step((98, 38), 0) == Command("left")
    assert nav.step((96, 38), .1) == Command.NONE
    assert nav.step((95, 38), .25) == Command.NONE
    assert nav.step((95, 38), .31) == Command("left")
    assert nav.step((93, 38), .42) == Command.NONE
    assert nav.step((91, 38), .6) == Command.NONE
    assert nav.state == "dwelling" and nav.idx == 0
    assert nav.step((91, 38), .95) == Command.NONE
    assert nav.idx == 1


def test_precision_walk_is_optional_and_cannot_bypass_floor_check():
    points = [{"x": 90, "y": 38}, {"x": 100, "y": 38}]
    ordinary = circuit(points, start_idx=0, tolerance_x=1)
    assert ordinary.step((94, 38), 0) == Command("left")
    precise = circuit(points, start_idx=0, precision_walk=True, tolerance_x=1,
                      platforms=[{"left": 89, "right": 110, "y": 38}])
    assert precise.step((94, 45), 0) == Command.NONE
    assert precise.state == "waiting_floor"


def test_precision_pulse_commands_carry_remaining_time_without_extending_on_fresh_frames():
    nav = circuit([{"x": 90, "y": 38}, {"x": 100, "y": 38}], start_idx=0,
                  precision_walk=True, approach_radius_x=6, tolerance_x=1)
    assert nav.step((98, 38), 0).valid_for_s is None  # ordinary far approach retains its watchdog
    assert nav.step((95, 38), .1) == Command.NONE
    pulse = nav.step((95, 38), .31)
    assert pulse.move_x == "left" and pulse.valid_for_s == pytest.approx(.1)
    renewed = nav.step((94, 38), .35)
    assert renewed.move_x == "left" and renewed.valid_for_s == pytest.approx(.06)
    assert nav.step((93, 38), .42) == Command.NONE


def test_nav_pulse_releases_keyboard_even_without_another_vision_decision():
    from maplebot.config import load_config
    from maplebot.control.backend import RecordingBackend
    from maplebot.control.keyboard import KeyboardController

    nav = circuit([{"x": 90, "y": 38}, {"x": 100, "y": 38}], start_idx=0,
                  precision_walk=True, approach_radius_x=6, tolerance_x=1)
    cfg = load_config()
    cfg["keys"]["pickup"] = ""
    cfg["buffs"] = {"keys": []}
    clock = [0.0]
    backend = RecordingBackend()
    keyboard = KeyboardController(cfg, backend, clock=lambda: clock[0], sleep=lambda _: None)
    nav.step((95, 38), 0)
    clock[0] = .21
    keyboard.set_command(nav.step((95, 38), .21), source_time=.21, navigation_only=True)
    keyboard.set_enabled(True)
    keyboard.apply_once(.21)
    assert backend.held == {"left"}
    # Simulate slow perception: only the independent keyboard gets another tick.
    clock[0] = .32
    keyboard.apply_once(.32)
    assert not backend.held and backend.events == [("down", "left"), ("up", "left")]


def test_starting_mid_rope_holds_up_instead_of_jumping_off():
    nav = circuit([{"x": 40, "y": 76}, {"x": 40, "y": 61, "action": "climb"}], start_idx=1)
    assert nav.step((40, 63), 0) == Command("none", "up")
    assert nav.step((40, 63), .3) == Command("none", "up")
    assert nav.step((40, 63), 1.1) == Command("none", "up")


def test_timeout_retries_are_bounded_and_do_not_skip_unreached_point():
    nav = circuit([{"x": 40, "y": 61, "action": "climb"}, {"x": 80, "y": 61}],
                  start_idx=0, timeout_s=.5, recovery_s=.2, max_retries=1)
    for i in range(20):
        nav.step((40, 76), i / 10)
    assert nav.state == "blocked" and nav.idx == 0 and nav.retries == 1
    assert nav.step((40, 76), 2.0) == Command.NONE


def test_pause_missing_position_and_long_gaps_do_not_exhaust_timeout():
    nav = circuit([{"x": 80, "y": 76}, {"x": 4, "y": 76}], start_idx=0, timeout_s=.5)
    nav.step((20, 76), 0)
    nav.suspend(.1)
    assert nav.step((20, 76), 100) == Command("right")
    assert nav.step(None, 100.1) == Command.NONE
    assert nav.step((20, 76), 200) == Command("right")
    assert nav.retries == 0


def test_invalid_circuit_fails_instead_of_silently_sweeping():
    with pytest.raises(ValueError):
        circuit([{"x": 10, "y": 76}])


def _floor_circuit(start_idx=0):
    return circuit([{"x": 20, "y": 61}, {"x": 84, "y": 76}, {"x": 4, "y": 76},
                    {"x": 26, "y": 46}], start_idx=start_idx,
                   platforms=[{"y": 76, "left": 4, "right": 84},
                              {"y": 61, "left": 20, "right": 97},
                              {"y": 46, "left": 26, "right": 91}])


def test_fall_rejoins_nearest_walk_on_stable_lower_floor_and_leaves_ground_slope():
    nav = _floor_circuit()
    assert nav.step((96, 80), 0) == Command.NONE
    assert nav.step((96, 80), .2) == Command.NONE and nav.idx == 0
    assert nav.step((96, 80), .5) == Command("left") and nav.idx == 1
    assert nav.step((94, 80), .6) == Command("left")  # slope still belongs to ground floor
    assert nav.step((87, 77), .7) == Command("left")
    assert nav.step((84, 76), .8) == Command.NONE


def test_floor_rejoin_requires_stable_height_and_never_skips_upward_during_jump():
    nav = _floor_circuit()
    for now, y in ((0, 68), (.1, 73), (.3, 76), (.5, 79)):
        assert nav.step((90, y), now) == Command.NONE and nav.idx == 0
    assert nav.step((90, 79), .7) == Command.NONE
    assert nav.step((90, 79), 1.0) == Command("left") and nav.idx == 1
    for now in (1.1, 1.5, 2.0):
        assert nav.step((50, 61), now) == Command.NONE and nav.idx == 1
    assert nav.step((50, 76), 2.1) == Command("right") and nav.idx == 1


def test_pause_and_missing_position_cancel_floor_stability_confirmation():
    nav = _floor_circuit()
    nav.step((90, 76), 0)
    nav.suspend(.3)
    assert nav.step((90, 76), 10) == Command.NONE and nav.idx == 0
    nav.step(None, 10.2)
    assert nav.step((90, 76), 10.4) == Command.NONE and nav.idx == 0
    assert nav.step((90, 76), 10.9) == Command("left") and nav.idx == 1


def test_unknown_floor_still_times_out_without_blind_horizontal_recovery():
    nav = circuit([{"x": 80, "y": 61}, {"x": 4, "y": 76}], start_idx=0,
                  platforms=[{"y": 61, "left": 20, "right": 97}, {"y": 76, "left": 4, "right": 84}],
                  timeout_s=.4, recovery_s=.2, max_retries=1)
    for i in range(20):
        assert nav.step((50, 68), i / 10) == Command.NONE
    assert nav.state == "blocked" and nav.retries == 1
