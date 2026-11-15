"""Regression for the recorded Construction Site stair knockback, without input."""
import pytest

from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command


def circuit(**overrides):
    return MinimapCircuit({
        "start_idx": 1, "loop": False, "tolerance_x": 1, "tolerance_y": 1,
        "platforms": [
            {"left": 66.787, "y": 54.413, "right": 89.6, "right_y": 38.038,
             "knots": [[70.6, 51.288], [72.412, 48.6], [80.35, 48.663],
                       [84.287, 43.725], [86.6, 42.1]]},
            {"left": 89, "right": 111, "y": 38},
            {"left": 65, "right": 85, "y": 55}],
        "waypoints": [{"x": 82, "y": 46.59348006096012},
                      {"x": 92, "y": 38, "action": "ramp", "from_x": 82,
                       "from_y": 46.59348006096012, "dwell_s": .6}],
        **overrides,
    })


def enter_knockback(nav):
    assert nav.step((81, 48), 0) == Command.NONE
    assert nav.step((81, 48), .3) == Command("right")
    assert nav.step((82, 47), .45) == Command("right")
    assert nav.step((79, 47), .6) == Command.NONE
    assert nav.state == "ramp_waiting_knockback" and nav.transition_active
    assert nav._ramp_retry_count == 1 and not nav.completed


def settle_knockback(nav):
    for now in (.9, 1.05, 1.2):
        assert nav.step((80, 49), now) == Command.NONE
    assert nav.state == "ramp_realigning"
    assert nav.step((80, 49), 1.3) == Command.NONE
    assert nav.step((80, 49), 1.55).move_x == "right"
    assert nav.step((81, 48), 1.65) == Command.NONE
    assert nav.step((81, 48), 1.95) == Command("right")


def test_recorded_airborne_knockback_releases_then_realigns_on_same_stair():
    nav = circuit()
    enter_knockback(nav)
    original_start = nav._started
    settle_knockback(nav)
    assert nav._started == original_start and nav._ramp_retry_count == 1
    assert not nav.completed
    assert nav.step((87, 42), 2.1) == Command("right")
    assert nav.step((90, 38), 2.3) == Command("right")
    for now in (2.5, 2.8, 3.2):
        assert nav.step((92, 38), now) == Command.NONE
    assert nav.completed


@pytest.mark.parametrize("position", [(78, 49), (79, 50), (79, 55), (79, 44), (65, 55)])
def test_backward_recovery_never_accepts_farther_knockback_fall_or_wrong_floor(position):
    nav = circuit()
    nav.step((81, 48), 0)
    nav.step((81, 48), .3)
    assert nav.step(position, .6) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "left ramp corridor"
    assert not nav.completed and nav._ramp_retry_count == 0


def test_unsupported_wait_has_fixed_deadline_and_never_moves():
    nav = circuit()
    enter_knockback(nav)
    for now in (.8, 1., 1.2, 1.81):
        assert nav.step((79, 47), now) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "ramp knockback did not settle"
    assert not nav.completed


def test_falling_during_wait_blocks_immediately_even_on_another_defined_floor():
    nav = circuit()
    enter_knockback(nav)
    assert nav.step((79, 55), .8) == Command.NONE
    assert nav.state == "blocked" and not nav.completed


def test_only_one_local_retry_is_allowed_per_waypoint():
    nav = circuit()
    enter_knockback(nav)
    settle_knockback(nav)
    assert nav.step((79, 47), 2.1) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "left ramp corridor"
    assert nav._ramp_retry_count == 1


def test_local_retry_never_replenishes_original_ramp_timeout():
    nav = circuit(timeout_s=1.1)
    enter_knockback(nav)
    for now in (.8, .95, 1.09):
        assert nav.step((80, 49), now) == Command.NONE
    assert nav.state == "ramp_realigning"
    assert nav.step((80, 49), 1.11) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "ramp timeout"


@pytest.mark.parametrize("missing", [False, True])
def test_pause_or_missing_position_discards_support_proof_without_resetting_budget(missing):
    nav = circuit()
    enter_knockback(nav)
    nav.step((80, 49), .9)
    if missing:
        assert nav.step(None, .95) == Command.NONE
    else:
        nav.suspend(.95)
    assert nav._ramp_retry_frames == 0 and nav._ramp_retry_count == 1
    assert nav._ramp_retry_since == .6
    for now in (10, 10.1):
        assert nav.step((80, 49), now) == Command.NONE
        assert nav.state == "ramp_waiting_knockback"
    nav.step((80, 49), 10.26)
    assert nav.state == "ramp_realigning" and nav._ramp_retry_count == 1


def test_repeated_same_timestamp_never_proves_stable_support():
    nav = circuit()
    enter_knockback(nav)
    for _ in range(8):
        assert nav.step((80, 49), .9) == Command.NONE
    assert nav._ramp_retry_frames == 1 and nav.state == "ramp_waiting_knockback"


def test_new_waypoint_resets_local_retry_budget_and_wait_state():
    nav = circuit()
    enter_knockback(nav)
    nav._set_waypoint(0, .7)
    assert nav._ramp_retry_count == 0 and nav._ramp_retry_since is None
    assert nav._ramp_retry_frames == 0


def test_exact_live_timestamps_allow_three_fresh_support_frames_before_retry_deadline():
    nav = circuit()
    recorded = [
        (1794671335.9002898, (81, 48)),
        (1794671336.1568544, (81, 48)),
        (1794671336.2553139, (81, 48)),
        (1794671336.5627449, (82, 47)),
        (1794671336.8721275, (79, 47)),
        (1794671337.1742253, (80, 49)),
        (1794671337.478163, (80, 49)),
        (1794671337.7859912, (80, 49)),
    ]
    for index, (timestamp, pos) in enumerate(recorded):
        command = nav.step(pos, timestamp-recorded[0][0])
        if index >= 4:
            assert command == Command.NONE
    assert nav.state == "ramp_realigning" and nav._ramp_retry_count == 1
    assert nav._started == 0 and not nav.completed
