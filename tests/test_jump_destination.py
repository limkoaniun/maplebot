"""Landing proof follows the reached floor, without correcting toward its edge."""
import pytest

from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command


def gap(**overrides):
    return MinimapCircuit({
        "start_idx": 1, "loop": False, "max_retries": 0,
        "tolerance_x": 1, "tolerance_y": .75,
        "jump_landing_overshoot_x": 3,
        "platforms": [{"left": 89, "right": 111, "y": 38},
                      {"left": 42, "right": 88.537, "y": 38}],
        "waypoints": [{"x": 91, "y": 38},
                      {"x": 85, "y": 38, "action": "jump", "from_x": 91,
                       "from_y": 38, "dwell_s": .5}],
        **overrides,
    })


def descend_to_destination(nav):
    assert nav.step((91, 38), 0) == Command.NONE
    assert nav.step((91, 38), .3) == Command("left", "none", "jump")
    nav.step((90, 36), .63)
    nav.step((87, 34), .96)
    assert nav.step((85, 38), 1.32) == Command.NONE
    assert nav.state == "jump_landing" and not nav.completed


def test_live_upper_gap_drift_confirms_floor_without_correcting_back_toward_edge():
    nav = gap()
    descend_to_destination(nav)
    # The recorded run drifted beyond the original three-pixel target margin.
    assert nav.step((83, 38), 1.58) == Command.NONE
    assert nav.step((81, 38), 1.70) == Command.NONE
    assert not nav.completed  # three observations still need the landing dwell
    assert nav.step((80, 38), 1.95) == Command.NONE
    assert nav.completed


def test_reached_floor_still_needs_three_fresh_landing_observations():
    nav = gap()
    descend_to_destination(nav)
    assert nav.step((81, 38), 1.9) == Command.NONE
    assert not nav.completed  # elapsed dwell alone is insufficient
    nav.step((80, 38), 2.0)
    assert nav.completed


@pytest.mark.parametrize("lost", [(81, 36), (81, 40), (41, 38), (89, 38)])
def test_unsupported_or_outside_destination_position_discards_arrival(lost):
    nav = gap()
    descend_to_destination(nav)
    nav.step(lost, 1.5)
    assert not nav._jump_destination_seen and not nav.completed
    nav.step((81, 38), 1.8)  # outside the original arrival margin cannot reacquire
    assert not nav._jump_destination_seen
    nav.step((81, 38), 2.4)
    assert nav.state == "blocked" and not nav.completed


@pytest.mark.parametrize("pause", [False, True])
def test_pause_or_missing_capture_clears_extended_destination_allowance(pause):
    nav = gap()
    descend_to_destination(nav)
    if pause:
        nav.suspend(1.4)
    else:
        nav.step(None, 1.4)
    assert not nav._jump_destination_seen
    nav.step((81, 38), 1.7)
    assert not nav.completed and not nav._jump_destination_seen


def test_destination_support_without_departure_never_latches_arrival():
    nav = gap()
    nav.step((91, 38), 0)
    nav.step((91, 38), .3)
    for now, x in [(.5, 85), (.8, 83), (1.1, 81)]:
        nav.step((x, 38), now)
        assert not nav._jump_destination_seen and not nav.completed


def higher_ledge():
    nav = gap(platforms=[{"left": 0, "right": 15, "y": 50},
                        {"left": 25, "right": 40, "y": 46}],
              waypoints=[{"x": 10, "y": 50},
                         {"x": 30, "y": 46, "action": "jump", "from_x": 10,
                          "from_y": 50, "dwell_s": 0}])
    nav.step((10, 50), 0)
    nav.step((10, 50), .3)
    return nav


def test_brief_higher_ledge_apex_and_horizontal_drift_do_not_complete():
    nav = higher_ledge()
    for now, x in [(.5, 30), (.6, 35), (.7, 36)]:
        assert nav.step((x, 46), now) == Command.NONE
        assert not nav.completed
    nav.step((36, 44), .8)
    assert not nav._jump_destination_seen and not nav.completed


def test_higher_ledge_drift_retains_long_dwell_and_minimum_flight_requirement():
    nav = higher_ledge()
    for now, x in [(.5, 30), (.8, 35), (1.1, 36)]:
        assert nav.step((x, 46), now) == Command.NONE
        assert not nav.completed
    assert nav.step((37, 46), 1.2) == Command.NONE
    assert nav.completed
