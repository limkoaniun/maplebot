from maplebot.config import load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command


def drop(source=31, target=46, **over):
    return MinimapCircuit({"start_idx": 1, "loop": False, "tolerance_y": .75,
        "timeout_s": 5, "max_retries": 1, "drop_interval_s": .8,
        "waypoints": [{"x": 78, "y": source, "dwell_s": 0},
                      {"x": 78, "y": target, "action": "drop", "dwell_s": 0},
                      {"x": 74, "y": target}], **over})


def test_drop_alignment_can_allow_a_pixel_without_relaxing_rope_alignment():
    nav = drop(align_tolerance_x=.4, drop_align_tolerance_x=1)
    assert nav.align_tolerance_x == .4
    assert nav.step((79, 31), 0) == Command.NONE
    assert nav.step((79, 31), .3).action == "jump"
    outside = drop(align_tolerance_x=.4, drop_align_tolerance_x=1)
    outside.step((80, 31), 0)
    assert outside.state == "aligning"


def test_single_drop_launch_releases_down_after_departure_and_never_repeats_in_flight():
    nav = drop()
    cfg = load_config()
    cfg["keys"]["pickup"] = ""
    backend = RecordingBackend()
    keyboard = KeyboardController(cfg, backend, sleep=lambda _: None)
    for now, y in [(0, 31), (.3, 31), (.35, 31), (.5, 35), (.8, 40), (1.1, 44), (1.2, 46)]:
        cmd = nav.step((78, y), now)
        keyboard.set_command(cmd)
        keyboard.apply_once(now)
        if y > 32:
            assert cmd == Command.NONE
            assert cfg["keys"]["down"] not in backend.held
    assert backend.events.count(("down", cfg["keys"]["jump"])) == 1
    assert nav.idx == 1 and nav.transition_active
    nav.step((78, 46), 1.5)
    assert nav.idx == 2


def test_passing_destination_in_flight_does_not_advance_and_overshoot_blocks():
    nav = drop()
    for now, y in [(0, 31), (.3, 31), (.5, 36), (.7, 43), (.8, 46)]:
        nav.step((78, y), now)
    assert nav.idx == 1 and not nav.completed
    assert nav.step((78, 47), .9) == Command.NONE
    assert nav.state == "blocked" and nav.reason == "passed destination floor"
    assert nav.step((78, 61), 1.1) == Command.NONE and nav.idx == 1


def test_intermediate_ledge_requires_stable_observation_before_second_drop():
    nav = drop(16, 31)
    nav.step((78, 16), 0)
    assert nav.step((78, 16), .3).action == "jump"
    assert nav.step((78, 24), .5) == Command.NONE
    assert nav.step((78, 27), .6) == Command.NONE
    assert nav.step((78, 27), .9) == Command.NONE
    assert nav.step((78, 27), 1.1) == Command.NONE  # intermediate floor now confirmed
    assert nav.step((78, 27), 1.2) == Command.NONE
    assert nav.step((78, 27), 1.5) == Command("none", "down", "jump")
    assert nav.step((78, 29), 1.7) == Command.NONE
    assert nav.step((78, 31), 1.9) == Command.NONE and nav.idx == 1
    nav.step((78, 31), 2.2)
    assert nav.idx == 2


def test_failed_drop_retries_only_on_stable_source_floor_and_is_bounded():
    nav = drop()
    nav.step((78, 31), 0)
    assert nav.step((78, 31), .3).action == "jump"
    assert nav.step((78, 31), .6) == Command.NONE
    assert nav.step((78, 31), 1.2) == Command.NONE
    nav.step((78, 31), 1.3)
    assert nav.step((78, 31), 1.6).action == "jump"
    assert nav.step((78, 31), 2.5) == Command.NONE
    assert nav.state == "blocked" and nav.retries == 1


def test_pause_never_reissues_launch_and_requires_new_stable_landing():
    nav = drop()
    nav.step((78, 31), 0)
    nav.step((78, 31), .3)
    nav.suspend(.31)
    assert nav.step((78, 31), 10) == Command.NONE
    nav.step((78, 40), 10.2)
    nav.step((78, 46), 10.4)
    nav.suspend(10.5)
    nav.step((78, 46), 20)
    nav.step((78, 46), 20.2)
    assert nav.idx == 1
    nav.step((78, 46), 20.3)
    assert nav.idx == 2


def test_fall_timeout_blocks_without_blind_retry():
    nav = drop(timeout_s=1)
    for now, y in [(0, 31), (.3, 31), (.5, 35), (.8, 38), (1.1, 40)]:
        cmd = nav.step((78, y), now)
    assert cmd == Command.NONE and nav.state == "blocked" and nav.reason == "drop timeout"
