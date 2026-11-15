import pytest

from maplebot.nav.loot import LootCollector
from maplebot.types import Command
from tests.test_keyboard import _leased_keyboard


def visit(goal=85.625, start=84, bounds=(42, 88.6), **overrides):
    loot = LootCollector({"enabled": True, "disappear_s": 0, "tolerance_x": 1,
                          "max_approach_s": 4, "settle_s": 1.2, **overrides}, 16)
    loot.observe((start, 71), (700, 500), (700+(goal-start)*16, 500), 0,
                 platform_bounds=bounds)
    return loot


def test_recorded_84_to_87_overshoot_brakes_before_reversing_and_stops_outside_original_floor():
    loot = visit()
    # Original trace held RIGHT for these first two observations, reversed at87,
    # then continued after89,72 had crossed the original floor's88.6 edge.
    assert loot.step((84, 71), 0, platform_bounds=(42, 88.6)) == Command.NONE
    assert loot.step((84, 71), .31, platform_bounds=(42, 88.6)) == Command.NONE
    assert loot.step((87, 71), .46, platform_bounds=(42, 88.6)) == Command.NONE
    assert loot.step((89, 72), .72, platform_bounds=(66.79, 89.6)) is None
    assert loot.goal is None and loot._pulse_until is None


def test_precision_pulse_expires_in_keyboard_without_next_vision_frame():
    loot = visit()
    assert loot.step((84, 71), 0) == Command.NONE
    pulse = loot.step((84, 71), .46)
    assert pulse == Command("right") and pulse.valid_for_s == pytest.approx(.04)
    kb, backend, clock = _leased_keyboard()
    kb.set_command(pulse, source_time=clock[0])
    kb.apply_once(0)
    assert backend.held == {"right"}
    clock[0] += .041
    kb.apply_once(.041)
    assert not backend.held and kb.cmd == Command.NONE


def test_repeated_observation_does_not_extend_a_pulse_or_skip_release():
    loot = visit()
    loot.step((84, 71), 0)
    assert loot.step((84, 71), .46).valid_for_s == pytest.approx(.04)
    assert loot.step((84, 71), .48).valid_for_s == pytest.approx(.02)
    assert loot.step((84, 71), .51) == Command.NONE
    assert loot.step((84, 71), .9) == Command.NONE
    assert loot.step((84, 71), .97).valid_for_s == pytest.approx(.04)


def test_direction_reversal_brakes_even_when_both_positions_are_far_from_goal():
    loot = visit(goal=50, start=30, bounds=(0, 100))
    assert loot.step((30, 71), 0) == Command("right")
    assert loot.step((60, 71), .3) == Command.NONE
    assert loot.step((60, 71), .6) == Command.NONE
    assert loot.step((60, 71), .76) == Command("left")


@pytest.mark.parametrize("reset", ["cancel", "observe", "visual_gap"])
def test_reset_paths_cannot_restart_an_old_pulse(reset):
    loot = visit()
    loot.step((84, 71), 0)
    assert loot.step((84, 71), .46) == Command("right")
    if reset == "visual_gap":
        assert loot.visual_lock(.47, False, ("floor", 0))
        assert not loot.visual_lock(.5, True, ("floor", 0))
    else:
        if reset == "cancel":
            loot.cancel()
        loot.observe((84, 71), (700, 500), (726, 500), .5, platform_bounds=(42, 88.6))
    assert loot._pulse_until is None
    assert loot.step((84, 71), .51) == Command.NONE


def test_precision_attempt_cannot_extend_total_approach_budget():
    loot = visit(max_approach_s=1)
    loot.step((84, 71), 0)
    for now in (.3, .6, .9):
        loot.step((84, 71), now)
    assert loot.step((84, 71), 1.01) is None and loot.goal is None


def test_small_pulses_reach_goal_with_inertia_without_crossing_platform_edge():
    # A deliberately simple damped-motion model: 10 minimap pixels/s when held,
    # 0.18s momentum decay, 0.30s vision cadence and an independent10ms key tick.
    loot = visit()
    pos, velocity, held_until, direction = 84., 0., 0., 0
    maximum = pos
    for tick in range(401):
        now = tick*.01
        if tick % 30 == 0:
            command = loot.step((round(pos), 71), now, platform_bounds=(42, 88.6))
            if command is None:
                break
            direction = {"left": -1, "right": 1}.get(command.move_x, 0)
            held_until = now+(command.valid_for_s or .3) if direction else now
        target_speed = 10*direction if now < held_until else 0
        velocity += (target_speed-velocity)*(.01/.18)
        pos += velocity*.01
        maximum = max(maximum, pos)
    assert maximum < 87 and abs(pos-85.625) <= 1.5
    assert loot.goal is None  # finite completed visit, not unlimited corrective motion
