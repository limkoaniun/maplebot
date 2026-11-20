"""Loot goals follow the platform surface, survive a newer target, and close distance in bounded pulses."""
import pytest

from maplebot.nav.circuit import Platform
from maplebot.nav.loot import LootCollector
from maplebot.types import Command

STAIR1 = Platform(87.71, 66.79, 89.6, 71.33,
                  ((70.6, 84.58), (72.41, 81.9), (80.35, 81.96), (84.29, 77.02), (86.6, 75.4)))
FLOOR = Platform(71, 42, 88.6)
BOUNDS = (FLOOR.left, FLOOR.right)


def collector(**over):
    return LootCollector({"enabled": True, "disappear_s": .25, "tolerance_x": 1, "max_approach_s": 4,
                          "settle_s": 1.2, **over}, 16)


def track(loot, start, goal_x, now, platform=FLOOR, visible=None, y=71):
    bounds = (platform.left, platform.right)
    loot.observe((start, y), (700, 500), (700 + (goal_x - start) * 16, 500), now,
                 platform_bounds=bounds, platform=platform, visible_x=visible)


def test_goal_height_follows_the_stair_and_the_visit_survives_walking_along_it():
    loot = collector(disappear_s=0)
    track(loot, 82, 85.75, 0, STAIR1, y=80)
    assert loot.goal == pytest.approx((85.75, STAIR1.y_at(85.75)))  # about 76.0, not the player's 80
    bounds = (STAIR1.left, STAIR1.right)
    # Recorded stair slope: the player climbs four minimap pixels in height on the way.
    for now, pos in ((0, (82, 80)), (.3, (83, 79)), (.6, (84, 77)), (.9, (85, 76))):
        assert loot.step(pos, now, platform_bounds=bounds) is not None, pos
    assert loot.state == "dwelling" and loot.goal is not None


def test_without_a_platform_the_same_stair_visit_cancels_after_two_pixels_of_rise():
    loot = collector(disappear_s=0)
    loot.observe((82, 80), (700, 500), (760, 500), 0, platform_bounds=(STAIR1.left, STAIR1.right))
    assert loot.step((83, 79), 0) is not None
    assert loot.step((84, 77), .3) is None and loot.goal is None


def test_leaving_the_stair_surface_still_cancels_a_surface_goal():
    loot = collector(disappear_s=0)
    track(loot, 82, 85.75, 0, STAIR1, y=80)
    assert loot.step((83, 75), 0, platform_bounds=(STAIR1.left, STAIR1.right)) is None  # knocked 4 px up
    assert loot.goal is None and loot.pending_count == 0


def _run(loot, pos, start, until, has=False, step=.15, bounds=BOUNDS, speed=9.5):
    """Kinematic walk: a leased pulse moves for its lease, an open hold until the next frame."""
    now, visited, states = start, [], []
    while now < until - 1e-9:
        cmd = loot.step((round(pos), 71), now, has, platform_bounds=bounds)
        states.append(loot.state)
        if cmd is not None and cmd.move_x in ("left", "right"):
            held = min(step, cmd.valid_for_s or step)
            pos += (1 if cmd.move_x == "right" else -1) * speed * held
        visited.append(pos)
        now += step
    return pos, visited, states


def test_begun_visit_displaced_by_a_new_target_is_resumed_after_that_fight():
    loot = collector(max_pending=2, walk_speed_x=10)
    track(loot, 75, 71.25, 0)
    assert loot.step((75, 71), .1) == Command.NONE and loot.step((75, 71), .4) is not None
    assert loot.state == "approaching"
    for now in (.55, .7, .85):  # monster B appears on the right and is fought
        track(loot, 75, 78.75, now, visible=[78.75])
    assert loot.goal == (78.75, 71.0) and loot.pending_count == 1
    pos, visited, states = _run(loot, 75., 1.0, 9.0)
    assert min(abs(x - 78.75) for x in visited) <= 1 and min(abs(x - 71.25) for x in visited) <= 1
    assert states.count("dwelling") >= 2 * 7  # both sites got a full 1.2 s dwell
    assert loot.goal is None and loot.pending_count == 0 and loot.state == "idle"


def test_default_keeps_a_single_goal_and_forgets_the_displaced_visit():
    loot = collector()
    track(loot, 75, 71.25, 0)
    loot.step((75, 71), .1), loot.step((75, 71), .4)
    track(loot, 75, 78.75, .55, visible=[78.75])
    assert loot.pending_count == 0
    loot.yield_to_target(.6)
    assert loot.goal is None and loot.pending_count == 0
    assert loot.walk_speed_x == 0 and loot.max_pending == 0  # both stay opt-in


def test_pursuit_keeps_a_begun_visit_but_never_remembers_a_merely_tracked_monster():
    tracked = collector(max_pending=2, walk_speed_x=10)
    track(tracked, 75, 71.25, 0)
    tracked.yield_to_target(.1)  # the monster only walked out of range
    assert tracked.goal is None and tracked.pending_count == 0
    begun = collector(max_pending=2, walk_speed_x=10)
    track(begun, 75, 71.25, 0)
    begun.step((75, 71), .1), begun.step((75, 71), .4)
    begun.yield_to_target(.5)
    assert begun.goal is None and begun.pending_count == 1


def test_displacing_a_resumed_site_never_renews_its_deadline():
    loot = collector(max_pending=1, pending_s=3, walk_speed_x=10)
    track(loot, 75, 60, 0, visible=[60, 79])
    track(loot, 75, 79, .15, visible=[79])  # site 60 remembered until 3.15
    now, starts = .3, 0
    while now < 8:
        loot.yield_to_target(now)  # a living monster wanders into pursuit range again
        loot.step((75, 71), now + .15, platform_bounds=BOUNDS)
        starts += loot.goal is not None and loot.goal[0] == 60
        now += .9
    assert 1 <= starts <= 4 and loot.pending_count == 0 and loot.goal is None


def test_nearest_remembered_site_is_visited_first_and_a_timeout_keeps_the_other():
    loot = collector(max_pending=2, walk_speed_x=10, max_approach_s=1)
    for index, x in enumerate((60, 70, 80)):
        track(loot, 75, x, index * .15, visible=[x])
    loot.yield_to_target(.5)
    assert loot.pending_count == 2
    loot.step((75, 71), .6, platform_bounds=BOUNDS)
    assert loot.goal[0] == 70 and loot.pending_count == 1
    for now in (.9, 1.2, 1.5, 1.8):  # never arrives: this visit runs out its own budget
        loot.step((75, 71), now, platform_bounds=BOUNDS)
    assert loot.pending_count + (loot.goal is not None) == 1
    loot.step((75, 71), 2.1, platform_bounds=BOUNDS)
    assert loot.goal is not None and loot.goal[0] == 60


@pytest.mark.parametrize("gap", [.5, .7, .75])
def test_walk_back_from_beyond_the_brake_zone_is_leased_and_stays_on_the_floor(gap):
    # Remembered sites make 10+ px walk-backs routine. Held speed is 9.2-9.8
    # minimap px/s (log/three-circuits-v3.jsonl), and one late frame must not
    # carry an open hold past the edge at 88.6.
    for late in range(1, 12):
        loot = collector(disappear_s=0, max_pending=1, walk_speed_x=10)
        track(loot, 72, 87.9, 0)
        pos, now, frame, furthest = 72., 0., 0, 72.
        while now < 6 and loot.goal is not None:
            cmd = loot.step((round(pos), 71), now, platform_bounds=BOUNDS)
            step = gap if frame == late else .15
            if cmd is not None and cmd.move_x == "right":
                assert cmd.valid_for_s is not None and cmd.valid_for_s <= .5
                pos += 9.8 * min(step, cmd.valid_for_s, .75)
            furthest, now, frame = max(furthest, pos), now + step, frame + 1
        assert furthest < 88.6, (late, furthest)


def test_running_pulse_is_resized_by_fresh_frames_and_never_resumed_after_a_frozen_clock():
    loot = collector(disappear_s=0, walk_speed_x=10)
    track(loot, 84, 88, 0)
    loot.step((84, 71), 0, platform_bounds=BOUNDS)
    first = loot.step((84, 71), .46, platform_bounds=BOUNDS)
    assert first.valid_for_s == pytest.approx(.7 * 3.6 / 10)
    closer = loot.step((86, 71), .5, platform_bounds=BOUNDS)  # moved faster than planned: shorten
    assert closer.valid_for_s == pytest.approx(.7 * 1.6 / 10)
    assert loot.step((86, 71), .52, platform_bounds=BOUNDS).valid_for_s == pytest.approx(.7 * 1.6 / 10 - .02)
    again = collector(disappear_s=0, walk_speed_x=10)
    track(again, 84, 88, 0)
    again.step((84, 71), 0, platform_bounds=BOUNDS)
    assert again.step((84, 71), .46, platform_bounds=BOUNDS) == Command("right")
    assert again.step((86, 71), 2.0, platform_bounds=BOUNDS) == Command.NONE  # 1.5 s stall: brake first


def test_pulse_cap_binds_when_the_configured_speed_is_low():
    loot = collector(disappear_s=0, walk_speed_x=2)
    track(loot, 75, 79.9, 0)
    loot.step((75, 71), 0, platform_bounds=BOUNDS)
    assert loot.step((75, 71), .46, platform_bounds=BOUNDS).valid_for_s == pytest.approx(.5)


def test_target_that_vanished_while_another_monster_is_fought_keeps_its_site():
    loot = collector(max_pending=2, walk_speed_x=10)
    track(loot, 75, 72.5, 0, visible=[72.5, 79])
    track(loot, 75, 79, .15, visible=[79])  # first monster gone; the fight moves to the second
    assert loot.goal == (79, 71.0) and loot.pending_count == 1


def test_target_switch_while_the_old_monster_is_still_visible_remembers_nothing():
    loot = collector(max_pending=2, walk_speed_x=10)
    track(loot, 75, 72.5, 0, visible=[72.5, 79])
    track(loot, 75, 79, .15, visible=[72.9, 79])  # old target merely lost the tie-break
    assert loot.pending_count == 0
    track(loot, 75, 79.4, .3, visible=[79.4])  # ordinary movement of the same monster
    assert loot.pending_count == 0


def test_monster_reappearing_on_a_remembered_site_removes_it():
    loot = collector(max_pending=2, walk_speed_x=10)
    track(loot, 75, 72.5, 0, visible=[72.5, 79])
    track(loot, 75, 79, .15, visible=[79])
    track(loot, 75, 79, .3, visible=[72.2, 79])  # the detection only flickered
    assert loot.pending_count == 0


@pytest.mark.parametrize("reset", ["cancel", "suspend", "other_floor", "expired", "left_floor"])
def test_remembered_sites_never_outlive_their_floor_or_deadline(reset):
    loot = collector(max_pending=2, pending_s=3, walk_speed_x=10)
    track(loot, 75, 72.5, 0, visible=[72.5, 79])
    track(loot, 75, 79, .15, visible=[79])
    assert loot.pending_count == 1
    if reset == "cancel":
        loot.cancel()
    elif reset == "suspend":
        loot.suspend(.2)
    elif reset == "left_floor":
        assert loot.step((75, 66), .3, platform_bounds=BOUNDS) is None
    else:
        loot.yield_to_target(.2)  # the living second target is pursued, nothing begun
        bounds = (66.79, 89.6) if reset == "other_floor" else BOUNDS
        now = .5
        for _ in range(30 if reset == "expired" else 1):
            assert loot.step((75, 71), now, has_detections=reset == "expired", platform_bounds=bounds) is None \
                or reset == "expired"
            now += .15
        assert loot.step((75, 71), now, platform_bounds=bounds) is None
    assert loot.goal is None and loot.pending_count == 0


def test_remembered_sites_are_bounded_and_deduplicated():
    loot = collector(max_pending=2, walk_speed_x=10)
    for index, x in enumerate((60, 60.5, 64, 68, 72)):
        track(loot, 75, x, index * .15, visible=[x])
    track(loot, 75, 80, 1, visible=[80])
    assert loot.pending_count == 2
    for bad in ({"max_pending": 9, "walk_speed_x": 10}, {"walk_speed_x": float("inf")}, {"walk_speed_x": 21},
                {"max_pending": 1}):  # an unleased walk-back to a remembered site is refused
        with pytest.raises(ValueError):
            collector(**bad)


def test_recorded_fixed_taps_cannot_cover_a_typical_drop_distance_in_time():
    # Lap 1 of log/final-circuit-v4.jsonl: nine visits ran the whole 4 s budget
    # while moving 0-3 minimap px. One 0.04 s tap per 0.45 s release is too slow.
    loot = collector(disappear_s=0)
    track(loot, 75, 78.75, 0)
    pos, visited, states = _run(loot, 75., 0, 5.0)
    assert "dwelling" not in states and loot.goal is None and abs(pos - 78.75) > 1


@pytest.mark.parametrize("speed", [7, 9.5, 12])
@pytest.mark.parametrize("step", [.08, .15, .3])
def test_distance_sized_pulses_reach_the_drop_without_overshoot(speed, step):
    loot = collector(disappear_s=0, walk_speed_x=10)
    track(loot, 75, 78.75, 0)
    pos, visited, states = _run(loot, 75., 0, 5.0, step=step, speed=speed)
    assert "dwelling" in states and states.index("dwelling") * step <= 3.0
    assert max(visited) <= 78.75 + 1  # never beyond the pickup tolerance


def test_distance_sized_pulse_stays_leased_and_short_of_the_platform_edge():
    loot = collector(disappear_s=0, walk_speed_x=10)
    track(loot, 84, 87.9, 0)  # goal 0.7 px inside the floor's right edge at 88.6
    assert loot.step((84, 71), 0, platform_bounds=BOUNDS) == Command.NONE  # brake first
    pulse = loot.step((84, 71), .46, platform_bounds=BOUNDS)
    assert pulse == Command("right") and pulse.valid_for_s is not None
    assert pulse.valid_for_s * 12 <= (88.6 - 1.0) - 84 and pulse.valid_for_s <= .5
    near = collector(disappear_s=0, walk_speed_x=10)
    track(near, 87.2, 88.5, 0)
    near.step((87.2, 71), 0, platform_bounds=BOUNDS)
    assert near.step((87.2, 71), .46, platform_bounds=BOUNDS).valid_for_s == pytest.approx(.04)
