"""Offline kinematic regressions for time spent near walk waypoints.

The model is deliberately simple. The character moves at ``speed`` minimap
px/s only while a movement key is down and slides ``coast`` px after each
key-up. A command reaches the keyboard ``latency`` seconds after its vision
frame; its lease runs from then for ``valid_for_s`` (0.75 s when unleased).
With ``tick_s`` the keyboard only presses and releases on its own tick, as the
30 Hz worker does, so a key-down waits for the next tick and a key-up is up to
one tick late. The circuit only ever sees the rounded minimap pixel. Capture
age is not modelled. This is not the game.

Measured facts the numbers below come from: held walking is 9.2-9.8 minimap
px/s, vision delivers 6-12 frames/s, the command lease is at most 0.75 s.
"""
import math
import random

import pytest

from maplebot.nav import recovery
from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Command

# Live calibration quoted in the measured problem report (config/user.yaml is not read here).
LIVE = {"precision_walk": True, "tolerance_x": 1.0, "tolerance_y": 1.0, "align_tolerance_x": .4,
        "jump_align_tolerance_x": .4, "drop_align_tolerance_x": 1.0, "approach_radius_x": 6,
        "approach_pulse_s": .04, "approach_release_s": .45, "timeout_s": 15}
# Recommended opt-in values. approach_walk_speed_x is an UPPER bound on the true held
# walking speed (measured 9.2-9.8 minimap px/s): a lower value makes every pulse too long.
# The edge margin exceeds the stop margin by the 0.75 px that slide plus rounding can add.
FAR = {"approach_fine_radius_x": 1.5, "approach_walk_speed_x": 11.0, "approach_stop_margin_x": 1.0,
       "approach_edge_margin_x": 1.75, "approach_far_pulse_max_s": .45, "approach_far_release_s": .3,
       "approach_resume_credit": True}
NO_CREDIT = {**FAR, "approach_resume_credit": False}
PLATFORM = {"left": 60, "right": 140, "y": 38}
SPEEDS = (8.5, 9.5, 10.5)  # true px/s, bracketing the measured range
COASTS = (0.0, .3, .6)  # px of slide after a key-up
STARTS = [1.75 + .25 * i for i in range(22)]  # 1.75 .. 7 px from the waypoint
LEASE_S = .75
TICK_S = 1 / 30
REALISTIC = {"tick_s": TICK_S, "latency": (.01, .05)}
FAR_PULSE = lambda distance: (distance - FAR["approach_stop_margin_x"]) / FAR["approach_walk_speed_x"]  # noqa: E731


def walk_nav(target=100.0, platform=PLATFORM, dwell=0, **options):
    return MinimapCircuit({"waypoints": [{"x": target, "y": 38, "dwell_s": dwell}, {"x": 130, "y": 38}],
                           "start_idx": 0, "loop": False, "platforms": [platform], **LIVE, **options})


def simulate(nav, start, *, speed=9.5, frame_s=.15, coast=0.0, gaps=None, limit_s=60.0, y=38,
             tick_s=None, latency=(0.0, 0.0), seed=0):
    """Return (arrival time or None, extreme true X, held seconds, commands)."""
    rng = random.Random(seed)
    phase = rng.uniform(0, tick_s) if tick_s else 0.0
    x, now, held, frame, commands = float(start), 0.0, 0.0, 0, []
    low = high = x
    key, key_up = None, math.inf  # held direction (+1/-1) and when the keyboard lets go

    def on_tick(moment):  # the first keyboard tick at or after ``moment``
        return moment if not tick_s else phase + math.ceil((moment - phase) / tick_s - 1e-9) * tick_s

    def release():
        nonlocal x, key, key_up, low, high
        if key is not None:
            x += coast * key
            low, high = min(low, x), max(high, x)
        key, key_up = None, math.inf

    def advance(clock, until):
        nonlocal x, held, low, high
        if key is not None:
            down = max(0.0, min(until, key_up) - clock)
            x += speed * down * key
            held += down
            low, high = min(low, x), max(high, x)
            if key_up <= until:
                release()

    while now < limit_s:
        command = nav.step((math.floor(x + .5), y), now)  # the minimap pixel holding the true X
        commands.append((now, x, command))
        if nav.idx != 0:
            return now, (low, high), held, commands
        following = now + (gaps or {}).get(frame, frame_s)
        received = now + rng.uniform(*latency)
        applied = on_tick(received)
        advance(now, applied)
        want = {"left": -1, "right": 1}.get(command.move_x)
        deadline = received + (LEASE_S if command.valid_for_s is None else command.valid_for_s)
        if want is None or applied >= deadline:
            release()  # a lease that ran out before a tick never presses the key
        else:
            if key not in (None, want):
                release()
            key, key_up = want, on_tick(deadline)
        advance(applied, following)
        now = following
        frame += 1
    return None, (low, high), held, commands


def test_reproduces_slow_precision_walk_from_approach_radius_with_live_numbers():
    arrived, _, held, commands = simulate(walk_nav(), 94)
    print(f"\nbaseline from 6 px at 6.7 fps, 9.5 px/s: arrived={arrived:.2f} s, key-down {held:.2f} s")
    # One 0.04 s tap (0.38 px at 9.5 px/s) per vision-gated 0.6 s cycle: five pixels take about 7.8 s.
    assert 6 < arrived < 10 and max(c.valid_for_s or 0 for _, _, c in commands) == pytest.approx(.04)


def test_baseline_needs_one_vision_gated_tap_per_cycle_at_the_slowest_speed():
    arrived, _, held, commands = simulate(walk_nav(), 94, speed=8.5)
    print(f"\nbaseline from 6 px at 6.7 fps, 8.5 px/s: arrived={arrived:.2f} s, key-down {held:.2f} s")
    taps = [now for now, _, c in commands if c.move_x == "right"]
    assert arrived > 7 and min(b - a for a, b in zip(taps, taps[1:])) >= .45 + .04


def sweep(fps, *, gaps=None, target=100.0, platform=PLATFORM, far=True, side=-1, realistic=True, starts=STARTS):
    """Worst case over starts, speeds, slides and (optionally) keyboard tick plus command latency.

    Returns (slowest arrival, mean arrival, furthest true travel beyond the waypoint, runs).
    ``side`` is -1 when approaching from the left of the waypoint and +1 from its right.
    """
    slowest = furthest = None
    total = runs = 0
    options = FAR if far else {"timeout_s": 60}
    for index, distance in enumerate(starts):
        for speed in SPEEDS:
            for coast in COASTS:
                arrived, (low, high), _, commands = simulate(
                    walk_nav(target, platform, **options), target + side * distance, speed=speed, coast=coast,
                    frame_s=1 / fps, gaps=gaps, seed=index, **(REALISTIC if realistic else {}))
                assert arrived is not None, (distance, speed, coast)
                moves = [c for _, x, c in commands if c.move_x != "none" and abs(target - math.floor(x + .5)) <= 6]
                # Every decision inside the radius is leased, so a late frame can only end a pulse early.
                assert all(c.valid_for_s is not None and c.valid_for_s <= FAR["approach_far_pulse_max_s"] + 1e-9
                           for c in moves)
                assert all(c.move_x == ("right" if side < 0 else "left") for c in moves)  # never had to come back
                beyond = (high - target) if side < 0 else (target - low)
                slowest = arrived if slowest is None else max(slowest, arrived)
                furthest = beyond if furthest is None else max(furthest, beyond)
                total, runs = total + arrived, runs + 1
    return slowest, total / runs, furthest, runs


@pytest.mark.parametrize("fps", [6, 8, 12])
@pytest.mark.parametrize("realistic", [False, True])
def test_far_pulses_reach_walk_waypoint_quickly_without_leaving_tolerance(fps, realistic):
    slowest, mean, furthest, runs = sweep(fps, realistic=realistic)
    base_slowest, base_mean, base_furthest, _ = sweep(fps, far=False, realistic=realistic)
    label = "30 Hz tick + 10-50 ms latency" if realistic else "ideal keyboard"
    print(f"\n{fps} fps, {label}, {runs} runs from 1.75-7 px: far slowest {slowest:.2f} s mean {mean:.2f} s "
          f"furthest {furthest:+.2f} px | baseline slowest {base_slowest:.2f} s mean {base_mean:.2f} s "
          f"furthest {base_furthest:+.2f} px")
    assert slowest < 4.0 and mean < base_mean / 2
    assert furthest < 1.0  # the true position never leaves the 1.0 px arrival tolerance


def test_one_dropped_half_second_frame_cannot_extend_a_far_pulse():
    results = [sweep(fps, gaps={frame: .5}) for fps in (6, 12) for frame in range(12)]
    slowest, furthest = max(r[0] for r in results), max(r[2] for r in results)
    print(f"\none dropped 0.5 s frame anywhere (6 and 12 fps, tick + latency): slowest arrival {slowest:.2f} s, "
          f"furthest {furthest:+.2f} px")
    assert slowest < 4.5 and furthest < 1.0


@pytest.mark.parametrize("inside", [0.0, .5])
def test_far_pulses_travel_no_further_past_a_platform_edge_than_fine_taps(inside):
    edge = {"left": 89, "right": 140, "y": 38}
    target = edge["left"] + inside  # a connection take-off at, or half a pixel inside, the edge
    worst = {}
    dense = [1.75 + .05 * i for i in range(106)]  # the extreme is one fine tap from a pixel boundary
    for far in (True, False):
        results = [sweep(fps, target=target, platform=edge, far=far, side=1, starts=dense) for fps in (6, 8, 12)]
        results += [sweep(6, gaps={frame: .5}, target=target, platform=edge, far=far, side=1, starts=dense)
                    for frame in (2, 5)]
        # ``furthest`` is travel beyond the waypoint; the edge is ``inside`` px further on.
        worst[far] = max(r[2] for r in results) - inside, max(r[0] for r in results)
    print(f"\nwaypoint {inside} px inside the edge: far approach {worst[True][0]:+.2f} px past the edge "
          f"(slowest {worst[True][1]:.2f} s), fine taps {worst[False][0]:+.2f} px (slowest {worst[False][1]:.2f} s)")
    assert worst[True][0] <= worst[False][0] + 1e-9
    assert worst[True][1] < worst[False][1]


def test_underestimated_walk_speed_overshoots_so_the_key_must_be_an_upper_bound():
    # The earlier "about 4 px/s" calibration, run at the true speed.
    slow = {**FAR, "approach_walk_speed_x": 4.5, "approach_stop_margin_x": .75, "approach_edge_margin_x": 1.5}
    worst = max(simulate(walk_nav(**slow), 100 - distance, speed=speed, coast=.6, frame_s=.15, gaps={3: .5},
                         seed=i, **REALISTIC)[1][1] - 100
                for i, distance in enumerate(STARTS) for speed in SPEEDS)
    print(f"\napproach_walk_speed_x 4.5 at a true 8.5-10.5 px/s: furthest {worst:+.2f} px past the waypoint")
    assert worst > 1.0


def test_far_pulse_is_sized_from_distance_and_only_shortened_by_fresh_frames():
    nav = walk_nav(**FAR)
    assert nav.step((94, 38), 0) == Command.NONE  # release first
    assert nav.step((94, 38), .15) == Command.NONE
    first = nav.step((94, 38), .31)
    assert first == Command("right") and first.valid_for_s == pytest.approx(.45)  # capped, far under the lease
    assert nav.step((95, 38), .46).valid_for_s == pytest.approx(.3)  # remaining time, never renewed
    assert nav.step((95, 38), .61).valid_for_s == pytest.approx(.15)
    assert nav.step((96, 38), .77) == Command.NONE  # key-up was at the 0.76 deadline
    assert nav.step((96, 38), .92) == Command.NONE
    # 0.3 s after the enforced key-up at 0.76; a release anchored at the frame
    # that saw it (0.77) would still be waiting at 1.065.
    second = nav.step((98, 38), 1.065)
    assert second == Command("right") and second.valid_for_s == pytest.approx(FAR_PULSE(2))
    assert nav.step((99, 38), 1.2) == Command.NONE and nav.idx == 1  # arrival is still the ordinary check


def test_far_release_already_served_by_a_late_frame_starts_the_next_pulse_at_once():
    nav = walk_nav(**FAR)
    for now in (0, .15):
        nav.step((94, 38), now)
    assert nav.step((94, 38), .31).valid_for_s == pytest.approx(.45)  # the keyboard ends it at 0.76
    late = nav.step((97, 38), 1.2)  # the first frame after it: 0.44 s released, more than the 0.3 s needed
    assert late == Command("right") and late.valid_for_s == pytest.approx(FAR_PULSE(3))


def test_far_pulse_shortens_when_the_player_is_closer_than_planned():
    nav = walk_nav(**FAR)
    for now in (0, .15):
        nav.step((94, 38), now)
    assert nav.step((94, 38), .31).valid_for_s == pytest.approx(.45)
    shortened = nav.step((98, 38), .46)  # four pixels in one frame: the old 0.3 s remainder is too long
    assert shortened == Command("right") and shortened.valid_for_s == pytest.approx(FAR_PULSE(2))
    assert nav.step((98, 38), .5).valid_for_s == pytest.approx(FAR_PULSE(2) - .04)


def test_fine_radius_keeps_existing_taps_and_release():
    nav = walk_nav(**FAR)
    assert nav.step((98.5, 38), 0) == Command.NONE
    assert nav.step((98.5, 38), .3) == Command.NONE  # fine release is the configured 0.45 s, not the far 0.3 s
    assert nav.step((98.5, 38), .44) == Command.NONE
    tap = nav.step((98.5, 38), .46)
    assert tap == Command("right") and tap.valid_for_s == pytest.approx(.04)
    assert nav.step((98.5, 38), .51) == Command.NONE
    assert nav.step((98.5, 38), .9) == Command.NONE
    assert nav.step((98.5, 38), .97).valid_for_s == pytest.approx(.04)


def test_far_pulse_that_ends_inside_fine_radius_releases_before_any_tap():
    nav = walk_nav(**FAR)
    for now in (0, .15):
        nav.step((96, 38), now)
    assert nav.step((96, 38), .31).valid_for_s == pytest.approx(FAR_PULSE(4))  # 0.27 s, until 0.58
    assert nav.step((98.5, 38), .46) == Command.NONE  # entered the fine radius: pulse is cut, not carried over
    assert nav.step((98.5, 38), .86) == Command.NONE
    assert nav.step((98.5, 38), .92).valid_for_s == pytest.approx(.04)


def test_far_approach_brakes_before_reversing():
    nav = walk_nav(**FAR)
    for now in (0, .15):
        nav.step((96, 38), now)
    assert nav.step((96, 38), .31) == Command("right")
    assert nav.step((104, 38), .46) == Command.NONE  # knocked past the point: release first
    assert nav.step((104, 38), .61) == Command.NONE
    reverse = nav.step((104, 38), .77)
    assert reverse == Command("left") and reverse.valid_for_s == pytest.approx(FAR_PULSE(4))


def test_far_pulse_stops_short_of_platform_edge_at_the_waypoint():
    edge = {"left": 89, "right": 110, "y": 38}
    nav = walk_nav(target=89, platform=edge, **FAR)
    for now in (0, .15):
        assert nav.step((92, 38), now) == Command.NONE
    pulse = nav.step((92, 38), .31)
    # Three pixels of floor remain: the 1.75 px edge margin governs, not the 1.0 px stop margin.
    assert pulse == Command("left") and pulse.valid_for_s == pytest.approx((3 - 1.75) / 11)
    wide = walk_nav(target=89, platform=edge, **{**FAR, "approach_edge_margin_x": 3.0})
    commands = [wide.step((92, 38), now / 100) for now in range(0, 200, 15)]
    assert max(c.valid_for_s or 0 for c in commands) == pytest.approx(.04)  # no room: fine taps only


def test_far_pulses_need_a_known_destination_platform_when_platforms_exist():
    nav = MinimapCircuit({"waypoints": [{"x": 100, "y": 38, "dwell_s": 0}, {"x": 130, "y": 38}], "start_idx": 0,
                          "platforms": [{"left": 60, "right": 99, "y": 38}], **LIVE, **FAR})
    commands = [nav.step((94, 38), now / 100) for now in range(0, 300, 15)]
    assert any(c.move_x == "right" for c in commands)
    assert max(c.valid_for_s or 0 for c in commands) <= .04 + 1e-9
    free = MinimapCircuit({"waypoints": [{"x": 100, "y": 38, "dwell_s": 0}, {"x": 130, "y": 38}],
                           "start_idx": 0, **LIVE, **FAR})
    assert max(free.step((94, 38), now / 100).valid_for_s or 0 for now in range(0, 64, 16)) == pytest.approx(.45)


def test_far_keys_do_not_change_floor_check_arrival_or_free_walk():
    nav = walk_nav(**FAR)
    assert nav.step((94, 45), 0) == Command.NONE and nav.state == "waiting_floor"
    free = walk_nav(**FAR).step((80, 38), 0)
    assert free == Command("right") and free.valid_for_s is None
    arrived = walk_nav(**FAR)
    arrived.step((98, 38), 0)
    assert arrived.idx == 0  # two pixels away is not arrival
    arrived.step((99, 38), .15)
    assert arrived.idx == 1


@pytest.mark.parametrize("action, extra", [("jump", {"from_x": 100, "from_y": 38, "y": 30}),
                                           ("ramp", {"from_x": 100, "from_y": 38, "y": 30}),
                                           ("drop", {"y": 50}), ("climb", {"y": 30})])
def test_transition_takeoff_alignment_keeps_fine_taps_only(action, extra):
    point = {"x": 100 if action in ("drop", "climb") else 106, "y": 38, "action": action, "dwell_s": 0, **extra}
    nav = MinimapCircuit({"waypoints": [point, {"x": 130, "y": 30}], "start_idx": 0, **LIVE, **FAR})
    commands = [nav.step((95, 38), now / 100) for now in range(0, 400, 15)]
    assert any(c.move_x == "right" for c in commands)
    assert max(c.valid_for_s or 0 for c in commands) == pytest.approx(.04)


def test_absent_far_keys_keep_every_near_correction_a_fine_tap():
    nav = walk_nav()
    assert nav.approach_fine_radius_x is None and not nav.approach_resume_credit
    commands = [nav.step((94, 38), now / 100) for now in range(0, 400, 15)]
    assert {round(c.valid_for_s, 6) for c in commands if c.move_x == "right"} == {.04}
    assert [c.move_x for c in commands[:5]] == ["none", "none", "none", "right", "none"]


@pytest.mark.parametrize("bad", [{"approach_walk_speed_x": 0}, {"approach_fine_radius_x": 6},
                                 {"approach_fine_radius_x": .7}, {"approach_stop_margin_x": 0},
                                 {"approach_far_pulse_max_s": .7}, {"approach_far_pulse_max_s": .01},
                                 {"approach_edge_margin_x": -1}, {"approach_far_release_s": 0},
                                 {"approach_walk_speed_x": 100}])
def test_far_approach_configuration_is_validated(bad):
    with pytest.raises(ValueError):
        walk_nav(**{**FAR, **bad})
    with pytest.raises(ValueError):
        walk_nav(approach_fine_radius_x=1.5)  # no walk speed: nothing to size a pulse from


@pytest.mark.parametrize("stop, edge", [(1.0, 1.0), (1.0, 1.7), (.75, 1.0), (.5, 1.2)])
def test_edge_margin_must_cover_slide_and_rounding_beyond_the_stop_margin(stop, edge):
    with pytest.raises(ValueError, match="approach_edge_margin_x"):
        walk_nav(**{**FAR, "approach_stop_margin_x": stop, "approach_edge_margin_x": edge})
    walk_nav(**{**FAR, "approach_stop_margin_x": stop, "approach_edge_margin_x": stop + .75})  # the boundary is valid
    # Without far pulses the two margins are unused and unconstrained, as before.
    walk_nav(approach_stop_margin_x=stop, approach_edge_margin_x=edge)


def test_return_to_start_sub_circuit_ignores_far_pulses_and_resume_credit(monkeypatch):
    route = {"waypoints": [{"x": 100, "y": 38, "dwell_s": 0, "platform_id": 1}, {"x": 130, "y": 38, "platform_id": 1}],
             "start_idx": 0, "loop": False, "platforms": [PLATFORM], "movement_test": True, **LIVE, **FAR}
    monkeypatch.setattr(recovery, "supported_platform", lambda cfg, pos: {"id": 1})
    monkeypatch.setattr(recovery, "plan_return_to_start", lambda cfg, pos: dict(route))
    nav = walk_nav(**FAR)
    assert nav.approach_fine_radius_x == 1.5 and nav.approach_resume_credit
    assert nav.request_return_to_start()
    commands = [nav.step((94, 38), now / 100) for now in range(0, 400, 15)]
    assert nav.recovery is not None and nav.recovery_state == "running"
    assert nav.recovery.approach_fine_radius_x is None and not nav.recovery.approach_resume_credit
    assert any(c.move_x == "right" for c in commands)
    assert max(c.valid_for_s or 0 for c in commands) == pytest.approx(.04)  # the same route far-pulses for 0.45 s
    plain = MinimapCircuit(route, far_approach=False)
    assert plain.approach_fine_radius_x is None and not plain.approach_resume_credit


def _flicker(nav, suspends, resume_pos=(98.5, 38), before=(0, .16)):
    """Fine release in progress, then suspend frames, then steps every 0.16 s."""
    for now in before:
        assert nav.step((98.5, 38), now) == Command.NONE
    for i in range(suspends):
        nav.suspend(before[-1] + .16 * (i + 1))
    start = before[-1] + .16 * (suspends + 1)
    return [c.move_x for c in (nav.step(resume_pos, start + .16 * i) for i in range(5))]


def test_default_suspend_still_restarts_the_full_release():
    assert _flicker(walk_nav(), 1) == ["none", "none", "none", "right", "none"]
    assert _flicker(walk_nav(**NO_CREDIT), 3) == ["none", "none", "none", "right", "none"]


@pytest.mark.parametrize("suspends", [1, 2, 3, 20])
def test_release_credit_counts_served_release_and_never_resumes_a_pulse(suspends):
    # 0.16 s were served before the first suspend and the interruption itself was at least 0.16 s.
    assert _flicker(walk_nav(approach_resume_credit=True), suspends)[:2] == ["none", "right"]
    # Nothing served before a one-frame flicker: 0.45 - 0.16 s remain, where the default costs a frame more.
    fresh = _flicker(walk_nav(approach_resume_credit=True), 1, before=(0,))
    assert fresh == ["none", "none", "right", "none", "none"]
    assert _flicker(walk_nav(), 1, before=(0,)) == ["none", "none", "none", "right", "none"]


def test_repeated_suspends_do_not_move_the_release_anchor():
    nav = walk_nav(approach_resume_credit=True)
    nav.step((98.5, 38), 0)
    nav.suspend(.1)
    nav.suspend(.2)
    nav.suspend(.3)  # a restart at each suspend would leave 0.35 s to wait after the next step
    assert nav.step((98.5, 38), .4) == Command.NONE  # still a release frame; 0.15 s remain from the first suspend
    assert nav.step((98.5, 38), .54) == Command.NONE
    assert nav.step((98.5, 38), .56).valid_for_s == pytest.approx(.04)


def test_suspended_pulse_is_not_resumed_even_with_credit():
    nav = walk_nav(approach_resume_credit=True)
    nav.step((98.5, 38), 0)
    assert nav.step((98.5, 38), .46) == Command("right")
    nav.suspend(.47)
    assert nav.step((98.5, 38), .48) == Command.NONE  # the old pulse had 0.02 s left
    assert nav.step((98.5, 38), .9) == Command.NONE  # a whole release from the interruption
    assert nav.step((98.5, 38), .94) == Command("right")


def test_release_credit_is_refused_after_observed_movement_or_reversal():
    moved = _flicker(walk_nav(approach_resume_credit=True), 3, resume_pos=(97.9, 38))
    assert moved == ["none", "none", "none", "right", "none"]
    reversed_ = _flicker(walk_nav(approach_resume_credit=True), 3, resume_pos=(101.5, 38))
    assert reversed_ == ["none", "none", "none", "left", "none"]


def test_suspend_during_a_far_pulse_requires_a_whole_release_from_the_interruption():
    nav = walk_nav(**FAR)
    for now in (0, .15):
        nav.step((94, 38), now)
    assert nav.step((94, 38), .31).valid_for_s == pytest.approx(.45)
    nav.suspend(.45)
    assert nav.step((94, 38), .6) == Command.NONE  # 0.15 s since the interruption: not yet settled
    assert nav.step((94, 38), .76) == Command("right")  # same pixel and 0.3 s released
    nav = walk_nav(**FAR)
    for now in (0, .15, .31):
        nav.step((94, 38), now)
    nav.suspend(.45)
    assert nav.step((95, 38), .6) == Command.NONE  # it moved: no credit at all
    assert nav.step((95, 38), .75) == Command.NONE
    assert nav.step((95, 38), .92) == Command("right")


def test_release_credit_does_not_survive_a_step_without_an_approach():
    nav = walk_nav(approach_resume_credit=True)
    nav.step((98.5, 38), 0)
    nav.step((98.5, 38), .4)
    nav.suspend(.5)
    assert nav.step(None, 5) == Command.NONE  # lost position: alignment and credit are gone
    commands = [nav.step((98.5, 38), 5.16 + .16 * i) for i in range(5)]
    assert [c.move_x for c in commands] == ["none", "none", "none", "right", "none"]


def test_dwelling_before_a_suspend_is_not_counted_as_served_release():
    # A far pulse from 97 is last corrected at 98, then the waypoint is reached and dwelt on.
    nav = walk_nav(dwell=5, **FAR)
    for now in (0, .15):
        assert nav.step((97, 38), now) == Command.NONE
    assert nav.step((97, 38), .31).valid_for_s == pytest.approx(FAR_PULSE(3))
    assert nav.step((98, 38), .45) == Command("right")
    assert nav.step((98, 38), .6) == Command.NONE
    for now in (.75, .9, 1.05, 1.2):
        assert nav.step((99, 38), now) == Command.NONE and nav.state == "dwelling"
    nav.suspend(1.3)  # e.g. combat; the player is pushed a pixel back to exactly the stale 98
    # The dwell is neither a served release nor evidence about pixel 98: a whole 0.3 s release follows.
    assert nav.step((98, 38), 1.42) == Command.NONE
    assert nav.step((98, 38), 1.5) == Command.NONE
    assert nav.step((98, 38), 1.71) == Command.NONE
    pulse = nav.step((98, 38), 1.73)
    assert pulse == Command("right") and pulse.valid_for_s == pytest.approx(FAR_PULSE(2))


@pytest.mark.parametrize("options", [{"approach_resume_credit": True}, FAR])
def test_a_non_approach_step_after_a_suspend_forfeits_credit_and_the_stale_cycle(options):
    nav = walk_nav(dwell=5, **options)
    nav.step((98.5, 38), 0)
    nav.step((98.5, 38), .4)  # 0.4 s of the 0.45 s release served
    nav.suspend(.5)
    assert nav.step((99, 38), .6) == Command.NONE and nav.state == "dwelling"
    commands = [nav.step((98.5, 38), .76 + .16 * i) for i in range(5)]
    assert [c.move_x for c in commands] == ["none", "none", "none", "right", "none"]


def test_waiting_for_the_floor_ends_the_measured_release():
    nav = walk_nav(**FAR)
    nav.step((98.5, 38), 0)
    nav.step((98.5, 38), .4)
    assert nav.step((98.5, 45), .5) == Command.NONE and nav.state == "waiting_floor"  # knocked off the platform
    nav.suspend(.6)
    commands = [nav.step((98.5, 38), .76 + .16 * i) for i in range(5)]
    assert [c.move_x for c in commands] == ["none", "none", "none", "right", "none"]


def test_credited_resume_never_pulses_on_the_very_next_frame():
    nav = walk_nav(**FAR)
    for frame in range(6):  # 12 fps: 0.415 s of the 0.45 s fine release served
        assert nav.step((98.5, 38), .083 * frame) == Command.NONE
    nav.suspend(.5)
    nav.suspend(.58)
    # Served plus interrupted time exceed the release, yet one same-pixel frame is not enough.
    assert nav.step((98.5, 38), .66) == Command.NONE
    assert nav.step((98.5, 38), .743) == Command.NONE  # 0.083 s after the resume frame
    assert nav.step((98.5, 38), .826).valid_for_s == pytest.approx(.04)  # 0.166 s: past the 0.15 s floor
    far = walk_nav(**FAR)
    for frame in range(3):  # 0.2 s of the 0.3 s far release served, the rest during the interruption
        far.step((94, 38), .1 * frame)
    far.suspend(.3)
    assert far.step((94, 38), .4) == Command.NONE
    assert far.step((94, 38), .5) == Command.NONE
    assert far.step((94, 38), .56).valid_for_s == pytest.approx(.45)


def test_movement_seen_after_a_credited_resume_restarts_the_whole_release():
    nav = walk_nav(approach_resume_credit=True)
    for frame in range(6):
        nav.step((98.5, 38), .083 * frame)
    nav.suspend(.5)
    assert nav.step((98.5, 38), .66) == Command.NONE  # same pixel: credited
    # ...but it was only passing through that pixel with another controller's key still down.
    moving = [nav.step((97.9, 38), .66 + .083 * frame) for frame in range(1, 8)]
    assert [c.move_x for c in moving] == ["none"] * 6 + ["right"]
