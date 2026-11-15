"""Opt-in combat during the pre-takeoff part of a transition (`attack.pre_transition_s`).

Every circuit here is a real MinimapCircuit driven through Brain.decide or
MinimapCircuit.step with scripted minimap positions. Nothing is live: no
capture, keyboard or game.
"""
import math

import pytest

from maplebot.brain.fsm import Brain, Perception
from maplebot.config import deep_merge, load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.nav.circuit import MinimapCircuit
from maplebot.types import Box, Command, Detection

W, H, P = 1282, 693, (641, 346)
SCALE = 16  # screen px per minimap px
ATTACKS = ("attack", "aoe", "normal_attack")

A = {"left": 0, "right": 40, "y": 50}    # jump takeoff floor
B = {"left": 50, "right": 90, "y": 40}   # jump landing / drop ledge
C = {"left": 0, "right": 100, "y": 60}   # floor under the ledge

JUMP = {"start_idx": 1, "loop": False, "platforms": [A, B, C], "waypoints": [
    {"x": 30, "y": 50, "dwell_s": 0},
    {"x": 55, "y": 40, "action": "jump", "from_x": 30, "from_y": 50, "dwell_s": 0},
    {"x": 30, "y": 50, "action": "jump", "from_x": 55, "from_y": 40, "dwell_s": 0}]}
DROP = {"start_idx": 1, "loop": False, "platforms": [B, C], "max_retries": 1, "drop_interval_s": .8,
        "waypoints": [{"x": 70, "y": 40, "dwell_s": 0},
                      {"x": 70, "y": 60, "action": "drop", "dwell_s": 0},
                      {"x": 60, "y": 60, "dwell_s": 0}]}
CLIMB = {"start_idx": 1, "loop": False, "platforms": [B, C],
         "waypoints": [{"x": 60, "y": 60, "dwell_s": 0},
                       {"x": 60, "y": 40, "action": "climb", "dwell_s": 0},
                       {"x": 70, "y": 40, "dwell_s": 0}]}
RAMP = {"start_idx": 1, "loop": False,
        "platforms": [{"left": 0, "right": 65, "y": 80}, {"left": 75, "right": 100, "y": 63}],
        "waypoints": [{"x": 63, "y": 80},
                      {"x": 80, "y": 63, "action": "ramp", "from_x": 63, "from_y": 80, "dwell_s": 0}]}
# One lap with a jump and a drop, for the default-off differential.
LAP = {"start_idx": 0, "loop": False, "platforms": [A, B, C], "waypoints": [
    {"x": 30, "y": 50, "dwell_s": 0},
    {"x": 55, "y": 40, "action": "jump", "from_x": 30, "from_y": 50, "dwell_s": 0},
    {"x": 70, "y": 40, "dwell_s": 0},
    {"x": 70, "y": 60, "action": "drop", "dwell_s": 0},
    {"x": 60, "y": 60, "dwell_s": 0}]}
CLIMB_DWELL = {**CLIMB, "waypoints": [CLIMB["waypoints"][0], {**CLIMB["waypoints"][1], "dwell_s": 1.0},
                                      CLIMB["waypoints"][2]]}
CLIMB_SLOW = {**CLIMB, "timeout_s": 1.0}
# After the last fight frame the circuit stays suspended for Brain._reacquire_s
# (0.4 s) before it is stepped again; tests resume this long after it.
QUIET = .5
FLIGHT = [(31, 48), (40, 38), (50, 36), (54, 38), (55, 40), (55, 40), (55, 40), (55, 40), (55, 40)]


def cfg_for(circuit, pre=2.0, **over):
    attack = {} if pre is None else {"pre_transition_s": pre}
    cfg = deep_merge(load_config(), {"attack": attack, "route": {"circuit": {
        "enabled": True, "pixels_per_minimap_x": SCALE, **circuit}}})
    return deep_merge(cfg, over)


def brain(circuit, pre=2.0, **over):
    return Brain(cfg_for(circuit, pre, **over), None, 1200)


def see(pos, now, dets=(), on_ladder=False):
    return Perception(W, H, P, None, pos, 150, on_ladder, list(dets), now, 0)


def mob(dx, dy=0, w=40, h=30):
    """Monster centred `dx` screen px beside the player (16 px = one minimap px)."""
    return Detection("m", Box(P[0] + dx - w // 2, P[1] + dy - h // 2, w, h), 0.1)


def run(b, frames, start, dets=(), dt=.1):
    """Feed positions at `dt` spacing; returns the (command, brain state) per frame."""
    out = []
    for i, pos in enumerate(frames):
        out.append((b.decide(see(pos, round(start + i * dt, 6), dets)), b.state))
    return out


# -- MinimapCircuit.transition_committed ---------------------------------------
def test_committed_is_false_outside_any_transition_and_true_for_a_return():
    nav = MinimapCircuit({"start_idx": 0, "waypoints": [{"x": 10, "y": 50}, {"x": 30, "y": 50}]})
    nav.step((5, 50), 0)
    assert not nav.transition_active and not nav.transition_committed
    assert not nav.takeoff_interruptible((5, 50))  # nothing to interrupt
    assert nav.request_return_to_start()
    assert nav.transition_committed and not nav.takeoff_interruptible((5, 50))


def test_jump_is_uncommitted_only_before_its_takeoff_key():
    nav = MinimapCircuit({**JUMP, "max_retries": 1, "jump_flight_timeout_s": .5, "recovery_s": .4})
    seen = {}
    script = [((30, 45), 0), ((25, 50), .1), ((25, 50), .4), ((30, 50), .5), ((30, 50), .6),
              ((30, 50), .8), ((31, 48), .9), ((40, 38), 1.0), ((40, 38), 1.5), ((40, 50), 1.6),
              ((30, 50), 2.1), ((30, 50), 2.4), ((40, 38), 2.5), ((54, 38), 2.6), ((55, 40), 2.7)]
    for pos, now in script:
        cmd = nav.step(pos, now)
        seen.setdefault(nav.state, set()).add(nav.transition_committed)
        if cmd.action == "jump":
            assert nav.transition_committed  # the frame that issues the key is already committed
        assert nav.takeoff_interruptible(pos) == (not nav.transition_committed)
    assert seen == {"jump_waiting_floor": {False}, "jump_aligning": {False}, "jump_settling": {False},
                    "jumping": {True}, "airborne": {True}, "jump_retry": {True}, "jump_landing": {True}}


def test_ramp_is_uncommitted_only_before_it_starts_walking_the_slope():
    nav = MinimapCircuit(RAMP)
    seen = {}
    for pos, now in [((63, 75), 0), ((58, 80), .1), ((63, 80), .2), ((63, 80), .5), ((67, 76), .7),
                     ((72, 71), .9), ((80, 63), 1.1)]:
        nav.step(pos, now)
        seen.setdefault(nav.state, set()).add(nav.transition_committed)
    assert seen == {"ramp_waiting_floor": {False}, "ramp_aligning": {False}, "ramp_settling": {False},
                    "ramping": {True}, "ramp_landing": {True}}


def test_ramp_knockback_wait_is_committed():
    nav = MinimapCircuit(RAMP)
    nav._ramp_retry_since = 0.0  # the release-and-wait window after a backward hit on the stair
    assert nav.transition_active and nav.transition_committed


def test_drop_is_committed_from_its_key_until_it_realigns_on_the_source_floor():
    nav = MinimapCircuit(DROP)
    nav.step((70, 40), 0)
    assert nav.state == "settling" and not nav.transition_committed  # first alignment: not yet a transition
    assert nav.step((70, 40), .3) == Command("none", "down", "jump")
    assert nav.transition_committed and not nav.takeoff_interruptible((70, 40))
    now = .3
    while nav._drop_launched_at is not None:  # knocked about, never left the ledge
        now = round(now + .1, 6)
        nav.step((70, 40), now)
        assert nav.transition_committed and now < 3
    assert nav.retries == 1 and nav.state == "dropping" and nav.transition_committed  # unstepped: still committed
    nav.step((64, 40), now + .1)
    assert nav.state == "aligning" and not nav.transition_committed
    assert nav.takeoff_interruptible((64, 40))
    # The fresh-position guard is the takeoff test's own +/-1 px, on both sides:
    # 2 px below is the start of a fall, 2 px above is Down+Jump hopping upward.
    assert nav._drop_source_y == 40
    assert nav.takeoff_interruptible((64, 41)) and nav.takeoff_interruptible((64, 39))
    assert not nav.takeoff_interruptible((64, 42)) and not nav.takeoff_interruptible((64, 38))
    # Knocked off the ledge since that frame: falling, or already on a lower floor.
    assert not nav.takeoff_interruptible((64, 47)) and not nav.takeoff_interruptible((64, 60))
    assert not nav.takeoff_interruptible(None)
    nav.step((64, 47), now + .2)
    assert nav.state == "dropping" and nav.transition_committed
    nav.step((64, 60), now + .3)
    assert nav.state == "dwelling" and nav.transition_committed


def test_every_climb_state_that_holds_the_route_is_committed():
    nav = MinimapCircuit(CLIMB_DWELL)
    states = set()
    for pos, now in [((60, 60), 0), ((60, 60), .3), ((60, 55), .5), ((60, 48), .7), ((60, 42), .9),
                     ((60, 40), 1.0), ((60, 40), 1.2), ((60, 40), 1.4)]:
        nav.step(pos, now)
        assert nav.idx == 1
        if nav.transition_active:
            states.add(nav.state)
            assert nav.transition_committed and not nav.takeoff_interruptible(pos)
    # Dwelling on the top platform is still before landing proof (the dwell is not served).
    assert states == {"settling", "climbing", "dwelling"}


def test_climb_rest_after_a_timeout_is_committed():
    nav = MinimapCircuit(CLIMB_SLOW)
    for i in range(12):
        nav.step((63, 60), round(i * .1, 6))  # never gets under the rope
    assert nav.state == "recovering" and nav.retries == 1 and nav.transition_active
    assert nav.transition_committed and not nav.takeoff_interruptible((63, 60))


# -- default off ----------------------------------------------------------------
def test_default_is_off_and_matches_the_bare_circuit_through_a_jump_and_a_drop():
    assert "pre_transition_s" not in load_config()["attack"]
    plain, zero, reference = brain(LAP, pre=None), brain(LAP, pre=0), MinimapCircuit(LAP)
    assert plain._pre_transition_s == 0 == zero._pre_transition_s
    positions = ([(30, 50)] * 5 + FLIGHT[:4] + [(55, 40)] * 4 + [(60, 40), (66, 40)] + [(70, 40)] * 6
                 + [(70, 45), (70, 55)] + [(70, 60)] * 4 + [(66, 60), (60, 60), (60, 60)])
    states, attacked_frames = set(), 0
    for i, pos in enumerate(positions):
        now = round(i * .1, 6)
        # A monster in range on the same floor whenever the transition holds the route.
        dets = [mob(-60)] if reference.transition_active else []
        attacked_frames += bool(dets)
        expected = reference.step(pos, now)
        for b in (plain, zero):
            assert b.decide(see(pos, now, dets)) == expected
            assert b.state == "circuit_" + reference.state and b.target is None
            assert (b.circuit.idx, b.circuit.state, b.circuit.retries) == (reference.idx, reference.state, 0)
        states.add(reference.state)
    assert reference.completed and plain.circuit.completed and zero.circuit.completed
    assert attacked_frames >= 15
    assert {"jump_settling", "jumping", "airborne", "jump_landing", "settling", "dropping", "dwelling"} <= states


@pytest.mark.parametrize("value", [-.1, 10.5, math.nan, math.inf, "2", True, None])
def test_budget_key_is_validated(value):
    with pytest.raises(ValueError, match="pre_transition_s"):
        brain(JUMP, pre=value) if value is not None else Brain(
            deep_merge(cfg_for(JUMP, pre=None), {"attack": {"pre_transition_s": None}}), None, 1200)


# -- attacking before the takeoff ----------------------------------------------------
def test_fresh_same_floor_monster_is_attacked_while_lining_up_and_only_on_frames_that_see_it():
    b = brain(JUMP)
    cmd = b.decide(see((25, 50), 0, [mob(-60)]))
    assert cmd == Command("none", "none", "attack", face_x="left")
    assert b.state == "hunting" and b.target == "left"
    assert b.circuit.idx == 1 and b.circuit.state == "initializing"  # the circuit was not stepped
    # The detection is gone: remembered pixels are never attacked. The fight holds
    # without any key for the short reacquire window, then the circuit resumes.
    for now in (.1, .2, .3):
        assert b.decide(see((25, 50), now)) == Command.NONE
        assert b.state == "reacquiring_target" and b.target is None and b._combat_memory is None
        assert b.circuit.state == "initializing"
    cmd = b.decide(see((25, 50), .5))
    assert cmd.action == "none" and cmd.face_x == "none"
    assert b.state == "circuit_jump_aligning" and b.target is None and b._combat_memory is None
    cmd = b.decide(see((25, 50), .6, [mob(48)]))
    assert cmd == Command("none", "none", "attack", face_x="right") and b.state == "hunting"


def test_reacquire_window_never_sends_a_key_and_ends_without_a_new_detection():
    b = brain(JUMP)
    assert b.decide(see((30, 50), 0, [mob(-60)])).action == "attack"
    out = run(b, [(30, 50)] * 12, .1)
    states = [state for _, state in out]
    assert states[:3] == ["reacquiring_target"] * 3 and states.count("reacquiring_target") <= 4
    assert all(cmd == Command.NONE for cmd, state in out if state == "reacquiring_target")
    assert all(state.startswith("circuit_") for state in states[4:])
    assert b.circuit.transition_committed  # the route went on to its takeoff


def test_reacquire_window_does_not_hold_off_the_platform_or_once_committed():
    b = brain(JUMP)
    assert b.decide(see((30, 50), 0, [mob(-60)])).action == "attack"
    b.decide(see((30, 45), .1))  # knocked into the air: not a fight frame
    assert b.state == "circuit_jump_waiting_floor"
    b = brain(DROP)
    now = _failed_drop(b)
    b.decide(see((70, 40), now + .1))
    assert b.decide(see((70, 40), now + .2, [mob(-60)])).action == "attack"
    assert b.decide(see((70, 42), now + .3)) == Command.NONE  # 2 px under the ledge: falling
    assert b.state == "circuit_dropping" and b.circuit.transition_committed


# -- facing with monsters on both sides -----------------------------------------------
def _surrounded(i, jitter=6):
    """One monster each side at about 70 px; which one is nearer alternates per frame."""
    shift = jitter if i % 2 else -jitter
    return [mob(-70 + shift), mob(70 + shift)]


def test_monsters_on_both_sides_do_not_flip_the_facing_every_frame():
    b = brain(JUMP, pre=5)
    faces = []
    for i in range(12):
        cmd = b.decide(see((25, 50), i * .1, _surrounded(i)))
        assert b.state == "hunting" and cmd.action == "attack" and cmd.move_x == "none"
        faces.append(cmd.face_x)
    assert faces[0] in ("left", "right") and faces == [faces[0]] * 12
    # The kept side is not stale: once it has no fresh detection the other is faced at once.
    other = "left" if faces[0] == "right" else "right"
    cmd = b.decide(see((25, 50), 1.2, [mob(-70 if other == "left" else 70)]))
    assert cmd == Command("none", "none", "attack", face_x=other)


def test_kept_side_is_forgotten_with_the_fight_and_the_waypoint():
    b = brain(JUMP, pre=5)
    assert b.decide(see((25, 50), 0, [mob(-70)])).face_x == "left"
    assert b._pre_side == "left"
    run(b, [(25, 50)] * 6, .1)  # the fight ends: nothing seen for the reacquire window
    assert b._pre_side is None and b._pre_seen is None
    # A fresh tie starts with the nearer monster, not with the old side.
    assert b.decide(see((25, 50), .8, [mob(-76), mob(64)])).face_x == "right"
    b.suspend(.9)
    assert b._pre_side is None and not b._pre_engaged


def test_surrounded_fight_presses_the_attack_key_instead_of_tapping_directions():
    cfg = cfg_for(JUMP, 5)
    wall = [0.0]
    backend = RecordingBackend()
    keyboard = KeyboardController(cfg, backend, clock=lambda: wall[0], sleep=lambda s: None)
    b = Brain(cfg, None, 1200)
    frame = 0
    for tick in range(90):  # 3 s of 30 Hz key ticks, a fresh frame every 0.1 s
        now = round(tick / 30, 6)
        wall[0] = now
        if tick % 3 == 0:
            keyboard.set_command(b.decide(see((25, 50), now, _surrounded(frame))), source_time=now)
            frame += 1
        keyboard.apply_once(now)
    downs = [key for kind, key in backend.events if kind == "down"]
    keys = cfg["keys"]
    assert downs.count(keys["attack"]) >= 3
    assert downs.count(keys["left"]) + downs.count(keys["right"]) <= 1  # one turn, no movement spam


def test_without_the_key_the_same_frames_never_attack():
    b = brain(JUMP, pre=None)
    for i in range(12):
        cmd = b.decide(see((25, 50), i * .1, [mob(-60)]))
        assert cmd.action not in ATTACKS and cmd.face_x == "none" and b.state.startswith("circuit_jump_")


def test_player_must_stand_on_a_known_platform():
    for pos, on_ladder in [((30, 45), False), ((45, 50), False), ((30, 50), True), (None, False)]:
        b = brain(JUMP)
        cmd = b.decide(see(pos, 0, [mob(-60)], on_ladder))
        assert cmd.action not in ATTACKS and b.state.startswith("circuit_"), (pos, on_ladder)
    b = brain(JUMP)
    perception = Perception(W, H, None, None, (30, 50), 150, False, [mob(-60)], 0, 0)  # no player fix
    assert b.decide(perception).action not in ATTACKS and b.state.startswith("circuit_")


def test_monster_past_the_platform_end_or_out_of_range_is_ignored():
    b = brain(JUMP)
    # 38 + 80/16 = 43 is over the gap beyond the floor that ends at 40.
    assert b.decide(see((38, 50), 0, [mob(80)])).action not in ATTACKS
    assert b.state.startswith("circuit_")
    # On the floor but outside the 350 px attack box.
    assert b.decide(see((38, 50), .1, [mob(-420)])).action not in ATTACKS
    assert b.state.startswith("circuit_") and b._pre_spent == 0


def test_monster_on_another_surface_is_ignored_with_the_surface_test_on():
    below = mob(-60, dy=70)  # sprite bottom 73 px under the feet line: the floor underneath
    wide = {"range_y": 220}
    loose = brain(JUMP, attack=wide)
    assert loose.decide(see((25, 50), 0, [below])).action == "attack"  # band overlap alone accepts it
    strict = brain(JUMP, attack={**wide, "surface": {"enabled": True}})
    for i in range(4):
        cmd = strict.decide(see((25, 50), i * .1, [below]))
        assert cmd.action not in ATTACKS and cmd.face_x == "none"
        assert strict.state.startswith("circuit_") and strict.target is None
    assert strict._pre_spent == 0
    assert strict.decide(see((25, 50), .5, [below, mob(-60)])).action == "attack"  # one on the player's surface


@pytest.mark.parametrize("pos,dx", [((30, 50), -300), ((30, 50), -40), ((30, 50), 150), ((1.5, 50), -20),
                                   ((38.5, 50), 20), ((0, 50), 100), ((40, 50), -200)])
def test_pre_takeoff_combat_never_holds_a_direction(pos, dx):
    b = brain(JUMP)
    for i in range(6):
        cmd = b.decide(see(pos, i * .1, [mob(dx)]))
        assert b.state == "hunting" and cmd.action == "attack"
        assert cmd.move_x == "none" and cmd.move_y == "none"
        assert cmd.face_x == ("left" if dx < 0 else "right")


def test_ordinary_combat_helper_still_walks_toward_a_far_target_and_stops_at_the_platform_end():
    b = brain(JUMP)
    assert b._combat(see((30, 50), 0, [mob(-300)]), Command.NONE, "left") == Command("left", "none", "attack")
    assert b._combat(see((0, 50), 0, [mob(-300)]), Command.NONE, "left") == Command(
        "none", "none", "attack", face_x="left")


def test_crowd_uses_the_ordinary_aoe_choice_without_moving():
    b = brain(JUMP)
    cmd = b.decide(see((25, 50), 0, [mob(-60), mob(-90), mob(70)]))
    assert cmd.action == "aoe" and cmd.move_x == "none" and b.state == "hunting"


# -- never while committed -------------------------------------------------------
def _jump_takeoff(b):
    run(b, [(30, 50)] * 3, 0)
    cmd, _ = run(b, [(30, 50)], .3)[0]
    assert cmd.action == "jump"
    return .3


def _to_jumping(b):
    return (30, 50), _jump_takeoff(b) + .05           # key issued, still standing on the takeoff floor


def _to_airborne(b):
    start = _jump_takeoff(b)
    run(b, FLIGHT[:2], start + .1)
    return (40, 38), start + .3


def _to_jump_landing(b):
    start = _jump_takeoff(b)
    run(b, FLIGHT[:5], start + .1)
    return (55, 40), start + .6                      # on the destination floor, landing not yet proven


def _to_jump_retry(b):
    start = _jump_takeoff(b)
    run(b, [(30, 50)] * 22, start + .1)               # never left the floor: flight times out
    return (30, 50), start + 2.3


def _to_drop_launched(b):
    run(b, [(70, 40)] * 3, 0)
    cmd, _ = run(b, [(70, 40)], .3)[0]
    assert cmd == Command("none", "down", "jump")
    return (70, 40), .5                              # key issued, still on the ledge


def _to_drop_falling(b):
    pos, now = _to_drop_launched(b)
    run(b, [(70, 45)], now)
    return (70, 50), now + .1


def _to_drop_landed(b):
    pos, now = _to_drop_launched(b)
    run(b, [(70, 45), (70, 55), (70, 60)], now)
    return (70, 60), now + .3                        # at the destination, dwell not yet served


def _to_climb_settling(b):
    run(b, [(60, 60)], 0)
    return (60, 60), .1


def _to_climbing(b):
    run(b, [(60, 60)] * 4, 0)
    return (60, 60), .4


def _to_ramping(b):
    run(b, [(63, 80)] * 4, 0)
    return (64, 79), .4


def _to_ramp_landing(b):
    run(b, [(63, 80)] * 4 + [(67, 76), (72, 71), (77, 66), (80, 63)], 0)
    return (80, 63), .8


def _to_returning(b):
    run(b, [(30, 50)], 0)
    assert b.circuit.request_return_to_start()
    return (30, 50), .1


def _to_climb_dwelling(b):
    run(b, [(60, 60)] * 4 + [(60, 55), (60, 48), (60, 42), (60, 40)], 0)
    return (60, 40), .8                              # on the top platform, dwell not yet served


def _to_climb_recovering(b):
    run(b, [(63, 60)] * 12, 0)                        # never under the rope: the attempt times out
    return (63, 60), 1.2


# (circuit, reach, circuit state there, positions that carry the transition on afterwards)
COMMITTED = [
    (JUMP, _to_jumping, "jumping", FLIGHT),
    (JUMP, _to_airborne, "airborne", FLIGHT[2:]),
    (JUMP, _to_jump_landing, "jump_landing", [(55, 40)] * 6),
    (JUMP, _to_jump_retry, "jump_retry", [(30, 50)] * 12),
    (DROP, _to_drop_launched, "dropping", [(70, 45), (70, 55)] + [(70, 60)] * 6),
    (DROP, _to_drop_falling, "dropping", [(70, 55)] + [(70, 60)] * 6),
    (DROP, _to_drop_landed, "dwelling", [(70, 60)] * 6),
    (CLIMB, _to_climb_settling, "settling", [(60, 60)] * 3 + [(60, 55), (60, 48), (60, 42)] + [(60, 40)] * 3),
    (CLIMB, _to_climbing, "climbing", [(60, 55), (60, 48), (60, 42)] + [(60, 40)] * 3),
    (CLIMB_DWELL, _to_climb_dwelling, "dwelling", [(60, 40)] * 14),
    (CLIMB_SLOW, _to_climb_recovering, "recovering", [(63, 60)] * 10),
    (RAMP, _to_ramping, "ramping", [(67, 76), (72, 71), (77, 66)] + [(80, 63)] * 6),
    (RAMP, _to_ramp_landing, "ramp_landing", [(80, 63)] * 6),
    (JUMP, _to_returning, None, [(30, 50)] * 5),
]
IDS = [reach.__name__[4:] for _, reach, _, _ in COMMITTED]
# In these the character stands on a drawn platform, so only the commit rule can refuse.
# (A landed drop is refused twice: by the rule and by the source-height check.)
ON_A_FLOOR = {"jumping", "jump_landing", "jump_retry", "drop_launched", "climb_settling",
              "climbing", "climb_dwelling", "climb_recovering", "ramp_landing"}


def _snapshot(nav):
    return (nav.idx, nav.state, nav.retries, nav.reason, nav.completed, nav.returning_to_start,
            round(nav._clock.time, 6))


@pytest.mark.parametrize("circuit,reach,state,onward", COMMITTED, ids=IDS)
def test_no_attack_once_the_transition_is_committed(circuit, reach, state, onward):
    b, reference = brain(circuit, pre=10), brain(circuit, pre=0)
    pos, now = reach(b)
    assert reach(reference) == (pos, now)
    if state is not None:
        assert b.circuit.state == state
    assert b.circuit.transition_active and b.circuit.transition_committed
    mobs, start, compared = [mob(-60), mob(60)], _snapshot(reference.circuit), 0
    for i, where in enumerate([pos] * 3 + onward):
        if not reference.circuit.transition_committed:
            break  # advanced, or back to a pre-takeoff phase where a fight is allowed
        at = round(now + i * .1, 6)
        cmd = b.decide(see(where, at, mobs))
        assert cmd.action not in ATTACKS and cmd.face_x == "none"
        assert b.state != "hunting" and b.state.startswith("circuit_") and b.target is None
        assert b._pre_spent == 0 and not b._pre_engaged
        # Monsters in view must leave the committed transition exactly as the key-off brain has it.
        assert cmd == reference.decide(see(where, at, mobs))
        assert _snapshot(b.circuit) == _snapshot(reference.circuit) and b.state == reference.state
        compared += 1
    assert compared >= 3
    if state is not None:
        # The comparison ran on into landing proof / the next phase, not three idle frames.
        assert _snapshot(reference.circuit)[:3] != start[:3]


@pytest.mark.parametrize("circuit,reach,state,onward", [c for c, name in zip(COMMITTED, IDS) if name in ON_A_FLOOR],
                         ids=[name for name in IDS if name in ON_A_FLOOR])
def test_the_commit_rule_is_what_refuses_those_attacks(monkeypatch, circuit, reach, state, onward):
    # Sensitivity check: with the rule knocked out the same frame would attack,
    # so the test above cannot pass by accident of position or range.
    b = brain(circuit, pre=10)
    pos, now = reach(b)
    monkeypatch.setattr(MinimapCircuit, "transition_committed", property(lambda self: False))
    assert b.decide(see(pos, now, [mob(-60), mob(60)])).action in ATTACKS


def test_returning_route_is_never_interrupted():
    b = brain(JUMP, pre=10)
    pos, now = _to_returning(b)
    for i in range(5):
        cmd = b.decide(see(pos, now + i * .1, [mob(-60)]))
        assert cmd.action not in ATTACKS and b.state != "hunting"


# -- budget ---------------------------------------------------------------------
def test_budget_exhaustion_restores_transition_priority_and_the_jump_then_takes_off():
    b = brain(JUMP, pre=1.0)
    out = run(b, [(30, 50)] * 40, 0, [mob(-60)])
    hunting = [i for i, (_, state) in enumerate(out) if state == "hunting"]
    assert hunting == list(range(len(hunting))) and 10 <= len(hunting) <= 12  # about one second of frames
    assert all(cmd.action == "attack" for cmd, _ in out[:len(hunting)])
    rest = out[len(hunting):]
    assert all(cmd.action not in ATTACKS and state.startswith("circuit_") for cmd, state in rest)
    jumps = [i for i, (cmd, _) in enumerate(rest) if cmd.action == "jump"]
    assert jumps and jumps[0] in (3, 4)  # a full settle was served after the last combat frame
    assert b.circuit.reason is None


def test_flickering_detection_cannot_stretch_the_budget():
    b = brain(JUMP, pre=1.0)
    attacks = 0
    for i in range(200):
        dets = [mob(-60)] if i % 2 == 0 else []
        # Off-floor frames keep the circuit from taking off, so only the budget can end the fight.
        cmd = b.decide(see((25, 50) if i % 2 == 0 else (25, 45), i * .1, dets))
        attacks += cmd.action in ATTACKS
    assert 5 <= attacks <= 12  # each frame is charged the time since the previous frame


def test_a_long_gap_is_charged_as_one_second():
    b = brain(JUMP, pre=2.0)
    assert b.decide(see((25, 50), 0, [mob(-60)])).action == "attack"
    assert b.decide(see((25, 50), 30, [mob(-60)])).action == "attack" and b._pre_spent == 1.0
    assert b.decide(see((25, 50), 60, [mob(-60)])).action == "attack" and b._pre_spent == 2.0
    assert b.decide(see((25, 50), 90, [mob(-60)])).action not in ATTACKS


def test_budget_is_per_waypoint_selection_and_resets_on_the_next_waypoint():
    b = brain(JUMP, pre=.5)
    mobs, now, hunted = [mob(-60)], 0.0, 0

    def until_jump(limit):
        nonlocal now
        states = []
        for _ in range(limit):
            cmd = b.decide(see((30, 50), now, mobs))
            states.append(b.state)
            now = round(now + .1, 6)
            if cmd.action == "jump":
                return states
        raise AssertionError("no takeoff")

    first = until_jump(30)
    assert 5 <= first.count("hunting") <= 7
    # The attempt fails on the spot: the same waypoint gets no new allowance.
    while b.circuit.retries == 0:
        assert b.decide(see((30, 50), now, mobs)).action not in ATTACKS and b.state != "hunting"
        now = round(now + .1, 6)
        assert now < 10
    assert "hunting" not in until_jump(40) and b.circuit.idx == 1
    for pos in FLIGHT:  # monsters in view all the way: committed, so never attacked
        if b.circuit.idx == 2:
            break
        assert b.decide(see(pos, now, mobs)).action not in ATTACKS and b.state != "hunting"
        now = round(now + .1, 6)
    assert b.circuit.idx == 2 and b._pre_spent >= .5  # landed; the old waypoint's allowance stayed spent
    cmd = b.decide(see((55, 40), now, [mob(60)]))
    assert cmd.action == "attack" and b.state == "hunting" and b._pre_spent <= .1 + 1e-9


def test_budget_resets_when_a_return_to_start_begins():
    b = brain(JUMP, pre=.3)
    run(b, [(25, 50)] * 8, 0, [mob(-60)])
    assert b._pre_spent >= .3
    assert b.circuit.request_return_to_start()
    b.decide(see((25, 50), .8, [mob(-60)]))
    assert b._pre_spent == 0 and b.state != "hunting"


# -- fresh alignment proof before takeoff ------------------------------------------
def test_jump_needs_a_fresh_settle_after_the_last_combat_frame():
    undisturbed = brain(JUMP)
    assert [c.action for c, _ in run(undisturbed, [(30, 50)] * 4, 0)] == ["none", "none", "none", "jump"]
    b = brain(JUMP)
    run(b, [(30, 50)] * 3, 0)
    assert b.circuit.state == "jump_settling"
    cmd = b.decide(see((30, 50), .3, [mob(-60)]))  # the frame that would have taken off
    assert cmd.action == "attack" and b.circuit._aligned_since is None and b.circuit._pulse_until is None
    held = run(b, [(30, 50)] * 3, .4)
    assert held == [(Command.NONE, "reacquiring_target")] * 3 and b.circuit._aligned_since is None
    after = run(b, [(30, 50)] * 4, .3 + QUIET)
    assert [c.action for c, _ in after] == ["none", "none", "none", "jump"]
    assert [state for _, state in after[:3]] == ["circuit_jump_settling"] * 3


def test_knockback_during_the_fight_is_realigned_before_the_jump():
    b = brain(JUMP)
    run(b, [(30, 50)] * 3, 0)
    b.decide(see((27, 50), .3, [mob(-60)]))  # hit and pushed 3 px while attacking
    out = run(b, [(27, 50)] * 8, .3 + QUIET)
    assert all(cmd.action != "jump" for cmd, _ in out)
    assert {state for _, state in out} == {"circuit_jump_aligning"}
    assert Command("right") in [cmd for cmd, _ in out]


def test_ramp_needs_a_fresh_settle_after_the_last_combat_frame():
    b = brain(RAMP)
    run(b, [(63, 80)] * 3, 0)
    assert b.circuit.state == "ramp_settling"
    assert b.decide(see((63, 80), .3, [mob(-60)])).action == "attack" and b.state == "hunting"
    assert not b.circuit._ramp_started
    out = run(b, [(63, 80)] * 4, .3 + QUIET)
    assert [cmd for cmd, _ in out] == [Command.NONE] * 3 + [Command("right")]
    assert b.circuit._ramp_started and b.circuit.transition_committed


def _failed_drop(b, knocked=(70, 40)):
    """Down+Jump pressed, the character stays on the ledge, and the circuit re-aligns."""
    run(b, [(70, 40)] * 3, 0)
    assert b.decide(see((70, 40), .3)) == Command("none", "down", "jump")
    now = .3
    while b.circuit._drop_launched_at is not None:
        now = round(now + .1, 6)
        assert b.decide(see(knocked, now, [mob(-60)])).action not in ATTACKS  # committed: no attack
        assert now < 3, "the failed drop was never recognised"  # a disturbed circuit fails, not hangs
    return now


def test_realigning_after_a_failed_drop_fights_and_then_needs_a_fresh_settle():
    b = brain(DROP)
    now = _failed_drop(b)
    assert b.circuit.retries == 1
    # One frame is stepped first so the circuit itself has seen the source floor again.
    cmd = b.decide(see((70, 40), now + .1, [mob(-60)]))
    assert cmd.action not in ATTACKS and b.state == "circuit_settling"
    fight = run(b, [(70, 40)] * 5, now + .2, [mob(-60)])
    assert all(cmd.action == "attack" and state == "hunting" for cmd, state in fight)
    after = run(b, [(70, 40)] * 4, now + .6 + QUIET)
    assert [cmd for cmd, _ in after] == [Command.NONE] * 3 + [Command("none", "down", "jump")]
    assert b.circuit.reason is None


def test_knocked_off_the_drop_ledge_mid_fight_is_never_treated_as_pre_takeoff():
    b = brain(DROP)
    now = _failed_drop(b)
    b.decide(see((70, 40), now + .1))
    assert b.decide(see((70, 40), now + .2, [mob(-60)])).action == "attack"
    # Next fresh frame: already standing on the lower floor. The circuit has not seen it yet.
    assert not b.circuit.transition_committed
    cmd = b.decide(see((70, 60), now + .3, [mob(-60)]))
    assert cmd == Command.NONE and b.state == "circuit_dwelling" and b.circuit.transition_committed


@pytest.mark.parametrize("y,fights", [(39, True), (40, True), (41, True), (38, False), (42, False)])
def test_drop_fight_needs_the_source_floor_height_within_one_pixel(y, fights):
    # 2 px off the ledge height is still "supported" for combat (tolerance_y 2), so
    # only the drop's own +/-1 px takeoff test stands between this frame and an attack.
    b = brain(DROP)
    now = _failed_drop(b)
    b.decide(see((70, 40), now + .1))
    assert b.circuit.state == "settling" and b._supported_combat_platform(see((70, y), 0)) is not None
    cmd = b.decide(see((70, y), now + .2, [mob(-60)]))
    assert (cmd.action in ATTACKS, b.state == "hunting") == (fights, fights)
    assert fights or (b.state.startswith("circuit_") and b._pre_spent == 0)


# -- timeouts ---------------------------------------------------------------------
def test_fight_time_does_not_count_toward_the_transition_timeout():
    b = brain(JUMP, pre=5.0, route={"circuit": {"timeout_s": 1.0}})
    fight = run(b, [(30, 50)] * 30, 0, [mob(-60)])  # three seconds, well past timeout_s
    assert all(state == "hunting" for _, state in fight)
    out = run(b, [(30, 50)] * 5, 2.9 + QUIET)
    assert out[-1][0].action == "jump" or any(cmd.action == "jump" for cmd, _ in out)
    assert b.circuit.retries == 0 and b.circuit.reason is None


def test_without_a_fight_the_same_timeout_still_fails_the_attempt():
    b = brain(JUMP, pre=5.0, route={"circuit": {"timeout_s": 1.0}})
    run(b, [(30, 45)] * 14, 0)  # never on the takeoff floor
    assert b.circuit.retries == 1


def test_drop_timeout_is_not_spent_by_the_fight():
    b = brain(DROP, pre=5.0, route={"circuit": {"timeout_s": 4.0}})
    now = _failed_drop(b)
    b.decide(see((70, 40), now + .1))
    fight = run(b, [(70, 40)] * 40, now + .2, [mob(-60)])
    assert all(state == "hunting" for _, state in fight)
    out = run(b, [(70, 40)] * 4, now + 4.1 + QUIET)
    assert out[-1][0] == Command("none", "down", "jump") and b.circuit.reason is None


def _bursts(b, pos, start, cycles):
    """Per 1.2 s: one quiet frame, 0.5 s of monster frames, 0.6 s without a detection.

    The circuit is stepped between the fights at gaps under a second, which is
    where its active clock would count fight time if the fight did not stop it.
    """
    hunted = 0
    for i in range(cycles * 12):
        dets = [mob(-60)] if 1 <= i % 12 <= 5 else []
        b.decide(see(pos, round(start + i * .1, 6), dets))
        hunted += b.state == "hunting"
    return hunted


def test_fights_interleaved_with_circuit_frames_do_not_run_the_jump_clock():
    b = brain(JUMP, pre=5.0, route={"circuit": {"timeout_s": 2.0}})
    assert _bursts(b, (25, 50), 0, 3) == 15  # 3.6 s of wall time lining up, well past timeout_s
    assert b.circuit.retries == 0 and b.circuit.reason is None and b.circuit.state == "jump_aligning"
    assert b.circuit._clock.time <= 1.0  # only the frames the circuit itself stepped


def test_fights_interleaved_with_circuit_frames_do_not_run_the_drop_clock():
    b = brain(DROP, pre=5.0, route={"circuit": {"timeout_s": 4.0}})
    now = _failed_drop(b)
    b.decide(see((64, 40), now + .1))  # knocked sideways on the ledge: re-aligning
    assert b.circuit.state == "aligning"
    before = b.circuit._clock.time
    assert _bursts(b, (64, 40), now + .2, 3) == 15
    assert b.circuit.reason is None and b.circuit.state == "aligning"
    assert b.circuit._clock.time - before <= 1.0


# -- a fight is a window, not isolated frames --------------------------------------------
@pytest.mark.parametrize("every", [1, 2, 3])
def test_intermittent_detections_do_not_burn_the_jump_timeout_or_stretch_the_fight(every):
    # About 7 fps with a qualifying detection on one frame in `every`, standing on the takeoff point.
    b = brain(JUMP, pre=4.0, route={"circuit": {"timeout_s": 3.0, "max_retries": 1}})
    combat, takeoff = [], None
    for i in range(200):
        now = round(i / 7, 6)
        cmd = b.decide(see((30, 50), now, [mob(40)] if i % every == 0 else []))
        assert b.circuit.retries == 0 and b.circuit.reason is None, now  # no timeout inside the fight
        if b.state == "hunting":
            assert cmd.action == "attack"
            combat.append(now)
        else:
            assert cmd.action not in ATTACKS
        if cmd.action == "jump":
            takeoff = now
            break
    assert takeoff is not None and len(combat) >= 4.0 * 7 / every - 2
    assert combat[-1] - combat[0] <= 4.0 + b._reacquire_s  # wall time of the fight is the allowance
    assert takeoff - combat[-1] <= b._reacquire_s + b.circuit.align_settle_s + 3 / 7
    assert b.circuit._clock.time <= 1.0


@pytest.mark.parametrize("every", [1, 2, 3])
def test_intermittent_detections_do_not_burn_the_drop_timeout_or_stretch_the_fight(every):
    b = brain(DROP, pre=4.0, route={"circuit": {"timeout_s": 5.0}})
    start = _failed_drop(b) + .1
    b.decide(see((70, 40), start))
    combat, takeoff = [], None
    for i in range(1, 200):
        now = round(start + i / 7, 6)
        cmd = b.decide(see((70, 40), now, [mob(40)] if (i - 1) % every == 0 else []))
        assert b.circuit.reason is None and b.circuit.state != "blocked", now  # no "drop timeout"
        if b.state == "hunting":
            combat.append(now)
        if cmd == Command("none", "down", "jump"):
            takeoff = now
            break
    assert takeoff is not None and len(combat) >= 4.0 * 7 / every - 2
    assert combat[-1] - combat[0] <= 4.0 + b._reacquire_s
    assert takeoff - combat[-1] <= b._reacquire_s + b.circuit.align_settle_s + 3 / 7


def test_fight_window_is_charged_on_frames_without_a_detection_too():
    b = brain(JUMP, pre=1.0)
    out = []
    for i in range(40):  # a detection on every third frame, 0.1 s apart
        out.append((b.decide(see((25, 50), round(i * .1, 6), [mob(-60)] if i % 3 == 0 else [])), b.state))
    fight = [i for i, (_, state) in enumerate(out) if state in ("hunting", "reacquiring_target")]
    assert fight == list(range(len(fight))) and 10 <= len(fight) <= 12  # one second of wall time
    assert all(state.startswith("circuit_") for _, state in out[len(fight):])
    assert sum(cmd.action in ATTACKS for cmd, _ in out) <= 4


# -- loot ---------------------------------------------------------------------------
def test_loot_is_not_tracked_or_started_by_pre_takeoff_combat():
    b = brain(JUMP, keys={"pickup": "x"}, loot={"enabled": True, "disappear_s": .1, "settle_s": .5})
    assert b.loot.enabled
    for i in range(5):
        assert b.decide(see((25, 50), i * .1, [mob(-60)])).action == "attack"
        assert b.loot.goal is None and b.loot.state == "idle" and b.loot.pending_count == 0
    for i in range(5, 25):  # the monster died: the route continues, no pickup visit
        b.decide(see((25, 50), i * .1))
        assert b.state.startswith("circuit_") or (b.state == "reacquiring_target" and i < 10)
        assert b.loot.goal is None and b.loot.state == "idle" and b.loot.pending_count == 0
    assert b.state.startswith("circuit_")
