"""Opt-in transition timing: commit_budget, takeoff_far_approach, recovery_far_approach.

Offline only. Real ``MinimapCircuit`` objects run against a small simulated
world built from the versioned Construction Site geometry
(examples/construction-site/user.yaml, read-only). Two failures of lap 3 of the
supervised 2026-11-16 run (trace log/claude-circuits-6.jsonl) are replayed:

1. waypoint 14 (ramp ``stair-2-finish``): 12.4 s of fine-tap alignment, then a
   progressing ramp cut by the waypoint clock about 2 px before its landing;
2. the return child's 3 px walk to the ``drop-right-3-2`` takeoff under
   knock-backs, ended by its 15 s waypoint timeout.

One ramp of the VERIFIED lap 1 of the same run is replayed too (waypoint 22):
the character stood 5 s on the first stair pixel with the key held and then
walked up. ``commit_budget`` must not end what the old clock let finish.

The world is deliberately simple: a held key walks at ``speed`` minimap px/s
after ``startup_s`` of each press (live fine taps moved about 0.16 px each),
slides ``coast`` px at key-up, and the circuit only sees the whole minimap
pixel. A leased command holds its key for at most ``valid_for_s``. This is not
the game and nothing here is a live verification.
"""
from collections import namedtuple
from copy import deepcopy
import hashlib
from itertools import product
import math
from pathlib import Path

import pytest
import yaml

from maplebot.nav.circuit import MinimapCircuit, Platform
from maplebot.nav.recovery import plan_return_to_start
from maplebot.types import Command

SITE = Path(__file__).resolve().parents[1] / "examples" / "construction-site" / "user.yaml"
_SITE_CIRCUIT = yaml.safe_load(SITE.read_text(encoding="utf-8"))["route"]["circuit"]
# The recommended far-approach values (tests/test_approach_efficiency.py): the walk
# speed is an UPPER bound on the measured 9.2-9.8 px/s.
FAR = {"approach_fine_radius_x": 1.5, "approach_walk_speed_x": 11.0, "approach_stop_margin_x": 1.0,
       "approach_edge_margin_x": 1.75, "approach_far_pulse_max_s": .45, "approach_far_release_s": .3}
A = {"commit_budget": True}
B = {**FAR, "takeoff_far_approach": True}
C = {**FAR, "recovery_far_approach": True}
ALL = {**A, **B, **C}
NEW_KEYS = ("commit_budget", "takeoff_far_approach", "recovery_far_approach")
SPEEDS = (8.5, 9.5, 10.5)
# Key-up slide. A fine tap plus its slide must stay under one pixel, or this model's
# exact-pixel takeoffs (tolerance 0.4) hop over their pixel for ever (10.5 px/s with 0.6).
COASTS = (0.0, .3, .5)
FRAMES = (.08, .12, .16)  # 12.5, 8.3 and 6.25 frames per second
Frame = namedtuple("Frame", "t x y pos cmd state idx committed")


def site(**overrides):
    return {**deepcopy(_SITE_CIRCUIT), **overrides}


def waypoint(connection_id):
    return next(i for i, p in enumerate(_SITE_CIRCUIT["waypoints"]) if p.get("connection_id") == connection_id)


RAMP = waypoint("stair-2-finish")      # 14: stair-2 x=82 -> floor-3-right x=92
GAP = waypoint("floor-3-gap")          # 17: floor-3-right x=91 -> floor-3 x=85
STAIR_JUMP = waypoint("jump-3a")       # 20: floor-3 x=66 -> stair-3 x=69
RAMP3 = waypoint("stair-3-finish")     # 22: stair-3 x=82 -> floor-4-right x=92
BARRICADE = waypoint("barricade-jump")  # 28: floor-4 x=59 -> barricade x=63
DROP = waypoint("drop-4-3")            # 35: floor-4 x=60 -> floor-3
LADDER = waypoint("base-ladder")       # 1: ground x=49 -> floor-1
TOP_LADDER = waypoint("top-ladder")    # 30: barricade x=63 -> top


class World:
    """Walks, falls, jumps, drops and climbs over the drawn map; reports whole pixels."""
    AIR, JUMP_VY, GRAVITY, CLIMB, TICK = 10.0, -24.0, 72.0, 8.0, 0.01

    def __init__(self, cfg, pos, *, speed=9.5, coast=0.0, startup_s=0.0, slope_factor=1.0):
        plan = cfg["editor_plan"]
        self.platforms = [Platform(p["y"], p["left"], p["right"], p.get("right_y"), p.get("knots", ()))
                          for p in plan["platforms"]]
        by_id = dict(zip((p["id"] for p in plan["platforms"]), self.platforms))
        self.ropes = sorted({(c["from_x"], *sorted((by_id[c["from_platform"]].y_at(c["from_x"]),
                                                    by_id[c["to_platform"]].y_at(c["to_x"]))))
                             for c in plan["connections"] if c["kind"] == "rope"})
        self.speed, self.coast, self.startup_s, self.slope_factor = speed, coast, startup_s, slope_factor
        self.frozen = False       # True: nothing moves the character
        self.rope_frozen = False  # True: Up/Down on a rope move nothing
        self.takeoffs = self.drops = 0
        self.put(pos)
        self.low = self.high = self.x

    def put(self, pos):
        """Place the character (a knock-back): on the floor under ``pos`` when there is one."""
        self.x, self.y = map(float, pos)
        self.vy, self.air, self.rope = 0.0, False, None
        self._held, self._held_for, self._moved = None, 0.0, False
        ground = self._ground(1.0)
        if ground is not None:
            self.y = ground[0]

    def _ground(self, reach=.6):
        floors = [(p.y_at(self.x), p) for p in self.platforms
                  if p.left <= self.x <= p.right and abs(p.y_at(self.x) - self.y) <= reach]
        return min(floors, key=lambda item: abs(item[0] - self.y)) if floors else None

    def _walk(self, dx):
        ground = self._ground()
        if ground is not None and ground[1].sloped:
            dx *= self.slope_factor
        steps = max(1, math.ceil(abs(dx) / .1))  # follow a steep stair in small steps
        for _ in range(steps):
            self.x += dx / steps
            ground = self._ground()
            if ground is None:
                self.air, self.vy = True, 0.0
                return
            self.y = ground[0]

    def advance(self, cmd, dt):
        if self.frozen:
            self._held = None
            return
        held_s = dt if cmd.valid_for_s is None else min(dt, max(0.0, cmd.valid_for_s))
        for tick in range(round(dt / self.TICK)):
            want = ({"left": -1, "right": 1}.get(cmd.move_x) if tick * self.TICK < held_s - 1e-9 else None)
            if want != self._held:
                if self._held is not None and self._moved and not self.air and self.rope is None and self.coast:
                    self._walk(self.coast * self._held)  # key-up slide
                self._held, self._held_for, self._moved = want, 0.0, False
            sign = 0
            if want is not None:
                if self._held_for >= self.startup_s - 1e-9:
                    sign, self._moved = want, True
                self._held_for += self.TICK
            self.low, self.high = min(self.low, self.x), max(self.high, self.x)
            if self.rope is not None:
                _, top, bottom = self.rope
                if not self.rope_frozen:
                    self.y += {"up": -1, "down": 1}.get(cmd.move_y, 0) * self.CLIMB * self.TICK
                if self.y <= top or self.y >= bottom:
                    self.y, self.rope = (top if self.y <= top else bottom), None
                continue
            if not self.air:
                rope = next((r for r in self.ropes if abs(r[0] - self.x) <= 1.0), None)
                if (tick == 0 and rope and cmd.move_y == "up" and cmd.action == "jump"
                        and abs(self.y - rope[2]) <= .6):
                    self.rope, self.x, self.y = rope, float(rope[0]), self.y - .7
                    continue
                if rope and cmd.move_y == "down" and cmd.action != "jump" and abs(self.y - rope[1]) <= .6:
                    self.rope, self.x, self.y = rope, float(rope[0]), self.y + .7
                    continue
                if tick == 0 and cmd.action == "jump" and cmd.move_y == "down":
                    self.air, self.vy = True, 0.0  # fall through the ledge
                    self.y += .7
                    self.drops += 1
                elif tick == 0 and cmd.action == "jump":
                    self.air, self.vy = True, self.JUMP_VY
                    self.takeoffs += 1
                else:
                    if sign:
                        self._walk(sign * self.speed * self.TICK)
                    continue
            self.x += sign * self.AIR * self.TICK
            previous = self.y
            self.vy += self.GRAVITY * self.TICK
            self.y += self.vy * self.TICK
            if self.vy > 0:
                floors = [p.y_at(self.x) for p in self.platforms
                          if p.left <= self.x <= p.right and previous <= p.y_at(self.x) <= self.y]
                if floors:
                    self.y, self.vy, self.air = min(floors), 0.0, False
        self.low, self.high = min(self.low, self.x), max(self.high, self.x)

    @property
    def pos(self):
        return math.floor(self.x + .5), math.floor(self.y + .5)


def drive(nav, world, *, limit_s, dt=.12, stop=None, events=(), suspend=(), dropped=(), t0=0.0):
    """Step ``nav`` on every frame for ``limit_s`` seconds from ``t0``.

    ``events`` are (time, fn(world)). ``suspend`` spans belong to another owner
    (combat, HP protection). ``dropped`` spans have no vision frame at all: the
    keyboard only serves what is left of the last command's lease.
    """
    log, pending = [], sorted(events, key=lambda event: event[0])
    last = None
    for frame in range(int(limit_s / dt) + 1):
        t = round(t0 + frame * dt, 6)
        while pending and pending[0][0] <= t + 1e-9:
            pending.pop(0)[1](world)
        if last is not None and any(a <= t < b for a, b in dropped):
            left = (.75 if last[1].valid_for_s is None else last[1].valid_for_s) - (t - last[0])
            world.advance(Command(last[1].move_x, last[1].move_y, valid_for_s=left)
                          if left > 1e-9 and last[1].action == "none" else Command.NONE, dt)
            continue
        if any(a <= t < b for a, b in suspend):
            nav.suspend(t)  # combat or HP protection owns the frame: no route keys
            cmd = Command.NONE
        else:
            cmd = nav.step(world.pos, t)
        log.append(Frame(t, world.x, world.y, world.pos, cmd, nav.state, nav.idx, nav.transition_committed))
        last = (t, cmd)
        if stop is not None and stop(nav):
            break
        world.advance(cmd, dt)
    return log


def digest(log):
    rows = [(f.pos, f.cmd.move_x, f.cmd.move_y, f.cmd.action,
             None if f.cmd.valid_for_s is None else round(f.cmd.valid_for_s, 6), f.state, f.idx, f.committed)
            for f in log]
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


def first(log, predicate):
    return next((f for f in log if predicate(f)), None)


def takeoff_frame(log):
    """The first frame that sends a takeoff key (Jump, Down+Jump) or starts a ramp/rope."""
    return first(log, lambda f: f.cmd.action == "jump" or f.state in ("ramping", "ramp_landing"))


def on_time(t, expected, dt=.1):
    """Not before ``expected`` and no later than the next frame."""
    return expected - 1e-6 <= t <= expected + dt + 1e-6


def max_lease(log):
    return max((f.cmd.valid_for_s or 0) for f in log if f.cmd.move_x != "none")


def run(idx, start, keys=None, *, limit_s=40.0, world=None, stop=None, cls=MinimapCircuit, **drive_options):
    """One waypoint of the lap from ``start`` until it is left or blocked (no return on block)."""
    cfg = site(start_idx=idx, return_on_block=False, **(keys or {}))
    nav, world = cls(cfg), World(cfg, start, **(world or {}))
    log = drive(nav, world, limit_s=limit_s, stop=stop or (lambda n: n.idx != idx or n.state == "blocked"),
                **drive_options)
    return nav, world, log


def script(nav, positions, *, dt=.1, t0=0.0):
    """Feed fixed positions (None = no position, "suspend" = another owner's frame)."""
    log = []
    for frame, pos in enumerate(positions):
        t = round(t0 + frame * dt, 6)
        if pos == "suspend":
            nav.suspend(t)
            cmd = Command.NONE
        else:
            cmd = nav.step(pos, t)
        log.append(Frame(t, None, None, pos, cmd, nav.state, nav.idx, nav.transition_committed))
    return log


def lap_nav(idx, keys=None, **overrides):
    return MinimapCircuit(site(start_idx=idx, return_on_block=False, **(keys or {}), **overrides))


def _returning(cls, keys, world_options, limit_s, start=(100, 55), idx=RAMP + 1, events=None, dt=.12):
    cfg = site(start_idx=idx, **keys)
    nav, world = cls(cfg), World(cfg, start, **world_options)
    assert nav.request_return_to_start()
    log = drive(nav, world, limit_s=limit_s, dt=dt, events=events(nav) if events else (),
                stop=lambda n: n.state == "returned_to_start" or n.recovery_terminal)
    return nav, world, log


def scenario(name, cls=MinimapCircuit, **keys):
    """Closed-loop reference runs. ``cls`` lets a saved copy of the old class run the same world."""
    slow = {"startup_s": .02}
    if name == "ramp":
        return run(RAMP, (78, 65), keys, world=slow, limit_s=14, cls=cls, stop=lambda n: n.idx == RAMP + 2)[2]
    if name == "ramp_blocked_then_return":  # nothing moves: ramp timeout, then the return's own failure
        cfg = site(start_idx=RAMP, **keys)
        nav, world = cls(cfg), World(cfg, (78, 65))
        world.frozen = True
        return drive(nav, world, limit_s=40, stop=lambda n: n.recovery_terminal)
    if name == "gap_jump":
        return run(GAP, (96, 55), keys, world=slow, limit_s=24, cls=cls, stop=lambda n: n.idx == GAP + 2)[2]
    if name == "stair_jump":
        return run(STAIR_JUMP, (72, 55), keys, limit_s=14, cls=cls, stop=lambda n: n.idx == STAIR_JUMP + 2)[2]
    if name == "drop":
        return run(DROP, (65, 38), keys, world=slow, limit_s=18, cls=cls, stop=lambda n: n.idx == DROP + 2)[2]
    if name == "climb":  # 14.8 s of alignment: the rope is entered just before the 15 s mark
        return run(LADDER, (54, 105), keys, world=slow, limit_s=22, cls=cls, stop=lambda n: n.idx == LADDER + 2)[2]
    if name == "return":
        return _returning(cls, keys, {}, 40)[2]
    if name == "rejoin":  # waypoint 10 expects floor-2; the character stands across the gap
        cfg = site(start_idx=GAP - 7, arrive_on_destination=True, platform_rejoin=True, **keys)
        nav, world = cls(cfg), World(cfg, (95, 71))
        return drive(nav, world, limit_s=20, stop=lambda n: n.idx == GAP - 5)
    raise KeyError(name)


# (waypoint, takeoff X, alignment tolerance, source platform id, sides to start from)
TAKEOFFS = {
    "ramp stair-2 x82": (RAMP, 82, 1.0, "stair-2", (-1, 1)),
    "gap jump x91 toward the gap": (GAP, 91, .4, "floor-3-right", (1,)),
    "gap jump x91 from the edge side": (GAP, 91, .4, "floor-3-right", (-1,)),
    "stair jump x66": (STAIR_JUMP, 66, .4, "floor-3", (-1, 1)),
    "barricade jump x59": (BARRICADE, 59, .4, "floor-4", (-1, 1)),
    "drop x60 floor-4": (DROP, 60, 1.0, "floor-4", (-1, 1)),
    "rope x49 ground": (LADDER, 49, .4, "ground", (-1, 1)),
    "rope x63 barricade": (TOP_LADDER, 63, .4, "barricade", (-1, 1)),
}
DISTANCES = (1.25, 2.0, 3.0, 4.5, 6.0, 7.0)


def _taken_off(nav):
    return (nav._jump_phase == "flight" or nav._ramp_started or nav._drop_launched_at is not None
            or nav._climb_jump is not None or nav.state == "blocked")


def _surface(platform_id):
    item = next(p for p in _SITE_CIRCUIT["editor_plan"]["platforms"] if p["id"] == platform_id)
    return Platform(item["y"], item["left"], item["right"], item.get("right_y"), item.get("knots", ()))


def takeoff_sweep(name, keys, *, dropped=None):
    """Every start/speed/slide/frame-rate case up to the takeoff key.

    Returns (slowest, mean, furthest true travel beyond the takeoff X, least
    distance kept from the platform end ahead, runs); asserts the unchanged
    takeoff proof on every run.
    """
    idx, target, tolerance, platform_id, sides = TAKEOFFS[name]
    surface = _surface(platform_id)
    times, beyond, clear = [], -math.inf, math.inf
    for side in sides:
        for distance in DISTANCES:
            start_x = target + side * distance
            if not surface.left + .1 <= start_x <= surface.right - .1:
                start_x = surface.left + .1 if side < 0 else surface.right - .1
            for speed in SPEEDS:
                for coast in COASTS:
                    for dt in FRAMES:
                        cfg = site(start_idx=idx, return_on_block=False, timeout_s=60, **keys)
                        nav = MinimapCircuit(cfg)
                        world = World(cfg, (start_x, surface.y_at(start_x)), speed=speed, coast=coast)
                        log = drive(nav, world, limit_s=60, dt=dt, stop=_taken_off,
                                    dropped=((dropped, dropped + .5),) if dropped else ())
                        case = (name, start_x, speed, coast, dt)
                        assert nav.state != "blocked" and _taken_off(nav), case
                        key = log[-1]
                        # Unchanged proof: the takeoff pixel is inside the kind's own tolerance and the
                        # whole settle before the key was spent aligned, on stable height, without keys.
                        assert abs(key.pos[0] - target) <= tolerance, case
                        settled = key.t
                        for f in reversed(log[:-1]):
                            if (abs(f.pos[0] - target) > tolerance or f.cmd.move_x != "none"
                                    or f.pos[1] != key.pos[1]):
                                break
                            settled = f.t
                        assert key.t - settled >= .25 - 1e-9, case
                        # Every correction inside the approach radius is a lease the keyboard ends itself.
                        near = [f.cmd for f in log[:-1] if f.cmd.move_x != "none" and abs(f.pos[0] - target) <= 6]
                        assert all(c.valid_for_s is not None and c.valid_for_s <= .45 + 1e-9 for c in near), case
                        # ...and the character never left the source platform's drawn span.
                        assert surface.left <= world.low and world.high <= surface.right, case
                        times.append(key.t)
                        beyond = max(beyond, world.high - target if side < 0 else target - world.low)
                        clear = min(clear, surface.right - world.high if side < 0 else world.low - surface.left)
    return max(times), sum(times) / len(times), beyond, clear, len(times)


# --------------------------------------------------------------------------
# 1. Keys: default off, validated, and off means the behaviour before this change
# --------------------------------------------------------------------------

def test_new_keys_default_off_and_budgets_are_not_read():
    nav = MinimapCircuit(site())
    assert (nav.commit_budget, nav.takeoff_far_approach, nav.recovery_far_approach) == (False, False, False)
    assert not hasattr(nav, "ramp_travel_timeout_s")  # budgets are not even read while off
    far = MinimapCircuit(site(**FAR, approach_resume_credit=True))
    assert not far.takeoff_far_approach and not far.recovery_far_approach and far.approach_resume_credit
    on = MinimapCircuit(site(**ALL))
    # A ramp and a rope get the waypoint's own timeout_s (15 s here) from their takeoff.
    assert (on.ramp_travel_timeout_s, on.drop_fall_timeout_s, on.climb_travel_timeout_s,
            on.landing_confirm_timeout_s, on.transition_stall_s) == (15.0, 6.0, 15.0, 3.0, 2.0)


@pytest.mark.parametrize("timeout_s, ramp, climb", [(1.5, 2.0, 2.0), (12, 12.0, 12.0), (45, 30.0, 45.0),
                                                    (61, 30.0, 60.0), (300, 30.0, 60.0)])
def test_budget_defaults_follow_timeout_s_and_never_reject_a_timeout_that_is_valid_without_the_key(
        timeout_s, ramp, climb):
    assert MinimapCircuit(site(timeout_s=timeout_s)).timeout_s == timeout_s
    nav = MinimapCircuit(site(commit_budget=True, timeout_s=timeout_s))  # the default is clamped, not refused
    assert (nav.timeout_s, nav.ramp_travel_timeout_s, nav.climb_travel_timeout_s) == (timeout_s, ramp, climb)
    # A configured value is still range-checked, under its own name.
    for key, bad in (("ramp_travel_timeout_s", 31), ("climb_travel_timeout_s", 61), ("climb_travel_timeout_s", 1.5)):
        with pytest.raises(ValueError, match=key):
            MinimapCircuit(site(commit_budget=True, timeout_s=timeout_s, **{key: bad}))


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf])
def test_invalid_timeout_s_is_reported_as_itself_with_commit_budget(bad):
    for keys in ({}, A):
        with pytest.raises(ValueError, match="tolerances/timers must be positive"):
            MinimapCircuit(site(timeout_s=bad, **keys))


@pytest.mark.parametrize("key", NEW_KEYS)
@pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0])
def test_new_keys_must_be_booleans(key, bad):
    with pytest.raises(ValueError, match=key):
        MinimapCircuit(site(**FAR, **{key: bad}))


@pytest.mark.parametrize("key", ["takeoff_far_approach", "recovery_far_approach"])
def test_far_keys_need_the_far_approach_configuration(key):
    with pytest.raises(ValueError, match=key):
        MinimapCircuit(site(**{key: True}))
    with pytest.raises(ValueError):  # the existing far validation still applies
        MinimapCircuit(site(**{**FAR, "approach_edge_margin_x": 1.0, key: True}))
    MinimapCircuit(site(**{key: False}))  # false needs nothing


@pytest.mark.parametrize("key, low, high", [("ramp_travel_timeout_s", 2, 30), ("drop_fall_timeout_s", 2, 30),
                                            ("climb_travel_timeout_s", 2, 60),
                                            ("landing_confirm_timeout_s", 1, 10), ("transition_stall_s", 1, 5)])
def test_commit_budgets_are_validated_only_when_the_key_is_on(key, low, high):
    for bad in (low - .01, high + .01, math.nan, math.inf, True, "3", None):
        with pytest.raises(ValueError, match=key):
            MinimapCircuit(site(commit_budget=True, **{key: bad}))
        MinimapCircuit(site(**{key: bad}))  # not read while commit_budget is off
    for good in (low, high):
        assert getattr(MinimapCircuit(site(commit_budget=True, **{key: good})), key) == good


# Digests of the command/state/index/transition_committed sequence each scenario produced
# with maplebot/nav/circuit.py as it was BEFORE these keys existed: on 2026-11-16 the old
# class was run through this same ``scenario`` function and world. They pin "off = unchanged".
PINNED = {
    "ramp": "7d4ef62d2b767b4a",
    "ramp_blocked_then_return": "c4c88df617f01ba1",
    "gap_jump": "3b0c8363ba06066b",
    "stair_jump": "27bec15a64fe332c",
    "drop": "f0f6ed041c0ed7dd",
    "climb": "3c9ea4c1a58971c3",
    "return": "51a5ddfbbbf45d6f",
    "rejoin": "a1ee9bf2490ed5bf",
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_keys_absent_or_false_reproduce_the_pinned_sequences(name):
    absent = scenario(name)
    assert len(absent) > 100 and digest(absent) == PINNED[name]
    assert digest(scenario(name, **{key: False for key in NEW_KEYS})) == PINNED[name]


@pytest.mark.parametrize("name", ["ramp", "stair_jump", "drop", "return", "rejoin", "ramp_blocked_then_return"])
def test_commit_budget_changes_nothing_when_no_takeoff_meets_an_expired_clock(name):
    # Same commands, states, indices and transition_committed flags, frame for frame.
    assert digest(scenario(name, commit_budget=True)) == PINNED[name]


@pytest.mark.parametrize("name", ["ramp", "gap_jump", "drop", "climb"])
def test_far_keys_alone_still_leave_every_takeoff_on_fine_taps(name):
    # approach_fine_radius_x without takeoff_far_approach: the waypoint under test is unchanged.
    log = [f for f in scenario(name, **FAR) if f.state.endswith("aligning")]
    assert log and max_lease(log) == pytest.approx(.04)


# --------------------------------------------------------------------------
# 2. Recorded ramps: lap 3 (alignment spent the clock, cut before landing) and lap 1 (5 s stand)
# --------------------------------------------------------------------------

# Status samples of log/claude-circuits-6.jsonl (lap 3, waypoint 14), elapsed seconds and
# canonical minimap pixel. "h": the frame belonged to HP protection (the circuit was not
# stepped). "i": interpolated; status is sampled about three times a second while the
# circuit saw 6-12 frames, and the ramp demonstrably started between 440.27 and 440.47.
LAP3_RAMP = """
427.16 81 64|427.47 79 65|427.77 78 65|428.18 78 65|428.48 78 65|428.79 78 65|429.09 78 65|429.50 78 65
429.80 78 65|430.11 78 65|430.41 78 65|430.82 78 65|431.13 78 65|431.43 78 65|431.83 78 65|432.14 78 65
432.44 78 65|432.75 79 65|433.16 79 65|433.46 79 65|433.77 79 65|434.07 79 65|434.48 79 65|434.78 79 65
435.09 79 65|435.19 79 65|435.49 79 65|435.80 80 65|436.10 80 65|436.51 80 65|436.82 80 65|437.02 80 65
437.12 80 65|437.52 80 65|437.73 80 65|437.83 80 65|438.23 80 65|438.44 80 65|438.54 80 65|438.85 80 65
439.15 80 65|439.25 80 65|439.56 80 65|439.76 80 65|439.96 81 65|440.12 81 64 i|440.27 81 64|440.40 81 64 i
440.47 81 64 h|440.78 81 64 h|440.88 82 64 h|441.19 82 64|441.29 82 64|441.59 82 63|441.90 84 60|442.30 87 58
442.61 89 55|443.02 90 54
"""
# What the character would have done next (it was walking right at 4-7 px/s): not recorded.
LAP3_CONTINUATION = "443.32 91 55|443.63 92 55|443.93 92 55|444.23 92 55|444.54 92 55"


def replay_ramp(keys, text=LAP3_RAMP, start=RAMP, ramp=RAMP):
    """Step a lap circuit from waypoint ``start`` through recorded samples until ``ramp`` is left."""
    nav, log = lap_nav(start, keys), []
    for row in text.replace("\n", "|").split("|"):
        if not row.strip():
            continue
        t, x, y, *kind = row.split()
        if kind == ["h"]:
            nav.suspend(float(t))
        else:
            nav.step((int(x), int(y)), float(t))
        log.append((float(t), (int(x), int(y)), nav.state))
        if nav.state == "blocked" or nav.idx > ramp:
            break
    return nav, log


def test_lap3_trace_reproduces_the_ramp_cut_two_pixels_before_its_landing():
    nav, log = replay_ramp({})
    aligning = [t for t, _, state in log if state == "ramp_aligning"]
    assert aligning[-1] - aligning[0] > 12  # 12.3 s of one-tap-per-cycle alignment for 3 px
    assert [state for _, _, state in log[-4:-1]] == ["ramping"] * 3 and nav._ramp_started
    assert log[-1] == (443.02, (90, 54), "blocked") and nav.reason == "ramp timeout"  # as observed live


def test_lap3_trace_with_commit_budget_lets_the_ramp_land_and_proves_the_landing():
    nav, log = replay_ramp(A, LAP3_RAMP + LAP3_CONTINUATION)
    states = [state for _, _, state in log]
    assert "blocked" not in states and dict((t, s) for t, _, s in log)[443.02] == "ramping"
    assert states[-3:] == ["ramp_landing"] * 3  # three stable supported frames over the 0.6 s dwell
    assert nav.idx == RAMP + 1 and nav.reason is None
    # The pre-takeoff bound is untouched: the same trace frozen in alignment still times out at 15 s.
    stuck = "|".join(f"{427.16 + .3 * i:.2f} 78 65" for i in range(60))
    nav, log = replay_ramp(A, stuck)
    assert nav.reason == "ramp timeout" and 15 <= log[-1][0] - 427.16 < 15.31


# Status samples of the same file, lap 1 (VERIFIED live), waypoints 21-22: the walk onto
# stair-3 and the ramp ``stair-3-finish``. Right was held from 121.67 s on. The character
# stood on (82,47) from 122.38 to 127.45 s: 5.07 s without one pixel of progress, every frame
# stepped, HP constant, the on-screen position constant too. The cause is not known. It then
# walked up and the landing was proven at 129.78 s.
LAP1_RAMP = """
120.66 80 49|120.96 80 48|121.26 81 48|121.37 81 48|121.67 81 48|122.08 81 48|122.38 82 47|122.69 82 47
122.99 82 47|123.29 82 47|123.70 82 47|124.01 82 47|124.31 82 47|124.71 82 47|125.02 82 47|125.32 82 47
125.42 82 47|125.73 82 47|126.03 82 47|126.34 82 47|126.74 82 47|126.94 82 47|127.05 82 47|127.45 82 47
127.75 84 44|128.06 86 43|128.47 88 40|128.77 90 38|129.07 91 38|129.18 94 38|129.48 95 38|129.78 95 38
"""


def lap1_ramp(keys, text=LAP1_RAMP):
    return replay_ramp(keys, text, start=RAMP3 - 1, ramp=RAMP3)


def test_lap1_trace_with_five_seconds_standing_on_the_stair_lands_with_and_without_commit_budget():
    off, expected = lap1_ramp({})
    assert off.idx == RAMP3 + 1 and expected[-1][0] == 129.78  # as observed live
    standing = [t for t, pos, state in expected if pos == (82, 47) and state == "ramping"]
    assert standing[-1] - standing[0] > 5  # the key was held and no pixel was gained
    # A ramp has no no-progress rule (no value of transition_stall_s would have let this lap
    # pass), and its budget is not shorter than what the waypoint clock had left.
    for keys in (A, ALL, {**A, "transition_stall_s": 1}, {**A, "transition_stall_s": 5}):
        nav, log = lap1_ramp(keys)
        assert log == expected and nav.idx == RAMP3 + 1 and nav.reason is None, keys


def test_ramp_that_starts_at_once_keeps_at_least_the_time_the_waypoint_clock_gave_it():
    # The same stand lasting 6 s longer. The landing is proven 14.5 s after the waypoint was
    # selected, inside the unchanged 15 s clock, so the key must not turn it into a block.
    rows = [row.split() for row in LAP1_RAMP.replace("\n", "|").split("|") if row.strip()]
    longer = ([(float(t), x, y) for t, x, y in rows if float(t) <= 127.45]
              + [(127.45 + .3 * k, "82", "47") for k in range(1, 21)]
              + [(float(t) + 6.0, x, y) for t, x, y in rows if float(t) > 127.45])
    text = "|".join(f"{t:.2f} {x} {y}" for t, x, y in longer)
    off, expected = lap1_ramp({}, text)
    assert off.idx == RAMP3 + 1 and expected[-1][0] == 135.78 and 135.78 - 121.26 < 15
    nav, log = lap1_ramp(A, text)
    assert log == expected and nav.reason is None and nav.ramp_travel_timeout_s == nav.timeout_s == 15.0
    # A configured shorter budget is honoured, measured from the takeoff (between 121.37 and 121.67).
    nav, log = lap1_ramp({**A, "ramp_travel_timeout_s": 8}, text)
    assert nav.reason == "ramp timeout" and 121.37 + 8 <= log[-1][0] <= 121.67 + 8 + .31


# Fine taps of about 0.19 px (live: about 0.16 px) and a slope walked at 4.3 px/s (live:
# 82 -> 92 in about 2.2 s) reproduce the live timing: about 12.4 s of alignment from x=78.8.
SLOW_STAIR = {"startup_s": .02, "slope_factor": .45}


def test_closed_loop_ramp_after_twelve_seconds_of_alignment_is_cut_only_without_commit_budget():
    nav, world, log = run(RAMP, (78.8, 65), {}, world=SLOW_STAIR)
    takeoff = first(log, lambda f: f.state == "ramping")
    assert 12 < takeoff.t < 13 and max_lease(log[:log.index(takeoff)]) == pytest.approx(.04)
    assert nav.reason == "ramp timeout" and log[-1].t == pytest.approx(15.0)
    assert nav._ramp_started and world.x > 90  # a progressing ramp, about a pixel from x=92
    progress = [f.pos[0] for f in log if f.state == "ramping"]
    assert progress == sorted(progress) and progress[-1] - progress[0] >= 8

    nav, world, log = run(RAMP, (78.8, 65), A, world=SLOW_STAIR)
    assert nav.idx == RAMP + 1 and nav.reason is None
    assert first(log, lambda f: f.state == "ramping").t == takeoff.t  # nothing before the takeoff changed
    landing = [f for f in log if f.state == "ramp_landing"]
    # Landing proof as always: three fresh frames, stable height, strict support on floor-3-right.
    assert len(landing) >= 3 and landing[-1].t - landing[0].t >= .6 - 1e-9
    assert {f.pos[1] for f in landing} == {55} and all(89 <= f.pos[0] <= 93 for f in landing)
    assert log[-1].t < takeoff.t + 8.0


@pytest.mark.parametrize("dt", FRAMES)
def test_takeoff_far_approach_alone_ends_the_long_alignment_of_failure_one(dt):
    slowest = 0.0
    for speed in SPEEDS:
        for coast in COASTS:
            for start in (78.0, 78.8):
                for dropped in ((), ((.5, 1.0),)):
                    model = {**SLOW_STAIR, "speed": speed, "coast": coast}
                    nav, world, log = run(RAMP, (start, 65), B, world=model, dt=dt, dropped=dropped,
                                          stop=lambda n: n._ramp_started or n.state == "blocked")
                    assert nav._ramp_started and world.high <= 83.0, (speed, coast, start)
                    slowest = max(slowest, log[-1].t)
                    # ...and the ramp then finishes far inside the waypoint clock, without commit_budget.
                    nav, world, log = run(RAMP, (start, 65), B, world=model, dt=dt, dropped=dropped)
                    assert nav.idx == RAMP + 1 and log[-1].t < 10, (speed, coast, start)
    print(f"\nfailure-1 start with takeoff_far_approach at {1 / dt:.1f} fps: slowest takeoff {slowest:.2f} s")
    assert slowest < 5.0  # 12.4 s with fine taps in the same model


# --------------------------------------------------------------------------
# 3. commit_budget: every committed transition still ends within its own bound
# --------------------------------------------------------------------------

RAMP8 = {**A, "ramp_travel_timeout_s": 8}  # a configured ramp budget shorter than timeout_s


def _pinned_ramp(keys):
    cfg = site(start_idx=RAMP, return_on_block=False, **keys)
    nav, world = MinimapCircuit(cfg), World(cfg, (81, 64.5))
    log = drive(nav, world, limit_s=5, stop=lambda n: n._ramp_started and world.x >= 85)
    world.frozen = True  # pinned on the slope: strictly supported, inside the corridor
    log += drive(nav, world, limit_s=40, t0=log[-1].t + .12, stop=lambda n: n.state == "blocked")
    return nav, log, first(log, lambda f: f.state == "ramping").t


def test_ramp_pinned_on_its_stair_ends_at_the_travel_budget_and_never_by_a_stall_rule():
    nav, log, takeoff = _pinned_ramp(A)
    assert nav.reason == "ramp timeout" and nav.transition_committed
    assert on_time(log[-1].t, takeoff + 15.0, .12)  # default budget: timeout_s, counted from the takeoff
    assert all(f.state == "ramping" for f in log if takeoff <= f.t < log[-1].t)
    nav, log, takeoff = _pinned_ramp({**RAMP8, "transition_stall_s": 1})
    assert nav.reason == "ramp timeout" and on_time(log[-1].t, takeoff + 8.0, .12)
    nav, log, takeoff = _pinned_ramp({})
    assert nav.reason == "ramp timeout" and log[-1].t == pytest.approx(15.0)  # the old cut, from the selection
    # Swaying a pixel back and forth on the stair is no different.
    nav = lap_nav(RAMP, RAMP8)
    log = script(nav, [(81, 64)] * 4 + [(83, 62), (84, 61)] * 60)
    assert nav.reason == "ramp timeout" and on_time(first(log, lambda f: f.state == "blocked").t, .3 + 8.0)


def test_slow_but_progressing_ramp_ends_at_its_travel_budget():
    crawl = {"slope_factor": .04}  # 0.38 px/s: under 6 px in 15 s, and the ramp is 10 px long
    for keys, budget in ((RAMP8, 8.0), (A, 15.0)):
        nav, world, log = run(RAMP, (81, 64.5), keys, world=crawl)
        takeoff = first(log, lambda f: f.state == "ramping")
        assert nav.reason == "ramp timeout" and budget - 1e-9 <= log[-1].t - takeoff.t < budget + .12 + 1e-9
        assert 81 + .3 * budget < world.x < 89  # it was moving the whole time and is still on the stair
    nav, world, off = run(RAMP, (81, 64.5), {}, world=crawl)
    assert nav.reason == "ramp timeout" and off[-1].t == pytest.approx(15.0)  # the old cut, from the selection


def test_ramp_landing_that_never_holds_still_ends_at_the_budget_plus_its_dwell():
    flicker = [(81, 64)] * 4 + [(92, 55), (92, 54)] * 80
    for keys, budget in ((A, 15.0), (RAMP8, 8.0)):
        log = script(lap_nav(RAMP, keys), flicker)
        takeoff = first(log, lambda f: f.state != "ramp_settling")
        blocked = first(log, lambda f: f.state == "blocked")
        assert takeoff.t == pytest.approx(.3) and on_time(blocked.t, .3 + budget + .6)
        assert all(f.state == "ramp_landing" for f in log if takeoff.t < f.t < blocked.t)
    off = lap_nav(RAMP)
    assert first(script(off, flicker), lambda f: f.state == "blocked").t == pytest.approx(15.0)
    assert off.reason == "ramp timeout"


def _knocked_ramp(keys, tail):
    nav = lap_nav(RAMP, keys)
    positions = [(79, 65)] * 100 + [(81, 64)] * 4 + [(83, 62), (84, 60)] + [(79, 65)] * 4 + tail
    return nav, script(nav, positions)


def test_ramp_knockback_rearm_credits_only_the_committed_time():
    nav, log = _knocked_ramp(A, [(79, 65)] * 80)
    takeoff = first(log, lambda f: f.state == "ramping").t
    rearmed = first(log, lambda f: f.state == "ramp_realigning").t
    assert (takeoff, rearmed) == (pytest.approx(10.3), pytest.approx(10.9))
    assert [f.committed for f in log if takeoff <= f.t < rearmed] == [True] * 6
    blocked = first(log, lambda f: f.state == "blocked")
    # 10.3 s were spent before the takeoff; the remaining 4.7 s run from the re-arm.
    assert nav.reason == "ramp timeout" and on_time(blocked.t, 15.0 + rearmed - takeoff)
    nav, log = _knocked_ramp({}, [(79, 65)] * 80)
    assert first(log, lambda f: f.state == "blocked").t == pytest.approx(15.0)


def test_second_ramp_takeoff_after_a_knockback_gets_its_own_budget():
    # Re-armed at 10.9 s, 4.7 s of pre-takeoff time left (until 15.6 s). The second takeoff
    # comes at 15.3 s and its landing is proven at 16.5 s, after that clock ran out.
    tail = [(79, 65)] * 40 + [(81, 64)] * 4 + [(83, 62), (85, 60), (87, 58), (89, 55)] + [(92, 55)] * 9
    nav, log = _knocked_ramp(A, tail)
    takeoffs = [f.t for a, f in zip(log, log[1:]) if f.state == "ramping" and a.state != "ramping"]
    assert takeoffs == [pytest.approx(10.3), pytest.approx(15.3)]
    assert nav.idx == RAMP + 1 and nav.reason is None and "blocked" not in [f.state for f in log]
    assert first(log, lambda f: f.idx == RAMP + 1).t == pytest.approx(16.5)
    nav, log = _knocked_ramp({}, tail)  # without the key the clock ends it before the second takeoff
    assert nav.reason == "ramp timeout" and first(log, lambda f: f.state == "blocked").t == pytest.approx(15.0)


def test_waypoint_clock_still_ends_a_ramp_inside_its_knockback_wait_without_the_key():
    # Takeoff at 14.6 s, knocked back at 14.9 s. Without the key the unchanged clock ends the
    # waypoint at 15.0 s, inside the 1.2 s knock-back wait. With it the wait is committed time:
    # re-armed at 15.2 s, and the 0.6 s since the takeoff are credited to the pre-takeoff clock.
    positions = [(79, 65)] * 143 + [(81, 64)] * 4 + [(83, 62), (84, 60)] + [(79, 65)] * 30
    nav = lap_nav(RAMP)
    log = script(nav, positions)
    assert first(log, lambda f: f.state == "ramp_waiting_knockback").t == pytest.approx(14.9)
    assert nav.reason == "ramp timeout" and first(log, lambda f: f.state == "blocked").t == pytest.approx(15.0)
    nav = lap_nav(RAMP, A)
    log = script(nav, positions)
    assert first(log, lambda f: f.state == "ramp_realigning").t == pytest.approx(15.2)
    assert nav.reason == "ramp timeout" and first(log, lambda f: f.state == "blocked").t == pytest.approx(15.6)


@pytest.mark.parametrize("keys", [{}, A])
def test_second_ramp_knockback_still_leaves_the_corridor(keys):
    nav, log = _knocked_ramp(keys, [(81, 64)] * 4 + [(83, 62), (79, 65), (79, 65)])
    assert log[-2].state == "blocked" and nav.reason == "left ramp corridor"


@pytest.mark.parametrize("gap", ["suspend", None])
def test_suspended_and_positionless_frames_do_not_spend_a_commit_budget(gap):
    keys = {**A, "ramp_travel_timeout_s": 4}
    climb = [(81, 64)] * 4 + [(83, 62)] + [gap] * 150 + [(85, 60), (87, 58), (89, 55), (91, 55)] + [(92, 55)] * 9
    nav = lap_nav(RAMP, keys)
    log = script(nav, climb)
    assert nav.idx == RAMP + 1 and "blocked" not in [f.state for f in log]  # 16 s after the takeoff
    # Active time is still counted: a ramp pinned after the gap ends when 4 s of it are used up.
    nav = lap_nav(RAMP, keys)
    log = script(nav, [(81, 64)] * 4 + [(83, 62)] + [gap] * 150 + [(83, 62)] * 60)
    # (0.1 s of ramp before the gap, the frame at 15.5 s that ends the gap counts nothing, then 3.9 s.)
    assert nav.reason == "ramp timeout" and on_time(first(log, lambda f: f.state == "blocked").t, 19.4)


# The same lap run by one circuit through several transitions: ramp 14, gap jump 17, stair
# jump 20, ramp 22. Commit state must never leak from one waypoint selection into the next.
def test_closed_loop_through_two_ramps_and_two_jumps_leaves_no_commit_state_behind():
    def lap(keys):
        cfg = site(start_idx=RAMP, return_on_block=False, **keys)
        nav, world = MinimapCircuit(cfg), World(cfg, (81, 64.5))
        selected = []

        def stop(n):
            if not selected or selected[-1][0] != n.idx:
                selected.append((n.idx, (n._commit_started, n._commit_progress_at, n._climb_commit_spent)))
            return n.state == "blocked" or n.idx == RAMP3 + 1
        return nav, drive(nav, world, limit_s=120, stop=stop), selected
    nav, log, selected = lap(A)
    assert nav.idx == RAMP3 + 1 and nav.reason is None
    assert [idx for idx, _ in selected] == list(range(RAMP, RAMP3 + 2))
    assert all(state == (None, None, 0.0) for _, state in selected)  # on the frame of every selection
    assert {f.idx for f in log if f.committed} == {RAMP, GAP, STAIR_JUMP, RAMP3}
    assert digest(log) == digest(lap({})[1])  # no takeoff met an expired clock: nothing changed


SCRIPTED_ROUTE = [_SITE_CIRCUIT["waypoints"][RAMP - 1], _SITE_CIRCUIT["waypoints"][RAMP],
                  {"x": 103, "y": 55.0, "action": "walk", "dwell_s": 0, "platform_id": "floor-3-right"},
                  {"x": 103, "y": 71.0, "action": "drop", "dwell_s": .6, "from_platform": "floor-3-right",
                   "to_platform": "floor-2-right", "connection_id": "drop-right-3-2"},
                  {"x": 100, "y": 71.0, "action": "walk", "dwell_s": .6, "platform_id": "floor-2-right"}]


def _ramp_walk_drop(keys, tail):
    """Ramp stair-2-finish (takeoff 0.3 s, proven 1.3 s), a walk, then the drop at x=103 (selected 1.7 s)."""
    nav = MinimapCircuit(site(waypoints=deepcopy(SCRIPTED_ROUTE), start_idx=1, loop=False, return_on_block=False,
                              **keys))
    ramp = [(81, 64)] * 4 + [(84, 60), (87, 58), (89, 55)] + [(92, 55)] * 8
    log = script(nav, ramp + [(97, 55), (101, 55), (103, 55)] + tail)
    assert first(log, lambda f: f.idx == 3).t == pytest.approx(1.7)
    return nav, log


@pytest.mark.parametrize("keys", [{}, A])
def test_waypoint_after_a_committed_ramp_has_its_own_whole_pretakeoff_clock(keys):
    nav, log = _ramp_walk_drop(keys, [(106, 55)] * 200)  # never lines up for the drop
    blocked = first(log, lambda f: f.state == "blocked")
    assert nav.reason == "drop timeout" and blocked.t == pytest.approx(1.7 + 15.0)
    assert all(f.state == "aligning" and not f.committed for f in log if 1.8 <= f.t < blocked.t)


def test_waypoint_after_a_committed_ramp_measures_its_budget_from_its_own_takeoff():
    # 10 s beside the drop point, Down+Jump at 12.1 s, then a fall of one pixel per 0.3 s that
    # lands 4.3 s later: after the waypoint clock (16.7 s), inside the 6 s from this launch.
    fall = [(103, 56 + i // 3) for i in range(45)]
    tail = [(106, 55)] * 100 + [(103, 55)] * 4 + fall + [(103, 71)] * 10
    nav, log = _ramp_walk_drop(A, tail)
    launch = first(log, lambda f: f.cmd.action == "jump")
    assert launch.t == pytest.approx(12.1) and nav.idx == 4 and nav.reason is None
    assert first(log, lambda f: f.idx == 4).t == pytest.approx(17.3)
    nav, log = _ramp_walk_drop({}, tail)
    assert nav.reason == "drop timeout" and first(log, lambda f: f.state == "blocked").t == pytest.approx(16.7)


def _late_takeoff(idx, parked, takeoff, keys, stop_extra=()):
    cfg = site(start_idx=idx, return_on_block=False, **keys)
    nav, world = MinimapCircuit(cfg), World(cfg, parked)
    world.frozen = True  # 14.5 s in which no tap moves the character

    def release(w):
        w.frozen = False
        w.put(takeoff)
    log = drive(nav, world, limit_s=30, events=[(14.5, release)],
                stop=lambda n: n.idx != idx or n.state in ("blocked", *stop_extra))
    return nav, world, log


def test_jump_taken_off_just_before_the_clock_is_not_cut_in_flight():
    nav, world, log = _late_takeoff(GAP, (93, 55), (91, 55), {}, ("jump_retry",))
    assert log[-1].state == "jump_retry" and log[-1].t == pytest.approx(15.0)
    assert world.takeoffs == 1 and world.air  # the old clock failed a jump that was in the air
    nav, world, log = _late_takeoff(GAP, (93, 55), (91, 55), A, ("jump_retry",))
    assert nav.idx == GAP + 1 and world.takeoffs == 1 and nav.retries == 0
    key = first(log, lambda f: f.cmd.action == "jump")
    assert 14.75 <= key.t < 15.0 and [f.state for f in log[-4:-1]] == ["jump_landing"] * 3
    assert 84 <= world.x <= 86.5 and world.y == 55


def test_jump_flight_and_landing_confirmation_keep_their_own_bounds():
    takeoff = [(91, 55)] * 4
    for keys in ({}, A):  # a flight that never reaches the landing: unchanged 2 s
        nav = lap_nav(GAP, keys)
        log = script(nav, takeoff + [(88, 50)] * 30)
        assert on_time(first(log, lambda f: f.state == "jump_retry").t, .3 + 2.0)
    wobble = takeoff + [(90, 53), (88, 51), (87, 53), (86, 54)] + [(85, 55), (85, 54)] * 120
    nav = lap_nav(GAP, A)
    log = script(nav, wobble)
    # At the landing but never still: flight timeout + landing_confirm_timeout_s + the 0.5 s dwell.
    retry = first(log, lambda f: f.state == "jump_retry").t
    assert on_time(retry, .3 + 2.0 + 3.0 + .5)
    blocked = first(log, lambda f: f.state == "blocked")
    assert nav.reason == "jump timeout" and nav.retries == 1  # max_retries still bounds the takeoffs
    assert on_time(blocked.t, retry + .6 + 15.0, .2)
    off = lap_nav(GAP)
    assert first(script(off, wobble), lambda f: f.state == "jump_retry").t == pytest.approx(15.0)


@pytest.mark.parametrize("positions, state", [
    ([(81, 55)] * 330, "jump_arrived_aligning"),                          # 4 px from the landing X, never closer
    ([(85, 55)] * 6 + [(85, 54.4), (85, 55)] * 162, "jump_arrived"),      # at it, height never still for the dwell
], ids=["beside the landing", "never still"])
def test_jump_accepted_without_a_takeoff_keeps_the_waypoint_clock_with_commit_budget(positions, state):
    # Proven on the destination platform (arrive_on_destination): no Jump was sent, so there is
    # no flight and no flight bound. Only the waypoint clock ends this phase, with the key too.
    logs = []
    for keys in ({}, A):
        nav = lap_nav(GAP, keys, arrive_on_destination=True)
        log = script(nav, positions)
        assert first(log, lambda f: f.state == state).t < 1.0 and not any(f.cmd.action == "jump" for f in log)
        assert first(log, lambda f: f.state == "jump_retry").t == pytest.approx(15.0)
        assert nav.reason == "jump timeout" and first(log, lambda f: f.state == "blocked").t == pytest.approx(30.6)
        logs.append([(f.state, f.cmd, f.committed) for f in log])
    assert logs[0] == logs[1]


def test_drop_launched_just_before_the_clock_is_not_cut_in_the_fall():
    nav, world, log = _late_takeoff(DROP, (63, 38), (60, 38), {})
    assert nav.reason == "drop timeout" and log[-1].t == pytest.approx(15.0) and world.drops == 1 and world.air
    nav, world, log = _late_takeoff(DROP, (63, 38), (60, 38), A)
    assert nav.idx == DROP + 1 and world.drops == 1 and world.pos == (60, 55)
    assert [f.state for f in log[-3:-1]] == ["dwelling"] * 2


@pytest.mark.parametrize("keys", [{}, A])
def test_drop_that_never_leaves_its_floor_keeps_the_existing_retry_rule(keys):
    cfg = site(start_idx=DROP, return_on_block=False, **keys)
    nav, world = MinimapCircuit(cfg), World(cfg, (60, 38))
    world.frozen = True
    log = drive(nav, world, limit_s=20, stop=lambda n: n.state == "blocked")
    assert nav.reason == "drop did not leave source floor"
    assert sum(f.cmd.action == "jump" for f in log) >= 2 and log[-1].t < 4  # identical with the key on


def test_stalled_fall_and_post_landing_correction_are_bounded():
    launch = [(60, 38)] * 4 + [(60, 42), (58, 47), (50, 52)]
    beside = launch + [(41, 55)] * 200  # level with floor-3 but one pixel left of its drawn end
    nav = lap_nav(DROP, A)
    log = script(nav, beside)
    assert nav.reason == "drop stalled" and on_time(first(log, lambda f: f.state == "blocked").t, .7 + 2.0)
    off = lap_nav(DROP)
    assert first(script(off, beside), lambda f: f.state == "blocked").t == pytest.approx(15.0)
    # Landed 4 px beside the point: the landing proof ends the commit, and the correction on
    # the destination floor is charged to the waypoint clock again (minus the 0.9 s fall).
    aside = [(60, 38)] * 4 + [(60, 45), (62, 52)] + [(64, 55)] * 200
    nav = lap_nav(DROP, A)
    log = script(nav, aside)
    assert nav.reason == "drop timeout" and on_time(first(log, lambda f: f.state == "blocked").t, 15.9)
    assert max_lease(log) == pytest.approx(.04)  # the post-landing correction keeps fine taps
    # A landing that never holds still ends at the fall budget plus its dwell.
    nav = lap_nav(DROP, A)
    log = script(nav, launch + [(60, 55), (60, 54)] * 80)
    assert nav.reason == "drop timeout" and on_time(first(log, lambda f: f.state == "blocked").t, .3 + 6.0 + .6)


LAUNCH = [(60, 38)] * 4  # drop-4-3: settled on the takeoff, Down+Jump on the frame at 0.3 s


def _blocked_at(log):
    return first(log, lambda f: f.state == "blocked").t


def _launches(log):
    return [f.t for a, f in zip([None] + log, log) if f.cmd.action == "jump" and (a is None or a.cmd.action != "jump")]


def test_failed_drop_launch_puts_the_realignment_back_on_the_credited_waypoint_clock():
    # Down+Jump at 0.3 s, knocked 3 px aside on the same floor, re-armed by the unchanged 0.8 s
    # rule at 1.1 s. Seven more seconds of re-alignment are pre-takeoff time, not fall time.
    retake = LAUNCH + [(63, 38)] * 75 + [(60, 38)] * 4 + [(60, 45), (60, 52)] + [(60, 55)] * 9
    for keys in ({}, A):
        nav = lap_nav(DROP, keys)
        log = script(nav, retake)
        assert _launches(log) == [pytest.approx(.3), pytest.approx(8.2)] and nav.idx == DROP + 1, keys
    # Never lining up again ends at the waypoint clock, later by exactly the 0.8 s that were committed.
    nav = lap_nav(DROP, A)
    log = script(nav, LAUNCH + [(63, 38)] * 200)
    assert first(log, lambda f: f.state == "aligning").t == pytest.approx(1.2)
    assert nav.reason == "drop timeout" and _blocked_at(log) == pytest.approx(15.0 + .8)
    assert [f.committed for f in log if .3 <= f.t < 1.2] == [True] * 9 and not log[12].committed
    off = lap_nav(DROP)
    assert _blocked_at(script(off, LAUNCH + [(63, 38)] * 200)) == pytest.approx(15.0)


def test_drop_caught_by_a_ledge_gets_a_fresh_fall_budget_for_the_next_launch():
    # Caught on a stable ledge at y=46 (0.5 s), 3 px beside the point for ten seconds, then the
    # second Down+Jump at 10.3 s: more than the 6 s fall budget after the first launch.
    ledge = LAUNCH + [(60, 42)] + [(63, 46)] * 95 + [(60, 46)] * 4
    for keys in ({}, A):
        nav = lap_nav(DROP, keys)
        log = script(nav, ledge + [(60, 50)] + [(60, 55)] * 9)
        assert _launches(log) == [pytest.approx(.3), pytest.approx(10.3)] and nav.idx == DROP + 1, keys
        assert not any(f.committed for f in log if 1.0 <= f.t < 10.3)  # lining up on the ledge is pre-takeoff
    # The second fall has its own budget: a landing that never holds still ends 6 s + dwell after it.
    never_still = ledge + [(60, 50)] + [(60, 55), (60, 54)] * 60
    nav = lap_nav(DROP, A)
    assert on_time(_blocked_at(script(nav, never_still)), 10.3 + 6.0 + .6) and nav.reason == "drop timeout"
    off = lap_nav(DROP)
    assert _blocked_at(script(off, never_still)) == pytest.approx(15.0)


def test_dip_below_the_drop_floor_and_back_is_not_a_fall_and_stays_on_the_waypoint_clock():
    aside = [(63, 38)] * 200
    # No launch: two frames 2-3 px lower at 1.0 s, back on the floor, seen stable there at 1.6 s.
    # While it lasts it is handled as a fall; once it is back it was pre-takeoff time after all.
    nav = lap_nav(DROP, A)
    log = script(nav, [(63, 38)] * 10 + [(63, 40), (63, 41)] + aside)
    assert nav.reason == "drop timeout" and _blocked_at(log) == pytest.approx(15.0) and not _launches(log)
    assert [f.committed for f in log if 1.0 <= f.t < 1.7] == [True] * 7 and not log[17].committed
    # A fall nobody launched right after such a dip still gets its own budget (off: cut at 15.0 s).
    then_falls = ([(63, 38)] * 100 + [(63, 40), (63, 41)] + [(63, 38)] * 40 + [(61, 42), (60, 47), (60, 52)]
                  + [(60, 55)] * 9)
    nav = lap_nav(DROP, A)
    log = script(nav, then_falls)
    assert nav.idx == DROP + 1 and first(log, lambda f: f.idx == DROP + 1).t > 15.0
    off = lap_nav(DROP)
    assert _blocked_at(script(off, then_falls)) == pytest.approx(15.0)
    # Launched at 0.3 s, the same dip: the launch stays committed until its own failed-launch
    # re-arm (drop_interval_s, here 2 s after the key), and that span is credited.
    keys = {**A, "drop_interval_s": 2.0}
    nav = lap_nav(DROP, keys)
    log = script(nav, LAUNCH + [(60, 40), (60, 41)] + aside)
    assert _launches(log) == [pytest.approx(.3)] and first(log, lambda f: f.state == "aligning").t == pytest.approx(2.5)
    assert nav.reason == "drop timeout" and _blocked_at(log) == pytest.approx(15.0 + 2.1)
    for dip in ([(63, 38)] * 10 + [(63, 40), (63, 41)], LAUNCH + [(60, 40), (60, 41)]):
        off = lap_nav(DROP, {"drop_interval_s": 2.0})
        assert _blocked_at(script(off, dip + aside)) == pytest.approx(15.0)


@pytest.mark.parametrize("dip, back", [([(63, 40)], 6), ([(63, 40), (63, 41)] * 5, 6)],
                         ids=["one frame", "ten frames"])
@pytest.mark.parametrize("dt", [.1, .16])
def test_height_reading_that_keeps_dipping_cannot_postpone_the_drop_timeout(dip, back, dt):
    # Beside the drop point, the dot reads 2-3 px lower for one frame (or ten, still under the
    # 2 s stall rule) again and again. Each dip looks like the start of a fall; none is credited.
    cycle = dip + [(63, 38)] * back
    flicker = [(63, 38)] * 5 + cycle * 60
    nav = lap_nav(DROP, A)
    log = script(nav, flicker, dt=dt)
    assert nav.reason == "drop timeout" and not _launches(log)
    assert 15.0 <= _blocked_at(log) < 15.0 + (len(cycle) + 1) * dt  # at most one dip later than without the key
    off = lap_nav(DROP)
    assert 15.0 <= _blocked_at(script(off, flicker, dt=dt)) < 15.0 + dt


def test_fall_nobody_launched_gets_the_fall_budget_instead_of_the_spent_waypoint_clock():
    # 14.5 s beside the drop point, then knocked off the ledge: landed at 14.9 s, proven at 15.6 s.
    knocked_off = [(63, 38)] * 146 + [(61, 42), (60, 47), (60, 52)] + [(60, 55)] * 9
    nav = lap_nav(DROP, A)
    log = script(nav, knocked_off)
    assert nav.idx == DROP + 1 and not _launches(log) and first(log, lambda f: f.idx == DROP + 1).t == pytest.approx(15.6)
    assert [f.state for f in log if 14.9 <= f.t < 15.6] == ["dwelling"] * 7
    off = lap_nav(DROP)
    assert _blocked_at(script(off, knocked_off)) == pytest.approx(15.0) and off.reason == "drop timeout"


def test_fall_that_keeps_descending_without_landing_ends_at_the_fall_budget():
    # Bouncing a pixel every frame and one pixel lower every 1.5 s: never a stable ledge, never
    # two seconds without progress, never at the destination.
    falling = LAUNCH + [(60, 42 + i // 15 + i % 2) for i in range(160)]
    assert max(y for _, y in falling) == 53
    nav = lap_nav(DROP, A)
    log = script(nav, falling)
    assert nav.reason == "drop timeout" and on_time(_blocked_at(log), .3 + 6.0)
    off = lap_nav(DROP)
    assert _blocked_at(script(off, falling)) == pytest.approx(15.0)


def test_no_progress_rule_counts_whole_pixels_toward_the_destination_only():
    # A fall swinging between 47 and 46: the deepest pixel was first seen at 0.5 s.
    nav = lap_nav(DROP, A)
    log = script(nav, LAUNCH + [(60, 43), (60, 47)] + [(60, 46), (60, 47)] * 60)
    assert nav.reason == "drop stalled" and on_time(_blocked_at(log), .5 + 2.0)
    # Touching the destination is progress while it lasts: bouncing between it and 3 px above
    # runs to the fall budget, not into the stall rule.
    nav = lap_nav(DROP, A)
    log = script(nav, LAUNCH + [(60, 45)] + [(60, 55), (60, 52)] * 60)
    assert nav.reason == "drop timeout" and on_time(_blocked_at(log), .3 + 6.0)
    # A rope: swinging between 100 and 101 rests 2 s after 100 was first reached (0.8 s)...
    entry = [(49, 105)] * 5 + [(49, 103), (49, 102), (49, 101), (49, 100)]
    nav = lap_nav(LADDER, A)
    log = script(nav, entry + [(49, 101), (49, 100)] * 60)
    assert on_time(first(log, lambda f: f.state == "recovering").t, .8 + 2.0)
    # ...and creeping a quarter pixel per second is not a pixel of progress either.
    nav = lap_nav(LADDER, A)
    log = script(nav, entry + [(49, 100 - .25 * (i // 10)) for i in range(1, 60)])
    assert on_time(first(log, lambda f: f.state == "recovering").t, .8 + 2.0)
    off = lap_nav(LADDER)
    assert not [f for f in script(off, entry + [(49, 101), (49, 100)] * 60) if f.state == "recovering"]


def test_rope_entered_just_before_the_clock_is_not_released_mid_climb():
    off = scenario("climb")
    entry = first(off, lambda f: f.cmd.action == "jump")
    assert 14.5 < entry.t < 15.0
    rest = [f for f in off if f.state == "recovering"]
    assert rest and rest[0].t == pytest.approx(15.0) and 89 < rest[0].y < 104  # Up released on the rope
    on = scenario("climb", commit_budget=True)
    assert digest(on[:off.index(entry) + 1]) == digest(off[:off.index(entry) + 1])
    arrived = first(on, lambda f: f.idx == LADDER + 1)
    assert not [f for f in on if f.state == "recovering"]
    assert all(f.cmd.move_y == "up" for f in on if entry.t <= f.t < arrived.t)
    assert arrived.t < first(off, lambda f: f.idx == LADDER + 1).t


def test_stalled_rope_takes_the_existing_retry_and_timeout_path():
    def stuck(keys):
        cfg = site(start_idx=LADDER, return_on_block=False, **keys)
        nav, world = MinimapCircuit(cfg), World(cfg, (49, 105))
        log = drive(nav, world, limit_s=40, events=[(1.0, lambda w: setattr(w, "rope_frozen", True))],
                    stop=lambda n: n.state == "blocked")
        return nav, world, log
    nav, world, log = stuck(A)
    assert nav.reason == "waypoint timeout" and nav.retries == 1 and world.rope is not None
    rests = [f.t for f in log if f.state == "recovering"]
    last_progress = max(f.t for a, f in zip(log, log[1:]) if f.pos[1] < a.pos[1])
    assert rests[0] - last_progress == pytest.approx(2.0, abs=.13)  # transition_stall_s, then recovery_s
    assert all(f.cmd == Command.NONE for f in log if f.state == "recovering") and log[-1].t < 7
    nav, world, log = stuck({})
    assert nav.reason == "waypoint timeout" and log[-1].t > 30  # two whole 15 s windows before


def test_rope_budget_is_cumulative_across_leaving_and_reentering():
    def rope(frames):  # one pixel up every 0.3 s
        return [(49, 102 - i // 3) for i in range(frames)]
    positions = [(49, 105)] * 6 + rope(15) + [(49, 105)] * 10 + rope(40)
    nav = lap_nav(LADDER, A, climb_travel_timeout_s=3)
    log = script(nav, positions)
    # 1.5 s on the rope, back on the floor, on again at 3.1 s: the budget ends 1.5 s later, not 3 s.
    rest = first(log, lambda f: f.state == "recovering").t
    assert on_time(rest, 4.6)
    # A spent rope budget is spent for this waypoint selection: after the one retry rest the
    # existing "waypoint timeout" follows at once instead of another whole window.
    assert nav.reason == "waypoint timeout" and nav.retries == 1
    assert on_time(first(log, lambda f: f.state == "blocked").t, rest + .6)
    off = lap_nav(LADDER)
    assert not [f for f in script(off, positions) if f.state == "recovering"]


@pytest.mark.parametrize("idx, start", [(LADDER, (54, 105)), (TOP_LADDER, (66, 34))],
                         ids=["base ladder", "top ladder"])
def test_rope_that_is_never_reached_keeps_the_waypoint_clock_with_commit_budget(idx, start):
    def frozen(keys):
        cfg = site(start_idx=idx, return_on_block=False, **keys)
        nav, world = MinimapCircuit(cfg), World(cfg, start)
        world.frozen = True  # on the source floor beside the rope; no tap moves the character
        return nav, drive(nav, world, limit_s=60, stop=lambda n: n.state == "blocked")
    nav, log = frozen(A)
    assert first(log, lambda f: f.state == "recovering").t == pytest.approx(15.0)
    assert nav.reason == "waypoint timeout" and nav.retries == 1 and log[-1].t == pytest.approx(30.6)
    assert {f.state for f in log} == {"aligning", "recovering", "blocked"}
    assert digest(log) == digest(frozen({})[1])  # frame for frame what the key-less circuit does


@pytest.mark.parametrize("pos", [(52, 100), (51, 100)])  # 3 and 2 px beside the rope, 5 px above the ground
def test_hanging_beside_the_rope_is_not_on_the_rope_and_keeps_the_waypoint_clock(pos):
    for keys in ({}, A):
        nav = lap_nav(LADDER, keys)
        log = script(nav, [pos] * 330)
        assert first(log, lambda f: f.state == "recovering").t == pytest.approx(15.0), keys
        assert nav.reason == "waypoint timeout" and _blocked_at(log) == pytest.approx(30.6), keys
    # One pixel beside it is inside climb_jitter_x: on the rope by the minimap, so the stall rule applies.
    nav = lap_nav(LADDER, A)
    assert on_time(first(script(nav, [(50, 100)] * 60), lambda f: f.state == "recovering").t, 2.0)


def test_top_of_the_rope_reached_beside_its_x_is_no_longer_rope_time():
    # Up the rope in 1.5 s (0.5 to 2.0 s), then standing at the destination height 1.5 px beside
    # the point: off the rope. The waypoint clock runs again, later by the credited rope time.
    positions = [(49, 105)] * 5 + [(49, 103 - i) for i in range(15)] + [(50.5, 88)] * 330
    nav = lap_nav(LADDER, A)
    log = script(nav, positions)
    assert first(log, lambda f: f.pos == (50.5, 88)).t == pytest.approx(2.0)
    assert first(log, lambda f: f.state == "recovering").t == pytest.approx(15.0 + 1.5)
    assert nav.reason == "waypoint timeout" and _blocked_at(log) == pytest.approx(15.0 + 1.5 + .6 + 15.0)
    off = lap_nav(LADDER)
    assert first(script(off, positions), lambda f: f.state == "recovering").t == pytest.approx(15.0)


ROPES = [_SITE_CIRCUIT["waypoints"][i] for i in (0, LADDER, 40, 41)]  # point 1, up the base ladder, back down


def test_second_rope_waypoint_gets_its_own_whole_rope_budget():
    def climb(keys, positions):
        nav = MinimapCircuit(site(waypoints=deepcopy(ROPES), start_idx=1, return_on_block=False, **keys))
        return nav, script(nav, positions)
    keys = {**A, "climb_travel_timeout_s": 3}
    # Up: 1.5 s on the rope, dropped back to the ground, on again and up in 1.4 s (2.9 s of 3).
    up = ([(49, 105)] * 5 + [(49, 103 - i // 3) for i in range(15)] + [(49, 105)] * 10
          + [(49, 103 - i) for i in range(14)] + [(49, 88)] * 6)
    # Down from floor-1: one pixel per 0.5 s, far too slow to arrive.
    down = [(49, 88)] * 12 + [(49, 90 + i // 5) for i in range(60)]
    nav, log = climb(keys, up + down)
    assert not [f for f in log if f.idx == 1 and f.state == "recovering"]
    selected = first(log, lambda f: f.idx == 3)
    on_rope = first(log, lambda f: f.idx == 3 and f.pos[1] >= 90)
    assert selected.t == pytest.approx(5.7) and on_rope.t == pytest.approx(6.2)
    # The whole 3 s from its own first rope frame, although the first rope used 2.9 s.
    assert on_time(first(log, lambda f: f.state == "recovering").t, on_rope.t + 3.0)
    assert nav.idx == 3 and nav._climb_commit_spent == pytest.approx(3.0, abs=.11)
    nav, log = climb({}, up + down)  # without the key nothing rests inside these 12 seconds
    assert nav.idx == 3 and not [f for f in log if f.state == "recovering"]


def test_walk_waypoint_between_two_heights_is_unchanged_by_commit_budget():
    # Only a climb waypoint is ever "on the rope". A walk whose previous walk waypoint lies at
    # another height (a slope) keeps the plain waypoint clock, frame for frame.
    def walking(keys):
        nav = MinimapCircuit({"waypoints": [{"x": 100, "y": 38, "dwell_s": 0}, {"x": 110, "y": 30, "dwell_s": 0}],
                              "start_idx": 1, "precision_walk": True, "tolerance_x": 1.0, "tolerance_y": 1.0,
                              "timeout_s": 15, "max_retries": 1, "recovery_s": .6, **keys})
        return nav, script(nav, [(110, 34)] * 330)  # under the waypoint's X, 4 px short of its height
    nav, log = walking(A)
    assert first(log, lambda f: f.state == "recovering").t == pytest.approx(15.0) and nav._commit_started is None
    assert nav.reason == "waypoint timeout" and first(log, lambda f: f.state == "blocked").t == pytest.approx(30.6)
    assert [(f.state, f.cmd) for f in log] == [(f.state, f.cmd) for f in walking({})[1]]


# --------------------------------------------------------------------------
# 4. takeoff_far_approach: faster alignment, same takeoff proof, never past an edge
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(TAKEOFFS))
def test_takeoff_far_approach_is_faster_and_stays_inside_the_tolerance_and_the_platform(name):
    _, target, tolerance, platform_id, sides = TAKEOFFS[name]
    surface = _surface(platform_id)
    fine = takeoff_sweep(name, {})
    far = takeoff_sweep(name, B)
    late = max((takeoff_sweep(name, B, dropped=at) for at in (.3, .48, .7, 1.0)), key=lambda r: (r[0], r[2]))
    print(f"\n{name}, {far[4]} runs: fine slowest {fine[0]:.2f} s mean {fine[1]:.2f} s beyond {fine[2]:+.2f} px "
          f"edge clearance {fine[3]:.2f} px | far slowest {far[0]:.2f} s mean {far[1]:.2f} s beyond {far[2]:+.2f} px "
          f"clearance {far[3]:.2f} px | far + one dropped 0.5 s frame: slowest {late[0]:.2f} s beyond {late[2]:+.2f} px")
    # Floor beyond the takeoff point in the walking direction. With less than the edge margin
    # plus one fine tap (1.75 + 0.44 px) no far pulse fits before the platform end.
    beyond = min(target - surface.left if side > 0 else surface.right - target for side in sides)
    if (beyond - FAR["approach_edge_margin_x"]) / FAR["approach_walk_speed_x"] < .04:
        assert name == "gap jump x91 toward the gap" and beyond == 2
        assert far[:4] == pytest.approx(fine[:4])  # toward the gap it is the fine taps, run for run
    else:
        assert far[0] < fine[0] and far[1] < fine[1]
        assert max(far[0], late[0]) < (3.0 if tolerance == 1.0 else 5.0)
    for result in (far, late):
        # The true position never left the takeoff tolerance (pixel rounding included) and kept
        # the validated 0.75 px of slide-and-rounding slack from the platform end ahead.
        assert result[2] <= .5 + 1e-9 and result[3] >= .75
    assert tolerance in (.4, 1.0)


def knocked_mid_key(keys, name, start, land, *, speed, coast, dt, delay, stale):
    """A hit ``delay`` s into the first alignment key puts the character on ``land`` (same floor).

    That key was sized from the position before the hit and the keyboard holds it for its
    lease; ``stale`` adds one frame of capture lag before the circuit sees the new position.
    An instant displacement along the floor is harsher than the game, where a knock-back
    lifts the character. Runs to the takeoff; returns (world, lease of that key).
    """
    idx, _, _, platform_id, _ = TAKEOFFS[name]
    surface = _surface(platform_id)
    cfg = site(start_idx=idx, return_on_block=False, timeout_s=60, **keys)
    nav, world = MinimapCircuit(cfg), World(cfg, (start, surface.y_at(start)), speed=speed, coast=coast)
    seen, lease, t = [], None, 0.0
    while t < 60:
        observed = seen[-1] if stale and seen else world.pos
        seen.append(world.pos)
        cmd = nav.step(observed, round(t, 6))
        if _taken_off(nav):
            break
        if lease is None and cmd.move_x != "none" and nav.state.endswith("aligning"):
            lease = cmd.valid_for_s
            world.advance(cmd, delay)
            world.put((land, surface.y_at(land)))
            world.advance(Command(cmd.move_x, valid_for_s=lease - delay) if lease > delay else Command.NONE,
                          dt - delay)
        else:
            world.advance(cmd, dt)
        t += dt
    assert lease is not None and _taken_off(nav) and nav.state != "blocked", (name, start, land, lease)
    return world, lease


@pytest.mark.parametrize("name", sorted(TAKEOFFS))
def test_hit_onto_the_takeoff_point_while_an_alignment_key_is_held_stays_on_the_source_platform(name):
    _, target, _, platform_id, sides = TAKEOFFS[name]
    surface = _surface(platform_id)
    for side in sides:
        for distance in (2.0, 3.5, 5.5):
            start = min(max(target + side * distance, surface.left + .1), surface.right - .1)
            for land in (target, target - side * .6):  # onto the takeoff point, and 0.6 px beyond it
                for speed, coast, dt, stale in product(SPEEDS, COASTS, FRAMES, (False, True)):
                    world, lease = knocked_mid_key(B, name, start, land, speed=speed, coast=coast, dt=dt,
                                                   delay=.02, stale=stale)
                    case = (name, start, land, speed, coast, dt, stale, lease)
                    assert surface.left <= world.low and world.high <= surface.right, case


@pytest.mark.parametrize("land", [90.4, 91.0, 92.0])
def test_gap_jump_hit_toward_the_gap_goes_exactly_as_far_as_with_fine_taps(land):
    name = "gap jump x91 toward the gap"  # takeoff x=91, floor-3-right ends at x=89
    for start, speed, coast, dt, delay, stale in product((94.5, 96.5), SPEEDS, COASTS, FRAMES, (.02, .04),
                                                         (False, True)):
        model = {"speed": speed, "coast": coast, "dt": dt, "delay": delay, "stale": stale}
        fine, tap = knocked_mid_key({}, name, start, land, **model)
        far, lease = knocked_mid_key(B, name, start, land, **model)
        assert lease == pytest.approx(.04) and tap == pytest.approx(.04), (start, model)
        assert far.low == pytest.approx(fine.low) and far.low >= 89, (start, model)


def test_gap_jump_two_pixels_from_the_gap_keeps_fine_taps_toward_it_and_far_pulses_away_from_it():
    nav = lap_nav(GAP, B)
    toward = script(nav, [(96, 55)] * 20, dt=.15)
    assert {f.state for f in toward} == {"jump_aligning"} and max_lease(toward) == pytest.approx(.04)
    nav = lap_nav(GAP, B)
    away = script(nav, [(89, 55)] * 4, dt=.15)  # from the gap side the far end is 20 px beyond the takeoff
    assert away[2].cmd.move_x == "right" and away[2].cmd.valid_for_s == pytest.approx((2 - 1) / 11)


def synthetic(action, platforms=None, previous=(100, 38), **keys):
    point = {"x": 100, "y": 50 if action == "drop" else 30 if action == "climb" else 38, "action": action,
             "dwell_s": 0}
    if action in ("jump", "ramp"):
        point.update(x=94, y=30, from_x=100, from_y=38)
    live = {"precision_walk": True, "tolerance_x": 1.0, "tolerance_y": 1.0, "align_tolerance_x": .4,
            "jump_align_tolerance_x": .4, "drop_align_tolerance_x": 1.0, "approach_radius_x": 6,
            "approach_pulse_s": .04, "approach_release_s": .45, "timeout_s": 15}
    return MinimapCircuit({"waypoints": [{"x": previous[0], "y": previous[1], "dwell_s": 0}, point,
                                         {"x": 130, "y": 38}],
                           "start_idx": 1, **({"platforms": platforms} if platforms else {}), **live, **keys})


def shelf(beyond):
    """Takeoff at x=100 on a platform at y=38 that has ``beyond`` px of floor left of it."""
    return [{"left": 100 - beyond, "right": 140, "y": 38}, {"left": 60, "right": 99, "y": 30},
            {"left": 60, "right": 140, "y": 50}]


def leases(nav, pos, frames=20):
    """Lease of every walking command over ``frames`` frames at ``pos`` (0.15 s apart)."""
    return [c.valid_for_s for c in (nav.step(pos, frame * .15) for frame in range(frames)) if c.move_x != "none"]


# A jump or ramp takes its source platform from its own takeoff point, a drop or rope from the
# walk waypoint before it. The jump and the ramp are therefore given a previous walk waypoint
# on ANOTHER platform (the floor at y=50), which must not be mistaken for the source.
SITES = [("jump", (120, 50)), ("ramp", (120, 50)), ("drop", (100, 38)), ("climb", (100, 38))]


@pytest.mark.parametrize("action, previous", SITES, ids=[action for action, _ in SITES])
def test_takeoff_far_pulse_is_sized_from_the_floor_beyond_the_takeoff_point(action, previous):
    def first_lease(beyond, x=104):
        return leases(synthetic(action, shelf(beyond), previous, **B), (x, 38))[0]
    # 4 px before a takeoff with 3 px of floor beyond it: the 1.75 px edge margin governs...
    assert first_lease(3) == pytest.approx((3 - 1.75) / 11)
    # ...with 8 px beyond it the stop margin does, as on a walk waypoint.
    assert first_lease(8) == pytest.approx((4 - 1) / 11)
    # The shortest far pulse is one fine tap long: 2.2 px beyond the takeoff still fit one.
    assert .04 < first_lease(2.2) == pytest.approx((2.2 - 1.75) / 11)
    # 2 px (the gap jumps) or a takeoff on the very end do not: fine taps only.
    for beyond in (2, 0):
        taps = leases(synthetic(action, shelf(beyond), previous, **B), (104, 38))
        assert len(taps) >= 3 and max(taps) == pytest.approx(.04), beyond
    # Walking away from that end, the far end of the platform is 40 px beyond the takeoff.
    assert first_lease(3, x=98) == pytest.approx((2 - 1) / 11)


@pytest.mark.parametrize("action, previous", SITES, ids=[action for action, _ in SITES])
def test_takeoff_far_approach_falls_back_to_fine_taps_without_a_proven_source_floor(action, previous):
    # No platforms configured: nothing bounds a pulse, so none is sized.
    taps = leases(synthetic(action, None, previous, **B), (104, 38))
    assert len(taps) >= 3 and max(taps) == pytest.approx(.04)
    # The key off: the same call sites pass no room at all.
    taps = leases(synthetic(action, shelf(8), previous, **FAR), (104, 38))
    assert len(taps) >= 3 and max(taps) == pytest.approx(.04)


@pytest.mark.parametrize("action", ["drop", "climb"])
def test_takeoff_far_pulse_needs_the_character_inside_the_source_platform_span(action):
    # Level with the source platform but on a neighbouring one, 4 px outside its left end. (A
    # jump or ramp does not line up at all there: it waits for its takeoff floor.)
    nav = synthetic(action, shelf(0) + [{"left": 80, "right": 97, "y": 38}], **B)
    taps = leases(nav, (96, 38))
    assert nav.state == "aligning" and len(taps) >= 3 and max(taps) == pytest.approx(.04)
    for waiting in ("jump", "ramp"):
        nav = synthetic(waiting, shelf(0) + [{"left": 80, "right": 97, "y": 38}], **B)
        assert not leases(nav, (96, 38)) and nav.state == waiting + "_waiting_floor"


@pytest.mark.parametrize("tolerance_y, on, off", [(1.0, (37, 38, 39), (36, 36.9, 39.1, 40)),
                                                  (.75, (37.25, 38, 38.75), (37, 37.2, 38.8, 39))])
def test_takeoff_far_pulse_needs_strict_support_on_the_source_surface(tolerance_y, on, off):
    # A rope waypoint lines up at any height, so here the guard inside _takeoff_room is the only one.
    def longest(y):
        return max(leases(synthetic("climb", shelf(8), **B, tolerance_y=tolerance_y), (104, y)))
    assert [longest(y) for y in on] == [pytest.approx(3 / 11)] * len(on)
    assert [longest(y) for y in off] == [pytest.approx(.04)] * len(off)


def test_takeoff_room_is_the_floor_beyond_the_takeoff_point_or_none():
    nav = synthetic("climb", shelf(3), **B)
    source = nav.platforms[0]  # 97..140 at y=38; the takeoff is x=100
    assert nav._takeoff_room(source, (104, 38), -4) == 3  # walking left: 100 - 97
    assert nav._takeoff_room(source, (140, 38), -40) == 3  # wherever on the platform the player stands
    assert nav._takeoff_room(source, (98, 38), 2) == nav._takeoff_room(source, (97, 38), 3) == 40  # 140 - 100
    for pos, dx in (((96.9, 38), 3.1), ((140.1, 38), -40.1)):  # level with it, outside its span
        assert nav._takeoff_room(source, pos, dx) is None
    for y, room in ((37, 3), (36.99, None), (39, 3), (39.01, None)):  # support error at most 1 px
        assert nav._takeoff_room(source, (104, y), -4) == room
    strict = synthetic("climb", shelf(3), **B, tolerance_y=.75)  # a return child's tolerance
    assert strict._takeoff_room(source, (104, 38.75), -4) == 3 and strict._takeoff_room(source, (104, 38.76), -4) is None
    assert nav._takeoff_room(None, (104, 38), -4) is None  # unknown source platform
    assert synthetic("climb", shelf(3), **FAR)._takeoff_room(source, (104, 38), -4) is None  # key off
    assert synthetic("climb", None, **B)._takeoff_room(source, (104, 38), -4) is None  # no platforms


def test_off_the_takeoff_floor_nothing_moves_and_far_pulses_never_start():
    for idx, pos, state in ((GAP, (95, 50), "jump_waiting_floor"), (RAMP, (79, 60), "ramp_waiting_floor"),
                            (DROP, (64, 33), "dropping")):
        nav = lap_nav(idx, B)
        log = script(nav, [pos] * 12)
        assert {f.state for f in log} == {state} and all(f.cmd == Command.NONE for f in log)
    # A rope waypoint already hanging above its source floor keeps fine taps.
    nav = lap_nav(LADDER, B)
    log = script(nav, [(53, 100)] * 30)
    assert max_lease(log) == pytest.approx(.04)


def test_settle_proof_before_a_takeoff_is_unchanged_by_the_far_approach():
    # The stair jump takes off at x=66 on floor-3, 24 px from the platform end ahead.
    nav = lap_nav(STAIR_JUMP, B)
    far = script(nav, [(71, 55)] * 4, dt=.15)
    assert far[2].cmd.move_x == "left" and far[2].cmd.valid_for_s == pytest.approx((5 - 1) / 11)
    # Aligned, then one frame a pixel off: the settle starts again and Jump waits for all of it.
    log = script(nav, [(66, 55), (66, 55), (67, 55), (66, 55), (66, 55), (66, 55), (66, 55)], t0=1.0)
    assert [f.state for f in log] == ["jump_settling", "jump_settling", "jump_aligning", "jump_settling",
                                      "jump_settling", "jump_settling", "jumping"]
    assert [f.cmd.action for f in log] == ["none"] * 6 + ["jump"]
    plain = lap_nav(STAIR_JUMP)
    script(plain, [(71, 55)] * 4, dt=.15)
    same = script(plain, [(66, 55), (66, 55), (67, 55), (66, 55), (66, 55), (66, 55), (66, 55)], t0=1.0)
    assert [f.state for f in same] == [f.state for f in log]


def test_suspend_during_a_takeoff_far_pulse_releases_and_needs_a_fresh_release_and_settle():
    nav = lap_nav(STAIR_JUMP, B)
    assert script(nav, [(71, 55)] * 3, dt=.15)[-1].cmd.valid_for_s == pytest.approx(4 / 11)
    nav.suspend(.4)  # a pre-takeoff fight takes the keys in the middle of the pulse
    assert nav._pulse_until is None and nav._aligned_since is None
    log = script(nav, [(71, 55)] * 4, dt=.1, t0=.5)
    assert [f.cmd.move_x for f in log] == ["none", "none", "none", "left"]  # a whole 0.3 s release first
    nav = lap_nav(STAIR_JUMP, B)
    script(nav, [(66, 55)] * 2)  # settling
    nav.suspend(.2)
    log = script(nav, [(66, 55)] * 4, t0=.3)
    assert [f.state for f in log] == ["jump_settling"] * 3 + ["jumping"]  # the settle restarted


def test_corrections_after_a_landing_keep_fine_taps_with_the_far_keys():
    # Drop drop-4-3 landed 4 px beside its point: the correction on the destination floor.
    aside = [(60, 38)] * 4 + [(60, 45), (62, 52)] + [(64, 55)] * 200
    for keys, blocked in ((B, 15.0), (ALL, 15.9)):  # ALL: the proven 0.9 s fall is credited
        nav = lap_nav(DROP, keys)
        log = script(nav, aside)
        correcting = [f for f in log if f.state == "dwelling" and f.cmd.move_x == "left"]
        assert len(correcting) > 20 and max_lease(log) == pytest.approx(.04), keys
        assert nav.reason == "drop timeout" and _blocked_at(log) == pytest.approx(blocked), keys
    # Gap jump accepted without a takeoff (arrive_on_destination), 4 px from its landing X.
    for keys in (B, ALL):
        nav = lap_nav(GAP, keys, arrive_on_destination=True)
        log = script(nav, [(81, 55)] * 60)
        walking = [f for f in log if f.state == "jump_arrived_aligning" and f.cmd.move_x == "right"]
        assert len(walking) >= 5 and max_lease(log) == pytest.approx(.04), keys


def test_regain_support_stage_is_untouched_by_the_keys():
    def regain(keys):  # a return requested on the undrawn entry pipes
        nav = MinimapCircuit(site(start_idx=RAMP + 1, **keys))
        assert nav.request_return_to_start()
        return script(nav, [(65, 98)] * 70, dt=.125)
    plain = regain({})
    walking = [f for f in plain if f.state == "return_regain_walking" and f.cmd.move_x != "none"]
    assert walking and max_lease(plain) == pytest.approx(.25)  # its own leased pulses
    for keys in (FAR, A, B, C, ALL):
        assert digest(regain(keys)) == digest(plain), keys


# --------------------------------------------------------------------------
# 5. recovery_far_approach and failure 2: the return child's 3 px walk under knock-backs
# --------------------------------------------------------------------------

def test_real_return_from_floor_3_right_starts_with_the_three_pixel_walk_and_the_drop():
    plan = plan_return_to_start(site(), (100, 55))
    legs = [(p["action"], p["x"], p.get("connection_id")) for p in plan["waypoints"]]
    assert legs[:3] == [("walk", 100, None), ("walk", 103, None), ("drop", 103, "drop-right-3-2")]
    assert plan["max_retries"] == 0 and plan["movement_test"] and plan["tolerance_y"] == .75


def _child_launched(nav):
    return nav.recovery is not None and nav.recovery.idx == 2 and nav.recovery._drop_launched_at is not None


def _to_first_drop(keys, model, dt=.12, events=None):
    cfg = site(start_idx=RAMP + 1, **keys)
    nav, world = MinimapCircuit(cfg), World(cfg, (100, 55), **model)
    assert nav.request_return_to_start()
    log = drive(nav, world, limit_s=40, dt=dt, events=events(nav) if events else (),
                stop=lambda n: _child_launched(n) or n.recovery_terminal)
    return nav, world, log


def test_return_child_walks_three_pixels_with_fine_taps_only_unless_recovery_far_approach():
    nav, world, log = _to_first_drop({}, {"startup_s": .02})
    child = first(log, lambda f: f.state == "return_walking")
    there = first(log, lambda f: f.pos[0] >= 102)
    assert nav.recovery.approach_fine_radius_x is None and max_lease(log) == pytest.approx(.04)
    assert there.t - child.t > 4  # 100 -> 102 at about 0.3 px/s, exposed to every monster on the way
    nav, world, log = _to_first_drop(FAR, {"startup_s": .02})  # the far keys alone never reach a child
    assert nav.recovery.approach_fine_radius_x is None and max_lease(log) == pytest.approx(.04)


@pytest.mark.parametrize("dt", FRAMES)
def test_recovery_far_approach_closes_the_three_pixels_quickly_and_far_from_the_platform_end(dt):
    slowest = 0.0
    for speed in SPEEDS:
        for coast in COASTS:
            for startup in (0.0, .02):
                nav, world, log = _to_first_drop(C, {"speed": speed, "coast": coast, "startup_s": startup}, dt)
                case = (speed, coast, startup)
                assert _child_launched(nav) and not nav.recovery_terminal, case
                child = first(log, lambda f: f.state == "return_walking")
                there = first(log, lambda f: f.pos[0] >= 102)
                slowest = max(slowest, there.t - child.t)
                assert world.high <= 111 - 1.75 and world.high < 104.5, case  # floor-3-right ends at 111
                key = log[-1]
                assert key.cmd == Command("none", "down", "jump") and abs(key.pos[0] - 103) <= 1.0, case
                settle = [f for f in log[:-1] if f.t >= key.t - .25 - 1e-9]
                assert all(f.cmd == Command.NONE and abs(f.pos[0] - 103) <= 1.0 for f in settle), case
                assert max_lease(log) <= .45 + 1e-9, case
    print(f"\nreturn child 100 -> pixel 102 with recovery_far_approach at {1 / dt:.1f} fps: slowest {slowest:.2f} s")
    assert slowest < 1.5


def _hits(period):
    """A monster knocks the character about until the child's drop is launched."""
    def events(nav):
        spots = [(97, 55), (105.5, 55), (111, 55), (108.5, 55)]  # the displacements seen in lap 3

        def hit(world):
            if nav.recovery is not None and not _child_launched(nav) and not nav.recovery_terminal:
                world.put(spots[hit.count % len(spots)])
                hit.count += 1
        hit.count = 0
        return [(.9 + period * k, hit) for k in range(60)]
    return events


def test_lap3_return_walk_under_knockbacks_fails_with_fine_taps_and_recloses_with_far_pulses():
    slow = {"startup_s": .02}
    nav, world, log = _to_first_drop({}, slow, events=_hits(1.5))
    child = first(log, lambda f: f.state == "return_walking")
    assert nav.recovery_terminal and nav.reason == "Return failed: waypoint timeout" and nav.state == "blocked"
    assert log[-1].t - child.t == pytest.approx(15.0, abs=.25) and world.drops == 0

    nav, world, log = _to_first_drop(ALL, slow, events=_hits(1.5))
    assert _child_launched(nav) and not nav.recovery_terminal
    assert log[-1].cmd == Command("none", "down", "jump") and abs(log[-1].pos[0] - 103) <= 1
    hits = sum(1 for a, b in zip(log, log[1:]) if abs(b.x - a.x) > 2.5)
    print(f"\nreturn child under a hit every 1.5 s, all keys on: {hits} hits, drop launched after "
          f"{log[-1].t - child.t:.2f} s")
    assert hits >= 2 and log[-1].t - child.t < 15  # re-closed after each hit, then the normal settle
    assert world.high <= 111 and world.low >= 89


@pytest.mark.parametrize("keys", [C, ALL])
def test_relentless_hits_still_end_in_the_same_terminal_return_failure(keys):
    nav, world, log = _to_first_drop(keys, {"startup_s": .02}, events=_hits(.5))
    child = first(log, lambda f: f.state == "return_walking")
    # Bounded and terminal exactly as before: guarded Hold follows, nothing is retried or reset.
    # (The walk leg or, once it was reached between two hits, the drop leg runs out of its 15 s.)
    assert nav.recovery_terminal and nav.reason in ("Return failed: waypoint timeout", "Return failed: drop timeout")
    assert 15 - .25 <= log[-1].t - child.t <= 30.5 and nav.idx == RAMP + 1
    assert not nav.returning_to_start and nav.recovery_state == "failed"
    assert nav.step(world.pos, log[-1].t + 1) == Command.NONE and nav.state == "blocked"


def test_children_get_far_pulses_only_by_the_key_and_never_resume_credit():
    def child_of(**keys):
        nav = _to_first_drop(keys, {})[0]
        return nav.recovery
    child = child_of(**ALL, approach_resume_credit=True)
    assert child.approach_fine_radius_x == 1.5 and child.takeoff_far_approach and child.commit_budget
    assert not child.approach_resume_credit and child.max_retries == 0 and child.tolerance_y == .75
    child = child_of(**C, approach_resume_credit=True)
    assert child.approach_fine_radius_x == 1.5 and not child.takeoff_far_approach
    assert not child.approach_resume_credit
    assert child._takeoff_room(child.platforms[0], (50, 105), -3) is None  # its takeoffs keep fine taps
    child = child_of(**B, approach_resume_credit=True)  # takeoff key without the recovery key
    assert child.approach_fine_radius_x is None and not child.takeoff_far_approach
    assert not child.approach_resume_credit


@pytest.mark.parametrize("keys, radius", [({}, None), (B, None), (C, 1.5)])
def test_rejoin_child_uses_far_pulses_only_with_recovery_far_approach(keys, radius):
    cfg = site(start_idx=GAP - 7, arrive_on_destination=True, platform_rejoin=True, approach_resume_credit=True,
               **{**FAR, **keys})
    nav, world = MinimapCircuit(cfg), World(cfg, (95, 71))
    log = drive(nav, world, limit_s=10, stop=lambda n: n.rejoining)
    assert nav.rejoining and nav.platform_rejoin.approach_fine_radius_x == radius
    assert not nav.platform_rejoin.approach_resume_credit and not nav.platform_rejoin.takeoff_far_approach
    log = drive(nav, world, limit_s=20, t0=log[-1].t + .12, stop=lambda n: n.idx == GAP - 5)
    assert nav.idx == GAP - 5 and nav.rejoin_state == "completed" and world.takeoffs == 1


def test_whole_return_with_all_keys_keeps_the_start_proof_and_the_return_bounds():
    cfg = site(start_idx=RAMP + 1, **ALL)
    nav, world = MinimapCircuit(cfg), World(cfg, (100, 55), startup_s=.02)
    assert nav.request_return_to_start() and nav.transition_committed
    log = drive(nav, world, limit_s=70, stop=lambda n: n.state == "returned_to_start" or n.recovery_terminal)
    assert nav.state == "returned_to_start" and nav.idx == 0 and nav.return_count == 1
    assert world.pos == (49, 105) and world.drops == 2
    confirming = [f for f in log if f.state == "return_confirming_start"]
    assert len(confirming) >= 3 and confirming[-1].t - confirming[0].t >= .36  # three fresh stable frames
    assert all(f.committed for f in log[:-1]) and log[-1].t < 30
    # The same world with fine taps only cannot finish a walk leg inside the child's 15 s.
    cfg = site(start_idx=RAMP + 1)
    nav, world = MinimapCircuit(cfg), World(cfg, (100, 55), startup_s=.02)
    assert nav.request_return_to_start()
    drive(nav, world, limit_s=70, stop=lambda n: n.state == "returned_to_start" or n.recovery_terminal)
    assert nav.recovery_terminal and nav.reason == "Return failed: waypoint timeout"
