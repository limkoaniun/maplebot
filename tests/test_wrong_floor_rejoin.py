"""Stably supported on a drawn platform the current waypoint does not expect.

Offline only. Real ``MinimapCircuit`` objects run against a small simulated
world built from the versioned Construction Site geometry
(examples/construction-site/user.yaml, read-only). Scenario A and B replay the
two failures of lap 2 of the supervised 2026-11-16 run: a knock-back across a
0.5 px gap onto a jump's destination, and a knock-back onto the previous
platform of a walk waypoint. Nothing here is a live verification.
"""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

from maplebot.brain.fsm import Brain, Perception
from maplebot.evidence import activity_bucket
from maplebot.nav.circuit import MinimapCircuit, Platform
from maplebot.nav.editor import compile_plan
from maplebot.nav.recovery import plan_rejoin, plan_return_to_start, supported_platform
from maplebot.types import Box, Command, Detection

SITE = Path(__file__).resolve().parents[1] / "examples" / "construction-site" / "user.yaml"
_SITE_CIRCUIT = yaml.safe_load(SITE.read_text(encoding="utf-8"))["route"]["circuit"]
DT = 0.1
ON = {"arrive_on_destination": True, "platform_rejoin": True}


def site(**overrides):
    return {**deepcopy(_SITE_CIRCUIT), **overrides}


class World:
    """Walks, falls and jumps over the drawn platforms; reports whole minimap pixels."""
    WALK, AIR, JUMP_VY, GRAVITY, TICK = 10.0, 10.0, -24.0, 72.0, 0.01

    def __init__(self, cfg, pos):
        self.platforms = [Platform(p["y"], p["left"], p["right"], p.get("right_y"), p.get("knots", ()))
                          for p in cfg["editor_plan"]["platforms"]]
        self.x, self.y = map(float, pos)
        self.vy, self.air = 0.0, False
        self.takeoffs = self.drops = 0
        self.frozen = False      # True: no key moves the character

    def put(self, pos):
        self.x, self.y = map(float, pos)
        self.vy, self.air = 0.0, False

    def _ground(self):
        heights = [p.y_at(self.x) for p in self.platforms
                   if p.left <= self.x <= p.right and abs(p.y_at(self.x) - self.y) <= .6]
        return min(heights, key=lambda h: abs(h - self.y)) if heights else None

    def advance(self, cmd, dt=DT):
        if self.frozen:
            return
        held = dt if cmd.valid_for_s is None else min(dt, max(0.0, cmd.valid_for_s))
        for tick in range(round(dt / self.TICK)):
            sign = ({"left": -1, "right": 1}.get(cmd.move_x, 0)
                    if tick * self.TICK < held - 1e-9 else 0)
            if not self.air:
                if cmd.action == "jump" and cmd.move_y == "down" and tick == 0:
                    self.air, self.vy = True, 0.0  # fall through the ledge
                    self.y += .7
                    self.drops += 1
                elif cmd.action == "jump" and tick == 0:
                    self.air, self.vy = True, self.JUMP_VY
                    self.takeoffs += 1
                else:
                    self.x += sign * self.WALK * self.TICK
                    ground = self._ground()
                    if ground is None:
                        self.air, self.vy = True, 0.0
                    else:
                        self.y = ground
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

    @property
    def pos(self):
        return math.floor(self.x + .5), math.floor(self.y + .5)


class Run:
    """Step one circuit against one world and keep (time, index, state, command, position)."""
    def __init__(self, cfg, pos, start_idx):
        self.nav = MinimapCircuit({**cfg, "start_idx": start_idx})
        self.world = World(cfg, pos)
        self.now = 0.0
        self.rows = []

    def step(self, pos="world"):
        seen = self.world.pos if pos == "world" else pos
        cmd = self.nav.step(seen, self.now)
        self.rows.append((round(self.now, 3), self.nav.idx, self.nav.state, cmd, seen))
        self.world.advance(cmd)
        self.now += DT
        return cmd

    def until(self, done, limit_s=60.0, each=None):
        end = self.now + limit_s
        while self.now < end:
            if each is not None:
                each(self)
            self.step()
            if done(self):
                return self
        raise AssertionError(f"not reached within {limit_s} s; last rows: {self.rows[-6:]}")

    def states(self, since=0):
        collapsed = []
        for _, _, state, _, _ in self.rows[since:]:
            if not collapsed or collapsed[-1] != state:
                collapsed.append(state)
        return collapsed

    def trace(self):
        return [(t, idx, state, (cmd.move_x, cmd.move_y, cmd.action, cmd.valid_for_s), pos)
                for t, idx, state, cmd, pos in self.rows]


def scenario_a(**overrides):
    """Waypoint 18 (index 17): floor-3-gap, knocked onto floor-3 while settling at the takeoff."""
    run = Run(site(**overrides), (93, 55), 16)
    run.until(lambda r: r.nav.idx == 17 and r.nav.state == "jump_settling", 20)
    assert run.world.takeoffs == 0
    run.world.put((87, 55))
    return run, len(run.rows)


def scenario_b(**overrides):
    """Waypoint 11 (index 10): walk on floor-2, knocked back onto floor-2-right."""
    run = Run(site(**overrides), (93, 71), 9)
    run.until(lambda r: r.nav.idx == 10, 20)
    assert run.world.takeoffs == 1 and supported_platform(run.nav._cfg, run.world.pos)["id"] == "floor-2"
    run.world.put((93, 71))
    return run, len(run.rows)


# --- geometry the scenarios rely on --------------------------------------------------------

def test_site_geometry_matches_the_observed_knock_back_positions():
    cfg = site()
    assert supported_platform(cfg, (87, 55))["id"] == "floor-3"
    assert supported_platform(cfg, (93, 71))["id"] == "floor-2-right"
    assert supported_platform(cfg, (88, 71))["id"] == "floor-2"
    for pos in ((89, 55), (89, 71)):  # stair end overlaps the right floor: ambiguous
        with pytest.raises(ValueError, match="ambiguous"):
            supported_platform(cfg, pos)
    for pos in ((91, 54), (89, 70)):
        with pytest.raises(ValueError, match="not supported"):
            supported_platform(cfg, pos)
    points = cfg["waypoints"]
    assert (points[17]["action"], points[17]["connection_id"], points[17]["to_platform"]) == (
        "jump", "floor-3-gap", "floor-3")
    assert (points[10]["action"], points[10]["platform_id"], points[9]["connection_id"]) == (
        "walk", "floor-2", "floor-2-gap")


# --- default off -----------------------------------------------------------------------------

def _legacy_a(**overrides):
    run, since = scenario_a(return_on_block=False, **overrides)
    run.until(lambda r: r.nav.state == "blocked", 40)
    return run, since


def _legacy_b(**overrides):
    run, since = scenario_b(return_on_block=False, **overrides)
    run.until(lambda r: r.nav.state == "blocked", 40)
    return run, since


def test_default_off_scenario_a_waits_retries_and_blocks_without_a_key():
    run, since = _legacy_a()
    assert run.states(since) == ["jump_waiting_floor", "jump_retry", "jump_waiting_floor", "blocked"]
    assert run.nav.reason == "jump timeout" and run.nav.idx == 17
    assert all(cmd == Command.NONE for _, _, _, cmd, _ in run.rows[since:])
    assert run.world.pos == (87, 55) and run.world.takeoffs == 0
    assert (run.nav.arrival_count, run.nav.rejoin_count, run.nav.rejoining) == (0, 0, False)


def test_default_off_scenario_b_waits_and_ends_in_waypoint_timeout():
    run, since = _legacy_b()
    assert run.states(since) == ["waiting_floor", "recovering", "waiting_floor", "blocked"]
    assert run.nav.reason == "waypoint timeout" and run.nav.idx == 10
    assert all(cmd == Command.NONE for _, _, _, cmd, _ in run.rows[since:])
    assert run.nav.rejoin_count == 0 and run.nav.platform_rejoin is None


@pytest.mark.parametrize("legacy", [_legacy_a, _legacy_b])
@pytest.mark.parametrize("variant", [
    {"arrive_on_destination": False, "platform_rejoin": False},
    {**ON, "movement_test": True},  # editor tests and return/rejoin children stay inert
])
def test_keys_absent_false_or_in_a_movement_test_are_step_for_step_identical(legacy, variant):
    baseline, _ = legacy(movement_test=variant.get("movement_test", False))
    other, _ = legacy(**variant)
    assert other.trace() == baseline.trace()
    assert not other.nav._arrive_enabled and not other.nav._rejoin_enabled
    assert (other.nav.arrival_count, other.nav.rejoin_count) == (0, 0)


def test_enabled_keys_change_nothing_on_an_undisturbed_stretch_of_the_lap():
    def stretch(**overrides):
        run = Run(site(**overrides), (96, 71), 7)
        return run.until(lambda r: r.nav.idx == 12, 60)
    plain, enabled = stretch(), stretch(**ON)
    assert enabled.trace() == plain.trace()
    assert plain.world.takeoffs == enabled.world.takeoffs == 1
    assert (enabled.nav.arrival_count, enabled.nav.rejoin_count) == (0, 0)


@pytest.mark.parametrize("key, value", [
    ("arrive_on_destination", 1), ("arrive_on_destination", "true"), ("arrive_on_destination", None),
    ("platform_rejoin", 0), ("platform_rejoin", "false"),
    ("max_platform_rejoins_per_lap", 0), ("max_platform_rejoins_per_lap", 6),
    ("max_platform_rejoins_per_lap", 2.0), ("max_platform_rejoins_per_lap", True),
    ("platform_rejoin_timeout_s", 4.9), ("platform_rejoin_timeout_s", 60.1),
    ("platform_rejoin_timeout_s", float("nan")), ("platform_rejoin_timeout_s", "20"),
    ("platform_rejoin_timeout_s", True),
    ("platform_rejoin_max_transitions", 0), ("platform_rejoin_max_transitions", 5),
    ("platform_rejoin_max_transitions", 1.5)])
def test_configuration_validation(key, value):
    with pytest.raises(ValueError, match=key):
        MinimapCircuit(site(**{key: value}))


def test_configuration_defaults_bounds_and_the_editor_plan_requirement():
    nav = MinimapCircuit(site())
    assert (nav.arrive_on_destination, nav.platform_rejoin_enabled) == (False, False)
    assert (nav.max_platform_rejoins_per_lap, nav.platform_rejoin_timeout_s,
            nav.platform_rejoin_max_transitions) == (3, 20.0, 1)
    assert (nav.rejoining, nav.platform_rejoin, nav.rejoin_count, nav.rejoin_state, nav.rejoin_reason,
            nav.arrival_count) == (False, None, 0, "idle", None, 0)
    nav = MinimapCircuit(site(**ON, max_platform_rejoins_per_lap=5, platform_rejoin_timeout_s=5,
                              platform_rejoin_max_transitions=4))
    assert nav._arrive_enabled and nav._rejoin_enabled
    without_plan = site()
    del without_plan["editor_plan"]
    MinimapCircuit(without_plan)
    MinimapCircuit({**without_plan, **ON, "movement_test": True})
    for key in ON:
        with pytest.raises(ValueError, match="saved editor plan"):
            MinimapCircuit({**without_plan, key: True})


# --- part 1: the jump's destination already supports the character ---------------------------

def feed(nav, pos, start, count, dt=DT):
    """Step fixed observations; returns the commands and the time after the last one."""
    commands = []
    for i in range(count):
        commands.append(nav.step(pos, round(start + i * dt, 6)))
    return commands, round(start + count * dt, 6)


def at_jump(**overrides):
    """Index 17 lined up on floor-3-right, one frame into its settle."""
    nav = MinimapCircuit(site(**{**ON, **overrides, "start_idx": 17}))
    assert nav.step((91, 55), 0) == Command.NONE and nav.state == "jump_settling"
    return nav


def arrived(**overrides):
    """Entry proof passed on floor-3 at (87,55); the landing proof has not started."""
    nav = at_jump(**overrides)
    commands, now = feed(nav, (87, 55), .1, 6)
    assert nav._jump_phase == "arrived" and nav.state == "jump_arrived_checking"
    assert all(c == Command.NONE for c in commands)
    return nav, now


def test_scenario_a_is_accepted_by_destination_proof_without_any_takeoff():
    run, since = scenario_a(**ON)
    run.until(lambda r: r.nav.idx == 18, 15)
    rows = run.rows[since:]
    assert run.states(since) == ["jump_arrived_checking", "jump_arrived_aligning", "jump_arrived"]
    assert run.world.takeoffs == 0 and run.world.drops == 0
    assert not any(cmd.action != "none" or cmd.move_y != "none" for _, _, _, cmd, _ in rows)
    checking = [row for row in rows if row[2] == "jump_arrived_checking"]
    assert len(checking) >= 3 and checking[-1][0] - checking[0][0] >= .5 - 1e-9
    assert all(row[3] == Command.NONE and row[4] == (87, 55) for row in checking)
    aligning = [row for row in rows if row[2] == "jump_arrived_aligning"]
    moves = [row[3] for row in aligning if row[3] != Command.NONE]
    assert moves and all(c.move_x == "left" and c.valid_for_s is not None and c.valid_for_s <= .05
                         for c in moves)  # vision-gated precision taps only
    proof = [row for row in rows if row[2] == "jump_arrived"]
    # Landing proof before the index moves: three frames and the 0.65 s no-descent dwell.
    assert all(row[1] == 17 for row in rows[:-1]) and rows[-1][1] == 18
    assert len(proof) >= 3 and proof[-1][0] - proof[0][0] >= .65 - 1e-9
    assert all(row[3] == Command.NONE and row[4] == (86, 55) for row in proof)
    assert supported_platform(run.nav._cfg, run.world.pos)["id"] == "floor-3"
    nav = run.nav
    assert (nav.arrival_count, nav.retries, nav.return_count, nav.rejoin_count) == (1, 0, 0, 0)
    assert not nav.returning_to_start and nav.reason is None


def test_scenario_a_takes_seconds_where_the_legacy_wait_took_two_timeouts():
    run, since = scenario_a(**ON)
    run.until(lambda r: r.nav.idx == 18, 15)
    assert run.rows[-1][0] - run.rows[since][0] < 5
    legacy, legacy_since = _legacy_a()
    assert legacy.rows[-1][0] - legacy.rows[legacy_since][0] > 29


@pytest.mark.parametrize("times, passes", [((.1, .2), False), ((.1, .3, .5), False),
                                           ((.1, .15, .2, .25, .3, .4, .5), False),
                                           ((.1, .35), True), ((.1, .2, .3, .4, .5), True)])
def test_entry_proof_needs_three_frames_and_half_a_second(times, passes):
    nav = at_jump()
    for now in times:
        assert nav.step((87, 55), now) == Command.NONE
        assert nav.state == "jump_arrived_checking" and nav._jump_phase == "align"
    assert nav.step((87, 55), .6 if passes else .55) == Command.NONE
    assert nav._jump_phase == ("arrived" if passes else "align")


@pytest.mark.parametrize("moved", [(88, 55), (86, 55)])
def test_a_moving_character_restarts_the_entry_proof(moved):
    nav = at_jump()
    feed(nav, (87, 55), .1, 5)
    assert nav.step(moved, .6) == Command.NONE and nav._jump_phase == "align"
    assert nav.step(moved, .7) == Command.NONE and nav.step(moved, 1.0) == Command.NONE
    assert nav._jump_phase == "align" and nav.state == "jump_arrived_checking"
    assert nav.step(moved, 1.1) == Command.NONE and nav._jump_phase == "arrived"


@pytest.mark.parametrize("pos, name", [((88, 54), "unsupported"), ((88, 56), "below the floor"),
                                       ((93, 71), "another platform"), ((70, 52), "the stair above floor-3"),
                                       ((60, 71), "the floor below")])
def test_unsupported_or_other_platform_never_starts_arrival(pos, name):
    nav = at_jump(platform_rejoin=False)
    for i in range(1, 31):
        assert nav.step(pos, i * DT) == Command.NONE
        assert nav.state == "jump_waiting_floor" and nav._jump_phase == "align"
        assert not nav.transition_committed and nav._off_frames == 0
    assert nav.arrival_count == 0 and nav.idx == 17


@pytest.mark.parametrize("pos", [(89, 55), (91, 54)])
def test_ambiguous_overlap_and_takeoff_floor_tolerance_stay_on_the_ordinary_takeoff_path(pos):
    """x=89 is strictly ambiguous (stair end over the right floor); (91,54) is within the
    takeoff floor's own 1 px support rule. Neither is ever treated as an arrival."""
    nav = at_jump()
    for i in range(1, 8):
        nav.step(pos, i * DT)
        assert not nav.state.startswith("jump_arrived") and nav._jump_phase in ("align", "flight")
    assert nav.arrival_count == 0 and nav.idx == 17


def test_support_lost_after_the_entry_proof_latches_nothing_and_a_real_jump_still_needs_flight_proof():
    run, _ = scenario_a(**ON)
    run.until(lambda r: r.nav.state == "jump_arrived", 10)
    run.step(pos=(87, 53))  # knocked into the air: unsupported
    assert run.nav.state == "jump_waiting_floor" and run.nav._jump_phase == "align"
    assert not run.nav._jump_destination_seen and run.nav._jump_landing_frames == 0
    run.world.put((93, 55))  # back on the takeoff floor
    since = len(run.rows)
    run.until(lambda r: r.nav.idx == 18, 20)
    assert run.world.takeoffs == 1 and run.nav.arrival_count == 0
    flown = run.states(since)
    assert flown[flown.index("jumping"):] == ["jumping", "airborne", "jump_landing"]
    assert not any(s.startswith("jump_arrived") for s in flown)
    assert supported_platform(run.nav._cfg, run.world.pos)["id"] == "floor-3"


def test_back_on_the_source_during_arrival_returns_to_the_ordinary_takeoff_path():
    nav, now = arrived()
    assert nav.step((91, 55), now) == Command.NONE and nav.state == "jump_waiting_floor"
    assert nav.step((91, 55), now + .1) == Command.NONE and nav.state == "jump_settling"
    assert nav._jump_phase == "align" and nav.arrival_count == 0


@pytest.mark.parametrize("x, expected", [(85, None), (86, None), (84, None), (82, None),
                                         (81, "right"), (88, "left")])
def test_landing_window_matches_the_flight_rule(x, expected):
    """Within tolerance_x or up to jump_landing_overshoot_x (3) beyond: proof only."""
    nav = at_jump()
    commands, now = feed(nav, (x, 55), .1, 7)
    assert all(c == Command.NONE for c in commands) and nav._jump_phase == "arrived"
    commands, now = feed(nav, (x, 55), now, 12)
    assert not any(c.action != "none" for c in commands)
    if expected is None:
        assert all(c == Command.NONE for c in commands)
        assert nav.idx == 18 and nav.arrival_count == 1
    else:
        assert nav.idx == 17 and nav.state == "jump_arrived_aligning" and nav.arrival_count == 0
        assert {c.move_x for c in commands} == {"none", expected}


@pytest.mark.parametrize("interrupt", ["suspend", "missing"])
@pytest.mark.parametrize("phase", ["checking", "arrived"])
def test_suspend_or_a_missing_position_clears_the_proof(interrupt, phase):
    nav = at_jump()
    feed(nav, (86, 55), .1, 5 if phase == "checking" else 9)
    assert nav._jump_phase == ("align" if phase == "checking" else "arrived")
    if phase == "arrived":
        assert nav._jump_landing_frames >= 2 and nav._jump_destination_seen and nav.idx == 17
    if interrupt == "suspend":
        nav.suspend(1.05)
    else:
        assert nav.step(None, 1.05) == Command.NONE
    assert nav._off_frames == 0 and nav._jump_landing_frames == 0 and not nav._jump_destination_seen
    frames = 0
    while nav.idx == 17:  # the proof starts again from nothing after the interruption
        assert nav.step((86, 55), round(50 + frames * DT, 6)) == Command.NONE and frames < 30
        frames += 1
    elapsed = (frames - 1) * DT
    assert elapsed >= (.5 + .65 if phase == "checking" else .65) - 1e-9 and nav.arrival_count == 1


def test_timeout_during_arrival_uses_the_ordinary_retry_then_block_and_is_not_replenished():
    nav, now = arrived(return_on_block=False)
    states = []
    for i in range(400):  # the character never reaches the landing window
        assert nav.step((88, 55), round(now + i * DT, 6)).action == "none"
        if not states or states[-1] != nav.state:
            states.append(nav.state)
        if nav.state == "blocked":
            break
    assert states == ["jump_arrived_aligning", "jump_retry", "jump_arrived_checking",
                      "jump_arrived_aligning", "blocked"]
    assert nav.reason == "jump timeout" and nav.retries == 1 and nav.arrival_count == 0
    assert 29 < now + i * DT < 33  # two 15 s windows plus the 0.6 s rest


def test_arrival_is_interruptible_only_until_its_entry_proof_passed():
    nav = at_jump()
    seen = {}
    for i in range(1, 30):
        nav.step((87 if i < 12 else 86, 55), i * DT)
        if nav.idx != 17:
            break
        seen.setdefault(nav.state, set()).add((nav._jump_phase, nav.transition_committed))
    assert seen == {"jump_arrived_checking": {("align", False), ("arrived", True)},
                    "jump_arrived_aligning": {("arrived", True)}, "jump_arrived": {("arrived", True)}}


def test_flight_phase_is_unchanged_destination_support_without_departure_is_not_landing():
    nav = at_jump()
    feed(nav, (91, 55), .1, 2)
    assert nav.step((91, 55), .3) == Command("left", "none", "jump") and nav._jump_phase == "flight"
    for now, x in [(.5, 85), (.8, 83), (1.1, 82)]:
        nav.step((x, 55), now)
        assert not nav._jump_destination_seen and nav.idx == 17 and nav.arrival_count == 0


# --- part 2: planner ----------------------------------------------------------------------------

def shape(result):
    return [(p["action"], p.get("connection_id"), p["x"]) for p in result["waypoints"]]


def test_rejoin_plan_for_scenario_b_is_the_saved_gap_jump_and_mutates_nothing():
    cfg = site(**ON)
    before = deepcopy(cfg)
    result = plan_rejoin(cfg, (93, 71), 10, 2)
    assert cfg == before
    assert shape(result) == [("walk", None, 93), ("walk", None, 91), ("jump", "floor-2-gap", 85),
                             ("walk", None, 85)]
    assert result["waypoints"][-1]["platform_id"] == "floor-2" and result["waypoints"][-1]["dwell_s"] == 0
    assert result["movement_test"] and not result["loop"] and not result["return_on_block"]
    assert result["max_retries"] == 0 and result["tolerance_y"] == .75 and result["start_idx"] == 0
    assert result["arrive_on_destination"] is False and result["platform_rejoin"] is False
    child = MinimapCircuit(result, far_approach=False)
    assert not child._arrive_enabled and not child._rejoin_enabled


def test_rejoin_plan_for_a_jump_or_ramp_ends_at_its_saved_takeoff():
    cfg = site()
    jump = plan_rejoin(cfg, (93, 38), 17, 2)  # floor-4-right -> takeoff of floor-3-gap
    assert shape(jump) == [("walk", None, 93), ("walk", None, 103), ("drop", "drop-right-4-3", 103),
                           ("walk", None, 91)]
    assert jump["waypoints"][-1]["platform_id"] == "floor-3-right"
    ramp = plan_rejoin(cfg, (60, 71), 14, 2)  # floor-2 -> takeoff of stair-2-finish on stair-2
    assert ramp["waypoints"][-1]["platform_id"] == "stair-2" and ramp["waypoints"][-1]["x"] == 82
    assert [c for _, c, _ in shape(ramp) if c] == ["jump-2a"]


def test_rejoin_plan_refusals():
    cfg = site()
    with pytest.raises(ValueError, match="too long"):  # drop, jump, ramp
        plan_rejoin(cfg, (80, 55), 16, 2)
    assert len([c for _, c, _ in shape(plan_rejoin(cfg, (80, 55), 16, 3)) if c]) == 3
    with pytest.raises(ValueError, match="Already on"):
        plan_rejoin(cfg, (88, 71), 10, 2)
    for index in (1, 33):  # climb, drop
        with pytest.raises(ValueError, match="does not support"):
            plan_rejoin(cfg, (93, 71), index, 2)
    with pytest.raises(ValueError, match="existing waypoint"):
        plan_rejoin(cfg, (93, 71), 44, 2)
    for pos, match in (((91, 54), "not supported"), ((89, 71), "ambiguous")):
        with pytest.raises(ValueError, match=match):
            plan_rejoin(cfg, pos, 10, 2)
    missing = site()
    del missing["waypoints"][10]["platform_id"]
    with pytest.raises(ValueError, match="no saved platform"):
        plan_rejoin(missing, (93, 71), 10, 2)
    severed = site()
    severed["editor_plan"]["connections"] = [c for c in severed["editor_plan"]["connections"]
                                             if c["from_platform"] != "floor-2-right"]
    with pytest.raises(ValueError, match="No connection"):  # never invents a reverse jump
        plan_rejoin(severed, (93, 71), 10, 2)
    without_plan = site()
    del without_plan["editor_plan"]
    with pytest.raises(ValueError, match="saved editor plan"):
        plan_rejoin(without_plan, (93, 71), 10, 2)


# --- part 2: circuit -----------------------------------------------------------------------------

def rejoin_started(**overrides):
    run, since = scenario_b(**{**ON, **overrides})
    run.until(lambda r: r.nav.rejoining, 5)
    return run, since


def test_scenario_b_physically_repeats_the_saved_jump_and_resumes_the_same_waypoint():
    run, since = scenario_b(**ON)
    started = run.nav._started
    run.until(lambda r: r.nav.rejoining, 5)
    waiting = run.rows[since:]
    # Three stable frames and half a second on floor-2-right, then one input-free frame.
    assert [row[2] for row in waiting] == ["waiting_floor"] * (len(waiting) - 1) + ["rejoin_starting"]
    assert len(waiting) >= 4 and waiting[-1][0] - waiting[0][0] >= .5 - 1e-9
    assert all(row[3] == Command.NONE and row[4] == (93, 71) for row in waiting)
    nav = run.nav
    assert nav.rejoin_count == 1 and nav.rejoin_state == "running" and nav.platform_rejoin is not None
    assert nav.transition_active and nav.transition_committed and not nav.takeoff_interruptible((93, 71))
    run.until(lambda r: not r.nav.rejoining, 30)
    rows = run.rows[since:]
    assert all(row[1] == 10 for row in rows) and nav.idx == 10  # the index never moves
    states = run.states(since)
    assert states[:2] == ["waiting_floor", "rejoin_starting"] and states[-2:] == ["rejoin_confirming", "rejoined"]
    middle = states[2:-2]
    assert all(s.startswith("rejoin_") for s in middle)
    for needed in ("rejoin_walking", "rejoin_jump_aligning", "rejoin_jump_settling", "rejoin_jumping",
                   "rejoin_airborne", "rejoin_jump_landing"):
        assert needed in middle
    assert "blocked" not in states
    # The saved gap jump was really flown a second time, with the child's landing proof.
    assert run.world.takeoffs == 2 and run.world.drops == 0
    jumps = [row for row in rows if row[3].action == "jump"]
    assert jumps and all(row[2] in ("rejoin_jumping", "rejoin_airborne") and row[3].move_x == "left"
                         for row in jumps)
    landing = [row for row in rows if row[2] == "rejoin_jump_landing"]
    assert len(landing) >= 3 and all(supported_platform(nav._cfg, row[4])["id"] == "floor-2" for row in landing)
    confirming = [row for row in rows if row[2] == "rejoin_confirming"]
    assert len(confirming) >= 2 and all(row[3] == Command.NONE for row in confirming)
    assert rows[-1][0] - confirming[0][0] >= .5 - 1e-9
    assert supported_platform(nav._cfg, run.world.pos)["id"] == "floor-2"
    assert (nav.rejoin_state, nav.rejoin_reason, nav.rejoin_count) == ("completed", None, 1)
    assert (nav.retries, nav.return_count, nav.arrival_count) == (0, 0, 0)
    assert not nav.returning_to_start and nav.reason is None and nav._started > started
    assert not nav.transition_active
    # The parent then finishes waypoint 11 itself: dwell, advance.
    since = len(run.rows)
    run.until(lambda r: r.nav.idx == 11, 10)
    assert run.states(since)[0] == "dwelling" and run.rows[-1][0] - run.rows[since][0] >= .6 - 1e-9
    assert nav.rejoin_count == 1 and run.world.takeoffs == 2


@pytest.mark.parametrize("case", ["moving", "airborne", "ambiguous", "missing", "suspended"])
def test_no_rejoin_without_a_stable_strict_wrong_platform(case):
    nav = MinimapCircuit(site(**ON, start_idx=10, return_on_block=False))
    for i in range(60):
        now = i * DT
        if case == "moving":
            pos = (93 + i % 2, 71)
        elif case == "airborne":
            pos = (93, 69)
        elif case == "ambiguous":
            pos = (89, 71)
        elif case == "missing":
            pos = (93, 71) if i % 3 else None
        else:
            pos = (93, 71)
            if i % 3 == 2:
                nav.suspend(now)
                continue
        assert nav.step(pos, now) == Command.NONE
        assert not nav.rejoining and nav.state in ("waiting_floor", "waiting_position")
    assert nav.rejoin_count == 0 and nav.idx == 10


def test_legacy_lower_floor_rejoin_keeps_precedence_step_for_step():
    def fall(**overrides):
        run = Run(site(**overrides), (96, 55), 15)
        run.step()
        run.world.put((93, 71))  # fell to floor-2-right, which has walk waypoints
        for _ in range(12):
            run.step()
        return run
    plain, enabled = fall(), fall(**ON)
    assert enabled.trace() == plain.trace()
    assert plain.nav.idx == 8 and enabled.nav.rejoin_count == 0 and not enabled.nav.rejoining


def test_jump_takeoff_is_rejoined_from_a_third_platform_through_the_saved_drop():
    run = Run(site(**ON), (93, 38), 17)  # floor-4-right; waypoint 18 takes off from floor-3-right
    run.until(lambda r: r.nav.rejoining, 5)
    assert run.states() == ["jump_waiting_floor", "rejoin_starting"]
    run.until(lambda r: not r.nav.rejoining, 30)
    nav = run.nav
    assert nav.rejoin_state == "completed" and nav.idx == 17 and nav.state == "rejoined"
    assert run.world.drops == 1 and run.world.takeoffs == 0
    assert supported_platform(nav._cfg, run.world.pos)["id"] == "floor-3-right"
    run.until(lambda r: r.nav.idx == 18, 30)  # then the parent's own jump, with flight proof
    assert run.world.takeoffs == 1 and nav.arrival_count == 0 and nav.rejoin_count == 1


def test_ramp_takeoff_is_rejoined_from_the_floor_below_its_stair():
    run = Run(site(**ON), (60, 71), 14)  # floor-2; waypoint 15 ramps from stair-2
    run.until(lambda r: r.nav.rejoining, 5)
    assert run.states() == ["ramp_waiting_floor", "rejoin_starting"]
    assert [p.action for p in run.nav.platform_rejoin.waypoints] == ["walk", "walk", "jump", "walk"]


@pytest.mark.parametrize("arrive", [False, True])
def test_the_jumps_own_destination_never_starts_a_rejoin(arrive):
    nav = MinimapCircuit(site(arrive_on_destination=arrive, platform_rejoin=True, start_idx=17))
    nav.step((91, 55), 0)
    commands, _ = feed(nav, (87, 55), .1, 12)
    assert not nav.rejoining and nav.rejoin_count == 0
    assert nav.state == ("jump_arrived_aligning" if arrive else "jump_waiting_floor")
    assert arrive or all(c == Command.NONE for c in commands)


def test_failed_child_is_not_a_block_and_the_waypoint_then_times_out_into_one_normal_return():
    run, since = rejoin_started()
    started = run.nav._started

    def freeze(r):
        if r.nav.state == "rejoin_jump_settling":
            r.world.frozen = True  # the jump key does nothing: a failed flight
    run.until(lambda r: not r.nav.rejoining, 30, each=freeze)
    nav = run.nav
    assert nav.state == "rejoin_failed" and nav.rejoin_state == "failed"
    assert nav.rejoin_reason == "jump timeout" and nav.reason is None
    assert nav.idx == 10 and nav._started == started and nav.retries == 0
    assert (nav.rejoin_count, nav.return_count) == (1, 0) and nav.platform_rejoin is None
    assert not nav.transition_active
    failed_at = len(run.rows)
    run.until(lambda r: r.nav.returning_to_start, 40)
    # No second rejoin at this waypoint; the ordinary wait, retry rest and timeout follow.
    assert run.states(failed_at)[:-1] == ["waiting_floor", "recovering", "waiting_floor"]
    assert all(row[3] == Command.NONE for row in run.rows[failed_at:])
    assert "blocked" not in run.states(since)
    assert nav.rejoin_count == 1 and nav.return_count == 1 and nav.idx == 10
    assert nav.blocked_origin_reason == "waypoint timeout" and not nav.recovery_terminal


def test_rejoin_timeout_counts_stepped_time_only():
    run, _ = rejoin_started(platform_rejoin_timeout_s=5)
    run.world.frozen = True  # the child can never reach its takeoff
    begun = run.now
    for _ in range(20):
        run.step()
    run.nav.suspend(run.now)
    run.now += 500  # e.g. the brain fought here: no circuit time passes
    run.step()
    assert run.nav.rejoining and run.nav.rejoin_state == "running"
    run.until(lambda r: not r.nav.rejoining, 10)
    assert run.nav.rejoin_reason == "Rejoin timed out" and run.nav.state == "rejoin_failed"
    assert 5 <= (run.now - begun - 500) < 5.6 and run.nav.idx == 10 and run.nav.rejoin_count == 1


def test_per_lap_cap_is_spent_by_started_rejoins_and_only_a_completed_lap_restores_it():
    run, _ = scenario_b(**ON, max_platform_rejoins_per_lap=1)
    run.until(lambda r: r.nav.rejoin_state == "completed", 30)
    run.world.put((93, 71))  # knocked back a second time
    since = len(run.rows)
    for _ in range(40):
        run.step()
    assert run.states(since) == ["waiting_floor"] and run.nav.rejoin_count == 1
    nav = run.nav
    nav._set_waypoint(0, run.now)  # what a return to point 1 does
    assert nav.rejoin_count == 1
    nav._set_waypoint(len(nav.waypoints) - 1, run.now)
    nav._advance(run.now)
    assert nav.rejoin_count == 0 and nav.idx == 0


def test_an_unplannable_rejoin_spends_nothing_and_is_latched_for_that_waypoint():
    nav = MinimapCircuit(site(**ON, start_idx=10, return_on_block=False,
                              platform_rejoin_max_transitions=1))
    commands, now = feed(nav, (93, 55), 0, 40)  # floor-3-right: two connections away
    assert all(c == Command.NONE for c in commands)
    assert nav.state == "waiting_floor" and not nav.rejoining and nav.rejoin_count == 0
    assert nav.rejoin_state == "failed" and nav.rejoin_reason == "Rejoin path is too long"
    assert nav._rejoin_failed_idx == 10
    feed(nav, (93, 71), now, 20)  # one connection away, but this waypoint is latched
    assert not nav.rejoining and nav.rejoin_count == 0
    nav._set_waypoint(10, nav._clock.time)  # a new selection of the waypoint clears the latch
    feed(nav, (93, 71), now + 2, 8)
    assert nav.rejoining and nav.rejoin_count == 1


def test_knocked_onto_the_expected_platform_before_the_child_commits_goes_straight_to_confirmation():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_walking", 5)
    run.world.put((86, 71))
    since = len(run.rows)
    run.until(lambda r: not r.nav.rejoining, 5)
    assert run.states(since) == ["rejoin_confirming", "rejoined"]
    assert all(row[3] == Command.NONE for row in run.rows[since:])
    assert run.world.takeoffs == 1 and run.nav.rejoin_state == "completed" and run.nav.idx == 10


def test_a_committed_child_keeps_control_on_the_expected_platform():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_jumping", 20)
    child = run.nav.platform_rejoin
    assert child.transition_committed
    run.step(pos=(86, 71))  # looks supported, but the flight has no landing proof yet
    assert run.nav.rejoining and run.nav.state != "rejoin_confirming" and run.nav.rejoin_state == "running"


def test_confirmation_restarts_when_the_character_leaves_and_a_completed_child_then_fails():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_confirming", 30)
    child = run.nav.platform_rejoin
    run.step(pos=(93, 71))  # knocked off again before the half second passed
    assert run.nav.rejoining and run.nav._off_frames == 0 and run.nav.state.startswith("rejoin_")
    child.completed = True
    run.step(pos=(93, 71))
    assert not run.nav.rejoining and run.nav.state == "rejoin_failed"
    assert run.nav.rejoin_reason == "Left the waypoint platform before confirmation"
    assert run.nav.idx == 10 and run.nav.reason is None


def test_a_return_request_supersedes_the_rejoin_and_spends_exactly_one_return():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_walking", 5)
    nav = run.nav
    assert nav.request_return_to_start()
    assert not nav.rejoining and nav.platform_rejoin is None and nav.rejoin_state == "failed"
    assert nav.rejoin_reason == "Superseded by return to point 1"
    assert nav.returning_to_start and nav.return_count == 1 and nav.idx == 10
    assert not nav.request_return_to_start() and nav.return_count == 1
    run.step()
    assert nav.state.startswith("return_") and nav.rejoin_count == 1


def test_abort_and_explicit_cancel_end_a_rejoin():
    run, _ = rejoin_started()
    run.nav.cancel_platform_rejoin("Map identity or saved settings changed")
    assert not run.nav.rejoining and run.nav.state == "rejoin_failed" and not run.nav.recovery_terminal
    assert run.nav.rejoin_reason == "Map identity or saved settings changed"
    other, _ = rejoin_started()
    other.nav.abort_return_to_start("HP is unreadable")
    assert not other.nav.rejoining and other.nav.recovery_terminal and other.nav.state == "blocked"


def test_suspend_mid_rejoin_keeps_the_rejoin_clears_the_proofs_and_it_can_still_finish():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_jump_settling", 20)
    child = run.nav.platform_rejoin
    before = child._clock.time
    run.nav.suspend(run.now)
    run.now += 100
    assert run.nav.rejoining and child._aligned_since is None and child._clock.time == before
    run.until(lambda r: r.nav.state == "rejoin_confirming", 20)
    run.step()
    assert run.nav._off_frames == 2
    run.nav.suspend(run.now)
    assert run.nav._off_frames == 0 and run.nav.rejoining
    run.now += 100
    since = len(run.rows)
    run.until(lambda r: not r.nav.rejoining, 5)
    assert len(run.rows) - since >= 3 and run.nav.rejoin_state == "completed"
    assert run.world.takeoffs == 2 and run.nav.idx == 10


def test_missing_position_during_a_rejoin_sends_nothing_and_suspends_the_child():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_walking", 5)
    assert run.step(pos=None) == Command.NONE
    assert run.nav.state == "rejoin_waiting_position" and run.nav.rejoining
    run.until(lambda r: not r.nav.rejoining, 30)
    assert run.nav.rejoin_state == "completed"


# --- brain and accounting -----------------------------------------------------------------------

def _brain(start_idx, **overrides):
    from maplebot.config import deep_merge, load_config
    cfg = deep_merge(load_config(), {"attack": {"pre_transition_s": 2.0}, "route": {"circuit": site(
        **{**ON, "start_idx": start_idx, **overrides})}})
    return Brain(cfg, None, 1200)


def _see(pos, now, monster=False):
    detections = [Detection("m", Box(641 + 30 - 20, 346 - 15, 40, 30), .1)] if monster else []
    return Perception(1282, 693, (641, 346), None, pos, 150, False, detections, now, 0)


def test_brain_gives_an_active_rejoin_sole_control_over_combat_and_loot(monkeypatch):
    brain = _brain(10)
    now = 0.0
    while not brain.circuit.rejoining:
        assert brain.decide(_see((93, 71), now)) == Command.NONE and now < 2
        now += DT
    suspended = []
    monkeypatch.setattr(brain, "_choose_target", lambda p: pytest.fail("combat ran during a rejoin"))
    monkeypatch.setattr(brain.loot, "step", lambda *args: pytest.fail("loot ran during a rejoin"))
    monkeypatch.setattr(brain.loot, "suspend", lambda t: suspended.append(t))
    for _ in range(12):
        cmd = brain.decide(_see((93, 71), now, monster=True))
        assert cmd.action == "none" and brain.state.startswith("circuit_rejoin_")
        assert brain.circuit.state != "blocked" and activity_bucket(brain.state) == "traversing"
        now += DT
    assert len(suspended) == 12 and brain.circuit.rejoining and brain.circuit.idx == 10


def test_brain_does_not_attack_once_the_arrival_entry_proof_passed(monkeypatch):
    brain = _brain(17)
    now = 0.0
    while brain.circuit._jump_phase != "arrived":
        brain.decide(_see((87, 55), now))
        assert now < 2
        now += DT
    monkeypatch.setattr(brain, "_choose_target", lambda p: pytest.fail("combat ran during arrival proof"))
    for _ in range(4):
        assert brain.decide(_see((87, 55), now, monster=True)).action == "none"
        assert brain.state == "circuit_jump_arrived_aligning"
        now += DT


def test_activity_bucket_files_rejoin_states_under_traversing():
    assert activity_bucket("circuit_rejoin_walking") == "traversing"
    assert activity_bucket("minimap_route_rejoin_jump_landing") == "traversing"
    assert activity_bucket("circuit_rejoin_failed") == "traversing"
    assert activity_bucket("circuit_return_walking") == "recovering"
    assert activity_bucket("circuit_recovering") == "recovering"
    assert activity_bucket("circuit_jump_arrived") == "traversing"


def test_return_plans_are_unchanged_by_the_new_keys():
    plain, enabled = site(), site(**ON)
    for pos in ((93, 71), (87, 55), (60, 38)):
        a, b = plan_return_to_start(plain, pos), plan_return_to_start(enabled, pos)
        assert a["waypoints"] == b["waypoints"] and b["movement_test"]
        child = MinimapCircuit(b, far_approach=False)
        assert not child._arrive_enabled and not child._rejoin_enabled


# --- review 3: a rejoin's own time never expires the waypoint on its platform --------------------

def jitter(run, a, b, done, limit_s=40.0):
    """Displace the character between two pixels every 0.3 s: supported, never 0.5 s still."""
    end, i = run.now + limit_s, 0
    while run.now < end:
        run.world.put(a if (i // 3) % 2 == 0 else b)
        run.step()
        i += 1
        if done(run):
            return run
    raise AssertionError(f"not reached within {limit_s} s; last rows: {run.rows[-6:]}")


def test_a_rejoin_that_times_out_on_the_walk_waypoints_platform_never_fires_the_stuck_recovery_jump():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_confirming", 30)
    nav, takeoffs = run.nav, run.world.takeoffs
    jitter(run, (85, 71), (87, 71), lambda r: not r.nav.rejoining)
    assert nav.state == "rejoin_failed" and nav.rejoin_reason == "Rejoin timed out"
    assert nav._rejoin_failed_idx == 10
    assert nav._started == pytest.approx(nav._clock.time)  # a fresh window, as after success
    since = len(run.rows)
    jitter(run, (85, 71), (87, 71), lambda r: len(r.rows) - since >= 20)
    assert "recovering" not in run.states(since) and "blocked" not in run.states(since)
    assert not any(row[3].action != "none" for row in run.rows[since:])
    assert (nav.retries, nav.return_count, nav.rejoin_count, nav.idx) == (0, 0, 1, 10)
    run.until(lambda r: r.nav.idx == 11, 20)  # left alone, the waypoint finishes by itself
    assert run.world.takeoffs == takeoffs and nav.retries == 0


def test_a_rejoin_that_times_out_on_the_jump_takeoff_platform_spends_no_retry():
    run = Run(site(**ON), (84, 60.35), 17)  # knocked onto stair-2 below the takeoff floor
    run.until(lambda r: r.nav.rejoining, 5)
    assert [p.action for p in run.nav.platform_rejoin.waypoints] == ["walk", "walk", "ramp", "walk"]
    run.until(lambda r: r.nav.state == "rejoin_confirming", 30)
    nav = run.nav
    assert supported_platform(nav._cfg, run.world.pos)["id"] == "floor-3-right"
    jitter(run, (93, 55), (95, 55), lambda r: not r.nav.rejoining)
    assert nav.state == "rejoin_failed" and nav.rejoin_reason == "Rejoin timed out"
    since = len(run.rows)
    jitter(run, (93, 55), (95, 55), lambda r: len(r.rows) - since >= 20)
    assert "jump_retry" not in run.states(since) and nav.retries == 0 and nav.reason is None
    run.until(lambda r: r.nav.idx == 18, 30)  # its own jump, with flight proof
    assert run.world.takeoffs == 1 and (nav.retries, nav.return_count, nav.arrival_count) == (0, 0, 0)


def test_a_rejoin_that_fails_off_the_expected_platform_keeps_the_waypoint_clock():
    run, _ = rejoin_started(platform_rejoin_timeout_s=5)
    started = run.nav._started
    run.world.frozen = True
    run.until(lambda r: not r.nav.rejoining, 10)
    assert run.nav.rejoin_reason == "Rejoin timed out" and run.nav._started == started


def test_rejoin_timeout_never_cuts_a_committed_flight():
    run, since = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_airborne", 20)
    nav, child = run.nav, run.nav.platform_rejoin
    assert child.transition_committed
    nav._rejoin_started -= 100  # the budget ran out while the character is in the air
    seen = len(run.rows)
    run.until(lambda r: not r.nav.rejoining, 10,
              each=lambda r: None if r.nav.transition_committed and r.nav.transition_active
              else pytest.fail("the flight was released"))
    flight = run.rows[seen:-1]
    assert flight and all(row[2] in ("rejoin_airborne", "rejoin_jump_landing") for row in flight)
    assert any(row[3].move_x == "left" for row in flight)  # the flight keys were still served
    assert not child.transition_committed and child.reason is None  # ended by landing proof
    assert nav.state == "rejoin_failed" and nav.rejoin_reason == "Rejoin timed out"
    assert supported_platform(nav._cfg, run.world.pos)["id"] == "floor-2" and run.world.takeoffs == 2
    assert nav._started == pytest.approx(nav._clock.time) and nav.idx == 10 and nav.retries == 0
    run.until(lambda r: r.nav.idx == 11, 10)
    assert "recovering" not in run.states(since)


def test_rejoin_timeout_during_a_failed_flight_ends_with_the_childs_own_block():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_jumping", 20)
    run.nav._rejoin_started -= 100
    run.world.frozen = True  # the jump never leaves the takeoff floor
    started = run.nav._started
    run.until(lambda r: not r.nav.rejoining, 10)
    assert run.nav.rejoin_reason == "jump timeout" and run.nav.state == "rejoin_failed"
    assert run.nav._started == started and run.nav.reason is None


# --- review 3: knocked onto the jump's destination before the takeoff walk finished --------------

@pytest.mark.parametrize("start, pos, jump", [(16, (87, 55), 17), (15, (87, 55), 17), (24, (87, 38), 25),
                                              (8, (87, 71), 9)])
def test_walk_before_a_jump_hands_over_to_the_jumps_destination_proof(start, pos, jump):
    nav = MinimapCircuit(site(**ON, start_idx=start))
    commands, now = feed(nav, pos, 0, 5)
    assert all(c == Command.NONE for c in commands) and nav.idx == start and nav.state == "waiting_floor"
    assert nav.step(pos, now) == Command.NONE  # three frames and half a second, then no input
    assert nav.idx == jump and nav.state == "jump_arrived_checking" and nav._jump_phase == "align"
    assert (nav.arrival_count, nav.rejoin_count, nav.rejoin_state, nav.retries) == (0, 0, "idle", 0)
    # The jump waypoint proves the destination itself: entry proof, landing X, 0.65 s dwell.
    world = World(nav._cfg, pos)
    now, frames = round(now + DT, 6), 0
    while nav.idx == jump:
        cmd = nav.step(world.pos, now)
        assert cmd.action == "none" and cmd.move_y == "none" and frames < 100
        world.advance(cmd)
        now, frames = round(now + DT, 6), frames + 1
    assert frames * DT >= .5 + .65 - 1e-9 and nav.idx == jump + 1
    assert nav.arrival_count == 1 and world.takeoffs == 0 and nav.rejoin_count == 0
    assert supported_platform(nav._cfg, world.pos)["id"] == nav._wp_meta(jump)["to_platform"]
    assert abs(world.pos[0] - nav.waypoints[jump].x) <= nav.tolerance_x + 3


@pytest.mark.parametrize("times", [(0, .6), (0, .1, .2, .3, .4)])
def test_walk_before_a_jump_needs_the_full_fresh_proof(times):
    nav = MinimapCircuit(site(**ON, start_idx=16))
    for now in times:
        nav.step((87, 55), now)
    assert nav.idx == 16
    nav.step((87, 53), .65)  # airborne: the proof starts again
    for now in (.7, .8, 1.1):
        nav.step((87, 55), now)
        assert nav.idx == 16
    nav.step((87, 55), 1.2)
    assert nav.idx == 17


@pytest.mark.parametrize("start, pos, why", [
    (10, (93, 71), "the next jump does not land where the character stands"),
    (18, (93, 55), "no jump from floor-3 lands on floor-3-right"),
    (16, (89, 55), "ambiguous column"), (16, (87, 54), "unsupported"),
    (16, (60, 38), "another platform than the jump's destination")])
def test_walk_before_a_jump_is_not_skipped_otherwise(start, pos, why):
    nav = MinimapCircuit(site(arrive_on_destination=True, start_idx=start, return_on_block=False))
    feed(nav, pos, 0, 30)
    assert nav.idx == start and nav.arrival_count == 0, why


def test_without_arrive_on_destination_the_same_knock_keeps_the_old_wait_on_floors_3_and_4():
    """Pinned: floor-3 -> floor-3-right needs three saved connections (no floor-3-return jump)."""
    for start, pos, limit in ((16, (87, 55), 1), (16, (87, 55), 2), (24, (87, 38), 2)):
        nav = MinimapCircuit(site(platform_rejoin=True, start_idx=start, return_on_block=False,
                                  platform_rejoin_max_transitions=limit))
        commands, _ = feed(nav, pos, 0, 400)
        assert all(c == Command.NONE for c in commands) and nav.idx == start
        assert (nav.rejoin_count, nav.rejoin_reason) == (0, "Rejoin path is too long")
        assert nav.state == "blocked" and nav.reason == "waypoint timeout"


def test_arrival_walk_crosses_a_stair_toe_column_that_overlaps_the_destination_floor():
    run = Run(site(**ON), (60, 71), 9)  # floor-2, left of the stair-2 toe at x=67
    with pytest.raises(ValueError, match="ambiguous"):
        supported_platform(run.nav._cfg, (67, 71))
    run.until(lambda r: r.nav.idx == 10, 40)
    assert (67, 71) in [row[4] for row in run.rows]
    assert not any(s in run.states() for s in ("jump_waiting_floor", "jump_retry", "blocked"))
    assert run.world.takeoffs == 0 and run.nav.arrival_count == 1 and run.nav.retries == 0
    assert supported_platform(run.nav._cfg, run.world.pos)["id"] == "floor-2"


def test_the_ambiguous_column_never_passes_the_entry_proof_and_other_support_still_ends_arrival():
    nav = MinimapCircuit(site(**ON, start_idx=9))
    feed(nav, (67, 71), 0, 20)
    assert nav.state == "jump_waiting_floor" and nav._jump_phase == "align"
    nav = MinimapCircuit(site(**ON, start_idx=9))
    feed(nav, (60, 71), 0, 7)
    assert nav._jump_phase == "arrived"
    nav.step((67, 71), .7)
    assert nav._jump_phase == "arrived" and nav.state == "jump_arrived_aligning"
    for pos in ((69, 69), (67, 70), (67, 72)):  # on the stair itself / off the floor's strict support
        other = MinimapCircuit(site(**ON, start_idx=9))
        feed(other, (60, 71), 0, 7)
        other.step(pos, .7)
        assert other._jump_phase == "align" and other.state == "jump_waiting_floor"


# --- review 3: the default budget fits the default path length --------------------------------------

def test_defaults_allow_one_connection_and_a_two_connection_ramp_rejoin_is_refused():
    nav = MinimapCircuit(site(**ON, start_idx=14, return_on_block=False))
    assert nav.platform_rejoin_max_transitions == 1
    feed(nav, (88, 55), 0, 20)  # floor-3: drop-3-2, then jump-2a
    assert nav.state == "ramp_waiting_floor" and not nav.rejoining and nav.rejoin_count == 0
    assert nav.rejoin_reason == "Rejoin path is too long"
    with pytest.raises(ValueError, match="too long"):
        plan_rejoin(site(), (88, 55), 14)
    assert [c for _, c, _ in shape(plan_rejoin(site(), (88, 55), 14, 2)) if c] == ["drop-3-2", "jump-2a"]


def test_two_connection_ramp_rejoin_that_times_out_on_the_stair_does_not_block_the_ramp():
    run = Run(site(**ON, platform_rejoin_max_transitions=2, return_on_block=False), (88, 55), 14)
    run.until(lambda r: r.nav.rejoining, 5)
    run.until(lambda r: r.nav.state == "rejoin_confirming", 40)
    assert supported_platform(run.nav._cfg, run.world.pos)["id"] == "stair-2"
    run.nav._rejoin_started = run.nav._clock.time - run.nav.platform_rejoin_timeout_s  # budget just spent
    run.step()
    nav = run.nav
    assert nav.state == "rejoin_failed" and nav.rejoin_reason == "Rejoin timed out"
    run.until(lambda r: r.nav.idx == 15 or r.nav.state == "blocked", 40)
    assert nav.idx == 15 and nav.reason is None and run.world.drops == 1 and run.world.takeoffs == 1


# --- review 3: budgets are untouched, not reset -------------------------------------------------------

def spent_budgets(**overrides):
    """Waypoint 11 after one real waypoint retry, with the lap's single return already used."""
    run, _ = scenario_b(**{**ON, **overrides})
    run.world.put((89, 71))  # ambiguous column: the ordinary wait, no rejoin
    run.until(lambda r: r.nav.state == "recovering", 20)
    run.until(lambda r: r.nav.state == "waiting_floor", 5)
    assert run.nav.retries == 1 and run.nav.rejoin_count == 0 and run.nav.idx == 10
    run.nav.return_count = 1
    run.world.put((93, 71))
    run.until(lambda r: r.nav.rejoining, 5)
    return run


def _then_blocks_without_a_second_return(run):
    nav = run.nav
    assert (nav.retries, nav.return_count, nav.idx) == (1, 1, 10)
    run.world.frozen = False
    run.world.put((89, 71))
    run.until(lambda r: r.nav.state == "blocked", 20)
    assert nav.recovery_terminal and not nav.returning_to_start and nav.return_count == 1
    assert nav.reason.startswith("Return failed: Return budget exhausted") and "waypoint timeout" in nav.reason


def test_successful_rejoin_leaves_spent_retry_and_return_budgets_spent():
    run = spent_budgets()
    run.until(lambda r: not r.nav.rejoining, 30)
    assert run.nav.rejoin_state == "completed" and run.nav._recovery_until is None
    _then_blocks_without_a_second_return(run)


def test_failed_rejoin_leaves_spent_retry_and_return_budgets_spent():
    run = spent_budgets()
    started = run.nav._started

    def freeze(r):
        if r.nav.state == "rejoin_jump_settling":
            r.world.frozen = True
    run.until(lambda r: not r.nav.rejoining, 30, each=freeze)
    assert run.nav.rejoin_state == "failed" and run.nav._started == started
    _then_blocks_without_a_second_return(run)


# --- review 3: brain control at every trigger site ------------------------------------------------------

@pytest.mark.parametrize("start, pos", [(10, (93, 71)), (17, (93, 38)), (14, (60, 71))])
def test_brain_refuses_combat_and_loot_during_a_rejoin_at_walk_jump_and_ramp_waypoints(monkeypatch, start, pos):
    brain = _brain(start)
    now = 0.0
    while not brain.circuit.rejoining:
        assert brain.decide(_see(pos, now)) == Command.NONE and now < 2
        now += DT
    nav = brain.circuit
    assert nav.transition_active and nav.transition_committed and not nav.takeoff_interruptible(pos)
    monkeypatch.setattr(brain, "_choose_target", lambda p: pytest.fail("combat ran during a rejoin"))
    monkeypatch.setattr(brain.loot, "step", lambda *args: pytest.fail("loot ran during a rejoin"))
    for _ in range(12):
        cmd = brain.decide(_see(pos, now, monster=True))
        assert cmd.action == "none" and brain.state.startswith("circuit_rejoin_")
        assert nav.rejoining and nav.idx == start
        now += DT


@pytest.mark.parametrize("start, pos", [(17, (93, 38)), (14, (60, 71))])
def test_a_rejoin_is_committed_at_jump_and_ramp_waypoints(start, pos):
    run = Run(site(**ON), pos, start)
    run.until(lambda r: r.nav.rejoining, 5)
    for _ in range(10):
        assert run.nav.rejoining and run.nav.transition_active and run.nav.transition_committed
        assert not run.nav.takeoff_interruptible(run.world.pos)
        run.step()


# --- review 3: the connection's own destination is never rejoined from ------------------------------------

@pytest.mark.parametrize("limit", [1, 2, 4])
@pytest.mark.parametrize("start, source, pos, waiting", [(17, (91, 55), (87, 55), "jump_waiting_floor"),
                                                         (14, (82, 63), (95, 55), "ramp_waiting_floor")])
def test_standing_on_the_connections_own_destination_never_evaluates_a_rejoin(limit, start, source, pos,
                                                                              waiting):
    nav = MinimapCircuit(site(platform_rejoin=True, start_idx=start, platform_rejoin_max_transitions=limit))
    nav.step(source, 0)
    assert nav.state != waiting
    commands, _ = feed(nav, pos, .1, 12)
    assert all(c == Command.NONE for c in commands) and nav.state == waiting
    assert not nav.rejoining and nav.rejoin_count == 0
    assert nav.rejoin_state == "idle" and nav.rejoin_reason is None and nav._rejoin_failed_idx is None
    assert nav._off_frames == 0


# --- review 3: the fixed proof is three fresh, consecutive frames -------------------------------------------

def test_two_frames_never_pass_the_arrival_entry_proof_however_far_apart():
    nav = at_jump()
    nav.step((87, 55), .1)
    nav.step((87, 55), .7)
    assert nav._jump_phase == "align" and nav.state == "jump_arrived_checking"
    nav.step((87, 55), .8)
    assert nav._jump_phase == "arrived"


def test_two_frames_never_trigger_a_rejoin_however_far_apart():
    nav = MinimapCircuit(site(**ON, start_idx=10))
    nav.step((93, 71), 0)
    nav.step((93, 71), .6)
    assert not nav.rejoining and nav._off_frames == 2
    nav.step((93, 71), .7)
    assert nav.rejoining


def test_two_frames_never_confirm_a_rejoin_however_far_apart():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_confirming", 30)
    assert run.nav._off_frames == 1 and run.nav.platform_rejoin._suspended
    run.now += .5
    run.step()
    assert run.nav.rejoining and run.nav.state == "rejoin_confirming" and run.nav._off_frames == 2
    run.step()
    assert not run.nav.rejoining and run.nav.rejoin_state == "completed"


def test_arrival_landing_proof_needs_three_frames_as_well_as_the_dwell():
    nav = at_jump()
    feed(nav, (86, 55), .1, 6)
    assert nav._jump_phase == "arrived" and nav._jump_landing_frames == 0
    nav.step((86, 55), .7)
    nav.step((86, 55), 1.4)  # 0.7 s later, but only the second frame
    assert nav.idx == 17 and nav._jump_landing_frames == 2 and nav.arrival_count == 0
    nav.step((86, 55), 1.5)
    assert nav.idx == 18 and nav.arrival_count == 1


def test_one_unsupported_frame_restarts_the_trigger_proof_at_the_same_pixel():
    nav = MinimapCircuit(site(**ON, start_idx=10))
    feed(nav, (93, 71), 0, 5)
    nav.step((93, 69), .5)
    assert nav._off_frames == 0
    for now in (.6, .7, .8, .9, 1.0):
        nav.step((93, 71), now)
        assert not nav.rejoining
    nav.step((93, 71), 1.1)
    assert nav.rejoining


@pytest.mark.parametrize("start, wrong, right", [(10, (93, 71), (80, 71)), (17, (93, 38), (95, 55)),
                                                 (14, (60, 71), (76, 65))])
def test_frames_on_the_expected_platform_discard_an_unfinished_trigger_proof(start, wrong, right):
    nav = MinimapCircuit(site(**ON, start_idx=start))
    feed(nav, wrong, 0, 2)
    assert nav._off_frames == 2
    for i in range(2, 11):
        nav.step(right, round(i * DT, 6))
        assert nav._off_frames == 0 and "waiting_floor" not in nav.state
    nav.step(wrong, 1.1)  # one frame back at the same pixel, a second after the first
    assert not nav.rejoining and nav._off_frames == 1 and nav.rejoin_count == 0


def test_a_missing_position_while_confirming_suspends_the_child_and_needs_three_more_frames():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_confirming", 30)
    run.step()
    child = run.nav.platform_rejoin
    assert run.nav._off_frames == 2 and child._suspended
    child._suspended = False
    assert run.step(pos=None) == Command.NONE
    assert run.nav.state == "rejoin_waiting_position" and run.nav._off_frames == 0 and child._suspended
    run.now += 1
    since = len(run.rows)
    run.until(lambda r: not r.nav.rejoining, 5)
    assert len(run.rows) - since >= 3 and run.nav.rejoin_state == "completed"


def test_missing_position_while_the_child_runs_suspends_it():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_walking", 5)
    child = run.nav.platform_rejoin
    assert not child._suspended
    before = child._clock.time
    assert run.step(pos=None) == Command.NONE
    assert child._suspended and child._clock.time == before and not child._clock.active


# --- review 3: guards at the walk trigger site ------------------------------------------------------------

@pytest.mark.parametrize("floor_rejoin_s", [.4, .8, 1.5])
def test_legacy_lower_floor_rejoin_keeps_precedence_for_any_delay(floor_rejoin_s):
    def fall(**overrides):
        run = Run(site(floor_rejoin_s=floor_rejoin_s, **overrides), (96, 55), 15)
        run.step()
        run.world.put((93, 71))
        for _ in range(22):
            run.step()
            if "arrive_on_destination" in overrides:
                assert run.nav.rejoin_state == "idle" and run.nav._rejoin_failed_idx is None
                assert run.nav._off_frames == 0
        return run
    plain, enabled = fall(), fall(**ON, platform_rejoin_max_transitions=4)
    assert enabled.trace() == plain.trace()
    assert 8 in [row[1] for row in plain.rows]  # the legacy index jump happened
    assert enabled.nav.rejoin_count == 0 and not enabled.nav.rejoining


def test_no_rejoin_starts_during_a_walk_waypoints_recovery_rest():
    run, _ = scenario_b(**ON, recovery_s=2.0)
    run.world.put((89, 71))
    run.until(lambda r: r.nav.state == "recovering", 20)
    run.world.put((93, 71))
    rest = 0
    while run.nav._recovery_until is not None and run.nav._clock.time < run.nav._recovery_until - DT:
        run.step()
        rest += 1
        assert not run.nav.rejoining and run.nav._off_frames == 0 and run.nav.state == "recovering"
    assert rest >= 15
    since = len(run.rows)
    run.until(lambda r: r.nav.rejoining, 5)
    assert len(run.rows) - since >= 4  # the proof only started after the rest


@pytest.mark.parametrize("start, pos", [(1, (93, 71)), (1, (60, 88)), (33, (93, 38)), (33, (60, 55))])
def test_climb_and_drop_waypoints_never_evaluate_a_rejoin(start, pos):
    nav = MinimapCircuit(site(**ON, start_idx=start, return_on_block=False,
                              platform_rejoin_max_transitions=4))
    for i in range(40):
        nav.step(pos, round(i * DT, 6))
        assert not nav.rejoining and nav._off_frames == 0
    assert (nav.rejoin_state, nav.rejoin_count, nav._rejoin_failed_idx, nav.arrival_count) == (
        "idle", 0, None, 0)


# --- review 3: accounting and the map hard stop -------------------------------------------------------------

def test_activity_bucket_keeps_a_rejoin_childs_retry_rest_and_block_under_traversing():
    for prefix in ("circuit_", "minimap_route_"):
        for child_state in ("recovering", "blocked", "jump_retry", "waiting_position"):
            assert activity_bucket(prefix + "rejoin_" + child_state) == "traversing"
    assert activity_bucket("circuit_blocked") == "recovering"
    assert activity_bucket("minimap_route_return_recovering") == "recovering"


def _bot_in_a_rejoin():
    import numpy as np
    from maplebot.bot import Bot
    from maplebot.config import deep_merge, load_config
    from maplebot.control.backend import RecordingBackend
    from maplebot.control.keyboard import KeyboardController
    from maplebot.vision.monsters import MonsterLibrary

    cfg = deep_merge(load_config(), {"route": {"circuit": site(**ON, start_idx=10)}})
    keyboard = KeyboardController(cfg, RecordingBackend(), sleep=lambda s: None)

    class Source:
        title = "synthetic"

        def get_frame(self):
            return np.zeros((693, 1282, 3), np.uint8)

    bot = Bot(cfg, Source(), keyboard, MonsterLibrary(), None)
    nav = bot.brain.circuit
    feed(nav, (93, 71), 0, 8)
    assert nav.rejoining and nav.platform_rejoin is not None and nav.rejoin_state == "running"
    return bot, nav


def test_map_hard_stop_cancels_an_active_rejoin_without_a_terminal_block():
    bot, nav = _bot_in_a_rejoin()
    bot.paused = bot.full_pause = False
    bot._hard_stop_map("Map identity or saved settings changed")
    assert not nav.rejoining and nav.platform_rejoin is None
    assert nav.rejoin_state == "failed" and nav.rejoin_reason == "Map identity or saved settings changed"
    assert not nav.recovery_terminal and nav.state == "rejoin_failed" and nav.reason is None
    assert not nav.returning_to_start and (nav.rejoin_count, nav.return_count, nav.idx) == (1, 0, 10)
    assert bot.paused and bot.full_pause and bot.map_reacquire_reason.startswith("Map identity")
    # The old map's child can never be stepped again, and this waypoint is latched.
    assert nav.step((93, 71), 5.0) == Command.NONE and not nav.rejoining


def test_a_confirmation_that_completes_on_the_timeout_frame_is_a_success():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_confirming", 30)
    run.now += .5
    run.step()
    run.nav._rejoin_started -= 100
    run.step()  # third frame: the proof passes on the frame the budget is found spent
    assert not run.nav.rejoining and run.nav.rejoin_state == "completed" and run.nav.state == "rejoined"
    assert run.nav._rejoin_failed_idx is None


def test_a_child_that_blocks_while_the_character_stands_on_the_expected_platform_gives_a_fresh_window():
    run, _ = rejoin_started()
    run.until(lambda r: r.nav.state == "rejoin_jumping", 20)
    while run.nav.rejoining:  # knocked across the gap at launch: no departure, so no flight proof
        assert run.step(pos=(86, 71)).action in ("none", "jump") and run.now < 60
    nav = run.nav
    assert nav.state == "rejoin_failed" and nav.rejoin_reason == "jump timeout" and nav.reason is None
    assert nav._started == pytest.approx(nav._clock.time) and nav._rejoin_failed_idx == 10
    run.world.put((86, 71))
    since = len(run.rows)
    run.until(lambda r: r.nav.idx == 11, 10)
    assert run.states(since) == ["dwelling"] and nav.retries == 0


@pytest.mark.parametrize("index, key, value", [(16, "platform_id", "floor-3"), (17, "from_platform", "floor-2-right"),
                                               (17, "to_platform", "floor-4"), (17, "to_platform", None)])
def test_walk_before_a_jump_only_passes_walks_and_a_takeoff_on_its_own_platform(index, key, value):
    cfg = site(arrive_on_destination=True, start_idx=15, return_on_block=False)
    cfg["waypoints"][index][key] = value
    nav = MinimapCircuit(cfg)
    feed(nav, (87, 55), 0, 30)
    assert nav.idx == 15 and nav.arrival_count == 0
