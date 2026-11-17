"""attack.gap_landing_band_x: the landing edge band after a LEVEL gap jump.

Offline only. Real ``Brain``, ``MinimapCircuit`` and ``LootCollector`` objects run
against the small simulated worlds of tests/test_wrong_floor_rejoin.py and
tests/test_takeoff_zones.py, built from the versioned Construction Site geometry
(examples/construction-site/user.yaml, read-only). The shapes follow the seven
mechanism-3 knocks of the supervised session 10 (knocked back across a level gap
after a confirmed landing, while the Brain hunted, pursued, reacquired or looted at
landing-floor x 84-88). Nothing here is a live verification and no counter here is a
kill, an item or a prevented knock.
"""
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

import maplebot.nav.circuit as circuit_module
from maplebot.brain.fsm import BAND_ENTRY_MARGIN_X, Brain, Perception
from maplebot.config import deep_merge, load_config
from maplebot.nav.circuit import LandingBand, MinimapCircuit, Platform
from maplebot.nav.loot import LootCollector
from maplebot.types import Command
from tests.test_gap_reentry import RE, _floor_2_variant, _plan_variant, mob, see
from tests.test_wrong_floor_rejoin import DT, ON, SITE, Run, World, _legacy_a, _legacy_b, scenario_b, site

ROOT = Path(__file__).resolve().parents[1]
# The live attack geometry (config/user.yaml): default.yaml's range_x 350 / deadzone 30 would
# turn the out-of-range pursuit shapes below into attacks.
LIVE_ATTACK = {"range_x": 70, "range_y": 48, "search_margin": 45, "deadzone_px": 28, "min_overlap_px": 300}
LIVE_LOOT = {"enabled": True, "tolerance_x": 1, "settle_s": 1.2, "max_approach_s": 4, "walk_speed_x": 10.5,
             "max_pending": 2}
BAND_NAMES = ("landing_band", "_landing_band_at")
EXPECTED = {10: ("floor-2", 88.6, 9), 11: ("floor-2", 88.6, 9), 18: ("floor-3", 88.475, 17),
            19: ("floor-3", 88.475, 17), 26: ("floor-4", 88.537, 25), 27: ("floor-4", 88.537, 25)}
NO_BAND_WALKS = [0, 2, 3, 5, 7, 8, 13, 15, 16, 21, 23, 24, 29, 31, 32, 34, 36, 38, 40, 42, 43]
STATE_BAND = ("settling", "approaching", "dwelling")


def band_brain(idx, band=8, priority=None, lap=None, *, pre=6.0, loot=None, attack=None, **circuit):
    """``brain_at`` of tests/test_gap_reentry.py with the live attack/loot geometry and the band keys."""
    keys = {"pre_transition_s": pre, "surface": {"enabled": True}, **LIVE_ATTACK, **(attack or {})}
    if band is not None:
        keys["gap_landing_band_x"] = band
    if priority is not None:
        keys["gap_landing_band_priority_s"] = priority
    if lap is not None:
        keys["gap_landing_band_lap_s"] = lap
    cfg = deep_merge(load_config(), {
        "keys": {"pickup": "z"},
        "loot": {**LIVE_LOOT, **(loot or {})},
        "attack": keys,
        "route": {"circuit": site(**{"start_idx": idx, **circuit})}})
    return Brain(cfg, None, 1200)


def at(mx, pos):
    """A monster standing at minimap x ``mx`` while the character is at ``pos``."""
    return mob(round((mx - pos[0]) * 16))


def perceive(pos, now=0.0, dets=(), *, on_ladder=False, player=(641, 346)):
    return Perception(1282, 693, player, None, pos, 150, on_ladder, list(dets), now, 0)


def surface_ids():
    return {Platform(p["y"], p["left"], p["right"], p.get("right_y"), p.get("knots", ())): p["id"]
            for p in site()["editor_plan"]["platforms"]}


def cmd_row(cmd):
    return (cmd.move_x, cmd.move_y, cmd.action, cmd.face_x,
            None if cmd.valid_for_s is None else round(cmd.valid_for_s, 6))


class Loop:
    """A Brain over one World: (now, idx before, idx after, brain state, command, position) per frame."""

    def __init__(self, brain, pos, dets_for=lambda loop: [], dt=DT):
        self.b, self.nav = brain, brain.circuit
        self.world = World(self.nav._cfg, pos)
        self.dets_for, self.dt = dets_for, dt
        self.now, self.rows = 0.0, []

    def step(self):
        pos, before = self.world.pos, self.nav.idx
        cmd = self.b.decide(see(pos, round(self.now, 6), self.dets_for(self)))
        self.rows.append((round(self.now, 3), before, self.nav.idx, self.b.state, cmd, pos))
        self.world.advance(cmd, self.dt)
        self.now += self.dt
        return cmd

    def run(self, frames):
        for _ in range(frames):
            self.step()
        return self

    def until(self, done, limit_s=60.0):
        end = self.now + limit_s
        while self.now < end:
            self.step()
            if done(self):
                return self
        raise AssertionError(f"not reached within {limit_s} s; last rows: {self.rows[-6:]}")

    def trace(self, since=0):
        return [(t, a, b, state, cmd_row(cmd), pos) for t, a, b, state, cmd, pos in self.rows[since:]]


def digest(rows):
    return hashlib.sha256(repr(rows).encode()).hexdigest()[:16]


# --- T1-T5: geometry and configuration -----------------------------------------------------------

def test_landing_band_comes_from_the_plan():
    ids = surface_ids()
    points = site()["waypoints"]
    walks = []
    for idx, point in enumerate(points):
        nav = MinimapCircuit(site(start_idx=idx))
        band = nav.landing_band()
        if point.get("action", "walk") == "walk":
            walks.append(idx)
        if idx in EXPECTED:
            assert isinstance(band, LandingBand)
            assert (ids[band.platform], band.edge_x, band.toward, band.jump_index) == (
                EXPECTED[idx][0], EXPECTED[idx][1], "right", EXPECTED[idx][2]), idx
            assert nav.landing_band() is band  # cached: the identical object
        else:
            assert band is None, idx
    assert sorted(walks) == sorted(NO_BAND_WALKS + list(EXPECTED))

    def stretch(probe):
        run = Run(site(), (96, 71), 7)
        return run.until(lambda r: r.nav.idx == 20, 90, each=(lambda r: r.nav.landing_band()) if probe else None)
    assert stretch(True).trace() == stretch(False).trace()  # read-only


def test_landing_band_is_none_outside_a_plain_selected_walk():
    nav = MinimapCircuit(site(start_idx=10))
    assert nav.landing_band() is not None
    for name, value in (("completed", True), ("reason", "waypoint timeout"), ("recovery_terminal", True),
                        ("_selected", False), ("rejoining", True), ("returning_to_start", True)):
        probe = MinimapCircuit(site(start_idx=10))
        setattr(probe, name, value)
        assert probe.landing_band() is None, name
    assert MinimapCircuit(site(start_idx=10, movement_test=True)).landing_band() is None
    bare = site(start_idx=10)
    bare["platforms"] = []
    assert MinimapCircuit(bare).landing_band() is None
    # Natively: a return to point 1 and a running rejoin child.
    run, _ = scenario_b()
    assert run.nav.landing_band() is not None and run.nav.request_return_to_start()
    for _ in range(30):
        run.step()
        assert run.nav.landing_band() is None
    run, _ = scenario_b(**ON)
    run.until(lambda r: r.nav.rejoining, 5)
    assert run.nav.landing_band() is None
    assert run.nav.platform_rejoin.landing_band() is None  # the child is a movement test


def test_synthetic_plans():
    from tests.test_gap_veto import R_GAP
    mirror = MinimapCircuit({**R_GAP, "start_idx": 2, "editor_plan": {}})
    band = mirror.landing_band()
    assert (band.toward, band.edge_x, band.jump_index) == ("left", 91.525, 1)
    assert MinimapCircuit({**R_GAP, "start_idx": 0, "editor_plan": {}}).landing_band() is None
    for change in (_floor_2_variant(right=87.0), _floor_2_variant(y=72)):  # a 2 px gap, a 1 px step
        assert MinimapCircuit(_plan_variant(change)).landing_band() is None
    for idx in (7, 13, 34, 36, 38, 40):  # after a ramp, a stair jump, drops and a rope
        assert MinimapCircuit(site(start_idx=idx)).landing_band() is None
    # Guards that are redundant on the example plan, each pinned by a plan where only it decides.
    src, gap_jump = R_GAP["waypoints"][0], R_GAP["waypoints"][1]

    def plan(points, idx, loop=False, extra=()):
        return MinimapCircuit({"start_idx": idx, "loop": loop, "platforms": [*R_GAP["platforms"], *extra],
                               "waypoints": points, "editor_plan": {}})
    landing_walk = {"x": 100, "y": 55, "dwell_s": 0}
    assert plan([src, gap_jump, landing_walk], 2).landing_band() == band  # the reference shape
    # No wrap across waypoint 0: the looped plan's last waypoint is the gap jump onto walk 0's floor.
    wrapped = plan([landing_walk, src, gap_jump], 0, loop=True)
    assert wrapped.landing_band() is None and wrapped._landing_band_at(0) is None
    # A second jump (a hop on the landing floor) right after the gap jump: not a walk, no band.
    hop = {"x": 110, "y": 55, "action": "jump", "from_x": 100, "from_y": 55, "dwell_s": 0}
    hopped = plan([src, gap_jump, hop, {"x": 120, "y": 55, "dwell_s": 0}], 2)
    assert hopped.landing_band() is None and hopped._landing_band_at(2) is not None  # only the walk guard
    assert plan([src, gap_jump, hop, {"x": 120, "y": 55, "dwell_s": 0}], 3).landing_band() is None
    # A walk on a third platform after the jump: the jump did not land on this walk's floor.
    third = {"left": 105, "right": 130, "y": 40}
    assert plan([src, gap_jump, {"x": 120, "y": 40, "dwell_s": 0}], 2, extra=[third]).landing_band() is None
    # A drop on the landing floor between the jump and the walk: only walks may lie between.
    drop = {"x": 98, "y": 55, "action": "drop", "dwell_s": 0}
    assert plan([src, gap_jump, drop, landing_walk], 3).landing_band() is None


@pytest.mark.parametrize("value", [True, 0, 1.9, 12.1, math.nan, math.inf, "8", -8])
def test_band_key_is_validated(value):
    with pytest.raises(ValueError, match="gap_landing_band_x must be a finite number from 2 to 12"):
        band_brain(10, band=value)


@pytest.mark.parametrize("value", [.4, 4.1, -1, math.nan, True, "2"])
def test_priority_key_is_validated_only_with_the_band_on(value):
    with pytest.raises(ValueError, match="gap_landing_band_priority_s must be 0"):
        band_brain(10, priority=value)
    assert band_brain(10, band=None, priority=value)._band_x is None
    assert band_brain(10, band=False, priority=value)._band_priority_s is None


@pytest.mark.parametrize("value", [2.4, 20.1, math.nan, True, "8"])
def test_lap_key_is_validated_only_with_the_band_and_priority_on(value):
    with pytest.raises(ValueError, match="gap_landing_band_lap_s must be finite"):
        band_brain(10, priority=2.5, lap=value)
    assert band_brain(10, band=None, lap=value)._band_lap_s is None
    b = band_brain(10, priority=0, lap=value)
    assert (b._band_x, b._band_priority_s, b._band_lap_s) == (8.0, 0.0, None)


def test_config_defaults_and_plan_requirement():
    b = band_brain(10)
    assert (b._band_x, b._band_priority_s, b._band_lap_s) == (8.0, 2.5, 8.0)
    for value, expected in ((2, 2.0), (12, 12.0), (6.5, 6.5)):
        assert band_brain(10, band=value)._band_x == expected
    assert band_brain(10, priority=.5, lap=.5)._band_lap_s == .5
    assert band_brain(10, priority=4, lap=20)._band_lap_s == 20.0
    off = band_brain(10, band=None)
    assert (off._band_x, off._band_priority_s, off._band_lap_s) == (None, None, None)
    assert band_brain(10, band=False)._band_x is None
    plain = deep_merge(load_config(), {"attack": {"gap_landing_band_x": 8},
                                       "route": {"circuit": site(start_idx=10)}})
    del plain["route"]["circuit"]["editor_plan"]
    with pytest.raises(ValueError, match="gap_landing_band_x needs a saved editor plan"):
        Brain(plain, None, 1200)
    plain["route"]["circuit"]["movement_test"] = True
    assert Brain(deepcopy(plain), None, 1200)._band_x == 8.0  # inert: decide() returns before any band
    plain["route"]["circuit"]["movement_test"] = False
    plain["route"]["circuit"]["enabled"] = False
    assert Brain(plain, None, 1200).circuit is None


def test_membership_jitter_and_ambiguous_columns():
    def band(idx, pos, **kw):
        return band_brain(idx)._edge_band(perceive(pos, **kw))
    for idx, y in ((11, 71), (19, 55), (27, 38)):
        for x in range(81, 89):
            b = band(idx, (x, y))
            assert b.on and b.inside, (idx, x, y)
        for x in (79, 80):
            b = band(idx, (x, y))
            assert b.on and not b.inside, (idx, x, y)
        for x in (89, 90):
            assert not band(idx, (x, y)).on, (idx, x, y)
        assert not band(idx, (84, y), on_ladder=True).on
        assert not band(idx, (84, y), player=None).on
    for idx, pos in ((27, (86, 37)), (19, (88, 54)), (19, (84, 54)), (27, (88, 39)), (11, (84, 70)),
                     (11, (84, 72))):
        assert band(idx, pos).inside, pos
    assert not band(11, (89, 72)).on            # stair-1 / floor-2-right
    assert not band(19, (88, 56)).on            # stair-2 at x 88 is at y 56.83: no band that frame
    b = band(11, (84, 71))
    assert (b.edge_x, b.inner_x, b.toward, b.jump_index) == (88.6, 88.6 - 8, "right", 9)
    assert (b.lo, b.hi) == pytest.approx((88.6 - 8 - 1, 88.6))
    assert band_brain(11)._edge_band(perceive(None)) is None
    assert band_brain(12)._edge_band(perceive((66, 71))) is None   # a jump waypoint
    assert band_brain(11, band=12)._edge_band(perceive((84, 71))).inner_x == pytest.approx(76.6)
    # A fractional width puts a whole pixel exactly on the inner line: that pixel is inside.
    b = band_brain(11, band=8.6)._edge_band(perceive((80, 71)))
    assert b.inner_x == pytest.approx(80.0) and b.inside
    assert not band_brain(11, band=8.6)._edge_band(perceive((79, 71))).inside
    # The leftward mirror: the loot exclusion widens past the inner line by loot.tolerance_x.
    from tests.test_gap_veto import R_GAP
    m = band_brain_for(R_GAP, 2, **LIVE_ATTACK)
    b = m._edge_band(perceive((100, 55)))
    tol = m.loot.tolerance_x
    assert tol > 0 and (b.edge_x, b.inner_x, b.toward) == (91.525, pytest.approx(99.525), "left")
    assert (b.lo, b.hi) == pytest.approx((91.525, 99.525 + tol)) and b.on and not b.inside
    b = band_brain_for(R_GAP, 2, band=7.475, **LIVE_ATTACK)._edge_band(perceive((99, 55)))
    assert b.inner_x == pytest.approx(99.0) and b.inside


def test_narrow_landing_floor_gives_no_band():
    from tests.test_gap_veto import R_GAP
    narrow = deepcopy(R_GAP)
    narrow["platforms"][1] = {"left": 91.525, "right": 105, "y": 55}  # 13.5 px < 2 * 8
    narrow["waypoints"][2]["x"] = 100
    b = band_brain_for(narrow, 2)
    assert b.circuit.landing_band() is not None and b._edge_band(perceive((95, 55))) is None


def band_brain_for(circuit, idx, band=8, priority=None, **attack):
    from tests.test_pre_transition_combat import brain
    keys = {"gap_landing_band_x": band, **attack}
    if priority is not None:
        keys["gap_landing_band_priority_s"] = priority
    return brain({**circuit, "start_idx": idx, "editor_plan": {}}, 6.0, attack=keys)


# --- T6-T15: behaviour ----------------------------------------------------------------------------

def one(b, pos, dets, now=0.0):
    return b.decide(see(pos, now, dets))


def test_combat_never_walks_into_the_band():
    """00:32:32 / 00:51:52 / 00:57:38: hunting walked 2-5 px toward the edge after landing."""
    on, off = band_brain(11), band_brain(11, band=None)
    a, b = one(on, (80, 71), [mob(60)]), one(off, (80, 71), [mob(60)])
    assert a == Command("none", "none", "attack", face_x="right") and on.state == "hunting"
    assert b.move_x == "right" and b.face_x == "none" and off.state == "hunting"
    assert (a.move_y, a.action, a.valid_for_s) == (b.move_y, b.action, b.valid_for_s)
    assert on.edge_band_move_clamps == 1 and off.edge_band_move_clamps == 0
    on, off = band_brain(11), band_brain(11, band=None)
    assert one(on, (79, 71), [mob(60)]).move_x == "right" == one(off, (79, 71), [mob(60)]).move_x
    assert on.edge_band_move_clamps == 0
    # Inside the band the walk-out window would own the frame; priority 0 isolates the clamp.
    on = band_brain(27, priority=0)
    assert one(on, (86, 37), [mob(40)]) == Command("none", "none", "attack", face_x="right")
    assert on.edge_band_move_clamps == 1 and on.edge_band_frames == 1 and on.edge_band_brain_frames == 1
    assert one(band_brain(27, band=None), (86, 37), [mob(40)]).move_x == "right"
    on, off = band_brain(11, priority=0), band_brain(11, band=None)
    a, b = one(on, (84, 71), [mob(-60)]), one(off, (84, 71), [mob(-60)])
    assert a == b and cmd_row(a) == cmd_row(b) and a.move_x == "left" and on.edge_band_move_clamps == 0
    # The rightward mirror: the edge is the landing floor's left end, 91.525; inner line 99.525.
    from tests.test_gap_veto import R_GAP
    on, off = band_brain_for(R_GAP, 2), band_brain_for(R_GAP, 2, band=False)
    a, b = one(on, (100, 55), [mob(-60)]), one(off, (100, 55), [mob(-60)])
    assert a == Command("none", "none", "attack", face_x="left") and b.move_x == "left"
    assert on.edge_band_move_clamps == 1
    on = band_brain_for(R_GAP, 2)
    assert one(on, (101, 55), [mob(-60)]).move_x == "left" and on.edge_band_move_clamps == 0
    # A band exists (a landing-run walk) but the character is not supported on its floor:
    # on the take-off floor beyond the gap, or on a ladder. Never clamped.
    for pos, ladder in (((93, 71), False), ((84, 71), True), ((80, 71), True)):
        rows = []
        for band in (8, None):
            b = band_brain(11, band=band, priority=0)
            cmd = b.decide(replace(see(pos, 0.0, [mob(60)]), on_ladder=ladder))
            rows.append((cmd_row(cmd), b.state))
            if band is not None:
                assert b._band is not None and not b._band.on and b.edge_band_move_clamps == 0, pos
        assert rows[0] == rows[1] and rows[0][0][0] == "right" and rows[0][1] == "hunting", (pos, rows)


def test_pursuit_never_targets_the_band():
    on, off = band_brain(11), band_brain(11, band=None)
    a, b = one(on, (78, 71), [mob(100)]), one(off, (78, 71), [mob(100)])  # target at about 84.3
    assert off.state == "pursuing_target" and b == Command("right")
    assert on.state.startswith("circuit_") and a.action == "none" and a.face_x == "none"
    assert on.edge_band_pursuit_blocks == 1 and off.edge_band_pursuit_blocks == 0
    assert on._band.inner_x - BAND_ENTRY_MARGIN_X < 78 + 100 / 16
    rows = {}
    for name, b in (("on", band_brain(11)), ("off", band_brain(11, band=None))):
        rows[name] = [(cmd_row(one(b, (78, 71), [mob(-100)], round(i * DT, 6))), b.state) for i in range(8)]
        assert b.edge_band_pursuit_blocks == 0
    assert rows["on"] == rows["off"] and rows["on"][0][1] == "pursuing_target"
    # Two band-side detections on one frame are one blocked frame.
    on = band_brain(11)
    one(on, (78, 71), [mob(100), mob(104)])
    assert on.state.startswith("circuit_") and on.edge_band_pursuit_blocks == 1


def _pursuit_pair(make, pos, dx):
    rows = []
    for band in (8, None):
        b = make(band)
        rows.append((cmd_row(one(b, pos, [mob(dx)])), b.state, b.edge_band_pursuit_blocks))
    return rows


@pytest.mark.parametrize("plan, pos, dx, blocked", [
    # floor-2 (rightward band, inner line 80.6, bound 79.6): target 80.0 blocked, 79.0 pursued.
    ("floor-2", (74, 71), 96, True), ("floor-2", (73, 71), 96, False),
    # the leftward mirror (inner line 99.525, bound 100.525): target 100.0 blocked, 101.0 pursued.
    ("mirror", (106, 55), -96, True), ("mirror", (107, 55), -96, False)])
def test_pursuit_bound_keeps_the_one_pixel_margin(plan, pos, dx, blocked):
    from tests.test_gap_veto import R_GAP
    if plan == "floor-2":
        def make(band):
            return band_brain(11, band=band)
    else:
        def make(band):
            return band_brain_for(R_GAP, 2, band=band if band is not None else False, **LIVE_ATTACK)
    on, off = _pursuit_pair(make, pos, dx)
    assert off[1] == "pursuing_target" and off[2] == 0
    if blocked:
        assert on[1].startswith("circuit_") and on[2] == 1 and on[0][2] == "none"
    else:
        assert on == off


@pytest.mark.parametrize("idx, pos, mx", [(11, (87, 71), 81.5), (11, (86, 71), 80.5), (10, (87, 71), 81.5),
                                          (11, (84, 71), 78.5)])
def test_pursuit_away_from_the_edge_is_unchanged_inside_the_band(idx, pos, mx):
    """F1 bounds only steps toward the edge: a west pursuit from inside the band is today's pursuit."""
    rows = []
    for band in (8, None):
        b = band_brain(idx, band=band, priority=0)  # priority 0: the window never owns the frame
        rows.append((cmd_row(one(b, pos, [at(mx, pos)])), b.state, b.edge_band_pursuit_blocks))
        if band is not None:
            assert b._band.inside
    assert rows[0] == rows[1] and rows[0][1] == "pursuing_target" and rows[0][0][0] == "left"
    assert rows[0][2] == 0


def test_loot_exclusion_unit():
    floor3 = Platform(55, 42, 88.475)
    bounds = (42, 88.475)

    def collector():
        c = LootCollector({**LIVE_LOOT, "tolerance_x": 1}, 16)
        c.exclusion = (floor3, 79.475, 88.475)
        return c

    def track(c, goal_x, now=0.0, pos=(80, 55)):
        c.observe(pos, (641, 346), (641 + round((goal_x - pos[0]) * 16), 346), now, None, bounds,
                  platform=floor3, visible_x=[])
        return c
    c = track(collector(), 86)
    assert c.state == "tracking" and c.goal[0] == 86
    states = []
    for i in range(1, 15):
        assert c.step((80, 55), i * DT, False, True, bounds) is None
        states.append(c.state)
    assert c.band_skipped == 1 and "settling" not in states and c.goal is None
    c = track(collector(), 79.0, pos=(82, 55))  # just outside lo = 79.475
    c.step((82, 55), .1, False, True, bounds)
    assert c.state == "settling" and c.band_skipped == 0
    for i in range(2, 6):
        c.step((82, 55), i * DT, False, True, bounds)
    assert c.state == "approaching" and c.goal[0] == 79.0
    c = collector()
    c._pending = [((87.0, 55.0), bounds, floor3, 50.0), ((75.0, 55.0), bounds, floor3, 40.0)]
    c.step((80, 55), 0.0, False, True, bounds)
    assert c.band_skipped == 1 and c.goal == (75.0, 55.0) and c._deadline == 40.0 and c.state == "approaching"
    c = track(collector(), 86)
    c._started = 0.0
    c._defer(1.0)
    assert c.pending_count == 0 and c.band_skipped == 1
    other = Platform(71, 42, 88.6)  # another floor: never excluded
    assert not c._excluded((86.0, 71.0), other) and c._excluded((86.0, 55.0), floor3)
    assert not c._excluded((79.4, 55.0), floor3) and c._excluded((79.475, 55.0), floor3)
    # step()'s band drop clears only the band goal: a remembered site outside the band survives.
    c = track(collector(), 86)
    c._pending = [((75.0, 55.0), bounds, floor3, 40.0)]
    assert c.step((80, 55), .1, False, True, bounds) is None
    assert c.goal is None and c.band_skipped == 1 and c._pending == [((75.0, 55.0), bounds, floor3, 40.0)]

    def drive(c):
        out = []
        for i in range(40):
            now = round(i * DT, 6)
            if i in (0, 12):
                track(c, 84 if i == 0 else 76, now, (80, 55))
            out.append((cmd_row(c.step((80 - (i > 20), 55), now, False, True, bounds) or Command.NONE),
                        c.state, c.goal, c.pending_count))
        return out
    cleared = collector()
    cleared.exclusion = None
    pristine = LootCollector({**LIVE_LOOT, "tolerance_x": 1}, 16)
    assert drive(cleared) == drive(pristine) and cleared.band_skipped == 0
    assert any(row[1] == "approaching" for row in drive(LootCollector({**LIVE_LOOT, "tolerance_x": 1}, 16)))


def _kill_loop(idx, pos, mx, keys, frames=80, alive=8):
    b = band_brain(idx, **keys)
    loop = Loop(b, pos, lambda lp: [at(mx, lp.world.pos)] if len(lp.rows) < alive else [])
    held = []
    for _ in range(frames):
        loop.step()
        goal = b.loot.goal
        held.append(goal is not None and b.loot.state in STATE_BAND and b.loot._platform is not None
                    and b.loot._excluded(goal, b.loot._platform)
                    if getattr(b.loot, "exclusion", None) is not None else False)
    return b, loop, held


def test_band_kills_are_not_looted_outside_kills_are():
    """00:36:01 / 00:52:30: loot visits walked 83-88 toward the edge after landing."""
    on, loop, held = _kill_loop(19, (79, 55), 84, {})
    states = [row[3] for row in loop.rows]
    assert not any(s.startswith("looting_") for s in states) and on.loot.band_skipped >= 1
    assert "hunting" in states and states[-1].startswith("circuit_") and loop.world.pos[0] < 79
    assert not any(held)  # the bot's loot visual lock never holds a band goal
    off, off_loop, _ = _kill_loop(19, (79, 55), 84, {"band": None})
    assert any(row[3].startswith("looting_") for row in off_loop.rows) and off.loot.band_skipped == 0
    for idx, pos, mx in ((19, (79, 55), 76), (38, (79, 71), 84)):
        _, a, _ = _kill_loop(idx, pos, mx, {})
        _, b, _ = _kill_loop(idx, pos, mx, {"band": None})
        assert a.trace() == b.trace(), (idx, mx)
        assert any(row[3].startswith("looting_") for row in a.rows)
    # The exclusion holds on every band frame, also off the landing floor: at (88,56) the
    # character reads as stair-2 (band.on False) and a floor-3 band goal is still dropped.
    for band in (8, None):
        b = band_brain(19, band=band)
        floor = b.circuit._containing_platform((86, 55))
        loot = b.loot
        loot.goal, loot._platform, loot._platform_bounds = (86.0, 55.0), floor, (floor.left, floor.right)
        loot.state, loot._seen = "tracking", 0.0
        one(b, (88, 56), [], .1)
        if band is None:
            assert loot.goal == (86.0, 55.0) and b.state == "looting_settling"
        else:
            assert b._band is not None and not b._band.on and loot.exclusion is not None
            assert loot.goal is None and loot.band_skipped == 1 and not b.state.startswith("looting_")


def _counted_suspends(monkeypatch, b):
    calls = []
    original = b.circuit.suspend
    monkeypatch.setattr(b.circuit, "suspend", lambda now: (calls.append(now), original(now))[1])
    return calls


def _with_pending(b, x=75.0):
    floor = b.circuit._containing_platform((80, 71))
    b.loot._pending = [((x, 71.0), (floor.left, floor.right), floor, 1e9)]
    return floor


def test_window_walks_out_instead_of_hunting(monkeypatch):
    """00:38:26: in-place hunting at 86-87 after the landing, then knocked back across the gap."""
    b = band_brain(10)
    _with_pending(b)
    calls = _counted_suspends(monkeypatch, b)
    loop = Loop(b, (87, 71), lambda lp: [mob(-20)])
    window, clock = [], []
    for _ in range(200):
        frames, suspends = b.edge_band_window_frames, len(calls)
        loop.step()
        if b.edge_band_window_frames == frames:
            break
        window.append(loop.rows[-1])
        clock.append(b.circuit._clock.time)
        assert len(calls) == suspends  # never suspended
        assert b.loot.pending_count == 1
    assert window and all(row[3].startswith("circuit_") for row in window)
    assert all(row[4].action == "none" and row[4].face_x == "none" for row in window)
    assert clock[-1] > clock[0]  # the walk clock runs through the window
    assert loop.world.pos[0] <= 80 and loop.now <= 2 * (b._band_priority_s + DT) + 1e-9
    assert {row[2] for row in window} <= {10, 11} and window[-1][2] == 11
    assert loop.rows[-1][3] == "hunting" and b.loot.pending_count == 1
    assert any(row[3] == "circuit_dwelling" for row in window) and b.edge_band_route_dwell_s > 0
    assert b.edge_band_brain_frames == 0 and b.edge_band_windows_spent == 0
    assert b.edge_band_last["window"] and b.edge_band_last["jump_index"] == 9
    loop.run(20)
    assert b.loot.pending_count == 1  # kept for a later visit
    off = band_brain(10, band=None)
    _with_pending(off)
    rows = Loop(off, (87, 71), lambda lp: [mob(-20)]).run(10).rows
    assert all(row[3] == "hunting" and row[5] == (87, 71) for row in rows)


@pytest.mark.parametrize("first, dx, state", [((80, 71), 60, "hunting"), ((70, 71), 100, "pursuing_target")])
def test_window_clears_combat_and_pursuit_memory(first, dx, state):
    """A fight or pursuit just outside the band, then a window frame: no stale reacquisition after it."""
    rows = {}
    for band in (8, None):
        b = band_brain(11, band=band)
        one(b, first, [mob(dx)], 0.0)
        assert b.state == state
        one(b, (86, 71), [], .1)
        if band is not None:
            assert b.edge_band_last["window"] and b.state.startswith("circuit_")
            assert b._combat_memory is None and b._pursuit_seen is None
        one(b, (80, 71), [], .2)
        rows[band] = b.state
    assert rows[8].startswith("circuit_") and rows[None] == "reacquiring_target"


def test_status_counters_are_exact():
    # inside_s is the summed frame dt of inside frames; frames outside the band add nothing.
    b = band_brain(11, priority=0)
    for now, x in ((0.0, 86), (.1, 86), (.25, 87), (.3, 79), (.45, 84)):
        one(b, (x, 71), [], now)
    assert b.edge_band_frames == 4 and b.edge_band_inside_s == pytest.approx(.1 + .15 + .15)
    assert b.edge_band_last == {"index": 11, "jump_index": 9, "edge_x": 88.6, "inner_x": pytest.approx(80.6),
                                "x": 84, "window": False}
    assert b.edge_band_window_frames == 0
    b = band_brain(10)
    one(b, (86, 71), [], 0.0)
    assert b.edge_band_last == {"index": 10, "jump_index": 9, "edge_x": 88.6, "inner_x": pytest.approx(80.6),
                                "x": 86, "window": True}
    # route_dwell_s counts only circuit_dwelling, never in-band walking frames.
    b = band_brain(10)
    loop = Loop(b, (87, 71))
    loop.run(3)
    assert [row[3] for row in loop.rows] == ["circuit_walking"] * 3 and b._band_inside
    assert b.edge_band_route_dwell_s == 0.0 and b.edge_band_frames == 3
    assert b.edge_band_inside_s == pytest.approx(2 * DT) and b.edge_band_window_s == pytest.approx(2 * DT)
    # The lap cap, not the selection cap, ends a window: priority 2.5, lap 3.0, frozen in the band.
    b = band_brain(10, priority=2.5, lap=3.0)
    loop = _frozen(b, (86, 71))
    loop.run(40)
    assert b._band_lap_spent >= 3.0 and b._band_spent < b._band_priority_s
    assert b.edge_band_windows_spent == 1 and b.edge_band_window_s == pytest.approx(b._band_lap_spent)
    b.circuit._set_waypoint(10, loop.now)  # a new selection: its own allowance is fresh, the lap's is not
    loop.run(5)
    assert b._band_spent == 0.0 and b.edge_band_windows_spent == 1
    assert all(r[3] == "hunting" for r in loop.rows[-5:])


def _frozen(b, pos, dets=lambda lp: [mob(-20)]):
    loop = Loop(b, pos, dets)
    loop.world.frozen = True
    return loop


def test_window_bounds_selection_and_lap():
    b = band_brain(10)
    loop = _frozen(b, (86, 71))
    spent = {}
    for _ in range(80):
        loop.step()
        spent[b.circuit.idx] = b._band_spent
    windowed = [row for row in loop.rows if row[3].startswith("circuit_")]
    first = len(windowed)
    assert [row[3].startswith("circuit_") for row in loop.rows] == [True] * first + [False] * (80 - first)
    assert {row[1] for row in windowed} == {10, 11}
    assert spent[11] <= b._band_priority_s + DT + 1e-9 and b.edge_band_windows_spent == 1
    assert b.edge_band_window_s <= 2 * (b._band_priority_s + DT) and b._band_lap_spent == pytest.approx(
        b.edge_band_window_s)
    off = band_brain(11, band=None)
    off_rows = _frozen(off, (86, 71)).run(80).rows
    assert [(cmd_row(r[4]), r[3], r[2]) for r in loop.rows[first:]] == [
        (cmd_row(r[4]), r[3], r[2]) for r in off_rows[first:]]
    assert all(r[3] == "hunting" for r in loop.rows[first:])
    # Leaving the band and coming back does not re-arm it.
    loop.world.put((79, 71))
    loop.run(5)
    loop.world.put((86, 71))
    loop.run(5)
    assert all(not r[3].startswith("circuit_") for r in loop.rows[-10:]) and b.edge_band_windows_spent == 1
    # A new waypoint selection does.
    b.circuit._set_waypoint(10, loop.now)
    loop.step()
    assert loop.rows[-1][3].startswith("circuit_") and b._band_spent <= DT

    # The lap cap: with lap_s == priority_s the second selection gets no window ...
    b = band_brain(11, priority=2.5, lap=2.5)
    loop = _frozen(b, (86, 71))
    loop.run(40)
    assert b._band_lap_spent >= 2.5 and b.edge_band_windows_spent == 1
    assert b._band_lap_spent <= 2.5 + DT + 1e-9
    b.circuit._set_waypoint(10, loop.now)
    for _ in range(5):
        spent = (b._band_spent, b._band_lap_spent)
        b.suspend(loop.now)  # a Hold never replenishes either allowance
        assert (b._band_spent, b._band_lap_spent) == spent
        loop.step()
        assert loop.rows[-1][3] == "hunting"
    b.circuit.laps_completed += 1  # ... until a lap is completed
    loop.step()
    assert loop.rows[-1][3].startswith("circuit_") and b._band_lap_spent <= DT
    # Another circuit object (the startup-return coordinator's swap) re-arms it too.
    b = band_brain(11, priority=2.5, lap=2.5)
    loop = _frozen(b, (86, 71))
    loop.run(40)
    b.circuit._set_waypoint(10, loop.now)
    loop.step()
    assert loop.rows[-1][3] == "hunting"
    b.circuit = loop.nav = MinimapCircuit(deepcopy(b.circuit._cfg))
    b.circuit._set_waypoint(11, loop.now)
    loop.step()
    assert loop.rows[-1][3].startswith("circuit_")

    # priority 0: no window ever; the clamp and the loot exclusion stay on.
    b = band_brain(10, priority=0)
    loop = _frozen(b, (86, 71)).run(40)
    assert all(r[3] == "hunting" for r in loop.rows) and b.edge_band_window_frames == 0
    assert b.edge_band_brain_frames == 40 and b.loot.exclusion is not None
    one(b, (80, 71), [mob(60)], loop.now)
    assert b.edge_band_move_clamps == 1


@pytest.mark.parametrize("idx, pos, window", [((10), (82, 71), False), (10, (86, 71), True),
                                              (18, (81, 55), True), (26, (84, 38), True),
                                              (11, (84, 71), True), (19, (88, 55), True),
                                              # exactly tolerance_x (1) short of the stop: dwells, so a window
                                              (10, (84, 71), True), (26, (81, 38), True),
                                              (10, (83, 71), False)])
def test_route_out_predicate(idx, pos, window):
    on, off = band_brain(idx), band_brain(idx, band=None)
    a, b = one(on, pos, [mob(-20)]), one(off, pos, [mob(-20)])
    assert on._band.inside and on._band.route_out is window
    assert on.state.startswith("circuit_") is window
    if not window:  # landing at 82-84 during walk 10 (x 85): the route itself walks east
        assert a == b and cmd_row(a) == cmd_row(b) and on.state == "hunting"
        assert on.edge_band_brain_frames == 1


@pytest.mark.parametrize("x, window", [(97, True), (96, True), (98, False)])
def test_route_out_predicate_mirror(x, window):
    """Leftward band (edge 91.525, inner 99.525), stop at x 95, circuit tolerance_x 2."""
    from tests.test_gap_veto import R_GAP
    plan = deepcopy(R_GAP)
    plan["waypoints"][2]["x"] = 95
    b = band_brain_for(plan, 2, **LIVE_ATTACK)
    assert b.circuit.tolerance_x == 2.0
    one(b, (x, 55), [mob(20)])
    assert b._band.inside and b._band.route_out is window
    assert b.state.startswith("circuit_") is window and b.edge_band_window_frames == int(window)


def _begun_visit(b, x=76.0, deadline=5.0):
    floor = b.circuit._containing_platform((80, 71))
    loot = b.loot
    loot.goal, loot._platform, loot._platform_bounds = (x, 71.0), floor, (floor.left, floor.right)
    loot.state, loot._seen, loot._started, loot._empty_since, loot._deadline = "approaching", 0.0, 0.0, -1.0, deadline
    return floor


def test_window_parks_a_begun_visit():
    b = band_brain(10)
    floor = _begun_visit(b)
    b.loot._pending = [((60.0, 71.0), (floor.left, floor.right), floor, 1e9)]
    loop = Loop(b, (86, 71))
    loop.step()
    assert b.state.startswith("circuit_") and b.edge_band_visits_parked == 1
    assert b.loot.goal is None and b.loot.state == "idle"
    assert sorted(site[0][0] for site in b.loot._pending) == [60.0, 76.0]
    assert next(site for site in b.loot._pending if site[0][0] == 76.0)[3] == 5.0  # its first deadline
    loop.until(lambda lp: lp.b.state.startswith("looting_"), 10)
    assert b.loot.goal == (76.0, 71.0) and b.loot._deadline == 5.0 and loop.world.pos[0] <= 80
    assert b.loot.pending_count == 1 and b.edge_band_visits_parked == 1

    b = band_brain(10, loot={"max_pending": 0})
    floor = _begun_visit(b)
    b.loot._pending = [((60.0, 71.0), (floor.left, floor.right), floor, 1e9)]
    Loop(b, (86, 71)).step()
    assert b.state.startswith("circuit_") and b.edge_band_visits_parked == 0
    assert b.loot.goal is None and [site[0][0] for site in b.loot._pending] == [60.0]


@pytest.mark.parametrize("state", STATE_BAND + ("tracking",))
def test_park_parks_only_a_begun_visit_and_resets_its_pulse(state):
    b = band_brain(10)
    _begun_visit(b)
    loot = b.loot
    loot.state = state
    if state == "tracking":
        loot._started = None
    loot._pulse_direction, loot._pulse_until, loot._pulse_on = "left", 9.0, True
    Loop(b, (86, 71)).step()
    assert b.edge_band_last["window"] and b.state.startswith("circuit_")
    assert (loot._pulse_direction, loot._pulse_until, loot._pulse_on) == (None, None, False)
    if state == "tracking":  # a target still being tracked is left to step(), never parked
        assert loot.goal == (76.0, 71.0) and loot.state == "tracking" and loot.pending_count == 0
        assert b.edge_band_visits_parked == 0
    else:
        assert loot.goal is None and loot.state == "idle" and b.edge_band_visits_parked == 1
        assert [(site[0], site[3]) for site in loot._pending] == [((76.0, 71.0), 5.0)]


def test_window_time_never_consumes_loot_time():
    """park() freezes the loot clock: a parked visit whose deadline is shorter than the window survives."""
    b = band_brain(10)
    _begun_visit(b, deadline=.5)
    clock = b.loot._clock.time
    loop = Loop(b, (86, 71))
    while loop.now < 5 and (not loop.rows or loop.rows[-1][3].startswith("circuit_")):
        loop.step()
        if loop.rows[-1][3].startswith("circuit_"):
            assert b.loot._clock.time == clock
    window = [row for row in loop.rows if row[3].startswith("circuit_")]
    assert len(window) >= 10 and b.edge_band_window_s > .5  # longer than the visit's whole deadline
    assert b.edge_band_visits_parked == 1
    loop.until(lambda lp: lp.b.state.startswith("looting_"), 10)
    assert b.loot.goal == (76.0, 71.0) and b.loot._deadline == .5
    # park() itself, called alone across 2.9 s, never advances the clock.
    loot = LootCollector(LIVE_LOOT, 16)
    before = loot._clock.time
    for i in range(30):
        loot.park(i * DT)
    assert loot._clock.time == before


def test_transition_frames_never_see_a_band():
    rows = {}
    for name, keys in (("on", {}), ("off", {"band": None})):
        b = band_brain(11, **keys)
        first = one(b, (80, 71), [mob(60)])
        clamps = b.edge_band_move_clamps
        b.circuit._set_waypoint(12, .1)
        out = []
        for i in range(40):
            cmd = one(b, (66, 71), [mob(60)], round(.2 + i * DT, 6))
            out.append((cmd_row(cmd), b.state, b.circuit.idx, b.circuit.state))
            assert b._band is None and b.loot.exclusion is None
        assert b.edge_band_move_clamps == clamps
        rows[name] = (first.move_x, out)
    assert rows["on"][0] == "none" and rows["off"][0] == "right"
    assert rows["on"][1] == rows["off"][1] and any(r[1] == "hunting" for r in rows["on"][1])
    for start, pos, jump in ((8, (93, 71), 9), (16, (93, 55), 17), (24, (93, 38), 25)):
        traces = []
        for keys in ({}, {"band": None}):
            b = band_brain(start, **keys)
            loop = Loop(b, pos, lambda lp, j=jump: [mob(60)] if lp.nav.idx == j else [])
            loop.until(lambda lp, j=jump: lp.nav.idx == j + 1, 40)
            traces.append(loop.trace())
            assert any(row[1] == jump and row[3] == "hunting" for row in loop.rows)  # the pre-takeoff fight
            assert b.edge_band_frames == 0 and b.edge_band_move_clamps == 0
        assert traces[0] == traces[1]

    def veto_frames(pos, veto, band):
        attack = {} if veto is None else {"pre_transition_gap_veto_x": veto}
        b = band_brain(17, band=band, attack=attack)
        return [(cmd_row(one(b, pos, [mob(70)], round(i * DT, 6))), b.state, b.circuit.state) for i in range(12)]
    for pos in ((89, 55), (89, 56), (93, 55)):
        for veto in (6.0, None):
            assert veto_frames(pos, veto, 8) == veto_frames(pos, veto, None), (pos, veto)


def test_reentry_precedes_and_never_overlaps(monkeypatch):
    import tests.test_gap_reentry as reentry
    # _window_run (tests/test_gap_reentry.py) builds its brain through brain_at: band off, then on.
    monkeypatch.setattr(reentry, "brain_at", lambda idx, **keys: band_brain(idx, band=None, **keys))
    with_band, out = reentry._window_run(monkeypatch, RE, [(93, 71)] * 20)
    monkeypatch.setattr(reentry, "brain_at", lambda idx, **keys: band_brain(idx, **keys))
    banded, out_band = reentry._window_run(monkeypatch, RE, [(93, 71)] * 20)
    assert banded._band_x == 8.0 and with_band._band_x is None
    assert banded.reentry_priority_frames == with_band.reentry_priority_frames > 0
    assert [(r["cmd"], r["state"], r["idx"]) for r in out_band] == [(r["cmd"], r["state"], r["idx"]) for r in out]
    assert banded.edge_band_frames == 0 and banded.circuit.reentry_count == 1
    entry = next(i for i, row in enumerate(out_band) if row["idx"] == 8)
    assert [r["state"] for r in out_band[:entry + 1]] == [r["state"] for r in out[:entry + 1]]
    # After the re-entry and the re-landing, walk 10 gets a fresh window.
    b = band_brain(10, **RE)
    loop = Loop(b, (93, 71))
    overlap = []
    while len(loop.rows) < 600 and not (b.circuit.reentry_count == 1 and b.edge_band_window_frames):
        pending = b.circuit.reentry_pending(loop.world.pos)
        loop.step()
        overlap.append(pending and b._band_inside)
    assert b.circuit.reentry_count == 1 and b.edge_band_window_frames > 0 and b.circuit.idx in (10, 11)
    assert b.reentry_priority_frames > 0 and not any(overlap)
    assert loop.rows[-1][3].startswith("circuit_")


# --- T16-T22: identity and safety -------------------------------------------------------------------

def brain_lap(keys, laps=1, limit_s=900.0, dt=.12):
    """A whole lap from point 1 through ``Brain.decide`` with no detections and loot off."""
    from tests.test_takeoff_zones import LapWorld
    from tests.test_transition_timing import Frame
    cfg = deep_merge(load_config(), {"attack": {**LIVE_ATTACK, **keys},
                                     "route": {"circuit": site(start_idx=0)}})
    b = Brain(cfg, None, 1200)
    nav = b.circuit
    world = LapWorld(nav._cfg, (49, 105))
    log, wraps, previous = [], 0, nav.idx
    for frame in range(int(limit_s / dt) + 1):
        t = round(frame * dt, 6)
        cmd = b.decide(see(world.pos, t))
        log.append(Frame(t, world.x, world.y, world.pos, cmd, nav.state, nav.idx, nav.transition_committed))
        if previous == len(nav.waypoints) - 1 and nav.idx == 0 and nav.state != "returned_to_start":
            wraps += 1
        previous = nav.idx
        if wraps >= laps or nav.recovery_terminal:
            break
        world.advance(cmd, dt)
    return b, log, wraps


def test_undisturbed_laps_identical_with_the_key_on():
    from tests.test_takeoff_zones import PINNED as ZONES_PINNED, off_scenario
    from tests.test_transition_timing import PINNED as TIMING_PINNED, digest as frame_digest, scenario

    on, log_on, wraps = brain_lap({"gap_landing_band_x": 8})
    _, log_off, _ = brain_lap({})
    assert wraps == 1 and log_on == log_off and len(log_on) > 300
    assert [cmd_row(f.cmd) for f in log_on] == [cmd_row(f.cmd) for f in log_off]
    assert on.edge_band_frames > 0 and on.edge_band_window_frames > 0  # the band was there, the route alone ran
    assert on.edge_band_move_clamps == 0 and on.edge_band_brain_frames == 0
    # The circuit-only lap through the Brain is the pinned lap of the zones differential.
    assert frame_digest(log_on) == ZONES_PINNED["lap"]
    for name, value in TIMING_PINNED.items():
        assert frame_digest(scenario(name)) == value, name
    for name in ("lap", "knocked lap", "lap + other keys"):
        assert frame_digest(off_scenario(name)) == ZONES_PINNED[name], name


def test_laps_completed_counter():
    from tests.test_takeoff_zones import LapWorld
    from tests.test_transition_timing import drive
    cfg = site(start_idx=0)
    nav = MinimapCircuit(cfg)
    assert nav.laps_completed == 0
    seen = {"previous": nav.idx, "wraps": 0, "counts": []}

    def stop(n):
        if seen["previous"] == len(n.waypoints) - 1 and n.idx == 0:
            seen["wraps"] += 1
            seen["counts"].append(n.laps_completed)
        seen["previous"] = n.idx
        return seen["wraps"] >= 2
    drive(nav, LapWorld(cfg, (49, 105)), limit_s=1800, dt=.12, stop=stop)
    assert seen["wraps"] == 2 and seen["counts"] == [1, 2] and nav.laps_completed == 2
    # Never on a re-entry, a return or a rejoin.
    run, _ = scenario_b(**RE)
    run.until(lambda r: r.nav.state == "reentered", 5)
    run.until(lambda r: r.nav.idx == 11, 30)
    assert run.nav.reentry_count == 1 and run.nav.laps_completed == 0
    from tests.test_transition_timing import _returning
    returned, _, _ = _returning(MinimapCircuit, {}, {}, 40)
    assert returned.state == "returned_to_start" and returned.idx == 0 and returned.laps_completed == 0
    run, _ = scenario_b(**ON)
    run.until(lambda r: r.nav.rejoining, 5)
    run.until(lambda r: not r.nav.rejoining, 30)
    assert run.nav.rejoin_state == "completed" and run.nav.laps_completed == 0


def test_filtered_detections_unchanged():
    """test_gap_reentry.py's cross-gap detection: at (93,71) a monster at landing x 86 is never attacked."""
    seen = []
    for keys in (ON, RE):
        b = band_brain(10, **keys)
        original = b._choose_target
        b._choose_target = lambda p, o=original: (seen.append(list(p.detections)), o(p))[1]
        for i in range(40):
            cmd = one(b, (93, 71), [mob((86 - 93) * 16)], round(i * DT, 6))
            assert cmd.action == "none" and cmd.face_x == "none" and b.state != "hunting"
            # No band on frames the re-entry window or a rejoin owns; never "on" from across the gap.
            assert (b._band is None or not b._band.on) and b.edge_band_frames == 0
    assert seen and all(d == [] for d in seen)


def _band_cfg(original):
    def wrapped(**over):
        cfg = original(**over)
        circuit = cfg.get("route", {}).get("circuit")
        if isinstance(circuit, dict) and circuit.get("enabled", True) and not isinstance(
                circuit.get("editor_plan"), dict):
            circuit["editor_plan"] = {}
        return deep_merge(cfg, {"attack": {"gap_landing_band_x": 8}})
    return wrapped


def test_hp_and_hold_unchanged_with_the_band_on(monkeypatch):
    import tests.test_farming_hold as hold
    import tests.test_health_protection_integration as integration

    monkeypatch.setattr(integration, "_cfg", _band_cfg(integration._cfg))
    game = integration._Game(monkeypatch)
    assert game.bot.brain._band_x == 8.0
    integration.test_normal_low_hp_holds_until_target_then_resumes_farming(monkeypatch)
    integration.test_three_actual_ineffective_hp_taps_latch_attention_and_stop_retrying(monkeypatch)
    for terminal in (False, True):
        hold.test_manual_hold_keeps_route_stopped_and_allows_only_hp_even_with_failure_opt_out(
            monkeypatch, terminal)
    game = integration._Game(monkeypatch)
    game.bot.hold_farming("band test")
    monkeypatch.setattr(game.bot.brain, "decide", lambda p: pytest.fail("decide called on a held frame"))
    for hp in (75, 55, 55, 65, 75):
        snap = game.step(hp)
        assert snap.command == Command.NONE
    assert game.bot.keyboard.hp_taps_sent >= 1 and game.bot.brain._band_spent == 0.0
    game.only_hp_pressed()


GUARD_NAMES = {"MinimapCircuit": ("landing_band", "_landing_band_at"), "Brain": ("_edge_band",),
               "LootCollector": ("_excluded", "park")}


def _raise(*args, **kwargs):
    raise AssertionError("a landing edge band helper was called with the key absent")


@pytest.fixture
def guarded(monkeypatch):
    for cls in (MinimapCircuit, Brain, LootCollector):
        for name in GUARD_NAMES[cls.__name__]:
            monkeypatch.setattr(cls, name, _raise)


def test_key_absent_never_calls_a_band_helper(guarded, monkeypatch):
    from tests.test_gap_reentry import wrong_floor_with_keys_on
    from tests.test_takeoff_zones import PINNED as ZONES_PINNED, off_scenario
    from tests.test_transition_timing import PINNED as TIMING_PINNED, digest as frame_digest, scenario

    for name, value in TIMING_PINNED.items():
        assert frame_digest(scenario(name)) == value, name
    for name in ("knocked lap", "knocked lap + other keys", "rejoin + other keys", "scripted jump",
                 "return from stair-2"):
        assert frame_digest(off_scenario(name)) == ZONES_PINNED[name], name
    for legacy in (_legacy_a, _legacy_b):
        legacy()
    wrong_floor_with_keys_on()
    for name in BRAIN_PINNED:
        assert digest(brain_scenario(name)) == BRAIN_PINNED[name], name
    with pytest.raises(AssertionError, match="band helper was called"):  # the guard itself works
        one(band_brain(11), (84, 71), [])


GUARD_PLUGIN = '''
import pytest

def pytest_configure(config):
    import maplebot.nav.circuit as circuit
    import maplebot.nav.loot as loot
    from maplebot.brain.fsm import Brain

    def boom(*args, **kwargs):
        raise AssertionError("a landing edge band helper was called with the key absent")
    for name in ("landing_band", "_landing_band_at"):
        setattr(circuit.MinimapCircuit, name, boom)
    Brain._edge_band = boom
    loot.LootCollector._excluded = boom
    loot.LootCollector.park = boom
    with open({marker!r}, "w", encoding="utf-8") as handle:
        handle.write("guarded landing_band _landing_band_at _edge_band _excluded park")
'''


def test_whole_combat_loot_and_fsm_files_never_call_a_band_helper_with_the_key_absent(tmp_path):
    plugin = tmp_path / "band_guard_plugin.py"
    marker = tmp_path / "guarded.txt"
    plugin.write_text(GUARD_PLUGIN.format(marker=str(marker)), encoding="utf-8")
    files = ["tests/test_pre_transition_combat.py", "tests/test_fsm.py", "tests/test_surface_targeting.py",
             "tests/test_gap_reentry.py", "tests/test_gap_veto.py", "tests/test_loot.py",
             "tests/test_loot_precision.py", "tests/test_loot_sites.py", "tests/test_loot_visual_gap.py"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
         "-p", "band_guard_plugin", "--basetemp", str(tmp_path / "inner"),
         "--deselect", "tests/test_gap_reentry.py::"
         "test_whole_combat_and_fsm_files_never_call_a_new_helper_with_the_keys_absent", *files],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(ROOT)})
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert marker.read_text(encoding="utf-8").startswith("guarded landing_band")  # the plugin really ran
    passed = re.search(r"(\d+) passed", result.stdout)
    assert passed and int(passed.group(1)) >= 50 and "failed" not in result.stdout


def test_shipped_configs_leave_the_band_off():
    names = {"gap_landing_band_x", "gap_landing_band_priority_s", "gap_landing_band_lap_s"}
    assert not names & set(load_config()["attack"])
    text = SITE.read_text(encoding="utf-8")
    example = yaml.safe_load(text)
    assert not names & set(example["attack"])
    assert example["route"]["circuit"]["editor_reference_sha256"] == (
        "b131badde8048e46b7f79abe17f4be1b6e966dc8295619fca0834e38aa964427")
    assert "# gap_landing_band_x: 8" in text
    default = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    assert "# gap_landing_band_x: false" in default and "gap_landing_band_lap_s (8; up to 20)" in default
    handoff = (ROOT / "docs" / "DEVELOPMENT_HANDOFF.md").read_text(encoding="utf-8")
    assert "Landing edge band (`attack.gap_landing_band_x`" in handoff and "test_gap_landing_band.py" in handoff


def test_status_export_fields():
    source = (ROOT / "scripts" / "run_active.py").read_text(encoding="utf-8")
    ast.parse(source)
    for word in ("'edge_band'", "'laps_completed'", "'inside_s'", "'brain_frames'", "'window_s'",
                 "'windows_spent'", "'move_clamps'", "'loot_skipped'", "'route_dwell_s'", "'pursuit_blocks'",
                 "'visits_parked'", "'lap_spent_s'", "'window_frames'", "'last'"):
        assert word in source, word
    assert "if getattr(bot.brain, '_band_x', None) is not None else None" in source


# --- key absent: Brain-level sequences pinned against the code before the band -------------------

def brain_scenario(name, **attack):
    """Brain-driven runs over the landing floors; ``attack`` adds keys (absent = the band off)."""
    keys = {"band": None, "attack": attack}
    if name == "walk 10 west mob, remembered site":
        b = band_brain(10, **keys)
        _with_pending(b)
        return Loop(b, (87, 71), lambda lp: [mob(-20)]).run(60).trace()
    if name == "walk 19 kill at 84":
        return _kill_loop(19, (79, 55), 84, keys)[1].trace()
    if name == "walk 19 kill at 76":
        return _kill_loop(19, (79, 55), 76, keys)[1].trace()
    if name == "walk 11 east pursuit":
        b = band_brain(11, **keys)
        return Loop(b, (78, 71), lambda lp: [at(84.25, lp.world.pos)] if len(lp.rows) < 15 else []).run(60).trace()
    if name == "walk 27 east mob":
        b = band_brain(27, **keys)
        return Loop(b, (82, 38), lambda lp: [at(86.5, lp.world.pos)] if len(lp.rows) % 7 < 5 else []).run(80).trace()
    if name == "landing run from 8 with east monsters":
        b = band_brain(8, **keys)
        dets = (lambda lp: [at(87, lp.world.pos)] if lp.nav.idx in (10, 11) and len(lp.rows) % 9 < 6
                else [mob(60)] if lp.nav.idx == 9 and len(lp.rows) % 5 < 3 else [])
        return Loop(b, (93, 71), dets).run(400).trace()
    if name == "begun visit at walk 10":
        b = band_brain(10, **keys)
        _begun_visit(b)
        return Loop(b, (86, 71)).run(60).trace()
    raise KeyError(name)


# Digests of the frame-by-frame (time, index before/after, brain state, command incl. face and
# lease, position) sequences produced on 2026-11-18 by maplebot/ as it was BEFORE the band
# (commit 45e0d3e): an exported copy of that package ran this same ``brain_scenario``.
BRAIN_PINNED = {
    "walk 10 west mob, remembered site": "dd93da70c26c9b88",
    "walk 19 kill at 84": "a315ac82b9f6183d",
    "walk 19 kill at 76": "ea7299d7e3f8e1b0",
    "walk 11 east pursuit": "18f0f86486a409c9",
    "walk 27 east mob": "41b616662eecb737",
    "landing run from 8 with east monsters": "34acacb0c0179b10",
    "begun visit at walk 10": "06b34012725e492c",
}


@pytest.mark.parametrize("name", sorted(BRAIN_PINNED))
def test_key_absent_brain_sequences_equal_the_code_before_the_band(name):
    assert digest(brain_scenario(name)) == BRAIN_PINNED[name]
    # Bogus priority/lap values are not read while the band is off.
    assert digest(brain_scenario(name, gap_landing_band_priority_s="x",
                                 gap_landing_band_lap_s=math.nan)) == BRAIN_PINNED[name]
    assert digest(brain_scenario(name, gap_landing_band_x=False)) == BRAIN_PINNED[name]


@pytest.mark.parametrize("name", sorted(BRAIN_PINNED))
def test_key_on_changes_the_landing_shapes_only_in_the_band(name):
    on = brain_scenario(name, gap_landing_band_x=8)
    off = brain_scenario(name)
    assert len(on) == len(off)
    for row_on, row_off in zip(on, off):
        if row_on != row_off:
            break
    else:
        return
    # The first difference is a frame on a landing walk of a level gap jump.
    assert row_on[1] in EXPECTED, (row_on, row_off)
