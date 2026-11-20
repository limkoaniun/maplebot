"""route.circuit.gap_reentry: a landing walk knocked back across a level gap re-runs the gap jump.

Offline only. Real ``MinimapCircuit`` and ``Brain`` objects run against the small
simulated world of tests/test_wrong_floor_rejoin.py, built from the versioned
Construction Site geometry (examples/construction-site/user.yaml, read-only). The
knock positions follow the supervised 2026-11-17 run (20:16:35 at walk 10,
20:17:49 at walk 18 with the rejoin cap spent). Nothing here is a live verification.
"""
import ast
import inspect
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import maplebot.nav.circuit as circuit_module
import maplebot.nav.zones as zones
from maplebot.brain.fsm import Brain, Perception
from maplebot.config import deep_merge, load_config
from maplebot.nav.circuit import MinimapCircuit, Platform
from maplebot.nav.recovery import plan_rejoin, plan_return_to_start, supported_platform
from maplebot.types import Box, Command, Detection
from tests.test_wrong_floor_rejoin import (DT, ON, SITE, Run, World, _legacy_a, _legacy_b, feed, scenario_a,
                                           scenario_b, site)

RE = {**ON, "gap_reentry": True}
ROOT = Path(__file__).resolve().parents[1]
REENTRY_NAMES = ("_reentry_target", "reentry_pending", "_do_reentry", "level_gap_takeoff")


def collapsed(values):
    out = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def nav_trace(nav, positions, start=0.0, dt=DT):
    rows = []
    for i, pos in enumerate(positions):
        now = round(start + i * dt, 6)
        cmd = nav.step(pos, now)
        rows.append((now, nav.idx, nav.state, cmd, nav.rejoining, nav.reason))
    return rows


# --- brain harness ---------------------------------------------------------------------------

def brain_at(start_idx, *, pre=6.0, veto=None, **circuit):
    attack = {"pre_transition_s": pre, "surface": {"enabled": True}}
    if veto is not None:
        attack["pre_transition_gap_veto_x"] = veto
    cfg = deep_merge(load_config(), {
        "keys": {"pickup": "z"},
        "loot": {"enabled": True, "walk_speed_x": 9.0, "max_pending": 2},
        "attack": attack,
        "route": {"circuit": site(**{"start_idx": start_idx, **circuit})}})
    return Brain(cfg, None, 1200)


def mob(dx, dy=0, w=40, h=30):
    """A monster centred ``dx`` screen px beside the player, its sprite bottom on the player's floor."""
    return Detection("m", Box(641 + dx - w // 2, 346 + dy - h // 2, w, h), .1)


def see(pos, now, dets=()):
    return Perception(1282, 693, (641, 346), None, pos, 150, False, list(dets), now, 0)


# --- 1-3: the circuit re-runs the gap jump ---------------------------------------------------

def test_landing_walk_knocked_back_across_floor_2_gap_reenters_walk_8():
    run, since = scenario_b(**RE)
    started = run.nav._started
    run.until(lambda r: r.nav.idx == 8, 5)
    waiting, entry = run.rows[since:-1], run.rows[-1]
    assert [row[2] for row in waiting] == ["waiting_floor"] * len(waiting)
    assert all(row[3] == Command.NONE and row[4] == (93, 71) and row[1] == 10 for row in waiting)
    # The unchanged proof: three fresh frames and half a second, then the re-entry frame itself.
    assert len(waiting) >= 3 and .5 - 1e-9 <= entry[0] - waiting[0][0] <= .5 + 3 * DT + 1e-9
    assert entry[2] == "reentered" and entry[3] == Command.NONE
    nav = run.nav
    assert (nav.reentry_count, nav.rejoin_count, nav.rejoining, nav.retries) == (1, 0, False, 0)
    assert nav.reentry_last == {"from_index": 10, "to_index": 8, "platform": "floor-2-right"}
    assert nav._started > started and nav._zone_open and nav._rejoin_failed_idx is None
    since = len(run.rows)
    run.until(lambda r: r.nav.idx == 11, 30)
    rows = run.rows[since:]
    assert collapsed(row[1] for row in rows) == [8, 9, 10, 11]
    jumped = [row for row in rows if row[3].action == "jump"]
    assert jumped and all(row[1] == 9 and row[4][0] in (90, 91, 92) for row in jumped[:1])
    assert run.world.takeoffs == 2 and supported_platform(nav._cfg, run.world.pos)["id"] == "floor-2"
    assert all(row[2] != "blocked" for row in run.rows) and nav.reason is None and nav.retries == 0
    assert (nav.reentry_count, nav.rejoin_count, nav.return_count, nav.arrival_count) == (1, 0, 0, 0)


def test_floor_3_landing_walk_knock_reenters_walk_16():
    run = Run(site(**RE), (93, 55), 16)
    run.until(lambda r: r.nav.idx == 18, 30)
    run.world.put((99, 55))
    since = len(run.rows)
    run.until(lambda r: r.nav.idx == 19, 40)
    assert collapsed(row[1] for row in run.rows[since:]) == [18, 16, 17, 18, 19]
    assert run.nav.reentry_last == {"from_index": 18, "to_index": 16, "platform": "floor-3-right"}
    assert run.nav.reentry_count == 1 and run.nav.rejoin_count == 0 and run.world.takeoffs == 2


@pytest.mark.parametrize("enabled", [True, False])
def test_cap_spent_knock_back_reenters_instead_of_timing_out(enabled):
    """The 20:17:49 shape: knocked back across the gap with the lap's rejoin cap spent."""
    run, since = scenario_b(**(RE if enabled else ON), return_on_block=False)
    run.nav.rejoin_count = 3
    if enabled:
        run.until(lambda r: r.nav.idx == 8, 1.0)
        assert run.nav.state == "reentered" and run.nav.rejoin_count == 3
        indices = []
        for _ in range(400):  # 40 s: never the 'waypoint timeout' of the lap lost at 20:18:23
            run.step()
            indices.append(run.nav.idx)
            assert run.nav.reason != "waypoint timeout"
        assert collapsed(indices)[:4] == [8, 9, 10, 11] and max(indices) >= 12
        assert run.world.takeoffs >= 2 and run.nav.reentry_count == 1 and run.nav.rejoin_count == 3
    else:
        run.until(lambda r: r.nav.state == "blocked", 40)
        assert run.states(since) == ["waiting_floor", "recovering", "waiting_floor", "blocked"]
        assert run.nav.reason == "waypoint timeout" and run.nav.idx == 10


# --- 4-6: the brain's bounded priority window -------------------------------------------------

def _window_run(monkeypatch, keys, positions, dets_for=lambda i: [mob(60)]):
    b = brain_at(10, **keys)
    nav = b.circuit
    floor = nav._containing_platform((96, 71))
    b.loot._pending = [((100.0, 71.0), (floor.left, floor.right), floor, 1e9)]
    suspended = []
    original = nav.suspend
    monkeypatch.setattr(nav, "suspend", lambda now: (suspended.append(now), original(now))[1])
    out = []
    for i, pos in enumerate(positions):
        before = nav._off_frames
        cmd = b.decide(see(pos, round(i * DT, 6), dets_for(i)))
        out.append({"cmd": cmd, "state": b.state, "idx": nav.idx, "off": (before, nav._off_frames),
                    "suspends": len(suspended), "pending": b.loot.pending_count})
    return b, out


def test_brain_gives_pending_reentry_bounded_priority_over_hunting_and_loot(monkeypatch):
    """The 20:16:35 shape: a same-platform monster beside the character, a remembered loot site."""
    b, out = _window_run(monkeypatch, RE, [(93, 71)] * 20)
    entry = next(i for i, row in enumerate(out) if row["idx"] == 8)
    window = out[:entry + 1]
    assert entry * DT <= b.circuit.gap_reentry_priority_s
    assert [row["state"] for row in window] == ["circuit_waiting_floor"] * entry + ["circuit_reentered"]
    for row in window:
        assert row["cmd"].move_x == "none" and row["cmd"].action == "none" and row["cmd"].face_x == "none"
        assert row["suspends"] == 0 and row["pending"] == 0
    assert all(row["off"][1] == row["off"][0] + 1 for row in window[:-1])
    assert window[-1]["off"][1] == 0  # the re-entry frame resets the proof with the waypoint
    assert b.reentry_priority_frames == entry + 1 and b.circuit.reentry_count == 1
    # Afterwards the walk 8 is on the same floor as the monster: today's code fights it.
    assert out[entry + 1]["state"] == "hunting" and out[entry + 1]["suspends"] == 1
    assert not any(row["state"].startswith("looting_") for row in out)
    b_off, off = _window_run(monkeypatch, ON, [(93, 71)] * 20)
    assert all(row["state"] == "hunting" and row["idx"] == 10 for row in off)
    assert b_off.reentry_priority_frames == 0 and b_off.circuit.reentry_count == 0


def test_priority_window_is_bounded_then_equals_today(monkeypatch):
    jitter = [(93 + i % 2, 71) for i in range(40)]  # moves a pixel every frame: no proof ever
    b, out = _window_run(monkeypatch, RE, jitter)
    window = [i for i, row in enumerate(out) if row["state"].startswith("circuit_")]
    assert window == list(range(len(window))) and window
    assert (len(window) - 1) * DT <= b.circuit.gap_reentry_priority_s + DT + 1e-9  # first frame dt is 0
    assert b._reentry_spent <= b.circuit.gap_reentry_priority_s + DT + 1e-9
    assert b.reentry_priority_frames == len(window) and b.circuit.reentry_count == 0
    assert all(out[i]["suspends"] == 0 for i in window)
    _, off = _window_run(monkeypatch, ON, jitter)
    after = len(window)
    assert [(row["cmd"], row["state"], row["idx"]) for row in out[after:]] == [
        (row["cmd"], row["state"], row["idx"]) for row in off[after:]]
    assert all(row["state"] == "hunting" for row in out[after:])


def test_single_offroute_call_per_frame(monkeypatch):
    nav = MinimapCircuit(site(**RE, start_idx=10))
    calls = []
    original = nav._offroute_stable
    monkeypatch.setattr(nav, "_offroute_stable", lambda *a: (calls.append(a), original(*a))[1])
    monkeypatch.setattr(nav, "_platform_rejoin_due", lambda *a: pytest.fail("rejoin evaluated"))
    frames = 0
    while nav.idx == 10:
        assert nav.reentry_pending((93, 71))
        before = len(calls)
        nav.step((93, 71), round(frames * DT, 6))
        assert len(calls) == before + 1
        frames += 1
        assert frames < 20
    assert nav.idx == 8 and len(calls) == frames and all(call[1] == "floor-2-right" for call in calls)


# --- 7-8: bounds -------------------------------------------------------------------------------

def test_budget_per_lap():
    run, _ = scenario_b(**RE)
    for count in (1, 2):
        run.until(lambda r: r.nav.state == "reentered", 5)
        assert run.nav.reentry_count == count and run.nav.idx == 8
        run.until(lambda r: r.nav.idx == 10, 30)
        run.world.put((93, 71))
    run.until(lambda r: r.nav.rejoining, 5)  # the third knock takes today's rejoin path
    nav = run.nav
    assert (nav.reentry_count, nav.rejoin_count, nav.idx) == (2, 1, 10)
    run.until(lambda r: not r.nav.rejoining, 30)
    assert nav.rejoin_state == "completed" and nav.reentry_count == 2
    nav.retries = 1
    nav._set_waypoint(0, run.now)  # what a return to point 1 does
    assert nav.reentry_count == 2 and nav.retries == 0
    nav._set_waypoint(len(nav.waypoints) - 1, run.now)
    assert nav.reentry_count == 2
    nav._advance(run.now)
    assert nav.reentry_count == 0 and nav.idx == 0 and nav.rejoin_count == 0


def test_replenishment_is_bounded_and_stated():
    b = brain_at(16, pre=6.0, **RE)
    nav = b.circuit
    world = World(nav._cfg, (93, 55))
    selections, retries, frozen = [], {}, 0.0
    original = nav._set_waypoint

    def record(idx, now):
        selections.append(idx)
        original(idx, now)
    nav._set_waypoint = record  # instance attribute: the circuit's own calls go through it
    knocks, now, seen_landing = 0, 0.0, False
    while nav.idx != 19:
        assert now < 200, (nav.idx, nav.state, b.state)
        if nav.idx == 18 and not seen_landing and nav.state != "rejoined" and not nav.rejoining:
            seen_landing = True
            if knocks < 3:
                world.put((99, 55))
                knocks += 1
        if nav.idx != 18:
            seen_landing = False
        dets = [mob(60)] if nav.idx == 17 else []
        cmd = b.decide(see(world.pos, now, dets))
        if nav.idx == 17:
            selection = selections.count(17)
            retries[selection] = max(nav.retries, retries.get(selection, 0))
            if b.state in ("hunting", "reacquiring_target"):
                frozen += b._frame_dt
        world.advance(cmd)
        now = round(now + DT, 6)
    assert knocks == 3 and nav.reentry_count == 2 and nav.rejoin_count == 1
    assert selections.count(17) <= 3 and selections.count(17) == 3
    assert all(value <= nav.max_retries for value in retries.values())
    assert frozen <= 3 * b._pre_transition_s + 3 * DT + 1e-9 and frozen > 2 * b._pre_transition_s
    # The stated contract: the docstring and config/default.yaml carry the bound.
    doc = MinimapCircuit._do_reentry.__doc__
    for words in ("max_gap_reentries_per_lap", "_rejoin_failed_idx", "timeout_s", "walk g+1", "100 s", "200 s"):
        assert words in doc
    default = (ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    handoff = (ROOT / "docs" / "DEVELOPMENT_HANDOFF.md").read_text(encoding="utf-8")
    assert "max_gap_reentries_per_lap: 2" in default and "100 s per" in default and "200 s per lap" in default
    assert "roughly 100 s per re-entry and 200 s per lap" in handoff
    # Review: the earlier 70 s / 140 s figure left out the landing walk's fresh window.
    for text in (doc, default):
        assert "70 s per" not in text and "140 s per" not in text


def _stall_until_near_timeout(run, idx, park, margin=1.2, limit=80):
    """Hold the world at ``park`` while ``idx`` is current until its last window nearly expires."""
    nav = run.nav
    run.world.put(park)
    run.world.frozen = True
    end = run.now + limit
    while run.now < end:
        run.step()
        assert nav.idx == idx and nav.reason is None, run.rows[-3:]
        if (nav.retries == nav.max_retries and nav._recovery_until is None
                and getattr(nav, "_jump_phase", None) != "retry"
                and nav._clock.time - nav._started >= nav.timeout_s - margin):
            run.world.frozen = False
            return
    raise AssertionError("no near-timeout")


def test_reentry_bound_counts_the_fresh_landing_walk_window():
    """Review finding: walk g+1 gets a whole new window too, so the stated bound is three windows.

    Walk 10 is knocked back 1.2 s before its own 'waypoint timeout'; walk 8 and jump 9 are
    held until shortly before their last windows end; walk 10 then runs a full window again.
    Key off the same knock blocks within about 1.2 s.
    """
    run = Run(site(**RE, return_on_block=False), (93, 71), 9)
    run.until(lambda r: r.nav.idx == 10, 20)
    _stall_until_near_timeout(run, 10, (75, 71))
    run.world.put((93, 71))
    t_knock = run.nav._clock.time
    run.until(lambda r: r.nav.state == "reentered", 2)
    assert run.nav.idx == 8 and run.nav.reentry_count == 1
    _stall_until_near_timeout(run, 8, (95, 71), margin=5.0)
    run.until(lambda r: r.nav.idx == 9, 10)
    _stall_until_near_timeout(run, 9, (92, 71), margin=8.0)
    run.until(lambda r: r.nav.idx == 10, 20)
    t_second = run.nav._clock.time
    run.world.put((75, 71))
    run.world.frozen = True
    run.until(lambda r: r.nav.reason == "waypoint timeout", 40)
    nav = run.nav
    window = nav.timeout_s * (nav.max_retries + 1) + nav.recovery_s
    assert nav._clock.time - t_second >= 2 * nav.timeout_s  # the landing walk's fresh window
    assert 2 * window < nav._clock.time - t_knock <= 3 * window + 10  # three windows plus flight


def test_key_off_same_late_knock_blocks_within_the_remaining_window():
    run = Run(site(arrive_on_destination=True, platform_rejoin=True, return_on_block=False,
                   max_platform_rejoins_per_lap=1), (93, 71), 9)
    run.nav.rejoin_count = 1  # cap spent: today's wait
    run.until(lambda r: r.nav.idx == 10, 20)
    _stall_until_near_timeout(run, 10, (75, 71))
    run.world.put((93, 71))
    t_knock = run.nav._clock.time
    run.until(lambda r: r.nav.reason == "waypoint timeout", 5)
    assert run.nav._clock.time - t_knock < 2


def test_priority_window_reopens_for_the_next_waypoint_selection():
    """Review finding: the window is per waypoint selection, not once per Brain."""
    b = brain_at(10, **RE)
    nav = b.circuit
    now = 0.0
    for i in range(30):  # 1 px jitter: no proof; the 1.5 s window is spent at idx 10
        b.decide(see((93 + i % 2, 71), round(now, 6), [mob(60)]))
        now += DT
    assert b.state == "hunting" and b._reentry_spent >= nav.gap_reentry_priority_s
    nav._set_waypoint(11, now)  # a new selection that also leads back across the floor-2 gap
    b.decide(see((93, 71), round(now, 6), [mob(60)]))
    assert b.state.startswith("circuit_"), b.state


def test_no_reentry_pending_while_the_walk_recovers():
    """Review finding: the brain window must not open during the walk's recovery rest."""
    b = brain_at(10, **RE, return_on_block=False)
    nav = b.circuit
    now = 0.0
    while nav._recovery_until is None:  # stuck on floor-2 short of walk 10: timeout, then the rest
        b.decide(see((70, 71), round(now, 6)))
        now += DT
        assert now < 30
    assert not nav.reentry_pending((93, 71))
    b.decide(see((93, 71), round(now, 6), [mob(60)]))  # knocked across during the rest
    assert b.state == "hunting" and b.reentry_priority_frames == 0


def test_no_reentry_target_across_waypoint_0():
    """A plan whose waypoint 0 is the gap jump: walk 1 never re-enters the last waypoint."""
    cfg = site(**RE)
    rotated = {**cfg, "waypoints": cfg["waypoints"][9:] + cfg["waypoints"][:9], "start_idx": 1}
    nav = MinimapCircuit(rotated)
    assert nav.waypoints[0].action == "jump" and nav._wp_meta(1)["platform_id"] == "floor-2"
    assert nav._wp_meta(len(nav.waypoints) - 1)["platform_id"] == "floor-2-right"
    assert nav._reentry_target((93, 71)) is None
    assert MinimapCircuit({**cfg, "start_idx": 10})._reentry_target((93, 71)) == (8, "floor-2-right")


def test_rejoin_failed_latch_is_cleared_only_by_set_waypoint():
    tree = ast.parse(inspect.getsource(circuit_module))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MinimapCircuit")
    writes = {}
    for method in (node for node in cls.body if isinstance(node, ast.FunctionDef)):
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Attribute) and t.attr == "_rejoin_failed_idx" for t in node.targets):
                writes.setdefault(method.name, set()).add(
                    "None" if isinstance(node.value, ast.Constant) and node.value.value is None else "latch")
    cleared = {name for name, kinds in writes.items() if "None" in kinds}
    assert cleared == {"__init__", "_set_waypoint"}
    assert "_do_reentry" not in writes and "_reentry_target" not in writes
    nav = MinimapCircuit(site(**RE, start_idx=10))
    nav._rejoin_failed_idx = 10
    assert nav._reentry_target((93, 71)) == (8, "floor-2-right") and nav._rejoin_failed_idx == 10


# --- 9-15: where it never fires ---------------------------------------------------------------

@pytest.mark.parametrize("start, pos", [(10, (89, 71)), (18, (89, 55))])
def test_ambiguous_columns_never_reenter(start, pos):
    traces = []
    for keys in (ON, RE):
        nav = MinimapCircuit(site(**keys, start_idx=start, return_on_block=False))
        traces.append(nav_trace(nav, [pos] * 400))
        assert nav.reentry_count == 0 and not nav.reentry_pending(pos)
    assert traces[0] == traces[1]
    assert traces[1][-1][2] == "blocked" and traces[1][-1][5] == "waypoint timeout"


@pytest.mark.parametrize("start, pos, target", [(18, (86, 59), 13), (13, (60, 71), 38)])
def test_lower_floor_fall_keeps_the_legacy_reselection(start, pos, target):
    """Walk 18 knocked onto stair-2, walk 13 knocked onto floor-2: the old lower-floor re-selection."""
    traces = []
    for keys in (ON, RE):
        nav = MinimapCircuit(site(**keys, start_idx=start, return_on_block=False))
        traces.append(nav_trace(nav, [pos] * 30))
        assert nav.reentry_count == 0 and nav.reentry_last is None
    assert traces[0] == traces[1]
    assert collapsed(row[1] for row in traces[1])[:2] == [start, target]


def _plan_variant(change):
    cfg = site(**{**RE, "start_idx": 10, "platform_rejoin": False, "return_on_block": False})
    for platforms in (cfg["platforms"], cfg["editor_plan"]["platforms"]):
        for item in platforms:
            change(item)
    return cfg


def _floor_2_variant(**values):
    def change(item):
        if item.get("id") == "floor-2":
            item.update(values)
    return change


def test_non_level_jumps_never_reenter():
    # A 2 px gap (floor-2 ends at 87): wider than the level-gap rule's 1.5 px.
    wide = _plan_variant(_floor_2_variant(right=87.0))
    nav = MinimapCircuit(wide)
    assert supported_platform(nav._cfg, (93, 71))["id"] == "floor-2-right"
    assert nav._reentry_target((93, 71)) is None
    plain = MinimapCircuit({**wide, "gap_reentry": False})
    assert nav_trace(nav, [(93, 71)] * 60) == nav_trace(plain, [(93, 71)] * 60) and nav.reentry_count == 0
    source = Platform(71, 89, 111)
    assert zones.level_gap(source, Platform(71, 42, 87.0), 91, 85) is None
    assert zones.level_gap(source, Platform(71, 42, 88.6), 91, 85) == (89.0, "left")
    # Floors one pixel apart in height.
    assert zones.level_gap(source, Platform(72, 42, 88.6), 91, 85) is None
    assert zones.level_gap(Platform(71, 42, 88.6), source, 86, 91) == (88.6, "right")
    assert zones.level_gap(Platform(71, 42, 88.6), source, 84, 91) is None  # 4.6 px from the gap
    assert zones.level_gap(source, Platform(71, 42, 88.6, 75), 91, 85) is None  # sloped
    # A sloped source, as _jump_run refuses it (review finding).
    sloped_source = Platform(55, 89, 111, 55.5)
    assert sloped_source.sloped
    assert zones._jump_run(sloped_source, Platform(55, 42, 88.475), 91, 85, 8.0, .4) is None
    assert zones.level_gap(sloped_source, Platform(55, 42, 88.475), 91, 85) is None
    assert zones.level_gap(source, Platform(71, 42, 88.6), 91, 91) is None
    assert zones.level_gap(source, Platform(71, 42, 88.6), 95, 85) is None  # 6 px from the gap
    # The stair jump 12 (floor-2 -> stair-2) is never a re-entry target.
    stair = MinimapCircuit(site(**RE, start_idx=13))
    assert stair._reentry_target((60, 71)) is None and stair._reentry_target((80, 71)) is None
    # The same predicate as the jump zone rule, for every jump connection of the example plan.
    plan = site()["editor_plan"]
    surfaces = {p["id"]: Platform(p["y"], p["left"], p["right"], p.get("right_y"), p.get("knots", ()))
                for p in plan["platforms"]}
    jumps = [c for c in plan["connections"] if c["kind"] == "jump"]
    assert len(jumps) >= 6
    for c in jumps:
        src, dst = surfaces[c["from_platform"]], surfaces[c["to_platform"]]
        gap = zones.level_gap(src, dst, c["from_x"], c["to_x"])
        assert (gap is not None) == (zones._jump_run(src, dst, c["from_x"], c["to_x"], 8.0, .4) is not None), c["id"]
    assert {c["id"] for c in jumps if zones.level_gap(surfaces[c["from_platform"]], surfaces[c["to_platform"]],
                                                       c["from_x"], c["to_x"])} == {
        "floor-2-gap", "floor-3-gap", "floor-4-gap"}  # floor-2-return takes off 4.6 px from its gap


def _knocked_run(keys, phase, pos=(93, 71)):
    run = Run(site(**keys, return_on_block=False), (93, 71), 8)
    run.until(lambda r: r.nav.state == phase, 20)
    run.world.put(pos)
    since = len(run.rows)
    for _ in range(80):
        run.step()
    return run, since


@pytest.mark.parametrize("phase", ["airborne", "jump_landing"])
def test_reentry_never_in_flight_or_landing(phase):
    plain, since = _knocked_run(ON, phase)
    enabled, _ = _knocked_run(RE, phase)
    assert enabled.trace() == plain.trace() and enabled.nav.reentry_count == 0


def test_reentry_never_during_a_rejoin_child_or_a_return():
    def rejoin(keys):
        run, since = scenario_b(**keys, return_on_block=False)
        if "gap_reentry" in keys:
            run.nav.reentry_count = run.nav.max_gap_reentries_per_lap  # budget spent: today's rejoin
        run.until(lambda r: r.nav.state == "rejoin_walking", 5)
        run.world.put((95, 71))
        if "gap_reentry" in keys:
            # Budget left for a moment: only the rejoining guard itself can refuse here.
            spent, run.nav.reentry_count = run.nav.reentry_count, 0
            assert run.nav.rejoining and run.nav._reentry_target((95, 71)) is None
            run.nav.reentry_count = spent
        assert run.nav._reentry_target((95, 71)) is None
        run.until(lambda r: not r.nav.rejoining, 30)
        return run, since
    plain, since = rejoin(ON)
    enabled, _ = rejoin(RE)
    assert enabled.trace() == plain.trace() and enabled.nav.rejoin_state == "completed"
    assert enabled.nav.reentry_count == 2 and enabled.nav.reentry_last is None

    def returning(keys):
        run, _ = scenario_b(**keys)
        assert run.nav.request_return_to_start()
        for _ in range(60):
            run.step()
            assert run.nav._reentry_target(run.world.pos) is None
        return run
    assert returning(RE).trace() == returning(ON).trace()


def test_cross_gap_detection_is_never_attacked(monkeypatch):
    """At (93,71) a monster on the landing floor at minimap x 86 is not on the character's platform."""
    seen = []
    for keys in (RE, ON):
        b = brain_at(10, **keys)
        original = b._choose_target
        monkeypatch.setattr(b, "_choose_target", lambda p, o=original: (seen.append(list(p.detections)), o(p))[1])
        for i in range(40):
            cmd = b.decide(see((93, 71), round(i * DT, 6), [mob((86 - 93) * 16)]))
            assert cmd.action == "none" and cmd.face_x == "none" and b.state != "hunting"
    assert seen and all(d == [] for d in seen)


def test_navigation_only_frames_reenter_from_the_circuit_alone():
    nav = MinimapCircuit(site(**RE, start_idx=10))
    commands, now = feed(nav, (93, 71), 0, 5)
    assert all(c == Command.NONE for c in commands) and nav.idx == 10 and nav.state == "waiting_floor"
    assert nav.step((93, 71), now) == Command.NONE  # third frame and half a second: the proof holds
    assert nav.idx == 8 and nav.state == "reentered" and nav.reentry_count == 1


def test_children_and_editor_tests_are_inert():
    cfg = site(**RE)
    for child_cfg in (plan_rejoin(cfg, (93, 71), 10, 1), plan_return_to_start(cfg, (93, 71))):
        child = MinimapCircuit(child_cfg, far_approach=False)
        assert not child._reentry_enabled and not child.gap_reentry_active
        assert child.max_gap_reentries_per_lap is None and child.gap_reentry_priority_s is None
    run, _ = scenario_b(**RE)
    run.nav.reentry_count = 2
    run.until(lambda r: r.nav.rejoining, 5)
    assert not run.nav.platform_rejoin._reentry_enabled
    baseline, _ = _legacy_b(movement_test=True)
    tested, _ = _legacy_b(movement_test=True, gap_reentry=True)
    assert tested.trace() == baseline.trace() and not tested.nav.gap_reentry_active
    for legacy in (_legacy_a, _legacy_b):
        a, _ = legacy(**ON, movement_test=True)
        b, _ = legacy(**RE, movement_test=True)
        assert a.trace() == b.trace()


# --- 20-23: default off, key on undisturbed, guard, validation --------------------------------

def wrong_floor_with_keys_on(**keys):
    """Scenarios A and B with arrive_on_destination and platform_rejoin on: arrival, then a rejoin."""
    traces = []
    for scenario in (scenario_a, scenario_b):
        run, _ = scenario(**ON, **keys)
        for _ in range(300):
            run.step()
        traces.append(run.trace())
    return traces


BOGUS = [{},{"max_gap_reentries_per_lap": 0, "gap_reentry_priority_s": math.nan},
         {"max_gap_reentries_per_lap": 9, "gap_reentry_priority_s": "x"}]


@pytest.mark.parametrize("bogus", BOGUS, ids=["false", "zero-nan", "nine-x"])
def test_default_off_timing_and_wrong_floor_traces_are_unchanged(bogus):
    from tests.test_transition_timing import PINNED as TIMING_PINNED, digest, scenario

    keys = {"gap_reentry": False, **bogus}
    for name, value in TIMING_PINNED.items():
        assert digest(scenario(name, **keys)) == value, name
    for name in ("ramp", "stair_jump", "drop", "return", "rejoin", "ramp_blocked_then_return"):
        assert digest(scenario(name, commit_budget=True, **keys)) == TIMING_PINNED[name], name
    for legacy in (_legacy_a, _legacy_b):
        assert legacy(**keys)[0].trace() == legacy()[0].trace()
    assert wrong_floor_with_keys_on(**keys) == wrong_floor_with_keys_on()


@pytest.mark.parametrize("bogus", [BOGUS[0], BOGUS[2]], ids=["false", "nine-x"])
def test_default_off_takeoff_zone_pinned_sequences_are_unchanged(bogus):
    from tests.test_takeoff_zones import PINNED as ZONES_PINNED, off_scenario
    from tests.test_transition_timing import digest

    keys = {"gap_reentry": False, **bogus}
    for name, value in ZONES_PINNED.items():
        assert digest(off_scenario(name, **keys)) == value, name


def test_key_on_changes_nothing_on_undisturbed_stretches_and_laps():
    from tests.test_takeoff_zones import PINNED as ZONES_PINNED, off_scenario
    from tests.test_transition_timing import PINNED as TIMING_PINNED, digest, scenario

    def stretch(**overrides):
        run = Run(site(**overrides), (96, 71), 7)
        return run.until(lambda r: r.nav.idx == 12, 60)
    plain, enabled = stretch(), stretch(**RE)
    assert enabled.trace() == plain.trace() and enabled.nav.reentry_count == 0
    for name in ("ramp", "gap_jump", "stair_jump", "drop", "climb", "return", "ramp_blocked_then_return"):
        assert digest(scenario(name, gap_reentry=True)) == TIMING_PINNED[name], name
    for name in ("lap", "lap + other keys"):
        assert digest(off_scenario(name, gap_reentry=True)) == ZONES_PINNED[name], name


def _raise(*args, **kwargs):
    raise AssertionError("a gap re-entry / veto helper was called with the keys absent")


@pytest.fixture
def guarded(monkeypatch):
    for name in REENTRY_NAMES:
        monkeypatch.setattr(MinimapCircuit, name, _raise)
    monkeypatch.setattr(zones, "level_gap", _raise)
    monkeypatch.setattr(Brain, "_gap_veto", _raise)


def test_with_the_keys_absent_no_new_helper_is_ever_called(guarded):
    from tests.test_takeoff_zones import PINNED as ZONES_PINNED, off_scenario
    from tests.test_transition_timing import PINNED as TIMING_PINNED, digest, scenario

    for name in ("knocked lap", "knocked lap + other keys", "rejoin + other keys", "scripted jump",
                 "return from stair-2"):
        assert digest(off_scenario(name)) == ZONES_PINNED[name], name
    for name, value in TIMING_PINNED.items():
        assert digest(scenario(name)) == value, name
    for legacy in (_legacy_a, _legacy_b):
        legacy()
    wrong_floor_with_keys_on()
    with pytest.raises(AssertionError, match="helper was called"):  # the guard itself works
        MinimapCircuit(site(**RE, start_idx=10)).step((93, 71), 0)


GUARD_PLUGIN = '''
import os
import pytest

def pytest_configure(config):
    import maplebot.nav.circuit as circuit
    import maplebot.nav.zones as zones
    from maplebot.brain.fsm import Brain

    def boom(*args, **kwargs):
        raise AssertionError("a gap re-entry / veto helper was called with the keys absent")
    for name in {names!r}:
        setattr(circuit.MinimapCircuit, name, boom)
    zones.level_gap = boom
    Brain._gap_veto = boom
    with open({marker!r}, "w", encoding="utf-8") as handle:
        handle.write("guarded " + " ".join(sorted({names!r})))
'''


def test_whole_combat_and_fsm_files_never_call_a_new_helper_with_the_keys_absent(tmp_path):
    plugin = tmp_path / "gap_guard_plugin.py"
    marker = tmp_path / "guarded.txt"
    plugin.write_text(GUARD_PLUGIN.format(names=REENTRY_NAMES, marker=str(marker)), encoding="utf-8")
    files = ["tests/test_pre_transition_combat.py", "tests/test_fsm.py", "tests/test_surface_targeting.py"]
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "--tb=short", "-p", "no:cacheprovider",
         "-p", "gap_guard_plugin", "--basetemp", str(tmp_path / "inner"), *files],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + str(ROOT)})
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert marker.read_text(encoding="utf-8").startswith("guarded _do_reentry")  # the plugin really ran
    passed = re.search(r"(\d+) passed", result.stdout)
    assert passed and int(passed.group(1)) >= 50 and "failed" not in result.stdout


@pytest.mark.parametrize("value", [1, 0, "true", None, 1.0])
def test_gap_reentry_must_be_a_boolean(value):
    with pytest.raises(ValueError, match="gap_reentry must be true or false"):
        MinimapCircuit(site(gap_reentry=value))


@pytest.mark.parametrize("value", [0, 6, 2.0, True, "2", None])
def test_budget_is_an_integer_from_1_to_5_when_on(value):
    with pytest.raises(ValueError, match="max_gap_reentries_per_lap"):
        MinimapCircuit(site(**RE, max_gap_reentries_per_lap=value))
    MinimapCircuit(site(max_gap_reentries_per_lap=value))  # not read while off


@pytest.mark.parametrize("value", [.49, 3.01, math.nan, math.inf, True, "1.5", None])
def test_priority_is_finite_between_half_and_three_seconds_when_on(value):
    with pytest.raises(ValueError, match="gap_reentry_priority_s"):
        MinimapCircuit(site(**RE, gap_reentry_priority_s=value))
    MinimapCircuit(site(gap_reentry_priority_s=value))


def test_defaults_bounds_plan_requirement_and_status_fields():
    nav = MinimapCircuit(site())
    assert not nav.gap_reentry_active and not nav.reentry_pending((93, 71))
    assert (nav.reentry_count, nav.reentry_last, nav.max_gap_reentries_per_lap, nav.gap_reentry_priority_s) == (
        0, None, None, None)
    nav = MinimapCircuit(site(**RE))
    assert nav.gap_reentry_active and (nav.max_gap_reentries_per_lap, nav.gap_reentry_priority_s) == (2, 1.5)
    for budget, priority in ((1, .5), (5, 3.0), (3, 1)):
        nav = MinimapCircuit(site(**RE, max_gap_reentries_per_lap=budget, gap_reentry_priority_s=priority))
        assert (nav.max_gap_reentries_per_lap, nav.gap_reentry_priority_s) == (budget, float(priority))
    without_plan = site()
    del without_plan["editor_plan"]
    with pytest.raises(ValueError, match="gap_reentry needs a saved editor plan"):
        MinimapCircuit({**without_plan, "gap_reentry": True})
    assert not MinimapCircuit({**without_plan, "gap_reentry": True, "movement_test": True}).gap_reentry_active
    no_platforms = site(**RE)
    no_platforms["platforms"] = []
    assert not MinimapCircuit(no_platforms).gap_reentry_active


@pytest.mark.parametrize("value", [0, -1, 20.01, math.nan, math.inf, True, "6"])
def test_veto_key_is_validated(value):
    with pytest.raises(ValueError, match="pre_transition_gap_veto_x"):
        brain_at(10, veto=value)


def test_veto_key_off_values_and_range():
    assert brain_at(10)._gap_veto_x is None
    for off in (None, False):
        cfg = deep_merge(load_config(), {"attack": {"pre_transition_gap_veto_x": off},
                                         "route": {"circuit": site(start_idx=10)}})
        assert Brain(cfg, None, 1200)._gap_veto_x is None
    for value, expected in ((20, 20.0), (.5, .5), (6, 6.0)):
        assert brain_at(10, veto=value)._gap_veto_x == expected


def test_shipped_configs_leave_every_new_key_off_and_the_example_digest_alone():
    default = load_config()
    assert "pre_transition_gap_veto_x" not in default["attack"]
    assert not {"gap_reentry", "max_gap_reentries_per_lap", "gap_reentry_priority_s"} & set(
        default.get("route", {}).get("circuit", {}))
    example = yaml.safe_load(SITE.read_text(encoding="utf-8"))
    circuit = example["route"]["circuit"]
    assert not {"gap_reentry", "max_gap_reentries_per_lap", "gap_reentry_priority_s"} & set(circuit)
    assert "pre_transition_gap_veto_x" not in example["attack"]
    assert circuit["editor_reference_sha256"] == (
        "b131badde8048e46b7f79abe17f4be1b6e966dc8295619fca0834e38aa964427")
    assert "# gap_reentry: true" in SITE.read_text(encoding="utf-8")
