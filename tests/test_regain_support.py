"""Regain-support stage before a return to point 1, in a small simulated world.

Offline only. The world is a model, not the game: walking is 9.5 px/s and only
while a command is inside its lease, Down on a rope descends 5 px/s, gravity
drops to the highest surface below, positions are observed as whole pixels and
frames arrive at 8 fps.
"""
from copy import deepcopy
import math

import pytest

from maplebot.nav import recovery as recovery_module
from maplebot.nav import regain
from maplebot.nav.circuit import MinimapCircuit
from maplebot.nav.editor import compile_plan
from maplebot.nav.regain import SupportRegain, regain_options, unsaved_rope_option
from maplebot.types import Command

FRAME = 0.125
UNSUPPORTED = "Current position is not supported by a drawn platform"


def site_config(**overrides):
    """Construction-Site-like: ground y=105, floor-1 y=88, point 1 at (49,105)."""
    plan = {"version": 1, "name": "Site", "image_size": [210, 128], "loop": True,
            "platforms": [{"id": "ground", "left": 0, "right": 101, "y": 105},
                          {"id": "floor-1", "left": 42, "right": 132, "y": 88}],
            "connections": [
                {"id": "base-ladder", "kind": "rope", "from_platform": "ground",
                 "to_platform": "floor-1", "from_x": 49, "to_x": 49},
                {"id": "base-ladder-down", "kind": "rope", "from_platform": "floor-1",
                 "to_platform": "ground", "from_x": 49, "to_x": 49}],
            "route": [{"platform_id": "ground", "x": 49, "dwell_s": 0},
                      {"platform_id": "floor-1", "x": 100, "dwell_s": 0}]}
    result = compile_plan(plan, {"editor_plan": plan, "return_on_block": False,
                                 "tolerance_x": 2, "tolerance_y": .75, **overrides})
    result["start_idx"] = 1  # so index 0 can only come from a verified return
    return result


def scripted_stage(pos=(65, 98), **options):
    """A bare stage stepped at 8 fps up to its option's entry frame; the next step may send a key."""
    stage = SupportRegain(site_config(), **{"timeout_s": 20, "max_walk_x": 16, **options})
    for i in range(6):
        assert stage.step(pos, i * FRAME, i * FRAME) == Command.NONE
    assert stage.phase in ("walking", "descending") and stage.moves == 1
    return stage, 6 * FRAME


def returning_nav(**overrides):
    nav = MinimapCircuit(site_config(**overrides))
    assert nav.idx == 1 and nav.request_return_to_start()
    return nav


class World:
    WALK, ROPE, SUBSTEPS = 9.5, 5.0, 25

    def __init__(self, cfg, pos, *, solids=(), rope=None, walls=(), fall=40.0, frame_s=FRAME):
        self.frame_s = frame_s
        self.surfaces = [(p["left"], p["right"], p["y"]) for p in cfg["editor_plan"]["platforms"]]
        self.surfaces += list(solids)      # undrawn (left, right, y)
        self.rope = rope                   # (x, bottom_y) while the character hangs on one
        self.walls = list(walls)           # (x, top_y, bottom_y): no crossing at that height
        self.fall = fall
        self.x, self.y = float(pos[0]), float(pos[1])
        self.frame_index = 0
        self.command, self.command_until = Command.NONE, 0.0
        self.log = []

    @property
    def now(self):
        return self.frame_index * self.frame_s

    def observed(self):
        return (math.floor(self.x + .5), math.floor(self.y + .5))

    def _floor(self):
        below = [y for left, right, y in self.surfaces if left <= self.x <= right and y >= self.y - 1e-9]
        return min(below) if below else None

    def _physics(self, time, dt):
        active = self.command if time < self.command_until - 1e-9 else Command.NONE
        if self.rope is not None:
            if active.move_y == "down":
                self.y += self.ROPE * dt
                if self.y >= self.rope[1]:
                    self.y, self.rope = float(self.rope[1]), None
            return  # Left/Right alone do nothing on a rope
        floor = self._floor()
        if floor is not None and self.y < floor - 1e-9:
            self.y = min(floor, self.y + self.fall * dt)
            return  # no air control
        if active.move_x in ("left", "right"):
            new_x = self.x + (self.WALK if active.move_x == "right" else -self.WALK) * dt
            for wall_x, top, bottom in self.walls:
                if top <= self.y <= bottom and min(self.x, new_x) <= wall_x <= max(self.x, new_x):
                    return
            self.x = new_x

    def frame(self, nav):
        pos, now = self.observed(), self.now
        command = nav.step(pos, now)
        self.log.append({"now": now, "pos": pos, "command": command, "lease": command.valid_for_s,
                         "recovery_state": nav.recovery_state, "state": nav.state,
                         "idx": nav.idx, "returning": nav.returning_to_start,
                         "reason": nav.recovery_reason,
                         "child": nav.recovery is not None})
        self.command = command
        self.command_until = now + (command.valid_for_s if command.valid_for_s is not None else .75)
        dt = self.frame_s / self.SUBSTEPS
        for i in range(self.SUBSTEPS):
            self._physics(now + i * dt, dt)
        self.frame_index += 1
        return command

    def run(self, nav, seconds=40):
        for _ in range(int(seconds / self.frame_s)):
            self.frame(nav)
            if nav.recovery_terminal or not nav.returning_to_start:
                break
        return self.log


def stage_rows(log):
    return [row for row in log if row["recovery_state"] == "regaining_support"]


def keys(rows):
    return [row for row in rows if row["command"] != Command.NONE]


def walk_runs(log):
    """Stage walk keys grouped into lease-chained runs: [direction, start, end, first row index].

    A later key row belongs to the run while it starts before the run's lease
    ends; it must then carry exactly the remaining time, never a new lease.
    """
    runs = []
    for index, row in enumerate(log):
        command = row["command"]
        if row["child"] or command.move_x == "none":
            continue
        end = row["now"] + row["lease"]
        if runs and runs[-1][0] == command.move_x and row["now"] < runs[-1][2] - 1e-9:
            assert end == pytest.approx(runs[-1][2]), "a fresh frame extended a running pulse"
        else:
            runs.append([command.move_x, row["now"], end, index])
    return runs


def assert_stage_keys_are_bounded(log):
    """Only Left, Right and Down, each leased, while no return route exists."""
    runs = walk_runs(log)
    for i, (_, start, end, index) in enumerate(runs):
        # The whole time a walk key can be down, across every fresh frame of the pulse.
        assert end - start <= regain.PULSE_S + 1e-9 and regain.PULSE_S == .25
        if i:
            released = runs[i - 1][2]
            assert start - released >= .3 - 1e-9
            # Two fresh frames after the release: at least one no-key frame, then this one.
            assert sum(1 for row in log[:index + 1] if row["now"] >= released - 1e-9) >= 2
    for row in log:
        command = row["command"]
        if row["child"]:
            continue
        assert command.action == "none" and command.face_x == "none"
        assert command.move_y in ("none", "down") and command.move_x in ("none", "left", "right")
        if command != Command.NONE:
            assert row["recovery_state"] == "regaining_support"
            assert row["lease"] is not None and 0 < row["lease"] <= .3 + 1e-9
            assert (command.move_x == "none") != (command.move_y == "none")
            if command.move_x != "none":
                assert row["lease"] <= .25 + 1e-9
    assert not any(row["command"].move_y == "up" or row["command"].action == "jump" for row in log)


def assert_completed_at_point_one(nav, log):
    assert not nav.recovery_terminal and not nav.returning_to_start
    assert nav.recovery_state == "completed" and nav.idx == 0 and nav.return_count == 1
    # Index 0 appears only on the last frame, after the stable point-1 proof.
    assert all(row["idx"] == 1 for row in log[:-1]) and log[-1]["idx"] == 0
    proof = [row for row in log if row["recovery_state"] == "confirming_start"]
    assert len(proof) >= 2 and proof[-1]["now"] - proof[0]["now"] >= .25
    final = log[-1]["pos"]
    assert final[1] == 105 and abs(final[0] - 49) <= 2
    assert all(row["pos"] == final for row in log[-3:])


@pytest.fixture
def plans(monkeypatch):
    calls = []
    original = recovery_module.plan_return_to_start

    def recording(cfg, pos):
        calls.append(tuple(pos))
        return original(cfg, pos)

    monkeypatch.setattr(recovery_module, "plan_return_to_start", recording)
    return calls


# -- 1. the pipes -----------------------------------------------------------

def test_pipes_walk_off_lands_on_drawn_ground_then_runs_the_normal_verified_return(plans):
    nav = returning_nav()
    world = World(nav._cfg, (65, 98), solids=[(60, 70, 98)])
    log = world.run(nav)
    # Three still frames over 0.5 s are complete at t=0.5; one more no-key frame follows.
    assert [row["command"] for row in log[:6]] == [Command.NONE] * 6
    assert log[6]["now"] == .75 and log[6]["command"] == Command("left") and log[6]["lease"] == .25
    assert all(row["recovery_state"] == "regaining_support" for row in log[:7])
    assert log[4]["state"] == "return_regain_walking" and log[0]["state"] == "return_regain_settling"
    stage = stage_rows(log)
    assert {row["command"].move_x for row in keys(stage)} == {"left"}
    assert_stage_keys_are_bounded(log)
    # The stage ends at the first strictly supported frame and never declares support itself.
    landed = next(i for i, row in enumerate(log) if row["pos"][1] == 105)
    assert log[landed]["pos"][0] <= 60 and log[landed]["recovery_state"] == "waiting_source"
    assert all(row["recovery_state"] == "regaining_support" for row in log[:landed])
    first_child = next(i for i, row in enumerate(log) if row["child"])
    assert first_child - landed >= 2  # 3 fresh frames
    assert log[first_child]["now"] - log[landed]["now"] >= .5
    assert all(row["command"] == Command.NONE for row in log[landed:first_child])
    assert plans == [log[landed]["pos"]]
    assert_completed_at_point_one(nav, log)
    assert nav._regain.moves == 1 and nav._regain.perches == 1


# -- 2. stacked pipes -------------------------------------------------------

def test_stacked_pipes_use_two_perches_and_keep_the_direction_that_produced_the_descent(plans):
    nav = returning_nav()
    world = World(nav._cfg, (46, 98), solids=[(40, 51.6, 98), (51.6, 60, 101)])
    log = world.run(nav)
    stage = stage_rows(log)
    second = [row for row in stage if row["pos"][1] == 101]
    # Point 1 (x=49) is now to the left of the character, outside the rope corridor.
    assert second and second[0]["pos"][0] > 51
    assert {row["command"].move_x for row in keys(stage)} == {"right"}
    assert keys(second) and keys(second)[0]["command"] == Command("right")
    assert nav._regain.perches == 2 and nav._regain.moves == 2
    assert_stage_keys_are_bounded(log)
    assert len(plans) == 1 and plans[0][1] == 105 and plans[0][0] >= 60
    assert_completed_at_point_one(nav, log)


def test_a_fifth_undrawn_perch_is_terminal():
    nav = returning_nav()
    steps = [(78, 84, 90), (72, 78, 93), (66, 72, 96), (60, 66, 99), (54, 60, 102)]
    world = World(nav._cfg, (80, 90), solids=steps)
    log = world.run(nav)
    assert nav.recovery_terminal and nav.recovery is None and nav.return_count == 1
    assert "Could not reach a drawn platform" in nav.recovery_reason
    assert "too many undrawn surfaces" in nav.recovery_reason
    assert nav._regain.perches == 5 and log[-1]["pos"][1] == 102
    assert {row["pos"][1] for row in log} >= {90, 93, 96, 99, 102}
    assert_stage_keys_are_bounded(log)
    assert world.frame(nav) == Command.NONE and not nav.request_return_to_start()


# -- 3. wall ----------------------------------------------------------------

def test_wall_on_the_first_side_gives_two_silent_pulses_then_the_other_direction(plans):
    nav = returning_nav()
    world = World(nav._cfg, (65, 98), solids=[(60, 70, 98)], walls=[(64.9, 95, 99)])
    log = world.run(nav)
    stage_keys = keys(stage_rows(log))
    starts = [row for i, row in enumerate(log) if row["command"] != Command.NONE
              and not row["child"] and log[i - 1]["command"] == Command.NONE]
    assert [row["command"].move_x for row in starts[:3]] == ["left", "left", "right"]
    assert all(row["pos"] == (65, 98) for row in stage_keys if row["command"].move_x == "left")
    assert [row["command"].move_x for row in starts].count("left") == 2
    assert nav._regain.moves == 2 and nav._regain.perches == 1
    assert_stage_keys_are_bounded(log)
    assert plans and plans[0][0] >= 70
    assert_completed_at_point_one(nav, log)


# -- 4./5./6. ropes ---------------------------------------------------------

def test_saved_rope_is_descended_with_leased_down_only_then_the_normal_return(plans):
    nav = returning_nav()
    world = World(nav._cfg, (49, 95), rope=(49, 105))
    log = world.run(nav)
    stage_keys = keys(stage_rows(log))
    assert len(stage_keys) >= 8
    assert all(row["command"] == Command("none", "down") and 0 < row["lease"] <= .3 for row in stage_keys)
    assert any(row["state"] == "return_regain_descending" for row in log)
    assert_stage_keys_are_bounded(log)
    assert world.rope is None and plans == [(49, 105)]
    assert nav._regain.moves == 1
    assert_completed_at_point_one(nav, log)


def test_rope_corridor_on_an_undrawn_object_stalls_down_then_walks_off(plans):
    nav = returning_nav()
    world = World(nav._cfg, (49, 98), solids=[(45, 53, 98)])
    log = world.run(nav)
    stage_keys = keys(stage_rows(log))
    downs = [row for row in stage_keys if row["command"].move_y == "down"]
    walks = [row for row in stage_keys if row["command"].move_x != "none"]
    assert downs and walks and downs[-1]["now"] < walks[0]["now"]
    assert all(row["pos"] == (49, 98) for row in downs)
    assert 1.25 <= downs[-1]["now"] - downs[0]["now"] <= 1.5
    assert {row["command"].move_x for row in walks} == {"right"}  # point 1 is at this X: floor midpoint
    assert nav._regain.moves == 2
    assert_stage_keys_are_bounded(log)
    assert len(plans) == 1 and plans[0][0] >= 53
    assert_completed_at_point_one(nav, log)


UNSAVED_DETAIL = "Climbing down an unsaved rope toward ground"


def enclosed_solid_world(nav, pos=(65, 98)):
    """An undrawn solid with a wall on both sides: neither Left, Right nor Down moves anything."""
    return World(nav._cfg, pos, solids=[(60, 70, 98)], walls=[(64.9, 95, 99), (65.1, 95, 99)])


def stage_at_unsaved_descent(pos=(96, 96), cfg=None, platform="ground", **options):
    """A bare stage stepped at one still position until its first unsaved-rope Down."""
    stage = SupportRegain(cfg or site_config(), **{"timeout_s": 20, "max_walk_x": 16, **options})
    t, sent = 0.0, []
    while True:
        command = stage.step(pos, t, t)
        if command.move_y == "down":
            break
        sent.append(command)
        t += FRAME
        assert t < 15 and not stage.failed
    assert stage._current["unsaved"] and stage.phase == "descending"
    assert stage.detail == f"Climbing down an unsaved rope toward {platform}"
    assert {c.move_x for c in sent} == {"none", "left", "right"} and stage.moves == 3
    return stage, t


# (96,96) is the spot recorded live on 2026-11-16 (log/claude-circuits-5.jsonl): a rope
# or pole that the saved map does not draw, between the ground (105) and floor-1 (88).
@pytest.mark.parametrize("pos", [(96, 96), (65, 95)])
def test_undrawn_rope_is_descended_with_down_after_both_walks_were_silent(pos, plans):
    nav = returning_nav()
    # The rope's lower end hangs 2 px above the ground: Down releases there and the fall lands.
    world = World(nav._cfg, pos, rope=(pos[0], 103))
    log = world.run(nav)
    stage_keys = keys(stage_rows(log))
    order = [row["command"].move_x if row["command"].move_x != "none" else "down" for row in stage_keys]
    first_down = order.index("down")
    # Both walk directions were pulsed without any movement before the first Down.
    assert {"left", "right"} == set(order[:first_down]) and set(order[first_down:]) == {"down"}
    assert all(row["pos"] == pos for row in stage_keys[:first_down + 1])
    downs = stage_keys[first_down:]
    assert all(row["command"] == Command("none", "down") and row["lease"] == .3 for row in downs)
    assert all(row["state"] == "return_regain_descending" and row["reason"] == UNSAVED_DETAIL
               for row in downs)
    assert all(row["pos"][0] == pos[0] for row in stage_rows(log))
    assert_stage_keys_are_bounded(log)
    # The stage never declares support: the strict check sees the ground, then the usual proof.
    landed = next(i for i, row in enumerate(log) if row["pos"][1] == 105)
    assert log[landed]["recovery_state"] == "waiting_source" and log[landed]["command"] == Command.NONE
    assert all(row["recovery_state"] == "regaining_support" for row in log[:landed])
    first_child = next(i for i, row in enumerate(log) if row["child"])
    assert first_child - landed >= 2 and log[first_child]["now"] - log[landed]["now"] >= .5
    assert world.rope is None and plans == [(pos[0], 105)]
    assert nav._regain.moves == 3 and nav._regain.perches == 1
    assert_completed_at_point_one(nav, log)


def test_down_that_does_not_descend_on_an_enclosed_undrawn_solid_is_terminal(plans):
    nav = returning_nav()
    world = enclosed_solid_world(nav)
    log = world.run(nav)
    assert nav.recovery_terminal and nav.recovery is None and not plans
    assert "Could not reach a drawn platform from (65,98): no descent with Down" in nav.recovery_reason
    assert "map editor" in nav.recovery_reason and nav.reason.startswith("Return failed: ")
    assert all(row["pos"] == (65, 98) for row in log)
    sent = keys(log)
    order = [row["command"].move_x if row["command"].move_x != "none" else "down" for row in sent]
    assert set(order) == {"left", "right", "down"}
    assert set(order[order.index("down"):]) == {"down"}  # Down only after both walks, nothing after it
    downs = [row for row in sent if row["command"].move_y == "down"]
    assert all(row["command"] == Command("none", "down") and row["lease"] == .3 for row in downs)
    # One Down leg, ended by the 1.5 s no-descent rule.
    assert 1.25 <= downs[-1]["now"] - downs[0]["now"] <= 1.5 and regain.DESCEND_STALL_S == 1.5
    assert_stage_keys_are_bounded(log)
    assert nav._regain.moves == 3 and nav._regain.perches == 1
    assert world.frame(nav) == Command.NONE
    assert not nav.request_return_to_start() and nav.return_count == 1


@pytest.mark.parametrize("wall_x, ends", [
    (64.9, ["no horizontal progress", "walk window exhausted"]),   # the silent walk first
    (65.1, ["walk window exhausted", "no horizontal progress"])])  # the silent walk last
def test_no_down_when_a_walk_ended_by_window_exhaustion(wall_x, ends, plans):
    # One side is walled (silent pulses); the other walks on a wide undrawn surface to its bound.
    nav = returning_nav(regain_max_walk_x=5)
    world = World(nav._cfg, (65, 98), solids=[(30, 100, 98)], walls=[(wall_x, 95, 99)])
    log = world.run(nav)
    assert nav.recovery_terminal and ends[-1] in nav.recovery_reason and not plans
    assert nav._regain._walk_ends == ends
    assert {row["command"].move_x for row in keys(log)} == {"left", "right"}
    assert not any(row["command"].move_y != "none" for row in log)
    assert nav._regain.moves == 2
    assert_stage_keys_are_bounded(log)


def test_no_down_when_a_walk_ended_by_leaving_the_object():
    stage, t = scripted_stage((65, 98))
    assert stage.step((65, 98), t, t) == Command("left")
    t += FRAME
    assert stage.step((65, 100), t, t) == Command.NONE  # moved down: the Left leg ends without a cause
    assert stage.phase == "settling" and stage._walk_ends == []
    commands = []
    while not stage.failed and t < 15:
        t += FRAME
        commands.append(stage.step((65, 98), t, t))  # back on the same perch; Right stays silent
    assert stage.failed and "no horizontal progress" in stage.reason
    assert {c.move_x for c in commands} == {"none", "right"}
    assert all(c.move_y == "none" for c in commands)
    assert stage._walk_ends == ["no horizontal progress"] and stage.moves == 2 and stage.perches == 1


def test_no_second_down_where_the_saved_rope_corridor_already_stalled(plans):
    # On an undrawn solid inside the base ladder's corridor, walled in on both sides.
    nav = returning_nav()
    world = World(nav._cfg, (49, 98), solids=[(45, 53, 98)], walls=[(48.9, 95, 99), (49.1, 95, 99)])
    log = world.run(nav)
    assert nav.recovery_terminal and "no horizontal progress" in nav.recovery_reason and not plans
    sent = keys(log)
    downs = [row for row in sent if row["command"].move_y == "down"]
    walks = [row for row in sent if row["command"].move_x != "none"]
    assert downs and walks and downs[-1]["now"] < walks[0]["now"]  # one Down leg, the saved one
    assert all(row["reason"] == "Climbing down the saved rope toward ground" for row in downs)
    assert nav._regain.moves == 3
    assert_stage_keys_are_bounded(log)


def test_unsaved_rope_option_uses_the_below_rule_and_does_not_mutate_the_config():
    cfg = site_config()
    before = deepcopy(cfg)
    assert unsaved_rope_option(cfg, (96, 96)) == {
        "kind": "descend", "unsaved": True, "x": 96.0, "start_y": 96.0, "end_y": 105.0,
        "platform_id": "ground"}
    assert unsaved_rope_option({"route": {"circuit": cfg}}, (60, 80))["platform_id"] == "floor-1"
    assert unsaved_rope_option(cfg, (150, 60)) is None and unsaved_rope_option({}, (60, 80)) is None
    assert unsaved_rope_option(cfg, (96, 104.5)) is None  # within .75 of the ground: nothing "below"
    assert cfg == before
    assert all(option.get("unsaved") is None for option in regain_options(cfg, (96, 96)))
    cfg["editor_plan"]["platforms"].append({"id": "twin", "left": 50, "right": 90, "y": 88.5})
    assert unsaved_rope_option(cfg, (60, 80)) is None  # two floors at nearly the same height


@pytest.mark.parametrize("pos, cause", [((99, 98), "x drift right"), ((93, 98), "x drift left"),
                                        ((96, 107), "below the floor"), ((96, 94), "moved up")])
def test_unsaved_descent_stops_keys_when_the_position_leaves_the_straight_descent(pos, cause):
    stage, t = stage_at_unsaved_descent()
    for y in (97, 98):
        t += FRAME
        down = stage.step((96, y), t, t)
        assert down == Command("none", "down") and down.valid_for_s == .3
    t += FRAME
    assert stage.step(pos, t, t) == Command.NONE, cause
    assert stage.phase == "settling" and stage._last_end == "left the unsaved rope"
    t += FRAME
    assert stage.step(pos, t, t) == Command.NONE  # settling again: no key
    assert regain.ROPE_X == 2


def test_unsaved_descent_tolerates_two_pixels_of_x_and_one_past_the_floor():
    stage, t = stage_at_unsaved_descent()
    for pos in ((98, 97), (94, 98), (96, 106)):
        t += FRAME
        assert stage.step(pos, t, t) == Command("none", "down")


def test_a_slow_unsaved_descent_ends_at_the_descent_leg_budget():
    stage, t = stage_at_unsaved_descent((96, 90), timeout_s=30)
    entered = t - FRAME
    while t < 30:
        command = stage.step((96, 90 + int(t - entered)), t, t)  # 1 px per second: never stalls
        if command == Command.NONE:
            break
        assert command == Command("none", "down") and command.valid_for_s == .3
        t += FRAME
    assert t - entered == 8.0 and stage.phase == "settling" and regain.DESCEND_LEG_TIMEOUT_S == 8
    assert stage._last_end == "descent with Down did not reach a drawn platform in time"


def test_unsaved_descent_counts_against_the_stage_clock(plans):
    class SlowRope(World):
        ROPE = 1.5  # still above the ground when the stage clock ends

    nav = returning_nav(regain_timeout_s=6)
    world = SlowRope(nav._cfg, (96, 96), rope=(96, 105))
    log = world.run(nav)
    assert nav.recovery_terminal and "timed out" in nav.recovery_reason and not plans
    assert log[-1]["now"] == pytest.approx(6.0) and world.rope is not None and world.y > 96.9
    assert any(row["command"].move_y == "down" for row in log)
    assert_stage_keys_are_bounded(log)


def test_unsaved_descent_counts_against_the_option_start_budget():
    stage = SupportRegain(site_config(), timeout_s=30, max_walk_x=16)
    t, commands, starts = 0.0, [], 0
    while not stage.failed and t < 30:
        commands.append(stage.step((96, 96), t, t))
        if stage.phase == "descending":
            starts += 1
            stage.interrupt()  # before the Down leg could send a key
        t += FRAME
    assert stage.failed and "too many attempts" in stage.reason
    # Left, Right, then six Down starts use the eight moves; the ninth start fails.
    assert starts == 6 and stage.moves == 9 and regain.MAX_MOVES == 8
    assert all(c.move_y == "none" for c in commands)


def test_unsaved_down_is_not_held_past_its_lease_when_frames_stop(plans):
    nav = returning_nav()
    world = World(nav._cfg, (96, 96), rope=(96, 105))
    while world.frame(nav).move_y != "down":
        assert world.frame_index < 200
    last = world.log[-1]
    assert last["lease"] == .3
    y_at_key = last["pos"][1]
    # No further fresh frame: nothing renews the lease, so the descent stops by itself.
    start, dt = world.now, FRAME / World.SUBSTEPS
    for i in range(8 * World.SUBSTEPS):
        world._physics(start + i * dt, dt)
    assert world.rope is not None and world.y - y_at_key <= World.ROPE * .3 + 1e-6


def test_a_duplicate_frame_cannot_extend_the_unsaved_down_lease():
    stage, t = stage_at_unsaved_descent()
    elapsed, progress = stage.elapsed, stage._descend_progress
    for _ in range(20):  # the same capture again and again
        again = stage.step((96, 96), t, t)
        assert again == Command("none", "down") and again.valid_for_s == .3  # still ends at t + 0.3
    assert stage.elapsed == elapsed and stage._descend_progress == progress and stage.moves == 3
    # Fresh timestamps with a stale position do not keep it alive either: 1.5 s without descent.
    began, last_down = stage._descend_progress, t
    while stage.phase == "descending":
        t += FRAME
        if stage.step((96, 96), t, t) != Command.NONE:
            last_down = t
    assert stage._last_end == "no descent with Down"
    assert t - began == 1.5 and last_down == t - FRAME


# -- review regressions: the unsaved rope descent ----------------------------

# One pixel beside the recorded spot, a fractional alignment offset, and the ground's left
# end: one walk option has under 2 px of room, so its pulses are too short to be judged
# one by one and the leg runs to its 6 s budget without moving anything.
@pytest.mark.parametrize("pos", [(97, 96), (96.1, 96), (96.5, 96), (4, 96)])
def test_a_walk_with_under_two_pixels_of_room_that_never_moves_counts_as_silent(pos):
    stage, t = stage_at_unsaved_descent(pos)
    assert stage._walk_ends == ["no horizontal progress"] * 2
    assert 6 < t < 12 and regain.WALK_LEG_TIMEOUT_S == 6


@pytest.mark.parametrize("pos", [(97, 96), (4, 96)])
def test_undrawn_rope_beside_a_walk_bound_is_still_descended(pos, plans):
    nav = returning_nav()
    world = World(nav._cfg, pos, rope=(pos[0], 103))
    log = world.run(nav, seconds=60)
    stage_keys = keys(stage_rows(log))
    order = [row["command"].move_x if row["command"].move_x != "none" else "down" for row in stage_keys]
    first_down = order.index("down")
    assert {"left", "right"} == set(order[:first_down]) and set(order[first_down:]) == {"down"}
    assert all(row["pos"] == pos for row in stage_keys[:first_down + 1])
    assert min(row["lease"] for row in stage_keys[:first_down]) == pytest.approx(.1)
    assert all(row["reason"] == UNSAVED_DETAIL and row["lease"] == .3 for row in stage_keys[first_down:])
    assert nav._regain._walk_ends == ["no horizontal progress"] * 2
    assert nav._regain.moves == 3 and nav._regain.perches == 1
    assert_stage_keys_are_bounded(log)
    assert world.rope is None and plans == [(pos[0], 105)]
    assert_completed_at_point_one(nav, log)


@pytest.mark.parametrize("y_after", [
    lambda pulses: 96.6 if pulses == 1 else 96,   # 0.6 px during the first pulse, then back
    lambda pulses: 96 + .3 * min(pulses, 3)])     # 0.3 px per pulse three times: 0.9 px in all
def test_a_short_pulse_walk_that_moved_at_all_is_a_leg_timeout_and_gets_no_down(y_after):
    stage = SupportRegain(site_config(), timeout_s=30, max_walk_x=16)
    t, rights, previous, commands = 0.0, 0, Command.NONE, []
    while not stage.failed and t < 30:
        # Right has 1.5 px of room (0.15 s pulses); the dot shifts during that leg.
        command = stage.step((96.5, y_after(rights)), t, t)
        if command.move_x == "right" and previous == Command.NONE:
            rights += 1
        previous = command
        commands.append(command)
        t += FRAME
    assert rights > 5
    assert stage._walk_ends == ["no horizontal progress", "walk did not reach a drawn platform in time"]
    assert stage.failed and "walk did not reach a drawn platform in time" in stage.reason
    assert all(c.move_y == "none" for c in commands) and stage.moves == 2


def test_no_down_when_a_walk_ended_by_leg_timeout_before_a_silent_walk():
    stage, t = scripted_stage((96, 96), timeout_s=30)
    x, previous, commands = 96, Command.NONE, []
    while not stage.failed and t < 30:
        command = stage.step((x, 96), t, t)
        if command == Command("left") and previous == Command.NONE:
            x -= 1  # every Left pulse moves one pixel; Right never moves anything
        previous = command
        commands.append(command)
        t += FRAME
    assert stage._walk_ends == ["walk did not reach a drawn platform in time", "no horizontal progress"]
    assert stage.failed and "no horizontal progress" in stage.reason
    assert {c.move_x for c in commands} == {"none", "left", "right"}
    assert all(c.move_y == "none" for c in commands)
    assert stage.moves == 2 and stage.perches == 1 and 80 < x < 96


def test_down_is_not_sent_where_no_walk_was_tried():
    stage = SupportRegain(site_config(), timeout_s=30, max_walk_x=16)
    t = 0.0
    while len(stage._walk_ends) < 2:
        assert stage.step((96, 96), t, t).move_y == "none" and t < 10
        t += FRAME
    assert stage.phase == "settling" and stage._walk_ends == ["no horizontal progress"] * 2
    # Same height, inside the old walk window (77..101), but another place: moved during the settle.
    sent = []
    while not stage.failed and t < 25:
        command = stage.step((85, 96), t, t)
        if command.move_y == "down":
            break
        sent.append(command)
        t += FRAME
    # Left and Right were both pulsed at (85,96) before Down was held there.
    assert command == Command("none", "down") and stage._current["x"] == 85
    assert {c.move_x for c in sent} == {"none", "left", "right"}
    assert stage.perches == 2 and stage.moves == 5
    assert stage._walk_ends == ["no horizontal progress"] * 2


def test_two_pixels_of_height_from_the_silent_walks_is_another_place_too():
    stage, t = scripted_stage((96, 96), timeout_s=30)  # the perch is settled at (96,96)
    while len(stage._walk_ends) < 2:
        assert stage.step((96, 97), t, t).move_y == "none" and t < 10  # both walks silent at 97
        t += FRAME
    assert stage._silent_at == [(96, 97)] * 2 and stage.perches == 1
    sent = []
    while not stage.failed and t < 25:
        command = stage.step((96, 95), t, t)  # one pixel from the perch, two from the silence
        if command.move_y == "down":
            break
        sent.append(command)
        t += FRAME
    assert command == Command("none", "down") and stage._current["start_y"] == 95
    assert {c.move_x for c in sent} == {"none", "left", "right"}
    assert stage.perches == 2 and stage.moves == 5


def test_a_two_pixel_shift_after_the_silent_walks_still_gets_the_down_leg():
    stage = SupportRegain(site_config(), timeout_s=30, max_walk_x=16)
    t = 0.0
    while len(stage._walk_ends) < 2:
        stage.step((96, 96), t, t)
        t += FRAME
    commands, t = step_until_key(stage, (94, 97), t)
    assert commands[-1] == Command("none", "down") and stage.perches == 1 and stage.moves == 3
    assert regain.ROPE_X == 2


def test_unsaved_descent_that_stops_after_one_pixel_ends_at_the_stall_time():
    stage, t = stage_at_unsaved_descent()
    t += FRAME
    first_seen = t
    assert stage.step((96, 97), t, t) == Command("none", "down")
    last_down = t
    while stage.phase == "descending":
        t += FRAME
        if stage.step((96, 97), t, t) != Command.NONE:
            last_down = t
    assert stage._last_end == "no descent with Down"
    assert t - first_seen == 1.5 and last_down == t - FRAME


@pytest.mark.parametrize("jitter", [(96, 95), (97, 96), (94, 96)])
def test_position_jitter_is_not_descent_progress(jitter):
    stage, t = stage_at_unsaved_descent()
    began, last_down, i = stage._descend_progress, t, 0
    while stage.phase == "descending":
        t += FRAME
        i += 1
        if stage.step(jitter if i % 2 else (96, 96), t, t) != Command.NONE:
            last_down = t
            assert stage._descend_progress == began
    assert stage._last_end == "no descent with Down"
    assert t - began == 1.5 and last_down == t - FRAME


def test_an_interrupted_unsaved_descent_gets_new_walks_and_a_new_down_leg_lower_down():
    stage, t = stage_at_unsaved_descent(timeout_s=30)
    for y in (97, 98, 99):
        t += FRAME
        assert stage.step((96, y), t, t) == Command("none", "down")
    stage.interrupt()  # a lost dot or a pause, as wired from the circuit
    sent = []
    while not stage.failed and t < 25:
        t += FRAME
        command = stage.step((96, 99), t, t)
        if command.move_y == "down":
            break
        sent.append(command)
    assert not stage.failed and command == Command("none", "down") and command.valid_for_s == .3
    assert stage.detail == UNSAVED_DETAIL and stage._current["start_y"] == 99
    assert {c.move_x for c in sent} == {"none", "left", "right"}  # silence is proven again here
    assert stage.perches == 2 and stage.moves == 6


def test_unsaved_descent_over_a_slope_stops_one_pixel_past_the_lower_edge_of_its_surface():
    cfg = site_config()
    cfg["editor_plan"]["platforms"].append({"id": "stair", "left": 110, "right": 130, "y": 80, "right_y": 60})
    # The stair's quantized surface at x=120 is 69.5-70.5 (see the slope options test).
    assert unsaved_rope_option(cfg, (120, 60)) == {
        "kind": "descend", "unsaved": True, "x": 120.0, "start_y": 60.0, "end_y": 70.5,
        "platform_id": "stair"}
    stage, t = stage_at_unsaved_descent((120, 60), cfg=cfg, platform="stair")
    t += FRAME
    assert stage.step((120, 71), t, t) == Command("none", "down")  # 71 <= 70.5 + 1
    t += FRAME
    assert stage.step((120, 72), t, t) == Command.NONE
    assert stage.phase == "settling" and stage._last_end == "left the unsaved rope"


# -- 7. level class ---------------------------------------------------------

def test_just_past_a_drawn_end_walks_inward_and_becomes_supported_without_a_fall(plans):
    nav = returning_nav()
    world = World(nav._cfg, (136, 88), solids=[(132, 145, 88)])
    for _ in range(80):
        world.frame(nav)
        if nav.recovery is not None:
            break
    log = world.log
    assert nav.recovery is not None and nav.recovery_state == "running"
    assert all(row["pos"][1] == 88 for row in log)
    assert {row["command"].move_x for row in keys(stage_rows(log))} == {"left"}
    assert_stage_keys_are_bounded(log)
    assert len(plans) == 1 and 130 <= plans[0][0] <= 132 and plans[0][1] == 88
    assert min(row["pos"][0] for row in log) >= 130  # stays near the bound one pixel inside the end


def test_nine_pixels_past_the_end_with_nothing_below_is_terminal_without_any_key(plans):
    nav = returning_nav()
    world = World(nav._cfg, (141, 88), solids=[(132, 145, 88)])
    log = world.run(nav)
    assert nav.recovery_terminal and nav.recovery is None and not plans
    assert all(row["command"] == Command.NONE for row in log)
    assert "no drawn platform below or beside" in nav.recovery_reason
    assert "Could not reach a drawn platform from (141,88)" in nav.recovery_reason
    assert log[-1]["now"] == .5  # as soon as the position is still


# -- 8. clamp ---------------------------------------------------------------

def test_walk_window_is_clamped_and_exhaustion_is_terminal(plans):
    nav = returning_nav(regain_max_walk_x=5)
    world = World(nav._cfg, (65, 98), solids=[(30, 100, 98)])
    log = world.run(nav)
    assert nav.recovery_terminal and nav.recovery is None and not plans
    assert "walk window exhausted" in nav.recovery_reason
    assert all(60 - 1 <= row["pos"][0] <= 70 + 1 and row["pos"][1] == 98 for row in log)
    starts = [row for i, row in enumerate(log)
              if row["command"] != Command.NONE and log[i - 1]["command"] == Command.NONE]
    assert {row["command"].move_x for row in starts} == {"left", "right"}
    for row in starts:
        room = row["pos"][0] - 60 if row["command"].move_x == "left" else 70 - row["pos"][0]
        assert room >= 1
        assert row["lease"] == pytest.approx(min(.25, max(.1, room / 10)))
    assert nav._regain.moves == 2
    assert not any(row["command"].move_y != "none" for row in log)  # exhausted walks: no Down leg
    assert_stage_keys_are_bounded(log)


# -- 9. mid-air and knock-back ---------------------------------------------

def test_falling_onto_a_drawn_platform_runs_the_normal_path_with_zero_stage_keys(plans):
    nav = returning_nav()
    world = World(nav._cfg, (20, 92), fall=10.0)
    log = world.run(nav)
    air = [row for row in log if row["pos"][1] < 105]
    assert len(air) >= 8 and all(row["command"] == Command.NONE for row in air)
    assert all(row["recovery_state"] == "regaining_support" for row in air)
    assert not keys(stage_rows(log)) and nav._regain.moves == 0
    assert plans == [(20, 105)]
    assert_completed_at_point_one(nav, log)


def test_a_position_that_never_settles_is_terminal_after_four_seconds_without_a_key():
    nav = returning_nav()
    commands = []
    for i in range(40):
        commands.append(nav.step((60 + i % 2 * 2, 95), i * FRAME))
        if nav.recovery_terminal:
            break
    assert nav.recovery_terminal and "position kept moving" in nav.recovery_reason
    assert i * FRAME == 4.0 and all(command == Command.NONE for command in commands)
    assert nav._regain.moves == 0 and nav._regain.perches == 0


# -- 10. interruptions ------------------------------------------------------

def to_first_pulse(nav, pos=(65, 98)):
    for i in range(6):
        assert nav.step(pos, i * FRAME) == Command.NONE
    command = nav.step(pos, 6 * FRAME)
    assert command == Command("left") and command.valid_for_s == .25
    assert nav.state == "return_regain_walking" and nav._regain.moves == 1
    return 6 * FRAME


@pytest.mark.parametrize("kind", ["suspend", "lost", "gap"])
def test_interruption_mid_pulse_needs_a_new_settle_and_keeps_the_budgets(kind):
    nav = returning_nav()
    now = to_first_pulse(nav)
    elapsed = nav._regain.elapsed
    assert elapsed == pytest.approx(.75)
    if kind == "suspend":
        nav.suspend(now + .05)
        now += FRAME
    elif kind == "lost":
        assert nav.step(None, now + FRAME) == Command.NONE
        now += 2 * FRAME
    else:
        now += 100
    assert nav._regain.phase == "settling" or kind == "gap"
    commands = [nav.step((65, 98), now + i * FRAME) for i in range(7)]
    assert commands[:6] == [Command.NONE] * 6
    assert commands[6] == Command("left") and commands[6].valid_for_s == .25
    assert nav._regain.moves == 2 and nav._regain.perches == 1  # restarted, never reset
    assert elapsed <= nav._regain.elapsed <= elapsed + .8  # the gap itself is not stage time
    assert nav.returning_to_start and nav.return_count == 1 and nav.recovery is None


# -- 11. flicker ------------------------------------------------------------

def test_one_supported_frame_builds_no_plan_and_budgets_continue(plans):
    nav = returning_nav()
    now = to_first_pulse(nav)
    elapsed = nav._regain.elapsed
    assert nav.step((65, 105), now + FRAME) == Command.NONE  # one strictly supported frame
    assert nav.recovery is None and not plans
    assert nav.recovery_state == "waiting_source" and nav.state == "return_waiting_source"
    assert nav.recovery_reason is None
    commands = [nav.step((65, 98), now + (2 + i) * FRAME) for i in range(7)]
    assert commands[:6] == [Command.NONE] * 6 and commands[6] == Command("left")
    assert nav.recovery_state == "regaining_support"
    assert nav.recovery is None and not plans
    assert nav._regain.moves == 2 and nav._regain.perches == 1 and nav._regain.elapsed > elapsed


# -- 12. budgets ------------------------------------------------------------

def test_stage_clock_is_terminal(plans):
    nav = returning_nav(regain_timeout_s=5)
    world = World(nav._cfg, (65, 98), solids=[(30, 100, 98)])
    log = world.run(nav)
    assert nav.recovery_terminal and "timed out" in nav.recovery_reason
    assert "Could not reach a drawn platform" in nav.recovery_reason
    assert log[-1]["now"] == pytest.approx(5.0) and not plans
    assert_stage_keys_are_bounded(log)


def test_option_starts_are_bounded_even_when_every_start_is_interrupted():
    nav = returning_nav()
    now, pops, commands = 0.0, 0, []
    while not nav.recovery_terminal and now < 30:
        commands.append(nav.step((65, 98), now))
        if nav.state == "return_regain_walking":
            pops += 1
            nav.suspend(now + .01)  # before the option could send a key
        now += FRAME
    assert nav.recovery_terminal and "too many attempts" in nav.recovery_reason
    assert pops == 8 and nav._regain.moves == 9 and regain.MAX_MOVES == 8
    assert all(command == Command.NONE for command in commands)
    assert nav._regain.elapsed < 20


def test_whole_return_timeout_still_applies_and_no_new_attempt_is_granted():
    nav = returning_nav(return_timeout_s=2)
    commands = []
    for i in range(20):
        commands.append(nav.step((65, 98), i * FRAME))
        if nav.recovery_terminal:
            break
    assert nav.recovery_terminal and nav.recovery_reason == "Return route timed out"
    assert i * FRAME == 2.0 and any(command == Command("left") for command in commands)
    assert nav.step((65, 98), 2.2) == Command.NONE
    assert not nav.request_return_to_start() and nav.return_count == 1


# -- 13. flag off -----------------------------------------------------------

def test_flag_off_is_the_legacy_wait_step_for_step():
    nav = returning_nav(return_regain_support=False)
    for i in range(24):
        assert nav.step((65, 98), i * FRAME) == Command.NONE
        assert nav.recovery_state == "waiting_source" and nav.state == "return_waiting_source"
        assert nav.recovery_reason == UNSUPPORTED and nav._regain is None
    assert nav.step((65, 98), 3.0) == Command.NONE
    assert nav.recovery_terminal and nav._regain is None
    assert nav.recovery_reason == "Stable source unavailable: " + UNSUPPORTED


def test_flag_does_not_change_a_return_from_a_supported_position():
    logs = []
    for flag in (True, False):
        nav = returning_nav(return_regain_support=flag)
        world = World(nav._cfg, (80, 105))
        logs.append(world.run(nav))
        assert nav._regain is None
        assert_completed_at_point_one(nav, world.log)
    assert logs[0] == logs[1]
    assert any(row["command"] == Command("left") for row in logs[0])


# -- 14. regain_options -----------------------------------------------------

def walk(direction, bound, platform_id):
    return {"kind": "walk", "direction": direction, "bound": bound, "platform_id": platform_id}


def test_options_pick_the_first_drawn_floor_underneath_and_do_not_mutate_the_config():
    cfg = site_config()
    before = deepcopy(cfg)
    assert regain_options(cfg, (60, 80)) == [walk("left", 45, "floor-1"), walk("right", 76, "floor-1")]
    assert regain_options({"route": {"circuit": cfg}}, (60, 98)) == [
        walk("left", 44, "ground"), walk("right", 76, "ground")]
    assert cfg == before
    assert regain_options(cfg, (150, 60)) == [] and regain_options({}, (60, 80)) == []


def test_options_use_the_quantized_slope_surface():
    cfg = site_config()
    cfg["editor_plan"]["platforms"].append({"id": "stair", "left": 110, "right": 130, "y": 80, "right_y": 60})
    assert [o["platform_id"] for o in regain_options(cfg, (120, 60))] == ["stair", "stair"]
    # 69 is within .75 of the stair's local range (69.5-70.5): the stair is not "below".
    assert {o["platform_id"] for o in regain_options(cfg, (120, 69))} == {"floor-1"}


def test_options_refuse_two_floors_at_nearly_the_same_height():
    cfg = site_config()
    cfg["editor_plan"]["platforms"].append({"id": "twin", "left": 50, "right": 90, "y": 88.5})
    assert regain_options(cfg, (60, 80)) == []
    assert regain_options(cfg, (45, 80))[0]["platform_id"] == "floor-1"  # outside the twin's span


def test_rope_option_is_deduplicated_and_limited_to_its_corridor():
    cfg = site_config()
    descend = {"kind": "descend", "x": 49.0, "end_y": 105.0, "platform_id": "ground"}
    options = regain_options(cfg, (49, 95))
    assert options[0] == descend and [o["kind"] for o in options].count("descend") == 1
    assert regain_options(cfg, (51, 95))[0] == descend
    for pos in ((52, 95), (49, 88.5), (49, 104.5), (49, 80)):
        assert all(o["kind"] != "descend" for o in regain_options(cfg, pos))
    cfg["editor_plan"]["connections"].append({"id": "second", "kind": "rope", "from_platform": "ground",
                                              "to_platform": "floor-1", "from_x": 52, "to_x": 52})
    assert [o["x"] for o in regain_options(cfg, (51, 95)) if o["kind"] == "descend"] == [52.0]
    assert [o["x"] for o in regain_options(cfg, (50, 95)) if o["kind"] == "descend"] == [49.0]


def test_direction_rule_prefer_and_window_maths():
    cfg = site_config()
    assert [o["direction"] for o in regain_options(cfg, (30, 98))] == ["right", "left"]
    assert [o["direction"] for o in regain_options(cfg, (65, 98))] == ["left", "right"]
    assert [o["direction"] for o in regain_options(cfg, (30, 98), prefer="left")] == ["left", "right"]
    # At point 1's own X the floor midpoint (50.5) decides; the rope corridor comes first.
    assert [o.get("direction") for o in regain_options(cfg, (49, 98))] == [None, "right", "left"]
    assert regain_options(cfg, (65, 98), max_walk_x=5) == [walk("left", 60, "ground"), walk("right", 70, "ground")]
    # Within 3 px of the floor's end that direction is skipped.
    assert regain_options(cfg, (99, 98)) == [walk("left", 83, "ground")]
    assert regain_options(cfg, (3, 98)) == [walk("right", 19, "ground")]
    assert regain_options(cfg, (97.5, 98)) == [walk("left", 81.5, "ground")]  # under 1 px of room
    cfg["editor_plan"]["platforms"][0]["right"] = 209
    assert regain_options(cfg, (207, 98)) == [walk("left", 191, "ground")]
    # The image caps are the binding term only when the floor reaches the image edge.
    cfg["editor_plan"]["platforms"][0]["right"] = 101
    cfg["editor_plan"]["image_size"] = [99, 128]
    assert regain_options(cfg, (90, 98)) == [walk("left", 74, "ground"), walk("right", 97, "ground")]
    cfg["editor_plan"]["platforms"][0]["left"] = -5
    assert regain_options(cfg, (3, 98)) == [walk("right", 19, "ground"), walk("left", 1, "ground")]


def test_level_option_needs_a_small_gap_and_the_same_height():
    cfg = site_config()
    assert regain_options(cfg, (136, 88)) == [walk("left", 131, "floor-1")]
    assert regain_options(cfg, (140, 88)) == [walk("left", 131, "floor-1")]
    assert regain_options(cfg, (141, 88)) == [] and regain_options(cfg, (136, 90)) == []
    assert regain_options(cfg, (38, 88))[0] == walk("right", 43, "floor-1")  # ground below comes after


def test_level_option_takes_the_nearest_end_and_on_a_tie_the_one_toward_point_one():
    cfg = site_config()
    cfg["editor_plan"]["platforms"].append({"id": "east", "left": 140, "right": 160, "y": 88})
    # 4 px from both ends: point 1 (x=49) is to the left.
    assert regain_options(cfg, (136, 88)) == [walk("left", 131, "floor-1")]
    assert regain_options(cfg, (135, 88)) == [walk("left", 131, "floor-1")]
    assert regain_options(cfg, (137, 88)) == [walk("right", 141, "east")]
    cfg["waypoints"][0]["x"] = 200  # point 1 to the right: the tie goes the other way
    assert regain_options(cfg, (136, 88)) == [walk("right", 141, "east")]
    assert regain_options(cfg, (135, 88)) == [walk("left", 131, "floor-1")]  # nearer still wins


def test_support_regain_reports_its_reason_text_and_stays_failed():
    stage = SupportRegain(site_config(), timeout_s=20, max_walk_x=16)
    for i in range(5):
        assert stage.step((150, 60), i * FRAME, i * FRAME) == Command.NONE
    assert stage.failed and stage.phase == "failed"
    assert stage.reason == ("Could not reach a drawn platform from (150,60): no drawn platform below or "
                            "beside. Move the character onto a drawn platform or draw this surface in "
                            "the map editor.")
    for i in range(10):  # a failed stage stays silent however long it is stepped
        assert stage.step((65, 98), 1 + i * FRAME, 1 + i * FRAME) == Command.NONE
    assert stage.moves == 0 and stage.phase == "failed"
    stage.interrupt()
    assert stage.failed


# -- 15. validation ---------------------------------------------------------

@pytest.mark.parametrize("key, value", [
    ("return_regain_support", 1), ("return_regain_support", "true"), ("return_regain_support", None),
    ("regain_timeout_s", 4.9), ("regain_timeout_s", 30.1), ("regain_timeout_s", float("nan")),
    ("regain_timeout_s", True), ("regain_timeout_s", "20"),
    ("regain_max_walk_x", 3.9), ("regain_max_walk_x", 40.5), ("regain_max_walk_x", float("inf")),
    ("regain_max_walk_x", None)])
def test_regain_config_validation(key, value):
    with pytest.raises(ValueError, match=key):
        MinimapCircuit(site_config(**{key: value}))


def test_regain_config_defaults_and_bounds_are_accepted():
    nav = MinimapCircuit(site_config())
    assert nav.return_regain_support is True and nav.regain_timeout_s == 20 and nav.regain_max_walk_x == 16
    nav = MinimapCircuit(site_config(return_regain_support=False, regain_timeout_s=5, regain_max_walk_x=40))
    assert nav.return_regain_support is False and nav.regain_timeout_s == 5 and nav.regain_max_walk_x == 40


# -- review regressions: a perch is a place, not only a height ---------------

def high_perch_stage():
    """Above floor-1 at (60,70): left bound 45, right bound 76; the first Left pulse is out."""
    stage, t = scripted_stage((60, 70))
    assert stage.step((60, 70), t, t) == Command("left") and stage._current["bound"] == 45
    return stage, t


def step_until_key(stage, pos, t, frames=40):
    commands = []
    for _ in range(frames):
        t += FRAME
        commands.append(stage.step(pos, t, t))
        if commands[-1] != Command.NONE or stage.failed:
            break
    return commands, t


@pytest.mark.parametrize("interrupted", [True, False])
def test_same_height_at_another_x_with_no_floor_below_sends_no_key(interrupted):
    stage, t = high_perch_stage()
    assert regain_options(site_config(), (140, 70)) == []
    if interrupted:
        stage.interrupt()  # a pause; the character was moved meanwhile
    commands, t = step_until_key(stage, (140, 70), t)
    assert all(command == Command.NONE for command in commands)
    assert stage.failed and "from (140,70): no drawn platform below or beside" in stage.reason


@pytest.mark.parametrize("interrupted", [True, False])
def test_same_height_at_another_x_gets_a_window_around_where_it_stands(interrupted):
    stage, t = high_perch_stage()
    if interrupted:
        stage.interrupt()
    commands, t = step_until_key(stage, (100, 70), t)
    assert commands[-1] == Command("left") and all(c == Command.NONE for c in commands[:-1])
    assert len(commands) >= 6  # a full new settle and entry frame first
    assert stage._current == regain_options(site_config(), (100, 70))[0]
    assert stage._current["bound"] == 84 and 100 - stage._current["bound"] <= 16
    assert stage.perches == 2 and stage.moves == 2  # counted, so displacement stays bounded


def test_pushed_past_the_bound_at_the_same_height_recomputes_the_other_direction():
    stage, t = scripted_stage((65, 98))
    assert stage.step((65, 98), t, t) == Command("left")
    assert [o["bound"] for o in stage._options] == [81]  # the right option, computed for x=65
    t += FRAME
    assert stage.step((20, 98), t, t) == Command.NONE
    assert stage.phase == "settling" and stage._last_end == "walk window exhausted"
    commands, t = step_until_key(stage, (20, 98), t)
    assert commands[-1] == Command("right") and stage._current["bound"] == 36
    assert stage._current["bound"] - 20 <= 16 and stage.perches == 2


def test_two_pixels_past_the_bound_ends_that_direction_and_stays_the_same_perch():
    stage, t = scripted_stage((65, 98))
    assert stage.step((65, 98), t, t) == Command("left") and stage._current["bound"] == 49
    t += FRAME
    assert stage.step((47, 98), t, t) == Command.NONE  # mid-pulse: no further Left
    assert stage.phase == "settling" and stage._last_end == "walk window exhausted"
    commands, t = step_until_key(stage, (47, 98), t)
    assert commands[-1] == Command("right") and all(c == Command.NONE for c in commands[:-1])
    assert stage._current["bound"] == 81 and stage.perches == 1 and stage.moves == 2


def test_moved_up_during_a_walk_stops_keys_and_prefers_the_other_direction():
    stage, t = scripted_stage((65, 98))
    assert stage.step((65, 98), t, t) == Command("left")
    t += FRAME
    assert stage.step((65, 96), t, t) == Command.NONE
    assert stage.phase == "settling" and stage._prefer == "right"
    commands, t = step_until_key(stage, (65, 96), t)
    assert commands[-1] == Command("right") and all(c == Command.NONE for c in commands[:-1])
    assert stage.perches == 2


# -- review regressions: budgets that interruptions cannot reset -------------

@pytest.mark.parametrize("positions", [
    [(65, 104), (65, 105)],             # 1 px dot jitter at a surface
    [(60, 87), (60, 88), (60, 88)],     # unsupported on every third frame
])
def test_flickering_support_is_terminal_at_the_settle_timeout_not_the_return_timeout(positions, plans):
    nav = returning_nav()
    commands = []
    for i in range(600):
        commands.append(nav.step(positions[i % len(positions)], i * FRAME))
        if nav.recovery_terminal:
            break
    assert nav.recovery_terminal and i * FRAME <= 4.0 + len(positions) * FRAME
    assert "position kept moving" in nav.recovery_reason and "Could not reach" in nav.recovery_reason
    assert all(command == Command.NONE for command in commands) and not plans
    assert nav._regain.moves == 0 and nav._regain.elapsed == pytest.approx(i * FRAME, abs=2 * FRAME)


def test_a_dot_lost_every_third_frame_cannot_keep_the_stage_waiting_for_the_return_timeout():
    nav = returning_nav()
    commands = []
    for i in range(600):
        commands.append(nav.step(None if i % 3 == 2 else (65, 98), i * FRAME))
        if nav.recovery_terminal:
            break
    assert nav.recovery_terminal and "position kept moving" in nav.recovery_reason
    assert i * FRAME < 15 and all(command == Command.NONE for command in commands)


def test_stage_clock_counts_active_time_across_interruptions():
    stage = SupportRegain(site_config(), timeout_s=5, max_walk_x=16)
    for i in range(41):
        assert stage.step((65, 98), i * FRAME, i * FRAME) == Command.NONE
        if stage.failed:
            break
        stage.interrupt()  # e.g. a pause before every frame: a fresh settle each time
    assert stage.failed and i * FRAME == 5.0 and "timed out" in stage.reason
    assert stage.moves == 0


def test_a_pause_gives_a_new_settle_deadline_but_a_flicker_does_not():
    for fresh, survives in ((True, True), (False, False)):
        stage = SupportRegain(site_config(), timeout_s=20, max_walk_x=16)
        for i in range(31):  # 3.75 s without ever standing still
            assert stage.step((60 + i % 2 * 2, 98), i * FRAME, i * FRAME) == Command.NONE
        stage.interrupt(fresh_deadline=fresh)
        for i in range(31, 38):
            stage.step((65, 98), i * FRAME, i * FRAME)
        assert stage.failed is not survives
        assert (stage.phase == "walking") is survives


# -- review regressions: source proof must be continuous ---------------------

def test_an_unsupported_frame_restarts_the_source_proof(plans):
    nav = returning_nav()
    frames = [(65, 105)] * 4 + [(65, 98)] + [(65, 105)] * 5
    states = []
    for i, pos in enumerate(frames):
        assert nav.step(pos, i * FRAME) == Command.NONE
        states.append((nav.recovery is not None, len(plans)))
    # Supported again from frame 5 (t=.625): three frames and 0.5 s end at frame 9.
    assert states[:9] == [(False, 0)] * 9 and states[9] == (True, 1)
    assert plans == [(65, 105)] and nav._regain.moves == 0


# -- review regressions: pulse length and leg budgets ------------------------

def test_a_pulse_near_the_walk_bound_is_shortened_to_the_room_left():
    stage, t = scripted_stage((65, 98), max_walk_x=4)
    first = stage.step((65, 98), t, t)
    assert first == Command("left") and first.valid_for_s == .25 and stage._current["bound"] == 61
    leases = []
    for _ in range(12):
        t += FRAME
        command = stage.step((62.6, 98), t, t)
        if command != Command.NONE:
            leases.append((t, command.valid_for_s))
    # One continuation frame of the first pulse, then a new pulse sized for 1.6 px of room.
    assert leases[0] == (.875, pytest.approx(.125))
    assert leases[1][0] == pytest.approx(1.375) and leases[1][1] == pytest.approx(.16)
    assert regain.PULSE_MIN_S <= leases[1][1] < regain.PULSE_S


@pytest.mark.parametrize("pos, cause", [((52, 97), "sideways"), ((49, 107), "below the lower end")])
def test_descent_stops_when_the_character_leaves_the_saved_rope(pos, cause):
    stage, t = scripted_stage((49, 95))
    assert stage.phase == "descending"
    down = stage.step((49, 96), t, t)
    assert down == Command("none", "down") and down.valid_for_s == .3
    t += FRAME
    assert stage.step(pos, t, t) == Command.NONE, cause
    assert stage.phase == "settling" and stage._last_end == "left the saved rope"


def test_a_slow_descent_ends_at_the_descent_leg_budget():
    stage, t = scripted_stage((49, 95))
    entered = t - FRAME
    while t < 12:
        command = stage.step((49, 95 + int(t - entered)), t, t)  # 1 px per second: never stalls
        if command == Command.NONE:
            break
        assert command == Command("none", "down") and command.valid_for_s == .3
        t += FRAME
    assert t - entered == 8.0 and stage.phase == "settling"
    assert stage._last_end == "no descent on the saved rope"


def test_a_walk_that_keeps_moving_without_landing_ends_at_the_walk_leg_budget():
    stage, t = scripted_stage((65, 98))
    entered, x, previous, last_key = t - FRAME, 65, Command.NONE, None
    while t < 12 and stage.phase == "walking":
        command = stage.step((x, 98), t, t)
        if command != Command.NONE:
            assert command == Command("left")
            last_key = t
            if previous == Command.NONE:
                x -= 1  # every pulse moves one pixel: never silent
        previous = command
        t += FRAME
    assert stage._last_end == "walk did not reach a drawn platform in time"
    assert t - FRAME - entered == 6.0 and last_key > entered + 5 and x > 49 + 1


# -- review regressions: frame rates other than 8 fps ------------------------

@pytest.mark.parametrize("frame_s", [.6, .5, .25, .125, 1 / 30])
@pytest.mark.parametrize("walls", [(), ((64.9, 95, 99),)])
def test_settle_and_release_hold_at_other_frame_rates(frame_s, walls):
    nav = returning_nav()
    world = World(nav._cfg, (65, 98), solids=[(60, 70, 98)], walls=walls, frame_s=frame_s)
    for _ in range(int(30 / frame_s)):
        world.frame(nav)
        if nav.recovery_state == "waiting_source" or nav.recovery_terminal:
            break
    log = world.log
    assert nav.recovery_state == "waiting_source" and log[-1]["pos"][1] == 105
    first = next(i for i, row in enumerate(log) if row["command"] != Command.NONE)
    # Three still frames over 0.5 s, then the entry frame: never fewer than four no-key frames.
    assert first >= 4 and log[first - 2]["now"] >= .5 - 1e-9
    assert first == max(2, math.ceil(.5 / frame_s - 1e-9)) + 2
    assert len(walk_runs(log)) >= 2
    assert_stage_keys_are_bounded(log)


# -- review regressions: a second return gets a fresh stage ------------------

def test_a_second_return_starts_with_fresh_regain_budgets_and_options(plans):
    nav = returning_nav(max_returns_per_lap=2)
    world = World(nav._cfg, (65, 98), solids=[(60, 70, 98)])
    world.run(nav)
    first = nav._regain
    assert nav.recovery_state == "completed" and first.moves == 1 and first._options
    assert nav.request_return_to_start() and nav.return_count == 2
    assert nav._regain is None
    commands = [nav.step((90, 98), 100 + i * FRAME) for i in range(7)]
    assert commands[:6] == [Command.NONE] * 6 and commands[6] == Command("left")
    assert nav._regain is not first
    assert nav._regain._current == regain_options(nav._cfg, (90, 98))[0]
    assert nav._regain.moves == 1 and nav._regain.perches == 1 and nav._regain.elapsed <= .75
