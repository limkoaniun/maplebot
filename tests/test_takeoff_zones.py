"""Opt-in take-off zones: ``route.circuit.takeoff_zones`` and ``maplebot/nav/zones.py``.

Offline only. The geometry tests read the versioned Construction Site plan
(examples/construction-site/user.yaml, read-only). The behaviour tests run real
``MinimapCircuit`` objects on scripted positions and in the small simulated world
of tests/test_transition_timing.py. That world is not the game and nothing here
is a live verification.

A zone only widens where "at the take-off" is true before the FIRST attempt of a
jump, ramp or drop. The tests therefore keep asking the same three questions:
is the settle still required (now with X stillness), does a retry go back to the
saved pixel, and is every landing still proven exactly as before.
"""
from copy import deepcopy
import math
from pathlib import Path
import random

import pytest

from maplebot.nav.circuit import MinimapCircuit, Platform
from maplebot.nav.editor import compile_plan
from maplebot.nav.recovery import plan_rejoin, plan_return_to_start, supported_platform
from maplebot.nav.zones import takeoff_zone
from maplebot.types import Command
from tests.test_pre_transition_combat import ATTACKS, brain, mob, see
from tests.test_transition_timing import (
    A, ALL, B, DROP, FAR, GAP, LADDER, PINNED as TIMING_PINNED, RAMP, STAIR_JUMP, World, _SITE_CIRCUIT,
    _child_launched, _hits, _returning, _to_first_drop, digest, drive, first, lap_nav, max_lease, on_time, run,
    scenario, script, site, waypoint)

ZONES = {"takeoff_zones": True}
PLAN = _SITE_CIRCUIT["editor_plan"]
TOP_DROP = waypoint("top-drop")  # 33: top x=75 -> floor-4
# Every other opt-in route.circuit key of this branch; EVERY adds the zones.
OTHER = {**ALL, "arrive_on_destination": True, "platform_rejoin": True, "approach_resume_credit": True}
EVERY = {**OTHER, **ZONES}
# The keys suggested for the first supervised run: every other key, zones for drops and gap jumps only.
SUGGESTED = {**OTHER, **ZONES, "takeoff_zone_kinds": ["drop", "jump"]}
SETTLING = ("settling", "jump_settling", "ramp_settling")


def zone(kind, source, destination, from_x, to_x=None, plan=PLAN, **options):
    """``takeoff_zone`` with the Construction Site tolerances: jump 0.4, ramp and drop 1.0."""
    tolerance = options.pop("legacy_tolerance_x", .4 if kind == "jump" else 1.0)
    return takeoff_zone(plan, kind=kind, from_platform=source, to_platform=destination, from_x=from_x,
                        to_x=from_x if to_x is None else to_x, legacy_tolerance_x=tolerance, **options)


def pixels(interval):
    """The whole minimap pixels inside a zone."""
    return list(range(math.ceil(interval[0] - 1e-9), math.floor(interval[1] + 1e-9) + 1))


def near(interval, expected):
    if interval is None or expected is None:
        return interval is None and expected is None
    return interval == pytest.approx(expected, abs=1e-3)


def surfaces(plan=PLAN):
    return {p["id"]: Platform(p["y"], p["left"], p["right"], p.get("right_y"), p.get("knots", ()))
            for p in plan["platforms"]}


def changed_plan(*, without=(), knots=None, connections=None):
    """A copy of the plan without some platforms (and what refers to them), or with extra knots."""
    plan = deepcopy(PLAN)
    plan["platforms"] = [p for p in plan["platforms"] if p["id"] not in without]
    plan["connections"] = [c for c in plan["connections"]
                           if c["from_platform"] not in without and c["to_platform"] not in without
                           and (connections is None or connections(c))]
    plan["route"] = [stop for stop in plan["route"] if stop["platform_id"] not in without]
    for platform_id, points in (knots or {}).items():
        next(p for p in plan["platforms"] if p["id"] == platform_id)["knots"] = points
    return plan


# --------------------------------------------------------------------------
# 1. Geometry on the real Construction Site plan
# --------------------------------------------------------------------------

# Zone of every saved connection in its saved direction (None: the saved spot alone), computed
# on 2026-11-16 from examples/construction-site/user.yaml with max_x 8 and margin_x 1.5.
#   ramps        8 px back along the stair, 1 px forward (the legacy tolerance)
#   gap jumps    from 0.75 px inside the platform end at x=89 to the saved pixel: 90 and 91
#   top-drop     the barricade (60.29-67.54) plus the 2.5 px clearance ends at 70.04
#   drop-N       the stairs begin at 66.787; minus the clearance that is 64.287
#   right drops  95 is the 8 px reach; floor-3-right and floor-2-right end at 111, and a fall
#                keeps 2.5 px from the destination's end (108.5); floor-1 runs on, so drop-right-2-1
#                is bounded by the 1.5 px margin from its source's end (109.5)
#   barricade-drop       None: the saved X=65 is 2 px from the top ladder at x=63
#   left-entry-pipe-drop None: the lower pipe lies in the fall column
EXPECTED = {
    "base-ladder": None, "base-ladder-down": None, "top-ladder": None, "top-ladder-down": None,
    "jump-1a": None, "jump-2a": None, "jump-3a": None, "barricade-jump": None, "floor-2-return": None,
    "stair-1-finish": (74, 83), "stair-2-finish": (74, 83), "stair-3-finish": (74, 83),
    "floor-2-gap": (89.75, 91.4), "floor-3-gap": (89.75, 91.4), "floor-4-gap": (89.75, 91.4),
    "top-drop": (70.0375, 83), "drop-4-3": (52, 64.287), "drop-3-2": (52, 64.287), "drop-2-1": (52, 64.287),
    "drop-right-4-3": (95, 108.5), "drop-right-3-2": (95, 108.5), "drop-right-2-1": (95, 109.5),
    "barricade-drop": None, "left-entry-pipe-drop": None, "left-entry-lower-drop": (11.5, 16.4),
    "left-entry-ground-drop": (11, 20.5), "right-entry-barricade-drop": (126, 129.85),
    "bottom-brick-stairs-to-ground": (101.4125, 110),
}
# One strictly supported point on every drawn platform, for the return paths.
STANDS = {"ground": (60, 105), "floor-1": (80, 88), "floor-2": (80, 71), "floor-3": (80, 55), "floor-4": (80, 38),
          "stair-1": (75, 82), "stair-2": (80, 65), "stair-3": (80, 49), "top": (100, 22),
          "floor-2-right": (100, 71), "floor-3-right": (100, 55), "floor-4-right": (100, 38),
          "barricade": (64, 34), "left-entry-pipe": (15, 81), "left-entry-lower-pipe": (15, 85),
          "left-entry-floor": (20, 88), "right-entry-barricade": (128, 84), "bottom-brick-stairs": (110, 99)}


def connection_zone(connection, plan=PLAN, **options):
    kind = "climb" if connection["kind"] == "rope" else connection["kind"]
    return zone(kind, connection["from_platform"], connection["to_platform"], connection["from_x"],
                connection["to_x"], plan, **options)


def test_every_connection_of_the_real_plan_has_its_pinned_zone_and_the_plan_is_not_modified():
    before = deepcopy(PLAN)
    assert {c["id"] for c in PLAN["connections"]} == set(EXPECTED)
    for connection in PLAN["connections"]:
        assert near(connection_zone(connection), EXPECTED[connection["id"]]), connection["id"]
    assert PLAN == before
    whole = {name: pixels(interval) for name, interval in EXPECTED.items() if interval}
    assert whole["stair-2-finish"] == list(range(74, 84)) and whole["floor-3-gap"] == [90, 91]
    assert whole["top-drop"] == list(range(71, 84)) and whole["drop-4-3"] == list(range(52, 65))
    assert whole["drop-right-3-2"] == list(range(95, 109)) and whole["drop-right-2-1"] == list(range(95, 110))
    assert whole["left-entry-lower-drop"] == list(range(12, 17))
    assert whole["left-entry-ground-drop"] == list(range(11, 21))
    assert whole["right-entry-barricade-drop"] == list(range(126, 130))
    assert whole["bottom-brick-stairs-to-ground"] == list(range(102, 111))


def test_route_transitions_and_the_walks_in_front_of_them_get_these_zones_from_the_circuit():
    compiled = compile_plan(PLAN, {k: v for k, v in _SITE_CIRCUIT.items() if k not in ("waypoints", "platforms")})
    assert compiled["waypoints"] == _SITE_CIRCUIT["waypoints"]  # the example is the route as it is compiled
    nav = MinimapCircuit(site(**ZONES))
    zoned = []
    for index, point in enumerate(_SITE_CIRCUIT["waypoints"]):
        if point["action"] == "walk":
            continue
        expected = EXPECTED[point["connection_id"]]
        assert near(nav._zone(index), expected), point["connection_id"]
        # The take-off walk carries the same zone; a rope has none and neither has its walk.
        assert nav._zone(index - 1) == nav._zone(index)
        zoned += [point["connection_id"]] * bool(expected)
    assert len(zoned) == 10 and sorted(set(zoned)) == [
        "drop-2-1", "drop-3-2", "drop-4-3", "floor-2-gap", "floor-3-gap", "floor-4-gap", "stair-1-finish",
        "stair-2-finish", "stair-3-finish", "top-drop"]
    walks = [i for i, p in enumerate(_SITE_CIRCUIT["waypoints"]) if p["action"] == "walk"]
    assert sum(nav._zone(i) is not None for i in walks) == 10  # and no other walk waypoint has one
    assert nav._cfg["editor_plan"] == PLAN and nav._cfg["waypoints"] == _SITE_CIRCUIT["waypoints"]  # nothing rewritten


def test_return_path_transitions_from_every_platform_get_these_zones_and_a_child_never_a_jump_zone():
    seen = set()
    for platform_id, pos in STANDS.items():
        cfg = site(**ZONES)
        assert supported_platform(cfg, pos)["id"] == platform_id
        child = MinimapCircuit(plan_return_to_start(cfg, pos), zone_kinds=("drop", "ramp"))
        for index, point in enumerate(child._cfg["waypoints"]):
            if point["action"] == "walk":
                continue
            assert point["action"] != "jump"  # this map returns by walks, ramps, drops and ropes
            assert near(child._zone(index), EXPECTED[point["connection_id"]]), (platform_id, point["connection_id"])
            seen.add(point["connection_id"])
    # Every drop and forward ramp of the plan lies on the lap or on one of these returns.
    assert seen >= {"drop-right-4-3", "drop-right-3-2", "drop-right-2-1", "barricade-drop", "left-entry-pipe-drop",
                    "left-entry-lower-drop", "left-entry-ground-drop", "right-entry-barricade-drop",
                    "bottom-brick-stairs-to-ground", "stair-1-finish", "stair-2-finish", "stair-3-finish",
                    "top-drop", "drop-4-3", "drop-3-2", "drop-2-1"}
    # A rejoin may cross the gap by its saved jump: the child takes off from the saved pixel only.
    rejoin = MinimapCircuit(plan_rejoin(site(**ZONES), (95, 71), 10), zone_kinds=("drop", "ramp"))
    jump = next(i for i, p in enumerate(rejoin._cfg["waypoints"]) if p.get("connection_id") == "floor-2-gap")
    assert rejoin._zone(jump) is None and rejoin._zone(jump - 1) is None
    assert near(MinimapCircuit(site(**ZONES))._zone(waypoint("floor-2-gap")), (89.75, 91.4))


def test_zone_lies_on_the_source_platform_and_contains_the_saved_spot_and_its_legacy_window():
    floors, before = surfaces(), deepcopy(PLAN)
    for connection in PLAN["connections"]:
        for options in ({}, {"max_x": 20}, {"max_x": 1}, {"margin_x": 1}, {"margin_x": 4},
                        {"max_x": 20, "margin_x": 4}):
            interval = connection_zone(connection, **options)
            if interval is None:
                continue
            source, x0 = floors[connection["from_platform"]], connection["from_x"]
            tolerance = .4 if connection["kind"] == "jump" else 1.0
            assert source.left <= interval[0] <= x0 <= interval[1] <= source.right, (connection["id"], options)
            assert interval[0] <= max(source.left, x0 - tolerance) and min(source.right, x0 + tolerance) <= interval[1]
            # ...and it is wider than that window, or it would have been None.
            assert interval[1] - interval[0] > min(source.right, x0 + tolerance) - max(source.left, x0 - tolerance)
            assert interval[0] >= x0 - options.get("max_x", 8) - 1e-9 or interval[0] >= x0 - tolerance - 1e-9
            assert interval[1] <= x0 + options.get("max_x", 8) + 1e-9 or interval[1] <= x0 + tolerance + 1e-9
    assert PLAN == before


def test_drop_zone_keeps_the_margin_from_platform_ends_and_the_reach_from_the_saved_spot():
    # floor-4 and floor-3 both begin at x=42; stair-3 begins at 66.787 and its foot is level with floor-3.
    # The source keeps margin_x from its ends; the destination's ends and every other platform keep the
    # fall clearance margin_x + 1, never less than 2.5 (half a hidden pixel plus 2 px of drift).
    assert near(zone("drop", "floor-4", "floor-3", 60), (52, 64.287))
    assert near(zone("drop", "floor-4", "floor-3", 60, max_x=3), (57, 63))
    assert near(zone("drop", "floor-4", "floor-3", 60, max_x=20), (44.5, 64.287))    # floor-3 from 42, + 2.5
    assert near(zone("drop", "floor-4", "floor-3", 60, margin_x=3), (52, 62.787))    # 66.787 - (3 + 1)
    assert near(zone("drop", "floor-4", "floor-3", 60, margin_x=1), (52, 64.287))    # 66.787 - 2.5, not - (1 + 1)
    assert near(zone("drop", "floor-4", "floor-3", 60, max_x=20, margin_x=4), (47, 61.787))
    # A saved spot inside the clearance keeps its own pixel only; just outside it the legacy window still counts.
    assert zone("drop", "floor-4", "floor-3", 44.4) is None
    assert near(zone("drop", "floor-4", "floor-3", 44.5), (43.5, 52.5))
    # floor-3-right and floor-2-right both end at x=111: the destination's end binds (111 - 2.5).
    # floor-1 runs on beyond x=111: under floor-2-right only the source margin binds (111 - 1.5).
    assert near(zone("drop", "floor-3-right", "floor-2-right", 103), (95, 108.5))
    assert near(zone("drop", "floor-3-right", "floor-2-right", 103, margin_x=1), (95, 108.5))
    assert near(zone("drop", "floor-3-right", "floor-2-right", 103, margin_x=4), (95, 106))
    assert near(zone("drop", "floor-2-right", "floor-1", 103), (95, 109.5))
    # The legacy window alone is no zone.
    assert zone("drop", "floor-4", "floor-3", 60, max_x=1) is None
    assert zone("drop", "floor-4", "floor-3", 60, legacy_tolerance_x=9) is None


def test_drop_zone_excludes_a_platform_in_or_beside_the_fall_column_and_needs_an_allowed_saved_spot():
    # stair-3 would catch a fall at x >= 66.787; with 2.5 px of clearance the saved spot must be left of 64.287.
    assert near(zone("drop", "floor-4", "floor-3", 64), (56, 65))  # 65 only as part of the legacy window
    assert zone("drop", "floor-4", "floor-3", 65) is None and zone("drop", "floor-4", "floor-3", 75) is None
    # The lower entry pipe (y=85) lies between the upper pipe (81) and the floor (88).
    assert zone("drop", "left-entry-pipe", "left-entry-floor", 15) is None
    assert near(zone("drop", "left-entry-lower-pipe", "left-entry-floor", 15), (11.5, 16.4))
    # From the top floor the barricade (y=34, x 60.29-67.54) lies in the column down to floor-4 (y=38).
    assert zone("drop", "top", "floor-4", 68) is None and zone("drop", "top", "floor-4", 70) is None
    assert near(zone("drop", "top", "floor-4", 72), (70.0375, 80))
    # On the right the top of stair-3 (higher than y=39.5 from x=88.52) and floor-4-right (from x=89)
    # are level with floor-4: the zone ends 2.5 px before the first of them.
    assert near(zone("drop", "top", "floor-4", 85, max_x=20), (70.0375, 86.02))
    # Without stair-3 the end of floor-4 itself (88.537) keeps the zone 2.5 px away.
    assert near(zone("drop", "top", "floor-4", 85, max_x=20, plan=changed_plan(without=("stair-3",))), (70.0375, 86.037))
    alone = changed_plan(without=("barricade", "stair-3", "floor-4-right"))
    assert near(zone("drop", "top", "floor-4", 72, plan=alone), (64, 80))
    # A platform above the source or below the destination does not matter: the top floor above
    # floor-4-right, and everything under floor-3-right.
    assert near(zone("drop", "floor-4-right", "floor-3-right", 103), (95, 108.5))


def test_drop_zone_needs_level_source_and_destination_and_a_destination_underneath():
    assert zone("drop", "stair-2", "floor-2", 84) is None           # on the slope of the source
    assert zone("drop", "floor-3", "floor-4", 60) is None           # the "destination" is above
    assert zone("drop", "floor-4", "floor-4-right", 60) is None     # no column the two share
    assert zone("drop", "floor-4", "floor-3", 60, 61) is None       # a drop lands under its take-off
    # A 1 px bump between x=55 and 57: no take-off whose 2.5 px clearance reaches its upper half.
    for platform_id, bump in (("floor-3", [[55, 55], [56, 56], [57, 55]]), ("floor-4", [[55, 38], [56, 37], [57, 38]])):
        assert near(zone("drop", "floor-4", "floor-3", 60, plan=changed_plan(knots={platform_id: bump})), (59, 64.287))
        assert zone("drop", "floor-4", "floor-3", 57, plan=changed_plan(knots={platform_id: bump})) is None
    # A slope of under half a pixel over the clearance is level enough.
    gentle = changed_plan(knots={"floor-3": [[56, 56]]})
    assert near(zone("drop", "floor-4", "floor-3", 60, plan=gentle), (52, 64.287))


def test_rope_columns_are_not_part_of_a_drop_zone():
    # The base ladder (x=49) hangs from floor-1 (88) down to the ground: in the column of a drop onto floor-1.
    assert near(zone("drop", "floor-2", "floor-1", 52), (51, 60))     # 51 only as part of the legacy window
    assert near(zone("drop", "floor-2", "floor-1", 46), (44.5, 47))  # floor-1 begins at 42: 2.5 px from it
    for inside in (47, 49, 50, 51):
        assert zone("drop", "floor-2", "floor-1", inside) is None
    # One floor higher the same ladder is far below the fall.
    assert near(zone("drop", "floor-3", "floor-2", 52), (44.5, 60))
    # The top ladder (x=63) ends on the barricade: the saved barricade drop at x=65 keeps its pixel.
    assert zone("drop", "barricade", "floor-4", 65) is None
    no_ladder = changed_plan(connections=lambda c: not c["id"].startswith("top-ladder"))
    assert near(zone("drop", "barricade", "floor-4", 65, plan=no_ladder), (61.7875, 66.0375))


def test_ramp_zone_runs_back_along_a_sloped_source_and_stops_where_support_is_ambiguous():
    assert near(zone("ramp", "stair-2", "floor-3-right", 82, 92), (74, 83))
    assert near(zone("ramp", "stair-2", "floor-3-right", 82, 92, max_x=3), (79, 83))
    # With a longer reach the zone ends where the stair foot comes within 1.5 px of floor-2 (y=71)...
    assert near(zone("ramp", "stair-2", "floor-3-right", 82, 92, max_x=20), (68.6624, 83))
    # ...and a saved spot inside that stretch, or at the stair top beside floor-3-right, has no zone.
    assert zone("ramp", "stair-2", "floor-3-right", 68, 92) is None
    assert zone("ramp", "stair-2", "floor-3-right", 89, 92) is None
    assert near(zone("ramp", "stair-2", "floor-3-right", 70, 92), (68.6624, 71))
    assert zone("ramp", "stair-2", "floor-3-right", 69, 92) is None  # nothing beyond its legacy window
    # Forward of the saved spot only the legacy tolerance, whatever the reach.
    assert near(zone("ramp", "stair-2", "floor-3-right", 82, 92, max_x=20, legacy_tolerance_x=.4), (68.6624, 82.4))
    # Walking down the brick stairs to the ground: back is to the right, and the margin holds at that end.
    assert near(zone("ramp", "bottom-brick-stairs", "ground", 102, 99), (101.4125, 110))
    assert near(zone("ramp", "bottom-brick-stairs", "ground", 118, 99, max_x=20), (117, 122.4125))


def test_reversed_ramp_is_judged_on_its_effective_source():
    # stair-2-finish walked backwards starts on the flat floor-3-right: no sloped source, no zone.
    assert zone("ramp", "floor-3-right", "stair-2", 92, 82) is None
    assert zone("ramp", "ground", "bottom-brick-stairs", 99, 102) is None
    # A ramp saved from a floor up onto a stair, compiled reversed for the way down.
    plan = {"version": 1, "name": "Down a stair", "image_size": [120, 100], "loop": False,
            "platforms": [{"id": "lower", "left": 0, "right": 40, "y": 50},
                          {"id": "stair", "left": 40, "right": 60, "y": 50, "right_y": 40},
                          {"id": "upper", "left": 60, "right": 100, "y": 40}],
            "connections": [{"id": "up", "kind": "ramp", "from_platform": "lower", "to_platform": "stair",
                             "from_x": 38, "to_x": 55}],
            "route": [{"platform_id": "stair", "x": 58, "dwell_s": 0}, {"platform_id": "lower", "x": 20, "dwell_s": 0}]}
    cfg = compile_plan(plan, {"editor_plan": plan, "tolerance_x": 1.0, "tolerance_y": 1.0, **ZONES})
    ramp = cfg["waypoints"][2]
    assert (ramp["action"], ramp["from_platform"], ramp["to_platform"], ramp["from_x"], ramp["x"]) == (
        "ramp", "stair", "lower", 55, 38)
    nav = MinimapCircuit(cfg)
    # Back from x=55 is up the stair: 1.5 px short of its top at 60, 1 px of legacy tolerance downhill.
    assert [nav._zone(i) for i in range(4)] == [None, (54.0, 58.5), (54.0, 58.5), None]
    assert zone("ramp", "lower", "stair", 38, 55, plan=plan) is None  # the saved direction starts on the flat floor


def test_jump_zone_is_only_the_last_pixels_before_a_level_gap():
    assert near(zone("jump", "floor-3-right", "floor-3", 91, 85), (89.75, 91.4))
    assert pixels(zone("jump", "floor-3-right", "floor-3", 91, 85)) == [90, 91]
    # Never away from the gap, whatever the reach and margin; toward it the reach still bounds the zone.
    assert near(zone("jump", "floor-3-right", "floor-3", 91, 85, max_x=20, margin_x=4), (89.75, 91.4))
    assert near(zone("jump", "floor-3-right", "floor-3", 91, 85, max_x=1), (90, 91.4))
    # The saved take-off must be 1 to 4 px from the platform end at x=89.
    assert near(zone("jump", "floor-3-right", "floor-3", 93, 85), (89.75, 93.4))
    for x0 in (89.9, 90, 94):  # too close, nothing beyond its own window, too far
        assert zone("jump", "floor-3-right", "floor-3", x0, 85) is None
    # floor-2-return takes off 4.6 px from the end of floor-2 (88.6): no zone. From 85 it would have one, rightward.
    assert zone("jump", "floor-2", "floor-2-right", 84, 91) is None
    assert near(zone("jump", "floor-2", "floor-2-right", 85, 91), (84.6, 87.85))
    # Stair-foot jumps (sloped destination), the barricade jump (4 px up) and a jump across two floors: none.
    assert zone("jump", "floor-3", "stair-3", 66, 69) is None
    assert zone("jump", "floor-4", "barricade", 59, 63) is None
    assert zone("jump", "floor-3-right", "floor-2", 91, 85) is None
    # A destination behind the jump direction, or a landing X that is not on it.
    assert zone("jump", "floor-3-right", "floor-3", 91, 95) is None
    assert zone("jump", "floor-3-right", "floor-3", 91, 88.9) is None
    wide = deepcopy(PLAN)
    floor_3 = next(p for p in wide["platforms"] if p["id"] == "floor-3")
    floor_3["right"] = 87.4  # a 1.6 px gap
    assert zone("jump", "floor-3-right", "floor-3", 91, 85, plan=wide) is None
    floor_3["right"] = 87.5  # 1.5 px: still the gap a standing jump clears
    assert near(zone("jump", "floor-3-right", "floor-3", 91, 85, plan=wide), (89.75, 91.4))
    assert zone("climb", "ground", "floor-1", 49) is None and zone("rope", "ground", "floor-1", 49) is None


MALFORMED = {
    "plan is None": {"plan": None}, "plan is empty": {"plan": {}}, "plan is a list": {"plan": []},
    "plan has platforms only": {"plan": {"platforms": PLAN["platforms"]}},
    "plan version": {"plan": {**PLAN, "version": 2}}, "platforms not a list": {"plan": {**PLAN, "platforms": "floors"}},
    "platform y not a number": {"plan": {**PLAN, "platforms": [{**PLAN["platforms"][0], "y": "low"},
                                                              *PLAN["platforms"][1:]]}},
    "bad knots": {"plan": {**PLAN, "platforms": [{**p, "knots": [[1, 2, 3]]} if p["id"] == "floor-4" else p
                                                 for p in PLAN["platforms"]]}},
    "bad connection": {"plan": {**PLAN, "connections": [{"id": "x", "kind": "rope"}]}},
    "unknown source": {"from_platform": "floor-9"}, "no destination": {"to_platform": None},
    "unhashable source": {"from_platform": ["floor-4"]}, "source is destination": {"to_platform": "floor-4"},
    "from_x None": {"from_x": None}, "from_x text": {"from_x": "60"}, "from_x nan": {"from_x": math.nan},
    "from_x inf": {"from_x": math.inf}, "from_x bool": {"from_x": True}, "from_x off the source": {"from_x": 30},
    "to_x nan": {"to_x": math.nan}, "to_x None": {"to_x": None},
    "kind walk": {"kind": "walk"}, "kind None": {"kind": None}, "kind climb": {"kind": "climb"},
    "max_x zero": {"max_x": 0}, "max_x negative": {"max_x": -1}, "max_x inf": {"max_x": math.inf},
    "max_x text": {"max_x": "8"}, "margin zero": {"margin_x": 0}, "margin nan": {"margin_x": math.nan},
    "tolerance negative": {"legacy_tolerance_x": -1}, "tolerance None": {"legacy_tolerance_x": None},
    "tolerance nan": {"legacy_tolerance_x": math.nan},
}


@pytest.mark.parametrize("name", sorted(MALFORMED))
def test_malformed_input_gives_no_zone_and_never_raises(name):
    arguments = {"plan": PLAN, "kind": "drop", "from_platform": "floor-4", "to_platform": "floor-3", "from_x": 60,
                 "to_x": 60, "max_x": 8.0, "margin_x": 1.5, "legacy_tolerance_x": 1.0}
    assert near(takeoff_zone(**arguments), (52, 64.287))
    assert takeoff_zone(**{**arguments, **MALFORMED[name]}) is None


def fall_meets(floors, source_id, x, drift, destination_id, steps=400):
    """Ids of the drawn platforms a straight fall meets first.

    The fall leaves ``source_id`` at ``x`` and has drifted ``drift`` px sideways by
    the time it is level with the destination. Every drawn platform but the source
    is checked along the way. Geometry only: nothing of maplebot/nav/zones.py is used.
    """
    start_y = floors[source_id].y_at(x)
    end_x = x + drift
    end_y = floors[destination_id].y_at(end_x) + 1.0  # one pixel through, so the destination itself is met
    previous = (x, start_y)
    for step in range(1, steps + 1):
        px, py = x + (end_x - x) * step / steps, start_y + (end_y - start_y) * step / steps
        met = {name for name, floor in floors.items()
               if name != source_id and floor.left <= px <= floor.right and floor.left <= previous[0] <= floor.right
               and previous[1] < floor.y_at(previous[0]) and py >= floor.y_at(px)}
        if met:
            return met
        previous = (px, py)
    return set()


@pytest.mark.parametrize("margin_x", [1.5, 1.0, 4.0])
def test_every_whole_pixel_of_every_drop_zone_falls_onto_its_saved_destination_and_nothing_else(margin_x):
    floors, cfg = surfaces(), site()
    checked = zoned_pixels = 0
    for connection in PLAN["connections"]:
        interval = connection_zone(connection, margin_x=margin_x)
        if connection["kind"] != "drop" or interval is None:
            continue
        source, destination = connection["from_platform"], connection["to_platform"]
        for pixel in pixels(interval):
            zoned_pixels += 1
            # A whole-pixel position hides where in that pixel the character really stands.
            for x in (pixel - .49, pixel - .25, pixel, pixel + .25, pixel + .49):
                # The take-off point is strictly and only on the source platform...
                assert supported_platform(cfg, (x, floors[source].y_at(x)))["id"] == source, (connection["id"], x)
                for drift in (-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2):
                    # ...the fall meets the destination before any other drawn platform...
                    assert fall_meets(floors, source, x, drift, destination) == {destination}, (
                        connection["id"], x, drift)
                    # ...and where it lands only the destination supports the character.
                    landed = (x + drift, floors[destination].y_at(x + drift))
                    assert supported_platform(cfg, landed)["id"] == destination, (connection["id"], x, drift)
                    checked += 1
    assert checked == 5 * 9 * zoned_pixels  # five points of each pixel, nine drifts each
    if margin_x == 1.5:  # the ten drop zones of the plan at the default margin
        assert zoned_pixels == 13 * 4 + 14 * 2 + 15 + 5 + 10 + 4
    # The check has teeth: one pixel right of the drop-4-3 zone a 2 px drift ends on the stair foot,
    # three pixels left of the top-drop zone on the barricade, and the upper entry pipe above the lower one.
    assert fall_meets(floors, "floor-4", 65, 2, "floor-3") == {"stair-3"}
    assert fall_meets(floors, "top", 68, -2, "floor-4") == {"barricade"}
    assert fall_meets(floors, "left-entry-pipe", 15, 0, "left-entry-floor") == {"left-entry-lower-pipe"}


# --------------------------------------------------------------------------
# 2. Keys: default off, validated, effective only with a plan and never in an editor test
# --------------------------------------------------------------------------

def test_key_defaults_off_and_its_numbers_are_not_read_while_off():
    nav = MinimapCircuit(site())
    assert nav.takeoff_zones is False and nav._zone_kinds == () and nav.zone_takeoffs == 0
    assert not hasattr(nav, "takeoff_zone_max_x") and not hasattr(nav, "takeoff_zone_margin_x")
    for bad in ("wide", None, True, -1, 99, math.nan):  # not read, so not validated
        MinimapCircuit(site(takeoff_zones=False, takeoff_zone_max_x=bad, takeoff_zone_margin_x=bad))
    on = MinimapCircuit(site(**ZONES))
    assert on._zone_kinds == ("jump", "ramp", "drop")
    assert (on.takeoff_zone_max_x, on.takeoff_zone_margin_x) == (8.0, 1.5)
    assert "takeoff_zones" not in _SITE_CIRCUIT  # the versioned example leaves it commented out


@pytest.mark.parametrize("bad", [1, 0, "true", None, 1.0])
def test_key_must_be_a_boolean(bad):
    with pytest.raises(ValueError, match="takeoff_zones must be true or false"):
        MinimapCircuit(site(takeoff_zones=bad))


@pytest.mark.parametrize("key, low, high", [("takeoff_zone_max_x", 1, 20), ("takeoff_zone_margin_x", 1, 4)])
def test_zone_numbers_are_range_checked_when_the_key_is_on(key, low, high):
    for bad in (low - .01, high + .01, math.nan, math.inf, True, "3", None):
        with pytest.raises(ValueError, match=key):
            MinimapCircuit(site(**ZONES, **{key: bad}))
        with pytest.raises(ValueError, match=key):  # a movement test validates what it is given too
            MinimapCircuit(site(**ZONES, movement_test=True, **{key: bad}))
    for good in (low, high):
        assert getattr(MinimapCircuit(site(**ZONES, **{key: good})), key) == good


def test_zones_need_the_saved_plan_and_a_movement_test_only_gets_them_from_its_parent():
    without_plan = site(**ZONES)
    del without_plan["editor_plan"]
    with pytest.raises(ValueError, match="takeoff_zones needs a saved editor plan"):
        MinimapCircuit(without_plan)
    assert MinimapCircuit({**without_plan, "movement_test": True})._zone_kinds == ()
    assert MinimapCircuit({**without_plan, "movement_test": True}, zone_kinds=("drop", "ramp"))._zone_kinds == ()
    # An editor connection test is a movement test built without the argument.
    editor = MinimapCircuit(site(**ZONES, movement_test=True, loop=False))
    assert editor.takeoff_zones and editor._zone_kinds == ()
    assert brain(site(**ZONES, movement_test=True, loop=False)).circuit._zone_kinds == ()
    child = MinimapCircuit(site(**ZONES, movement_test=True, loop=False), zone_kinds=("drop", "ramp"))
    assert child._zone_kinds == ("drop", "ramp")
    # The argument alone does nothing: the key decides.
    assert MinimapCircuit(site(movement_test=True, loop=False), zone_kinds=("drop", "ramp"))._zone_kinds == ()
    assert MinimapCircuit(site(), zone_kinds=("drop", "ramp"))._zone_kinds == ()
    assert MinimapCircuit(site(**ZONES), zone_kinds=("drop",))._zone_kinds == ("jump", "ramp", "drop")
    # A parent hands its children drops and ramps, never jumps, and only while its own zones are in effect.
    assert MinimapCircuit(site(**ZONES))._child_zone_kinds() == ("drop", "ramp")
    assert MinimapCircuit(site())._child_zone_kinds() == () == editor._child_zone_kinds()


@pytest.mark.parametrize("bad", [[], (), "drop", ["drop", "drop"], ["rope"], ["climb"], ["Drop"], [None], [["drop"]],
                                 None, {"drop": True}, ["drop", 1], True])
def test_zone_kinds_must_be_a_non_empty_list_of_known_kinds_without_repeats(bad):
    with pytest.raises(ValueError, match="takeoff_zone_kinds"):
        MinimapCircuit(site(**ZONES, takeoff_zone_kinds=bad))
    with pytest.raises(ValueError, match="takeoff_zone_kinds"):  # a movement test validates it too
        MinimapCircuit(site(**ZONES, takeoff_zone_kinds=bad, movement_test=True, loop=False), zone_kinds=("drop",))
    MinimapCircuit(site(takeoff_zones=False, takeoff_zone_kinds=bad))  # not read while the key is off
    assert not MinimapCircuit(site(takeoff_zone_kinds=bad))._zone_kinds


@pytest.mark.parametrize("kinds, route, child", [
    (None, ("jump", "ramp", "drop"), ("drop", "ramp")),
    (["drop", "jump", "ramp"], ("jump", "ramp", "drop"), ("drop", "ramp")),
    (["drop", "jump"], ("jump", "drop"), ("drop",)), (("ramp",), ("ramp",), ("ramp",)),
    (["jump"], ("jump",), ()), (["drop"], ("drop",), ("drop",))])
def test_zone_kinds_choose_the_route_kinds_and_children_get_their_drops_and_ramps(kinds, route, child):
    keys = {**ZONES, **({} if kinds is None else {"takeoff_zone_kinds": kinds})}
    nav = MinimapCircuit(site(**keys))
    assert nav._zone_kinds == route and nav._child_zone_kinds() == child
    for idx, kind in ((GAP, "jump"), (RAMP, "ramp"), (DROP, "drop"), (TOP_DROP, "drop")):
        # The transition and the walk in front of it: a zone only for a chosen kind.
        assert (nav._zone(idx) is not None) == (kind in route) and nav._zone(idx - 1) == nav._zone(idx), (kinds, kind)
    # A child built by this parent: a return from the lap-3 spot and from a stair.
    for start, idx in (((100, 55), RAMP + 1), ((76, 65), RAMP - 1)):
        parent = MinimapCircuit(site(start_idx=idx, **keys))
        assert parent.request_return_to_start()
        for frame in range(30):
            parent.step(start, frame * .12)
            if parent.recovery is not None:
                break
        assert parent.recovery._zone_kinds == child
        kinds_used = {p["action"] for i, p in enumerate(parent.recovery._cfg["waypoints"])
                      if p["action"] != "walk" and parent.recovery._zone(i) is not None}
        assert kinds_used <= set(child) and ("drop" in kinds_used) == ("drop" in child)
    # A movement test never takes more than the parent hands it, and never a jump.
    assert MinimapCircuit(site(**keys, movement_test=True, loop=False),
                          zone_kinds=("jump", "drop", "ramp"))._zone_kinds == child


def test_zone_kinds_without_ramp_keep_every_ramp_on_its_saved_spot_on_the_route_and_in_a_return():
    keys = {**ZONES, "takeoff_zone_kinds": ["drop", "jump"]}
    # The lap's ramp from inside what would be its zone: lined up exactly as without zones.
    for idx, pos in ((RAMP, (76, 65)), (RAMP - 1, (75, 65))):
        log, plain = script(lap_nav(idx, keys), [pos] * 12), script(lap_nav(idx), [pos] * 12)
        assert [(f.state, f.cmd) for f in log] == [(f.state, f.cmd) for f in plain]
    # Drops and gap jumps still use theirs.
    assert states(script(lap_nav(DROP, keys), [(56, 38)] * 4))[-1] == "dropping"
    assert states(script(lap_nav(GAP, keys), [(90, 55)] * 4))[-1] == "jumping"
    # A return from the stair: the ramp walks to its saved spot first, both drops use their zones.
    nav, world, log = _returning(MinimapCircuit, keys, {}, 90, start=(76, 65), idx=RAMP - 1)
    assert nav.state == "returned_to_start" and world.pos == (49, 105)
    ramp = first(log, lambda f: f.state == "return_ramping")
    assert abs(ramp.pos[0] - 82) <= 1 and [f for f in log if f.t < ramp.t and f.cmd.move_x == "right"]
    assert nav.zone_takeoffs == 2 and world.drops == 2


def test_unusable_plan_or_waypoints_without_saved_platforms_keep_the_saved_spot_and_never_raise():
    bare_points = [{k: v for k, v in p.items() if k not in ("platform_id", "from_platform", "to_platform")}
                   for p in _SITE_CIRCUIT["waypoints"]]
    for overrides in ({"editor_plan": {}}, {"editor_plan": {"platforms": []}}, {"editor_plan": {**PLAN, "version": 9}},
                      {"waypoints": bare_points}):
        nav = lap_nav(DROP, ZONES, **overrides)
        assert nav._zone_kinds == ("jump", "ramp", "drop")
        assert [nav._zone(i) for i in range(len(nav.waypoints))] == [None] * len(nav.waypoints)
        for idx, pos in ((DROP, (56, 38)), (RAMP, (76, 65)), (GAP, (90, 55)), (DROP - 1, (64, 38))):
            log = script(lap_nav(idx, ZONES, **overrides), [pos] * 8)
            assert [(f.state, f.cmd) for f in log] == [(f.state, f.cmd) for f in script(lap_nav(idx), [pos] * 8)]


def test_status_publishes_the_counter_beside_the_other_circuit_counters():
    # scripts/run_active.py has live side effects and is never imported or run: its text is read.
    source = (Path(__file__).resolve().parents[1] / "scripts" / "run_active.py").read_text(encoding="utf-8")
    assert "'zone_takeoffs': getattr(circuit, 'zone_takeoffs', 0)," in source
    assert "'arrival_count': getattr(circuit, 'arrival_count', 0)," in source
    assert MinimapCircuit(site()).zone_takeoffs == 0  # present with the key off too


# --------------------------------------------------------------------------
# 3. Off means the behaviour before this change
# --------------------------------------------------------------------------

class LapWorld(World):
    """The shared world's hop peaks 0.1 px under the 4 px barricade jump of the lap; this one clears it."""
    JUMP_VY = -26.0


def knock(world, dx):
    """A hit: displaced along the floor it stands on (a stair too), falling when that is past the floor's end."""
    if world.air or world.rope is not None:
        return
    ground = world._ground()
    x = world.x + dx
    on_floor = ground is not None and ground[1].left <= x <= ground[1].right
    world.put((x, ground[1].y_at(x) if on_floor else world.y))
    if not on_floor and world._ground(1.0) is None:
        world.air = True


def knocks(seed, *, until=900.0):
    """Seeded hits of 2-6 px every 4-9 s, as ``drive`` events."""
    rng, events, t = random.Random(seed), [], 3.0
    while t < until:
        dx = rng.choice((-1, 1)) * rng.uniform(2.0, 6.0)
        events.append((round(t, 2), lambda world, dx=dx: knock(world, dx)))
        t += rng.uniform(4.0, 9.0)
    return events


def lap(keys, *, cls=MinimapCircuit, model=None, events=(), limit_s=900.0, dt=.12):
    """The whole lap from point 1 until it wraps or the route ends for good; returns (nav, world, log, wrapped)."""
    cfg = site(start_idx=0, **keys)
    nav, world = cls(cfg), LapWorld(cfg, (49, 105), **(model or {}))
    seen = {"previous": nav.idx, "wrapped": False}

    def stop(n):
        if seen["previous"] == len(n.waypoints) - 1 and n.idx == 0 and n.state != "returned_to_start":
            seen["wrapped"] = True
        seen["previous"] = n.idx
        return seen["wrapped"] or n.recovery_terminal
    log = drive(nav, world, limit_s=limit_s, dt=dt, events=events, stop=stop)
    return nav, world, log, seen["wrapped"]


SLOW = {"startup_s": .02, "coast": .3}
# Scripted frames, 0.1 s apart, through the branches the zones touch: a gap jump whose first
# flight fails and is retried, a ramp knocked back once and then walked, a drop that does not
# leave its floor, is retried, caught by a ledge and lands beside its point, and a rope.
SCRIPTS = {
    "scripted jump": (GAP, [(96, 55)] * 3 + [(90, 55)] * 6 + [(91, 55)] * 4 + [(88, 50)] * 24 + [(90, 55)] * 12
                      + [(91, 55)] * 4 + [(90, 53), (88, 51), (87, 53), (86, 54)] + [(85, 55)] * 9),
    "scripted ramp": (RAMP, [(76, 65)] * 6 + [(81, 64)] * 4 + [(83, 62), (84, 60)] + [(79, 65)] * 5 + [(76, 65)] * 4
                      + [(81, 64)] * 4 + [(83, 62), (85, 60), (87, 58), (89, 55)] + [(92, 55)] * 9),
    "scripted drop": (DROP, [(56, 38)] * 6 + [(60, 38)] * 16 + [(57, 38)] * 3 + [(60, 38)] * 2 + [(60, 42)]
                      + [(56, 46)] * 8 + [(60, 46)] * 4 + [(60, 50)] + [(57, 55)] * 30 + [(59, 55)] * 9),
    "scripted climb": (LADDER, [(54, 105)] * 4 + [(49, 105)] * 5 + [(49, 103 - i) for i in range(15)]
                       + [(49, 88)] * 8),
}
TIMING_SCENARIOS = ("ramp", "gap_jump", "stair_jump", "drop", "climb", "return", "rejoin", "ramp_blocked_then_return")


def off_scenario(name, cls=MinimapCircuit, **keys):
    """Runs for the default-off differential. ``cls`` lets the saved pre-change class run the same code.

    ``"<kind> + other keys"`` turns every other opt-in key of this branch on.
    """
    kind, _, other = name.partition(" + ")
    if kind in TIMING_SCENARIOS:  # the closed-loop runs of tests/test_transition_timing.py
        return scenario(kind, cls=cls, **({**ALL, "approach_resume_credit": True} if other else {}), **keys)
    keys = {**(OTHER if other else {}), **keys}
    if kind in SCRIPTS:
        idx, positions = SCRIPTS[kind]
        return script(cls(site(start_idx=idx, return_on_block=False, **keys)), positions)
    if kind == "lap":
        return lap(keys, cls=cls, model=SLOW if other else None)[2]
    if kind == "knocked lap":
        return lap(keys, cls=cls, events=knocks(5 if other else 3))[2]
    if kind == "return from the top":
        return _returning(cls, keys, {}, 90, start=(100, 22), idx=TOP_DROP - 2)[2]
    if kind == "return from stair-2":
        return _returning(cls, keys, {}, 90, start=(80, 65), idx=RAMP - 1)[2]
    if kind == "return under hits":
        return _returning(cls, keys, SLOW, 90, events=_hits(1.5))[2]
    raise KeyError(name)


# Digests of the frame-by-frame (position, command, lease, state, index, transition_committed)
# sequences produced on 2026-11-16 by maplebot/nav/circuit.py as it was BEFORE the zones existed
# (commit f001d9d): a saved copy of that class was run through this same ``off_scenario``.
PINNED = {
    "lap": "a407db5c5db300ec", "lap + other keys": "f356ab2ecf234a73",
    "knocked lap": "eed1ac32e11ccba9", "knocked lap + other keys": "ab00f86f804377af",
    "return from the top": "74b10b9c0de26e70", "return from the top + other keys": "bef6e2510c36f5e0",
    "return from stair-2": "20b424f1b9966457", "return from stair-2 + other keys": "341ab46b6e34422c",
    "return under hits": "973fd3894ec1a58f", "return under hits + other keys": "f80a8171e978b93d",
    "scripted climb": "3595203807f2c1a4", "scripted climb + other keys": "af977e0eac5a5d43",
    "scripted drop": "d2fd3ee406a6054d", "scripted drop + other keys": "85f3ef8c8ad4d260",
    "scripted jump": "dec0807ba0d4102f", "scripted jump + other keys": "dec0807ba0d4102f",
    "scripted ramp": "5df026512afd0085", "scripted ramp + other keys": "b2679ca1a263b6df",
    "ramp": "7d4ef62d2b767b4a", "ramp + other keys": "1f3f105ef731fa15",
    "gap_jump": "3b0c8363ba06066b", "gap_jump + other keys": "a9411416afc56eff",
    "stair_jump": "27bec15a64fe332c", "stair_jump + other keys": "545f803f9005d8b0",
    "drop": "f0f6ed041c0ed7dd", "drop + other keys": "ef5bc3b91a3a7ec5",
    "climb": "3c9ea4c1a58971c3", "climb + other keys": "ef355dd791744f9e",
    "return": "51a5ddfbbbf45d6f", "return + other keys": "a836efec64ae794c",
    "rejoin": "a1ee9bf2490ed5bf", "rejoin + other keys": "2961a69d0c5ee031",
    "ramp_blocked_then_return": "c4c88df617f01ba1", "ramp_blocked_then_return + other keys": "d827fc135986802d",
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_key_absent_or_false_reproduces_the_sequences_of_the_code_before_the_zones(name):
    absent = off_scenario(name)
    assert len(absent) > 30 and digest(absent) == PINNED[name]
    # False, with the two numbers set to values that would be refused if they were read.
    off = off_scenario(name, takeoff_zones=False, takeoff_zone_max_x="wide", takeoff_zone_margin_x=None,
                       takeoff_zone_kinds=["rope"])
    assert digest(off) == PINNED[name]


# Everything of the feature except its gate ``_zone_kinds``: with the key off none of it may be read.
ZONE_NAMES = ("_zone_open", "_zone_x", "_zone_start_x", "_zone_land_since", "_zone_cache", "zone_takeoffs",
              "takeoff_zone_max_x", "takeoff_zone_margin_x", "_zone", "_open_zone", "_zone_settle", "_zone_takeoff",
              "_zone_landed", "_legacy_tolerance", "ZONE_LANDING_DRIFT_X")


class Unzoned(MinimapCircuit):
    def __getattribute__(self, name):
        if name in ZONE_NAMES:
            raise AssertionError(f"{name} was read although takeoff_zones is off")
        return super().__getattribute__(name)


@pytest.mark.parametrize("name", ["knocked lap", "knocked lap + other keys", "return from stair-2",
                                  "rejoin + other keys", "scripted jump", "scripted ramp + other keys",
                                  "scripted drop + other keys"])
def test_with_the_key_off_nothing_of_the_feature_but_its_gate_is_ever_read(name, monkeypatch):
    import maplebot.nav.circuit as circuit_module

    monkeypatch.setattr(circuit_module, "MinimapCircuit", Unzoned)  # return and rejoin children too
    assert digest(off_scenario(name, cls=Unzoned)) == PINNED[name]
    assert digest(off_scenario(name, cls=Unzoned, takeoff_zones=False)) == PINNED[name]
    with pytest.raises(AssertionError, match="was read although"):  # the guard itself works
        off_scenario(name, cls=Unzoned, **ZONES)


# --------------------------------------------------------------------------
# 4. Inside a zone: no approach key, the unchanged settle plus X stillness, first attempt only
# --------------------------------------------------------------------------

def stair_y(x, stair="stair-2"):
    """The whole minimap pixel the dot shows on a stair."""
    return math.floor(surfaces()[stair].y_at(x) + .5)


def states(log):
    return [f.state for f in log]


def keys_sent(log):
    return [(f.cmd.move_x, f.cmd.move_y, f.cmd.action) for f in log]


TAKEOFF = {GAP: ("jump_settling", "jumping", ("left", "none", "jump")),
           RAMP: ("ramp_settling", "ramping", ("right", "none", "none")),
           DROP: ("settling", "dropping", ("none", "down", "jump"))}
HEIGHT = {GAP: lambda x: 55, RAMP: stair_y, DROP: lambda x: 38}
LEGACY = {GAP: [91], RAMP: [81, 82, 83], DROP: [59, 60, 61]}  # pixels inside the saved spot's own tolerance


@pytest.mark.parametrize("idx, zone_pixels", [(GAP, [90, 91]), (RAMP, list(range(74, 84))),
                                              (DROP, list(range(52, 65)))], ids=["gap jump", "ramp", "drop"])
def test_every_pixel_of_a_zone_takes_off_after_the_unchanged_settle_without_one_approach_key(idx, zone_pixels):
    settling, launched, key = TAKEOFF[idx]
    assert pixels(MinimapCircuit(site(**ZONES))._zone(idx)) == zone_pixels
    for x in zone_pixels:
        nav, pos = lap_nav(idx, ZONES), (x, HEIGHT[idx](x))
        log = script(nav, [pos] * 4)
        # align_settle_s (0.25 s) on the same pixel and height, then the take-off key from where it stands.
        assert states(log) == [settling] * 3 + [launched], (idx, x)
        assert [f.cmd for f in log[:3]] == [Command.NONE] * 3 and keys_sent(log)[3] == key, (idx, x)
        assert [f.committed for f in log] == [False, False, False, True], (idx, x)
        assert nav.zone_takeoffs == (x not in LEGACY[idx]) and nav.retries == 0, (idx, x)
        assert nav._zone_start_x == x and not nav._zone_open, (idx, x)
        # Without the key the same pixel takes off only inside the saved spot's own tolerance.
        plain = script(lap_nav(idx), [pos] * 4)
        assert (states(plain) == states(log)) == (x in LEGACY[idx]), (idx, x)
    for x in (zone_pixels[0] - 1, zone_pixels[-1] + 1):  # one pixel outside: lined up exactly as before
        pos = (x, HEIGHT[idx](x))
        log, plain = script(lap_nav(idx, ZONES), [pos] * 12), script(lap_nav(idx), [pos] * 12)
        assert [(f.state, f.cmd) for f in log] == [(f.state, f.cmd) for f in plain], (idx, x)
        assert {f.state for f in log} == {settling.replace("settling", "aligning")}, (idx, x)


@pytest.mark.parametrize("idx, first_x, second_x", [(GAP, 90, 91), (RAMP, 76, 77), (DROP, 56, 57), (DROP, 59, 60),
                                                   (RAMP, 82, 83)],
                         ids=["gap jump", "ramp", "drop", "drop inside its legacy window",
                              "ramp inside its legacy window"])
def test_a_settle_restarts_when_x_moves_and_takes_off_only_after_a_whole_settle_on_one_pixel(idx, first_x, second_x):
    settling, launched, key = TAKEOFF[idx]
    moved = [(first_x, HEIGHT[idx](first_x))] * 2 + [(second_x, HEIGHT[idx](first_x))] * 4
    nav = lap_nav(idx, ZONES)
    log = script(nav, moved)  # the height reading did not change: only X restarts this settle
    assert states(log) == [settling] * 5 + [launched] and keys_sent(log)[5] == key
    assert all(f.cmd == Command.NONE for f in log[:5]) and nav._zone_start_x == second_x
    # Both pixels inside the legacy window: without the key the shift does not restart the settle.
    # That X stillness is the one thing a zone adds there.
    if first_x in LEGACY[idx] and second_x in LEGACY[idx]:
        assert states(script(lap_nav(idx), moved))[:4] == [settling] * 3 + [launched]
    # Never still for a whole settle: no take-off at all, and the pre-takeoff clock ends it as before.
    restless = [(first_x, HEIGHT[idx](first_x)), (second_x, HEIGHT[idx](first_x))] * 80
    nav = lap_nav(idx, ZONES)
    log = script(nav, restless)
    assert not any(f.cmd.action == "jump" for f in log) and launched not in states(log)
    assert first(log, lambda f: f.state != settling).t == pytest.approx(15.0)
    assert (nav.reason or "jump timeout") in ("jump timeout", "ramp timeout", "drop timeout") and nav.zone_takeoffs == 0


@pytest.mark.parametrize("idx, pos, shifted", [(GAP, (90, 55), (90, 54)), (RAMP, (76, 65), (76, 66)),
                                               (DROP, (56, 38), (56, 39))], ids=["gap jump", "ramp", "drop"])
def test_a_height_change_restarts_the_settle_inside_a_zone_as_on_the_saved_pixel(idx, pos, shifted):
    settling, launched, key = TAKEOFF[idx]
    # Still supported on the source floor, the dot one pixel off in height: the settle starts again
    # (the drop keeps its own rule for it: the height has to be stable for the settle time as well).
    log = script(lap_nav(idx, ZONES), [pos] * 2 + [shifted] * 4)
    assert states(log) == [settling] * 5 + [launched] and keys_sent(log)[5] == key
    saved = {GAP: (91, 55), RAMP: (82, 63), DROP: (60, 38)}[idx]
    lifted = (saved[0], saved[1] + shifted[1] - pos[1])
    assert states(script(lap_nav(idx), [saved] * 2 + [lifted] * 4)) == [settling] * 5 + [launched]  # unchanged rule


def test_a_suspended_frame_restarts_the_settle_inside_a_zone_like_on_the_saved_pixel():
    for idx, x in ((GAP, 90), (RAMP, 76), (DROP, 56)):
        settling, launched, _ = TAKEOFF[idx]
        pos = (x, HEIGHT[idx](x))
        log = script(lap_nav(idx, ZONES), [pos, pos, "suspend", pos, pos, pos, pos])
        assert states(log)[3:] == [settling] * 3 + [launched], idx  # a full 0.25 s again after the fight frame


def test_a_pulse_leased_before_the_zone_is_cut_on_the_first_frame_inside_it():
    # Drop drop-4-3 with far take-off pulses: 6 px right of the saved spot, 2 px outside the zone.
    nav = lap_nav(DROP, {**B, **ZONES})
    log = script(nav, [(66, 38)] * 4 + [(65, 38)] + [(64, 38)] * 4)
    assert [f.cmd for f in log[:3]] == [Command.NONE] * 3  # the release before a far pulse
    assert log[3].cmd.move_x == "left" and log[3].cmd.valid_for_s == pytest.approx(.45)
    assert log[4].cmd.move_x == "left" and log[4].cmd.valid_for_s == pytest.approx(.35)  # the same pulse, running out
    # First frame inside the zone: no key at all, and nothing of the pulse is kept.
    assert log[5].state == "settling" and log[5].cmd == Command.NONE
    assert nav._pulse_until is None and not nav._pulse_on
    assert states(log)[5:] == ["settling"] * 3 + ["dropping"] and keys_sent(log)[8] == ("none", "down", "jump")
    assert nav._zone_start_x == 64
    # The walk waypoint in front of that drop, with far walk pulses.
    nav = lap_nav(DROP - 1, {**FAR, **ZONES})
    log = script(nav, [(66, 38)] * 4 + [(65, 38)] + [(64, 38)] * 2)
    assert log[3].cmd.valid_for_s == pytest.approx(.45) and log[4].cmd.valid_for_s == pytest.approx(.35)
    assert (log[5].state, log[5].cmd) == ("dwelling", Command.NONE) and nav._pulse_until is None


def test_after_a_failed_flight_the_gap_jump_needs_its_saved_pixel_again():
    nav = lap_nav(GAP, ZONES)
    log = script(nav, [(90, 55)] * 4 + [(88, 50)] * 21 + [(90, 55)] * 20 + [(91, 55)] * 4)
    assert log[3].state == "jumping" and log[3].pos == (90, 55)  # the first attempt, from the zone
    assert on_time(first(log, lambda f: f.state == "jump_retry").t, .3 + 2.0) and nav.retries == 1
    # Back on pixel 90 after the rest: lined up toward 91 with the fine taps, never settled there.
    again = [f for f in log if f.pos == (90, 55) and f.t > 3.05]
    assert len(again) == 14 and {f.state for f in again} == {"jump_aligning"}
    assert {f.cmd.move_x for f in again} == {"none", "right"} and max_lease(again) == pytest.approx(.04)
    assert states(log)[-4:] == ["jump_settling"] * 3 + ["jumping"] and log[-1].pos == (91, 55)
    assert nav.zone_takeoffs == 1 and nav._zone_start_x is None
    # A waypoint clock that ran out while lining up is a failed attempt too.
    nav = lap_nav(GAP, ZONES)
    log = script(nav, [(89, 55)] * 157 + [(90, 55)] * 12)
    assert on_time(first(log, lambda f: f.state == "jump_retry").t, 15.0)
    back = [f for f in log if f.pos == (90, 55)]
    assert {f.state for f in back[2:]} == {"jump_aligning"} and not any(f.cmd.action == "jump" for f in log)
    assert nav.zone_takeoffs == 0


def test_after_a_knockback_the_ramp_needs_its_saved_spot_again():
    nav = lap_nav(RAMP, ZONES)
    log = script(nav, [(76, 65)] * 4 + [(78, 65)] + [(74, 65)] * 5 + [(76, 65)] * 12
                 + [(81, 64)] * 4 + [(83, 62), (80, 65)])
    assert states(log)[:5] == ["ramp_settling"] * 3 + ["ramping"] * 2
    # 2 px behind where it started (x=76): the one bounded wait, then re-armed on this stair.
    assert states(log)[5:10] == ["ramp_waiting_knockback"] * 3 + ["ramp_realigning", "ramp_aligning"]
    # Pixel 76 again, still inside the zone's span: lined up toward x=82 as without a zone.
    again = log[10:22]
    assert {f.state for f in again} == {"ramp_aligning"} and {f.cmd.move_x for f in again} == {"none", "right"}
    assert states(log)[22:26] == ["ramp_settling"] * 3 + ["ramping"] and nav._zone_start_x is None
    # The second start is judged from the saved X: 2 px behind it leaves the corridor, as it always did.
    assert log[-1].state == "blocked" and nav.reason == "left ramp corridor" and nav.zone_takeoffs == 1


def test_after_a_launch_that_did_not_leave_the_floor_the_drop_needs_its_saved_spot_again():
    nav = lap_nav(DROP, ZONES)
    log = script(nav, [(56, 38)] * 23)
    assert log[3].cmd == Command("none", "down", "jump") and log[3].pos == (56, 38)
    retried = first(log, lambda f: f.state == "aligning")
    assert retried.t == pytest.approx(1.2) and nav.retries == 1  # the unchanged 0.8 s rule, one retry spent
    again = log[12:23]
    assert {f.state for f in again} == {"aligning"} and {f.cmd.move_x for f in again} == {"none", "right"}
    log += script(nav, [(60, 38)] * 4 + [(60, 45), (58, 52)] + [(57, 55)] * 30 + [(59, 55)] * 2, t0=2.3)
    assert states(log)[23:27] == ["settling"] * 3 + ["dropping"] and log[26].pos == (60, 38)
    # Launched from the saved spot this time: the landing 3 px beside it is corrected as before.
    beside = [f for f in log if f.pos == (57, 55)]
    assert {f.state for f in beside} == {"dwelling"} and nav.idx == DROP + 1
    assert [f for f in beside if f.cmd.move_x == "right"] and first(log, lambda f: f.idx == DROP + 1).pos == (59, 55)
    assert nav.zone_takeoffs == 1
    # Knocked off the floor while lining up again, without a second launch: the failed attempt left
    # nothing behind, so this landing beside the saved X is corrected as before too.
    nav = lap_nav(DROP, ZONES)
    log = script(nav, [(56, 38)] * 14 + [(57, 43), (57, 49), (57, 53)] + [(57, 55)] * 14)
    assert len(launches(log)) == 1 and nav.retries == 1 and nav._zone_start_x is None and nav.idx == DROP
    assert [f for f in log if f.pos == (57, 55) and f.state == "dwelling" and f.cmd.move_x == "right"]


@pytest.mark.parametrize("launched", [True, False], ids=["launched from the zone", "knocked off before the launch"])
def test_an_intermediate_ledge_gets_no_zone(launched):
    start = [(56, 38)] * (4 if launched else 2)
    nav = lap_nav(DROP, ZONES)
    log = script(nav, start + [(56, 42)] + [(56, 46)] * 16 + [(60, 46)] * 4 + [(60, 50), (58, 53)] + [(57, 55)] * 12)
    assert any(f.cmd.action == "jump" for f in log[:4]) == launched
    ledge = [f for f in log if f.pos == (56, 46)]
    # Caught at y=46, stable there: a new source floor. The zone was drawn for floor-4, so on the ledge
    # the same X is lined up toward the saved spot again.
    proven = states(ledge).index("aligning")
    assert proven in (5, 6) and states(ledge)[:proven] == ["dropping"] * proven  # 0.4 s stable on the ledge
    assert {f.state for f in ledge[proven:]} == {"aligning"}
    assert {f.cmd.move_x for f in ledge[proven:]} == {"none", "right"}
    assert not any(f.cmd.action == "jump" for f in ledge)
    relaunch = first(log, lambda f: f.pos == (60, 46) and f.cmd.action == "jump")
    assert relaunch is not None and nav._zone_start_x is None
    # ...and the landing after it keeps the saved spot's own tolerance.
    assert {f.state for f in log if f.pos == (57, 55)} == {"dwelling"} and nav.idx == DROP
    assert nav.zone_takeoffs == launched


# --------------------------------------------------------------------------
# 5. The walk waypoint in front of a zoned take-off
# --------------------------------------------------------------------------

def test_walk_in_front_of_a_zoned_drop_arrives_anywhere_in_the_zone_with_its_unchanged_dwell():
    walk = DROP - 1  # walk (60,38) on floor-4, dwell 0.6 s, then drop-4-3
    assert _SITE_CIRCUIT["waypoints"][walk]["dwell_s"] == .6
    for x in (52, 57, 64):
        nav = lap_nav(walk, ZONES)
        log = script(nav, [(x, 38)] * 9)
        assert states(log)[:6] == ["dwelling"] * 6 and all(f.cmd == Command.NONE for f in log), x
        assert on_time(first(log, lambda f: f.idx == DROP).t, .6), x  # the whole dwell, then the drop
        assert log[-1].state == "settling", x
        plain = script(lap_nav(walk), [(x, 38)] * 8)
        assert {f.state for f in plain} == {"walking"} and plain[-1].idx == walk, x
    for x in (51, 65):  # one pixel outside the zone: the walk goes on toward the saved X
        nav = lap_nav(walk, ZONES)
        log = script(nav, [(x, 38)] * 8)
        assert {f.state for f in log} == {"walking"} and nav.idx == walk, x
    # Knocked out of the zone during the dwell: the dwell starts again once it is back inside.
    nav = lap_nav(walk, ZONES)
    log = script(nav, [(64, 38)] * 4 + [(66, 38)] * 2 + [(64, 38)] * 8)
    assert states(log)[:6] == ["dwelling"] * 4 + ["walking"] * 2
    assert on_time(first(log, lambda f: f.idx == DROP).t, .6 + .6)


def test_walk_in_front_of_a_zoned_ramp_or_jump_arrives_in_its_zone_too():
    for walk, pos, idle in ((RAMP - 1, (75, 65), "ramp_settling"), (GAP - 1, (90, 55), "jump_settling")):
        nav = lap_nav(walk, ZONES)
        log = script(nav, [pos] * 3)
        assert _SITE_CIRCUIT["waypoints"][walk]["dwell_s"] == 0
        assert (log[0].state, log[0].idx) == ("dwelling", walk + 1) and states(log)[1:] == [idle] * 2
        assert all(f.cmd == Command.NONE for f in log)
    assert {f.state for f in script(lap_nav(RAMP - 1), [(75, 65)] * 3)} == {"walking"}


def test_walk_rule_still_needs_the_walk_floor_and_its_height():
    walk = DROP - 1
    # Inside the zone's X range but 2 px above the floor (a hop): not arrived.
    log = script(lap_nav(walk, ZONES), [(64, 36)] * 6)
    assert "dwelling" not in states(log) and log[-1].idx == walk
    # Inside it but on floor-3 underneath: the floor is wrong, nothing arrives at this waypoint.
    nav = lap_nav(walk, ZONES)
    log = script(nav, [(62, 55)] * 3)
    assert states(log) == ["waiting_floor"] * 3 and nav.idx == walk
    # On the stair before the ramp, in the zone's X range but 3 px above its surface.
    log = script(lap_nav(RAMP - 1, ZONES), [(76, 62)] * 6)
    assert "dwelling" not in states(log) and log[-1].idx == RAMP - 1


WALK_103 = {"x": 103, "y": 55.0, "action": "walk", "dwell_s": 0, "platform_id": "floor-3-right"}
DROP_32 = {"x": 103, "y": 71.0, "action": "drop", "dwell_s": .6, "from_platform": "floor-3-right",
           "to_platform": "floor-2-right", "connection_id": "drop-right-3-2"}
AFTER = {"x": 100, "y": 71.0, "action": "walk", "dwell_s": .6, "platform_id": "floor-2-right"}


def hand_made(points, **overrides):
    return MinimapCircuit(site(waypoints=deepcopy(points), start_idx=0, return_on_block=False, **ZONES, **overrides))


def test_walk_rule_is_only_for_the_walk_that_leads_to_that_take_off():
    lap_circuit = MinimapCircuit(site(**ZONES))
    points = _SITE_CIRCUIT["waypoints"]
    # Route stop 15 (x=96) is followed by the take-off walk 16 (x=91), not by the jump: no zone, and
    # standing inside the gap jump's zone is not an arrival there.
    assert (points[GAP - 2]["x"], points[GAP - 1]["x"]) == (96, 91) and lap_circuit._zone(GAP - 2) is None
    log = script(lap_nav(GAP - 2, ZONES), [(91, 55)] * 8)
    assert {f.state for f in log} == {"walking"} and log[-1].idx == GAP - 2
    # A walk in front of a rope, and the last waypoint of the loop (the next one is point 1): none.
    assert lap_circuit._zone(LADDER - 1) is None and lap_circuit._zone(len(points) - 1) is None
    zone_32 = (95.0, 108.5)
    assert [hand_made([WALK_103, DROP_32, AFTER])._zone(i) for i in range(3)] == [zone_32, zone_32, None]
    gap = points[GAP]  # the gap jump takes off from floor-3-right at x=91 and has a zone of its own
    for unrelated in ({**WALK_103, "x": 96},                                 # another X on the right platform
                      {**WALK_103, "x": 91, "platform_id": "floor-3"},       # the right X on another platform
                      {"x": 91, "y": 55.0, "action": "walk", "dwell_s": 0}):  # no saved platform at all
        nav = hand_made([unrelated, gap, points[GAP + 1]])
        assert nav._zone(0) is None and near(nav._zone(1), (89.75, 91.4)), unrelated
    nav = hand_made([{**WALK_103, "x": 96}, gap, points[GAP + 1]])
    log = script(nav, [(91, 55)] * 8)  # inside the jump's zone, 5 px from this walk's X
    assert {f.state for f in log} == {"walking"} and nav.idx == 0
    # A drop takes off from the walk waypoint before it: without that walk at its X on its platform
    # the drop has no zone either.
    for unrelated in ({**WALK_103, "x": 100}, {**WALK_103, "platform_id": "floor-4-right", "y": 38.0},
                      {"x": 103, "y": 55.0, "action": "walk", "dwell_s": 0}):
        nav = hand_made([unrelated, DROP_32, AFTER])
        assert [nav._zone(i) for i in range(3)] == [None, None, None], unrelated
    # No wrap past the end of a route that does not loop.
    assert [hand_made([DROP_32, AFTER, WALK_103], loop=True)._zone(i) for i in range(3)] == [zone_32, None, zone_32]
    assert [hand_made([DROP_32, AFTER, WALK_103], loop=False)._zone(i) for i in range(3)] == [zone_32, None, None]


# --------------------------------------------------------------------------
# 6. A ramp started inside its zone is judged from where it started (X_t)
# --------------------------------------------------------------------------

def started_ramp(x, keys=None):
    """The lap's ramp stair-2-finish, stepped up to the frame that starts it in its zone at ``x``."""
    nav = lap_nav(RAMP, {**ZONES, **(keys or {})})
    log = script(nav, [(x, stair_y(x))] * 4)
    assert log[-1].state == "ramping" and log[-1].cmd == Command("right") and nav._zone_start_x == x
    return nav


def test_ramp_corridor_starts_one_tolerance_behind_the_take_off_instead_of_behind_the_saved_x():
    nav = started_ramp(76)
    log = script(nav, [(77, 65), (79, 65), (75, 65), (76, 65), (80, 65)], t0=.4)
    assert states(log) == ["ramping"] * 5 and all(f.cmd == Command("right") for f in log)
    assert all(f.committed for f in log) and nav.reason is None
    # Without a zone the same ramp starts at its saved spot and its corridor begins at x=81.
    plain = lap_nav(RAMP)
    script(plain, [(82, 63)] * 4)
    assert script(plain, [(80, 65)], t0=.4)[0].state == "ramp_waiting_knockback"
    # A take-off inside the legacy window is measured from where it stood as well while the zone is open.
    assert script(started_ramp(81), [(80, 65)], t0=.4)[0].state == "ramping"


@pytest.mark.parametrize("back, state", [(1, "ramping"), (2, "ramp_waiting_knockback"), (3, "ramp_waiting_knockback"),
                                         (4, "blocked")])
def test_ramp_knockback_window_and_its_three_pixel_limit_are_measured_from_the_take_off(back, state):
    nav = started_ramp(78)  # 4 px behind the saved spot
    log = script(nav, [(78 - back, 65)], t0=.4)
    assert log[0].state == state and log[0].cmd == (Command("right") if state == "ramping" else Command.NONE)
    assert nav.reason == ("left ramp corridor" if state == "blocked" else None)
    assert nav._ramp_retry_count == (state == "ramp_waiting_knockback")


def test_ramp_knockback_wait_keeps_its_three_pixel_limit_from_the_take_off():
    nav = started_ramp(78)
    log = script(nav, [(76, 65), (76, 65), (74, 65)], t0=.4)  # 2 px back, then 4 px back while waiting
    assert states(log) == ["ramp_waiting_knockback"] * 2 + ["blocked"] and nav.reason == "left ramp corridor"
    # Re-armed after the wait: the next start is a legacy one, judged from the saved spot.
    nav = started_ramp(78)
    log = script(nav, [(76, 65)] * 4 + [(82, 63)] * 4 + [(80, 65)], t0=.4)
    assert states(log) == (["ramp_waiting_knockback"] * 3 + ["ramp_realigning"] + ["ramp_settling"] * 3
                           + ["ramping", "blocked"])
    assert nav.reason == "left ramp corridor" and nav._zone_start_x is None


def test_ramp_from_a_zone_still_needs_the_unchanged_landing_proof():
    climb = [(83, 62), (84, 61), (87, 58), (89, 55)]
    nav = started_ramp(76)
    log = script(nav, climb + [(92, 55)] * 8, t0=.4)
    landing = [f for f in log if f.state == "ramp_landing" and f.idx == RAMP]
    # Strict support on floor-3-right at the saved landing X, three stable frames, the 0.6 s dwell.
    assert states(log)[:4] == ["ramping"] * 4 and len(landing) >= 6 and {f.pos for f in landing} == {(92, 55)}
    assert on_time(first(log, lambda f: f.idx == RAMP + 1).t, landing[0].t + .6) and nav.zone_takeoffs == 1
    # The same climb from the saved spot without a zone: the same states and keys after the start.
    plain = lap_nav(RAMP)
    script(plain, [(82, 63)] * 4)
    reference = script(plain, climb + [(92, 55)] * 8, t0=.4)
    assert [(f.state, f.cmd, f.idx) for f in log] == [(f.state, f.cmd, f.idx) for f in reference]
    # At the landing X but 2 px above the floor, or never still: no arrival.
    for tail in ([(92, 53)] * 30, [(92, 55), (92, 54)] * 15):
        nav = started_ramp(76)
        log = script(nav, climb + tail, t0=.4)
        assert nav.idx == RAMP and nav.reason is None and all(f.cmd == Command.NONE for f in log[4:])


def test_fall_off_the_stair_after_a_zone_start_still_blocks_with_left_ramp_corridor():
    for fallen in ((78, 71), (80, 71), (77, 70), (74, 71)):  # down on floor-2, 5 px under the stair
        nav = started_ramp(76)
        log = script(nav, [(77, 65), fallen, fallen], t0=.4)
        assert states(log) == ["ramping", "blocked", "blocked"] and nav.reason == "left ramp corridor"
        assert log[1].cmd == Command.NONE and nav._ramp_retry_count == 0  # a fall is never a knock-back retry


def test_zone_start_adds_no_time_to_a_ramp():
    pinned = [(79, 65)] * 160  # standing on the stair with the key held
    # Without commit_budget the waypoint clock runs from the selection, wherever the ramp started.
    nav = started_ramp(76)
    log = script(nav, pinned, t0=.4)
    assert nav.reason == "ramp timeout" and on_time(first(log, lambda f: f.state == "blocked").t, 15.0)
    # With it the ramp budget runs from this take-off, as from any other.
    nav = started_ramp(76, A)
    log = script(nav, pinned, t0=.4)
    assert nav.reason == "ramp timeout" and on_time(first(log, lambda f: f.state == "blocked").t, .3 + 15.0)


# --------------------------------------------------------------------------
# 7. A drop launched from its zone: the arrival window after the unchanged landing proof
# --------------------------------------------------------------------------

def dropped(x, keys=None):
    """The lap's drop-4-3, stepped up to its Down+Jump from ``x`` on floor-4 (first attempt)."""
    nav = lap_nav(DROP, {**ZONES, **(keys or {})})
    log = script(nav, [(x, 38)] * 4)
    assert log[-1].cmd == Command("none", "down", "jump") and nav._zone_start_x == x
    return nav


def test_drop_from_a_zone_is_accepted_beside_the_saved_x_only_after_the_unchanged_landing_proof():
    nav = dropped(56)
    log = script(nav, [(56, 43), (56, 49), (56, 53)] + [(56, 55)] * 8, t0=.4)
    landed = first(log, lambda f: f.state == "dwelling")
    assert landed.pos == (56, 55) and states(log)[:3] == ["dropping"] * 3
    # Strictly on floor-3, the height still for the 0.6 s dwell: then accepted 4 px beside x=60.
    assert on_time(first(log, lambda f: f.idx == DROP + 1).t, landed.t + .6)
    assert all(f.cmd == Command.NONE for f in log) and nav.zone_takeoffs == 1
    # Without a zone a landing there is walked back to the saved X, as it always was.
    plain = lap_nav(DROP)
    log = script(plain, [(60, 38)] * 4 + [(59, 43), (57, 49), (56, 53)] + [(56, 55)] * 20)
    assert plain.idx == DROP and [f for f in log if f.state == "dwelling" and f.cmd.move_x == "right"]
    # A landing that never holds still is no landing: nothing is accepted, the old bound ends it.
    nav = dropped(56)
    log = script(nav, [(56, 43), (56, 49), (56, 53)] + [(56, 55), (56, 54)] * 80, t0=.4)
    assert nav.reason == "drop timeout" and on_time(first(log, lambda f: f.state == "blocked").t, 15.0)
    assert nav.idx == DROP and all(f.cmd == Command.NONE for f in log)
    # Two pixels above the destination is not on it: still falling.
    nav = dropped(56)
    log = script(nav, [(56, 43), (56, 49)] + [(56, 53)] * 3, t0=.4)
    assert nav.idx == DROP and {f.state for f in log} == {"dropping"}


@pytest.mark.parametrize("x, accepted", [(50, True), (49, False), (66, True), (67, False)])
def test_drop_arrival_window_is_the_zone_plus_two_pixels_of_drift(x, accepted):
    # drop-4-3: zone 52..64.287, so 50..66.287 on floor-3.
    nav = dropped(56)
    log = script(nav, [(56, 43), (x, 49), (x, 53)] + [(x, 55)] * 16, t0=.4)
    assert (nav.idx == DROP + 1) == accepted
    if not accepted:
        # Outside it the correction toward the saved X runs after the proven landing, as today...
        toward = "right" if x < 60 else "left"
        correcting = [f for f in log if f.cmd.move_x != "none"]
        assert correcting and {f.cmd.move_x for f in correcting} == {toward}
        assert {f.state for f in correcting} == {"dwelling"} and correcting[0].t >= .7 + .6 - 1e-6
        # ...and ends on the first frame inside the window.
        inside = (x + 1, 55) if x < 60 else (x - 1, 55)
        assert script(nav, [inside], t0=3.0)[0].idx == DROP + 1


def test_drop_arrival_window_needs_the_destination_platform_and_a_launch_outside_the_legacy_window():
    nav = dropped(56)
    point = nav.waypoints[DROP]
    floor_3 = nav._containing_platform((point.x, point.y))
    assert nav._zone_landed(point, (56, 55), floor_3)
    assert not nav._zone_landed(point, (56, 55), None)
    assert not nav._zone_landed(point, (56, 55), Platform(55, 57, 88))   # not on the destination's span
    assert not nav._zone_landed(point, (49, 55), floor_3) and not nav._zone_landed(point, (67, 55), floor_3)
    # Launched inside the legacy window: today's rule entirely, key on or off.
    tail = [(61, 43), (62, 49), (63, 53)] + [(63, 55)] * 20 + [(61, 55)] * 2
    logs = []
    for keys in ({}, ZONES):
        nav = lap_nav(DROP, keys)
        log = script(nav, [(61, 38)] * 4 + tail)
        assert log[3].cmd == Command("none", "down", "jump") and nav.idx == DROP + 1
        beside = [f for f in log if f.pos == (63, 55)]
        assert {f.state for f in beside} == {"dwelling"} and [f for f in beside if f.cmd.move_x == "left"]
        assert first(log, lambda f: f.idx == DROP + 1).pos == (61, 55)
        logs.append([(f.state, f.cmd, f.idx) for f in log])
    assert logs[0] == logs[1]


def test_drop_arrival_beside_the_saved_x_needs_its_whole_dwell_on_the_destination_not_one_frame():
    # top-drop (saved x=75, zone 70.04-83) launched from 83. The character comes to rest on stair-3 at
    # (89,39), just past floor-4's end (88.54): the height is stable but it is not on the destination.
    # Then it is displaced to (83,39) on floor-4, inside the arrival window 68.04-85.
    dwell = MinimapCircuit(site()).waypoints[TOP_DROP].dwell_s
    assert dwell == .6
    fall = [(83, 30), (88, 36)] + [(89, 39)] * 20
    nav = lap_nav(TOP_DROP, ZONES)
    log = script(nav, [(83, 22)] * 4 + fall + [(83, 39)] * 9)
    assert log[3].cmd == Command("none", "down", "jump") and log[3].pos == (83, 22) and nav.zone_takeoffs == 1
    beside = [f for f in log if f.pos == (83, 39)]
    assert {f.state for f in log if f.pos == (89, 39)} == {"dropping"}
    # The height alone had been still for 2 s; the destination is proven frame after frame from (83,39) on.
    assert beside[0].idx == TOP_DROP and beside[0].state == "dwelling" and beside[0].cmd == Command.NONE
    assert on_time(first(log, lambda f: f.idx == TOP_DROP + 1).t, beside[0].t + dwell)
    # No correction key while that dwell runs.
    assert all(f.cmd == Command.NONE for f in log[5:] if f.idx == TOP_DROP)
    # Leaving the destination during that dwell starts it again.
    nav = lap_nav(TOP_DROP, ZONES)
    log = script(nav, [(83, 22)] * 4 + fall + [(83, 39)] * 3 + [(89, 39)] + [(83, 39)] * 9)
    back = [f for f in log if f.pos == (83, 39)][3]
    assert on_time(first(log, lambda f: f.idx == TOP_DROP + 1).t, back.t + dwell)
    # A landing straight onto the destination keeps its unchanged proof: the dwell from the landing frame.
    nav = lap_nav(TOP_DROP, ZONES)
    log = script(nav, [(83, 22)] * 4 + [(83, 30), (83, 36)] + [(83, 38)] * 9)
    landed = first(log, lambda f: f.pos == (83, 38))
    assert on_time(first(log, lambda f: f.idx == TOP_DROP + 1).t, landed.t + dwell)
    # Without zones the same displacement is corrected toward the saved X, as before.
    nav = lap_nav(TOP_DROP)
    log = script(nav, [(75, 22)] * 4 + [(75, 30), (88, 36)] + [(89, 39)] * 20 + [(83, 39)] * 4)
    assert nav.idx == TOP_DROP and [f for f in log if f.pos == (83, 39) and f.cmd.move_x == "left"]


# --------------------------------------------------------------------------
# 8. The level-gap jump in the simulated world
# --------------------------------------------------------------------------

@pytest.mark.parametrize("start, pixel", [(89.2, 90), (90, 90), (90.4, 90), (91, 91), (92, 91), (96, 91)])
def test_gap_jump_takes_off_from_pixel_90_or_91_and_proves_its_landing(start, pixel):
    nav, world, log = run(GAP, (start, 55), ZONES)
    key = first(log, lambda f: f.cmd.action == "jump")
    # From the gap side it stops on 90, from the far side on 91 as before; 89 and 92 never launch.
    assert key.pos == (pixel, 55) and key.cmd == Command("left", "none", "jump")
    settle = [f for f in log if key.t - .25 - 1e-9 <= f.t < key.t]
    assert len(settle) >= 2 and all(f.cmd == Command.NONE and f.pos == key.pos for f in settle)
    assert nav.idx == GAP + 1 and world.takeoffs == 1 and nav.retries == 0 and nav.zone_takeoffs == (pixel == 90)
    # The landing proof of the flight: descent seen, strict support on floor-3, three stable frames.
    assert states(log)[-4:-1] == ["jump_landing"] * 3 and world.y == 55 and 84 <= world.x <= 88.475
    taps = [f.cmd.valid_for_s for f in log if f.state == "jump_aligning" and f.cmd.move_x != "none"]
    assert all(lease is not None and lease <= .04 + 1e-9 for lease in taps)  # lined up by fine taps, as before


# --------------------------------------------------------------------------
# 9. Return and rejoin children; editor movement tests
# --------------------------------------------------------------------------

def launches(log):
    return [f for a, f in zip([None] + log, log) if f.cmd.action == "jump" and (a is None or a.cmd.action != "jump")]


def test_return_child_from_the_lap3_spot_drops_where_it_stands_and_still_proves_point_1():
    # (100,55) on floor-3-right: the return of lap 3 in log/claude-circuits-6.jsonl needed x=103 +/- 1 for
    # its first walk waypoint and never left "return_walking".
    nav, world, log = _returning(MinimapCircuit, ZONES, {}, 60)
    assert nav.recovery._zone_kinds == ("drop", "ramp") and nav.recovery.movement_test
    assert [f.pos for f in launches(log)] == [(100, 55), (100, 71)]  # drop-right-3-2, then drop-right-2-1
    floor_1 = first(log, lambda f: f.pos == (100, 88))
    assert all(f.cmd.move_x == "none" for f in log if f.t <= floor_1.t)  # not one walk key before floor-1
    for landing in ((100, 71), (100, 88)):
        proving = [f for f in log if f.pos == landing and f.state == "return_dwelling"]
        assert len(proving) >= 6 and proving[-1].t - proving[0].t >= .6 - 1e-6, landing  # the unchanged dwell
    # Then the walk to the base ladder, the rope, and the unchanged proof of point 1.
    assert [f for f in log if f.state == "return_walking" and f.cmd.move_x == "left"]
    assert [f for f in log if f.state == "return_climbing" and f.cmd.move_y == "down"]
    confirming = [f for f in log if f.state == "return_confirming_start"]
    assert len(confirming) >= 3 and confirming[-1].t - confirming[0].t >= .36
    assert {f.pos for f in confirming} == {(49, 105)}
    assert nav.state == "returned_to_start" and nav.idx == 0 and world.pos == (49, 105) and world.drops == 2
    assert all(f.committed for f in log[:-1])  # a return is never interruptible
    # Both take-offs were outside the saved spot's own tolerance; the parent reports the child's count.
    assert (nav.zone_takeoffs, nav.recovery.zone_takeoffs, nav.return_count) == (2, 2, 1)


def test_same_return_with_the_zones_off_walks_to_the_saved_spot_as_before():
    nav, world, log = _returning(MinimapCircuit, {}, {}, 40)
    assert digest(log) == TIMING_PINNED["return"] and nav.recovery._zone_kinds == ()
    first_drop = launches(log)[0]
    assert abs(first_drop.pos[0] - 103) <= 1 and [f for f in log if f.t < first_drop.t and f.cmd.move_x == "right"]
    assert (nav.zone_takeoffs, nav.recovery.zone_takeoffs) == (0, 0)
    # The key false with the child handed kinds by hand: the key decides, nothing is used.
    child = MinimapCircuit(plan_return_to_start(site(takeoff_zones=False), (100, 55)), zone_kinds=("drop", "ramp"))
    assert child._zone_kinds == ()


def test_lap3_return_under_knockbacks_launches_its_drop_from_inside_the_zone():
    slow = {"startup_s": .02}
    nav, world, log = _to_first_drop({}, slow, events=_hits(1.5))
    assert nav.recovery_terminal and nav.reason == "Return failed: waypoint timeout" and world.drops == 0  # as live
    nav, world, log = _to_first_drop(ZONES, slow, events=_hits(1.5))
    key = log[-1]
    assert _child_launched(nav) and not nav.recovery_terminal and key.cmd == Command("none", "down", "jump")
    assert key.pos[0] in range(95, 110) and key.pos[1] == 55 and key.t < 5
    # The settle before the key: a quarter second on that pixel without a key.
    settle = [f for f in log if key.t - .25 - 1e-9 <= f.t < key.t]
    assert len(settle) >= 2 and all(f.cmd == Command.NONE and f.pos == key.pos for f in settle)
    assert all(f.cmd.move_x == "none" for f in log)  # every hit left it inside the zone: no walk key at all


def test_return_child_from_a_stair_starts_its_ramp_from_the_zone_and_proves_every_landing():
    nav, world, log = _returning(MinimapCircuit, ZONES, {}, 90, start=(76, 65), idx=RAMP - 1)
    assert nav.state == "returned_to_start" and world.pos == (49, 105)
    ramp = first(log, lambda f: f.state == "return_ramping")
    assert ramp.pos == (76, 65) and not [f for f in log if f.t < ramp.t and f.cmd.move_x != "none"]
    assert len([f for f in log if f.state == "return_ramp_landing"]) >= 6
    assert nav.zone_takeoffs == 3 and world.drops == 2  # the ramp and the two right-hand drops


def test_rejoin_child_takes_its_gap_jump_from_the_saved_pixel_only():
    # Waypoint 10 expects floor-2; the character stands on floor-2-right at pixel 90, inside the zone
    # the lap's own floor-2-gap jump would use.
    cfg = site(start_idx=GAP - 7, arrive_on_destination=True, platform_rejoin=True, **ZONES)
    nav, world = MinimapCircuit(cfg), World(cfg, (90, 71))
    log = drive(nav, world, limit_s=10, stop=lambda n: n.rejoining)
    child = nav.platform_rejoin
    assert child is not None and child._zone_kinds == ("drop", "ramp")
    jump = next(i for i, p in enumerate(child._cfg["waypoints"]) if p["action"] == "jump")
    assert child._cfg["waypoints"][jump]["connection_id"] == "floor-2-gap" and child._zone(jump) is None
    log += drive(nav, world, limit_s=30, t0=log[-1].t + .12, stop=lambda n: n.idx == GAP - 5)
    key = first(log, lambda f: f.cmd.action == "jump")
    assert key.pos == (91, 71) and key.state == "rejoin_jumping"
    assert [f for f in log if f.state == "rejoin_jump_aligning" and f.cmd.move_x == "right"]  # lined up, 90 -> 91
    assert nav.rejoin_state == "completed" and world.takeoffs == 1 and nav.zone_takeoffs == 0
    # The lap's own jump one floor higher, from the same pixel: the zone.
    assert first(run(GAP, (90, 55), ZONES)[2], lambda f: f.cmd.action == "jump").pos == (90, 55)


def test_rejoin_child_drops_from_its_zone_and_the_parent_counts_the_take_off():
    # Waypoint 18 expects floor-3; the character stands on floor-4 at x=56, inside the zone of drop-4-3.
    def rejoin(keys):
        cfg = site(start_idx=STAIR_JUMP - 2, platform_rejoin=True, **keys)
        nav, world = MinimapCircuit(cfg), World(cfg, (56, 38))
        log = drive(nav, world, limit_s=30, stop=lambda n: n.rejoin_state in ("completed", "failed"))
        return nav, world, log
    nav, world, log = rejoin(ZONES)
    key = first(log, lambda f: f.cmd.action == "jump")
    assert key.pos == (56, 38) and key.state == "rejoin_dropping" and all(f.cmd.move_x == "none" for f in log)
    proving = [f for f in log if f.pos == (56, 55) and f.state == "rejoin_dwelling"]
    assert len(proving) >= 6 and proving[-1].t - proving[0].t >= .6 - 1e-6  # the child's unchanged landing proof
    assert states(log)[-6:] == ["rejoin_confirming"] * 5 + ["rejoined"]  # and the parent's own proof on floor-3
    assert (nav.rejoin_state, nav.rejoin_count, nav.zone_takeoffs, world.drops) == ("completed", 1, 1, 1)
    assert nav.idx == STAIR_JUMP - 2 and nav.retries == 0
    nav, world, log = rejoin({})  # without the zones the child first walks to the saved spot
    assert first(log, lambda f: f.cmd.action == "jump").pos == (59, 38) and nav.zone_takeoffs == 0
    assert [f for f in log if f.state == "rejoin_walking" and f.cmd.move_x == "right"]


def test_editor_movement_test_never_uses_a_zone():
    for idx, pos, state in ((DROP, (56, 38), "aligning"), (RAMP, (76, 65), "ramp_aligning"),
                            (GAP, (90, 55), "jump_aligning"), (DROP - 1, (64, 38), "walking")):
        logs = []
        for keys in ({}, ZONES, {**ZONES, "takeoff_zone_max_x": 20}):
            nav = MinimapCircuit(site(start_idx=idx, movement_test=True, loop=False, **keys))
            logs.append(script(nav, [pos] * 12))
            assert {f.state for f in logs[-1]} == {state} and nav._zone_kinds == () and nav.zone_takeoffs == 0
        assert digest(logs[0]) == digest(logs[1]) == digest(logs[2])


# --------------------------------------------------------------------------
# 10. Pre-takeoff combat keeps its meaning
# --------------------------------------------------------------------------

@pytest.mark.parametrize("idx, pos, settling", [(GAP, (90, 55), "jump_settling"), (RAMP, (76, 65), "ramp_settling"),
                                                (DROP, (56, 38), "settling")], ids=["gap jump", "ramp", "drop"])
def test_pre_takeoff_combat_is_allowed_while_settling_inside_a_zone_and_refused_once_committed(idx, pos, settling):
    b = brain(site(start_idx=idx, return_on_block=False, **ZONES))
    nav = b.circuit
    for now in (0.0, .1):
        assert b.decide(see(pos, now)) == Command.NONE
    # Settling on a zone pixel the legacy window does not hold: a pre-takeoff phase like any other.
    assert nav.state == settling and nav.transition_active and not nav.transition_committed
    assert nav.takeoff_interruptible(pos)
    cmd = b.decide(see(pos, .2, [mob(60)]))
    assert cmd == Command("none", "none", "attack", face_x="right") and b.state == "hunting"
    # The fight cleared the settle: after it the whole settle is served again, then the take-off.
    seen, now = [], .3
    while not nav.transition_committed and now < 3:
        seen.append((b.decide(see(pos, round(now, 6))), b.state))
        now += .1
    assert nav.transition_committed and nav._zone_start_x == pos[0] and nav.zone_takeoffs == 1
    assert all(cmd == Command.NONE for cmd, _ in seen[:-1])
    assert [state for _, state in seen[-4:-1]] == ["circuit_" + settling] * 3
    # Committed: a monster in range no longer gets a frame.
    cmd = b.decide(see(pos, round(now, 6), [mob(60)]))
    assert cmd.action not in ATTACKS and b.state.startswith("circuit_") and not nav.takeoff_interruptible(pos)


# --------------------------------------------------------------------------
# 11. The whole lap with every opt-in key of this branch on
# --------------------------------------------------------------------------

# States in which the code before the zones already holds a direction key without a lease: a walk
# or alignment from beyond the approach radius, a ramp, a flight, and the stuck-recovery hop.
HOLDS = {"walking", "aligning", "jump_aligning", "ramp_aligning", "jump_arrived_aligning", "dwelling", "ramping",
         "jumping", "airborne", "recovering"}
NAMED = ("waypoint timeout", "jump timeout", "ramp timeout", "drop timeout", "drop stalled", "left ramp corridor",
         "ramp knockback did not settle", "passed destination floor", "drop did not leave source floor",
         "Return route timed out", "Return budget exhausted", "Stable source unavailable",
         "Current platform is ambiguous", "Could not reach a drawn platform")


def bare(state):
    for prefix in ("return_", "rejoin_"):
        if state.startswith(prefix):
            return state[len(prefix):]
    return state


def assert_keys_leased_as_before(log, longest):
    for f in log:
        if f.cmd.move_x == "none":
            continue
        state = bare(f.state)
        assert state not in SETTLING, f  # no direction key while settling, in a zone or on the saved pixel
        if f.cmd.valid_for_s is None:
            assert state in HOLDS, f
        else:
            assert 0 < f.cmd.valid_for_s <= (.25 if state.startswith("regain_") else longest) + 1e-9, f


@pytest.mark.parametrize("model", [{}, SLOW], ids=["plain world", "slow taps and slide"])
def test_undisturbed_lap_with_every_key_completes_and_uses_the_zones(model):
    nav, world, log, wrapped = lap(EVERY, model=model)
    assert wrapped and nav.idx == 0 and nav.reason is None and log[-1].t < 200
    assert (nav.return_count, nav.rejoin_count, nav.arrival_count, nav.retries) == (0, 0, 0, 0)
    assert world.takeoffs == 7 and world.drops == 4  # every jump and drop of the lap exactly once
    # Three ramps and four drops took off away from their saved spot; the gap jumps are reached
    # from the far side and stop on pixel 91, inside the saved tolerance.
    assert nav.zone_takeoffs == 7
    assert_keys_leased_as_before(log, .45)
    reference = lap(OTHER, model=model)
    assert reference[3] and log[-1].t <= reference[2][-1].t  # and the lap is not slower for it


def test_undisturbed_lap_with_the_suggested_live_keys_keeps_every_ramp_on_its_saved_spot():
    nav, world, log, wrapped = lap(SUGGESTED)
    assert wrapped and nav.reason is None and (nav.return_count, nav.rejoin_count, nav.retries) == (0, 0, 0)
    assert world.takeoffs == 7 and world.drops == 4 and nav.zone_takeoffs == 4  # the four drops only
    starts = [f for a, f in zip(log, log[1:]) if f.state == "ramping" and a.state != "ramping"]
    assert len(starts) == 3 and all(abs(f.pos[0] - 82) <= 1 for f in starts)  # the saved spot and its tolerance
    assert_keys_leased_as_before(log, .45)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("model", [{}, SLOW], ids=["plain world", "slow taps and slide"])
@pytest.mark.parametrize("keys", ["every key", "suggested live keys"])
def test_knocked_about_lap_with_every_key_ends_wrapped_or_with_a_named_bounded_failure(seed, model, keys):
    limit = 900.0
    nav, world, log, wrapped = lap(EVERY if keys == "every key" else SUGGESTED, model=model, events=knocks(seed),
                                   limit_s=limit)
    assert log[-1].t < limit - 1  # it never hangs: a wrap, or the terminal state behind guarded HP Hold
    if not wrapped:
        assert nav.recovery_terminal and nav.state == "blocked" and nav.reason.startswith("Return failed: ")
        assert any(name in nav.reason for name in NAMED), nav.reason
        assert nav.step(world.pos, log[-1].t + 1) == Command.NONE and nav.state == "blocked"
    assert nav.return_count <= 1 and nav.rejoin_count <= 3 and nav.retries <= 1
    assert_keys_leased_as_before(log, .45)


def test_lap_with_the_zones_alone_keeps_every_correction_on_fine_taps():
    nav, world, log, wrapped = lap(ZONES)
    assert wrapped and nav.zone_takeoffs == 7 and world.takeoffs == 7 and world.drops == 4
    assert_keys_leased_as_before(log, .04)
    reference = lap({})
    assert reference[3] and log[-1].t < reference[2][-1].t
