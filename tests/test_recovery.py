from copy import deepcopy

import pytest

from maplebot.nav.circuit import MinimapCircuit
from maplebot.nav.editor import compile_plan
from maplebot.nav.recovery import UnsupportedPosition, plan_return_to_start, supported_platform
from maplebot.types import Command


def recovery_config(**overrides):
    plan = {"version": 1, "name": "Return map", "image_size": [120, 100], "loop": False,
            "platforms": [{"id": "ground", "left": 0, "right": 100, "y": 80},
                          {"id": "middle", "left": 20, "right": 90, "y": 60},
                          {"id": "upper", "left": 30, "right": 85, "y": 40}],
            "connections": [
                {"id": "base-rope", "kind": "rope", "from_platform": "ground", "to_platform": "middle",
                 "from_x": 40, "to_x": 40},
                {"id": "up-jump", "kind": "jump", "from_platform": "middle", "to_platform": "upper",
                 "from_x": 60, "to_x": 60},
                {"id": "return-drop", "kind": "drop", "from_platform": "upper", "to_platform": "middle",
                 "from_x": 70, "to_x": 70}],
            "route": [{"platform_id": "ground", "x": 10, "dwell_s": 0},
                      {"platform_id": "upper", "x": 60, "dwell_s": 0}]}
    result = compile_plan(plan, {"editor_plan": plan, "return_on_block": True,
                                 "tolerance_x": 1, "tolerance_y": .75, **overrides})
    result["start_idx"] = len(result["waypoints"]) - 1
    return result


def blocked_nav(**overrides):
    nav = MinimapCircuit(recovery_config(**overrides))
    nav.state, nav.reason = "blocked", "waypoint timeout"
    return nav


def start_child(nav):
    origin = nav.idx
    for now in (0, .3):
        assert nav.step((65, 40), now) == Command.NONE
        assert nav.idx == origin and nav.recovery is None
    assert nav.step((65, 40), .6) == Command.NONE
    assert nav.recovery is not None and nav.returning_to_start
    assert nav.idx == origin and nav.recovery_from_index == origin


def finish_child(nav):
    observations = [((65, 40), .7), ((70, 40), .8), ((70, 40), .9),
                    ((70, 40), 1.2), ((70, 50), 1.3), ((70, 60), 1.4),
                    ((70, 60), 1.7), ((60, 60), 1.8), ((40, 60), 2.0),
                    ((40, 60), 2.1), ((40, 60), 2.4), ((40, 70), 2.6),
                    ((40, 80), 2.7), ((40, 80), 3.1), ((30, 80), 3.2),
                    ((10, 80), 3.3), ((10, 80), 3.6), ((10, 80), 3.95)]
    for pos, now in observations:
        nav.step(pos, now)
    assert nav.recovery.completed and nav.returning_to_start
    assert nav.recovery_state == "confirming_start"
    nav.step((10, 80), 4.2)
    assert nav.returning_to_start
    nav.step((10, 80), 4.5)
    assert not nav.returning_to_start and nav.recovery_state == "completed"


def test_return_planner_uses_actual_floor_and_declared_drop_rope_path_without_mutation():
    cfg = recovery_config()
    before = deepcopy(cfg)
    result = plan_return_to_start(cfg, (65, 40))
    assert cfg == before
    assert result["waypoints"][0]["platform_id"] == "upper"
    assert result["waypoints"][0]["x"] == 65
    transitions = [p for p in result["waypoints"] if p["action"] != "walk"]
    assert [(p["connection_id"], p["action"]) for p in transitions] == [
        ("return-drop", "drop"), ("base-rope", "climb")]
    assert result["waypoints"][-1]["platform_id"] == "ground"
    assert result["waypoints"][-1]["x"] == 10
    assert result["waypoints"][-1]["dwell_s"] >= .6
    assert result["movement_test"] and not result["loop"] and not result["return_on_block"]
    assert result["max_retries"] == 0


def test_physical_return_preserves_original_index_until_stable_point_one_then_restarts():
    nav = blocked_nav()
    original_points, origin = list(nav.waypoints), nav.idx
    start_child(nav)
    assert nav.transition_active and nav.return_count == 1
    finish_child(nav)
    assert nav.waypoints == original_points and nav.idx == 0 and nav.recovery_from_index == origin
    assert nav.return_count == 1 and not nav.recovery_terminal and nav.reason is None
    nav.step((10, 80), 4.6)
    assert nav.idx == 1 and nav.return_count == 1


def test_return_child_preserves_the_precision_pulse_command_deadline():
    nav = blocked_nav(precision_walk=True, approach_radius_x=6)
    start_child(nav)
    assert nav.step((65, 40), .7) == Command.NONE
    command = nav.step((65, 40), .91)
    assert command.move_x == "right" and command.valid_for_s == pytest.approx(.1)
    renewed = nav.step((66, 40), .95)
    assert renewed.move_x == "right" and renewed.valid_for_s == pytest.approx(.06)
    assert nav.returning_to_start and nav.recovery.idx == 1


def test_return_prefers_longer_declared_drop_rope_path_over_jump_shortcut():
    cfg = recovery_config()
    cfg["editor_plan"]["connections"].append({
        "id": "jump-shortcut", "kind": "jump", "from_platform": "upper", "to_platform": "ground",
        "from_x": 65, "to_x": 10,
    })
    before = deepcopy(cfg)
    result = plan_return_to_start(cfg, (65, 40))
    assert cfg == before
    assert [p["connection_id"] for p in result["waypoints"] if p["action"] != "walk"] == [
        "return-drop", "base-rope"]
    assert not any(c["kind"] == "jump" for c in result["editor_plan"]["connections"])


def test_return_preserves_jump_only_path_when_no_nonjump_route_exists():
    cfg = recovery_config()
    cfg["editor_plan"]["connections"] = [
        c for c in cfg["editor_plan"]["connections"] if c["id"] != "return-drop"]
    cfg["editor_plan"]["connections"].append({
        "id": "jump-return", "kind": "jump", "from_platform": "upper", "to_platform": "ground",
        "from_x": 65, "to_x": 10,
    })
    before = deepcopy(cfg)
    result = plan_return_to_start(cfg, (65, 40))
    assert cfg == before
    transitions = [p for p in result["waypoints"] if p["action"] != "walk"]
    assert [(p["connection_id"], p["action"]) for p in transitions] == [("jump-return", "jump")]
    assert result["editor_plan"]["connections"] == cfg["editor_plan"]["connections"]


def test_second_failure_cannot_replenish_return_budget_but_normal_lap_completion_can():
    nav = blocked_nav()
    start_child(nav)
    finish_child(nav)
    nav._set_waypoint(len(nav.waypoints) - 1, 4.6)
    nav.reason, nav.state = "waypoint timeout", "blocked"
    assert nav.step((65, 40), 4.7) == Command.NONE
    assert nav.recovery_terminal and not nav.returning_to_start and nav.return_count == 1
    assert not nav.request_return_to_start()

    normal = blocked_nav()
    start_child(normal)
    finish_child(normal)
    normal._set_waypoint(len(normal.waypoints) - 1, 4.6)
    normal.step((60, 40), 4.7)
    assert normal.completed and normal.return_count == 0


def test_exhausted_return_reports_new_failure_point_and_original_reason():
    nav = blocked_nav()
    start_child(nav)
    assert nav.blocked_origin_reason == "waypoint timeout"
    prior_origin = nav.recovery_from_index
    finish_child(nav)
    nav._set_waypoint(2, 4.6)
    nav.reason, nav.state = "left ramp corridor", "blocked"
    assert nav.idx != prior_origin
    assert not nav.request_return_to_start()
    assert nav.recovery_terminal and nav.return_count == 1
    assert nav.recovery_from_index == 2
    assert nav.blocked_origin_reason == "left ramp corridor"
    assert "Return budget exhausted" in nav.reason and "left ramp corridor" in nav.reason
    assert "point 3" in nav.recovery_reason


@pytest.mark.parametrize("reason", ["unsupported", "ambiguous", "moving"])
def test_return_waits_without_input_then_stops_when_source_never_stabilizes(reason):
    # (65,42) has "middle" drawn below it, so the regain-support stage would walk
    # there; this case pins the legacy wait with the stage switched off.
    cfg = recovery_config(**({"return_regain_support": False} if reason == "unsupported" else {}))
    if reason == "ambiguous":
        cfg["editor_plan"]["platforms"].append({"id": "overlap", "left": 30, "right": 85, "y": 40})
    nav = MinimapCircuit(cfg)
    assert nav.request_return_to_start()
    for i, now in enumerate((0, .6, 1.2, 1.8, 2.4, 3.01)):
        pos = (65, 42) if reason == "unsupported" else (65 + (i % 2 if reason == "moving" else 0), 40)
        assert nav.step(pos, now) == Command.NONE
    assert nav.recovery_terminal and nav.recovery is None and nav.return_count == 1
    assert not nav.request_return_to_start()


def test_return_never_invents_reverse_jump_when_saved_graph_has_no_path():
    cfg = recovery_config()
    cfg["editor_plan"]["connections"] = cfg["editor_plan"]["connections"][:2]
    with pytest.raises(ValueError, match="No connection"):
        plan_return_to_start(cfg, (65, 40))
    nav = MinimapCircuit(cfg)
    nav.request_return_to_start()
    for now in (0, .3, .6):
        assert nav.step((65, 40), now) == Command.NONE
    assert nav.recovery_terminal and nav.recovery is None


def test_return_source_pause_resets_stability_and_does_not_consume_active_timeout():
    nav = blocked_nav()
    nav.step((65, 40), 0)
    nav.step((65, 40), .3)
    nav.suspend(.4)
    nav.step((65, 40), 100)
    nav.step((65, 40), 100.3)
    assert nav.recovery is None and not nav.recovery_terminal
    nav.step((65, 40), 100.6)
    assert nav.recovery is not None
    before = nav.recovery._clock.time
    nav.suspend(100.7)
    nav.step((65, 40), 200)
    assert nav.recovery._clock.time == before and not nav.recovery_terminal


def test_failed_child_or_whole_return_timeout_is_terminal_without_new_attempt():
    nav = blocked_nav()
    start_child(nav)
    nav.recovery.reason = "waypoint timeout"
    nav.step((65, 40), .7)
    assert nav.recovery_terminal and "waypoint timeout" in nav.recovery_reason
    assert nav.step((65, 40), .9) == Command.NONE and nav.return_count == 1
    timeout = blocked_nav(return_timeout_s=1)
    start_child(timeout)
    timeout.step((65, 40), .9)
    assert timeout.step((65, 40), 1.1) == Command.NONE
    assert timeout.recovery_terminal and "timed out" in timeout.recovery_reason


def test_explicit_return_works_when_automatic_option_is_off_and_abort_preserves_origin():
    nav = blocked_nav(return_on_block=False)
    assert nav.step((65, 40), 0) == Command.NONE and not nav.returning_to_start
    origin = nav.idx
    assert nav.request_return_to_start()
    assert not nav.request_return_to_start() and nav.return_count == 1
    nav.abort_return_to_start("Map changed")
    assert nav.step((65, 40), .3) == Command.NONE and nav.recovery_terminal
    assert nav.idx == origin and nav.recovery_from_index == origin and nav.return_count == 1


def test_movement_verification_never_recovers_automatically_or_manually():
    nav = blocked_nav(movement_test=True)
    for now in (0, .3, .6):
        assert nav.step((65, 40), now) == Command.NONE
    assert nav.state == "blocked" and nav.return_count == 0 and nav.recovery is None
    assert not nav.request_return_to_start()


def test_source_matching_keeps_strict_horizontal_bounds_and_quantized_slope_support():
    cfg = recovery_config()
    cfg["editor_plan"]["platforms"].append({"id": "slope", "left": 90, "right": 110, "y": 50, "right_y": 30})
    assert supported_platform(cfg, (100, 39))["id"] == "slope"
    with pytest.raises(ValueError, match="not supported"):
        supported_platform(cfg, (111, 30))
    with pytest.raises(ValueError, match="not supported"):
        supported_platform(cfg, (100, 37))
    with pytest.raises(UnsupportedPosition):
        supported_platform(cfg, (100, 37))
    cfg["editor_plan"]["platforms"].append({"id": "overlap", "left": 30, "right": 85, "y": 40})
    with pytest.raises(ValueError, match="ambiguous") as ambiguous:
        supported_platform(cfg, (65, 40))
    assert not isinstance(ambiguous.value, UnsupportedPosition)


@pytest.mark.parametrize("value", [0, 4, 1.5, True])
def test_return_budget_validation(value):
    with pytest.raises(ValueError, match="return budget"):
        MinimapCircuit(recovery_config(max_returns_per_lap=value))


@pytest.mark.parametrize("value", [-1, 101, float("nan"), float("inf")])
def test_return_health_threshold_validation(value):
    with pytest.raises(ValueError, match="return_min_hp_percent"):
        MinimapCircuit(recovery_config(return_min_hp_percent=value))


def _legacy_plan_return_to_start(cfg, pos):
    """The planner as it was before ``_plan_route`` was extracted (reference copy)."""
    from maplebot.nav.circuit import Platform
    from maplebot.nav.editor import validate_plan
    from maplebot.nav.recovery import _circuit_config

    cfg = _circuit_config(cfg)
    source = supported_platform(cfg, pos)
    points = cfg.get("waypoints") or []
    if not points:
        raise ValueError("Return needs an original first waypoint")
    first = points[0]
    plan = deepcopy(cfg["editor_plan"])
    target_id = first.get("platform_id")
    if target_id is None:
        target_id = supported_platform(cfg, (first["x"], first["y"]))["id"]
    target = next((p for p in plan["platforms"] if p["id"] == target_id), None)
    if target is None:
        raise ValueError("Point 1 refers to a missing platform")
    surface = Platform(target["y"], target["left"], target["right"],
                       target.get("right_y"), target.get("knots", ()))
    if (not surface.left <= first["x"] <= surface.right
            or abs(surface.y_at(first["x"]) - first["y"]) > .75):
        raise ValueError("Point 1 is not on its saved platform")
    plan.update(loop=False, route=[
        {"platform_id": source["id"], "x": pos[0], "dwell_s": 0},
        {"platform_id": target_id, "x": first["x"], "dwell_s": max(.6, first.get("dwell_s", 0))},
    ])
    without_jumps = deepcopy(plan)
    without_jumps["connections"] = [c for c in without_jumps["connections"] if c["kind"] != "jump"]
    if not validate_plan(without_jumps):
        plan = without_jumps
    result = compile_plan(plan, cfg)
    result.update(editor_plan=plan, movement_test=True, return_on_block=False, max_retries=0,
                  tolerance_y=min(.75, float(cfg.get("tolerance_y", 2))))
    return result


def _outcome(planner, cfg, pos):
    try:
        return "ok", planner(cfg, pos)
    except (ValueError, KeyError) as exc:
        return type(exc).__name__, str(exc)


def _planner_variants():
    jump_only = recovery_config()
    jump_only["editor_plan"]["connections"] = [
        c for c in jump_only["editor_plan"]["connections"] if c["id"] != "return-drop"]
    jump_only["editor_plan"]["connections"].append({
        "id": "jump-return", "kind": "jump", "from_platform": "upper", "to_platform": "ground",
        "from_x": 65, "to_x": 10})
    severed = recovery_config()
    severed["editor_plan"]["connections"] = severed["editor_plan"]["connections"][:2]
    no_id = recovery_config()
    del no_id["waypoints"][0]["platform_id"]
    missing = recovery_config()
    missing["waypoints"][0]["platform_id"] = "gone"
    off = recovery_config()
    off["waypoints"][0]["y"] = 70
    empty = recovery_config()
    empty["waypoints"] = []
    no_plan = recovery_config()
    del no_plan["editor_plan"]
    return [recovery_config(), {"route": {"circuit": recovery_config(tolerance_y=.5)}}, jump_only, severed,
            no_id, missing, off, empty, no_plan]


@pytest.mark.parametrize("pos", [(65, 40), (50, 60), (10, 80), (65, 42), (200, 40)])
def test_extracted_route_planner_keeps_return_plans_and_messages_identical(pos):
    outcomes = set()
    for cfg in _planner_variants():
        before = deepcopy(cfg)
        new, old = _outcome(plan_return_to_start, cfg, pos), _outcome(_legacy_plan_return_to_start, cfg, pos)
        assert new == old and cfg == before
        outcomes.add(new[0] if new[0] != "ok" else "ok")
    assert "ValueError" in outcomes or "UnsupportedPosition" in outcomes  # error paths were compared too


def test_rejoin_planner_targets_the_waypoint_platform_with_both_keys_off():
    from maplebot.nav.recovery import plan_rejoin

    cfg = recovery_config(arrive_on_destination=True, platform_rejoin=True)
    before = deepcopy(cfg)
    points = cfg["waypoints"]
    jump = next(i for i, p in enumerate(points) if p["action"] == "jump")
    result = plan_rejoin(cfg, (65, 40), jump, 2)  # on "upper": back to the takeoff on "middle"
    assert cfg == before
    assert [p["connection_id"] for p in result["waypoints"] if p["action"] != "walk"] == ["return-drop"]
    assert (result["waypoints"][-1]["platform_id"], result["waypoints"][-1]["x"]) == ("middle", 60)
    assert result["waypoints"][-1]["dwell_s"] == 0 and result["movement_test"] and result["max_retries"] == 0
    assert result["arrive_on_destination"] is False and result["platform_rejoin"] is False
    assert any(c["kind"] == "jump" for c in result["editor_plan"]["connections"])  # graph kept whole
    with pytest.raises(ValueError, match="too long"):
        plan_rejoin(cfg, (65, 40), 0, 1)  # point 1 on "ground" needs the drop and the rope
    assert len(plan_rejoin(cfg, (65, 40), 0, 2)["waypoints"]) >= 4
    with pytest.raises(ValueError, match="Already on"):
        plan_rejoin(cfg, (50, 60), jump, 2)
    climb = next(i for i, p in enumerate(points) if p["action"] == "climb")
    with pytest.raises(ValueError, match="does not support climb"):
        plan_rejoin(cfg, (65, 40), climb, 2)
    with pytest.raises(UnsupportedPosition):
        plan_rejoin(cfg, (65, 42), jump, 2)
