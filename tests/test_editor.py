from copy import deepcopy

import pytest

from maplebot.nav.editor import (
    canvas_to_image, compile_plan, fit_transform, image_to_canvas,
    plan_from_circuit, snap_to_platform, validate_plan,
    platform_y,
)


def plan(platforms, connections=(), route=(), loop=False):
    return {"version": 1, "name": "Test map", "image_size": [111, 98],
            "platforms": platforms, "connections": list(connections),
            "route": [{"platform_id": pid, "x": x} for pid, x in route], "loop": loop}


def platform(ident, left, right, y):
    return {"id": ident, "left": left, "right": right, "y": y}


def staircase():
    return dict(platform("stairs", 66.79, 89.60, 87.71), right_y=71.33,
                knots=[[70.60, 84.58], [72.41, 81.90], [80.35, 81.96],
                       [84.29, 77.02], [86.60, 75.40]])


def test_piecewise_staircase_compiles_observed_shelf_and_connection_heights():
    stairs = staircase()
    p = plan([platform("low", 10, 66, 88), stairs, platform("high", 90, 108, 71)],
             [connection("onto", "jump", "low", "stairs", 65, 69),
              connection("off", "jump", "stairs", "high", 82, 93)],
             [("low", 60), ("stairs", 79), ("high", 100)])
    assert platform_y(stairs, 79) == pytest.approx(81.9498, abs=.001)
    assert platform_y(stairs, 82) == pytest.approx(79.8912, abs=.001)
    result = compile_plan(p)
    assert result["platforms"][1]["knots"] == stairs["knots"]
    assert next(w for w in result["waypoints"] if w["action"] == "walk" and w["x"] == 79)["y"] == pytest.approx(81.9498, abs=.001)
    assert next(w for w in result["waypoints"] if w.get("connection_id") == "off")["from_y"] == pytest.approx(79.8912, abs=.001)
    snapped = snap_to_platform(p, 79, 82, max_distance=.1)
    assert snapped[0] == "stairs" and snapped[1:] == pytest.approx((79, 81.95), abs=.001)


@pytest.mark.parametrize("knots", [None, "bad", [[10]], [[10, float("nan")]], [[True, 50]],
                                  [[0, 40]], [[70, 40]], [[40, 40], [30, 30]],
                                  [[30, 40], [30, 20]], [[30, -1]], [[30, 98]]])
def test_piecewise_knots_reject_invalid_interior_geometry(knots):
    p = plan([dict(platform("floor", 0, 70, 50), knots=knots)], route=[("floor", 10), ("floor", 60)])
    assert any("knots" in error for error in validate_plan(p))
    if knots not in ([[30, -1]], [[30, 98]]):
        assert snap_to_platform(p, 30, 40) is None


def test_piecewise_path_distance_counts_surface_and_migration_keeps_vertices():
    from maplebot.nav.editor import _surface_distance
    stairs = staircase()
    distance = _surface_distance(stairs, 70, 85)
    assert distance > 18 and distance == _surface_distance(stairs, 85, 70)
    source = {"loop": False, "platforms": [stairs], "waypoints": [{"x": 70, "y": platform_y(stairs, 70)},
                                                               {"x": 82, "y": platform_y(stairs, 82)}]}
    result = plan_from_circuit(source, [111, 98])
    assert result["platforms"][0]["knots"] == stairs["knots"]
    assert [w["y"] for w in compile_plan(result)["waypoints"]] == [point["y"] for point in source["waypoints"]]


def test_compiled_piecewise_geometry_matches_runtime_surface():
    from maplebot.nav.circuit import MinimapCircuit
    stairs = staircase()
    p = plan([stairs], route=[("stairs", 70), ("stairs", 82)])
    runtime = MinimapCircuit(compile_plan(p)).platforms[0]
    for x in (66.79, 69, 70.60, 72.41, 74, 79, 80.35, 82, 84.29, 86.60, 89.60):
        assert runtime.y_at(x) == pytest.approx(platform_y(stairs, x))


def test_piecewise_migration_expansion_preserves_original_endpoints_and_height():
    shaped = dict(platform("floor", 10, 70, 60), knots=[[30, 30], [50, 50]])
    source = {"loop": False, "platforms": [shaped], "waypoints": [{"x": 5, "y": 67.5}, {"x": 75, "y": 62.5}]}
    result = plan_from_circuit(source, [111, 98])
    expanded = result["platforms"][0]
    assert expanded["knots"] == [[10., 60.], [30., 30.], [50., 50.], [70., 60.]]
    assert (expanded["left"], expanded["y"], expanded["right"], expanded["right_y"]) == (5, 67.5, 75, 62.5)


def test_sloped_platform_interpolates_walk_and_jump_endpoint_heights():
    p = plan([platform("low", 5, 60, 88),
              dict(platform("slope", 67, 84, 88), right_y=71),
              platform("upper", 88, 105, 71)],
             [connection("onto", "jump", "low", "slope", 59, 69),
              connection("off", "jump", "slope", "upper", 83, 89)],
             [("low", 55), ("slope", 70), ("slope", 83), ("upper", 95)])
    output = compile_plan(p)
    jumps = [w for w in output["waypoints"] if w["action"] == "jump"]
    assert (jumps[0]["x"], jumps[0]["y"]) == (69, 86)
    assert (jumps[1]["from_x"], jumps[1]["from_y"]) == (83, 72)
    assert any(w["action"] == "walk" and (w["x"], w["y"]) == (70, 85) for w in output["waypoints"])
    assert output["platforms"][1]["right_y"] == 71


def test_slope_snap_projects_onto_segment_and_height_extrapolation_is_explicit():
    slope = dict(platform("slope", 67, 84, 88), right_y=71)
    assert platform_y(slope, 69) == 86 and platform_y(slope, 70) == 85
    assert platform_y(slope, 65) == 90
    assert platform_y(platform("flat", 0, 90, 60), 40) == 60
    snapped = snap_to_platform(plan([slope]), 70, 87)
    assert snapped[0] == "slope" and snapped[1:] == pytest.approx((69, 86))
    assert snap_to_platform(plan([slope]), 60, 90, max_distance=5) is None


@pytest.mark.parametrize("right_y", [-1, 98, float("nan"), True])
def test_invalid_sloped_endpoint_is_rejected(right_y):
    p = plan([dict(platform("slope", 10, 80, 70), right_y=right_y)], route=[("slope", 20), ("slope", 60)])
    assert validate_plan(p)


def test_vertical_connections_use_local_height_not_left_endpoint_height():
    p = plan([dict(platform("slope", 0, 80, 80), right_y=0), platform("lower", 0, 80, 40)],
             [connection("down", "drop", "slope", "lower", 60)], [("slope", 60), ("lower", 60)])
    assert validate_plan(p) == []
    output = compile_plan(p)
    assert output["waypoints"][0]["y"] == 20
    assert next(w for w in output["waypoints"] if w["action"] == "drop")["y"] == 40


def test_migration_keeps_sloped_boundaries_and_walk_heights():
    slope = dict(platform("slope", 67, 84, 88), right_y=71)
    source = {"loop": False, "platforms": [slope], "waypoints": [{"x": 69, "y": 86}, {"x": 83, "y": 72}]}
    migrated = plan_from_circuit(source, [111, 98])
    assert migrated["platforms"][0]["right_y"] == 71
    assert len(migrated["platforms"]) == 1
    assert [w["y"] for w in compile_plan(migrated)["waypoints"]] == [86, 72]


def connection(ident, kind, source, target, fx, tx=None):
    return {"id": ident, "kind": kind, "from_platform": source, "to_platform": target,
            "from_x": fx, "to_x": fx if tx is None else tx}


def test_disconnected_same_height_platforms_cannot_be_walked_between():
    p = plan([platform("left", 0, 30, 50), platform("right", 40, 80, 50)],
             route=[("left", 20), ("right", 60)])
    assert "No connection from route stop 1" in validate_plan(p)[0]
    with pytest.raises(ValueError, match="Draw a rope, jump, drop, or ramp"):
        compile_plan(p)
    p["connections"].append(connection("gap", "jump", "left", "right", 28, 43))
    result = compile_plan(p)
    jump = next(w for w in result["waypoints"] if w["action"] == "jump")
    assert (jump["from_x"], jump["from_y"], jump["x"], jump["y"]) == (28, 50, 43, 50)
    assert jump["connection_id"] == "gap"


def test_directed_jump_requires_a_return_connection_for_loop():
    p = plan([platform("a", 0, 20, 70), platform("b", 35, 60, 50)],
             [connection("out", "jump", "a", "b", 19, 36)], [("a", 10), ("b", 50)], loop=True)
    assert "stop 2 (b) to stop 1 (a)" in validate_plan(p)[0]
    p["connections"].append(connection("back", "jump", "b", "a", 36, 19))
    assert not validate_plan(p)
    assert [w["connection_id"] for w in compile_plan(p)["waypoints"] if w["action"] == "jump"] == ["out", "back"]


def test_rope_is_bidirectional_and_preserves_tuning_without_mutating_inputs():
    p = plan([platform("low", 0, 80, 76), platform("high", 20, 70, 61)],
             [connection("rope", "rope", "low", "high", 40)], [("high", 60), ("low", 10)], loop=True)
    base = {"timeout_s": 22, "align_tolerance_x": .4, "waypoints": [{"old": True}],
            "platforms": [], "enabled": False, "start_idx": 12}
    original, old_base = deepcopy(p), deepcopy(base)
    output = compile_plan(p, base)
    climbs = [w for w in output["waypoints"] if w["action"] == "climb"]
    assert [(w["from_platform"], w["to_platform"], w["y"]) for w in climbs] == [
        ("high", "low", 76), ("low", "high", 61)]
    assert output["timeout_s"] == 22 and output["align_tolerance_x"] == .4
    assert output["enabled"] and output["start_idx"] == 0 and output["loop"]
    assert p == original and base == old_base


def test_multihop_route_emits_takeoffs_and_landing_coordinates():
    p = plan([platform("a", 0, 30, 76), platform("b", 0, 30, 46),
              platform("c", 40, 80, 46), platform("d", 40, 80, 76)],
             [connection("up", "rope", "a", "b", 20),
              connection("across", "jump", "b", "c", 28, 42),
              connection("down", "drop", "c", "d", 60)], [("a", 5), ("d", 75)])
    output = compile_plan(p)
    assert [w["action"] for w in output["waypoints"]] == [
        "walk", "walk", "climb", "walk", "jump", "walk", "drop", "walk"]
    assert output["waypoints"][-1]["x"] == 75
    assert output["loop"] is False


def test_connection_choice_accounts_for_walking_to_and_from_connectors():
    p = plan([platform("top", 30, 80, 16), platform("lower", 20, 90, 31)],
             [connection("far-rope", "rope", "lower", "top", 40),
              connection("near-drop", "drop", "top", "lower", 78)],
             [("top", 74), ("lower", 78)])
    transitions = [w for w in compile_plan(p)["waypoints"] if w["action"] != "walk"]
    assert [w["connection_id"] for w in transitions] == ["near-drop"]


def test_route_stop_dwell_and_coincident_stops_remain_executable():
    p = plan([platform("floor", 0, 70, 50)], route=[("floor", 20), ("floor", 20)])
    p["route"][1]["dwell_s"] = 1.2
    output = compile_plan(p)
    assert len(output["waypoints"]) == 2
    assert output["waypoints"][0]["dwell_s"] == 1.2


@pytest.mark.parametrize("mutate, expected", [
    (lambda p: p["platforms"][0].update(left=float("nan")), "finite"),
    (lambda p: p["platforms"][0].update(right=111), "inside"),
    (lambda p: p["connections"][0].update(to_platform="missing"), "missing platform"),
    (lambda p: p["connections"][0].update(to_x=25), "same X"),
    (lambda p: p["connections"][0].update(kind="drop"), "lower platform"),
    (lambda p: p["route"][0].update(x=-1), "lie on its platform"),
])
def test_invalid_geometry_is_rejected(mutate, expected):
    p = plan([platform("a", 0, 50, 70), platform("b", 0, 50, 40)],
             [connection("c", "rope", "a", "b", 20)], [("a", 10), ("b", 10)])
    mutate(p)
    assert any(expected in e for e in validate_plan(p))


def test_drafts_can_be_empty_or_disconnected_but_invalid_shapes_are_reported():
    p = plan([])
    p["capture_metadata"] = {"title": "Game"}
    assert validate_plan(p, require_route=False) == []
    assert validate_plan(p)
    p = plan([platform("a", 0, 20, 40), platform("b", 40, 70, 40)], route=[("a", 5), ("b", 50)])
    assert validate_plan(p, require_route=False) == []
    assert validate_plan(p)
    p["connections"] = [connection("unfinished", "jump", "a", "missing", 10, 50)]
    assert validate_plan(p, require_route=False)


def test_migration_represents_existing_five_floor_circuit_and_vertical_descent():
    ys = [76, 61, 46, 31, 16]
    bounds = [(4, 84), (24, 90), (30, 84), (35, 79), (40, 74)]
    points = [{"x": 78, "y": 76, "action": "drop", "dwell_s": 0}]
    for i, (y, (left, right)) in enumerate(zip(ys, bounds)):
        if i:
            points.append({"x": [40, 63, 51, 66][i - 1], "y": y, "action": "climb", "dwell_s": .35})
        points.extend({"x": x, "y": y, "action": "walk", "dwell_s": .3} for x in (left, right))
        if i < 4:
            points.append({"x": [40, 63, 51, 66][i], "y": y, "action": "walk", "dwell_s": .3})
    points.extend({"x": 78, "y": y, "action": "drop", "dwell_s": 0} for y in (31, 46, 61))
    circuit = {"platforms": [{"y": y, "left": b[0], "right": b[1]} for y, b in zip(ys, bounds)],
               "waypoints": points, "timeout_s": 22}
    assert len(points) == 22
    before = deepcopy(circuit)
    p = plan_from_circuit(circuit, [111, 98], "Henesys Hunting Ground II")
    assert len(p["platforms"]) == 5 and len(p["route"]) == 22
    assert len(p["connections"]) == 8
    assert p["platforms"][-1]["right"] == 78
    assert validate_plan(p) == []
    result = compile_plan(p, circuit)
    assert [w["action"] for w in result["waypoints"] if w["action"] != "walk"] == [
        "climb", "climb", "climb", "climb", "drop", "drop", "drop", "drop"]
    assert circuit == before


def test_migration_preserves_explicit_same_height_jump():
    circuit = {"loop": False, "platforms": [platform("a", 0, 20, 50), platform("b", 40, 70, 50)],
               "waypoints": [{"x": 10, "y": 50, "action": "walk"},
                             {"x": 45, "y": 50, "action": "jump", "from_x": 18, "from_y": 50}]}
    p = plan_from_circuit(circuit, [111, 98])
    assert validate_plan(p) == []
    assert compile_plan(p)["waypoints"][-2]["action"] == "jump"


def test_snap_uses_segment_distance_not_only_height():
    p = plan([platform("a", 0, 20, 50), platform("b", 40, 70, 50)])
    assert snap_to_platform(p, 42, 53) == ("b", 42, 50)
    assert snap_to_platform(p, 23, 50) == ("a", 20, 50)
    assert snap_to_platform(p, 30, 50, max_distance=5) is None


def test_zero_distance_jump_between_touching_segments_is_not_executable():
    p = plan([platform("a", 0, 20, 50), platform("b", 20, 40, 50)],
             [connection("zero", "jump", "a", "b", 20, 20)], [("a", 10), ("b", 30)])
    assert any("distinct takeoff" in error for error in validate_plan(p))


def test_letterboxed_coordinate_conversion_is_reversible_and_clamps_padding():
    image, viewport = [111, 98], [800, 400]
    scale, ox, oy = fit_transform(image, viewport, 20)
    assert scale == pytest.approx(360 / 98)
    assert ox > 20 and oy == pytest.approx(20)
    point = (63.4, 45.8)
    drawn = image_to_canvas(point, image, viewport, 20)
    assert canvas_to_image(drawn, image, viewport, 20) == pytest.approx(point)
    assert canvas_to_image((-10, -10), image, viewport, 20) == (0, 0)
    assert canvas_to_image((1000, 1000), image, viewport, 20) == (110, 97)
    with pytest.raises(ValueError):
        fit_transform(image, viewport, padding=201)


def test_ramp_is_bidirectional_and_compiles_explicit_slope_endpoints():
    p = plan([platform("low", 5, 65, 80), platform("high", 75, 100, 63)],
             [connection("slope", "ramp", "low", "high", 63, 80)],
             [("high", 90), ("low", 55)], loop=True)
    output = compile_plan(p, {"ramp_corridor_y": 4.5})
    slopes = [w for w in output["waypoints"] if w["action"] == "ramp"]
    assert [(w["from_x"], w["from_y"], w["x"], w["y"]) for w in slopes] == [
        (80, 63, 63, 80), (63, 80, 80, 63)]
    assert [w["connection_id"] for w in slopes] == ["slope", "slope"]
    assert output["ramp_corridor_y"] == 4.5
    from maplebot.nav.circuit import MinimapCircuit
    assert len(MinimapCircuit(output).waypoints) == len(output["waypoints"])


@pytest.mark.parametrize("from_x,to_x,high_y", [(63, 63, 63), (63, 80, 80)])
def test_ramps_reject_vertical_or_flat_connections(from_x, to_x, high_y):
    p = plan([platform("low", 5, 90, 80), platform("high", 5, 100, high_y)],
             [connection("slope", "ramp", "low", "high", from_x, to_x)],
             [("low", 55), ("high", 90)])
    assert any("ramp needs different X" in error for error in validate_plan(p))


def test_migration_preserves_ramp_and_deduplicates_reverse_connection():
    circuit = {"loop": True,
               "platforms": [platform("low", 0, 65, 80), platform("high", 75, 100, 63)],
               "waypoints": [{"x": 63, "y": 80},
                             {"x": 80, "y": 63, "action": "ramp", "from_x": 63, "from_y": 80},
                             {"x": 63, "y": 80, "action": "ramp", "from_x": 80, "from_y": 63}]}
    p = plan_from_circuit(circuit, [111, 98])
    assert validate_plan(p) == []
    assert len(p["connections"]) == 1 and p["connections"][0]["kind"] == "ramp"
    assert len([w for w in compile_plan(p)["waypoints"] if w["action"] == "ramp"]) == 2
