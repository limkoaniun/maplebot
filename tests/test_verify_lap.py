from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maplebot.editor_bridge import capture_signature


spec = importlib.util.spec_from_file_location("verify_lap", Path(__file__).parents[1] / "scripts/verify_lap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    platforms = [{"id": "ground", "left": 5, "right": 95, "y": 75},
                 {"id": "upper", "left": 10, "right": 90, "y": 45}]
    points = [{"x": 20, "y": 75, "action": "walk", "platform_id": "ground"},
              {"x": 40, "y": 75, "action": "walk", "platform_id": "ground"},
              {"x": 40, "y": 45, "action": "climb", "to_platform": "upper"},
              {"x": 50, "y": 45, "action": "walk", "platform_id": "upper"},
              {"x": 60, "y": 45, "action": "walk", "platform_id": "upper"},
              {"x": 30, "y": 45, "action": "walk", "platform_id": "upper"},
              {"x": 40, "y": 45, "action": "walk", "platform_id": "upper"},
              {"x": 40, "y": 75, "action": "drop", "to_platform": "ground"},
              {"x": 50, "y": 75, "action": "walk", "platform_id": "ground"},
              {"x": 60, "y": 75, "action": "walk", "platform_id": "ground"}]
    cfg = {"game_window": {}, "minimap": {}, "route": {"circuit": {
        "enabled": True, "loop": True, "timeout_s": 35, "custom_tuning": {"stay": 1},
        "waypoints": points, "platforms": platforms}}}
    plan = {"capture": capture_signature(cfg)}
    position = [20, 75]
    state = {"running": True, "paused": True, "updated": module.time.time(),
             "capture": capture_signature(cfg)}
    bridge = SimpleNamespace(log=tmp_path, _cfg=lambda: cfg,
        load_current=lambda: (plan, np.zeros((98, 111, 3), np.uint8), position),
        _check_context=lambda *args: None)
    monkeypatch.setattr(module, "read_json", lambda path: state)
    monkeypatch.setattr(module, "health", lambda path: 50)
    return bridge, cfg, plan, position, state


def test_prepare_preserves_saved_waypoints_and_tuning_without_writing(prepared):
    bridge, cfg, _, _, _ = prepared
    original = deepcopy(cfg)
    request, reference, hp = module.prepare_lap(bridge)
    assert cfg == original
    assert request["circuit"]["waypoints"] == original["route"]["circuit"]["waypoints"]
    assert request["circuit"]["custom_tuning"] == {"stay": 1}
    assert request["circuit"]["timeout_s"] == 22
    assert request["circuit"]["start_idx"] == 0 and request["circuit"]["max_retries"] == 1
    assert request["circuit"]["movement_test"] and not request["circuit"]["loop"]
    assert request["source_platform"]["id"] == "ground"
    assert reference.shape == (98, 111, 3) and hp == 50
    assert list(bridge.log.iterdir()) == []


def test_chunks_share_walk_anchor_and_preserve_entire_original_order(prepared):
    request, _, _ = module.prepare_lap(prepared[0])
    before = deepcopy(request)
    chunks = module.split_lap(request)
    assert len(chunks) == 2 and request == before
    flattened = chunks[0]["circuit"]["waypoints"][:]
    for previous, chunk in zip(chunks, chunks[1:]):
        assert previous["circuit"]["waypoints"][-1] == chunk["circuit"]["waypoints"][0]
        flattened.extend(chunk["circuit"]["waypoints"][1:])
    assert flattened == request["circuit"]["waypoints"]
    assert all(2 <= len(c["circuit"]["waypoints"]) <= 8 for c in chunks)
    assert all(c["circuit"]["waypoints"][-1]["action"] == "walk" for c in chunks)
    assert chunks[1]["source_platform"]["id"] == "upper"
    assert [c["lap_start_index"] for c in chunks] == [0, 6]
    assert len({c["test_id"] for c in chunks}) == 2


@pytest.mark.parametrize("problem", ["unpaused", "wrong_capture", "wrong_floor", "map_changed", "low_hp", "missing_hp"])
def test_preflight_rejects_bad_context_without_writing(prepared, monkeypatch, problem):
    bridge, _, plan, pos, state = prepared
    if problem == "unpaused":
        state["paused"] = False
    elif problem == "wrong_capture":
        state["capture"] = {}
    elif problem == "wrong_floor":
        pos[1] = 45
    elif problem == "map_changed":
        plan["context_warning"] = "Wrong map"
    elif problem == "low_hp":
        monkeypatch.setattr(module, "health", lambda _: 24)
    else:
        monkeypatch.setattr(module, "health", lambda _: None)
    with pytest.raises(ValueError):
        module.prepare_lap(bridge, min_hp=1)
    assert list(bridge.log.iterdir()) == []


def test_new_chunk_requires_its_actual_source_floor(prepared):
    bridge, _, _, pos, _ = prepared
    request, _, _ = module.prepare_lap(bridge)
    upper_chunk = module.split_lap(request)[1]
    with pytest.raises(ValueError, match="starting platform"):
        module._prepare_chunk(bridge, upper_chunk, 25)
    pos[1] = 45
    reference, hp = module._prepare_chunk(bridge, upper_chunk, 25)
    assert hp == 50 and reference.shape[0] == 98


def test_split_rejects_terminal_transition_without_walk_anchor(prepared):
    request, _, _ = module.prepare_lap(prepared[0])
    request["circuit"]["waypoints"] = request["circuit"]["waypoints"][:8]
    with pytest.raises(ValueError, match="end at a saved walk"):
        module.split_lap(request)


def test_transition_evidence_maps_to_destination_platform(prepared):
    _, cfg, _, _, _ = prepared
    transitions = module.transitions(cfg["route"]["circuit"])
    assert [(t["waypoint_index"], t["action"], t["destination"]["id"]) for t in transitions] == [
        (2, "climb", "upper"), (7, "drop", "ground")]
    assert not any(t["confirmed"] for t in transitions)


def test_on_platform_uses_interpolated_slope_and_strict_support_bounds():
    slope = {"left": 67, "right": 76, "y": 88, "right_y": 79}
    assert module.on_platform([69, 86], slope, .75)
    assert module.on_platform([70, 85], slope, .75)
    assert not module.on_platform([70, 88], slope, .75)
    assert not module.on_platform([76.5, 78.5], slope, .75)


def test_transition_and_chunk_destination_use_slope_height():
    slope = {"id": "plank", "left": 67, "right": 76, "y": 88, "right_y": 79}
    ground = {"id": "floor", "left": 20, "right": 65, "y": 88}
    points = [{"x": 60, "y": 88, "platform_id": "floor"},
              {"x": 70, "y": 85, "action": "jump", "to_platform": "plank"},
              {"x": 71, "y": 84, "action": "walk", "platform_id": "plank"},
              {"x": 76, "y": 79, "action": "ramp", "to_platform": "plank"},
              {"x": 75, "y": 80, "action": "walk", "platform_id": "plank"}]
    circuit = {"platforms": [ground, slope], "waypoints": points}
    landings = module.transitions(circuit)
    assert [v["action"] for v in landings] == ["jump", "ramp"]
    assert all(v["destination"] == slope for v in landings)
    chunks = module.split_lap({"circuit": circuit}, max_points=3)
    assert len(chunks) == 2
    assert chunks[1]["source_platform"] == slope


def test_new_chunk_accepts_only_actual_source_height_on_slope(prepared):
    bridge, _, _, pos, _ = prepared
    request, _, _ = module.prepare_lap(bridge)
    request["source_platform"] = {"id": "plank", "left": 67, "right": 76, "y": 88, "right_y": 79}
    pos[:] = [70, 85]
    module._prepare_chunk(bridge, request, 25)
    pos[:] = [70, 88]
    with pytest.raises(ValueError, match="starting platform"):
        module._prepare_chunk(bridge, request, 25)


def test_interior_knots_preserved_in_lap_and_chunk_sources(prepared):
    bridge, cfg, _, pos, _ = prepared
    circuit = cfg["route"]["circuit"]
    circuit["platforms"][0].update(y=70, right_y=70, knots=[[20, 75], [80, 75]])
    circuit["platforms"][1].update(y=40, right_y=40, knots=[[30, 45], [60, 45]])
    original = deepcopy(circuit)
    request, _, _ = module.prepare_lap(bridge)
    assert request["source_platform"]["knots"] == [[20, 75], [80, 75]]
    assert request["circuit"]["platforms"] == original["platforms"]
    assert circuit == original
    second = module.split_lap(request)[1]
    assert second["source_platform"]["knots"] == [[30, 45], [60, 45]]
    # Neither flat endpoints nor a straight interpolation between them describe
    # the chunk's starting shelf atX40/Y45.
    pos[:] = [40, 45]
    module._prepare_chunk(bridge, second, 25)
    second["source_platform"]["knots"][0][1] = 999
    assert circuit["platforms"][1]["knots"][0][1] == 45


def test_lowest_floor_preflight_includes_interior_valley(prepared):
    bridge, cfg, _, _, _ = prepared
    platforms = cfg["route"]["circuit"]["platforms"]
    platforms[0].update(y=70, right_y=70, knots=[[20, 75], [80, 75]])
    # Another platform's endpoints are higher, but its interior is lower than
    # our nominal ground. Starting on the higher support must still be refused.
    platforms.append({"id": "lower_valley", "left": 0, "right": 4, "y": 60,
                      "right_y": 60, "knots": [[2, 80]]})
    with pytest.raises(ValueError, match="ground platform"):
        module.prepare_lap(bridge)
    assert list(bridge.log.iterdir()) == []


def test_transition_destination_uses_interior_shelf_height():
    surface = {"id": "stair", "left": 67, "right": 90, "y": 88, "right_y": 71,
               "knots": [[73, 82], [80, 82], [84, 77]]}
    result = module.transitions({"platforms": [surface], "waypoints": [
        {"x": 79, "y": 82, "action": "jump", "to_platform": "stair"}]})
    assert result[0]["destination"] == surface
    assert module.on_platform([82, 79.5], surface, .75)
    assert not module.on_platform([82, 76.9], surface, .75)


def test_sloped_preflight_uses_executor_rounding_uncertainty():
    surface = {"left": 80.35, "right": 84.287, "y": 81.85, "right_y": 76.912}
    assert module.on_platform([82, 79], surface, .75)
    assert not module.on_platform([82, 77], surface, .75)
    assert not module.on_platform([84.3, 76.9], surface, .75)
    assert not module.on_platform([75, 83], {"left": 70, "right": 80, "y": 82}, .75)


@pytest.fixture
def resumable(prepared, monkeypatch):
    bridge, cfg, plan, pos, state = prepared
    bridge.user_config = bridge.log / "user.yaml"
    bridge.user_config.write_text("unchanged: saved settings")
    bridge._check_context = lambda *args: (None, {"updated": module.time.time(), "player_pos": pos})
    monkeypatch.setattr(module, "read_json", lambda path: state if path.name == "active-status.json"
                        else json.loads(path.read_text()))
    calls = []
    def run_chunk(bridge, request, min_hp, evidence, deadline_s, pin=None):
        calls.append(request["lap_start_index"])
        evidence.mkdir(parents=True)
        summary = {"result": "completed", "transitions": [], "floors_observed": ["upper"]}
        module.atomic_json(evidence / "summary.json", summary)
        module.atomic_json(evidence / "trace.json", [{"circuit_index": 0}])
        return summary
    monkeypatch.setattr(module, "_run_chunk", run_chunk)
    return bridge, cfg, plan, pos, state, calls


def test_resume_runs_only_selected_chunks_and_labels_partial_result(resumable):
    bridge, _, _, pos, _, calls = resumable
    pos[:] = [40, 45]  # chunk2's saved upper-platform anchor
    result = module.run_lap(bridge, start_chunk=2)
    assert calls == [6]
    assert result["mode"] == "partial lap" and result["result"] == "completed"
    assert result["start_chunk"] == 2 and result["requested_chunks"] == [2]
    assert result["chunk_count"] == 2 and result["requested_chunk_count"] == 1
    assert result["all_requested_chunks_completed"]
    assert not result["full_lap_completed"] and not result["uninterrupted"]
    assert [chunk["chunk"] for chunk in result["chunks"]] == [2]
    directory = next(bridge.log.glob("lap-check-*"))
    assert not (directory / "chunk-01").exists()
    combined = json.loads((directory / "trace.json").read_text())
    assert combined[0]["chunk"] == 2 and combined[0]["global_waypoint_index"] == 6


@pytest.mark.parametrize("start", [0, -1, 3, 1.5, True])
def test_invalid_resume_chunk_is_rejected_before_queue_or_evidence(resumable, start):
    bridge, _, _, _, _, calls = resumable
    before = set(bridge.log.iterdir())
    with pytest.raises(ValueError, match="chunk"):
        module.run_lap(bridge, start_chunk=start)
    assert calls == [] and set(bridge.log.iterdir()) == before


@pytest.mark.parametrize("problem", ["wrong_support", "outside_support", "unpaused", "stale", "wrong_map", "low_hp"])
def test_resume_keeps_source_and_live_context_preflight(resumable, monkeypatch, problem):
    bridge, _, plan, pos, state, calls = resumable
    pos[:] = [40, 45]
    if problem == "wrong_support":
        pos[1] = 75
    elif problem == "outside_support":
        pos[0] = 95
    elif problem == "unpaused":
        state["paused"] = False
    elif problem == "stale":
        state["updated"] -= 20
    elif problem == "wrong_map":
        plan["context_warning"] = "Wrong map"
    else:
        monkeypatch.setattr(module, "health", lambda _: 24)
    before = set(bridge.log.iterdir())
    with pytest.raises((ValueError, RuntimeError)):
        module.run_lap(bridge, start_chunk=2)
    assert calls == [] and set(bridge.log.iterdir()) == before


def test_default_start_still_requires_ground(resumable):
    bridge, _, _, pos, _, calls = resumable
    pos[:] = [40, 45]
    with pytest.raises(ValueError, match="ground platform"):
        module.run_lap(bridge)
    assert calls == []
