from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from maplebot.editor_bridge import capture_signature


spec = importlib.util.spec_from_file_location("verify_route_checks", Path(__file__).parents[1] / "scripts/verify_route.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def environment(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(module.time, "time", lambda: now[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    cfg = {"game_window": {"working_size": [1366, 768]}, "minimap": {"fixed_rect": [20, 82, 210, 128]},
           "route": {"circuit": {"editor_profile": "current-map"}}}
    plan = {"capture": capture_signature(cfg), "name": "Current map",
            "platforms": [{"id": "source", "left": 10, "right": 90, "y": 75},
                          {"id": "target", "left": 10, "right": 90, "y": 45}],
            "connections": [{"id": "rope", "kind": "rope", "from_platform": "source",
                             "to_platform": "target", "from_x": 40, "to_x": 40}]}
    config_path = tmp_path / "user.yaml"
    config_path.write_text("settings: original")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "current-map.json").write_text("{}")
    (profiles / "current-map.png").write_bytes(b"reference bytes")
    log = tmp_path / "log"
    log.mkdir()
    state = {"running": True, "paused": True, "capture": capture_signature(cfg),
             "updated": 1000, "minimap": [44, 45], "minimap_seen": True}
    changing = {"wrong_map": False, "queued": 0, "stopped": 0, "test_state": "running"}

    def context(*args):
        if changing["wrong_map"]:
            raise ValueError("The current minimap differs from this profile.")
        return np.zeros((128, 210, 3), np.uint8), {"updated": now[0], "player_pos": [44, 45]}

    def queue(*args):
        changing["queued"] += 1

    def stop():
        changing["stopped"] += 1

    bridge = SimpleNamespace(user_config=config_path, profiles=profiles, log=log,
        _cfg=lambda: deepcopy(cfg), load_current=lambda: (plan, np.zeros((128, 210, 3), np.uint8), (20, 75)),
        _check_context=context, test_connection=queue,
        test_status=lambda: {"test_id": "test-one", "state": changing["test_state"]}, stop_test=stop)
    monkeypatch.setattr(module, "read_json", lambda path: {**state, "updated": now[0]})
    monkeypatch.setattr(module, "health", lambda log: 50)
    return bridge, plan, state, changing, now


def test_pinned_context_allows_duplicate_fresh_observations(environment):
    bridge, _, state, _, now = environment
    pin = module.PinnedContext(bridge)
    pin.check(state)
    now[0] += .5
    pin.check(state)  # the same recent status is not a capture failure


@pytest.mark.parametrize("change", ["config", "profile", "map", "capture", "guard"])
def test_pinned_context_rejects_changed_profile_settings_or_map(environment, change):
    bridge, _, state, changing, _ = environment
    pin = module.PinnedContext(bridge)
    if change == "config":
        bridge.user_config.write_text("settings: changed")
    elif change == "profile":
        (bridge.profiles / "current-map.json").write_text('{"new":true}')
    elif change == "map":
        changing["wrong_map"] = True
    elif change == "capture":
        state["capture"] = {}
    else:
        state.update(map_state="unknown", map_ready=False, map_reason="new map")
    with pytest.raises((RuntimeError, ValueError)):
        pin.check(state)


def test_context_warning_cannot_become_new_test_baseline(environment):
    bridge, plan, _, _, _ = environment
    plan["context_warning"] = "Unknown map; empty draft shown"
    with pytest.raises(ValueError, match="Unknown map"):
        module.PinnedContext(bridge)


def test_startup_candidate_and_recent_duplicate_capture_are_allowed(environment):
    bridge, _, state, _, _ = environment
    pin = module.PinnedContext(bridge)
    state.update(map_state="candidate", map_ready=False, editor_test={"state": "queued"})
    pin.check(state)
    state.update(map_state="matched", map_ready=True, editor_test={"state": "running"},
                 capture_state="duplicate", capture_sequence=12, capture_age_s=.5)
    pin.check(state)
    pin.check(state)


@pytest.mark.parametrize("fields", [
    {"map_state": "candidate", "map_ready": False},
    {"map_state": "matched", "map_ready": True, "capture_state": "stale_or_missing"},
    {"map_state": "matched", "map_ready": True, "capture_state": "duplicate", "capture_age_s": .76},
])
def test_running_test_requires_map_ready_and_current_capture(environment, fields):
    bridge, _, state, _, _ = environment
    pin = module.PinnedContext(bridge)
    state.update(editor_test={"state": "running"}, **fields)
    with pytest.raises(RuntimeError):
        pin.check(state)


def test_stable_landing_requires_fresh_distinct_samples_and_sideways_probe(environment):
    _, _, _, _, now = environment
    proof = module.LandingProof({"left": 10, "right": 90, "y": 45}, from_x=40, away_px=2)
    assert not proof.observe([40, 45], 1000)
    now[0] = 1000.8
    for _ in range(10):
        assert not proof.observe([44, 45], 1000)
    assert len(proof.samples) == 1
    assert not proof.observe([40, 45], 1000.3)
    assert not proof.observe([40, 45], 1000.6)
    assert proof.observe([42, 45], 1000.8)


def test_inflight_floor_crossing_and_observation_gap_reset_landing(environment):
    _, _, _, _, now = environment
    proof = module.LandingProof({"left": 10, "right": 90, "y": 45})
    proof.observe([40, 45], 1000)
    now[0] = 1000.3
    assert not proof.observe([40, 46], 1000.3)
    assert proof.samples == []
    now[0] = 1002
    proof.observe([40, 45], 1002)
    now[0] = 1004
    assert not proof.observe([40, 45], 1004)
    assert len(proof.samples) == 1
    assert not proof.observe([40, 45], 1003.9)  # out-of-order metadata is ignored
    assert len(proof.samples) == 1


def test_health_grace_is_bounded_and_low_health_stops_immediately():
    guard = module.HealthGuard(1)  # cannot lower the supervised minimum below 25
    guard.check(None, now=0)
    guard.check(None, now=1.9)
    with pytest.raises(RuntimeError, match="unreadable"):
        guard.check(None, now=2.1)
    guard.check(50, now=3)
    assert guard.missing_since is None
    with pytest.raises(RuntimeError, match="25"):
        guard.check(24, now=3.1)


def test_failed_live_monitor_retains_trace_and_cancels_without_next_test(environment, monkeypatch):
    bridge, _, _, changing, _ = environment
    values = iter([50, 20])
    monkeypatch.setattr(module, "health", lambda log: next(values))
    with pytest.raises(RuntimeError, match="Health fell"):
        module.run_connections(bridge, ["rope", "rope"])
    assert changing["queued"] == 1 and changing["stopped"] == 1
    evidence = next(bridge.log.glob("route-check-*"))
    trace = json.loads((evidence / "step-01-trace.json").read_text())
    failure = json.loads((evidence / "step-01-failure.json").read_text())
    assert trace[0]["hp"] == 20 and failure["result"] == "failed"


def test_map_change_after_queue_stops_without_repeating_connection(environment):
    bridge, _, _, changing, _ = environment
    def queue(*args):
        changing["queued"] += 1
        changing["wrong_map"] = True
    bridge.test_connection = queue
    with pytest.raises(ValueError, match="minimap differs"):
        module.run_connections(bridge, ["rope", "rope"])
    assert changing["queued"] == 1 and changing["stopped"] == 1
    assert len(list(bridge.log.glob("route-check-*/step-01-trace.json"))) == 1


def test_completed_connection_requires_fresh_stable_paused_destination(environment):
    bridge, _, _, changing, _ = environment
    changing["test_state"] = "completed"
    results = module.run_connections(bridge, ["rope"])
    observations = results[0]["landing_observations"]
    assert len(observations) >= 3
    assert observations[-1]["updated"] - observations[0]["updated"] >= .6
    assert changing["queued"] == 1 and changing["stopped"] == 1


def test_duplicate_destination_snapshot_cannot_prove_final_landing(environment):
    bridge, _, state, changing, _ = environment
    changing["test_state"] = "completed"
    def repeated(*args):
        return None, {"updated": 1000, "player_pos": [44, 45]}
    bridge._check_context = repeated
    with pytest.raises(RuntimeError, match="fresh matching minimap"):
        module.run_connections(bridge, ["rope"])
    assert changing["queued"] == 1 and changing["stopped"] == 1


def test_slope_support_follows_observed_x_while_walking(environment):
    _, _, _, _, now = environment
    slope = {"left": 67, "right": 76, "y": 88, "right_y": 79}
    proof = module.LandingProof(slope, from_x=67, away_px=2)
    assert not proof.observe([69, 86], now[0])
    now[0] += .35
    assert not proof.observe([70, 85], now[0])
    now[0] += .35
    assert proof.observe([71, 84], now[0])
    assert [sample["expected_y"] for sample in proof.samples] == [86, 85, 84]
    assert all(sample["support_error_y"] == 0 for sample in proof.samples)


def test_final_slope_landing_requires_stable_xy_after_sideways_probe(environment):
    _, _, _, _, now = environment
    proof = module.LandingProof({"left": 67, "right": 76, "y": 88, "right_y": 79},
                                from_x=67, away_px=2, stationary=True)
    for x in [69, 70, 71]:
        assert not proof.observe([x, 155 - x], now[0])
        now[0] += .35
    assert len(proof.samples) == 1  # walking two pixels resets the final dwell
    assert not proof.observe([71, 84], now[0])
    now[0] += .35
    assert proof.observe([71, 84], now[0])


@pytest.mark.parametrize("bad_pos", [[70, 88], [70, 87], [77, 78], [66, 89]])
def test_slope_landing_rejects_wrong_height_and_outside_support(environment, bad_pos):
    _, _, _, _, now = environment
    proof = module.LandingProof({"left": 67, "right": 76, "y": 88, "right_y": 79})
    assert not proof.observe([69, 86], now[0])
    now[0] += .35
    assert not proof.observe(bad_pos, now[0])
    assert proof.samples == []


def test_slope_landing_duplicates_do_not_count(environment):
    _, _, _, _, now = environment
    proof = module.LandingProof({"left": 67, "right": 76, "y": 88, "right_y": 79},
                                stationary=True)
    assert not proof.observe([70, 85], now[0])
    stamp = now[0]
    now[0] += .8
    for _ in range(5):
        assert not proof.observe([70, 85], stamp)
    assert len(proof.samples) == 1


def test_completed_connection_reports_slope_height_at_final_position(environment):
    bridge, plan, _, changing, now = environment
    plan["platforms"][1].update(left=67, right=76, y=88, right_y=79)
    plan["connections"][0].update(to_x=67)
    bridge._check_context = lambda *args: (None, {"updated": now[0], "player_pos": [70, 85]})
    changing["test_state"] = "completed"
    result = module.run_connections(bridge, ["rope"])[0]
    assert result["position"] == [70, 85]
    assert result["destination_y"] == 88
    assert result["expected_y_at_position"] == 85
    assert len(result["landing_observations"]) >= 3
    assert all(row["support_error_y"] == 0 for row in result["landing_observations"])


def test_interior_shelf_knots_override_incorrect_endpoint_slope(environment):
    _, _, _, _, now = environment
    surface = {"left": 67, "right": 90, "y": 88, "right_y": 71,
               "knots": [[73, 82], [80, 82], [84, 77]]}
    proof = module.LandingProof(surface, from_x=74, away_px=2, stationary=True)
    for dt in [0, .35, .7]:
        now[0] = 1000 + dt
        result = proof.observe([79, 82], now[0])
    assert result
    assert all(row["expected_y"] == 82 for row in proof.samples)
    # The endpoint-only line is around79 here. That is below/above the actual
    # saved shelf and cannot become landing evidence simply by being stationary.
    wrong = module.LandingProof(surface)
    for dt in [0, .35, .7]:
        now[0] = 1001 + dt
        assert not wrong.observe([79, 79], now[0])


def test_completed_knotted_connection_uses_final_local_surface_height(environment):
    bridge, plan, _, changing, now = environment
    plan["platforms"][1].update(left=67, right=90, y=88, right_y=71,
                                knots=[[73, 82], [80, 82], [84, 77]])
    plan["connections"][0]["to_x"] = 75
    bridge._check_context = lambda *args: (None, {"updated": now[0], "player_pos": [79, 82]})
    changing["test_state"] = "completed"
    result = module.run_connections(bridge, ["rope"])[0]
    assert result["position"] == [79, 82]
    assert result["expected_y_at_position"] == 82
    assert len(result["landing_observations"]) >= 3


def test_rounded_minimap_x_does_not_reject_real_sloped_landing(environment):
    _, _, _, _, now = environment
    surface = {"left": 66.787, "right": 89.6, "y": 87.6, "right_y": 71.225,
               "knots": [[70.6, 84.475], [72.412, 81.787], [80.35, 81.85],
                         [84.287, 76.912], [86.6, 75.287]]}
    proof = module.LandingProof(surface, stationary=True)
    for dt in [0, .35, .7]:
        now[0] = 1000 + dt
        accepted = proof.observe([82, 79], now[0])
    assert accepted
    assert abs(proof.samples[-1]["nominal_error_y"]) > .75
    assert abs(proof.samples[-1]["support_error_y"]) < .75
    for pos in [[82, 77], [90, 71]]:
        bad = module.LandingProof(surface)
        for dt in [0, .35, .7]:
            now[0] = 1001 + dt
            assert not bad.observe(pos, now[0])


def test_quantization_does_not_widen_flat_landing_tolerance(environment):
    _, _, _, _, now = environment
    proof = module.LandingProof({"left": 70, "right": 80, "y": 82})
    for dt in [0, .35, .7]:
        now[0] = 1000 + dt
        assert not proof.observe([75, 83], now[0])
