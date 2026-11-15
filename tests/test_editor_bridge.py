import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import maplebot.editor_bridge as module
from maplebot.editor_bridge import EditorBridge, atomic_json, capture_signature, map_similarity


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    (tmp_path / "config").mkdir()
    custom = {"game_window": {"title": "Test game", "working_size": [1366, 768]},
              "minimap": {"fixed_rect": [56, 81, 111, 98]},
              "monsters": {"names": ["slime"]},
              "route": {"other_route_setting": "keep", "circuit": {"timeout_s": 22,
                        "pixels_per_minimap_x": 16, "custom_tuning": {"keep": 7}}},
              "unrelated_user_setting": {"key": [1, "two", True]},
              "keys": {"pickup": "x", "unrecognized_binding": "v"}}
    config_path = tmp_path / "config/user.yaml"
    config_path.write_text(yaml.safe_dump(custom), encoding="utf-8")
    bridge = EditorBridge(tmp_path)
    image = np.random.default_rng(7).integers(0, 256, (98, 111, 3), dtype=np.uint8)
    plan = {"version": 1, "name": "Test map", "image_size": [111, 98], "loop": True,
            "platforms": [{"id": "low", "left": 0, "right": 70, "y": 70},
                          {"id": "high", "left": 20, "right": 80, "y": 40}],
            "connections": [{"id": "rope-1", "kind": "rope", "from_platform": "low",
                             "to_platform": "high", "from_x": 40, "to_x": 40}],
            "route": [{"platform_id": "low", "x": 10}, {"platform_id": "high", "x": 60}],
            "capture": capture_signature(bridge._cfg()), "extra_metadata": {"keep": "yes"}}

    def snapshot(img=None, prefix="editor-minimap", updated=1000.0, capture=None, player=(30, 70)):
        img = image if img is None else img
        cv2.imwrite(str(bridge.log / f"{prefix}.png"), img)
        atomic_json(bridge.log / f"{prefix}.json", {
            "updated": updated, "size": list(img.shape[1::-1]),
            "capture": capture_signature(bridge._cfg()) if capture is None else capture,
            "player_pos": player, "source": "test"})

    def status(**overrides):
        value = {"running": True, "paused": True, "updated": 1000.0, "minimap_seen": True,
                 "minimap": [30, 70], "capture": capture_signature(bridge._cfg())}
        value.update(overrides)
        atomic_json(bridge.log / "active-status.json", value)

    snapshot()
    status()
    return bridge, plan, image, custom, snapshot, status


def test_profile_roundtrip_preserves_metadata_and_image_without_config_changes(setup):
    bridge, plan, image, _, _, _ = setup
    before = bridge.user_config.read_bytes()
    assert "Draft saved" in bridge.save_profile(plan, image, apply=False)
    loaded, reference, player = bridge.load_profile(plan["name"])
    assert loaded == plan and np.array_equal(reference, image) and player == (30, 70)
    assert bridge.user_config.read_bytes() == before
    assert bridge.list_profiles() == [plan["name"]]
    assert not (bridge.log / "resume-active").exists()
    assert not (bridge.log / "reload-active").exists()


@pytest.mark.parametrize("rect", [[56, 81, 111, 98], [20, 82, 210, 128]])
def test_select_profile_uses_full_frame_and_its_own_crop_over_old_minimap(setup, rect):
    from copy import deepcopy
    from maplebot.map_profiles import map_settings_from_config
    bridge, old_plan, old_image, original, _, _ = setup
    bridge.save_profile(old_plan, old_image)
    plan = deepcopy(old_plan)
    plan.update(name="Different map", image_size=rect[2:])
    plan["capture"]["minimap_rect"] = rect
    reference = np.random.default_rng(30).integers(0, 255, (rect[3], rect[2], 3), dtype=np.uint8)
    settings = map_settings_from_config(bridge._cfg())
    settings["minimap"]["fixed_rect"] = rect
    settings["monsters"]["names"] = ["stump"]
    bridge.save_profile(plan, reference, apply=False, map_settings=settings)
    # The old editor-minimap remains fresh with the old map's content/signature.
    prepared = np.zeros((768, 1366, 3), np.uint8)
    x, y, w, h = rect
    prepared[y:y+h, x:x+w] = reference
    cv2.imwrite(str(bridge.log / "controller-prepared.png"), prepared)
    atomic_json(bridge.log / "controller-prepared.json", {"updated": 1000., "capture": capture_signature(bridge._cfg())})
    bridge.select_profile(plan["name"])
    saved = yaml.safe_load(bridge.user_config.read_text())
    assert saved["minimap"]["fixed_rect"] == rect
    assert saved["monsters"]["names"] == ["stump"]
    assert saved["map_profile"]["active"] == bridge._profile_key(plan["name"])
    assert saved["keys"] == original["keys"]
    assert (bridge.log / "pause-active").exists() and not (bridge.log / "resume-active").exists()


def test_crop_calibration_invalidates_old_map_without_changing_character_controls(setup):
    bridge, plan, image, original, _, _ = setup
    bridge.save_profile(plan, image)
    prepared = np.zeros((768, 1366, 3), np.uint8)
    metadata = {"updated": 1000., "capture": capture_signature(bridge._cfg())}
    blank, crop, player = bridge.calibrate_crop([20, 82, 210, 128], prepared, metadata)
    saved = yaml.safe_load(bridge.user_config.read_text())
    assert crop.shape == (128, 210, 3) and player is None
    assert blank["platforms"] == [] and blank["connections"] == []
    assert saved["route"]["circuit"]["enabled"] is False
    assert saved["route"]["circuit"]["waypoints"] == []
    assert not saved["map_profile"].get("active") and saved["monsters"]["names"] == []
    assert saved["keys"] == original["keys"]
    assert (bridge.profiles / f"{bridge._profile_key(plan['name'])}.json").exists()


def test_legacy_resave_cannot_borrow_current_map_monsters(setup):
    bridge, plan, image, _, _, _ = setup
    bridge.profiles.mkdir()
    key = bridge._profile_key(plan["name"])
    cv2.imwrite(str(bridge.profiles / f"{key}.png"), image)
    atomic_json(bridge.profiles / f"{key}.json", {"version": 1, "plan": plan})
    with pytest.raises(ValueError, match="older profile"):
        bridge.save_profile(plan, image, apply=True)
    assert bridge.settings_for_plan(plan)["monsters"]["names"] == []


def test_ramp_landing_probe_continues_past_slope_instead_of_turning_back(setup):
    bridge, plan, image, _, _, _ = setup
    plan["connections"][0].update(kind="ramp", from_x=40, to_x=60)
    bridge.test_connection(plan, "rope-1", image)
    request = json.loads((bridge.log / "editor-test-request.json").read_text())
    assert request["circuit"]["waypoints"][-1]["x"] == 64
    bridge.stop_test()
    plan["platforms"][1]["right"] = 61
    with pytest.raises(ValueError, match="too narrow"):
        bridge.test_connection(plan, "rope-1", image)


def test_connection_source_preflight_accepts_local_slope_height(setup):
    bridge, plan, image, _, snapshot, _ = setup
    plan["platforms"][0]["right_y"] = 35
    snapshot(player=(30, 55))
    bridge.test_connection(plan, "rope-1", image)
    request = json.loads((bridge.log / "editor-test-request.json").read_text())
    assert request["source_platform"]["right_y"] == 35
    assert request["circuit"]["waypoints"][0]["y"] == 50


def test_connection_source_preflight_keeps_piecewise_surface_and_uses_shelf_height(setup):
    bridge, plan, image, _, snapshot, _ = setup
    plan["platforms"][0].update(knots=[[20, 55], [40, 55]])
    snapshot(player=(30, 55))
    bridge.test_connection(plan, "rope-1", image)
    request = json.loads((bridge.log / "editor-test-request.json").read_text())
    assert request["source_platform"]["knots"] == [[20, 55], [40, 55]]
    assert request["circuit"]["waypoints"][0]["y"] == 55
    assert request["circuit"]["platforms"][0]["knots"] == [[20, 55], [40, 55]]


def test_profile_apply_and_reload_keep_piecewise_knots_and_detect_geometry_drift(setup):
    from maplebot.map_profiles import read_profile, profile_config_matches
    bridge, plan, image, _, _, _ = setup
    plan["platforms"][0].update(knots=[[20, 55], [40, 55]])
    bridge.save_profile(plan, image)
    loaded, _, _ = bridge.load_profile(plan["name"])
    assert loaded["platforms"][0]["knots"] == [[20, 55], [40, 55]]
    cfg = bridge._cfg()
    profile = read_profile(bridge.profiles / f"{bridge._profile_key(plan['name'])}.json")
    assert profile_config_matches(cfg, profile)
    cfg["route"]["circuit"]["platforms"][0]["knots"][0][1] += 1
    assert not profile_config_matches(cfg, profile)


@pytest.mark.parametrize("paused", [False, True])
def test_manual_return_queues_request_without_changing_settings_or_pause_files(setup, paused):
    bridge, plan, image, _, _, status = setup
    bridge.save_profile(plan, image)
    status(paused=paused, map_ready=True, circuit_enabled=True)
    (bridge.log / "resume-active").touch()
    before = bridge.user_config.read_bytes()
    profile_path = bridge.profiles / f"{bridge._profile_key(plan['name'])}.json"
    profile_before = profile_path.read_bytes()
    assert "check your position" in bridge.return_to_start()
    request = json.loads((bridge.log / "return-to-start-active").read_text())
    assert request == {"created": 1000., "source": "settings"}
    assert (bridge.log / "pause-active").exists() and (bridge.log / "resume-active").exists()
    assert bridge.user_config.read_bytes() == before and profile_path.read_bytes() == profile_before
    with pytest.raises(ValueError, match="already running"):
        bridge.return_to_start()


@pytest.mark.parametrize("fields,message", [
    ({"running": False}, "Start the controller"),
    ({"updated": 996}, "Start the controller"),
    ({"map_ready": False}, "recognized"),
    ({"minimap_seen": False}, "recognized"),
    ({"capture": {}}, "recognized"),
    ({"circuit_enabled": False}, "enabled map circuit"),
    ({"returning_to_start": True}, "already running"),
    ({"editor_test": {"state": "running"}}, "Stop the movement test"),
])
def test_manual_return_rejects_stale_or_incompatible_session(setup, fields, message):
    bridge, plan, image, _, _, status = setup
    bridge.save_profile(plan, image)
    status(**dict({"map_ready": True, "circuit_enabled": True}, **fields))
    with pytest.raises(ValueError, match=message):
        bridge.return_to_start()
    assert not (bridge.log / "return-to-start-active").exists()


def test_sync_active_settings_persists_recovery_limit_in_map_profile(setup):
    from maplebot.map_profiles import read_profile, apply_profile, profile_config_matches
    bridge, plan, image, _, _, _ = setup
    bridge.save_profile(plan, image)
    data = yaml.safe_load(bridge.user_config.read_text())
    data["route"]["circuit"].update(return_on_block=True, max_returns_per_lap=2)
    bridge.sync_active_settings(data)
    profile = read_profile(bridge.profiles / f"{bridge._profile_key(plan['name'])}.json")
    assert profile.map_settings["circuit_settings"]["return_on_block"] is True
    assert profile.map_settings["circuit_settings"]["max_returns_per_lap"] == 2
    reapplied = apply_profile(bridge._cfg(), profile)
    assert reapplied["route"]["circuit"]["max_returns_per_lap"] == 2
    assert profile_config_matches(reapplied, profile)


def test_pause_cancels_a_queued_return_before_it_can_unpause_the_editor(setup):
    bridge, plan, image, _, _, status = setup
    bridge.save_profile(plan, image)
    status(map_ready=True, circuit_enabled=True)
    bridge.return_to_start()
    bridge.pause()
    assert not (bridge.log / "return-to-start-active").exists()
    assert (bridge.log / "pause-active").exists()


@pytest.mark.parametrize("paused", [False, True])
def test_hold_farming_queues_only_request_and_is_independent_of_automatic_failure_preference(setup, paused):
    bridge, _, _, _, _, status = setup
    cfg = yaml.safe_load(bridge.user_config.read_text())
    cfg.setdefault("health", {}).update(enable=True, protect_on_route_failure=False)
    bridge.user_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    status(paused=paused, full_pause=paused, map_ready=True)
    (bridge.log / "pause-active").touch()
    (bridge.log / "resume-active").touch()
    before = bridge.user_config.read_bytes()
    assert "guarded HP protection stays on" in bridge.hold_farming()
    request = json.loads((bridge.log / "hold-farming-active").read_text())
    assert request == {"created": 1000., "source": "settings"}
    assert bridge.user_config.read_bytes() == before
    assert (bridge.log / "pause-active").exists() and (bridge.log / "resume-active").exists()
    assert not (bridge.log / "reload-active").exists()
    assert "already requested" in bridge.hold_farming()
    assert json.loads((bridge.log / "hold-farming-active").read_text()) == request
    bridge.pause()
    assert not (bridge.log / "hold-farming-active").exists()


@pytest.mark.parametrize("fields,message", [
    ({"running": False}, "Start the controller"),
    ({"updated": 996}, "Start the controller"),
    ({"map_ready": False}, "recognized"),
    ({"minimap_seen": False}, "recognized"),
    ({"capture": {}}, "recognized"),
    ({"editor_test": {"state": "running"}}, "Finish the movement or potion test"),
    ({"potion_test": {"state": "running"}}, "Finish the movement or potion test"),
])
def test_hold_farming_requires_fresh_normal_session(setup, fields, message):
    bridge, _, _, _, _, status = setup
    status(**dict({"paused": False, "map_ready": True}, **fields))
    with pytest.raises(ValueError, match=message):
        bridge.hold_farming()
    assert not (bridge.log / "hold-farming-active").exists()


@pytest.mark.parametrize("disabled", ["health", "key"])
def test_hold_farming_requires_health_readings_and_hp_key(setup, disabled):
    bridge, _, _, _, _, status = setup
    cfg = yaml.safe_load(bridge.user_config.read_text())
    if disabled == "health":
        cfg["health"] = {"enable": False}
    else:
        cfg["keys"]["hp_potion"] = ""
    bridge.user_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    status(paused=False, map_ready=True)
    with pytest.raises(ValueError, match="Enable health readings"):
        bridge.hold_farming()
    assert not (bridge.log / "hold-farming-active").exists()


@pytest.mark.parametrize("held", [
    {"manual_health_hold": True, "paused": True},
    {"manual_health_hold": True, "paused": False},
    {"recovery_terminal": True, "paused": True, "health_protection_active": False},
    {"recovery_terminal": True, "paused": False, "health_protection_active": True},
    {"circuit_state": "blocked", "paused": True},
])
def test_existing_permanent_hold_is_idempotent_without_writing_a_request(setup, held):
    bridge, _, _, _, _, status = setup
    status(map_ready=True, full_pause=False, **held)
    before = bridge.user_config.read_bytes()
    assert "already held" in bridge.hold_farming()
    assert "already held" in bridge.hold_farming()
    assert not (bridge.log / "hold-farming-active").exists()
    assert bridge.user_config.read_bytes() == before


@pytest.mark.parametrize("temporary", [
    {"health_protection_active": True, "health_protection_state": "healing"},
    {"health_protection_active": True, "health_protection_state": "awaiting_response"},
    {"health_protection_active": False, "health_protection_state": "monitoring"},
    {"manual_health_hold": True, "startup_return_state": "checking"},
    {"manual_health_hold": True, "startup_return_state": "returning"},
])
def test_temporary_refill_or_startup_hold_queues_a_permanent_hold(setup, temporary):
    bridge, _, _, _, _, status = setup
    status(map_ready=True, paused=True, full_pause=False, **temporary)
    assert "Hold requested" in bridge.hold_farming()
    request = json.loads((bridge.log / "hold-farming-active").read_text())
    assert request == {"created": 1000., "source": "settings"}


def test_terminal_route_with_protection_opted_out_still_accepts_explicit_hold(setup):
    bridge, _, _, _, _, status = setup
    cfg = yaml.safe_load(bridge.user_config.read_text())
    cfg.setdefault("health", {})["protect_on_route_failure"] = False
    bridge.user_config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    status(map_ready=True, paused=True, full_pause=False, recovery_terminal=True)
    assert "Hold requested" in bridge.hold_farming()
    assert (bridge.log / "hold-farming-active").exists()


@pytest.mark.parametrize("fields,message", [
    ({"updated": 996.}, "current game image"),
    ({"map_ready": False}, "recognized"),
    ({"minimap_seen": False}, "recognized"),
    ({"capture": {}}, "recognized"),
    ({"editor_test": {"state": "running"}}, "Finish the movement"),
])
def test_idempotent_hold_still_requires_fresh_compatible_session(setup, fields, message):
    bridge, _, _, _, _, status = setup
    status(**dict({"map_ready": True, "paused": True, "full_pause": False,
                   "manual_health_hold": True}, **fields))
    with pytest.raises(ValueError, match=message):
        bridge.hold_farming()
    assert not (bridge.log / "hold-farming-active").exists()


def test_idempotent_hold_does_not_cancel_pause_stop_or_rewrite_pending_request(setup):
    bridge, _, _, _, _, status = setup
    status(map_ready=True, paused=True, full_pause=False, health_protection_active=True)
    pending = bridge.log / "hold-farming-active"
    atomic_json(pending, {"created": 999., "source": "previous"})
    before = pending.read_bytes()
    for name in ("pause-active", "stop-active", "resume-active"):
        (bridge.log / name).touch()
    assert "already requested" in bridge.hold_farming()
    assert pending.read_bytes() == before
    assert all((bridge.log / name).exists() for name in ("pause-active", "stop-active", "resume-active"))


@pytest.mark.parametrize("state,updated,allowed", [("preflight", 1000., False),
    ("awaiting_response", 1000., False), ("passed", 1000., True), ("failed", 1000., True),
    ("awaiting_response", 969., True)])
def test_hold_farming_does_not_overlap_fresh_standalone_potion_test(setup, state, updated, allowed):
    bridge, _, _, _, _, status = setup
    status(paused=False, map_ready=True)
    atomic_json(bridge.log / "potion-verification.json", {"state": state, "updated": updated})
    if allowed:
        bridge.hold_farming()
        assert (bridge.log / "hold-farming-active").exists()
    else:
        with pytest.raises(ValueError, match="Finish the movement or potion test"):
            bridge.hold_farming()
        assert not (bridge.log / "hold-farming-active").exists()


def test_jump_onto_slope_compiles_landing_probe_at_its_local_height(setup):
    bridge, plan, image, _, _, _ = setup
    plan["platforms"][1].update(y=60, right_y=30)
    plan["connections"][0].update(kind="jump", from_x=40, to_x=60)
    bridge.test_connection(plan, "rope-1", image)
    request = json.loads((bridge.log / "editor-test-request.json").read_text())
    jump = next(w for w in request["circuit"]["waypoints"] if w["action"] == "jump")
    assert jump["y"] == 40
    final = request["circuit"]["waypoints"][-1]
    assert (final["x"], final["y"]) == (56, 42)


def test_apply_preserves_unknown_user_settings_and_never_resumes(setup):
    bridge, plan, image, original, _, _ = setup
    (bridge.log / "resume-active").touch()
    atomic_json(bridge.log / "editor-test-request.json", {"old": True})
    assert "stays paused" in bridge.save_profile(plan, image)
    saved = yaml.safe_load(bridge.user_config.read_text())
    assert saved["unrelated_user_setting"] == original["unrelated_user_setting"]
    assert saved["keys"] == original["keys"]
    assert saved["route"]["other_route_setting"] == "keep"
    circuit = saved["route"]["circuit"]
    assert circuit["custom_tuning"] == {"keep": 7} and circuit["timeout_s"] == 22
    assert circuit["editor_plan"] == plan
    assert len(circuit["editor_reference_sha256"]) == 64
    assert (bridge.log / "pause-active").exists() and (bridge.log / "stop-editor-test").exists()
    assert (bridge.log / "reload-active").exists()
    assert not (bridge.log / "resume-active").exists()
    assert not (bridge.log / "editor-test-request.json").exists()
    loaded, _, _ = bridge.load_current()
    assert loaded == plan


def test_wrong_map_cannot_inherit_stored_applied_coordinates(setup):
    bridge, plan, image, _, snapshot, _ = setup
    bridge.save_profile(plan, image)
    saved_before = bridge.user_config.read_bytes()
    different = np.random.default_rng(12).integers(0, 256, image.shape, dtype=np.uint8)
    snapshot(different)
    fresh, reference, _ = bridge.load_current()
    assert fresh["name"] == "New map" and fresh["platforms"] == [] and fresh["route"] == []
    assert fresh["connections"] == [] and "context_warning" in fresh
    assert np.array_equal(reference, different)
    assert bridge.user_config.read_bytes() == saved_before
    with pytest.raises(ValueError, match="current minimap differs"):
        bridge.save_profile(plan, image)
    # Offline loading remains possible but must not show a current player dot.
    old, _, player = bridge.load_profile(plan["name"])
    assert old == plan and player is None


def test_capture_change_does_not_rebind_applied_plan(setup):
    bridge, plan, image, _, snapshot, _ = setup
    bridge.save_profile(plan, image)
    cfg = yaml.safe_load(bridge.user_config.read_text())
    cfg["game_window"]["title_bar_px"] = 10
    bridge.user_config.write_text(yaml.safe_dump(cfg))
    snapshot()
    fresh, _, _ = bridge.load_current()
    assert fresh["platforms"] == [] and "differ" in fresh["context_warning"]
    stored = yaml.safe_load(bridge.user_config.read_text())["route"]["circuit"]["editor_plan"]
    assert stored["capture"] == plan["capture"]
    assert fresh["capture"] != stored["capture"]


def test_applied_reference_digest_blocks_profile_image_replacement(setup):
    bridge, plan, image, _, _, _ = setup
    bridge.save_profile(plan, image)
    other = np.random.default_rng(8).integers(0, 256, image.shape, dtype=np.uint8)
    bridge.save_profile(plan, other, apply=False)
    fresh, _, _ = bridge.load_current()
    assert fresh["platforms"] == []
    assert "reference image changed" in fresh["context_warning"]


def test_profile_image_tampering_is_detected(setup):
    bridge, plan, image, _, _, _ = setup
    bridge.save_profile(plan, image, apply=False)
    key = bridge._profile_key(plan["name"])
    cv2.imwrite(str(bridge.profiles / f"{key}.png"), np.zeros_like(image))
    with pytest.raises(ValueError, match="changed independently"):
        bridge.load_profile(plan["name"])


def test_profile_names_cannot_escape_storage_directory(setup):
    bridge, plan, image, _, _, _ = setup
    plan["name"] = "../../CON: map\\evil"
    bridge.save_profile(plan, image, apply=False)
    key = bridge._profile_key(plan["name"])
    assert "/" not in key and "\\" not in key and ":" not in key and ".." not in key
    assert {p.suffix for p in bridge.profiles.iterdir()} == {".json", ".png"}
    assert all(p.parent == bridge.profiles for p in bridge.profiles.iterdir())
    assert bridge.load_profile(plan["name"])[0]["name"] == plan["name"]
    for name in ("", " " * 3, "x" * 101):
        with pytest.raises(ValueError):
            bridge._profile_key(name)


@pytest.mark.parametrize("stamp", [990, None, float("nan"), 1005])
def test_snapshot_rejects_stale_invalid_and_future_timestamps(setup, stamp):
    bridge, _, _, _, snapshot, _ = setup
    snapshot(updated=stamp)
    with pytest.raises(ValueError, match="No fresh minimap"):
        bridge._snapshot()


def test_snapshot_uses_only_matching_capture_and_can_fall_back_to_relay(setup):
    bridge, _, image, _, snapshot, _ = setup
    snapshot(capture={"wrong": "settings"})
    with pytest.raises(ValueError, match="No fresh minimap"):
        bridge._snapshot()
    snapshot(prefix="relay-editor-minimap", player=None)
    frame, metadata = bridge._snapshot()
    assert np.array_equal(frame, image) and metadata["player_pos"] is None


def test_context_independently_checks_metadata_and_requires_profile_capture(setup, monkeypatch):
    bridge, plan, image, _, _, _ = setup
    metadata = {"updated": 1000, "capture": {"wrong": True}}
    monkeypatch.setattr(bridge, "_snapshot", lambda cfg=None: (image, metadata))
    with pytest.raises(ValueError, match="stale or uses different"):
        bridge._check_context(plan, image, bridge._cfg())
    metadata["capture"] = capture_signature(bridge._cfg())
    del plan["capture"]
    with pytest.raises(ValueError, match="settings differ"):
        bridge._check_context(plan, image, bridge._cfg())


def test_connection_test_queues_only_selected_edge_once_and_does_not_change_config(setup):
    bridge, plan, image, _, _, _ = setup
    plan["connections"].append({"id": "other", "kind": "rope", "from_platform": "low",
                                "to_platform": "high", "from_x": 60, "to_x": 60})
    before = bridge.user_config.read_bytes()
    (bridge.log / "resume-active").touch()
    assert "Test queued" in bridge.test_connection(plan, "rope-1", image)
    request = json.loads((bridge.log / "editor-test-request.json").read_text())
    circuit = request["circuit"]
    transitions = [w for w in circuit["waypoints"] if w["action"] != "walk"]
    assert len(transitions) == 1 and transitions[0]["connection_id"] == "rope-1"
    assert circuit["movement_test"] and circuit["loop"] is False and circuit["max_retries"] == 1
    assert circuit["tolerance_y"] == .75 and circuit["tolerance_x"] == 2
    assert circuit["waypoints"][0]["dwell_s"] == 0
    last = circuit["waypoints"][-1]
    assert last["action"] == "walk" and last["platform_id"] == "high"
    assert last["x"] == 44 and last["y"] == 40 and last["dwell_s"] >= .6
    assert circuit["pixels_per_minimap_x"] == 16 and circuit["custom_tuning"] == {"keep": 7}
    assert request["reference"] == Path(request["reference"]).name
    assert (bridge.log / request["reference"]).exists()
    assert (bridge.log / "pause-active").exists() and not (bridge.log / "resume-active").exists()
    assert not (bridge.log / "reload-active").exists()
    assert bridge.user_config.read_bytes() == before
    with pytest.raises(ValueError, match="already queued"):
        bridge.test_connection(plan, "rope-1", image)
    bridge.stop_test()
    assert not (bridge.log / "editor-test-request.json").exists()
    assert (bridge.log / "stop-editor-test").exists()


@pytest.mark.parametrize("landing,left,right,expected", [(78, 20, 80, 74), (40, 37, 43, 43)])
def test_connection_probe_stays_on_destination_and_moves_even_on_small_platform(setup, landing, left, right, expected):
    from maplebot.nav.circuit import MinimapCircuit
    bridge, plan, image, _, _, _ = setup
    # The source platform must also contain a vertical rope's chosen X.
    plan["platforms"][0]["right"] = 90
    plan["platforms"][1].update(left=left, right=right)
    plan["connections"][0].update(from_x=landing, to_x=landing)
    plan["route"][1]["x"] = landing
    bridge.test_connection(plan, "rope-1", image)
    circuit = json.loads((bridge.log / "editor-test-request.json").read_text())["circuit"]
    final = circuit["waypoints"][-1]
    assert final["x"] == expected and left <= final["x"] <= right
    assert abs(expected - landing) - circuit["tolerance_x"] >= 2
    assert circuit["tolerance_x"] == min(2, abs(expected - landing) - 2)
    circuit["start_idx"] = len(circuit["waypoints"]) - 1
    nav = MinimapCircuit(circuit)
    assert nav.step((landing, 40), 0).move_x != "none"
    assert not nav.completed
    nav.step((expected, 42), .1)
    assert not nav.completed  # hanging two pixels below is not a landing
    # The nearest accepted arrival still proves at least two pixels of movement.
    direction = 1 if expected > landing else -1
    accepted_x = expected - direction * circuit["tolerance_x"]
    assert abs(accepted_x - landing) >= 2
    nav.step((accepted_x, 40), .2)
    nav.step((accepted_x, 40), .9)
    assert nav.completed


def test_connection_test_rejects_destination_too_narrow_for_a_landing_probe(setup):
    bridge, plan, image, _, _, _ = setup
    plan["platforms"][1].update(left=39, right=41)
    plan["route"][1]["x"] = 40
    before = bridge.user_config.read_bytes()
    with pytest.raises(ValueError, match="too narrow.*at least 3"):
        bridge.test_connection(plan, "rope-1", image)
    assert not (bridge.log / "editor-test-request.json").exists()
    assert bridge.user_config.read_bytes() == before


@pytest.mark.parametrize("status_fields, message", [
    ({"running": False}, "Start the controller"),
    ({"updated": 990}, "Start the controller"),
    ({"paused": False}, "Pause the controller"),
    ({"capture": {}}, "not loaded the current capture"),
])
def test_connection_test_requires_current_paused_controller(setup, status_fields, message):
    bridge, plan, image, _, _, status = setup
    status(**status_fields)
    with pytest.raises(ValueError, match=message):
        bridge.test_connection(plan, "rope-1", image)
    assert not (bridge.log / "editor-test-request.json").exists()


def test_connection_start_uses_matching_snapshot_position_over_old_status(setup):
    bridge, plan, image, _, snapshot, _ = setup
    snapshot(player=(40, 40))
    with pytest.raises(ValueError, match="starting platform"):
        bridge.test_connection(plan, "rope-1", image)


def test_empty_draft_can_be_saved_without_being_executable(setup):
    bridge, plan, image, _, _, _ = setup
    plan.update(platforms=[], connections=[], route=[])
    bridge.save_profile(plan, image, apply=False)
    assert bridge.load_profile(plan["name"])[0] == plan
    with pytest.raises(ValueError, match="Draw at least one platform"):
        bridge.save_profile(plan, image)


def test_map_similarity_returns_floats_and_rejects_different_structure():
    black = np.zeros((30, 40, 3), np.uint8)
    assert type(map_similarity(black, black)) is float
    assert map_similarity(black, black) == 1.0
    assert map_similarity(black, black + 100) == 0.0
    rng = np.random.default_rng(5)
    a, b = [rng.integers(0, 256, (98, 111, 3), dtype=np.uint8) for _ in range(2)]
    moved_dot = a.copy()
    cv2.circle(moved_dot, (40, 50), 3, (0, 255, 255), -1)
    assert map_similarity(a, moved_dot) > .55
    assert map_similarity(a, b) < .55
    assert map_similarity(a, b[:50]) == 0.0


def test_atomic_json_retries_transient_windows_reader_lock(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    original = Path.replace
    attempts, delays = [], []

    def busy_twice(path, destination):
        attempts.append(destination)
        if len(attempts) < 3:
            raise PermissionError("reader still open")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", busy_twice)
    monkeypatch.setattr(module.time, "sleep", delays.append)
    atomic_json(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}
    assert len(attempts) == 3 and delays == [.01, .02]
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}')
    calls = []

    def busy(path, destination):
        calls.append(destination)
        raise PermissionError("still busy")

    monkeypatch.setattr(Path, "replace", busy)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        atomic_json(target, {"new": True})
    assert json.loads(target.read_text()) == {"old": True}
    assert len(calls) == 4 and list(tmp_path.glob("*.tmp")) == []


def _scroll(image, dx, dy=0):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                          image.shape[1::-1], flags=cv2.INTER_NEAREST)


@pytest.mark.parametrize("dx", [-9, 8])
def test_scrolled_active_snapshot_and_editor_use_reference_coordinates(setup, dx):
    bridge, plan, image, _, snapshot, status = setup
    bridge.save_profile(plan, image)
    snapshot(_scroll(image, dx), player=(30+dx, 70))
    status(minimap=[30+dx, 70])  # an untagged raw status cannot override alignment
    frame, metadata = bridge._snapshot()
    assert np.array_equal(frame, image)
    assert metadata["coordinate_space"] == "profile"
    assert metadata["minimap_offset"] == [-dx, 0]
    assert metadata["raw_player_pos"] == (30+dx, 70)
    assert metadata["player_pos"] == (30, 70)
    loaded, reference, player = bridge.load_current()
    assert loaded == plan and np.array_equal(reference, image) and player == (30, 70)


def test_scrolled_profile_can_apply_without_rebasing_saved_geometry_or_image(setup):
    bridge, plan, image, _, snapshot, _ = setup
    bridge.save_profile(plan, image)
    key = bridge._profile_key(plan["name"])
    before = (bridge.profiles / f"{key}.png").read_bytes()
    snapshot(_scroll(image, 8), player=(38, 70))
    loaded, reference, _ = bridge.load_current()
    bridge.save_profile(loaded, reference)
    assert (bridge.profiles / f"{key}.png").read_bytes() == before
    saved = yaml.safe_load(bridge.user_config.read_text())
    assert saved["route"]["circuit"]["editor_plan"] == plan
    assert not (bridge.log / "resume-active").exists()


def test_scrolled_connection_preflight_uses_canonical_dot_and_reference(setup):
    bridge, plan, image, _, snapshot, status = setup
    plan["platforms"][0]["right"] = 34
    plan["connections"][0].update(from_x=30, to_x=30)
    bridge.save_profile(plan, image)
    snapshot(_scroll(image, 8), player=(38, 70))
    status(minimap=[38, 70])  # outside source support in raw screen coordinates
    bridge.test_connection(plan, "rope-1", image)
    request = json.loads((bridge.log / "editor-test-request.json").read_text())
    assert request["source_platform"] == plan["platforms"][0]
    reference = cv2.imread(str(bridge.log / request["reference"]))
    np.testing.assert_array_equal(reference, image)


def test_missing_current_dot_never_falls_back_to_status_on_saved_route(setup):
    bridge, plan, image, _, snapshot, status = setup
    bridge.save_profile(plan, image)
    snapshot(_scroll(image, 8), player=None)
    status(minimap=[30, 70], minimap_seen=True)
    assert bridge.load_current()[2] is None
    with pytest.raises(ValueError, match="starting platform"):
        bridge.test_connection(plan, "rope-1", image)
    assert not (bridge.log / "editor-test-request.json").exists()


def test_draft_profile_alignment_does_not_require_already_active_profile(setup):
    bridge, plan, image, _, snapshot, _ = setup
    bridge.save_profile(plan, image, apply=False)
    snapshot(_scroll(image, -9), player=(21, 70))
    loaded, reference, player = bridge.load_profile(plan["name"])
    assert loaded == plan and np.array_equal(reference, image) and player == (30, 70)


def test_new_map_without_applied_profile_retains_raw_image_and_dot(setup):
    bridge, _, image, _, snapshot, _ = setup
    shifted = _scroll(image, 8)
    snapshot(shifted, player=(38, 70))
    plan, reference, player = bridge.load_current()
    assert plan["name"] == "New map" and not plan["platforms"]
    assert np.array_equal(reference, shifted) and player == (38, 70)


def test_canonical_export_cannot_be_mistaken_for_a_raw_snapshot(setup):
    bridge, plan, image, _, _, _ = setup
    bridge.save_profile(plan, image)
    path = bridge.log / "editor-minimap.json"
    metadata = json.loads(path.read_text())
    metadata.update(coordinate_space="profile", player_pos=[30, 70],
                    minimap_offset=[-8, 0], raw_player_pos=[38, 70])
    atomic_json(path, metadata)
    with pytest.raises(ValueError, match="No fresh minimap"):
        bridge._raw_snapshot()


def test_shifted_raw_image_cannot_rebase_an_applied_route_silently(setup):
    bridge, plan, image, _, snapshot, _ = setup
    bridge.save_profile(plan, image)
    shifted = _scroll(image, 8)
    snapshot(shifted, player=(38, 70))
    before = bridge.user_config.read_bytes()
    with pytest.raises(ValueError, match="saved coordinate system"):
        bridge.save_profile(plan, shifted)
    assert bridge.user_config.read_bytes() == before


def test_canonical_dot_outside_reference_is_not_replaced_by_raw_or_status_dot(setup):
    bridge, plan, image, _, snapshot, status = setup
    bridge.save_profile(plan, image)
    snapshot(_scroll(image, 8), player=(2, 70))
    status(minimap=[30, 70], minimap_seen=True)
    loaded, reference, player = bridge.load_current()
    assert loaded == plan and np.array_equal(reference, image) and player is None


def test_uncertain_alignment_shows_raw_new_map_and_cannot_queue_saved_connection(setup, monkeypatch):
    import maplebot.map_profiles as profiles
    bridge, plan, image, _, snapshot, _ = setup
    bridge.save_profile(plan, image)
    shifted = _scroll(image, 8)
    snapshot(shifted, player=(38, 70))
    monkeypatch.setattr(profiles, "align_map", lambda *args: (0., (-8, 0), .01))
    loaded, reference, player = bridge.load_current()
    assert loaded["name"] == "New map" and not loaded["route"]
    assert np.array_equal(reference, shifted) and player == (38, 70)
    with pytest.raises(ValueError, match="alignment is uncertain"):
        bridge.test_connection(plan, "rope-1", image)
    assert not (bridge.log / "editor-test-request.json").exists()
