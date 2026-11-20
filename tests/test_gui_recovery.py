from copy import deepcopy
from types import SimpleNamespace

import pytest

import maplebot.gui as gui


def test_recovery_status_reports_original_point_and_progress_when_waiting_for_game():
    label, detail = gui._recovery_display({"returning_to_start": True, "recovery_from_index": 8,
        "recovery_index": 2, "recovery_count": 7, "return_count": 1, "recovery_state": "walking"})
    assert label == "Returning to point 1"
    assert "point 9" in detail and "step 3/7" in detail and "attempt 1" in detail
    assert "Bring MapleStory" in detail


def test_terminal_recovery_reason_takes_precedence_over_original_block():
    label, detail = gui._recovery_display({"recovery_terminal": True, "recovery_from_index": 3,
        "paused": True, "circuit_state": "blocked", "recovery_reason": "No path from this floor."})
    assert label == "Recovery stopped"
    assert "point 4" in detail and "No path from this floor." in detail
    assert gui._recovery_display({"returning_to_start": False, "recovery_terminal": False}) is None


def test_manual_request_preflight_failure_is_visible_without_active_recovery():
    result = gui._recovery_display({"returning_to_start": False, "recovery_terminal": False,
        "recovery_state": "idle", "recovery_reason": "A fresh minimap position is required."})
    assert result == ("Return unavailable", "A fresh minimap position is required.")
    assert gui._recovery_display({"recovery_state": "completed", "recovery_reason": "Reached point 1."}) is None


def test_regaining_support_stage_is_shown_with_its_progress_and_failure_reason():
    status = {"returning_to_start": True, "recovery_from_index": 0, "return_count": 1, "foreground": True,
              "recovery_state": "regaining_support", "circuit_state": "return_regain_walking",
              "recovery_reason": "Walking left toward drawn platform ground"}
    label, detail = gui._recovery_display(status)
    assert label == "Returning to point 1"
    assert "point 1" in detail and "attempt 1" in detail
    assert "Walking left toward drawn platform ground" in detail and "Bring MapleStory" not in detail
    _, detail = gui._recovery_display({**status, "recovery_reason": None})
    assert "Regaining support" in detail
    reason = ("Could not reach a drawn platform from (65,95): no horizontal progress. Move the character "
              "onto a drawn platform or draw this surface in the map editor.")
    label, detail = gui._recovery_display({"recovery_terminal": True, "recovery_from_index": 0,
        "recovery_state": "failed", "recovery_reason": reason})
    assert label == "Recovery stopped" and reason in detail


def test_paused_return_is_identified_in_status():
    _, detail = gui._recovery_display({"returning_to_start": True, "paused": True})
    assert "Paused." in detail and "Bring MapleStory" not in detail


def test_resume_preflight_and_failed_return_are_visible_before_recovery_is_armed():
    label, detail = gui._recovery_display({"startup_return_state": "checking",
        "startup_return_reason": "Waiting for HP to recover.", "foreground": False})
    assert label == "Preparing return to point 1"
    assert "Waiting for HP" in detail and "Bring MapleStory" in detail
    label, detail = gui._recovery_display({"startup_return_state": "failed",
        "startup_return_reason": "No path from the current platform.", "recovery_terminal": False})
    assert label == "Return to point 1 stopped" and "No path" in detail
    assert "Farming stays held" in detail
    assert gui._recovery_display({"startup_return_state": "completed"}) is None


def test_resume_farming_from_manual_hold_keeps_pause_all_available_and_queues_no_live_input(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    messages = []
    status = {"paused": True, "full_pause": False, "manual_health_hold": True,
              "health_protection_active": True, "health_protection_state": "monitoring",
              "circuit_enabled": True}
    window = SimpleNamespace(status=status, _editing_map=lambda: False, _running=lambda: True,
        message=SimpleNamespace(set=messages.append))
    assert gui._can_resume_manual_hold(status) and gui._pause_label(status) == "Pause all"
    gui.SettingsWindow._start(window)
    assert (tmp_path / "resume-active").exists() and (tmp_path / "arm-active").exists()
    assert not (tmp_path / "pause-active").exists() and not (tmp_path / "return-to-start-active").exists()
    assert "returns to point 1 before farming" in messages[-1]


@pytest.mark.parametrize("extra", [{"startup_return_state": "checking"},
    {"startup_return_state": "returning"}, {"returning_to_start": True},
    {"health_protection_state": "dead"}, {"editor_test": {"state": "running"}}, {"manual_health_hold": False}])
def test_resume_manual_hold_action_is_unavailable_during_active_return_or_other_session(extra):
    assert not gui._can_resume_manual_hold({"paused": True, "full_pause": False, "manual_health_hold": True, **extra})


def test_pause_cancels_pending_return_even_when_last_status_was_already_paused(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    request = tmp_path / "return-to-start-active"
    request.write_text("queued")
    messages = []
    window = SimpleNamespace(_editing_map=lambda: False, _running=lambda: True,
        status={"paused": True}, message=SimpleNamespace(set=messages.append))
    gui.SettingsWindow._pause_resume(window)
    assert not request.exists() and (tmp_path / "pause-active").exists()
    assert not (tmp_path / "resume-active").exists()
    assert "cancelled" in messages[0]


@pytest.fixture
def settings(monkeypatch, tmp_path):
    (tmp_path / "character.png").write_bytes(b"test")
    monkeypatch.setattr(gui, "ROOT", tmp_path)
    data = {"unrelated": {"preserve": True}, "route": {"circuit": {"editor_plan": {"preserve": True},
            "waypoints": [{"x": 10, "y": 40}, {"x": 50, "y": 40}]}}}
    monkeypatch.setattr(gui, "_read_user", lambda: deepcopy(data))
    values = {
        "keys.attack": "z", "keys.normal_attack": "alt", "keys.jump": "c", "keys.hp_potion": "q",
        "keys.mp_potion": "a", "keys.pickup": "x", "keys.pickup_interval_s": ".5",
        "attack.range_x": "70", "attack.range_y": "48", "attack.cooldown_s": ".65",
        "attack.mp_fallback_percent": "5", "attack.deadzone_px": "28", "search.flip_frames": "60",
        "health.hp_percent": "60", "health.mp_percent": "20", "health.minimum_hp_percent": "50",
        "health.retry_s": "2", "health.max_retries": "3",
        "health.protection_retry_s": "1.0", "health.protection_max_taps": "3",
        "player.nametag.diff_thres": ".1", "monsters.diff_thres": ".1", "safety.max_missing_frames": "3",
        "route.circuit.tolerance_x": "1", "route.circuit.tolerance_y": "1", "route.circuit.timeout_s": "15",
        "route.circuit.max_returns_per_lap": "1", "loot.settle_s": ".8", "loot.max_approach_s": "2",
        "player.nametag.path": "character.png",
    }
    variable = lambda value: SimpleNamespace(get=lambda: value)
    return SimpleNamespace(values={k: variable(v) for k, v in values.items()},
        monsters={"stump": variable(True)}, loot_enabled=variable(True), post_combat_loot_enabled=variable(True),
        circuit_enabled=variable(True), return_on_block=variable(True), protect_on_route_failure=variable(True))


def test_settings_validation_preserves_geometry_and_saves_recovery_options(settings):
    data = gui.SettingsWindow._validated(settings)
    circuit = data["route"]["circuit"]
    assert circuit["return_on_block"] is True and circuit["max_returns_per_lap"] == 1
    assert circuit["editor_plan"] == {"preserve": True} and len(circuit["waypoints"]) == 2
    assert data["unrelated"] == {"preserve": True}


@pytest.mark.parametrize("value", ["0", "4", "1.5", "nan"])
def test_settings_reject_unbounded_or_noninteger_recovery_limits(settings, value):
    settings.values["route.circuit.max_returns_per_lap"] = SimpleNamespace(get=lambda: value)
    with pytest.raises(ValueError, match="max returns per lap|Max returns per lap"):
        gui.SettingsWindow._validated(settings)
