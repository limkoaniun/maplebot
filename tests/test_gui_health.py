import json
from types import SimpleNamespace

import pytest

import maplebot.gui as gui
from tests.test_gui_recovery import settings


def test_health_settings_save_independent_hp_mp_and_route_failure_protection(settings):
    data = gui.SettingsWindow._validated(settings)
    assert data["health"]["hp_percent"] == 60
    assert data["health"]["minimum_hp_percent"] == 50
    assert data["health"]["mp_percent"] == 20
    assert data["health"]["protect_on_route_failure"] is True
    assert data["health"]["protection_retry_s"] == 1.0 and data["health"]["protection_max_taps"] == 3
    assert data["health"]["retry_s"] == 2 and data["health"]["max_retries"] == 3
    settings.protect_on_route_failure = SimpleNamespace(get=lambda: False)
    assert gui.SettingsWindow._validated(settings)["health"]["protect_on_route_failure"] is False


@pytest.mark.parametrize("path,value", [("protection_retry_s", ".4"), ("protection_retry_s", "11"),
    ("protection_max_taps", "0"), ("protection_max_taps", "6"), ("protection_max_taps", "1.5")])
def test_hp_protection_retry_fields_enforce_bounded_values(settings, path, value):
    settings.values[f"health.{path}"] = SimpleNamespace(get=lambda: value)
    with pytest.raises(ValueError):
        gui.SettingsWindow._validated(settings)


@pytest.mark.parametrize("minimum,trigger,valid", [(50, 50, False), (50, 49, False),
    (-1, 60, False), (100, 100, False), (99, 100, True), (0, 1, True), (50, 101, False)])
def test_minimum_hp_and_potion_trigger_have_consistent_bounds(settings, minimum, trigger, valid):
    settings.values["health.minimum_hp_percent"] = SimpleNamespace(get=lambda: str(minimum))
    settings.values["health.hp_percent"] = SimpleNamespace(get=lambda: str(trigger))
    if valid:
        assert gui.SettingsWindow._validated(settings)["health"]["minimum_hp_percent"] == minimum
    else:
        with pytest.raises(ValueError):
            gui.SettingsWindow._validated(settings)


@pytest.mark.parametrize("state,active,label", [
    ("monitoring", True, "Route stopped — HP protection on"),
    ("healing", True, "Route stopped — HP protection on"),
    ("awaiting_response", True, "Route stopped — HP protection on"),
    ("needs_attention", False, "Health needs attention"),
    ("dead", False, "Character down — input stopped"),
])
def test_health_status_keeps_reason_and_does_not_equate_key_presses_with_healing(state, active, label):
    result = gui._health_protection_display({"health_protection_state": state,
        "health_protection_active": active, "health_protection_reason": "Runtime explanation.",
        "health_protection_taps": 2})
    assert result[0] == label
    assert "Runtime explanation." in result[1] and "key presses: 2" in result[1]
    assert "does not confirm HP restoration" in result[1]


def test_explicit_full_pause_is_not_displayed_as_active_hp_protection():
    assert gui._health_protection_display({"health_protection_state": "suspended",
        "health_protection_active": False, "full_pause": True}) is None


def test_hold_preflight_error_is_visible_without_claiming_active_protection():
    result = gui._health_protection_display({"health_protection_state": "idle", "full_pause": True,
        "hold_request_error": "A current map is required."})
    assert result == ("Hold unavailable", "A current map is required.")
    assert gui._health_protection_display({"health_protection_state": "dead",
        "hold_request_error": "A current map is required."})[0] == "Character down — input stopped"


def test_failed_resume_return_stays_visible_while_hp_protection_monitors():
    state = {"health_protection_state": "monitoring", "health_protection_active": True,
             "startup_return_state": "failed", "startup_return_reason": "No path from this platform."}
    label, detail = gui._health_protection_display(state)
    assert label == "Return stopped — HP protection on"
    assert "No path from this platform" in detail and "Watching the current HP" in detail
    state["health_protection_state"] = "needs_attention"
    assert gui._health_protection_display(state)[0] == "Health needs attention"


@pytest.mark.parametrize("status,label", [
    ({"paused": True, "full_pause": False}, "Pause all"),
    ({"paused": True, "health_protection_active": True}, "Pause all"),
    ({"paused": True, "full_pause": True}, "Resume"),
    ({"paused": False, "full_pause": False}, "Pause"),
])
def test_pause_button_distinguishes_automatic_hold_from_full_pause(status, label):
    assert gui._pause_label(status) == label


def test_queued_hold_has_pause_all_even_before_controller_reports_mode_change():
    assert gui._pause_label({"paused": True, "full_pause": True}, hold_pending=True) == "Pause all"


@pytest.mark.parametrize("action", ["pause", "stop"])
def test_pause_and_stop_cancel_queued_health_hold(tmp_path, monkeypatch, action):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    (tmp_path / "hold-farming-active").write_text("queued")
    (tmp_path / "return-to-start-active").write_text("queued")
    window = SimpleNamespace(_editing_map=lambda: False, _running=lambda: True,
        status={"paused": True, "full_pause": True}, message=SimpleNamespace(set=lambda value: None))
    method = gui.SettingsWindow._pause_resume if action == "pause" else gui.SettingsWindow._stop
    method(window)
    assert not (tmp_path / "hold-farming-active").exists()
    assert not (tmp_path / "return-to-start-active").exists()
    assert (tmp_path / ("pause-active" if action == "pause" else "stop-active")).exists()
    assert not (tmp_path / "resume-active").exists()


def test_pause_all_from_automatic_hold_never_sends_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    (tmp_path / "resume-active").touch()
    messages = []
    window = SimpleNamespace(_editing_map=lambda: False, _running=lambda: True,
        status={"paused": True, "full_pause": False, "health_protection_active": True},
        message=SimpleNamespace(set=messages.append))
    gui.SettingsWindow._pause_resume(window)
    assert (tmp_path / "pause-active").exists()
    assert not (tmp_path / "resume-active").exists() and not (tmp_path / "arm-active").exists()
    assert "all input" in messages[-1]


def test_health_bell_only_rings_on_new_alert_transitions_and_warning_survives_stale_status():
    bells = []
    window = SimpleNamespace(status={}, _health_alert_state=None, _health_alert_display=None,
                             bell=lambda: bells.append("bell"))
    for state in ("idle", "needs_attention", "needs_attention", "dead", "dead"):
        window.status = {"health_protection_state": state}
        gui.SettingsWindow._update_health_alert(window, True)
    assert bells == ["bell", "bell"]
    warning = window._health_alert_display
    window.status = {}
    gui.SettingsWindow._update_health_alert(window, False)
    assert window._health_alert_display == warning
    window.status = {"health_protection_state": "monitoring", "health_protection_active": True}
    gui.SettingsWindow._update_health_alert(window, True)
    assert window._health_alert_display is None and window._health_alert_state is None
    window.status = {"health_protection_state": "needs_attention"}
    gui.SettingsWindow._update_health_alert(window, True)
    assert len(bells) == 3


def test_live_status_prioritizes_health_over_paused_and_route_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    monkeypatch.setattr(gui, "USER_CONFIG", tmp_path / "missing.yaml")
    status = {"running": True, "paused": True, "full_pause": False, "recovery_terminal": True,
              "recovery_reason": "Route unavailable", "health_protection_state": "needs_attention",
              "health_protection_reason": "HP did not increase", "health_protection_active": False}
    (tmp_path / "active-status.json").write_text(json.dumps(status))
    values, configurations = {}, {}
    var = lambda key: SimpleNamespace(set=lambda value: values.update({key: value}))
    button = lambda key: SimpleNamespace(configure=lambda **kwargs: configurations.update({key: kwargs}))
    window = SimpleNamespace(status={}, _health_alert_state=None, _health_alert_display=None,
        _running=lambda: True, _editing_map=lambda: False, bell=lambda: None,
        state_text=var("state"), activity=var("activity"), vitals=var("vitals"), message=var("message"),
        state_label=button("label"), start_button=button("start"), pause_button=button("pause"),
        stop_button=button("stop"), hold_button=button("hold"), return_button=button("return"), save_button=button("save"),
        after=lambda *args: None, launch_deadline=0., _poll=lambda: None)
    window._update_health_alert = lambda running: gui.SettingsWindow._update_health_alert(window, running)
    gui.SettingsWindow._poll(window)
    assert values["state"] == "Health needs attention" and values["activity"] == "HP did not increase"
    assert configurations["label"]["style"] == "Alert.State.TLabel"
    assert configurations["pause"]["text"] == "Pause all"
