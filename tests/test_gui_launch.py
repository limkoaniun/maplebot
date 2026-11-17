"""Settings-app Start: the pending-launch guard, the launcher call and the failure message.

No window is created: the methods run on SimpleNamespace stand-ins, and the launcher is a recorder.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from maplebot import first_run, gui, launcher
from tests.packaging_guard import install

PNG = b"\x89PNG\r\n\x1a\n" + b"data"


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


def _window(**values):
    messages = []
    base = dict(_editing_map=lambda: False, _running=lambda: False, launch_deadline=0.0, launch_process=None,
                status={}, message=SimpleNamespace(set=messages.append),
                start_button=SimpleNamespace(configure=lambda **kwargs: None))
    base.update(values)
    return SimpleNamespace(**base), messages


def _no_launch(*args, **kwargs):
    raise AssertionError("the launcher must not be called")


def test_pending_prompt_blocks_a_second_start(monkeypatch):
    monkeypatch.setattr(gui.launcher, "start_controller", _no_launch)
    saves = []
    window, _ = _window(launch_process=SimpleNamespace(poll=lambda: None), _save=lambda: saves.append(1) or True)
    gui.SettingsWindow._start(window)
    assert saves == []


def test_start_calls_the_launcher_after_saving(monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "ROOT", tmp_path)
    calls = []

    def start(root, wait_for_start=False, hwnd=0, **kwargs):
        calls.append((root, wait_for_start, hwnd))
        return launcher.LaunchHandle()

    monkeypatch.setattr(gui.launcher, "start_controller", start)
    finished = SimpleNamespace(poll=lambda: 0)
    window, messages = _window(launch_process=finished, _save=lambda: True)
    gui.SettingsWindow._start(window)
    assert calls == [(tmp_path, False, 0)]  # no winfo_id on the stand-in, so hwnd falls back to 0
    assert window.launch_deadline > 0 and "administrator prompt" in messages[-1]
    assert window.launch_process.poll() is None


def test_launcher_os_error_is_reported(monkeypatch):
    def fails(*args, **kwargs):
        raise OSError("signal file locked")

    errors = []
    monkeypatch.setattr(gui.launcher, "start_controller", fails)
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, text, parent=None: errors.append(text))
    window, _ = _window(_save=lambda: True)
    gui.SettingsWindow._start(window)
    assert errors == ["signal file locked"] and window.launch_deadline == 0


def test_running_controller_without_launch_attribute_still_resumes(monkeypatch, tmp_path):
    """test_gui_recovery's stand-in has no launch_process; the guard must sit after _running()."""
    monkeypatch.setattr(gui, "LOG", tmp_path)
    monkeypatch.setattr(gui.launcher, "start_controller", _no_launch)
    status = {"paused": True, "full_pause": False, "manual_health_hold": True, "circuit_enabled": False}
    messages = []
    window = SimpleNamespace(status=status, _editing_map=lambda: False, _running=lambda: True,
                             message=SimpleNamespace(set=messages.append))
    gui.SettingsWindow._start(window)
    assert (tmp_path / "resume-active").exists() and (tmp_path / "arm-active").exists()


def _poll_window(tmp_path, launch_process):
    values, configurations = {}, {}
    var = lambda key: SimpleNamespace(set=lambda value: values.update({key: value}))  # noqa: E731
    button = lambda key: SimpleNamespace(configure=lambda **kwargs: configurations.update({key: kwargs}))  # noqa: E731
    window = SimpleNamespace(status={}, _health_alert_state=None, _health_alert_display=None,
        _running=lambda: False, _editing_map=lambda: False, bell=lambda: None,
        state_text=var("state"), activity=var("activity"), vitals=var("vitals"), message=var("message"),
        state_label=button("label"), start_button=button("start"), pause_button=button("pause"),
        stop_button=button("stop"), hold_button=button("hold"), return_button=button("return"),
        save_button=button("save"), after=lambda *args: None, launch_deadline=1e18,
        launch_process=launch_process, _poll=lambda: None)
    window._update_health_alert = lambda running: None
    return window, values


@pytest.mark.parametrize("code, text", [
    (183, "Another Maplebot controller is already running (a source-checkout Maplebot tool is running (pid 4))."),
    (1223, "Start was cancelled at the Windows administrator prompt."),
    (1, "The launch did not complete. Try Start active and approve the Windows prompt."),
])
def test_poll_reports_a_finished_launch_once(monkeypatch, tmp_path, code, text):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    monkeypatch.setattr(gui, "USER_CONFIG", tmp_path / "missing.yaml")
    handle = launcher.LaunchHandle.done(code, "a source-checkout Maplebot tool is running (pid 4)")
    window, values = _poll_window(tmp_path, handle)
    gui.SettingsWindow._poll(window)
    if code == 183:
        assert values["message"] == text + " Stop it before starting again."
    else:
        assert values["message"] == text
    assert window.launch_deadline == 0 and window.launch_process is None and values["state"] == "Stopped"


def test_poll_keeps_starting_while_the_prompt_is_open(monkeypatch, tmp_path):
    monkeypatch.setattr(gui, "LOG", tmp_path)
    monkeypatch.setattr(gui, "USER_CONFIG", tmp_path / "missing.yaml")
    window, values = _poll_window(tmp_path, launcher.LaunchHandle())
    gui.SettingsWindow._poll(window)
    assert values["state"] == "Starting…" and "message" not in values


FORM_KEYS = ("keys.attack", "keys.hp_potion", "health.hp_percent", "health.minimum_hp_percent")
SEED_FORM = {"keys": {"attack": "w", "hp_potion": "1"}, "health": {"hp_percent": 60.0, "minimum_hp_percent": 50.0}}


def _settings_window(monkeypatch, tmp_path):
    """A stand-in settings window whose form was loaded from the empty seed (no Tk window)."""
    root = tmp_path / "Maplebot"
    first_run.ensure_user_layout(root)
    monkeypatch.setattr(gui, "ROOT", root)
    monkeypatch.setattr(gui, "USER_CONFIG", root / "config" / "user.yaml")
    monkeypatch.setattr(gui, "LOG", root / "log")
    events = {"loads": 0, "warnings": [], "errors": []}
    monkeypatch.setattr(gui.messagebox, "showwarning",
                        lambda title, text, parent=None: events["warnings"].append(text))
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, text, parent=None: events["errors"].append(text))
    values = {key: SimpleNamespace(get=lambda: "") for key in FORM_KEYS}
    messages = []

    def load():
        events["loads"] += 1
        window._loaded_form_values = gui._form_snapshot(gui._read_user(), values)

    def validated():
        data = gui._read_user()
        for key in FORM_KEYS:  # the stale form, as _validated overlays it
            gui._set(data, key, gui._get(SEED_FORM, key))
        return data

    window = SimpleNamespace(values=values, _editing_map=lambda: False, _running=lambda: False,
                             _load=load, _validated=validated, message=SimpleNamespace(set=messages.append))
    load()
    events["loads"] = 0
    return window, root, events, messages


def test_save_refuses_a_stale_form_after_a_cli_import(monkeypatch, tmp_path):
    """maplebot-cli import-settings while the settings app is open: the next Save (or Start active,
    which saves first) must not overlay the old form on the imported keys."""
    window, root, events, messages = _settings_window(monkeypatch, tmp_path)
    source = tmp_path / "checkout"
    (source / "config").mkdir(parents=True)
    imported = "keys:\n  attack: c\n  hp_potion: end\nhealth:\n  hp_percent: 72\n  minimum_hp_percent: 45\n"
    (source / "config" / "user.yaml").write_text(imported, encoding="utf-8")
    first_run.import_settings(root, source, alive=lambda: None)

    assert gui.SettingsWindow._save(window) is False
    assert (root / "config" / "user.yaml").read_text(encoding="utf-8") == imported
    assert events["loads"] == 1 and events["warnings"] == [gui.SETTINGS_CHANGED_MESSAGE]
    assert messages[-1] == gui.SETTINGS_CHANGED_MESSAGE
    # The form now reflects the file, so the following save goes through.
    window._validated = lambda: gui._read_user()
    assert gui.SettingsWindow._save(window) is True
    assert gui._read_user()["keys"]["hp_potion"] == "end"


def test_save_writes_when_nothing_changed_and_tracks_its_own_write(monkeypatch, tmp_path):
    window, root, events, _ = _settings_window(monkeypatch, tmp_path)
    assert gui.SettingsWindow._save(window) is True
    assert gui._read_user()["keys"]["hp_potion"] == "1" and events["loads"] == 0
    # Its own write is not "changed elsewhere"; a change to a key the form does not own is not either.
    data = gui._read_user()
    data["route"] = {"circuit": {"tolerance_x": 2}, "map": None}
    data["supervisor"] = {"note": "kept"}
    (root / "config" / "user.yaml").write_text(gui.yaml.safe_dump(data), encoding="utf-8")
    assert gui.SettingsWindow._save(window) is True and events["loads"] == 0
    assert gui._read_user()["supervisor"] == {"note": "kept"}
    # A form-owned key written elsewhere (here the HP potion key) is.
    data = gui._read_user()
    data["keys"]["hp_potion"] = "end"
    (root / "config" / "user.yaml").write_text(gui.yaml.safe_dump(data), encoding="utf-8")
    assert gui.SettingsWindow._save(window) is False and events["loads"] == 1
    assert gui._read_user()["keys"]["hp_potion"] == "end"


def test_load_example_button_passes_the_controller_probe(monkeypatch, tmp_path, guards):
    window, root, events, _ = _settings_window(monkeypatch, tmp_path)
    example = root / "examples" / "construction-site"
    example.mkdir(parents=True)
    for name in first_run.EXAMPLE_FILES["construction-site"]:
        (example / name).write_bytes(b"keys:\n  attack: z\n" if name == "user.yaml" else b"{}")
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *args, **kwargs: True)
    guards.mutexes.add(launcher.GLOBAL_MUTEX)
    gui.SettingsWindow._load_example(window)
    assert events["errors"] and "Stop the running" in events["errors"][-1]
    assert (root / "config" / "user.yaml").read_text(encoding="utf-8") == first_run.USER_SEED
    assert events["loads"] == 0
    guards.mutexes.clear()
    gui.SettingsWindow._load_example(window)
    assert (root / "config" / "user.yaml").read_bytes() == b"keys:\n  attack: z\n" and events["loads"] == 1


def test_import_button_probes_the_source_folder_too(monkeypatch, tmp_path, guards):
    import tkinter.filedialog
    window, root, events, _ = _settings_window(monkeypatch, tmp_path)
    source = tmp_path / "checkout"
    (source / "config").mkdir(parents=True)
    (source / "config" / "user.yaml").write_text("keys:\n  attack: c\n", encoding="utf-8")
    monkeypatch.setattr(tkinter.filedialog, "askdirectory", lambda **kwargs: str(source))
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *args, **kwargs: True)
    guards.mutexes.add(launcher.per_root_mutex_name(source))  # only the SOURCE folder's controller runs
    gui.SettingsWindow._import_settings(window)
    assert events["errors"] and "Stop every running" in events["errors"][-1]
    assert (root / "config" / "user.yaml").read_text(encoding="utf-8") == first_run.USER_SEED
    guards.mutexes.clear()
    gui.SettingsWindow._import_settings(window)
    assert (root / "config" / "user.yaml").read_text(encoding="utf-8") == "keys:\n  attack: c\n"
    assert events["loads"] == 1


def test_character_image_is_copied_into_assets(tmp_path):
    root = tmp_path / "Maplebot"
    root.mkdir()
    picked = tmp_path / "Desktop" / "my-tag.png"
    picked.parent.mkdir()
    picked.write_bytes(PNG)
    assert gui._import_character_image(root, picked) == "assets/my-tag.png"
    assert (root / "assets" / "my-tag.png").read_bytes() == PNG
    assert gui._import_character_image(root, picked) == "assets/my-tag.png"  # identical: fine
    assert gui._import_character_image(root, root / "assets" / "my-tag.png") == "assets/my-tag.png"
    picked.write_bytes(PNG + b"other")
    with pytest.raises(ValueError, match="already in the assets folder"):
        gui._import_character_image(root, picked)
    assert (root / "assets" / "my-tag.png").read_bytes() == PNG
    text = tmp_path / "notes.png"
    text.write_text("not an image")
    with pytest.raises(ValueError, match="not a PNG"):
        gui._import_character_image(root, text)
    named = tmp_path / "명찰.png"
    named.write_bytes(PNG)
    with pytest.raises(ValueError, match="Rename"):
        gui._import_character_image(root, named)
