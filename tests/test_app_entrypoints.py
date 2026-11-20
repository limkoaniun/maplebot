"""maplebot.app modes, run_script and the controller-mode order (offline; stub scripts only)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import maplebot
from maplebot import app, first_run, launcher, paths
from tests.packaging_guard import FakeApi, install


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


@pytest.fixture
def root(tmp_path, monkeypatch):
    folder = tmp_path / "Maplebot"
    folder.mkdir()
    monkeypatch.setattr(paths, "app_root", lambda: folder)
    return folder


def _exe(monkeypatch, name, *, frozen=True, root=None):
    monkeypatch.setattr(paths, "exe_name", lambda: name)
    monkeypatch.setattr(paths, "is_frozen", lambda: frozen)
    if root is not None:
        monkeypatch.setattr(paths, "resource_dir", lambda: root / "_internal")


def test_mode_allowlists(monkeypatch, root, capsys):
    _exe(monkeypatch, "maplebot-cli.exe")
    assert app.main(["controller"]) == 2 and app.main(["relay"]) == 2 and app.main(["gui"]) == 2
    assert app.main([]) == 2
    assert app.main(["unknown"]) == 2
    _exe(monkeypatch, "maplebot.exe")
    assert app.main(["supervise", "--help"]) == 2 and app.main(["start"]) == 2
    assert app.main(["import-settings", "x"]) == 2
    assert "usage error" in (root / "log" / "launcher.log").read_text(encoding="utf-8")
    _exe(monkeypatch, "renamed.exe")
    assert app.main(["version"]) == 2
    assert app.main(["--help"]) == 0


def test_no_arguments_open_the_gui_in_the_windowed_exe_and_source(monkeypatch, root):
    ran = []
    monkeypatch.setattr(app, "mode_gui", lambda args: ran.append(args) or 0)
    monkeypatch.setitem(app.MODES, "gui", app.mode_gui)
    _exe(monkeypatch, "maplebot.exe")
    assert app.main([]) == 0
    _exe(monkeypatch, "", frozen=False)
    assert app.main([]) == 0
    assert ran == [[], []]


def _gui_stub(monkeypatch):
    seen = {}
    stub = types.ModuleType("maplebot.gui")
    stub.main = lambda: seen.setdefault("cwd", Path.cwd())
    monkeypatch.setitem(sys.modules, "maplebot.gui", stub)
    monkeypatch.setattr(maplebot, "gui", stub, raising=False)
    return seen


def test_gui_mode_runs_the_stub_in_the_root(monkeypatch, root):
    seen = _gui_stub(monkeypatch)
    monkeypatch.setattr(first_run, "path_supported", lambda r: True)
    assert app.main(["gui"]) == 0
    assert seen["cwd"] == root
    assert (root / "config" / "user.yaml").read_text(encoding="utf-8") == first_run.USER_SEED
    assert (root / "log").is_dir() and (root / "config" / "map_profiles").is_dir()


def test_gui_mode_refuses_an_unwritable_root(monkeypatch, root):
    seen = _gui_stub(monkeypatch)
    boxes = []
    monkeypatch.setattr(app, "_message_box", lambda text, flags=0: boxes.append((text, flags)))
    monkeypatch.setattr(first_run, "check_writable", lambda r: False)
    assert app.main(["gui"]) == 3
    assert "cannot write" in boxes[0][0] and boxes[0][1] == 0x10 and "cwd" not in seen


def test_gui_mode_warns_about_a_non_ascii_folder(monkeypatch, root):
    _gui_stub(monkeypatch)
    boxes = []
    monkeypatch.setattr(app, "_message_box", lambda text, flags=0: boxes.append(text))
    monkeypatch.setattr(first_run, "is_ascii_path", lambda r: False)
    assert app.main(["gui"]) == 0
    assert boxes == [app.NON_ASCII_WARNING]


def _stub_script(monkeypatch, root, body):
    internal = root / "_internal" / "scripts"
    internal.mkdir(parents=True, exist_ok=True)
    (internal / "stub.py").write_text(body, encoding="utf-8")
    monkeypatch.setattr(paths, "BUNDLED_SCRIPTS", paths.BUNDLED_SCRIPTS + ("stub.py",))
    monkeypatch.setattr(paths, "resource_dir", lambda: root / "_internal")
    monkeypatch.setattr(paths, "is_frozen", lambda: True)


STUB = '''
import sys
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Point:
    x: int

RECORD.update(file=__file__, name=__name__, argv=list(sys.argv), main=sys.modules["__main__"].__dict__ is globals(),
              root=Path(__file__).resolve().parents[1], point=Point(3).x)
if "--raise" in sys.argv:
    raise RuntimeError("stub failure line")
'''


def test_run_script_virtual_file_and_restoration(monkeypatch, root):
    _stub_script(monkeypatch, root, STUB)
    record = {}
    monkeypatch.setattr(sys.modules["builtins"], "RECORD", record, raising=False)
    saved_main, saved_argv = sys.modules.get("__main__"), sys.argv[:]
    app.run_script("stub.py", ["--wait-for-start"], root=root)
    virtual = str(root / "scripts" / "stub.py")
    assert record["file"] == virtual and record["name"] == "__main__" and record["main"]
    assert record["argv"] == [virtual, "--wait-for-start"]
    assert record["root"] == root.resolve() and record["point"] == 3
    assert not (root / "scripts").exists()
    assert sys.modules.get("__main__") is saved_main and sys.argv == saved_argv
    app.run_script("stub.py", [], root=root)
    assert record["argv"] == [virtual]

    with pytest.raises(RuntimeError) as failure:
        app.run_script("stub.py", ["--raise"], root=root)
    import traceback
    text = "".join(traceback.format_exception(failure.value))
    assert 'raise RuntimeError("stub failure line")' in text and virtual in text
    assert sys.modules.get("__main__") is saved_main and sys.argv == saved_argv


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, name, args, **kwargs):
        self.calls.append((name, list(args), kwargs))


def _controller(monkeypatch, root, *, create=(1234, 0), elevated=True, processes=()):
    api = FakeApi(create=lambda name: create, elevated=elevated, processes=processes)
    monkeypatch.setattr(launcher, "Win32Api", lambda *a, **k: api)
    recorder = _Recorder()
    monkeypatch.setattr(app, "run_script", recorder)
    _exe(monkeypatch, "maplebot.exe", root=root)
    return api, recorder


def test_controller_duplicate_exits_before_touching_user_files(monkeypatch, root):
    api, recorder = _controller(monkeypatch, root, create=(99, 183))
    assert app.main(["controller"]) == 0
    assert recorder.calls == [] and api.closed == [99]
    assert not (root / "config").exists()
    assert "duplicate launch ignored" in (root / "log" / "launcher.log").read_text(encoding="utf-8")
    api, recorder = _controller(monkeypatch, root, create=(0, 5))
    assert app.main(["controller"]) == 0 and recorder.calls == []


def test_controller_refuses_unelevated(monkeypatch, root):
    api, recorder = _controller(monkeypatch, root, elevated=False)
    assert app.main(["controller"]) == 4
    assert recorder.calls == [] and api.closed == [1234] and not (root / "config").exists()


def test_controller_refuses_a_foreign_controller(monkeypatch, root):
    api, recorder = _controller(monkeypatch, root, processes=[(7, "pythonw.exe")])
    api.command_lines[7] = r"pythonw.exe -u C:\repo\scripts\run_active.py"
    assert app.main(["controller"]) == 0
    assert recorder.calls == [] and api.closed == [1234]
    assert "controller refused: a source-checkout" in (root / "log" / "launcher.log").read_text(encoding="utf-8")


def test_controller_refuses_shadowing_files(monkeypatch, root):
    (root / "win32api.py").write_text("", encoding="utf-8")
    api, recorder = _controller(monkeypatch, root)
    assert app.main(["controller"]) == 5
    assert recorder.calls == [] and api.closed == [1234]
    assert "importable files" in (root / "log" / "launcher.log").read_text(encoding="utf-8")


def test_controller_normal_path(monkeypatch, root):
    api, recorder = _controller(monkeypatch, root)
    assert app.main(["controller", "--wait-for-start"]) == 0
    assert [(c[0], c[1]) for c in recorder.calls] == [("run_active.py", ["--wait-for-start"])]
    assert recorder.calls[0][2]["root"] == root
    assert api.closed == [1234] and (root / "config" / "user.yaml").is_file()
    assert ("create_mutex", launcher.GLOBAL_MUTEX) in api.calls
    assert app.main(["controller", "--other"]) == 2


@pytest.mark.parametrize("code, expected", [(None, 0), (0, 0), (3, 3)])
def test_controller_system_exit_codes(monkeypatch, root, code, expected):
    api, _ = _controller(monkeypatch, root)

    def exits(name, args, **kwargs):
        raise SystemExit(code)

    monkeypatch.setattr(app, "run_script", exits)
    assert app.main(["controller"]) == expected and api.closed == [1234]


def test_controller_crash_is_logged(monkeypatch, root):
    api, _ = _controller(monkeypatch, root)

    def crashes(name, args, **kwargs):
        raise RuntimeError("controller exploded")

    monkeypatch.setattr(app, "run_script", crashes)
    assert app.main(["controller"]) == 1 and api.closed == [1234]
    assert "controller exploded" in (root / "log" / "launcher.log").read_text(encoding="utf-8")


def test_relay_and_supervise_refuse_with_a_shadow_hazard(monkeypatch, root):
    recorder = _Recorder()
    monkeypatch.setattr(app, "run_script", recorder)
    (root / "scripts").mkdir()
    _exe(monkeypatch, "maplebot.exe", root=root)
    assert app.main(["relay"]) == 5
    assert "relay refused" in (root / "log" / "relay-helper.log").read_text(encoding="utf-8")
    _exe(monkeypatch, "maplebot-cli.exe", root=root)
    assert app.main(["supervise", "--minutes", "1"]) == 5
    assert recorder.calls == []


def test_relay_and_supervise_run_their_scripts(monkeypatch, root):
    recorder = _Recorder()
    monkeypatch.setattr(app, "run_script", recorder)
    _exe(monkeypatch, "maplebot.exe", root=root)
    assert app.main(["relay"]) == 0
    _exe(monkeypatch, "maplebot-cli.exe", root=root)
    assert app.main(["supervise", "--output", "s.jsonl", "--minutes", "1"]) == 0
    assert app.main(["supervise", "--root", "elsewhere", "--minutes", "1"]) == 0
    assert app.main(["supervise", "--root=elsewhere"]) == 0
    assert recorder.calls[0][:2] == ("relay_client.py", [])
    assert recorder.calls[1][:2] == ("supervise_farming.py", ["--root", str(root), "--output", "s.jsonl", "--minutes", "1"])
    assert recorder.calls[2][1] == ["--root", "elsewhere", "--minutes", "1"]
    assert recorder.calls[3][1] == ["--root=elsewhere"]


def test_scripts_are_never_imported():
    assert "run_active" not in sys.modules and "scripts.run_active" not in sys.modules


def test_stop_and_status_on_an_empty_root(monkeypatch, root, capsys):
    assert app.main(["stop"]) == 0
    assert [p.relative_to(root).as_posix() for p in root.rglob("*")] == ["log", "log/stop-active"]
    capsys.readouterr()
    assert app.main(["status"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report == {"verdict": "not running", "status_age_s": None, "status": None}


def test_paths_without_probe_creates_nothing(monkeypatch, root, capsys):
    assert app.main(["paths"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["writable"] == "not probed" and Path(report["app_root"]) == root
    assert list(root.iterdir()) == []
    assert app.main(["paths", "--probe"]) == 0
    assert json.loads(capsys.readouterr().out)["writable"] is True
    assert sorted(p.name for p in root.iterdir()) == ["config", "log"]
    assert not any(p.is_file() for p in root.rglob("*"))


def test_start_mode_codes(monkeypatch, root, capsys):
    api = FakeApi(mutexes={launcher.GLOBAL_MUTEX})
    monkeypatch.setattr(launcher, "Win32Api", lambda *a, **k: api)
    assert app.main(["start"]) == 3
    assert "already running" in capsys.readouterr().out
    api.mutexes.clear()
    api.runas = lambda *args: 1223
    assert app.main(["start", "--wait-for-start"]) == 1
    assert "cancelled" in capsys.readouterr().out
    assert api.calls[-1][0] == "runas" and api.calls[-1][2].endswith("--wait-for-start")
    api.runas = lambda *args: 0
    assert app.main(["start"]) == 0
    assert app.main(["start", "--bogus"]) == 2


def test_restore_example_mode(monkeypatch, root, capsys):
    assert app.main(["restore-example"]) == 1  # this folder has no example
    assert "does not include" in capsys.readouterr().out
    assert app.main(["restore-example", "a", "b"]) == 2


def _example(root):
    folder = root / "examples" / "construction-site"
    folder.mkdir(parents=True)
    for name in first_run.EXAMPLE_FILES["construction-site"]:
        (folder / name).write_bytes(b"keys:\n  attack: z\n" if name == "user.yaml" else b"{}")


@pytest.mark.parametrize("probe", ["global", "root", "foreign"])
def test_restore_example_mode_refuses_while_running(monkeypatch, root, capsys, probe):
    _example(root)
    api = FakeApi()
    if probe == "global":
        api.mutexes.add(launcher.GLOBAL_MUTEX)
    elif probe == "root":
        api.mutexes.add(launcher.per_root_mutex_name(root))
    else:
        api.processes = [(31, "pythonw.exe")]
        api.command_lines[31] = r"pythonw.exe -u C:\repo\scripts\run_active.py"
    monkeypatch.setattr(launcher, "Win32Api", lambda *a, **k: api)
    assert app.main(["restore-example"]) == 1
    assert "Stop the running" in capsys.readouterr().out
    assert (root / "config" / "user.yaml").read_text(encoding="utf-8") == first_run.USER_SEED
    assert list((root / "config" / "map_profiles").iterdir()) == []
    api.mutexes.clear()
    api.processes = []
    assert app.main(["restore-example"]) == 0
    assert (root / "config" / "user.yaml").read_bytes() == b"keys:\n  attack: z\n"


def test_import_settings_mode_prints_the_settings_app_note(monkeypatch, root, tmp_path, capsys):
    source = tmp_path / "checkout"
    (source / "config").mkdir(parents=True)
    (source / "config" / "user.yaml").write_text("keys:\n  attack: z\n", encoding="utf-8")
    assert app.main(["import-settings", str(source)]) == 0
    assert app.IMPORT_GUI_NOTE in capsys.readouterr().out


def test_version_mode(monkeypatch, capsys):
    assert app.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"Maplebot {maplebot.__version__} (source)"
    info = types.ModuleType("maplebot._build_info")
    info.VERSION, info.COMMIT, info.DIRTY = "0.1.0", "abc1234", True
    monkeypatch.setitem(sys.modules, "maplebot._build_info", info)
    assert app.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "Maplebot 0.1.0 (abc1234+dirty)"


def test_app_module_imports_no_logger_module():
    source = Path(app.__file__).read_text(encoding="utf-8")
    top = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert "from maplebot import first_run, launcher, paths" in top
    assert not any(name in line for line in top for name in ("maplebot.bot", "maplebot.log", "maplebot.gui",
                                                             "maplebot.capture", "maplebot.config"))
