"""launcher.py: probes, signal clearing and the UAC launch, all through a fake Win32 API."""
from __future__ import annotations

import ctypes
import hashlib
import json
import re
import sys
import threading
from pathlib import Path

import pytest

from maplebot import launcher
from tests.packaging_guard import FakeApi, install

REPO = Path(__file__).resolve().parents[1]
OTHER_FILES = ("editor-test-request.json", "stop-editor-test", "stop-relay", "supervise-off", "active-status.json")


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "Maple bot root"
    (folder / "log").mkdir(parents=True)
    return folder


def _status(root, **values):
    (root / "log" / "active-status.json").write_text(json.dumps(values), encoding="utf-8")


NOW = 1_000_000.0


@pytest.mark.parametrize("image", ["maplebot.exe", "pythonw.exe", "python.exe"])
def test_fresh_status_with_a_controller_image_is_running(root, image):
    _status(root, running=True, updated=NOW - 2, pid=77)
    api = FakeApi(states={77: ("alive", image)})
    assert launcher.controller_alive(root, api=api, now=lambda: NOW)


def test_denied_process_counts_as_running(root):
    _status(root, running=True, updated=NOW - 1, pid=77)
    api = FakeApi(states={77: ("denied", None)})
    assert "access denied" in launcher.controller_alive(root, api=api, now=lambda: NOW)


@pytest.mark.parametrize("status", [
    {"running": True, "updated": NOW - 6, "pid": 77},       # stale
    {"running": False, "updated": NOW - 1, "pid": 77},      # stopped
    {"running": True, "updated": NOW - 1, "pid": 78},       # dead pid
    {"running": True, "updated": NOW - 1, "pid": 79},       # reused pid with another image
])
def test_not_running(root, status):
    _status(root, **status)
    api = FakeApi(states={77: ("alive", "pythonw.exe"), 79: ("alive", "notepad.exe")})
    assert launcher.controller_alive(root, api=api, now=lambda: NOW) is None


@pytest.mark.parametrize("content", [None, "{not json", "[1, 2]"])
def test_missing_or_corrupt_status_is_not_running(root, content):
    if content is not None:
        (root / "log" / "active-status.json").write_text(content, encoding="utf-8")
    assert launcher.controller_alive(root, api=FakeApi(), now=lambda: NOW) is None


@pytest.mark.parametrize("pid", ["abc", None, [1], True])
def test_malformed_fresh_pid_counts_as_running(root, pid):
    _status(root, running=True, updated=NOW - 1, pid=pid)
    assert launcher.controller_alive(root, api=FakeApi(), now=lambda: NOW) == "running (unverifiable pid)"


@pytest.mark.parametrize("which", ["global", "root"])
def test_either_mutex_means_running_even_without_status(root, which):
    name = launcher.GLOBAL_MUTEX if which == "global" else launcher.per_root_mutex_name(root)
    _status(root, running=True, updated=0, pid=1)
    assert launcher.controller_alive(root, api=FakeApi(mutexes={name}), now=lambda: NOW)
    (root / "log" / "active-status.json").unlink()
    assert launcher.controller_alive(root, api=FakeApi(mutexes={name}), now=lambda: NOW)


def test_foreign_controllers():
    processes = [(1, "pythonw.exe"), (2, "python.exe"), (3, "python.exe"), (4, "maplebot.exe"),
                 (5, "pythonw.exe"), (6, "python.exe"), (7, "notepad.exe"), (8, "python.exe"), (9, "python.exe")]
    lines = {1: r'"C:\venv\Scripts\pythonw.exe" -u C:\repo\scripts\run_active.py --wait-for-start',
             2: None,
             3: r"python -m pytest tests/test_verify_route.py -q",
             4: r'"C:\Maplebot\Maplebot.exe" controller',
             5: r"pythonw.exe -m maplebot.gui",
             6: r"python scripts/watch_circuits.py --minutes 5",
             8: r"python -m maplebot run --cfg x.yaml",
             9: r"python C:\self\scripts\run_active.py"}
    api = FakeApi(processes=processes, command_lines=lines)
    reasons = launcher.foreign_controllers(Path("."), api=api, self_pid=9)
    joined = "\n".join(reasons)
    assert "(pid 1)" in joined and "process 2 could not be inspected" in joined
    assert "(pid 6)" in joined and "(pid 8)" in joined
    for pid in (3, 4, 5, 7, 9):
        assert f"pid {pid})" not in joined and f"process {pid} " not in joined
    assert len(reasons) == 4


@pytest.mark.parametrize("line", [
    "python -m scripts.run_active",
    "python -m scripts.run_active --wait-for-start",
    r'"C:\venv\Scripts\pythonw.exe" -u -m scripts.run_active',
    "python -mscripts.run_active",
    "python -m scripts.verify_potions",
    "python -m scripts.verify_route --x",
    "python -m scripts.verify_lap",
    "python -m scripts.watch_full_circuit",
    "python -m scripts.watch_circuits --minutes 5",
    "python -m SCRIPTS.Run_Active",
    "python run_active.py",                       # started from inside scripts\
    r'"C:\Repo\Scripts\RUN_ACTIVE.PY"',
    "python -u verify_potions.py --x",
    "python -c \"import runpy; runpy.run_path('scripts/run_active.py')\"",
])
def test_controller_command_lines(line):
    assert launcher.command_line_is_controller(line), line


@pytest.mark.parametrize("line", [
    "python tests/test_run_active.py",
    "python -m pytest tests/test_verify_route.py -q",
    "python -m scripts.run_active_report",
    "python -m maplebot.gui",
    "python -m scripts.relay_client",
    "python my_run_active.py",
])
def test_other_command_lines(line):
    assert not launcher.command_line_is_controller(line), line


def test_an_unreadable_command_line_that_raises_still_fails_closed():
    class Raising(FakeApi):
        def command_line(self, pid):
            raise OSError("query failed")

    reasons = launcher.foreign_controllers(Path("."), api=Raising(processes=[(12, "python.exe")]), self_pid=1)
    assert reasons == ["Python process 12 could not be inspected (it may be an elevated Maplebot controller)"]


def test_per_root_mutex_name_resolves_relative_and_dotted_paths(tmp_path):
    (tmp_path / "Maplebot").mkdir()
    (tmp_path / "other").mkdir()
    expected = _formula(str((tmp_path / "Maplebot").resolve()))
    assert launcher.per_root_mutex_name(tmp_path / "Maplebot") == expected
    assert launcher.per_root_mutex_name(Path("Maplebot")) == expected  # the cwd is tmp_path
    assert launcher.per_root_mutex_name(tmp_path / "other" / ".." / "Maplebot") == expected
    assert launcher.per_root_mutex_name(tmp_path / "MAPLEBOT") == expected  # the formula lowercases


def test_process_enumeration_failure_fails_closed():
    api = FakeApi(list_error=OSError("snapshot failed"))
    assert launcher.foreign_controllers(Path("."), api=api) == ["process list unavailable"]


def _signals(root):
    for name in launcher.START_SIGNALS + OTHER_FILES:
        (root / "log" / name).write_text("x")


def test_foreign_reason_refuses_start_and_clears_nothing(root):
    _signals(root)
    api = FakeApi(processes=[(5, "pythonw.exe")])  # unreadable command line
    reason = launcher.controller_alive(root, api=api)
    assert "could not be inspected" in reason
    handle = launcher.start_controller(root, api=api, threaded=False)
    assert handle.poll() == 183 and "could not be inspected" in handle.reason
    assert all((root / "log" / n).exists() for n in launcher.START_SIGNALS + OTHER_FILES)
    assert "start refused" in (root / "log" / "launcher.log").read_text(encoding="utf-8")


def test_alive_start_returns_183_without_clearing(root):
    _signals(root)
    api = FakeApi(mutexes={launcher.per_root_mutex_name(root)})
    handle = launcher.start_controller(root, api=api)
    assert handle.poll() == 183
    assert all((root / "log" / n).exists() for n in launcher.START_SIGNALS)
    assert not any(c[0] == "runas" for c in api.calls)


@pytest.mark.parametrize("code", [0, 1223, 5])
def test_start_clears_exactly_the_start_signals_and_waits_for_the_prompt(root, code):
    _signals(root)
    release = threading.Event()
    seen = []

    def runas(exe, params, cwd, hwnd):
        seen.append((exe, params, cwd, hwnd))
        release.wait(5)
        return code

    api = FakeApi(runas=runas)
    handle = launcher.start_controller(root, wait_for_start=True, hwnd=42, api=api, threaded=True)
    assert handle.poll() is None
    assert not any((root / "log" / n).exists() for n in launcher.START_SIGNALS)
    assert all((root / "log" / n).exists() for n in OTHER_FILES)
    release.set()
    assert handle.wait(5) == code and handle.poll() == code
    exe, params, cwd, hwnd = seen[0]
    assert cwd == str(root) and hwnd == 42 and params.endswith("--wait-for-start")
    log = (root / "log" / "launcher.log").read_text(encoding="utf-8")
    assert f"-> {code}" in log


def test_a_raising_launch_reports_1(root):
    def runas(*args):
        raise OSError("boom")

    handle = launcher.start_controller(root, api=FakeApi(runas=runas), threaded=False)
    assert handle.poll() == 1
    assert "boom" in (root / "log" / "launcher.log").read_text(encoding="utf-8")


def test_start_signals_match_start_active_ps1():
    text = (REPO / "scripts" / "start_active.ps1").read_text(encoding="utf-8")
    first = re.findall(r"'log/([a-z-]+-active)'", text)
    loop = re.search(r"foreach \(\$signal in @\(([^)]*)\)\)", text).group(1)
    names = first + re.findall(r"'([a-z-]+-active)'", loop)
    assert tuple(names) == launcher.START_SIGNALS


def _formula(root_text):
    return "Local\\MaplebotActive_" + hashlib.sha256(root_text.lower().encode()).hexdigest()[:16]


def test_per_root_mutex_matches_run_active_with_a_virtual_file(tmp_path):
    for name in ("plain", "with spaces", "Mixed Case"):
        root = tmp_path / name
        root.mkdir()
        assert not (root / "scripts").exists()
        virtual = Path(str(root / "scripts" / "run_active.py")).resolve().parents[1]
        assert launcher.per_root_mutex_name(root) == _formula(str(virtual))
    text = (REPO / "scripts" / "run_active.py").read_text(encoding="utf-8")
    assert "mutex_name = 'Local\\\\MaplebotActive_' + hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:16]" in text
    assert "ROOT = Path(__file__).resolve().parents[1]" in text


def test_controller_command_frozen(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "maplebot-cli.exe"))
    exe, params = launcher.controller_command(tmp_path, False)
    assert exe == str(tmp_path / "Maplebot.exe") and params == "controller"
    assert launcher.controller_command(tmp_path, True)[1] == "controller --wait-for-start"
    assert launcher.relay_command(tmp_path) == [str(tmp_path / "Maplebot.exe"), "relay"]


def test_controller_command_source(root):
    exe, params = launcher.controller_command(root, False)
    assert exe.lower().endswith("pythonw.exe")
    script = str(root / "scripts" / "run_active.py")
    assert params == f'-u "{script}"'  # the root has a space, so the path is quoted
    assert launcher.controller_command(root, True)[1] == f'-u "{script}" --wait-for-start'
    assert launcher.relay_command(root) == [sys.executable, str(root / "scripts/relay_client.py")]


def test_launch_failure_messages():
    assert launcher.launch_failure_message(1223) == "Start was cancelled at the Windows administrator prompt."
    assert launcher.launch_failure_message(183, "a Maplebot controller is running") == (
        "Another Maplebot controller is already running (a Maplebot controller is running). "
        "Stop it before starting again.")
    assert launcher.launch_failure_message(1) == (
        "The launch did not complete. Try Start active and approve the Windows prompt.")


def test_shell_execute_uses_noasync_only():
    info = launcher.build_execute_info(r"C:\M\Maplebot.exe", "controller", r"C:\M", 0)
    assert info.fMask == 0x100 and info.lpVerb == "runas" and info.nShow == 0
    assert info.lpDirectory == r"C:\M" and info.cbSize == ctypes.sizeof(type(info))
    recorded = {}

    class Fake:
        def CoInitializeEx(self, *args):
            return 0

        def CoUninitialize(self):
            recorded["uninit"] = True

        def ShellExecuteExW(self, pointer):
            structure = pointer._obj
            recorded.update(mask=structure.fMask, verb=structure.lpVerb, file=structure.lpFile,
                            params=structure.lpParameters, show=structure.nShow)
            return 1

    # The real class with fake DLLs: nothing reaches Windows.
    api = REAL_WIN32API(dll_loader=lambda name: Fake())
    assert api.shell_execute_runas(r"C:\M\Maplebot.exe", "controller", r"C:\M", 0) == 0
    assert recorded == {"mask": 0x100, "verb": "runas", "file": r"C:\M\Maplebot.exe", "params": "controller",
                        "show": 0, "uninit": True}


# Captured at import, before the autouse guard replaces launcher.Win32Api with the fake.
REAL_WIN32API = launcher.Win32Api


class _FakeFunction:
    """A DLL export stand-in that accepts argtypes/restype like a ctypes function."""

    def __init__(self, body):
        self.body = body

    def __call__(self, *args):
        return self.body(*args)


class _FakeDll:
    def __init__(self, **bodies):
        self.closed = []
        defaults = {"CloseHandle": lambda handle: self.closed.append(handle) or 1}
        for name, body in {**defaults, **bodies}.items():
            setattr(self, name, _FakeFunction(body))

    def __getattr__(self, name):  # any other export: present, returns 0
        function = _FakeFunction(lambda *args: 0)
        setattr(self, name, function)
        return function


def _real_api(monkeypatch, last_error=0, **dlls):
    monkeypatch.setattr(REAL_WIN32API, "_last_error", staticmethod(lambda: last_error))
    return REAL_WIN32API(dll_loader=lambda name: dlls[name])


@pytest.mark.parametrize("handle, error, held", [
    (0, launcher.ERROR_ACCESS_DENIED, True),   # an elevated owner probed from a normal process
    (0, launcher.ERROR_FILE_NOT_FOUND, False),
    (0, 6, True),                              # any other error fails closed
    (77, 0, True),
])
def test_real_mutex_probe_fails_closed(monkeypatch, handle, error, held):
    kernel32 = _FakeDll(OpenMutexW=lambda access, inherit, name: handle)
    api = _real_api(monkeypatch, error, kernel32=kernel32)
    assert api.mutex_exists(launcher.GLOBAL_MUTEX) is held
    assert kernel32.closed == ([77] if handle else [])


def test_real_empty_process_list_fails_closed(monkeypatch):
    kernel32 = _FakeDll(CreateToolhelp32Snapshot=lambda flags, pid: 900, Process32FirstW=lambda snap, entry: 0)
    api = _real_api(monkeypatch, 18, kernel32=kernel32)
    with pytest.raises(OSError):
        api.list_processes()
    assert kernel32.closed == [900]
    assert launcher.foreign_controllers(Path("."), api=api) == ["process list unavailable"]


def test_real_process_list_reads_entries(monkeypatch):
    remaining = [(41, "Python.EXE"), (42, "notepad.exe")]

    def fill(snapshot, pointer):
        if not remaining:
            return 0
        pid, image = remaining.pop(0)
        pointer._obj.th32ProcessID = pid
        pointer._obj.szExeFile = image
        return 1

    kernel32 = _FakeDll(CreateToolhelp32Snapshot=lambda flags, pid: 900, Process32FirstW=fill, Process32NextW=fill)
    api = _real_api(monkeypatch, kernel32=kernel32)
    assert api.list_processes() == [(41, "python.exe"), (42, "notepad.exe")]


@pytest.mark.parametrize("open_handle, status", [(0, 0), (55, -1073741790), (55, -1073741823)])
def test_real_command_line_failures_return_none(monkeypatch, open_handle, status):
    kernel32 = _FakeDll(OpenProcess=lambda access, inherit, pid: open_handle)
    ntdll = _FakeDll(NtQueryInformationProcess=lambda handle, cls, buffer, size, needed: status)
    api = _real_api(monkeypatch, 5, kernel32=kernel32, ntdll=ntdll)
    assert api.command_line(1234) is None
    assert kernel32.closed == ([55] if open_handle else [])


def test_real_command_line_reads_the_unicode_string(monkeypatch):
    text = r"pythonw.exe -u C:\repo\scripts\run_active.py"
    wide = ctypes.create_unicode_buffer(text)

    class UnicodeString(ctypes.Structure):
        _fields_ = [("Length", ctypes.c_ushort), ("MaximumLength", ctypes.c_ushort), ("Buffer", ctypes.c_void_p)]

    def query(handle, cls, buffer, size, needed):
        assert cls == launcher.PROCESS_COMMAND_LINE_INFORMATION
        value = UnicodeString.from_address(buffer if isinstance(buffer, int) else ctypes.addressof(buffer))
        value.Length = len(text) * 2
        value.MaximumLength = value.Length + 2
        value.Buffer = ctypes.addressof(wide)
        return 0

    kernel32 = _FakeDll(OpenProcess=lambda access, inherit, pid: 55)
    api = _real_api(monkeypatch, kernel32=kernel32, ntdll=_FakeDll(NtQueryInformationProcess=query))
    assert api.command_line(1234) == text and kernel32.closed == [55]
    assert launcher.command_line_is_controller(api.command_line(1234))


def test_request_stop_creates_only_the_stop_file(tmp_path):
    launcher.request_stop(tmp_path)
    assert [p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")] == ["log", "log/stop-active"]


def test_clear_start_signals_order(root):
    _signals(root)
    assert launcher.clear_start_signals(root) == list(launcher.START_SIGNALS)
    assert launcher.clear_start_signals(root) == []
