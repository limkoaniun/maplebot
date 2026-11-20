"""Shared offline guards for the packaging tests (not a conftest, so collection is unchanged).

Each packaging test module installs these through its own autouse fixture:
- launcher.Win32Api is replaced by FakeApi: no mutex is created and no UAC prompt is raised
  unless a test installs a recorder; probes report "nothing running" by default;
- maplebot.gui's subprocess.Popen raises, so no relay or script can start;
- app._message_box raises, so no window opens;
- the working directory is tmp_path.
"""
from __future__ import annotations

import subprocess
import types


class FakeApi:
    def __init__(self, *, mutexes=(), processes=(), command_lines=None, states=None, elevated=True,
                 runas=None, create=None, list_error=None):
        self.mutexes = set(mutexes)
        self.processes = list(processes)
        self.command_lines = dict(command_lines or {})
        self.states = dict(states or {})
        self.elevated = elevated
        self.runas = runas
        self.create = create
        self.list_error = list_error
        self.calls = []
        self.closed = []

    def mutex_exists(self, name):
        self.calls.append(("mutex_exists", name))
        return name in self.mutexes

    def create_mutex(self, name):
        self.calls.append(("create_mutex", name))
        if self.create is None:
            raise AssertionError("create_mutex is not allowed in this test")
        return self.create(name)

    def close_handle(self, handle):
        self.closed.append(handle)

    def process_state(self, pid):
        self.calls.append(("process_state", pid))
        return self.states.get(pid, ("dead", None))

    def list_processes(self):
        if self.list_error is not None:
            raise self.list_error
        return list(self.processes)

    def command_line(self, pid):
        return self.command_lines.get(pid)

    def is_elevated(self):
        return self.elevated

    def shell_execute_runas(self, exe, params, cwd, hwnd=0):
        self.calls.append(("runas", exe, params, cwd, hwnd))
        if self.runas is None:
            raise AssertionError("shell_execute_runas is not allowed in this test")
        return self.runas(exe, params, cwd, hwnd)


def install(monkeypatch, tmp_path, default_api=None):
    from maplebot import app, gui, launcher

    api = default_api or FakeApi()
    monkeypatch.setattr(launcher, "Win32Api", lambda *a, **k: api)

    def no_popen(*args, **kwargs):
        raise AssertionError(f"subprocess.Popen is not allowed in packaging tests: {args!r}")

    # Replace gui's module reference only: the tests themselves may still run git or a
    # selftest subprocess through the real subprocess module.
    monkeypatch.setattr(gui, "subprocess", types.SimpleNamespace(
        Popen=no_popen, DEVNULL=subprocess.DEVNULL, CREATE_NO_WINDOW=getattr(subprocess, "CREATE_NO_WINDOW", 0)))

    def no_box(*args, **kwargs):
        raise AssertionError(f"a message box would open: {args!r}")

    monkeypatch.setattr(app, "_message_box", no_box)
    monkeypatch.chdir(tmp_path)
    return api
