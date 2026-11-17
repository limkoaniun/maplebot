"""The offline selftest, run in a subprocess from a temporary working directory.

A subprocess keeps pynput, windows_capture and the capture modules out of the pytest process
(test_source.py asserts lazy imports). The repository's log/ is compared only by its
maplebot_*.log names, because the live controller may write other files there.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from maplebot import selftest
from tests.packaging_guard import install

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


def _logs():
    log = REPO / "log"
    return sorted(p.name for p in log.glob("maplebot_*.log")) if log.is_dir() else []


def test_selftest_passes_in_source_mode_and_writes_only_its_report(tmp_path):
    work = tmp_path / "cwd"
    work.mkdir()
    report = work / "r.json"
    before = _logs()
    code = f"import sys; from maplebot import selftest; sys.exit(selftest.run(False, {str(report)!r}))"
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONHOME", "PYTHONSTARTUP")}
    env["PYTHONPATH"] = str(REPO)
    result = subprocess.run([sys.executable, "-c", code], cwd=str(work), env=env, capture_output=True,
                            text=True, timeout=300)
    data = json.loads(report.read_text(encoding="utf-8"))
    failed = [c for c in data["checks"] if not c["ok"]]
    assert result.returncode == 0 and failed == [], (result.stdout[-4000:], result.stderr[-4000:], failed)
    names = {c["name"] for c in data["checks"]}
    for expected in ("root coherence", "maplebot modules import", "cv2", "pynput.keyboard", "pywin32",
                     "windows_capture", "mss", "tkinter", "config and data", "images", "example profile",
                     "null key backend", "script run_active.py", "script relay_client.py",
                     "script supervise_farming.py", "controller probe"):
        assert expected in names, expected
    assert _logs() == before
    created = sorted(p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file())
    assert all(p == "r.json" or (p.startswith("log/maplebot_") and p.endswith(".log")) for p in created), created


def test_script_import_scan_and_module_walk():
    text = "import os\nfrom maplebot.bot import x\nfrom . import y\nfrom __future__ import annotations\n" \
           "def f():\n    import msvcrt\nif __name__ == '__main__':\n    import win32api\n"
    assert selftest.script_import_names(text) == ["maplebot.bot", "msvcrt", "os", "win32api"]
    names = selftest.source_module_names(REPO / "maplebot")
    assert "maplebot.gui" in names and "maplebot.nav.loot" in names and "maplebot" in names
    assert "maplebot.__main__" not in names


def test_pins_hold_in_the_current_scripts():
    run_active = (REPO / "scripts" / "run_active.py").read_text(encoding="utf-8")
    for pin in selftest.RUN_ACTIVE_PINS:
        assert pin in run_active
    assert run_active.count("__file__") == 1
    relay = (REPO / "scripts" / "relay_client.py").read_text(encoding="utf-8")
    assert relay.count("__file__") == 1 and selftest.RELAY_PINS[0] in relay
    assert selftest.SUPERVISE_PINS[0] in (REPO / "scripts" / "supervise_farming.py").read_text(encoding="utf-8")


def test_app_selftest_mode_runs_in_a_temporary_folder(monkeypatch, tmp_path):
    from maplebot import app
    seen = {}

    def fake_run(strict=False, report=None):
        seen.update(cwd=Path.cwd(), strict=strict, report=report)
        Path("log").mkdir()
        (Path("log") / "maplebot_x.log").write_text("x")
        return 0

    from maplebot import log as log_module
    monkeypatch.setattr(log_module, "_LOGGERS", {})  # the mode closes logger handlers; keep the suite's
    monkeypatch.setattr(selftest, "run", fake_run)
    start = Path.cwd()
    assert app.main(["selftest", "--strict", "--report", "out.json"]) == 0
    assert seen["strict"] is True and seen["report"] == str(start / "out.json")
    assert seen["cwd"] != start and not seen["cwd"].exists()
    assert Path.cwd() == start
    assert app.main(["selftest", "--bogus"]) == 2
