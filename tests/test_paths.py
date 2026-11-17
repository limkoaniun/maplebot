"""maplebot.paths in source mode and with a simulated frozen exe (offline)."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from maplebot import config, gui, paths
from tests.packaging_guard import install

REPO = Path(__file__).resolve().parents[1]


def _same(a, b):
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


def test_source_mode_values_are_unchanged():
    assert not paths.is_frozen()
    assert _same(paths.app_root(), REPO)
    assert paths.resource_dir() == paths.app_root()
    assert gui.ROOT == Path(gui.__file__).resolve().parents[1]
    assert config.PACKAGE_ROOT == os.path.dirname(os.path.dirname(os.path.abspath(config.__file__)))
    assert config.DEFAULT_CONFIG_PATH == os.path.join(config.PACKAGE_ROOT, "config", "default.yaml")
    assert gui.USER_CONFIG == gui.ROOT / "config/user.yaml" and gui.LOG == gui.ROOT / "log"
    assert paths.exe_name() == ""
    assert paths.script_source("run_active.py") == paths.app_root() / "scripts" / "run_active.py"
    assert paths.user_config_path(Path("x")) == Path("x") / "config" / "user.yaml"
    assert paths.profiles_dir(Path("x")) == Path("x") / "config" / "map_profiles"
    assert paths.log_dir(Path("x")) == Path("x") / "log"
    assert paths.default_config_path(Path("x")) == Path("x") / "config" / "default.yaml"


def _frozen_layout(monkeypatch, tmp_path):
    """A simulated exe in one folder, _MEIPASS in another, and the cwd in a third."""
    install, bundle, elsewhere = tmp_path / "install", tmp_path / "bundle", tmp_path / "elsewhere"
    for folder in (install, bundle, elsewhere):
        folder.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(install / "Maplebot.exe"))
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    return install, bundle, elsewhere


def test_frozen_layout_resolves_beside_the_exe(monkeypatch, tmp_path):
    install, bundle, elsewhere = _frozen_layout(monkeypatch, tmp_path)
    assert paths.is_frozen()
    assert _same(paths.app_root(), install) and not _same(paths.app_root(), elsewhere)
    assert paths.resource_dir() == bundle
    assert paths.script_source("run_active.py") == bundle / "scripts" / "run_active.py"
    assert paths.user_config_path() == paths.app_root() / "config" / "user.yaml"
    assert paths.exe_name() == "maplebot.exe"
    with pytest.raises(ValueError):
        paths.script_source("x.py")
    with pytest.raises(ValueError):
        paths.script_source("verify_lap.py")


def test_nothing_is_cached(monkeypatch, tmp_path):
    first = paths.app_root()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "maplebot-cli.exe"))
    assert paths.app_root() != first and paths.exe_name() == "maplebot-cli.exe"


def test_frozen_config_and_gui_import_point_at_the_exe_folder(tmp_path):
    """A subprocess simulates the frozen exe (cwd, exe folder and _MEIPASS all differ), then
    imports maplebot.config and maplebot.gui (no window is created); the repo is untouched."""
    install, bundle, elsewhere = tmp_path / "install", tmp_path / "bundle", tmp_path / "elsewhere"
    for folder in (install, bundle, elsewhere):
        folder.mkdir()
    log = REPO / "log"
    before = sorted(p.name for p in log.glob("maplebot_*.log")) if log.is_dir() else []
    code = ("import sys, json; sys.frozen = True; sys.executable = sys.argv[1] + '\\\\Maplebot.exe'; "
            "sys._MEIPASS = sys.argv[2]; from maplebot import config, gui; "
            "print(json.dumps([config.PACKAGE_ROOT, config.DEFAULT_CONFIG_PATH, str(gui.ROOT), "
            "str(gui.USER_CONFIG), str(gui.LOG)]))")
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    result = subprocess.run([sys.executable, "-c", code, str(install), str(bundle)], cwd=str(elsewhere), env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    import json
    package_root, default, gui_root, user_config, gui_log = json.loads(result.stdout.strip().splitlines()[-1])
    assert _same(package_root, install) and _same(gui_root, install)
    assert _same(default, install / "config" / "default.yaml")
    assert _same(user_config, install / "config" / "user.yaml") and _same(gui_log, install / "log")
    after = sorted(p.name for p in log.glob("maplebot_*.log")) if log.is_dir() else []
    assert after == before
    assert list(install.iterdir()) == [] and list(bundle.iterdir()) == []  # importing creates nothing there


def test_frozen_map_profiles_load_from_the_exe_folder(monkeypatch, tmp_path):
    """bot.build_bot's profile lookup must use the install folder, never _internal or the cwd."""
    from maplebot import bot, map_profiles
    install, bundle, elsewhere = _frozen_layout(monkeypatch, tmp_path)
    seen = {}

    def load_profiles(root, directory=None, **kwargs):
        seen.update(root=Path(root), directory=directory)
        return []

    monkeypatch.setattr(map_profiles, "load_profiles", load_profiles)
    monkeypatch.setattr(map_profiles, "MapProfileRecognizer", lambda profiles, **kwargs: ("recognizer", profiles, kwargs))
    assert bot.load_map_recognizer({"map_profile": {"recognition": {"x": 1}}}) == ("recognizer", [], {"x": 1})
    assert _same(seen["root"], install) and seen["directory"] is None
    assert _same(bot.map_profile_root(), install)
    assert not _same(seen["root"], bundle) and not _same(seen["root"], elsewhere)


def test_paths_module_imports_nothing_heavy():
    source = Path(paths.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert sorted(imports) == ["from __future__ import annotations", "from pathlib import Path", "import sys"]
