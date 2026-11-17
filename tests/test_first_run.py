"""First-run layout, writability, example restore and install checks (tmp_path roots only)."""
from __future__ import annotations

import builtins
import os
import shutil
from pathlib import Path

import pytest

from maplebot import first_run
from maplebot.config import load_config
from maplebot.map_profiles import read_profile
from tests.packaging_guard import install

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "construction-site"


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


@pytest.fixture
def root(tmp_path):
    folder = tmp_path / "Maplebot"
    folder.mkdir()
    return folder


def _example_root(root):
    target = root / "examples" / "construction-site"
    target.mkdir(parents=True)
    for name in first_run.EXAMPLE_FILES["construction-site"]:
        shutil.copyfile(EXAMPLE / name, target / name)
    return target


def test_layout_creates_folders_and_an_empty_override(root):
    created = first_run.ensure_user_layout(root)
    for relative in ("config", "config/map_profiles", "log"):
        assert (root / relative).is_dir()
    user = root / "config" / "user.yaml"
    assert user.read_text(encoding="utf-8") == first_run.USER_SEED
    assert str(user) in created
    assert load_config(str(user)) == load_config()
    assert first_run.is_empty_override(user)


def test_layout_never_rewrites_a_file(root):
    first_run.ensure_user_layout(root)
    user = root / "config" / "user.yaml"
    stamp = user.stat().st_mtime_ns
    data = user.read_bytes()
    assert first_run.ensure_user_layout(root) == []
    assert user.read_bytes() == data and user.stat().st_mtime_ns == stamp
    user.write_text("keys:\n  attack: z\n", encoding="utf-8")
    first_run.ensure_user_layout(root)
    assert user.read_text(encoding="utf-8") == "keys:\n  attack: z\n"
    assert not first_run.is_empty_override(user)


def test_check_writable(root, monkeypatch):
    assert first_run.check_writable(root)
    assert sorted(p.name for p in (root / "config").iterdir()) == []
    assert sorted(p.name for p in (root / "log").iterdir()) == []
    real_open = builtins.open

    def denied(path, mode="r", *args, **kwargs):
        if "x" in mode:
            raise PermissionError("read-only")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", denied)
    assert not first_run.check_writable(root)


def test_path_supported(root, monkeypatch):
    assert first_run.path_supported(root) is True

    def refuse(text):
        raise UnicodeEncodeError("mbcs", text, 0, 1, "not in the code page")

    monkeypatch.setattr(first_run, "_encode_ansi", refuse)
    assert first_run.path_supported(root) is False


def test_shadow_hazards(root):
    for name in ("config", "log", "monster", "assets", "examples", "_internal"):
        (root / name).mkdir()
    (root / "README-FIRST.txt").write_text("x")
    (root / "Maplebot.exe").write_bytes(b"MZ")
    assert first_run.shadow_hazards(root, frozen=True) == []
    (root / "x.py").write_text("")
    (root / "y.pyd").write_bytes(b"")
    (root / "z.pth").write_text("")
    (root / "w.pyc").write_bytes(b"")
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "maplebot").mkdir()
    (root / "scripts").mkdir()
    hazards = first_run.shadow_hazards(root, frozen=True)
    assert set(hazards) == {"x.py", "y.pyd", "z.pth", "w.pyc", "pkg\\", "maplebot\\", "scripts\\"}
    assert "scripts\\" not in first_run.shadow_hazards(root, frozen=False)


def test_shadow_hazards_cover_every_importable_suffix(root):
    """.pyw and tagged .pyd files import from sys.path[0] on Windows, and so do packages whose
    only __init__ is .pyw, .pyc or .pyd."""
    import importlib.machinery
    for suffix in importlib.machinery.all_suffixes():
        assert suffix.lower() in first_run.SHADOW_SUFFIXES
    (root / "mss.pyw").write_text("")
    (root / "win32event.cp313-win_amd64.pyd").write_bytes(b"")
    (root / "Yaml.PYW").write_text("")
    for package, init in (("pynput", "__init__.pyw"), ("cv2", "__init__.pyc"), ("numpy", "__init__.pyd")):
        (root / package).mkdir()
        (root / package / init).write_bytes(b"")
    (root / "plain").mkdir()  # a folder without __init__ is only a namespace portion: not a hazard
    (root / "plain" / "notes.txt").write_text("x")
    assert set(first_run.shadow_hazards(root, frozen=True)) == {
        "mss.pyw", "win32event.cp313-win_amd64.pyd", "Yaml.PYW", "pynput\\", "cv2\\", "numpy\\"}


def test_shadow_hazards_report_an_entry_that_cannot_be_inspected(root, monkeypatch):
    (root / "locked").mkdir()
    (root / "config").mkdir()
    real_is_dir = Path.is_dir

    def is_dir(self):
        if self.name == "locked":
            raise PermissionError("denied")
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    assert first_run.shadow_hazards(root, frozen=True) == ["locked"]


def test_restore_example_over_the_seed(root):
    _example_root(root)
    first_run.ensure_user_layout(root)
    copied = first_run.restore_example(root, alive=lambda: None)
    assert len(copied) == 3
    for name in first_run.EXAMPLE_FILES["construction-site"][1:]:
        assert (root / "config" / "map_profiles" / name).read_bytes() == (EXAMPLE / name).read_bytes()
    assert (root / "config" / "user.yaml").read_bytes() == (EXAMPLE / "user.yaml").read_bytes()
    profile = read_profile(root / "config" / "map_profiles" / "kerning-city-construction-site-7bcaa4fc6a.json")
    assert profile is not None
    assert not (root / "config" / "user.yaml.tmp").exists()
    assert not (root / "config" / first_run.RESTORE_TMP).exists()


def test_restore_example_never_uses_the_gui_temp_file(root, monkeypatch):
    _example_root(root)
    first_run.ensure_user_layout(root)
    seen = []
    real_replace = os.replace
    monkeypatch.setattr(first_run.os, "replace", lambda a, b: (seen.append(Path(a).name), real_replace(a, b)))
    first_run.restore_example(root)
    assert seen == [first_run.RESTORE_TMP]


@pytest.mark.parametrize("setup", ["settings", "profile", "alive", "unknown", "missing"])
def test_restore_example_refusals_leave_everything_unchanged(root, setup):
    _example_root(root)
    first_run.ensure_user_layout(root)
    user = root / "config" / "user.yaml"
    profiles = root / "config" / "map_profiles"
    name, alive = "construction-site", None
    if setup == "settings":
        user.write_text("keys:\n  attack: z\n", encoding="utf-8")
    elif setup == "profile":
        (profiles / "kerning-city-construction-site-7bcaa4fc6a.png").write_bytes(b"mine")
    elif setup == "alive":
        alive = lambda: "this folder's controller is running"  # noqa: E731
    elif setup == "unknown":
        name = "elsewhere"
    elif setup == "missing":
        (root / "examples" / "construction-site" / "user.yaml").unlink()
    before = {p: p.read_bytes() for p in (root / "config").rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        first_run.restore_example(root, name, alive=alive)
    after = {p: p.read_bytes() for p in (root / "config").rglob("*") if p.is_file()}
    assert after == before


def test_zone_identifier_and_manifest_checks(root):
    exe = root / "Maplebot.exe"
    exe.write_bytes(b"MZ")
    assert first_run.has_zone_identifier(exe) is False
    try:
        with open(str(exe) + ":Zone.Identifier", "w") as stream:
            stream.write("[ZoneTransfer]\nZoneId=3\n")
    except OSError:
        pytest.skip("alternate data streams are not supported here")
    assert first_run.has_zone_identifier(exe) is True

    (root / "config").mkdir()
    (root / "config" / "default.yaml").write_text("a: 1\n", encoding="utf-8")
    (root / "monster").mkdir()
    (root / "monster" / "x.png").write_bytes(b"png")
    manifest = {"files": {"config/default.yaml": first_run.file_sha256(root / "config" / "default.yaml"),
                          "monster/x.png": first_run.file_sha256(root / "monster" / "x.png"),
                          "monster/gone.png": "0" * 64}}
    (root / first_run.INSTALL_MANIFEST).write_text(__import__("json").dumps(manifest), encoding="utf-8")
    assert first_run.read_install_manifest(root) == manifest
    assert first_run.modified_program_files(root, manifest) == ["monster/gone.png"]
    (root / "config" / "default.yaml").write_text("a: 2\n", encoding="utf-8")
    assert first_run.modified_program_files(root, manifest) == ["config/default.yaml", "monster/gone.png"]
    assert first_run.read_install_manifest(root / "nowhere") is None


def test_first_run_imports_no_maplebot_module():
    source = Path(first_run.__file__).read_text(encoding="utf-8")
    assert "import maplebot" not in source and "from maplebot" not in source
