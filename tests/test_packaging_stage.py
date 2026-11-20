"""packaging/stage.py, audit_dist.py, build.py and lock_requirements.py pure parts (offline).

The packaging tools are loaded from their files: the folder is not a Python package (an
__init__.py there would shadow the PyPI ``packaging`` library).
"""
from __future__ import annotations

import fnmatch
import importlib.util
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tests.packaging_guard import install

REPO = Path(__file__).resolve().parents[1]
PACKAGING = REPO / "packaging"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"maplebot_packaging_{name}", PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage = _load("stage")
audit_dist = _load("audit_dist")
lock = _load("lock_requirements")


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


def test_packaging_is_not_a_package():
    assert not (PACKAGING / "__init__.py").exists()


def test_select():
    denied_paths = ["config/user.yaml", "config/map_profiles/a.json", "log/active-status.json", "screenshot/x.png",
                    "video/y.mp4", "scripts/calibrate_player.py", "tests/test_x.py", "maplebot/__pycache__/a.pyc",
                    "foo.jsonl", "scripts/start_active.ps1", "maplebot/_build_info.py", "config/user.yaml.tmp"]
    allowed_paths = ["maplebot/gui.py", "scripts/run_active.py", "scripts/relay_client.py",
                     "scripts/supervise_farming.py", "config/default.yaml", "config/user.example.yaml",
                     "monster/snail/snail_0.png", "assets/gain-line-exp-prefix.png",
                     "examples/construction-site/user.yaml", "packaging/maplebot.spec"]
    dropped = ["scripts/verify_lap.py", "docs/x.md", "README.md", "packaging/build.py", "pyproject.toml"]
    allowed, denied = stage.select(denied_paths + allowed_paths + dropped)
    assert allowed == sorted(allowed_paths)
    assert denied == sorted(denied_paths)


def test_real_tracked_selection():
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    allowed, denied = stage.select(stage.tracked_files(REPO))
    for name in ("README.md", "user.yaml", "kerning-city-construction-site-7bcaa4fc6a.json",
                 "kerning-city-construction-site-7bcaa4fc6a.png"):
        assert f"examples/construction-site/{name}" in allowed
    assert "assets/gain-line-exp-prefix.png" in allowed and "assets/gain-line-meso-prefix.png" in allowed
    assert not any(stage.is_denied(p) for p in allowed)
    assert all(p.startswith(("scripts/", "tests/", "config/")) or p.endswith(".ps1") for p in denied), denied
    assert not any(p.startswith(("config/user.yaml", "config/map_profiles", "log/")) for p in allowed)


def test_module_names_come_from_the_file_tree(tmp_path):
    for relative in ("maplebot/__init__.py", "maplebot/__main__.py", "maplebot/app.py", "maplebot/nav/__init__.py",
                     "maplebot/nav/loot.py", "maplebot/__pycache__/app.cpython-313.pyc", "maplebot/_build_info.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise SystemExit('never imported')\n")
    assert stage.module_names(tmp_path) == ["maplebot", "maplebot._build_info", "maplebot.app", "maplebot.nav",
                                            "maplebot.nav.loot"]


def test_script_import_scan_reads_text_only():
    before = set(sys.modules)
    names = set(stage.script_imports(REPO))
    for expected in ("win32api", "win32event", "hashlib", "maplebot.bot", "msvcrt", "argparse"):
        assert expected in names
    assert not any(n.startswith("scripts") for n in names)
    assert "run_active" not in sys.modules and not (set(sys.modules) - before) & {"win32api", "win32event"}


def test_privacy_scan_reports_each_file_and_token(tmp_path):
    for relative, text in (("scripts/run_active.py", "print('Starting for SamplePlayer')\n"),
                           ("monster/octopus/SOURCE.md", "ok\ncopied from /home/someone/refs\n")):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    hits = stage.privacy_scan(tmp_path, ["sampleplayer", "/home/"])
    assert ("scripts/run_active.py", 1, "sampleplayer") in hits
    assert ("monster/octopus/SOURCE.md", 2, "/home/") in hits


def test_private_tokens_come_from_the_ignored_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("USERNAME", "ab")
    assert stage.private_tokens(tmp_path) == [] and stage.deny_tokens(repo=tmp_path) == sorted(stage.BASE_DENY_TOKENS)
    path = tmp_path / stage.PRIVATE_TOKENS_FILE
    path.parent.mkdir(parents=True)
    path.write_text("# comment\nSamplePlayer  # my IGN\n\nab\nsomeuser\n", encoding="utf-8")
    assert stage.private_tokens(tmp_path) == ["SamplePlayer", "someuser"]  # "ab" is too short
    assert {"sampleplayer", "someuser"} <= set(stage.deny_tokens(repo=tmp_path))
    assert "sampleplayer" not in stage.deny_tokens()


def test_repository_ignores_the_private_files():
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    for relative in ("assets/player-nametag.png", "assets/player-head.png", "tests/fixtures/nametags/x.png",
                     stage.PRIVATE_TOKENS_FILE, "scripts/calibrate_player.py"):
        result = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "--no-index", relative])
        assert result.returncode == 0, relative
    tracked = stage.tracked_files(REPO)
    assert not [p for p in tracked if fnmatch.fnmatchcase(p, stage.PERSONAL_ASSET_PATTERN)]
    assert not [p for p in tracked if p.startswith("tests/fixtures/nametags/") and p.endswith(".png")]
    assert stage.PRIVATE_TOKENS_FILE not in tracked


def _commit(repo, message):
    """Commit everything with signing and hooks off, so no gpg/pinentry window or hook can run."""
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.invalid",
           "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={repo / '.no-hooks'}"]
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "--no-verify", "-m", message], check=True)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    files = {
        "pyproject.toml": '[project]\nname = "maplebot"\nversion = "0.1.0"\n',
        "maplebot/__init__.py": "", "maplebot/app.py": "", "maplebot/gui.py": "",
        "scripts/run_active.py": "import win32api\n", "scripts/relay_client.py": "import msvcrt\n",
        "scripts/supervise_farming.py": "import argparse\n", "scripts/verify_lap.py": "",
        "config/default.yaml": "a: 1\n", "config/user.example.yaml": "{}\n",
        "packaging/maplebot.spec": "", "packaging/maplebot_entry.py": "", "packaging/README-FIRST.txt": "x\n",
        "monster/snail/snail.png": "png", "assets/gain.png": "png",
        "examples/construction-site/user.yaml": "player:\n  nametag: {path: assets/player-nametag.png}\n",
        ".gitignore": "config/user.yaml\nconfig/map_profiles/\nlog/\nassets/player-*\npackaging/private-tokens.txt\n",
        # Public builds drop these two (they name private paths in the real repo).
        "assets/README.md": "crops from /home/someone/captures\n", "monster/snail/SOURCE.md": "from /home/someone\n",
    }
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    _commit(repo, "fixture")
    # Private, ignored data that must never reach the stage.
    (repo / "config" / "user.yaml").write_text("keys: {attack: z}\n", encoding="utf-8")
    (repo / "config" / "map_profiles").mkdir()
    (repo / "config" / "map_profiles" / "mine.json").write_text("{}", encoding="utf-8")
    (repo / "log").mkdir()
    (repo / "log" / "active-status.json").write_text("{}", encoding="utf-8")
    # The character's crop and the private tokens are git-ignored like in the real repo; personal
    # builds copy the crop from disk and public builds read the tokens.
    (repo / "assets" / "player-nametag.png").write_bytes(b"\x89PNG private nametag")
    (repo / "assets" / "player-old.png").write_bytes(b"\x89PNG stray crop the example does not name")
    (repo / "packaging" / "private-tokens.txt").write_text("sampleplayer\n", encoding="utf-8")
    return repo


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_stage_from_git_archive_and_dirty_handling(tmp_path):
    repo = _git_repo(tmp_path)
    manifest = stage.stage(repo, tmp_path / "stage")
    staged = sorted(p.relative_to(tmp_path / "stage").as_posix() for p in (tmp_path / "stage").rglob("*") if p.is_file())
    assert "maplebot/_build_info.py" in staged and "packaging/build-modules.json" in staged
    assert not any(p.startswith(("config/user.yaml", "config/map_profiles", "log/")) for p in staged)
    assert "scripts/verify_lap.py" not in staged
    modules = json.loads((tmp_path / "stage" / "packaging" / "build-modules.json").read_text())
    assert "maplebot._build_info" in modules["maplebot_modules"] and "maplebot._build_info" in manifest["modules"]
    assert set(modules["script_imports"]) == {"win32api", "msvcrt", "argparse"}
    assert manifest["dirty"] is False and manifest["example_included"] is True and manifest["public"] is False
    assert "config/default.yaml" in manifest["files"] and "assets/player-nametag.png" in manifest["files"]
    assert (tmp_path / "stage" / "assets" / "player-nametag.png").read_bytes() == b"\x89PNG private nametag"
    assert not (tmp_path / "stage" / "assets" / "player-old.png").exists()  # only what the example names
    assert stage.dirty_paths(repo) == []  # the ignored crop does not make the tree dirty
    info = (tmp_path / "stage" / "maplebot" / "_build_info.py").read_text()
    assert f"COMMIT = '{manifest['commit']}'" in info and "VERSION = '0.1.0'" in info

    lean = stage.stage(repo, tmp_path / "lean", no_example=True)
    assert lean["example_included"] is False
    assert not (tmp_path / "lean" / "examples").exists() and not (tmp_path / "lean" / "assets" / "player-nametag.png").exists()

    (repo / "maplebot" / "gui.py").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(stage.StageError, match="uncommitted"):
        stage.stage(repo, tmp_path / "dirty")
    dirty = stage.stage(repo, tmp_path / "dirty", allow_dirty=True)
    assert dirty["dirty"] is True
    assert (tmp_path / "dirty" / "maplebot" / "gui.py").read_text() == "# edited\n"
    assert (tmp_path / "dirty" / "assets" / "player-nametag.png").is_file()
    assert not (tmp_path / "dirty" / "config" / "user.yaml").exists()
    assert not (tmp_path / "dirty" / "config" / "map_profiles").exists()

    with pytest.raises(stage.StageError, match="outside the repository"):
        stage.stage(repo, repo / "stage-inside")


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_public_stage_fails_on_private_text(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "scripts" / "run_active.py").write_text("print('hello SamplePlayer')\n", encoding="utf-8")
    _commit(repo, "greeting")
    with pytest.raises(stage.StageError, match=r"scripts/run_active.py:1 \(sampleplayer\)"):
        stage.stage(repo, tmp_path / "public", public=True)  # token from the git-ignored file


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_public_stage_refuses_without_private_tokens(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "packaging" / "private-tokens.txt").unlink()
    with pytest.raises(stage.StageError, match="private tokens"):
        stage.stage(repo, tmp_path / "public", public=True)
    (repo / "packaging" / "private-tokens.txt").write_text("﻿# only a comment\nab\n", encoding="utf-8")
    with pytest.raises(stage.StageError, match="private tokens"):
        stage.stage(repo, tmp_path / "public", public=True)
    assert stage.stage(repo, tmp_path / "flag", public=True, extra_tokens=["sampleplayer"])["public"] is True
    assert stage.stage(repo, tmp_path / "personal")["public"] is False  # personal builds do not need it
    (repo / "packaging" / "private-tokens.txt").write_text("﻿sampleplayer\n", encoding="utf-8")
    assert stage.private_tokens(repo) == ["sampleplayer"]  # a BOM does not hide the first token


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_personal_stage_requires_the_local_character_crops(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "assets" / "player-nametag.png").unlink()
    with pytest.raises(stage.StageError, match=r"local character crops \(git-ignored\): assets/player-nametag.png"):
        stage.stage(repo, tmp_path / "personal")
    (repo / "examples" / "construction-site" / "user.yaml").write_text(
        "player:\n  nametag: {path: assets/mine.png}\n", encoding="utf-8")
    _commit(repo, "odd crop name")
    with pytest.raises(stage.StageError, match="must match assets/player-"):
        stage.stage(repo, tmp_path / "odd")
    assert stage.stage(repo, tmp_path / "lean", no_example=True)["example_included"] is False
    assert stage.stage(repo, tmp_path / "public", public=True)["public"] is True


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_public_stage_drops_personal_files_and_honours_deny_tokens(tmp_path):
    repo = _git_repo(tmp_path)
    manifest = stage.stage(repo, tmp_path / "public", public=True)
    public = tmp_path / "public"
    assert manifest["public"] is True and manifest["example_included"] is False  # public implies no example
    assert not (public / "examples").exists()
    assert not (public / "assets" / "player-nametag.png").exists()
    assert not (public / "assets" / "README.md").exists() and not (public / "monster" / "snail" / "SOURCE.md").exists()
    assert (public / "assets" / "gain.png").is_file() and (public / "monster" / "snail" / "snail.png").is_file()
    assert "PUBLIC = True" in (public / "maplebot" / "_build_info.py").read_text()

    (repo / "packaging" / "README-FIRST.txt").write_text("Questions? Ask SecretName.\n", encoding="utf-8")
    _commit(repo, "readme")
    assert stage.stage(repo, tmp_path / "plain", public=True)["public"] is True
    with pytest.raises(stage.StageError, match=r"packaging/README-FIRST.txt:1 \(secretname\)"):
        stage.stage(repo, tmp_path / "denied", public=True, extra_tokens=["SecretName"])


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")
def test_dirty_stage_never_ships_untracked_data_files(tmp_path):
    """Browse... copies a private nametag crop into <checkout>\\assets\\ (not gitignored)."""
    repo = _git_repo(tmp_path)
    (repo / "assets" / "my-nametag.png").write_bytes(b"\x89PNG private")
    (repo / "monster" / "mine").mkdir()
    (repo / "monster" / "mine" / "a.png").write_bytes(b"png")
    (repo / "examples" / "construction-site" / "extra.json").write_text("{}", encoding="utf-8")
    assert stage.dirty_paths(repo) == []  # untracked data files are never staged, so not "dirty"
    clean = stage.stage(repo, tmp_path / "clean")
    assert clean["dirty"] is False and not (tmp_path / "clean" / "assets" / "my-nametag.png").exists()

    (repo / "maplebot" / "new_module.py").write_text("X = 1\n", encoding="utf-8")
    (repo / "packaging" / "hooks").mkdir()
    (repo / "packaging" / "hooks" / "hook-x.py").write_text("", encoding="utf-8")
    assert stage.dirty_paths(repo) == ["maplebot/new_module.py", "packaging/hooks/hook-x.py"]
    dirty = stage.stage(repo, tmp_path / "dirty", allow_dirty=True)
    staged = tmp_path / "dirty"
    assert dirty["dirty"] is True and (staged / "maplebot" / "new_module.py").is_file()
    assert (staged / "packaging" / "hooks" / "hook-x.py").is_file()
    assert "maplebot.new_module" in dirty["modules"]
    for private in ("assets/my-nametag.png", "monster/mine/a.png", "examples/construction-site/extra.json"):
        assert not (staged / private).exists() and private not in dirty["files"]
    assert (staged / "assets" / "gain.png").is_file()  # tracked data still ships

    with pytest.raises(stage.StageError, match="public build must come from a clean"):
        stage.stage(repo, tmp_path / "public-dirty", public=True, allow_dirty=True)
    with pytest.raises(stage.StageError, match="public build must come from a clean"):
        stage.stage(repo, tmp_path / "public-dirty", public=True)


def test_deny_tokens(monkeypatch):
    monkeypatch.setenv("USERNAME", "SomeUser")
    tokens = stage.deny_tokens(["Secret", ""])
    assert "secret" in tokens and "someuser" in tokens
    assert set(stage.BASE_DENY_TOKENS) <= set(tokens) and all(t == t.lower() for t in tokens)
    monkeypatch.setenv("USERNAME", "ab")  # too short to be a useful token
    assert "ab" not in stage.deny_tokens()


@pytest.mark.parametrize("suffix", [".txt", ".yaml", ".yml", ".json", ".md", ".py"])
def test_privacy_scan_is_case_insensitive_across_text_files(tmp_path, suffix):
    (tmp_path / f"note{suffix}").write_text("ok\nsaved under C:\\USERS\\SomeUser\\maplebot\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"C:\\Users\\SomeUser")  # not text: not scanned
    hits = stage.privacy_scan(tmp_path, stage.deny_tokens(["someuser"]))
    assert (f"note{suffix}", 2, "c:\\users\\") in hits and (f"note{suffix}", 2, "someuser") in hits
    assert not any(path == "image.png" for path, _, _ in hits)


def test_version_info_and_build_info_text():
    assert stage.version_tuple("0.1.0") == (0, 1, 0, 0)
    text = stage.version_info_text("0.1.0", "abc1234")
    assert "filevers=(0, 1, 0, 0)" in text and "commit abc1234" in text
    namespace = {}
    exec(stage.build_info_text("0.1.0", "abc1234", True, "2026-01-01T00:00:00Z", False, True), namespace)
    assert namespace["COMMIT"] == "abc1234" and namespace["DIRTY"] is True and namespace["EXAMPLE_INCLUDED"] is True


def _pe(subsystem):
    data = bytearray(512)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<H", data, 0x80 + 24 + 68, subsystem)
    return bytes(data)


GOOD_MANIFEST = ('<assembly><trustInfo><requestedExecutionLevel level="asInvoker" uiAccess="false"/></trustInfo>'
                 '<application><windowsSettings><longPathAware>true</longPathAware></windowsSettings>'
                 '</application></assembly>')
DPI = "the manifest declares DPI awareness (pythonw.exe does not)"
LONG_PATHS = "the manifest does not declare longPathAware"
ELEVATION = "the manifest requests elevation"
INVOKER = 'requestedExecutionLevel is not level="asInvoker"'


@pytest.mark.parametrize("manifest, expected", [
    (GOOD_MANIFEST.replace("</windowsSettings>", "<dpiAware>true/pm</dpiAware></windowsSettings>"), [DPI]),
    (GOOD_MANIFEST.replace("</windowsSettings>", "<dpiAwareness>PerMonitorV2</dpiAwareness></windowsSettings>"), [DPI]),
    (GOOD_MANIFEST.replace("<longPathAware>true</longPathAware>", ""), [LONG_PATHS]),
    (GOOD_MANIFEST.replace("</trustInfo>", "<!-- highestAvailable --></trustInfo>"), [ELEVATION]),
    (GOOD_MANIFEST.replace("</trustInfo>", "<x level='requireAdministrator'/></trustInfo>"), [ELEVATION]),
    (GOOD_MANIFEST.replace('level="asInvoker"', 'level="highestAvailable"'), [INVOKER, ELEVATION]),
    (GOOD_MANIFEST.replace('"asInvoker"', "'asInvoker'"), []),
])
def test_manifest_problems_one_defect_at_a_time(manifest, expected):
    assert audit_dist.manifest_problems(manifest) == expected


def test_audit_shadow_suffixes(tmp_path):
    import importlib.machinery
    assert set(s.lower() for s in importlib.machinery.all_suffixes()) <= set(audit_dist.SHADOW_SUFFIXES)
    app = tmp_path / "Maplebot"
    app.mkdir()
    for name in ("a.pyw", "b.pyd", "c.pth", "d.pyc", "e.cp313-win_amd64.pyd", "notes.txt"):
        (app / name).write_bytes(b"")
    for package, init in (("pkgw", "__init__.pyw"), ("pkgc", "__init__.pyc"), ("pkgd", "__init__.pyd"),
                          ("plain", "readme.txt")):
        (app / package).mkdir()
        (app / package / init).write_bytes(b"")
    assert set(audit_dist.shadow_files(app)) == {"a.pyw", "b.pyd", "c.pth", "d.pyc", "e.cp313-win_amd64.pyd",
                                                "pkgw/", "pkgc/", "pkgd/"}


def _fake_dist(tmp_path, monkeypatch, *, example=False):
    """A dist tree that passes audit(); the exe resource readers are replaced (no build venv)."""
    app = tmp_path / "dist" / "Maplebot"
    stage_dir = tmp_path / "stage"
    for exe, subsystem in audit_dist.EXPECTED_SUBSYSTEM.items():
        (app / exe).parent.mkdir(parents=True, exist_ok=True)
        (app / exe).write_bytes(_pe(subsystem))
    for name in audit_dist.SCRIPTS:
        for folder in (app / "_internal" / "scripts", stage_dir / "scripts"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_text("print('hello')\n", encoding="utf-8")
    (app / "README-FIRST.txt").write_text("x", encoding="utf-8")
    shipped = {"config/default.yaml": "a: 1\n", "assets/gain.png": "png"}
    if example:
        shipped["examples/construction-site/user.yaml"] = "{}\n"
    for relative, text in shipped.items():
        (app / relative).parent.mkdir(parents=True, exist_ok=True)
        (app / relative).write_text(text, encoding="utf-8")
    manifest = {"files": {r: audit_dist._sha256(app / r) for r in shipped}, "modules": ["maplebot._build_info"],
                "example_included": example}
    (app / "maplebot-install.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(audit_dist, "read_manifest_resource", lambda exe: GOOD_MANIFEST)
    monkeypatch.setattr(audit_dist, "pyz_filenames", lambda exe: {"maplebot\\app.py"})
    return app, stage_dir, manifest


def test_audit_passes_a_clean_dist(tmp_path, monkeypatch):
    app, stage_dir, _ = _fake_dist(tmp_path, monkeypatch)
    assert audit_dist.audit(app, stage=stage_dir, public=True, tokens=["sampleplayer"]) == ([], [])
    app, stage_dir, _ = _fake_dist(tmp_path / "with-example", monkeypatch, example=True)
    assert audit_dist.audit(app, stage=stage_dir) == ([], [])


def test_audit_private_text_in_scripts_fails_public_and_warns_personal(tmp_path, monkeypatch):
    app, stage_dir, _ = _fake_dist(tmp_path, monkeypatch)
    for folder in (app / "_internal" / "scripts", stage_dir / "scripts"):
        (folder / "run_active.py").write_text("print('hello SamplePlayer')\n", encoding="utf-8")
    errors, warnings = audit_dist.audit(app, stage=stage_dir, public=True, tokens=["sampleplayer"])
    assert any("_internal/scripts run_active.py: sampleplayer" in e for e in errors) and warnings == []
    errors, warnings = audit_dist.audit(app, stage=stage_dir, public=False, tokens=["sampleplayer"])
    assert errors == [] and any("run_active.py: sampleplayer" in w for w in warnings)


def test_audit_pyz_build_root_paths(tmp_path, monkeypatch):
    app, stage_dir, _ = _fake_dist(tmp_path, monkeypatch)
    monkeypatch.setattr(audit_dist, "pyz_filenames", lambda exe: {r"C:\MBuild\stage\maplebot\app.py"})
    errors, _ = audit_dist.audit(app, stage=stage_dir, public=True, tokens=[], build_root=r"C:\MBuild")
    assert any("PYZ path" in e for e in errors)

    def unreadable(exe):
        raise RuntimeError("no PyInstaller here")

    monkeypatch.setattr(audit_dist, "pyz_filenames", unreadable)
    errors, warnings = audit_dist.audit(app, stage=stage_dir, public=True)
    assert any("PYZ code paths not checked" in e for e in errors)
    errors, warnings = audit_dist.audit(app, stage=stage_dir, public=False)
    assert errors == [] and any("PYZ code paths not checked" in w for w in warnings)


@pytest.mark.parametrize("defect, message", [
    ("backup", "forbidden path in dist: config/user.yaml.before-import-20260101-000000"),
    ("user", "forbidden path in dist: config/user.yaml"),
    ("unlisted", "file beside the exe is not in the manifest: assets/my-nametag.png"),
    ("example", "example_included does not match the examples folder"),
    ("hash", "manifest hash mismatch: config/default.yaml"),
    ("build-info", "maplebot._build_info is missing from the manifest module list"),
    ("script", "_internal/scripts/run_active.py differs from the staged copy"),
    ("shadow", "importable file beside the exes: yaml.pyw"),
    ("dpi", "Maplebot.exe: " + DPI),
    ("subsystem", "maplebot-cli.exe subsystem is 2, expected 3"),
])
def test_audit_failures(tmp_path, monkeypatch, defect, message):
    app, stage_dir, manifest = _fake_dist(tmp_path, monkeypatch)
    if defect == "backup":
        (app / "config" / "user.yaml.before-import-20260101-000000").write_text("{}")
    elif defect == "user":
        (app / "config" / "user.yaml").write_text("{}")
    elif defect == "unlisted":
        (app / "assets" / "my-nametag.png").write_bytes(b"private")
    elif defect == "example":
        (app / "maplebot-install.json").write_text(json.dumps(dict(manifest, example_included=True)))
    elif defect == "hash":
        (app / "config" / "default.yaml").write_text("a: 2\n")
    elif defect == "build-info":
        (app / "maplebot-install.json").write_text(json.dumps(dict(manifest, modules=[])))
    elif defect == "script":
        (app / "_internal" / "scripts" / "run_active.py").write_text("print('changed')\n")
    elif defect == "shadow":
        (app / "yaml.pyw").write_text("")
    elif defect == "dpi":
        monkeypatch.setattr(audit_dist, "read_manifest_resource", lambda exe: GOOD_MANIFEST.replace(
            "</windowsSettings>", "<dpiAware>true</dpiAware></windowsSettings>") if exe.name == "Maplebot.exe"
            else GOOD_MANIFEST)
    elif defect == "subsystem":
        (app / "maplebot-cli.exe").write_bytes(_pe(2))
    errors, _ = audit_dist.audit(app, stage=stage_dir)
    assert message in errors, errors


def test_audit_helpers(tmp_path):
    assert audit_dist.pe_subsystem(_pe(2)) == 2 and audit_dist.pe_subsystem(_pe(3)) == 3
    with pytest.raises(ValueError):
        audit_dist.pe_subsystem(b"not a pe")
    assert audit_dist.manifest_problems(GOOD_MANIFEST) == []

    app = tmp_path / "Maplebot"
    for relative in ("Maplebot.exe", "maplebot-cli.exe", "README-FIRST.txt", "config/default.yaml",
                     "_internal/scripts/run_active.py", "_internal/numpy/_core/tests/test_x.py"):
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert audit_dist.forbidden_paths(app) == [] and audit_dist.shadow_files(app) == []
    (app / "config" / "user.yaml").write_text("{}")
    (app / "config" / "map_profiles").mkdir()
    (app / "log").mkdir()
    (app / "log" / "active.log").write_text("x")
    (app / "evil.py").write_text("")
    (app / "scripts").mkdir()
    forbidden = audit_dist.forbidden_paths(app)
    assert {"config/user.yaml", "config/map_profiles", "log", "log/active.log"} <= set(forbidden)
    assert set(audit_dist.shadow_files(app)) == {"evil.py", "scripts/"}


def test_build_helpers(tmp_path):
    build = _load("build")
    assert build.zip_name("0.1.0", "abc1234", False, False) == "Maplebot-0.1.0-abc1234-personal-win64.zip"
    assert build.zip_name("0.1.0", "abc1234", True, True) == "Maplebot-0.1.0-abc1234+dirty-win64.zip"
    snapshot = {"porcelain": "", "tracked_config": {"config/default.yaml": "1"}, "untracked_config": {},
                "maplebot_logs": ["maplebot_a.log"]}
    assert build.compare_snapshots(snapshot, dict(snapshot)) == []
    changed = dict(snapshot, untracked_config={"config/user.yaml": (1, 2)})
    assert "rerun while it is idle" in build.compare_snapshots(snapshot, changed)[0]
    assert build.compare_snapshots(snapshot, dict(snapshot, maplebot_logs=[]))
    env = build.isolated_env({"X": "1"})
    assert "PYTHONPATH" not in env and env["X"] == "1"
    app = tmp_path / "Maplebot"
    (app / "config").mkdir(parents=True)
    (app / "config" / "default.yaml").write_text("a: 1")
    target = tmp_path / "out.zip"
    build.write_zip(app, target)
    import zipfile
    assert zipfile.ZipFile(target).namelist() == ["Maplebot/config/default.yaml"]


def test_build_isolation_guards(tmp_path, monkeypatch):
    import types
    build = _load("build")
    repo, build_root = tmp_path / "repo", tmp_path / "build"
    (build_root / "venv").mkdir(parents=True)
    repo.mkdir()
    monkeypatch.setattr(build, "sys", types.SimpleNamespace(prefix=str(build_root / "venv")))
    # maplebot is importable from this test interpreter (the repository is on sys.path), so the
    # real find_spec makes the guard refuse.
    with pytest.raises(build.BuildError, match="maplebot is importable"):
        build.guards(repo, build_root)
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda name: None)
    build.guards(repo, build_root)
    monkeypatch.setattr(build, "sys", types.SimpleNamespace(prefix=str(repo / "build" / "venv")))
    with pytest.raises(build.BuildError, match="outside the repository"):
        build.guards(repo, repo / "build")
    monkeypatch.setattr(build, "sys", types.SimpleNamespace(prefix=str(tmp_path / "elsewhere")))
    with pytest.raises(build.BuildError, match="must be the venv"):
        build.guards(repo, build_root)
    live = Path(build.os.environ.get("USERPROFILE", "")) / ".venvs" / "maplebot"
    monkeypatch.setattr(build, "sys", types.SimpleNamespace(prefix=str(live)))
    with pytest.raises(build.BuildError, match="live venv"):
        build.guards(repo, build_root)


def test_build_environment_and_command_are_isolated(monkeypatch, tmp_path):
    build = _load("build")
    for name in build.ISOLATION_ENV:
        monkeypatch.setenv(name, "x")
    env = build.isolated_env({"MAPLEBOT_STAGE": "s"})
    assert not any(k.upper() in build.ISOLATION_ENV for k in env)
    assert {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"} <= set(build.ISOLATION_ENV)
    assert env["MAPLEBOT_STAGE"] == "s"
    command = build.pyinstaller_command("py.exe", tmp_path / "dist", tmp_path / "work", tmp_path / "stage")
    assert command[:4] == ["py.exe", "-I", "-m", "PyInstaller"]
    assert command[-1] == str(tmp_path / "stage" / "packaging" / "maplebot.spec")


def test_build_snapshot_comparison_is_strict_for_tracked_config():
    build = _load("build")
    snapshot = {"porcelain": "", "tracked_config": {"config/default.yaml": "1"}, "untracked_config": {},
                "maplebot_logs": []}
    problems = build.compare_snapshots(snapshot, dict(snapshot, tracked_config={"config/default.yaml": "2"}))
    assert problems == ["the repository changed during the build (tracked_config)"]
    problems = build.compare_snapshots(snapshot, dict(snapshot, porcelain=" M maplebot/gui.py\n"))
    assert problems == ["the repository changed during the build (porcelain)"]


def test_lock_lines(tmp_path):
    (tmp_path / "numpy-2.5.4-cp313-cp313-win_amd64.whl").write_bytes(b"a")
    (tmp_path / "PyYAML-6.0.3-cp313-cp313-win_amd64.whl").write_bytes(b"b")
    lines = lock.lock_lines(tmp_path)
    assert lines[0].startswith("numpy==2.5.4 --hash=sha256:") and lines[1].startswith("pyyaml==6.0.3 --hash=sha256:")


def test_shipped_text_has_no_private_tokens():
    for relative in ("packaging/README-FIRST.txt", "packaging/maplebot.spec", "packaging/maplebot_entry.py"):
        text = (REPO / relative).read_text(encoding="utf-8").lower()
        for token in stage.deny_tokens(repo=REPO):  # includes the local private tokens when present
            assert token not in text, (relative, token)
