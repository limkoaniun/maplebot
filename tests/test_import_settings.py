"""import-settings: copy another folder's user.yaml and map profiles into an install (tmp dirs only).

These tests never read or copy the real repository config: both folders are built in tmp_path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maplebot import app, first_run, launcher, paths
from tests.packaging_guard import FakeApi, install

USER_YAML = "# Local Maplebot settings\nkeys:\n  attack: z\nplayer:\n  nametag:\n    path: assets/me.png\n"
PROFILE = {"version": 2, "plan": {"name": "Test Map"}}


@pytest.fixture(autouse=True)
def guards(monkeypatch, tmp_path):
    return install(monkeypatch, tmp_path)


@pytest.fixture
def folders(tmp_path):
    source = tmp_path / "checkout"
    (source / "config" / "map_profiles").mkdir(parents=True)
    (source / "config" / "user.yaml").write_text(USER_YAML, encoding="utf-8")
    (source / "config" / "map_profiles" / "test-map.json").write_text(json.dumps(PROFILE), encoding="utf-8")
    (source / "config" / "map_profiles" / "test-map.png").write_bytes(b"\x89PNG\r\n\x1a\nminimap")
    (source / "config" / "map_profiles" / "half-written.json.tmp").write_text("{", encoding="utf-8")
    target = tmp_path / "Maplebot"
    target.mkdir()
    first_run.ensure_user_layout(target)
    return source, target


def _tree(folder: Path) -> dict:
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_imports_into_an_empty_install(folders):
    source, target = folders
    before_source = _tree(source)
    result = first_run.import_settings(target, source, alive=lambda: None)
    assert (target / "config" / "user.yaml").read_text(encoding="utf-8") == USER_YAML
    assert json.loads((target / "config" / "map_profiles" / "test-map.json").read_text()) == PROFILE
    assert (target / "config" / "map_profiles" / "test-map.png").read_bytes().endswith(b"minimap")
    assert not (target / "config" / "map_profiles" / "half-written.json.tmp").exists()
    assert len(result["copied"]) == 3 and result["backups"] == []
    assert any("assets/me.png" in w for w in result["warnings"])  # the nametag crop is not in the install
    assert _tree(source) == before_source  # the source is only read
    leftovers = [p.name for p in (target / "config").rglob("*") if p.name.endswith(first_run.IMPORT_TMP_SUFFIX)]
    assert leftovers == []


@pytest.mark.parametrize("state", ["settings", "profiles"])
def test_refuses_over_existing_settings_without_force(folders, state):
    source, target = folders
    if state == "settings":
        (target / "config" / "user.yaml").write_text("keys:\n  attack: x\n", encoding="utf-8")
    else:
        (target / "config" / "map_profiles" / "mine.json").write_text("{}", encoding="utf-8")
    before = _tree(target)
    with pytest.raises(ValueError, match="--force"):
        first_run.import_settings(target, source)
    assert _tree(target) == before


def test_force_keeps_backups_and_deletes_nothing(folders):
    source, target = folders
    mine = "keys:\n  attack: x\n"
    (target / "config" / "user.yaml").write_text(mine, encoding="utf-8")
    (target / "config" / "map_profiles" / "mine.json").write_text("{}", encoding="utf-8")
    (target / "config" / "map_profiles" / "test-map.json").write_text('{"old": true}', encoding="utf-8")
    result = first_run.import_settings(target, source, force=True)
    profiles = target / "config" / "map_profiles"
    assert (profiles / "mine.json").read_text() == "{}"  # untouched, never deleted
    assert json.loads((profiles / "test-map.json").read_text()) == PROFILE
    assert len(result["backups"]) == 2
    backups = sorted(Path(p).name for p in result["backups"])
    assert backups[0].startswith("test-map.json.before-import-") and backups[1].startswith("user.yaml.before-import-")
    assert any(Path(p).read_text() == mine for p in result["backups"])
    # A backup is not a *.json file, so it can never be loaded as a profile.
    assert sorted(p.name for p in profiles.glob("*.json")) == ["mine.json", "test-map.json"]
    again = first_run.import_settings(target, source, force=True)
    assert again["copied"] == [] and len(again["unchanged"]) == 3


@pytest.mark.parametrize("problem", ["yaml", "list", "json", "no-user", "same", "missing"])
def test_refuses_bad_sources_without_writing(folders, problem):
    source, target = folders
    if problem == "yaml":
        (source / "config" / "user.yaml").write_text("keys: [unclosed\n", encoding="utf-8")
    elif problem == "list":
        (source / "config" / "user.yaml").write_text("- a\n- b\n", encoding="utf-8")
    elif problem == "json":
        (source / "config" / "map_profiles" / "broken.json").write_text("{nope", encoding="utf-8")
    elif problem == "no-user":
        (source / "config" / "user.yaml").unlink()
    elif problem == "same":
        source = target
    elif problem == "missing":
        source = source / "nowhere"
    before = _tree(target)
    with pytest.raises(ValueError):
        first_run.import_settings(target, source)
    assert _tree(target) == before


def test_corrupt_destination_settings_are_not_treated_as_empty(folders):
    """A hand-edited user.yaml that no longer parses must not be overwritten without --force."""
    source, target = folders
    (target / "config" / "user.yaml").write_text("keys: [unclosed\n", encoding="utf-8")
    assert not first_run.is_empty_override(target / "config" / "user.yaml")
    before = _tree(target)
    with pytest.raises(ValueError, match="--force"):
        first_run.import_settings(target, source)
    assert _tree(target) == before
    result = first_run.import_settings(target, source, force=True)
    assert len(result["backups"]) == 1
    assert Path(result["backups"][0]).read_text(encoding="utf-8") == "keys: [unclosed\n"


@pytest.mark.parametrize("content", ["[1, 2]", '"text"', "null"])
def test_profiles_must_be_json_objects(folders, content):
    source, target = folders
    (source / "config" / "map_profiles" / "odd.json").write_text(content, encoding="utf-8")
    before = _tree(target)
    with pytest.raises(ValueError, match="JSON object"):
        first_run.import_settings(target, source)
    assert _tree(target) == before


def test_source_backups_and_temporary_files_are_not_imported(folders):
    source, target = folders
    profiles = source / "config" / "map_profiles"
    (profiles / "test-map.json.before-import-20260101-000000").write_text("{}", encoding="utf-8")
    (profiles / "test-map.json.import.tmp").write_text("{", encoding="utf-8")
    (profiles / "other.json.tmp").write_text("{", encoding="utf-8")
    first_run.import_settings(target, source)
    assert sorted(p.name for p in (target / "config" / "map_profiles").iterdir()) == ["test-map.json", "test-map.png"]


def test_every_write_is_atomic_and_profiles_come_before_user_yaml(folders, monkeypatch):
    source, target = folders
    replaced = []
    real_replace = first_run.os.replace

    def recording_replace(pending, destination):
        replaced.append((Path(pending).name, Path(destination).name))
        return real_replace(pending, destination)

    monkeypatch.setattr(first_run.os, "replace", recording_replace)
    first_run.import_settings(target, source)
    assert replaced == [("test-map.json.import.tmp", "test-map.json"), ("test-map.png.import.tmp", "test-map.png"),
                        ("user.yaml.import.tmp", "user.yaml")]


def test_a_failed_replace_changes_nothing_and_leaves_no_temporary_file(folders, monkeypatch):
    source, target = folders
    before = _tree(target)

    def failing_replace(pending, destination):
        raise PermissionError("file in use")

    monkeypatch.setattr(first_run.os, "replace", failing_replace)
    with pytest.raises(PermissionError):
        first_run.import_settings(target, source)
    assert _tree(target) == before  # the first write failed: no profile, no temp file, seed intact


def test_refuses_while_a_controller_runs(folders):
    source, target = folders
    before = _tree(target)
    with pytest.raises(ValueError, match="Stop every running"):
        first_run.import_settings(target, source, alive=lambda: "this folder's controller is running")
    assert _tree(target) == before


def test_cli_mode_checks_both_folders_and_the_foreign_scan(folders, monkeypatch, capsys):
    source, target = folders
    api = FakeApi(mutexes={launcher.per_root_mutex_name(source)})
    monkeypatch.setattr(launcher, "Win32Api", lambda *a, **k: api)
    monkeypatch.setattr(paths, "app_root", lambda: target)
    before = _tree(target)
    assert app.main(["import-settings", str(source)]) == 1
    assert "refused" in capsys.readouterr().out and _tree(target) == before

    api.mutexes.clear()
    api.processes = [(4242, "pythonw.exe")]  # an unreadable Python process fails closed
    assert app.main(["import-settings", str(source)]) == 1
    assert "could not be inspected" in capsys.readouterr().out and _tree(target) == before

    api.processes = []
    assert app.main(["import-settings", str(source)]) == 0
    out = capsys.readouterr().out
    assert "copied" in out and (target / "config" / "user.yaml").read_text(encoding="utf-8") == USER_YAML
    assert app.main(["import-settings", str(source)]) == 1  # no longer empty: needs --force
    assert app.main(["import-settings", str(source), "--force"]) == 0
    assert app.main(["import-settings"]) == 2
    assert not any(call[0] in ("runas", "create_mutex") for call in api.calls)


def test_relative_source_is_resolved_against_the_callers_folder(folders, monkeypatch, tmp_path):
    source, target = folders
    monkeypatch.setattr(paths, "app_root", lambda: target)
    monkeypatch.chdir(tmp_path)
    assert app.main(["import-settings", "checkout"]) == 0
    assert (target / "config" / "user.yaml").read_text(encoding="utf-8") == USER_YAML


def test_private_settings_are_never_staged():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "maplebot_stage_for_import_test", Path(__file__).resolve().parents[1] / "packaging" / "stage.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    allowed, denied = stage.select(["config/user.yaml", "config/map_profiles/a.json", "config/map_profiles/a.png",
                                    "config/user.yaml.before-import-20260101-000000",
                                    "config/map_profiles/a.json.before-import-x", "config/default.yaml"])
    assert allowed == ["config/default.yaml"]
    assert len(denied) == 5
