"""Build the standalone Maplebot folder and zip (developer tool; run by build_exe.ps1).

Run with the SEPARATE build venv's python, never the live venv. Steps:
guards -> repo snapshot -> stage -> PyInstaller (isolated) -> copy program data -> manifest ->
audit -> strict selftest from a temp cwd -> version check -> optional signing -> zip ->
repo snapshot again. The build never starts gui/controller/relay/start/supervise, never sends
input and never writes into the repository.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _sibling(name):
    """Load a sibling tool by path (packaging/ is not a package and is never put on sys.path)."""
    spec = importlib.util.spec_from_file_location(f"maplebot_packaging_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit_dist = _sibling("audit_dist")
stage_mod = _sibling("stage")

DATA_FILES = ("config/default.yaml", "config/user.example.yaml")
DATA_DIRS = ("monster", "assets")
EXAMPLE_DIR = "examples/construction-site"
ISOLATION_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONNOUSERSITE")


class BuildError(RuntimeError):
    pass


def _inside(child, parent) -> bool:
    child, parent = Path(os.path.realpath(child)), Path(os.path.realpath(parent))
    return child == parent or parent in child.parents


def guards(repo: Path, build_root: Path) -> None:
    live = Path(os.environ.get("USERPROFILE", "")) / ".venvs" / "maplebot"
    if _inside(sys.prefix, live):
        raise BuildError("refusing to build with the live venv; use packaging\\build_exe.ps1")
    if not _inside(sys.prefix, build_root):
        raise BuildError(f"the build interpreter ({sys.prefix}) must be the venv under {build_root}")
    if _inside(build_root, repo):
        raise BuildError("the build root must be outside the repository")
    if importlib.util.find_spec("maplebot") is not None:
        raise BuildError("maplebot is importable in the build venv; the build must use the staged copy only")


def _git(repo, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_snapshot(repo: Path) -> dict:
    tracked_config = [p for p in _git(repo, "ls-files", "-z", "--", "config").split("\0") if p]
    untracked = {}
    for path in sorted((repo / "config").rglob("*")):
        relative = path.relative_to(repo).as_posix()
        if path.is_file() and relative not in tracked_config:
            stat = path.stat()
            untracked[relative] = (stat.st_size, stat.st_mtime_ns)
    log = repo / "log"
    return {
        "porcelain": _git(repo, "status", "--porcelain"),
        "tracked_config": {p: _sha256(repo / p) for p in tracked_config if (repo / p).is_file()},
        "untracked_config": untracked,
        "maplebot_logs": sorted(p.name for p in log.glob("maplebot_*.log")) if log.is_dir() else [],
    }


def compare_snapshots(before: dict, after: dict) -> list[str]:
    problems = []
    for key in ("porcelain", "tracked_config", "maplebot_logs"):
        if before[key] != after[key]:
            problems.append(f"the repository changed during the build ({key})")
    if before["untracked_config"] != after["untracked_config"]:
        problems.append("config changed during the build (the live session may have saved settings); "
                        "rerun while it is idle")
    return problems


def isolated_env(extra=None) -> dict:
    env = {k: v for k, v in os.environ.items() if k.upper() not in ISOLATION_ENV}
    env.update(extra or {})
    return env


def pyinstaller_command(python, dist: Path, work: Path, stage_dir: Path) -> list[str]:
    """PyInstaller in isolated mode (-I: no user site, no PYTHON* variables, no cwd on sys.path)."""
    return [str(python), "-I", "-m", "PyInstaller", "--noconfirm", "--clean",
            "--distpath", str(dist), "--workpath", str(work / "pyi"),
            str(stage_dir / "packaging" / "maplebot.spec")]


def tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*")):
        relative = path.relative_to(folder).as_posix()
        digest.update(relative.encode() + b"\0")
        if path.is_file():
            digest.update(_sha256(path).encode())
    return digest.hexdigest()


def zip_name(version: str, commit: str, dirty: bool, public: bool) -> str:
    return f"Maplebot-{version}-{commit}{'+dirty' if dirty else ''}{'' if public else '-personal'}-win64.zip"


def write_zip(app: Path, destination: Path) -> None:
    """One top-level Maplebot\\ folder, so extracting into %LOCALAPPDATA%\\Programs\\ cannot nest."""
    pending = destination.with_name(destination.name + ".tmp")
    with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(app.rglob("*")):
            if path.is_file():
                archive.write(path, (Path("Maplebot") / path.relative_to(app)).as_posix())
    os.replace(pending, destination)


def installed_packages() -> dict:
    from importlib import metadata
    return {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"]}


def copy_program_data(stage: Path, app: Path, example_included: bool) -> None:
    for relative in DATA_FILES:
        target = app / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stage / relative, target)
    for relative in DATA_DIRS + ((EXAMPLE_DIR,) if example_included else ()):
        if (stage / relative).is_dir():
            shutil.copytree(stage / relative, app / relative)
    shutil.copyfile(stage / "packaging" / "README-FIRST.txt", app / "README-FIRST.txt")


def run(args) -> int:
    repo = Path(args.repo).resolve()
    build_root = Path(args.build_root).resolve()
    guards(repo, build_root)
    before = repo_snapshot(repo)
    stage_dir, dist, work = build_root / "stage", build_root / "dist", build_root / "work"
    manifest = stage_mod.stage(repo, stage_dir, allow_dirty=args.allow_dirty, no_example=args.no_example,
                               public=args.public, extra_tokens=args.deny_token)
    print(f"staged {manifest['version']} ({manifest['commit']}{'+dirty' if manifest['dirty'] else ''})")
    modules = json.loads((stage_dir / "packaging" / "build-modules.json").read_text(encoding="utf-8"))
    if "maplebot._build_info" not in modules["maplebot_modules"]:
        raise BuildError("build-modules.json lacks maplebot._build_info")

    for folder in (dist, work):
        if folder.exists():
            shutil.rmtree(folder)
    work.mkdir(parents=True)
    command = pyinstaller_command(sys.executable, dist, work, stage_dir)
    env = isolated_env({"MAPLEBOT_STAGE": str(stage_dir), "MAPLEBOT_KEEP_FFMPEG": "1" if args.keep_ffmpeg else "0"})
    subprocess.run(command, cwd=str(work), env=env, check=True)

    app = dist / "Maplebot"
    copy_program_data(stage_dir, app, manifest["example_included"])
    manifest.update(python=platform.python_version(), packages=installed_packages())
    (app / "maplebot-install.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    tokens = stage_mod.deny_tokens(args.deny_token, repo=repo)
    errors, warnings = audit_dist.audit(app, stage=stage_dir, public=args.public, tokens=tokens,
                                        build_root=build_root)
    for line in warnings:
        print(f"audit warning: {line}")
    if errors:
        raise BuildError("audit failed:\n  " + "\n  ".join(errors))

    before_tree = tree_hash(app)
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    selftest_env = isolated_env({"PATH": f"{system_root}\\System32;{system_root}"})
    report = build_root / "selftest.json"
    with tempfile.TemporaryDirectory(prefix="maplebot-build-selftest-") as cwd:
        result = subprocess.run([str(app / "maplebot-cli.exe"), "selftest", "--strict", "--report", str(report)],
                                cwd=cwd, env=selftest_env, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise BuildError(f"strict selftest failed ({result.returncode}); see {report}\n{result.stderr}")
        version = subprocess.run([str(app / "maplebot-cli.exe"), "version"], cwd=cwd, env=selftest_env,
                                 capture_output=True, text=True)
    if manifest["commit"] not in version.stdout:
        raise BuildError(f"version output lacks the commit: {version.stdout!r}")
    if tree_hash(app) != before_tree:
        raise BuildError("the dist tree changed while the selftest ran")

    if args.sign_thumbprint:
        for exe in audit_dist.EXES:
            subprocess.run(["signtool", "sign", "/fd", "sha256", "/tr", args.timestamp_url, "/td", "sha256",
                            "/sha1", args.sign_thumbprint, str(app / exe)], check=True)

    if not args.skip_zip:
        name = zip_name(manifest["version"], manifest["commit"], manifest["dirty"], manifest["public"])
        destination = build_root / name
        write_zip(app, destination)
        (build_root / "SHA256SUMS.txt").write_text(f"{_sha256(destination)}  {name}\n", encoding="utf-8")
        (build_root / "build-requirements.txt").write_text(
            "\n".join(f"{k}=={v}" for k, v in sorted(manifest["packages"].items())) + "\n", encoding="utf-8")
        print(f"wrote {destination}")

    problems = compare_snapshots(before, repo_snapshot(repo))
    if problems:
        raise BuildError("; ".join(problems))
    print(f"build finished {datetime.datetime.now().isoformat(timespec='seconds')}: {app}")
    return 0


def parse(argv=None):
    parser = argparse.ArgumentParser(description="Build the standalone Maplebot programs")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--build-root", default=str(Path(os.environ.get("LOCALAPPDATA", ".")) / "maplebot-build"))
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--keep-ffmpeg", action="store_true")
    parser.add_argument("--no-example", action="store_true")
    parser.add_argument("--public", action="store_true", help="implies --no-example and a strict privacy scan")
    parser.add_argument("--deny-token", action="append", default=[])
    parser.add_argument("--skip-zip", action="store_true")
    parser.add_argument("--sign-thumbprint")
    parser.add_argument("--timestamp-url", default="http://timestamp.digicert.com")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    try:
        return run(args)
    except (BuildError, stage_mod.StageError, subprocess.CalledProcessError) as error:
        print(f"build failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
