"""Stage the files that go into the standalone Maplebot build (developer tool).

Pure functions plus a small CLI. Run only by packaging/build.py in the separate build venv;
nothing here imports maplebot, and nothing is ever written into the repository.

A clean tree is staged from ``git archive HEAD`` (so a concurrent editor cannot race the
copy); ``allow_dirty`` stages tracked files plus untracked-but-not-ignored files under
maplebot/ and packaging/ from the working tree and stamps the build +dirty (a public build
refuses a dirty tree). Private data (config/user.yaml, config/map_profiles/, log/, ...)
is denied by pattern on top of the allowlist and asserted absent again after the copy.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ALLOW_PREFIXES = ("maplebot/", "monster/", "assets/", "examples/construction-site/", "packaging/hooks/")
ALLOW_FILES = ("scripts/run_active.py", "scripts/relay_client.py", "scripts/supervise_farming.py",
               "config/default.yaml", "config/user.example.yaml",
               "packaging/maplebot.spec", "packaging/maplebot_entry.py", "packaging/README-FIRST.txt",
               "packaging/maplebot.ico")
DENY_PATTERNS = ("config/user.yaml", "config/user.yaml.*", "config/map_profiles", "config/map_profiles/*",
                 "log/*", "screenshot/*", "video/*", "tests/*", "*/__pycache__/*", "__pycache__/*",
                 "*.pyc", "*.jsonl", "*.log", "*.egg-info/*", ".git/*", "*.ps1", "*.tmp", "*.before-import-*",
                 "scripts/calibrate_player.py", "maplebot/_build_info.py", "*active-status.json", "*relay-*.png")
SCRIPTS = ("run_active.py", "relay_client.py", "supervise_farming.py")
EXAMPLE_PREFIX = "examples/construction-site/"
PERSONAL_ASSET_PATTERN = "assets/player-*"
MANIFEST_PREFIXES = ("monster/", "assets/", "examples/construction-site/")
# An --allow-dirty build takes untracked files only from these source folders. Data folders
# (assets/, monster/, examples/) ship tracked files only: the settings app's Browse... copies a
# private nametag crop into <checkout>\assets\, which .gitignore does not cover.
UNTRACKED_SOURCE_PREFIXES = ("maplebot/", "packaging/")
MANIFEST_FILES = ("config/default.yaml", "config/user.example.yaml")
TEXT_SUFFIXES = (".py", ".md", ".yaml", ".yml", ".json", ".txt")
BASE_DENY_TOKENS = ("<user>", "<user>", "player", "/home/", "c:\\users\\")


class StageError(RuntimeError):
    pass


def _posix(path) -> str:
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def is_denied(path: str) -> bool:
    lowered = _posix(path).lower()
    return any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in DENY_PATTERNS)


def is_allowed(path: str) -> bool:
    path = _posix(path)
    return path in ALLOW_FILES or any(path.startswith(prefix) for prefix in ALLOW_PREFIXES)


def select(paths) -> tuple[list[str], list[str]]:
    """(allowed, denied). A path that is neither allowlisted nor denied is dropped silently."""
    allowed, denied = [], []
    for raw in paths:
        path = _posix(raw)
        if is_denied(path):
            denied.append(path)
        elif is_allowed(path):
            allowed.append(path)
    return sorted(set(allowed)), sorted(set(denied))


def _git(repo, *args, binary=False):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise StageError(f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', 'replace').strip()}")
    return result.stdout if binary else result.stdout.decode("utf-8")


def tracked_files(repo, include_untracked=False) -> list[str]:
    args = ["ls-files", "-z"] + (["-co", "--exclude-standard"] if include_untracked else [])
    return [p for p in _git(repo, *args).split("\0") if p]


def dirty_paths(repo) -> list[str]:
    roots = ["maplebot", "monster", "assets", "examples/construction-site", "scripts", "config", "packaging"]
    out = _git(repo, "status", "--porcelain", "--untracked-files=all", "--", *roots)
    dirty = []
    for line in out.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if line.startswith("??") and not path.startswith(UNTRACKED_SOURCE_PREFIXES):
            continue  # never staged (see stageable_dirty_files), so it does not make the build dirty
        if is_allowed(path) and not is_denied(path):
            dirty.append(path)
    return dirty


def stageable_dirty_files(repo) -> list[str]:
    """Working-tree paths for an --allow-dirty stage: every tracked file, plus untracked
    (not ignored) files under UNTRACKED_SOURCE_PREFIXES only."""
    tracked = set(tracked_files(repo))
    untracked = [p for p in tracked_files(repo, include_untracked=True)
                 if p not in tracked and _posix(p).startswith(UNTRACKED_SOURCE_PREFIXES)]
    return sorted(tracked | set(untracked))


def project_version(repo) -> str:
    import tomllib
    data = tomllib.loads((Path(repo) / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def module_names(stage_root) -> list[str]:
    """Dotted maplebot module names from the staged file tree (nothing is imported)."""
    package = Path(stage_root) / "maplebot"
    names = []
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        parts = list(path.relative_to(Path(stage_root)).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name != "maplebot.__main__":
            names.append(name)
    return sorted(names)


def script_imports_from_text(text: str) -> list[str]:
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and node.module != "__future__":
            names.add(node.module)
    return sorted(n for n in names if n.split(".")[0] != "scripts")


def script_imports(stage_root) -> list[str]:
    names = set()
    for name in SCRIPTS:
        names.update(script_imports_from_text((Path(stage_root) / "scripts" / name).read_text(encoding="utf-8")))
    return sorted(names)


def deny_tokens(extra=()) -> list[str]:
    tokens = list(BASE_DENY_TOKENS)
    user = os.environ.get("USERNAME", "")
    if len(user) >= 3:
        tokens.append(user)
    tokens.extend(t for t in extra if t)
    return sorted({t.lower() for t in tokens})


def privacy_scan(root, tokens) -> list[tuple[str, int, str]]:
    """(relative path, line number, token) for each private token in a staged text file."""
    root = Path(root)
    tokens = [t.lower() for t in tokens if t]
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            lowered = line.lower()
            for token in tokens:
                if token in lowered:
                    hits.append((path.relative_to(root).as_posix(), number, token))
    return hits


def version_tuple(version: str) -> tuple[int, int, int, int]:
    numbers = [int(n) for n in re.findall(r"\d+", version)[:4]]
    return tuple((numbers + [0, 0, 0, 0])[:4])


def version_info_text(version: str, commit: str) -> str:
    v = version_tuple(version)
    return f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(filevers={v}, prodvers={v}, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Maplebot'),
      StringStruct('FileDescription', 'Maplebot'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('ProductName', 'Maplebot'),
      StringStruct('ProductVersion', '{version}'),
      StringStruct('Comments', 'commit {commit}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def build_info_text(version, commit, dirty, built_utc, public, example_included) -> str:
    return ("\"\"\"Written by packaging/stage.py; never committed.\"\"\"\n"
            f"VERSION = {version!r}\nCOMMIT = {commit!r}\nDIRTY = {bool(dirty)!r}\n"
            f"BUILT_UTC = {built_utc!r}\nPUBLIC = {bool(public)!r}\nEXAMPLE_INCLUDED = {bool(example_included)!r}\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _drop(paths, *, no_example, public) -> list[str]:
    kept = []
    for path in paths:
        if no_example and (path.startswith(EXAMPLE_PREFIX) or fnmatch.fnmatchcase(path, PERSONAL_ASSET_PATTERN)):
            continue
        if public and (path == "assets/README.md" or fnmatch.fnmatchcase(path, "monster/*/SOURCE.md")):
            continue
        kept.append(path)
    return kept


def _inside(child: Path, parent: Path) -> bool:
    child, parent = Path(os.path.realpath(child)), Path(os.path.realpath(parent))
    return child == parent or parent in child.parents


def stage(repo, dest, *, allow_dirty=False, no_example=False, public=False, extra_tokens=(), now=None) -> dict:
    repo, dest = Path(repo).resolve(), Path(dest)
    if _inside(dest, repo):
        raise StageError(f"the stage folder {dest} must be outside the repository")
    no_example = bool(no_example or public)
    dirty = dirty_paths(repo)
    if dirty and public:
        raise StageError("a public build must come from a clean, committed tree (--allow-dirty is not "
                         "accepted with --public): " + ", ".join(dirty[:20]))
    if dirty and not allow_dirty:
        raise StageError("the working tree has uncommitted changes in staged paths (commit them, or pass "
                         "--allow-dirty, which stamps +dirty): " + ", ".join(dirty[:20]))
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    if dirty and allow_dirty:
        allowed, _ = select(stageable_dirty_files(repo))
        allowed = _drop(allowed, no_example=no_example, public=public)
        for relative in allowed:
            source = repo / relative
            if source.is_file():
                target = dest / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    else:
        archive = tarfile.open(fileobj=io.BytesIO(_git(repo, "archive", "--format=tar", "HEAD", binary=True)))
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        allowed, _ = select(members)
        allowed = _drop(allowed, no_example=no_example, public=public)
        for relative in allowed:
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(members[relative]) as reader, open(target, "wb") as writer:
                shutil.copyfileobj(reader, writer)

    staged = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    leaked = [p for p in staged if is_denied(p)]
    if leaked:
        raise StageError("denied files reached the stage: " + ", ".join(leaked))
    missing = [f"scripts/{n}" for n in SCRIPTS if not (dest / "scripts" / n).is_file()]
    missing += [p for p in ("config/default.yaml", "packaging/maplebot.spec", "packaging/maplebot_entry.py",
                            "packaging/README-FIRST.txt", "maplebot/app.py") if not (dest / p).is_file()]
    if missing:
        raise StageError("required files are missing from the stage: " + ", ".join(missing))
    if no_example and "player" in (dest / "config" / "default.yaml").read_text(encoding="utf-8").lower():
        raise StageError("config/default.yaml references the personal example assets")

    version = project_version(repo)
    commit = _git(repo, "rev-parse", "--short", "HEAD").strip()
    built_utc = (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    example_included = (dest / "examples" / "construction-site").is_dir()
    # Order matters: _build_info first, so the module list below contains it.
    (dest / "maplebot" / "_build_info.py").write_text(
        build_info_text(version, commit, bool(dirty), built_utc, public, example_included), encoding="utf-8")
    (dest / "packaging").mkdir(exist_ok=True)
    (dest / "packaging" / "version_info.txt").write_text(version_info_text(version, commit), encoding="utf-8")
    modules = module_names(dest)
    if "maplebot._build_info" not in modules:
        raise StageError("maplebot._build_info is missing from the module list")
    (dest / "packaging" / "build-modules.json").write_text(
        json.dumps({"maplebot_modules": modules, "script_imports": script_imports(dest)}, indent=2),
        encoding="utf-8")

    if public:
        hits = privacy_scan(dest, deny_tokens(extra_tokens))
        if hits:
            raise StageError("a public build would ship private text: "
                             + "; ".join(f"{p}:{n} ({t})" for p, n, t in hits[:20]))

    files = {}
    for relative in staged:
        if relative in MANIFEST_FILES or relative.startswith(MANIFEST_PREFIXES):
            files[relative] = _sha256(dest / relative)
    return {"version": version, "commit": commit, "dirty": bool(dirty), "public": bool(public),
            "example_included": example_included, "built_utc": built_utc, "modules": modules, "files": files}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Stage the Maplebot build tree (no build).")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--no-example", action="store_true")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--deny-token", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        manifest = stage(args.repo, args.dest, allow_dirty=args.allow_dirty, no_example=args.no_example,
                         public=args.public, extra_tokens=args.deny_token)
    except StageError as error:
        print(f"stage failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
