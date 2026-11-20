"""Audit a built dist\\Maplebot folder before it is zipped (developer tool).

Fails on: private or runtime files in the tree, wrong bundled scripts, manifest hash
mismatches, wrong PE subsystems, a manifest that is not asInvoker / longPathAware or that
declares DPI awareness, importable files beside the exes, and an oversized tree. A --public
build also fails on private tokens in the bundled scripts and in PYZ code paths; a personal
build only warns about them. Runs in the build venv; it does not import maplebot.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.machinery
import json
import marshal
import struct
import sys
from pathlib import Path

EXES = ("Maplebot.exe", "maplebot-cli.exe")
EXPECTED_SUBSYSTEM = {"Maplebot.exe": 2, "maplebot-cli.exe": 3}  # GUI, console
SCRIPTS = ("relay_client.py", "run_active.py", "supervise_farming.py")
TOP_LEVEL_EXTRA = ("README-FIRST.txt", "maplebot-install.json")
FORBIDDEN_PATTERNS = ("config/user.yaml", "config/user.yaml.*", "*/map_profiles", "*/map_profiles/*",
                      "map_profiles", "log", "log/*", "screenshot", "screenshot/*", "video", "video/*",
                      "tests", "tests/*", "*.jsonl", "*.log", "*active-status.json", "*relay-*.png", "*.ps1",
                      "*calibrate_*.py", "*maplebot/_build_info.py", "*maplebot/_build_info.pyc")
# Everything the interpreter imports from a sys.path folder (.py, .pyw, .pyc, .pyd, tagged .pyd)
# plus .pth; a folder holding __init__ with any module suffix is a package.
MODULE_SUFFIXES = tuple(sorted({suffix.lower() for suffix in importlib.machinery.all_suffixes()}))
SHADOW_SUFFIXES = tuple(sorted(set(MODULE_SUFFIXES) | {".pth"}))
SIZE_BUDGET = 220 * 1024 * 1024
LOAD_LIBRARY_AS_DATAFILE = 0x2
RT_MANIFEST = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def forbidden_paths(app: Path) -> list[str]:
    hits = []
    for path in sorted(app.rglob("*")):
        relative = path.relative_to(app).as_posix()
        lowered = relative.lower()
        if any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in FORBIDDEN_PATTERNS):
            hits.append(relative)
    return hits


def shadow_files(app: Path) -> list[str]:
    """Importable files beside the exes, or a scripts\\ folder outside _internal."""
    hits = []
    for entry in sorted(app.iterdir()):
        if entry.is_dir():
            if entry.name.lower() in ("scripts", "maplebot"):
                hits.append(entry.name + "/")
            elif any((entry / ("__init__" + suffix)).exists() for suffix in MODULE_SUFFIXES):
                hits.append(entry.name + "/")
        elif entry.name.lower().endswith(SHADOW_SUFFIXES):
            hits.append(entry.name)
    return hits


def pe_subsystem(data: bytes) -> int:
    """The optional-header Subsystem field of a PE image."""
    if data[:2] != b"MZ":
        raise ValueError("not an MZ image")
    offset = struct.unpack_from("<I", data, 0x3C)[0]
    if data[offset:offset + 4] != b"PE\0\0":
        raise ValueError("no PE signature")
    optional = offset + 24
    return struct.unpack_from("<H", data, optional + 68)[0]


def manifest_problems(text: str) -> list[str]:
    problems = []
    lowered = text.lower()
    if 'level="asinvoker"' not in lowered.replace("'", '"'):
        problems.append('requestedExecutionLevel is not level="asInvoker"')
    if "requireadministrator" in lowered or "highestavailable" in lowered:
        problems.append("the manifest requests elevation")
    if "longpathaware" not in lowered:
        problems.append("the manifest does not declare longPathAware")
    if "dpiaware" in lowered:  # covers dpiAware and dpiAwareness
        problems.append("the manifest declares DPI awareness (pythonw.exe does not)")
    return problems


def read_manifest_resource(exe: Path) -> str:
    import win32api  # build venv only
    handle = win32api.LoadLibraryEx(str(exe), 0, LOAD_LIBRARY_AS_DATAFILE)
    try:
        return win32api.LoadResource(handle, RT_MANIFEST, 1).decode("utf-8", "replace")
    finally:
        win32api.FreeLibrary(handle)


def tree_size(app: Path) -> int:
    return sum(p.stat().st_size for p in app.rglob("*") if p.is_file())


def text_token_hits(folder: Path, tokens) -> list[str]:
    hits = []
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8").lower()
            except (OSError, UnicodeDecodeError):
                continue
            hits.extend(f"{path.name}: {t}" for t in tokens if t in text)
    return hits


def _code_filenames(code, seen=None):
    seen = set() if seen is None else seen
    seen.add(code.co_filename)
    for const in code.co_consts:
        if hasattr(const, "co_filename"):
            _code_filenames(const, seen)
    return seen


def pyz_filenames(exe: Path) -> set[str]:
    """co_filename strings of the code objects in the PYZ embedded in an exe (PyInstaller 6)."""
    from PyInstaller.archive.readers import CArchiveReader  # build venv only
    archive = CArchiveReader(str(exe))
    names = set()
    for entry_name, entry in archive.toc.items():
        typecode = entry[-1] if isinstance(entry, tuple) else None
        if typecode != "z" and not entry_name.lower().endswith(".pyz"):
            continue
        pyz = archive.open_embedded_archive(entry_name)
        for module in pyz.toc:
            value = pyz.extract(module)
            if isinstance(value, (bytes, bytearray)):
                value = marshal.loads(value)
            if hasattr(value, "co_filename"):
                _code_filenames(value, names)
    return names


def audit(app, *, stage=None, public=False, tokens=(), build_root=None) -> tuple[list[str], list[str]]:
    app = Path(app)
    errors, warnings = [], []
    errors += [f"forbidden path in dist: {p}" for p in forbidden_paths(app)]
    errors += [f"importable file beside the exes: {p}" for p in shadow_files(app)]
    for exe in EXES:
        if not (app / exe).is_file():
            errors.append(f"{exe} is missing")
    scripts = app / "_internal" / "scripts"
    present = sorted(p.name for p in scripts.iterdir()) if scripts.is_dir() else []
    if present != sorted(SCRIPTS):
        errors.append(f"_internal/scripts holds {present}, expected {sorted(SCRIPTS)}")
    elif stage is not None:
        for name in SCRIPTS:
            if (scripts / name).read_bytes() != (Path(stage) / "scripts" / name).read_bytes():
                errors.append(f"_internal/scripts/{name} differs from the staged copy")

    manifest_path = app / "maplebot-install.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        manifest = {}
        errors.append(f"maplebot-install.json unreadable: {error}")
    files = manifest.get("files") or {}
    for relative, expected in sorted(files.items()):
        path = app / relative
        if not path.is_file():
            errors.append(f"manifest file missing: {relative}")
        elif _sha256(path) != expected:
            errors.append(f"manifest hash mismatch: {relative}")
    for path in sorted(app.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(app).as_posix()
        if relative.startswith("_internal/") or relative in EXES or relative in TOP_LEVEL_EXTRA:
            continue
        if relative not in files:
            errors.append(f"file beside the exe is not in the manifest: {relative}")
    if "maplebot._build_info" not in (manifest.get("modules") or []):
        errors.append("maplebot._build_info is missing from the manifest module list")
    if bool(manifest.get("example_included")) != (app / "examples" / "construction-site").is_dir():
        errors.append("example_included does not match the examples folder")

    for exe, expected in EXPECTED_SUBSYSTEM.items():
        path = app / exe
        if not path.is_file():
            continue
        try:
            subsystem = pe_subsystem(path.read_bytes()[:4096])
            if subsystem != expected:
                errors.append(f"{exe} subsystem is {subsystem}, expected {expected}")
        except (ValueError, struct.error) as error:
            errors.append(f"{exe}: {error}")
        try:
            errors += [f"{exe}: {p}" for p in manifest_problems(read_manifest_resource(path))]
        except Exception as error:  # noqa: BLE001
            errors.append(f"{exe}: could not read the embedded manifest ({error})")

    size = tree_size(app)
    if size > SIZE_BUDGET:
        errors.append(f"dist is {size / 2**20:.0f} MB, over the {SIZE_BUDGET / 2**20:.0f} MB budget")

    tokens = [t.lower() for t in tokens if t]
    private = []
    if scripts.is_dir():
        private += [f"_internal/scripts {hit}" for hit in text_token_hits(scripts, tokens)]
    path_tokens = tokens + ([str(build_root).lower()] if build_root else [])
    for exe in EXES:
        path = app / exe
        if not path.is_file():
            continue
        try:
            names = pyz_filenames(path)
        except Exception as error:  # noqa: BLE001
            (errors if public else warnings).append(f"{exe}: PYZ code paths not checked ({error})")
            continue
        private += sorted({f"{exe} PYZ path {n}" for n in names for t in path_tokens if t in n.lower()})[:20]
    if private:
        (errors if public else warnings).extend(private)
    return errors, warnings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit dist\\Maplebot")
    parser.add_argument("app")
    parser.add_argument("--stage")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--token", action="append", default=[])
    parser.add_argument("--build-root")
    args = parser.parse_args(argv)
    errors, warnings = audit(args.app, stage=args.stage, public=args.public, tokens=args.token,
                             build_root=args.build_root)
    for line in warnings:
        print(f"warning: {line}")
    for line in errors:
        print(f"error: {line}", file=sys.stderr)
    print("audit " + ("FAILED" if errors else "passed"))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
