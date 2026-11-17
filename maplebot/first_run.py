"""First-run layout, install checks and explicit settings imports for a Maplebot folder.

Everything here works on an explicit ``root`` (the folder holding Maplebot.exe, or the
checkout in source mode). Nothing runs automatically except ``ensure_user_layout``, which
only creates missing folders and an empty-override config/user.yaml and never rewrites a
file. ``restore_example`` and ``import_settings`` are explicit user actions that refuse to
overwrite existing settings and refuse while a controller runs.

Imports are limited to the standard library and yaml: no maplebot module, no get_logger.
"""
from __future__ import annotations

import datetime
import hashlib
import importlib.machinery
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

USER_SEED = "# Local Maplebot settings. Only keys set here override config/default.yaml.\n{}\n"
EXAMPLE_FILES = {"construction-site": ("user.yaml",
                                       "kerning-city-construction-site-7bcaa4fc6a.json",
                                       "kerning-city-construction-site-7bcaa4fc6a.png")}
INSTALL_MANIFEST = "maplebot-install.json"
# Every suffix this interpreter imports from a sys.path folder (.py, .pyw, .pyc, .pyd and the
# tagged .cp313-win_amd64.pyd), plus .pth. A folder holding __init__ with any module suffix is
# a package.
MODULE_SUFFIXES = tuple(sorted({suffix.lower() for suffix in importlib.machinery.all_suffixes()}))
SHADOW_SUFFIXES = tuple(sorted(set(MODULE_SUFFIXES) | {".pth"}))
RESTORE_TMP = "user.yaml.restore.tmp"
IMPORT_TMP_SUFFIX = ".import.tmp"


def ensure_user_layout(root) -> list[str]:
    """Create config/, config/map_profiles/, log/ and an empty-override user.yaml if missing."""
    root = Path(root)
    created = []
    for folder in (root / "config", root / "config" / "map_profiles", root / "log"):
        if not folder.is_dir():
            folder.mkdir(parents=True, exist_ok=True)
            created.append(str(folder))
    user = root / "config" / "user.yaml"
    if not user.exists():
        try:
            with open(user, "x", encoding="utf-8", newline="\n") as handle:
                handle.write(USER_SEED)
            created.append(str(user))
        except FileExistsError:
            pass
    return created


def check_writable(root) -> bool:
    """Probe config/ and log/ with an exclusive-create file that is removed again."""
    root = Path(root)
    try:
        for folder in (root / "config", root / "log"):
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / f".write-probe-{os.getpid()}"
            with open(probe, "xb") as handle:
                handle.write(b"ok")
            probe.unlink()
    except OSError:
        return False
    return True


def _encode_ansi(text: str) -> bytes:
    return text.encode("mbcs", "strict")


def path_supported(root) -> bool:
    """True when the resolved root fits the active ANSI code page (OpenCV's narrow paths)."""
    try:
        _encode_ansi(str(Path(root).resolve()))
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def is_ascii_path(root) -> bool:
    return str(Path(root).resolve()).isascii()


def shadow_hazards(root, *, frozen=None) -> list[str]:
    """Top-level entries of root that could shadow a bundled module once root is on sys.path."""
    root = Path(root)
    if frozen is None:
        frozen = bool(getattr(sys, "frozen", False))
    hazards = []
    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return hazards
    for entry in entries:
        name = entry.name.lower()
        try:
            if entry.is_dir():
                if name == "maplebot":
                    hazards.append(entry.name + "\\")
                elif frozen and name == "scripts":
                    hazards.append(entry.name + "\\")
                elif any((entry / ("__init__" + suffix)).exists() for suffix in MODULE_SUFFIXES):
                    hazards.append(entry.name + "\\")
            elif name.endswith(SHADOW_SUFFIXES):
                hazards.append(entry.name)
        except OSError:
            hazards.append(entry.name)
    return hazards


def is_empty_override(path) -> bool:
    """True when the user settings file is absent or holds no settings (None or {})."""
    path = Path(path)
    if not path.exists():
        return True
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return False
    return value is None or value == {}


def _reason(alive):
    if alive is None:
        return None
    return alive()


def restore_example(root, name="construction-site", *, alive=None, source_root=None) -> list[str]:
    """Copy a versioned example into an install whose settings are still empty.

    ``alive`` returns a reason string while a controller runs (launcher.controller_alive).
    ``source_root`` defaults to root (the example ships beside the program files).
    """
    root = Path(root)
    if name not in EXAMPLE_FILES:
        raise ValueError(f"Unknown example {name!r}. Known examples: {', '.join(sorted(EXAMPLE_FILES))}.")
    reason = _reason(alive)
    if reason:
        raise ValueError(f"Stop the running Maplebot controller before loading the example ({reason}).")
    user = root / "config" / "user.yaml"
    if not is_empty_override(user):
        raise ValueError("config/user.yaml already contains settings. The example is only loaded into empty settings.")
    profiles = root / "config" / "map_profiles"
    source = Path(source_root or root) / "examples" / name
    user_name, *profile_names = EXAMPLE_FILES[name]
    existing = [n for n in profile_names if (profiles / n).exists()]
    if existing:
        raise ValueError("The example map profile already exists: " + ", ".join(existing))
    missing = [n for n in (user_name, *profile_names) if not (source / n).is_file()]
    if missing:
        raise ValueError(f"This build does not include the {name} example (missing {', '.join(missing)}).")
    profiles.mkdir(parents=True, exist_ok=True)
    copied = []
    for n in profile_names:
        destination = profiles / n
        with open(source / n, "rb") as reader, open(destination, "xb") as writer:
            shutil.copyfileobj(reader, writer)
        copied.append(str(destination))
    pending = root / "config" / RESTORE_TMP
    pending.write_bytes((source / user_name).read_bytes())
    os.replace(pending, user)
    copied.append(str(user))
    return copied


def has_zone_identifier(path) -> bool:
    """True when the file still carries the downloaded-from-the-internet mark."""
    try:
        with open(str(path) + ":Zone.Identifier", "rb"):
            return True
    except OSError:
        return False


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_install_manifest(root) -> dict | None:
    try:
        value = json.loads((Path(root) / INSTALL_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def modified_program_files(root, manifest) -> list[str]:
    """Shipped files whose hash differs from the manifest, or that are missing."""
    root = Path(root)
    changed = []
    for relative, expected in sorted(((manifest or {}).get("files") or {}).items()):
        try:
            if file_sha256(root / relative) != expected:
                changed.append(relative)
        except OSError:
            changed.append(relative)
    return changed


# --- importing settings from another Maplebot folder (for example a source checkout) --------------

def _same_folder(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return False


def _atomic_write(destination: Path, data: bytes) -> None:
    pending = destination.with_name(destination.name + IMPORT_TMP_SUFFIX)
    try:
        with open(pending, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, destination)
    except BaseException:
        try:
            pending.unlink()
        except OSError:
            pass
        raise


def _backup(path: Path, stamp: str) -> Path:
    """Copy an existing file aside (exclusive create) before it is replaced; never deletes."""
    for index in range(100):
        suffix = f".before-import-{stamp}" + (f"-{index}" if index else "")
        backup = path.with_name(path.name + suffix)
        try:
            with open(path, "rb") as reader, open(backup, "xb") as writer:
                shutil.copyfileobj(reader, writer)
            return backup
        except FileExistsError:
            continue
    raise OSError(f"Could not choose a backup name for {path}")


def _setting(data, dotted):
    node = data
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def import_settings(root, source, *, alive=None, force=False) -> dict:
    """Copy <source>/config/user.yaml and <source>/config/map_profiles/* into root/config/.

    Refuses (ValueError) while ``alive()`` reports a running controller, when the source is
    this folder or has no user.yaml, when a file does not parse, and - unless ``force`` - when
    this folder already has settings or map profiles. Every file is written through a
    temporary file and os.replace. Nothing is deleted: with ``force``, a different existing
    file is first copied to <name>.before-import-<time>. Returns
    {"copied", "unchanged", "backups", "warnings"}.
    """
    root, source = Path(root), Path(source)
    if not source.is_dir():
        raise ValueError(f"{source} is not a folder.")
    if _same_folder(source, root):
        raise ValueError("The source folder is this Maplebot folder; choose the other Maplebot folder.")
    source_user = source / "config" / "user.yaml"
    source_profiles = source / "config" / "map_profiles"
    if not source_user.is_file():
        raise ValueError(f"{source} has no config/user.yaml to import.")
    reason = _reason(alive)
    if reason:
        raise ValueError(f"Stop every running Maplebot controller before importing settings ({reason}).")

    user_bytes = source_user.read_bytes()
    try:
        settings = yaml.safe_load(user_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"{source_user} is not valid YAML: {error}") from None
    if settings is not None and not isinstance(settings, dict):
        raise ValueError(f"{source_user} must contain named settings.")
    payload = []
    warnings = []
    if source_profiles.is_dir():
        for path in sorted(source_profiles.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.name.endswith((".tmp", IMPORT_TMP_SUFFIX)) or ".before-import-" in path.name:
                continue
            data = path.read_bytes()
            if path.suffix.lower() == ".json":
                try:
                    parsed = json.loads(data.decode("utf-8"))
                except (UnicodeDecodeError, ValueError) as error:
                    raise ValueError(f"Map profile {path.name} is not valid JSON: {error}") from None
                if not isinstance(parsed, dict):
                    raise ValueError(f"Map profile {path.name} must contain a JSON object.")
                if not path.with_suffix(".png").is_file():
                    warnings.append(f"Map profile {path.name} has no matching .png; the program cannot load it.")
            payload.append((path.name, data))

    config = root / "config"
    destination_user = config / "user.yaml"
    destination_profiles = config / "map_profiles"
    if not force:
        if not is_empty_override(destination_user):
            raise ValueError("config/user.yaml in this folder already contains settings. "
                             "Use --force to replace it (the current file is kept as a backup).")
        if destination_profiles.is_dir() and any(destination_profiles.iterdir()):
            raise ValueError("config/map_profiles in this folder is not empty. "
                             "Use --force to add the imported profiles (replaced files are kept as backups).")

    destination_profiles.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result = {"copied": [], "unchanged": [], "backups": [], "warnings": warnings}
    # Profiles first, then user.yaml, so the settings never name a profile that is not there yet.
    targets = [(destination_profiles / name, data) for name, data in payload] + [(destination_user, user_bytes)]
    for destination, data in targets:
        if destination.exists():
            if destination.read_bytes() == data:
                result["unchanged"].append(str(destination))
                continue
            if not (destination == destination_user and is_empty_override(destination)):
                result["backups"].append(str(_backup(destination, stamp)))
        _atomic_write(destination, data)
        result["copied"].append(str(destination))

    for dotted in ("player.nametag.path", "player.appearance.path"):
        value = _setting(settings or {}, dotted)
        if isinstance(value, str) and value:
            if os.path.isabs(value):
                warnings.append(f"{dotted} is an absolute path ({value}); prefer a path inside this folder's assets\\.")
            elif not (root / value).is_file():
                warnings.append(f"{dotted} names {value}, which is not in this folder; copy it from {source}.")
    directory = _setting(settings or {}, "map_profile.directory")
    if isinstance(directory, str) and directory:
        warnings.append(f"map_profile.directory is set to {directory}; profiles were copied to config\\map_profiles.")
    return result
