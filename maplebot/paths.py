"""Where Maplebot's program files and user data live, in source mode and as a frozen exe.

Source mode: the root is the repository checkout (the parent of this package), exactly as
``Path(__file__).resolve().parents[1]`` in gui.py and the scripts.

Frozen (PyInstaller onedir): the root is the folder that holds Maplebot.exe; the bundled
runtime and the three script sources live in ``sys._MEIPASS`` (``<root>\\_internal``).

This module imports only sys and pathlib. It must never import maplebot.log,
maplebot.config or anything else that calls get_logger, and nothing here is cached, so
tests can monkeypatch sys.frozen, sys.executable and sys._MEIPASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

BUNDLED_SCRIPTS = ("run_active.py", "relay_client.py", "supervise_farming.py")


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_dir() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return app_root()


def script_source(name: str) -> Path:
    if name not in BUNDLED_SCRIPTS:
        raise ValueError(f"{name!r} is not a bundled Maplebot script")
    return resource_dir() / "scripts" / name


def _root(root) -> Path:
    return Path(root) if root is not None else app_root()


def default_config_path(root=None) -> Path:
    return _root(root) / "config" / "default.yaml"


def user_config_path(root=None) -> Path:
    return _root(root) / "config" / "user.yaml"


def profiles_dir(root=None) -> Path:
    return _root(root) / "config" / "map_profiles"


def log_dir(root=None) -> Path:
    return _root(root) / "log"


def exe_name() -> str:
    return Path(sys.executable).name.lower() if is_frozen() else ""
