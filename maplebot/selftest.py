"""Offline installation self-test for the standalone program and the source checkout.

Read-only: no capture, no input, no keyboard hook, no mutex creation (mutexes are only
opened), no network, and the three bundled scripts are never executed (they are compiled,
their imports are scanned with ast and imported, and key lines of text are pinned).

Run it from a temporary working directory (``maplebot-cli selftest`` does this), because
importing maplebot modules calls get_logger, which creates ./log/maplebot_<stamp>.log.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
import traceback
from pathlib import Path

from maplebot import first_run, launcher, paths

SKIPPED_MODULES = {"maplebot.__main__"}  # the legacy CLI is not part of the program
RUN_ACTIVE_PINS = (
    "ROOT = Path(__file__).resolve().parents[1]",
    "'--wait-for-start' in sys.argv",
    "'Local\\\\MaplebotActive_' + hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:16]",
)
RELAY_PINS = ("ROOT = Path(__file__).resolve().parents[1]",)
SUPERVISE_PINS = ('"--root"',)


def _same(a, b) -> bool:
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


def _under(child, parent) -> bool:
    child = os.path.normcase(os.path.realpath(str(child)))
    parent = os.path.normcase(os.path.realpath(str(parent)))
    return child == parent or child.startswith(parent.rstrip("\\/") + os.sep)


class Checks:
    def __init__(self):
        self.items = []

    def add(self, name, ok, detail="", informational=False):
        self.items.append({"name": name, "ok": bool(ok) or informational, "detail": str(detail),
                           **({"informational": True} if informational else {})})

    def run(self, name, function):
        try:
            result = function()
        except Exception as error:  # noqa: BLE001 - every failure is a reported check
            self.add(name, False, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}")
            return None
        if isinstance(result, tuple):
            ok, detail = result
        else:
            ok, detail = (result is not False), ("" if result in (None, True, False) else result)
        self.add(name, ok, detail)
        return result

    @property
    def ok(self):
        return all(item["ok"] for item in self.items)


def source_module_names(package_dir: Path) -> list[str]:
    names = []
    for path in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_dir.parent).with_suffix("")
        parts = list(relative.parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if name not in SKIPPED_MODULES:
            names.append(name)
    return names


def script_import_names(text: str) -> list[str]:
    """Absolute module names imported anywhere in a script (functions and __main__ included)."""
    names = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module != "__future__":
                names.add(node.module)
    return sorted(names)


def _manifest():
    return first_run.read_install_manifest(paths.app_root()) if paths.is_frozen() else None


def _example_included(root: Path, manifest) -> bool:
    if paths.is_frozen():
        return bool((manifest or {}).get("example_included"))
    return (root / "examples" / "construction-site").is_dir()


def run(strict=False, report=None) -> int:
    checks = Checks()
    root = paths.app_root()
    manifest = _manifest()
    if paths.is_frozen() and manifest is None:
        checks.add("install manifest", False, f"{root / first_run.INSTALL_MANIFEST} is missing or unreadable")

    # 1. Root coherence.
    def coherence():
        from maplebot import bot, config, gui
        problems = []
        if not _same(config.PACKAGE_ROOT, root):
            problems.append(f"config.PACKAGE_ROOT={config.PACKAGE_ROOT}")
        if not _same(gui.ROOT, root):
            problems.append(f"gui.ROOT={gui.ROOT}")
        if not _same(bot.map_profile_root(), root):
            problems.append(f"map profile root={bot.map_profile_root()}")
        if not _under(config.DEFAULT_CONFIG_PATH, root / "config"):
            problems.append(f"DEFAULT_CONFIG_PATH={config.DEFAULT_CONFIG_PATH}")
        if paths.is_frozen():
            if not _same(root, Path(sys.executable).parent):
                problems.append("app_root is not the exe folder")
            if not _same(paths.resource_dir(), getattr(sys, "_MEIPASS", "")):
                problems.append("resource_dir is not _MEIPASS")
        return (not problems), (", ".join(problems) or str(root))
    checks.run("root coherence", coherence)

    # 2. Every maplebot module imports.
    if paths.is_frozen():
        modules = [m for m in (manifest or {}).get("modules") or [] if m not in SKIPPED_MODULES]
        if not modules:
            checks.add("module list", False, "the install manifest lists no modules")
    else:
        modules = source_module_names(Path(__file__).resolve().parent)
    failed = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001
            failed.append(f"{name}: {type(error).__name__}: {error}")
    checks.add("maplebot modules import", not failed and bool(modules),
               "; ".join(failed) or f"{len(modules)} modules")

    # 3. Third-party probes (no capture, no Controller or Listener).
    def numpy_probe():
        import numpy as np
        return bool(np.arange(6).reshape(2, 3).sum() == 15), np.__version__
    checks.run("numpy", numpy_probe)

    def cv2_probe():
        import cv2
        import numpy as np
        image = np.zeros((20, 20, 3), np.uint8)
        image[5:10, 5:10] = 255
        score = cv2.matchTemplate(image, image[4:12, 4:12].copy(), cv2.TM_SQDIFF_NORMED)
        ok, encoded = cv2.imencode(".png", image)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
        return (score.size > 0 and decoded is not None and decoded.shape == image.shape), cv2.__version__
    checks.run("cv2", cv2_probe)

    def yaml_probe():
        import yaml
        return yaml.safe_load("a: [1, 2]") == {"a": [1, 2]}
    checks.run("yaml", yaml_probe)

    def pynput_probe():
        import pynput.keyboard  # noqa: F401
        return "pynput.keyboard._win32" in sys.modules, "backend _win32 loaded"
    checks.run("pynput.keyboard", pynput_probe)

    def pywin32_probe():
        for name in ("win32api", "win32con", "win32event", "win32gui", "pywintypes"):
            importlib.import_module(name)
        return True
    checks.run("pywin32", pywin32_probe)
    checks.run("windows_capture", lambda: importlib.import_module("windows_capture") is not None)

    def mss_probe():
        importlib.import_module("mss")
        importlib.import_module("mss.windows")
        return True
    checks.run("mss", mss_probe)

    def tcl_probe():
        import tkinter
        return True, tkinter.Tcl().eval("info patchlevel")
    checks.run("tkinter", tcl_probe)

    # 4. Config and shipped data.
    example = root / "examples" / "construction-site"
    included = _example_included(root, manifest)

    def config_probe():
        from maplebot.config import deep_merge, load_config, load_yaml
        cfg = load_config(None)
        problems = []
        monsters = root / str(cfg.get("monsters", {}).get("dir", "monster"))
        if not monsters.is_dir():
            problems.append(f"monsters.dir {monsters} missing")
        templates = (cfg.get("evidence") or {}).get("templates") or {}
        if not {"exp", "meso"} <= set(templates):
            problems.append("evidence.templates must name exp and meso")
        for kind, value in templates.items():
            if value and not (root / str(value)).is_file():
                problems.append(f"evidence template {kind} {value} missing")
        notes = []
        if included:
            merged = deep_merge(cfg, load_yaml(str(example / "user.yaml")))
            player = merged.get("player") or {}
            for key in ("nametag", "appearance"):
                value = (player.get(key) or {}).get("path")
                if value and not (root / str(value)).is_file():
                    # A personal build must ship them; a source checkout of the public repository
                    # has no character crops until the user adds their own (they are git-ignored).
                    (problems if paths.is_frozen() else notes).append(f"example player.{key}.path {value} missing")
        elif paths.is_frozen():
            if (root / "examples").exists():
                problems.append("examples\\ is present in a build without the example")
            # Only what the build shipped: a user's own assets\player-* crop (Browse..., or copied after
            # Import settings) is legitimate in any install.
            shipped = (manifest or {}).get("files") or {}
            if any(name.startswith("assets/player-") for name in shipped):
                problems.append("assets\\player-* was shipped in a build without the example")
        detail = "; ".join(problems) or "; ".join([f"example included: {included}"] + notes)
        return (not problems), detail
    checks.run("config and data", config_probe)

    # 5. Images decode from absolute paths (OpenCV takes narrow paths on Windows).
    def image_probe():
        import cv2
        problems = []
        folder = root / "monster"
        for sub in sorted(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("_")):
            pngs = sorted(sub.glob("*.png"))
            if not any(cv2.imread(str(p)) is not None for p in pngs[:3]):
                problems.append(f"{sub.name}: no decodable PNG")
        gain = root / "assets" / "gain-line-exp-prefix.png"
        if cv2.imread(str(gain)) is None:
            problems.append(f"{gain} does not decode")
        if problems and not first_run.path_supported(root):
            problems.insert(0, "install path unsupported by OpenCV")
        return (not problems), ("; ".join(problems) or "monster templates and gain templates decode")
    checks.run("images", image_probe)

    # 6. The example profile loads.
    if included:
        def profile_probe():
            from maplebot.map_profiles import load_profiles
            errors = []
            profiles = load_profiles(root, directory=example, errors=errors)
            return (len(profiles) == 1 and not errors), (f"{len(profiles)} profile(s); " + "; ".join(errors))
        checks.run("example profile", profile_probe)

    # 7. The null key backend.
    def backend_probe():
        from maplebot.control.backend import make_backend
        return make_backend("null") is not None
    checks.run("null key backend", backend_probe)

    # 8. The bundled scripts: compile, import scan, text pins. Never executed.
    pins = {"run_active.py": RUN_ACTIVE_PINS, "relay_client.py": RELAY_PINS,
            "supervise_farming.py": SUPERVISE_PINS}
    for name in paths.BUNDLED_SCRIPTS:
        def script_probe(name=name):
            source = paths.script_source(name)
            text = source.read_text(encoding="utf-8")
            compile(text, str(root / "scripts" / name), "exec", dont_inherit=True)
            problems = []
            for module in script_import_names(text):
                if module.split(".")[0] == "scripts":
                    continue
                try:
                    importlib.import_module(module)
                except Exception as error:  # noqa: BLE001
                    problems.append(f"import {module}: {type(error).__name__}: {error}")
            for pin in pins[name]:
                if pin not in text:
                    problems.append(f"missing pinned text {pin!r}")
            if name != "supervise_farming.py" and text.count("__file__") != 1:
                problems.append(f"expected exactly one __file__, found {text.count('__file__')}")
            return (not problems), ("; ".join(problems) or str(source))
        checks.run(f"script {name}", script_probe)

    # 9. Informational only.
    try:
        alive = launcher.controller_alive(root)
    except Exception as error:  # noqa: BLE001
        alive = f"probe failed: {error}"
    checks.add("controller probe", True, alive or "not running", informational=True)
    checks.add("install path", True,
               f"path_supported={first_run.path_supported(root)} ascii={first_run.is_ascii_path(root)}",
               informational=True)
    checks.add("shadow hazards", True, ", ".join(first_run.shadow_hazards(root)) or "none", informational=True)
    if paths.is_frozen():
        marked = first_run.has_zone_identifier(Path(sys.executable))
        checks.add("downloaded mark", True, "Maplebot is marked as downloaded; Unblock the zip" if marked else "none",
                   informational=True)

    # 10. Build-only strict checks.
    if strict:
        leftovers = [p for p in ("config/user.yaml", "config/map_profiles", "log") if (root / p).exists()]
        checks.add("strict: no user data", not leftovers, ", ".join(leftovers) or "clean")
        changed = first_run.modified_program_files(root, manifest) if manifest else ["(no manifest)"]
        checks.add("strict: manifest hashes", not changed, ", ".join(changed) or "all match")
        scripts_dir = paths.resource_dir() / "scripts"
        present = sorted(p.name for p in scripts_dir.iterdir()) if scripts_dir.is_dir() else []
        checks.add("strict: bundled scripts", present == sorted(paths.BUNDLED_SCRIPTS), ", ".join(present))
        hazards = first_run.shadow_hazards(root, frozen=True)
        checks.add("strict: no shadow files", not hazards, ", ".join(hazards) or "none")

    result = {"ok": checks.ok, "frozen": paths.is_frozen(), "app_root": str(root), "strict": bool(strict),
              "checks": checks.items}
    if report:
        Path(report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for item in checks.items:
        mark = "ok  " if item["ok"] else "FAIL"
        if item.get("informational"):
            mark = "info"
        first_line = item["detail"].splitlines()[0] if item["detail"] else ""
        if sys.stdout is not None:
            print(f"[{mark}] {item['name']}: {first_line}", flush=True)
    if sys.stdout is not None:
        print("selftest " + ("passed" if checks.ok else "FAILED"), flush=True)
    return 0 if checks.ok else 1
