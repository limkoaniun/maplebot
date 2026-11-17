"""Single entry point for the standalone Maplebot programs (and ``python -m maplebot.app``).

Maplebot.exe (windowed):   gui (default) | controller [--wait-for-start] | relay
maplebot-cli.exe (console): start [--wait-for-start] | stop | status | paths [--probe] |
                            selftest [--strict] [--report PATH] | supervise ... |
                            restore-example [name] | import-settings <folder> [--force] | version
Source mode accepts every mode.

The controller, relay and supervisor are the repository scripts, executed from their bundled
source with a virtual ``__file__`` = <root>\\scripts\\<name>, so their own ROOT, chdir, log
paths and mutex name resolve to this folder and every safety behaviour is the same code.

Module-level imports stay light: nothing here imports a module that calls get_logger before
the mode has changed into the root (or a temporary folder for selftest).
"""
from __future__ import annotations

import builtins
import importlib
import json
import linecache
import os
import sys
import time
import traceback
import types
from pathlib import Path

from maplebot import first_run, launcher, paths

WINDOWED_MODES = ("gui", "controller", "relay")
CLI_MODES = ("start", "stop", "status", "paths", "selftest", "supervise", "restore-example",
             "import-settings", "version")

EXIT_USAGE, EXIT_NOT_WRITABLE, EXIT_NOT_ELEVATED, EXIT_SHADOWED = 2, 3, 4, 5

USAGE = """usage:
  Maplebot.exe [gui]                      open the settings app
  Maplebot.exe controller [--wait-for-start]   (started by Start active through the Windows prompt)
  Maplebot.exe relay                      (started by the settings app)
  maplebot-cli.exe start [--wait-for-start]
  maplebot-cli.exe stop | status | version
  maplebot-cli.exe paths [--probe]
  maplebot-cli.exe selftest [--strict] [--report PATH]
  maplebot-cli.exe supervise --output session.jsonl --minutes 60 [...]
  maplebot-cli.exe restore-example [construction-site]
  maplebot-cli.exe import-settings <other Maplebot folder> [--force]
"""

NON_ASCII_WARNING = ("Maplebot is installed in a folder whose name contains non-ASCII characters. "
                     "Some image features may not work. Move it to a folder such as C:\\Maplebot.")
IMPORT_GUI_NOTE = ("note: if the settings app was open during the import, its next Save & apply or Start "
                   "active does not save its old form: it reloads the imported settings for you to "
                   "review first. Closing and reopening the settings app also loads them.")
ZONE_WARNING = ("Windows marked these files as downloaded. Close Maplebot, right-click the zip, "
                "choose Properties > Unblock, and extract it again.")


def _say(text: str) -> None:
    stream = sys.stdout
    if stream is not None:
        try:
            print(text, file=stream, flush=True)
        except (OSError, ValueError):
            pass


def _message_box(text: str, flags: int = 0x40) -> None:
    """A blocking Windows message box (0x10 error, 0x40 information); silent if unavailable."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, str(text), "Maplebot", int(flags))
    except (AttributeError, OSError):
        pass


def _redirect_stdio(path: Path) -> None:
    if sys.stdout is not None and sys.stderr is not None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream


def _exit_code(code) -> int:
    if code is None:
        return 0
    if isinstance(code, bool):
        return int(code)
    if isinstance(code, int):
        return code
    _say(str(code))
    return 1


def run_script(name, args, *, root=None, run_name="__main__") -> None:
    """Execute a bundled script as ``__main__`` with __file__ = <root>/scripts/<name>."""
    source = paths.script_source(name)
    text = source.read_text(encoding="utf-8")
    root = Path(root) if root is not None else paths.app_root()
    if paths.is_frozen() or os.path.normcase(str(root)) != os.path.normcase(str(paths.app_root())):
        virtual = str(root / "scripts" / name)
    else:
        virtual = str(source)
    code = compile(text, virtual, "exec", dont_inherit=True)
    linecache.cache[virtual] = (len(text), None, text.splitlines(True), virtual)
    module = types.ModuleType(run_name)
    module.__file__ = virtual
    module.__builtins__ = builtins
    module.__loader__ = None
    module.__spec__ = None
    module.__package__ = None
    saved_main = sys.modules.get(run_name)
    saved_argv = sys.argv[:]
    sys.modules[run_name] = module
    sys.argv = [virtual, *[str(a) for a in args]]
    try:
        exec(code, module.__dict__)
    finally:
        sys.argv = saved_argv
        if saved_main is None:
            sys.modules.pop(run_name, None)
        else:
            sys.modules[run_name] = saved_main


def _enter_root() -> Path:
    root = paths.app_root()
    os.chdir(root)
    return root


def _shadow_guard(root: Path) -> list[str]:
    """Importable files in a frozen install folder would shadow bundled modules (sys.path[0])."""
    if not paths.is_frozen():
        return []  # a source checkout is the code itself
    return first_run.shadow_hazards(root, frozen=True)


# --- windowed modes -------------------------------------------------------------------------------

def _startup_warnings(root: Path) -> None:
    shown = []
    if not first_run.path_supported(root) or not first_run.is_ascii_path(root):
        shown.append(NON_ASCII_WARNING)
    if paths.is_frozen():
        exe = Path(sys.executable)
        if first_run.has_zone_identifier(exe):
            shown.append(ZONE_WARNING)
        manifest = first_run.read_install_manifest(root)
        if manifest:
            changed = first_run.modified_program_files(root, manifest)
            if changed:
                _say("warning: local edits to " + ", ".join(changed)
                     + " are replaced on upgrade; put overrides in config/user.yaml")
    for text in shown:
        _say("warning: " + text)
        _message_box(text, 0x40)


def mode_gui(args) -> int:
    if args:
        return _usage_error(f"gui takes no arguments: {' '.join(args)}")
    root = _enter_root()
    if not first_run.check_writable(root):
        _message_box(f"Maplebot cannot write to {root}. Move the Maplebot folder to a user-writable location "
                     "such as %LOCALAPPDATA%\\Programs\\Maplebot (not Program Files or OneDrive).", 0x10)
        return EXIT_NOT_WRITABLE
    _redirect_stdio(root / "log" / "gui.log")
    first_run.ensure_user_layout(root)
    _startup_warnings(root)
    gui = importlib.import_module("maplebot.gui")
    gui.main()
    return 0


def mode_controller(args) -> int:
    if args not in ([], ["--wait-for-start"]):
        return _usage_error("controller accepts only --wait-for-start")
    root = _enter_root()
    api = launcher.Win32Api()
    handle, error = api.create_mutex(launcher.GLOBAL_MUTEX)
    if not handle or error == launcher.ERROR_ALREADY_EXISTS:
        launcher.append_log(root, "controller: a controller is already running; duplicate launch ignored")
        if handle:
            api.close_handle(handle)
        return 0
    try:
        if paths.is_frozen() and not api.is_elevated():
            launcher.append_log(root, "controller refused: not elevated (approve the Windows prompt)")
            return EXIT_NOT_ELEVATED
        reasons = launcher.foreign_controllers(root, api=api)
        if reasons:
            launcher.append_log(root, "controller refused: " + "; ".join(reasons))
            return 0
        hazards = _shadow_guard(root)
        if hazards:
            launcher.append_log(root, "controller refused: importable files in the install folder: "
                                + ", ".join(hazards))
            return EXIT_SHADOWED
        (root / "log").mkdir(exist_ok=True)
        first_run.ensure_user_layout(root)
        run_script("run_active.py", args, root=root)
        return 0
    except SystemExit as stop:
        return _exit_code(stop.code)
    except BaseException:  # noqa: BLE001 - recorded for the user, reported as failure
        launcher.append_log(root, "controller failed:\n" + traceback.format_exc())
        return 1
    finally:
        api.close_handle(handle)


def mode_relay(args) -> int:
    if args:
        return _usage_error("relay takes no arguments")
    root = paths.app_root()
    hazards = _shadow_guard(root)
    if hazards:
        launcher.append_log(root, "relay refused: importable files in the install folder: " + ", ".join(hazards),
                            name="relay-helper.log")
        return EXIT_SHADOWED
    os.chdir(root)
    _redirect_stdio(root / "log" / "relay-helper.log")
    try:
        run_script("relay_client.py", [], root=root)
        return 0
    except SystemExit as stop:
        return _exit_code(stop.code)
    except BaseException:  # noqa: BLE001
        launcher.append_log(root, "relay failed:\n" + traceback.format_exc(), name="relay-helper.log")
        return 1


# --- console modes --------------------------------------------------------------------------------

def mode_supervise(args) -> int:
    root = paths.app_root()
    hazards = _shadow_guard(root)
    if hazards:
        _say("supervise refused: importable files in the install folder: " + ", ".join(hazards))
        return EXIT_SHADOWED
    os.chdir(root)
    args = list(args)
    if not any(a == "--root" or a.startswith("--root=") for a in args):
        args = ["--root", str(root), *args]
    try:
        run_script("supervise_farming.py", args, root=root)
        return 0
    except SystemExit as stop:
        return _exit_code(stop.code)


def mode_start(args) -> int:
    if args not in ([], ["--wait-for-start"]):
        return _usage_error("start accepts only --wait-for-start")
    root = _enter_root()
    if not first_run.check_writable(root):
        _say(f"Maplebot cannot write to {root}. Move it to a user-writable folder.")
        return EXIT_NOT_WRITABLE
    first_run.ensure_user_layout(root)
    handle = launcher.start_controller(root, wait_for_start=bool(args), hwnd=0, threaded=False)
    code = handle.poll()
    if code == 0:
        _say("Windows prompt approved; controller launched.")
        return 0
    _say(launcher.launch_failure_message(code, handle.reason))
    return 3 if code == launcher.ERROR_ALREADY_EXISTS else 1


def mode_stop(args) -> int:
    if args:
        return _usage_error("stop takes no arguments")
    root = _enter_root()
    path = launcher.request_stop(root)
    _say(f"Stop requested ({path}).")
    return 0


def mode_status(args) -> int:
    if args:
        return _usage_error("status takes no arguments")
    root = _enter_root()
    verdict = launcher.controller_alive(root) or "not running"
    status = launcher.read_status(root)
    age = None
    if status is not None:
        try:
            age = round(time.time() - float(status.get("updated")), 1)
        except (TypeError, ValueError):
            age = None
    _say(json.dumps({"verdict": verdict, "status_age_s": age, "status": status}, indent=2, default=str))
    return 0


def mode_paths(args) -> int:
    if args not in ([], ["--probe"]):
        return _usage_error("paths accepts only --probe")
    root = _enter_root()
    report = {
        "frozen": paths.is_frozen(),
        "exe": sys.executable,
        "app_root": str(root),
        "resource_dir": str(paths.resource_dir()),
        "default_config": str(paths.default_config_path(root)),
        "user_config": str(paths.user_config_path(root)),
        "profiles_dir": str(paths.profiles_dir(root)),
        "log_dir": str(paths.log_dir(root)),
        "scripts": {name: paths.script_source(name).is_file() for name in paths.BUNDLED_SCRIPTS},
        "path_supported": first_run.path_supported(root),
        "ascii_path": first_run.is_ascii_path(root),
        "shadow_hazards": first_run.shadow_hazards(root, frozen=paths.is_frozen()),
        "writable": first_run.check_writable(root) if args == ["--probe"] else "not probed",
    }
    _say(json.dumps(report, indent=2))
    return 0


def mode_restore_example(args) -> int:
    if len(args) > 1:
        return _usage_error("restore-example takes at most one example name")
    name = args[0] if args else "construction-site"
    root = _enter_root()
    first_run.ensure_user_layout(root)
    try:
        copied = first_run.restore_example(root, name, alive=lambda: launcher.controller_alive(root))
    except (OSError, ValueError) as error:
        _say(f"restore-example refused: {error}")
        return 1
    for path in copied:
        _say(f"copied {path}")
    return 0


def mode_import_settings(args) -> int:
    force = "--force" in args
    rest = [a for a in args if a != "--force"]
    if len(rest) != 1 or rest[0].startswith("-"):
        return _usage_error("import-settings needs exactly one source folder (and optionally --force)")
    source = Path(os.path.abspath(Path(rest[0]).expanduser()))  # relative to the caller's folder
    root = _enter_root()
    try:
        result = first_run.import_settings(
            root, source, force=force,
            alive=lambda: launcher.any_controller_alive([root, source]))
    except (OSError, ValueError) as error:
        _say(f"import-settings refused: {error}")
        return 1
    for path in result["backups"]:
        _say(f"kept a backup at {path}")
    for path in result["copied"]:
        _say(f"copied {path}")
    for path in result["unchanged"]:
        _say(f"already identical {path}")
    for warning in result["warnings"]:
        _say(f"warning: {warning}")
    _say(IMPORT_GUI_NOTE)
    return 0


def _version_text() -> str:
    try:
        info = importlib.import_module("maplebot._build_info")
    except ImportError:
        import maplebot
        return f"Maplebot {maplebot.__version__} (source)"
    dirty = "+dirty" if getattr(info, "DIRTY", False) else ""
    return f"Maplebot {info.VERSION} ({info.COMMIT}{dirty})"


def mode_version(args) -> int:
    _say(_version_text())
    return 0


def mode_selftest(args) -> int:
    import shutil
    import tempfile
    strict = False
    report = None
    rest = list(args)
    while rest:
        item = rest.pop(0)
        if item == "--strict":
            strict = True
        elif item == "--report" and rest:
            report = os.path.abspath(rest.pop(0))
        else:
            return _usage_error(f"selftest: unexpected argument {item}")
    previous = os.getcwd()
    temporary = tempfile.mkdtemp(prefix="maplebot-selftest-")
    os.chdir(temporary)
    try:
        selftest = importlib.import_module("maplebot.selftest")
        return selftest.run(strict=strict, report=report)
    finally:
        log_module = sys.modules.get("maplebot.log")
        for logger in list(getattr(log_module, "_LOGGERS", {}).values()):
            for handler in list(logger.handlers):
                try:
                    handler.close()
                except Exception:  # noqa: BLE001
                    pass
                logger.removeHandler(handler)
        if log_module is not None:
            log_module._LOGGERS.clear()
        os.chdir(previous)
        shutil.rmtree(temporary, ignore_errors=True)


MODES = {
    "gui": mode_gui, "controller": mode_controller, "relay": mode_relay,
    "start": mode_start, "stop": mode_stop, "status": mode_status, "paths": mode_paths,
    "selftest": mode_selftest, "supervise": mode_supervise, "restore-example": mode_restore_example,
    "import-settings": mode_import_settings, "version": mode_version,
}


def _allowed_modes(exe: str) -> tuple[str, ...]:
    if exe == "maplebot.exe":
        return WINDOWED_MODES
    if exe == "maplebot-cli.exe":
        return CLI_MODES
    if exe == "":
        return WINDOWED_MODES + CLI_MODES
    return ()


def _usage_error(detail: str) -> int:
    if paths.exe_name() == "maplebot.exe":
        launcher.append_log(paths.app_root(), "usage error: " + detail)
    else:
        _say(detail)
        _say(USAGE)
    return EXIT_USAGE


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    exe = paths.exe_name()
    if argv and argv[0] in ("-h", "--help", "help"):
        _say(USAGE)
        return 0
    if argv:
        mode, args = argv[0], argv[1:]
    else:
        mode, args = ("gui" if exe != "maplebot-cli.exe" else None), []
    if mode == "--version":
        mode = "version"
    if mode is None:
        return _usage_error("choose a mode")
    if mode not in _allowed_modes(exe):
        return _usage_error(f"mode {mode!r} is not available in {exe or 'source mode'}")
    return MODES[mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
