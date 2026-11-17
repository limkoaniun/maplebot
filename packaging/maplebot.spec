# -*- mode: python -*-
# PyInstaller spec for the standalone Maplebot programs. Run ONLY through packaging/build.py,
# against the staged tree (MAPLEBOT_STAGE), never against the live repository.
#
# Two EXEs share one _internal (one COLLECT):
#   Maplebot.exe      windowed: gui (default) | controller | relay
#   maplebot-cli.exe  console:  start | stop | status | paths | selftest | supervise | ...
# Both are asInvoker (uac_admin=False) and declare no DPI awareness, matching pythonw.exe:
# capture/window.py and gui.py set the DPI mode themselves at run time.
import json
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs

STAGE = Path(os.environ["MAPLEBOT_STAGE"]).resolve()
KEEP_FFMPEG = os.environ.get("MAPLEBOT_KEEP_FFMPEG") == "1"
mods = json.loads((STAGE / "packaging" / "build-modules.json").read_text(encoding="utf-8"))
assert "maplebot._build_info" in mods["maplebot_modules"], "stage.py must write _build_info first"
hidden = sorted(set(mods["maplebot_modules"] + mods["script_imports"] + [
    "maplebot._build_info",
    "win32api", "win32con", "win32event", "win32gui", "pywintypes",
    "pynput.keyboard._win32", "pynput.mouse._win32", "pynput._util.win32", "pynput._util.win32_vks",
    "windows_capture", "mss", "mss.windows", "yaml", "hashlib", "linecache",
    "ctypes.wintypes", "msvcrt", "tkinter", "tkinter.ttk", "tkinter.messagebox", "tkinter.filedialog"]))
# The three scripts are DATA (source text) executed by maplebot.app.run_script with a virtual
# __file__. They are never Analysis scripts: PyInstaller runs every listed script at startup.
datas = [(str(STAGE / "scripts" / n), "scripts")
         for n in ("run_active.py", "relay_client.py", "supervise_farming.py")]
hooks = STAGE / "packaging" / "hooks"
icon = STAGE / "packaging" / "maplebot.ico"

a = Analysis(
    [str(STAGE / "packaging" / "maplebot_entry.py")],
    pathex=[str(STAGE)],
    binaries=collect_dynamic_libs("windows_capture"),
    datas=datas,
    hiddenimports=hidden,
    hookspath=[str(hooks)] if hooks.is_dir() else [],
    excludes=["pytest", "_pytest", "pip", "setuptools", "pkg_resources", "IPython", "matplotlib",
              "tkinter.test", "test", "lib2to3", "pydoc_data", "maplebot.__main__"],
    noarchive=False,
    optimize=0,
)
if not KEEP_FFMPEG:
    a.binaries = [b for b in a.binaries
                  if not os.path.basename(b[0]).lower().startswith("opencv_videoio_ffmpeg")]
pyz = PYZ(a.pure)
common = dict(exclude_binaries=True, debug=False, strip=False, upx=False, uac_admin=False,
              uac_uiaccess=False, version=str(STAGE / "packaging" / "version_info.txt"),
              icon=str(icon) if icon.is_file() else None)
gui = EXE(pyz, a.scripts, [], name="Maplebot", console=False, **common)
cli = EXE(pyz, a.scripts, [], name="maplebot-cli", console=True, **common)
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name="Maplebot")
