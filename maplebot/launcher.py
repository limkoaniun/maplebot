"""Start and probe the elevated controller without PowerShell (a Python port of start_active.ps1).

The settings app and maplebot-cli use this module. It adds three checks the script lacks:
the per-root mutex, the global controller mutex, and a scan for source-checkout controller
processes. Any of them makes Start refuse (183) WITHOUT clearing a signal file.

Every Win32 call goes through an injectable ``api`` object; ctypes is loaded inside the
methods, so importing this module is cheap and has no side effects.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from maplebot import paths

START_SIGNALS = ("stop-active", "arm-active", "pause-active", "resume-active",
                 "reload-active", "return-to-start-active", "hold-farming-active")  # start_active.ps1 order
GLOBAL_MUTEX = "Local\\MaplebotController"
CONTROLLER_IMAGES = frozenset({"maplebot.exe", "pythonw.exe", "python.exe"})
PYTHON_IMAGES = frozenset({"pythonw.exe", "python.exe"})
LIVE_INPUT_MARKERS = ("run_active.py", "verify_potions.py", "verify_route.py", "verify_lap.py",
                      "watch_full_circuit.py", "watch_circuits.py")
LEGACY_RUN_RE = r"-m\s+maplebot(\.__main__)?\s+run\b|maplebot(\.exe)?\"?\s+run\b"
STATUS_FRESH_S = 6.0

ERROR_FILE_NOT_FOUND, ERROR_ACCESS_DENIED, ERROR_INVALID_PARAMETER = 2, 5, 87
ERROR_ALREADY_EXISTS, ERROR_CANCELLED = 183, 1223
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
SEE_MASK_NOASYNC = 0x100
SW_HIDE = 0
COINIT_APARTMENTTHREADED, COINIT_DISABLE_OLE1DDE = 0x2, 0x4
PROCESS_COMMAND_LINE_INFORMATION = 60
TH32CS_SNAPPROCESS = 0x2

# A marker counts only as a whole file name: "...\scripts\run_active.py" matches,
# "tests/test_verify_route.py" does not.
_MARKER_RE = re.compile(r"(?:^|[\\/\s\"'=])(?:" + "|".join(re.escape(m) for m in LIVE_INPUT_MARKERS)
                        + r")(?=$|[\s\"'])", re.IGNORECASE)
# The same tools started as modules from a checkout root: "python -m scripts.run_active"
# (scripts\ resolves as a namespace package there). "-mscripts.x" is accepted by Python too.
_MODULE_RE = re.compile(r"-m\s*[\"']?scripts\.(?:" + "|".join(re.escape(m[:-len(".py")]) for m in LIVE_INPUT_MARKERS)
                        + r")\b", re.IGNORECASE)


def per_root_mutex_name(root) -> str:
    """Must equal run_active.py's 'Local\\MaplebotActive_' + sha256(str(ROOT).lower())[:16]."""
    resolved = Path(root).resolve()
    return "Local\\MaplebotActive_" + hashlib.sha256(str(resolved).lower().encode()).hexdigest()[:16]


class Win32Api:
    """The real Win32 calls. Tests replace this class with a fake."""

    def __init__(self, dll_loader=None):
        self._loader = dll_loader

    def _dll(self, name):
        if self._loader is not None:
            return self._loader(name)
        import ctypes
        return ctypes.WinDLL(name, use_last_error=True)

    @staticmethod
    def _last_error():
        import ctypes
        return ctypes.get_last_error()

    def _kernel32(self):
        import ctypes
        from ctypes import wintypes
        k = self._dll("kernel32")
        k.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        k.OpenMutexW.restype = wintypes.HANDLE
        k.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        k.CreateMutexW.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                 ctypes.POINTER(wintypes.DWORD)]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k.GetCurrentProcess.argtypes = []
        k.GetCurrentProcess.restype = wintypes.HANDLE
        return k

    def mutex_exists(self, name) -> bool:
        k = self._kernel32()
        handle = k.OpenMutexW(SYNCHRONIZE, False, name)
        if handle:
            k.CloseHandle(handle)
            return True
        # ACCESS_DENIED (an elevated owner probed from a normal process) or any other error
        # counts as held: only "not found" proves the mutex is absent.
        return self._last_error() != ERROR_FILE_NOT_FOUND

    def create_mutex(self, name):
        k = self._kernel32()
        handle = k.CreateMutexW(None, False, name)
        return handle, self._last_error()

    def close_handle(self, handle) -> None:
        if handle:
            self._kernel32().CloseHandle(handle)

    def process_state(self, pid):
        import ctypes
        from ctypes import wintypes
        k = self._kernel32()
        handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return ("dead", None) if self._last_error() == ERROR_INVALID_PARAMETER else ("denied", None)
        try:
            code = wintypes.DWORD()
            if not k.GetExitCodeProcess(handle, ctypes.byref(code)):
                return "denied", None
            if code.value != STILL_ACTIVE:
                return "dead", None
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not k.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return "denied", None
            return "alive", os.path.basename(buffer.value).lower()
        finally:
            k.CloseHandle(handle)

    def list_processes(self):
        import ctypes
        from ctypes import wintypes

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

        k = self._kernel32()
        k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k.Process32FirstW.restype = wintypes.BOOL
        k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k.Process32NextW.restype = wintypes.BOOL
        snapshot = k.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if not snapshot or snapshot == ctypes.c_void_p(-1).value:
            raise OSError(self._last_error(), "CreateToolhelp32Snapshot failed")
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            result = []
            ok = k.Process32FirstW(snapshot, ctypes.byref(entry))
            while ok:
                result.append((int(entry.th32ProcessID), entry.szExeFile.lower()))
                ok = k.Process32NextW(snapshot, ctypes.byref(entry))
            if not result:
                raise OSError(self._last_error(), "the process list is empty")
            return result
        finally:
            k.CloseHandle(snapshot)

    def command_line(self, pid):
        import ctypes
        from ctypes import wintypes
        try:
            k = self._kernel32()
            ntdll = self._dll("ntdll")
            ntdll.NtQueryInformationProcess.argtypes = [wintypes.HANDLE, wintypes.ULONG, ctypes.c_void_p,
                                                        wintypes.ULONG, ctypes.POINTER(wintypes.ULONG)]
            ntdll.NtQueryInformationProcess.restype = wintypes.LONG
        except (OSError, AttributeError):
            return None

        class UNICODE_STRING(ctypes.Structure):
            _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                        ("Buffer", ctypes.c_void_p)]

        handle = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return None
        try:
            size = 4096
            for _ in range(3):
                buffer = ctypes.create_string_buffer(size)
                needed = wintypes.ULONG(0)
                status = ntdll.NtQueryInformationProcess(handle, PROCESS_COMMAND_LINE_INFORMATION,
                                                         buffer, size, ctypes.byref(needed))
                if status == 0:
                    text = UNICODE_STRING.from_buffer(buffer)
                    if not text.Buffer or text.Length == 0:
                        return ""
                    return ctypes.wstring_at(text.Buffer, text.Length // 2)
                if needed.value > size:
                    size = needed.value + 64
                    continue
                return None
            return None
        except (OSError, ValueError):
            return None
        finally:
            k.CloseHandle(handle)

    def is_elevated(self) -> bool:
        import ctypes
        from ctypes import wintypes
        k = self._kernel32()
        advapi = self._dll("advapi32")
        advapi.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
        advapi.OpenProcessToken.restype = wintypes.BOOL
        advapi.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                               wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        advapi.GetTokenInformation.restype = wintypes.BOOL
        token = wintypes.HANDLE()
        if not advapi.OpenProcessToken(k.GetCurrentProcess(), 0x0008, ctypes.byref(token)):  # TOKEN_QUERY
            return False
        try:
            elevation = wintypes.DWORD(0)
            returned = wintypes.DWORD(0)
            if not advapi.GetTokenInformation(token, 20, ctypes.byref(elevation), ctypes.sizeof(elevation),
                                              ctypes.byref(returned)):  # TokenElevation
                return False
            return bool(elevation.value)
        finally:
            k.CloseHandle(token)

    def shell_execute_runas(self, exe, params, cwd, hwnd=0) -> int:
        import ctypes
        ole32 = self._dll("ole32")
        shell32 = self._dll("shell32")
        info = build_execute_info(exe, params, cwd, hwnd)
        initialized = ole32.CoInitializeEx(None, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE)
        try:
            if shell32.ShellExecuteExW(ctypes.byref(info)):
                return 0
            return self._last_error() or 1
        finally:
            if initialized in (0, 1):  # S_OK or S_FALSE: balance with CoUninitialize
                ole32.CoUninitialize()


def _execute_info_type():
    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
                    ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR),
                    ("lpParameters", wintypes.LPCWSTR), ("lpDirectory", wintypes.LPCWSTR),
                    ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE), ("lpIDList", ctypes.c_void_p),
                    ("lpClass", wintypes.LPCWSTR), ("hkeyClass", wintypes.HKEY), ("dwHotKey", wintypes.DWORD),
                    ("hIconOrMonitor", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]
    return SHELLEXECUTEINFOW


def build_execute_info(exe, params, cwd, hwnd=0):
    """SHELLEXECUTEINFOW for the "runas" verb. SEE_MASK_NOASYNC only: no FLAG_NO_UI, so a
    Windows security or error dialog is shown instead of becoming a silent failure."""
    import ctypes
    kind = _execute_info_type()
    info = kind()
    info.cbSize = ctypes.sizeof(kind)
    info.fMask = SEE_MASK_NOASYNC
    info.hwnd = hwnd or None
    info.lpVerb = "runas"
    info.lpFile = str(exe)
    info.lpParameters = str(params)
    info.lpDirectory = str(cwd)
    info.nShow = SW_HIDE
    return info


def _api(api):
    return api if api is not None else Win32Api()


def append_log(root, line: str, name: str = "launcher.log") -> None:
    """Append one timestamped line to log/<name>; never raises."""
    try:
        folder = Path(root) / "log"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        with open(folder / name, "a", encoding="utf-8") as handle:
            handle.write(f"{stamp} {line.rstrip()}\n")
    except OSError:
        pass


def read_status(root) -> dict | None:
    try:
        value = json.loads((Path(root) / "log" / "active-status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def command_line_is_controller(command_line: str) -> bool:
    return bool(_MARKER_RE.search(command_line) or _MODULE_RE.search(command_line)
                or re.search(LEGACY_RUN_RE, command_line, re.IGNORECASE))


def foreign_controllers(root, *, api=None, self_pid=None) -> list[str]:
    """Reasons a source-checkout tool that sends input may be running (fails closed).

    A Python process whose command line names run_active.py, a live verify/watch tool or
    the legacy ``maplebot run``, or whose command line cannot be read (for example an
    elevated controller probed from a normal process), is a reason. Maplebot.exe processes
    are not inspected: an exe controller holds GLOBAL_MUTEX.
    """
    api = _api(api)
    self_pid = os.getpid() if self_pid is None else self_pid
    try:
        processes = list(api.list_processes())
    except Exception:  # noqa: BLE001 - any enumeration failure fails closed
        return ["process list unavailable"]
    reasons = []
    for pid, image in processes:
        if pid == self_pid or str(image).lower() not in PYTHON_IMAGES:
            continue
        try:
            command = api.command_line(pid)
        except Exception:  # noqa: BLE001
            command = None
        if command is None:
            reasons.append(f"Python process {pid} could not be inspected (it may be an elevated Maplebot controller)")
        elif command_line_is_controller(command):
            reasons.append(f"a source-checkout Maplebot tool is running (pid {pid})")
    return reasons


def controller_alive(root, *, api=None, now=time.time, scan_foreign=True) -> str | None:
    """A reason string when a controller must be assumed running, else None."""
    api = _api(api)
    if api.mutex_exists(GLOBAL_MUTEX):
        return "a Maplebot controller is running"
    if api.mutex_exists(per_root_mutex_name(root)):
        return "this folder's controller is running"
    status = read_status(root)
    if status and status.get("running"):
        try:
            fresh = now() - float(status.get("updated", 0)) < STATUS_FRESH_S
        except (TypeError, ValueError):
            fresh = False
        if fresh:
            raw = status.get("pid")
            try:
                if isinstance(raw, bool):
                    raise TypeError
                pid = int(raw)
            except (TypeError, ValueError):
                return "running (unverifiable pid)"
            state, image = api.process_state(pid)
            if state == "denied":
                return "this folder's controller process is running (access denied)"
            if state == "alive" and image in CONTROLLER_IMAGES:
                return "this folder's controller process is running"
    if scan_foreign:
        reasons = foreign_controllers(root, api=api)
        if reasons:
            return "; ".join(reasons)
    return None


def any_controller_alive(roots, *, api=None, now=time.time) -> str | None:
    """controller_alive over several roots (the foreign scan runs once)."""
    api = _api(api)
    for index, root in enumerate(roots):
        reason = controller_alive(root, api=api, now=now, scan_foreign=index == 0)
        if reason:
            return reason
    return None


def clear_start_signals(root) -> list[str]:
    folder = Path(root) / "log"
    removed = []
    for name in START_SIGNALS:
        try:
            (folder / name).unlink()
            removed.append(name)
        except FileNotFoundError:
            pass
    return removed


def _source_pythonw() -> str:
    sibling = Path(sys.executable).with_name("pythonw.exe")
    if sibling.is_file():
        return str(sibling)
    return str(Path(os.environ.get("USERPROFILE", "")) / ".venvs" / "maplebot" / "Scripts" / "pythonw.exe")


def controller_command(root, wait_for_start=False) -> tuple[str, str]:
    """(exe, parameters) for the elevated controller launch."""
    flag = ["--wait-for-start"] if wait_for_start else []
    if paths.is_frozen():
        exe = str(Path(sys.executable).with_name("Maplebot.exe"))
        args = ["controller", *flag]
    else:
        exe = _source_pythonw()
        args = ["-u", str(Path(root) / "scripts" / "run_active.py"), *flag]
    return exe, subprocess.list2cmdline(args)


def relay_command(root) -> list[str]:
    if paths.is_frozen():
        return [str(Path(sys.executable).with_name("Maplebot.exe")), "relay"]
    return [sys.executable, str(Path(root) / "scripts/relay_client.py")]


class LaunchHandle:
    """Popen.poll() semantics: None while the UAC prompt is pending, then the result code."""

    def __init__(self, code=None, reason=None):
        self._code = code
        self.reason = reason
        self._done = threading.Event()
        if code is not None:
            self._done.set()

    @classmethod
    def done(cls, code, reason=None):
        return cls(int(code), reason)

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        self._done.wait(timeout)
        return self._code

    def _finish(self, code):
        self._code = int(code)
        self._done.set()


def start_controller(root, wait_for_start=False, hwnd=0, *, api=None, threaded=True) -> LaunchHandle:
    root = Path(root)
    api = _api(api)
    reason = controller_alive(root, api=api)
    if reason:
        append_log(root, f"start refused: {reason}")
        return LaunchHandle.done(ERROR_ALREADY_EXISTS, reason)
    clear_start_signals(root)  # may raise OSError; the caller reports it
    exe, params = controller_command(root, wait_for_start)
    handle = LaunchHandle()

    def work():
        code = 1
        try:
            code = int(api.shell_execute_runas(exe, params, str(root), hwnd))
        except BaseException:  # noqa: BLE001 - recorded, reported as a failed launch
            append_log(root, "start: launch raised\n" + traceback.format_exc())
            code = 1
        finally:
            append_log(root, f"start: {exe} {params} -> {code}")
            handle._finish(code)

    if threaded:
        threading.Thread(target=work, name="maplebot-uac-launch", daemon=True).start()
    else:
        work()
    return handle


def launch_failure_message(code, reason=None) -> str:
    if code == ERROR_CANCELLED:
        return "Start was cancelled at the Windows administrator prompt."
    if code == ERROR_ALREADY_EXISTS:
        detail = f" ({reason})" if reason else ""
        return f"Another Maplebot controller is already running{detail}. Stop it before starting again."
    return "The launch did not complete. Try Start active and approve the Windows prompt."


def request_stop(root) -> Path:
    folder = Path(root) / "log"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "stop-active"
    path.touch()
    return path
