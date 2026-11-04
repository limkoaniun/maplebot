"""Live capture of the game window on Windows (mss + win32gui). Imports are lazy so the
rest of the package stays importable on Linux/macOS for tests."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from maplebot.log import get_logger

log = get_logger()


def find_window_title(token: str) -> str | None:
    import win32gui

    matches: list[str] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if token.lower() in title.lower():
            matches.append(title)

    win32gui.EnumWindows(cb, None)
    return matches[0] if matches else None


def window_rect(title: str) -> tuple[int, int, int, int] | None:
    """(left, top, width, height) of the window's client area in screen pixels."""
    import win32gui

    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return None
    l, t, r, b = win32gui.GetClientRect(hwnd)
    sl, st = win32gui.ClientToScreen(hwnd, (l, t))
    return sl, st, r - l, b - t


def is_foreground(title: str) -> bool:
    import win32gui

    try:
        return win32gui.GetWindowText(win32gui.GetForegroundWindow()) == title
    except Exception:
        return False


def set_dpi_aware() -> None:
    """Make GetClientRect/ClientToScreen report physical pixels, like mss does, on scaled displays."""
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def activate_window(title: str) -> bool:
    import win32con
    import win32gui

    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return False
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return True
    except Exception as e:  # SetForegroundWindow is picky about focus rules
        log.warning("activate_window failed: %s", e)
        return False


class WindowCapture:
    """Grabs the game window's client area in a background thread."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        set_dpi_aware()
        token = str(cfg["game_window"]["title"])
        self.title = find_window_title(token)
        if self.title is None:
            raise RuntimeError(f"no visible window whose title contains {token!r}")
        self.fps = float(cfg.get("system", {}).get("fps_capture", 15))
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def _run(self) -> None:
        import mss

        period = 1.0 / max(1.0, self.fps)
        with mss.mss() as sct:
            while not self._stop.is_set():
                t0 = time.monotonic()
                rect = window_rect(self.title)
                if rect is not None and rect[2] > 0 and rect[3] > 0:
                    left, top, w, h = rect
                    try:
                        raw = np.asarray(sct.grab({"left": left, "top": top, "width": w, "height": h}))
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                        with self._lock:
                            self._frame = frame
                    except Exception as e:
                        log.debug("grab failed: %s", e)
                else:
                    with self._lock:
                        self._frame = None  # minimised / gone: report no frame rather than a stale one
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
