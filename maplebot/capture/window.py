"""Live client-area capture using desktop pixels or Windows Graphics Capture.

Windows imports stay lazy so image processing remains usable on other platforms.
The graphics backend captures the selected window even when another app covers it.
"""
from __future__ import annotations

import threading
import time
import math
from dataclasses import dataclass

import cv2
import numpy as np

from maplebot.log import get_logger

log = get_logger()


@dataclass(frozen=True)
class FrameSample:
    """A successfully captured image and its monotonic producer metadata."""

    frame: np.ndarray
    sequence: int
    captured_at: float


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
    if not hwnd or win32gui.IsIconic(hwnd) or not win32gui.IsWindowVisible(hwnd):
        return None
    try:
        l, t, r, b = win32gui.GetClientRect(hwnd)
        sl, st = win32gui.ClientToScreen(hwnd, (l, t))
    except Exception:
        return None  # The window may close between lookup and querying its bounds.
    return sl, st, r - l, b - t


def crop_graphics_client(raw: np.ndarray, client: tuple[int, int, int, int],
                         bounds: tuple[int, int, int, int]) -> np.ndarray | None:
    """Copy BGR client pixels from a WGC frame; reject geometry during a resize.

    ``client`` is screen x/y/width/height; ``bounds`` is the captured window's
    left/top/right/bottom, normally DWM extended frame bounds (without shadows).
    """
    left, top, width, height = client
    x, y = left - bounds[0], top - bounds[1]
    if width <= 0 or height <= 0 or x < 0 or y < 0:
        return None
    if raw.shape[:2] != (bounds[3] - bounds[1], bounds[2] - bounds[0]):
        return None
    if x + width > raw.shape[1] or y + height > raw.shape[0]:
        return None
    return raw[y:y + height, x:x + width, :3].copy(order="C")


def graphics_client_frame(raw: np.ndarray, title: str) -> np.ndarray | None:
    """Convert a full HWND capture to the same physical client pixels as mss."""
    import ctypes
    from ctypes import wintypes
    import win32gui

    client = window_rect(title)
    if client is None:
        return None
    hwnd = win32gui.FindWindow(None, title)
    if not hwnd:
        return None
    # Borderless windows can already be client-sized.
    if raw.shape[:2] == (client[3], client[2]):
        return raw[:, :, :3].copy(order="C")
    rect = wintypes.RECT()
    result = ctypes.windll.dwmapi.DwmGetWindowAttribute(
        wintypes.HWND(hwnd), 9, ctypes.byref(rect), ctypes.sizeof(rect))
    bounds = (rect.left, rect.top, rect.right, rect.bottom) if result == 0 else None
    if bounds is not None:
        frame = crop_graphics_client(raw, client, bounds)
        if frame is not None:
            return frame
    return crop_graphics_client(raw, client, tuple(win32gui.GetWindowRect(hwnd)))


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

    def __init__(self, cfg: dict, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.max_frame_age_s = float(cfg.get("safety", {}).get("max_frame_age_s", 0.75))
        if not math.isfinite(self.max_frame_age_s) or self.max_frame_age_s <= 0:
            raise ValueError("safety.max_frame_age_s must be finite and positive")
        set_dpi_aware()
        token = str(cfg["game_window"]["title"])
        self.title = find_window_title(token)
        if self.title is None:
            raise RuntimeError(f"no visible window whose title contains {token!r}")
        self.fps = float(cfg.get("system", {}).get("fps_capture", 15))
        self.backend = str(cfg["game_window"].get("capture_backend", "mss")).lower()
        if self.backend not in ("mss", "graphics"):
            raise ValueError(f"unknown capture backend: {self.backend!r}")
        self.last_error: str | None = None
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._captured_at: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._ready.clear()
        self.last_error = None
        target = self._run_graphics if self.backend == "graphics" else self._run
        self._thread = threading.Thread(target=target, name="capture", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)
        if self.last_error is not None:
            self.stop()
            raise RuntimeError(self.last_error)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if not self._thread.is_alive():
                self._thread = None
        self._publish(None)

    def get_frame(self) -> np.ndarray | None:
        """Compatibility API; returns the latest image only while it is fresh."""
        sample = self.get_sample()
        return None if sample is None else sample.frame

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return self._sequence

    @property
    def captured_at(self) -> float | None:
        with self._lock:
            return self._captured_at

    def _publish(self, frame: np.ndarray | None, *, captured_at: float | None = None) -> None:
        """Only capture producers publish; reading a frame never renews its age."""
        with self._lock:
            self._frame = frame
            self._captured_at = (None if frame is None else
                                 self.clock() if captured_at is None else captured_at)
            if frame is not None:
                self._sequence += 1

    def get_sample(self, *, after_sequence: int | None = None,
                   max_age_s: float | None = None) -> FrameSample | None:
        max_age = self.max_frame_age_s if max_age_s is None else float(max_age_s)
        if not math.isfinite(max_age) or max_age <= 0:
            raise ValueError("max_age_s must be finite and positive")
        if self._stop.is_set() or window_rect(self.title) is None:
            self._publish(None)
            return None
        with self._lock:
            if self._frame is None or self._captured_at is None:
                return None
            age = self.clock() - self._captured_at
            if age < 0 or age >= max_age:
                return None
            if after_sequence is not None and self._sequence <= after_sequence:
                return None
            return FrameSample(self._frame.copy(), self._sequence, self._captured_at)

    def _run_graphics(self) -> None:
        control = None
        try:
            import win32gui
            from windows_capture import WindowsCapture

            hwnd = win32gui.FindWindow(None, self.title)
            if not hwnd:
                raise RuntimeError("game window closed before capture started")
            capture = WindowsCapture(
                window_hwnd=hwnd, cursor_capture=False,
                minimum_update_interval=max(1, round(1000 / max(1.0, self.fps))))

            @capture.event
            def on_frame_arrived(frame, capture_control):
                if self._stop.is_set():
                    capture_control.stop()
                    return
                arrived_at = self.clock()
                try:
                    client_frame = graphics_client_frame(frame.frame_buffer, self.title)
                    self._publish(client_frame, captured_at=arrived_at)
                    if client_frame is not None:
                        self._ready.set()
                except Exception as exc:
                    self._publish(None)
                    self.last_error = f"window capture failed: {exc}"
                    self._ready.set()
                    self._stop.set()
                    capture_control.stop()

            @capture.event
            def on_closed():
                self._publish(None)
                self.last_error = "game window closed"
                self._stop.set()
                self._ready.set()

            control = capture.start_free_threaded()
            while not self._stop.wait(0.1):
                if control.is_finished():
                    self.last_error = self.last_error or "window capture stopped"
                    break
                if window_rect(self.title) is None:
                    self._publish(None)
        except Exception as exc:
            self.last_error = f"graphics capture unavailable: {exc}"
            log.warning("%s", self.last_error)
        finally:
            if control is not None:
                try:
                    control.stop()
                except Exception as exc:
                    log.debug("capture shutdown: %s", exc)
            self._publish(None)
            self._stop.set()
            self._ready.set()

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
                        captured_at = self.clock()
                        raw = np.asarray(sct.grab({"left": left, "top": top, "width": w, "height": h}))
                        frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                        self._publish(frame, captured_at=captured_at)
                        self._ready.set()
                    except Exception as e:
                        self._publish(None)
                        log.debug("grab failed: %s", e)
                else:
                    self._publish(None)  # minimised / gone
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
