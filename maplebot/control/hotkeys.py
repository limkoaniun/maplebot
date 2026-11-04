"""Global function-key hotkeys (F1 pause, F2 screenshot, F12 quit) via pynput, imported lazily."""
from __future__ import annotations

import time
from typing import Callable


class Hotkeys:
    def __init__(self, bindings: dict[str, Callable[[], None]], debounce_s: float = 1.0):
        self.bindings = {k.lower(): v for k, v in bindings.items()}
        self.debounce_s = float(debounce_s)
        self._last: dict[str, float] = {}
        self._listener = None

    def _on_press(self, key) -> None:
        name = getattr(key, "name", None)
        if name is None:
            return
        name = name.lower()
        fn = self.bindings.get(name)
        if fn is None:
            return
        now = time.monotonic()
        if now - self._last.get(name, -1e9) < self.debounce_s:
            return
        self._last[name] = now
        try:
            fn()
        except Exception:
            pass

    def start(self) -> None:
        from pynput import keyboard

        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
