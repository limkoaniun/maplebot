"""Applies Commands to the keyboard: holds for movement, timed taps for actions."""
from __future__ import annotations

import threading
import time
from collections import deque

from maplebot.brain.combat import AttackTimer
from maplebot.health import BuffScheduler
from maplebot.log import get_logger
from maplebot.types import Command

TAP_S = 0.05
log = get_logger()


class KeyboardController:
    def __init__(self, cfg: dict, backend, clock=time.monotonic, sleep=time.sleep):
        self.cfg = cfg
        self.keys = cfg["keys"]
        self.backend = backend
        self.clock = clock
        self.sleep = sleep
        self.fps = float(cfg.get("system", {}).get("fps_keyboard", 30))
        self.cmd = Command.NONE
        self.enabled = True
        self._lock = threading.Lock()
        self._requests: deque[str] = deque()
        self._held_x: str | None = None
        self._held_y: str | None = None
        self._last_cmd: Command | None = None
        self._attack = AttackTimer(cfg["attack"])
        self._buffs = BuffScheduler(cfg.get("buffs", {}))
        self._freeze_until = 0.0
        self._last_pickup: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- public API ------------------------------------------------------
    def set_command(self, cmd: Command) -> None:
        with self._lock:
            self.cmd = cmd

    def request(self, name: str) -> None:
        """Queue a one-shot: "hp_potion" or "mp_potion"."""
        with self._lock:
            self._requests.append(name)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self.release_all()

    def release_all(self) -> None:
        for k in (self._held_x, self._held_y):
            if k:
                self.backend.up(k)
        self._held_x = self._held_y = None
        self._last_cmd = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="keyboard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.release_all()

    # -- one tick --------------------------------------------------------
    def tap(self, key: str, duration: float = TAP_S) -> None:
        if not key:
            return
        self.backend.down(key)
        self.sleep(duration)
        self.backend.up(key)

    def _key(self, name: str) -> str:
        return str(self.keys.get(name) or "")

    def apply_once(self, now: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            cmd = self.cmd
            req = self._requests.popleft() if self._requests else None

        if req is not None:
            self.tap(self._key(req))
            return

        buff = self._buffs.due(now)
        if buff is not None:
            self._release_movement()
            self.tap(buff)
            self._buffs.mark(buff, now)
            self._freeze_until = now + self._buffs.freeze_s
            return
        if now < self._freeze_until:
            self._release_movement()
            return

        self._apply_hold(cmd.move_x, "x", now)
        self._apply_hold(cmd.move_y, "y", now)
        self._apply_action(cmd, now)
        self._apply_pickup(now)
        self._last_cmd = cmd

    # -- pieces ----------------------------------------------------------
    def _release_movement(self) -> None:
        for attr in ("_held_x", "_held_y"):
            k = getattr(self, attr)
            if k:
                self.backend.up(k)
                setattr(self, attr, None)

    def _apply_hold(self, want: str, axis: str, now: float) -> None:
        attr = "_held_x" if axis == "x" else "_held_y"
        held = getattr(self, attr)
        target = self._key(want) if want in ("left", "right", "up", "down") else None
        if held == target:
            return
        if held:
            self.backend.up(held)
        if target:
            self.backend.down(target)
        setattr(self, attr, target)

    def _apply_action(self, cmd: Command, now: float) -> None:
        a = cmd.action
        if a in ("attack", "aoe"):
            kind = "aoe" if (a == "aoe" and self._key("aoe")) else "attack"
            if self._attack.ready(kind, now):
                self.tap(self._key(kind))
                self._attack.mark(kind, now)
            return
        changed = cmd != self._last_cmd
        if not changed:
            return
        if a == "jump":
            self.tap(self._key("jump"))
        elif a == "teleport":
            self.tap(self._key("teleport") or self._key("jump"))

    def _apply_pickup(self, now: float) -> None:
        key = self._key("pickup")
        if not key or not (self._held_x or self._held_y):
            return
        interval = float(self.keys.get("pickup_interval_s", 0.3))
        if self._last_pickup is None or now - self._last_pickup >= interval:
            self.tap(key)
            self._last_pickup = now

    def _run(self) -> None:
        period = 1.0 / max(1.0, self.fps)
        while not self._stop.is_set():
            t0 = self.clock()
            try:
                self.apply_once(t0)
            except Exception as e:  # keep the key thread alive, but say so
                log.error("keyboard tick failed: %s", e)
            dt = self.clock() - t0
            if dt < period:
                time.sleep(period - dt)
