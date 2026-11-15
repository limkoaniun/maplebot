"""Applies Commands to the keyboard: holds for movement, timed taps for actions."""
from __future__ import annotations

import threading
import time
import math
from collections import deque

from maplebot.brain.combat import AttackTimer
from maplebot.health import BuffScheduler
from maplebot.log import get_logger
from maplebot.types import Command

TAP_S = 0.05
FACE_DELAY_S = 0.08
log = get_logger()


class KeyboardController:
    def __init__(self, cfg: dict, backend, clock=time.monotonic, sleep=time.sleep, is_active=None):
        self.cfg = cfg
        self.is_active = is_active or (lambda: True)  # e.g. "is the game window in the foreground"
        self.keys = cfg["keys"]
        self.backend = backend
        self.clock = clock
        self.sleep = sleep
        self.fps = float(cfg.get("system", {}).get("fps_keyboard", 30))
        safety = cfg.get("safety", {})
        self.command_timeout_s = float(safety.get("command_timeout_s", 0.75))
        self.max_frame_age_s = float(safety.get("max_frame_age_s", 0.75))
        for name, value in (("command_timeout_s", self.command_timeout_s),
                            ("max_frame_age_s", self.max_frame_age_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"safety.{name} must be finite and positive")
        self.cmd = Command.NONE
        self._command_deadline: float | None = None
        # Capture stamp of the last stamped set_command, and the deadline each
        # leased (valid_for_s) command was first granted for that capture.  A
        # repeat send for the same capture may only shorten such a pulse.
        self._command_source_time: float | None = None
        self._pulse_deadlines: dict[Command, float] = {}
        self._navigation_only = False
        self._hp_only = False
        self.hp_taps_sent = 0
        self.hp_tap_time: float | None = None
        self.enabled = True
        self._lock = threading.Lock()
        self._tick_lock = threading.RLock()
        self._requests: deque[tuple[str, float]] = deque()
        self._held_x: str | None = None
        self._held_y: str | None = None
        self._facing_x: str | None = None
        self._face_ready_at = 0.0
        self._last_cmd: Command | None = None
        self._attack = AttackTimer(cfg["attack"])
        self._buffs = BuffScheduler(cfg.get("buffs", {}))
        self._freeze_until = 0.0
        self._last_pickup: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- public API ------------------------------------------------------
    @property
    def hp_only(self) -> bool:
        return self._hp_only

    def set_command(self, cmd: Command, *, source_time: float | None = None,
                    valid_for_s: float | None = None, navigation_only: bool = False) -> None:
        """Renew input permission, optionally bounded by the source frame's age.

        Lease time always uses ``clock``; ``apply_once(now)`` independently drives
        action cooldowns so offline tests can simulate those without sleeping.

        A command carrying its own ``valid_for_s`` is a positioning pulse sized
        for one capture.  Sending an equal command again for the same
        ``source_time`` never extends it: the earliest deadline granted for that
        capture wins, also after the worker already expired the pulse.  A new
        capture stamp, a different command, and unleased holds renew normally.
        """
        deadline = self._lease_deadline(source_time, valid_for_s, command_valid_for_s=cmd.valid_for_s)
        # Serialize mode changes with a complete tick, so a prior combat tick
        # cannot press an automatic action after navigation-only takes effect.
        with self._tick_lock:
            with self._lock:
                if self._hp_only:
                    self._requests.clear()
                self._hp_only = False
                self._navigation_only = bool(navigation_only)
                if self._navigation_only:
                    cmd = cmd.replace(face_x="none", action=cmd.action if cmd.action == "jump" else "none")
                    self._requests.clear()
                if source_time is None or source_time != self._command_source_time:
                    self._pulse_deadlines.clear()
                self._command_source_time = source_time
                if cmd.valid_for_s is not None and source_time is not None:
                    # Command equality ignores valid_for_s, so a repeat with a
                    # different pulse length is still the same pulse.
                    deadline = min(deadline, self._pulse_deadlines.get(cmd, deadline))
                    self._pulse_deadlines[cmd] = deadline
                self.cmd = cmd
                self._command_deadline = deadline

    def _forget_capture(self) -> None:
        """Drop same-capture pulse memory. Caller holds ``_lock``."""
        self._command_source_time = None
        self._pulse_deadlines.clear()

    def _lease_deadline(self, source_time: float | None, valid_for_s: float | None = None,
                        *, command_valid_for_s: float | None = None) -> float:
        now = self.clock()
        duration = self.command_timeout_s if valid_for_s is None else float(valid_for_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("valid_for_s must be finite and positive")
        if command_valid_for_s is not None:
            duration = min(duration, command_valid_for_s)
        deadline = now + min(duration, self.command_timeout_s)
        if source_time is not None:
            stamp = float(source_time)
            if not math.isfinite(stamp) or stamp > now:
                deadline = now  # invalid/future capture stamps cannot extend permission
            else:
                deadline = min(deadline, stamp + self.max_frame_age_s)
        return deadline

    def set_hp_only(self, *, source_time: float | None) -> None:
        """Permit only explicitly requested HP taps after the caller enables input.

        An ordinary action already in flight finishes before this returns. All
        movement is released immediately; a queued HP request survives successive
        fresh renewals of this mode, but never an expired lease or a mode change.
        """
        deadline = self._lease_deadline(source_time)
        with self._tick_lock:
            with self._lock:
                now = self.clock()
                preserve = (self._hp_only and self.enabled and self.is_active()
                            and self._command_deadline is not None and now < self._command_deadline
                            and now < deadline)
                if not preserve:
                    self._requests.clear()
                else:
                    pending = next((item for item in self._requests
                                    if item[0] == "hp_potion" and item[1] > now), None)
                    self._requests.clear()
                    if pending is not None:
                        self._requests.append(pending)
                self._hp_only = True
                self._navigation_only = False
                self.cmd = Command.NONE
                self._command_deadline = deadline
                self._forget_capture()
            self.release_all()

    def request(self, name: str) -> None:
        """Queue a one-shot: "hp_potion" or "mp_potion"."""
        with self._lock:
            if self.enabled and not self._navigation_only and self.is_active():
                now = self.clock()
                if self._hp_only:
                    if (name != "hp_potion" or self._command_deadline is None
                            or now >= self._command_deadline):
                        return
                    while self._requests and self._requests[0][1] <= now:
                        self._requests.popleft()
                    if self._requests:
                        return
                    self._requests.append((name, min(now + self.command_timeout_s, self._command_deadline)))
                else:
                    self._requests.append((name, now + self.command_timeout_s))

    def clear_requests(self) -> None:
        with self._lock:
            self._requests.clear()

    def set_enabled(self, enabled: bool) -> None:
        # Finish an in-flight tick before releasing keys, so it cannot re-press
        # movement after disable returns.
        with self._tick_lock:
            with self._lock:
                self.enabled = bool(enabled)
                if not self.enabled:
                    self.cmd = Command.NONE
                    self._command_deadline = None
                    self._forget_capture()
                    self._requests.clear()
            if not self.enabled:
                self.release_all()

    def release_all(self) -> None:
        with self._tick_lock:
            for k in (self._held_x, self._held_y):
                if k:
                    self.backend.up(k)
            self._held_x = self._held_y = None
            self._facing_x = None
            self._face_ready_at = 0.0
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
        self.set_enabled(False)

    # -- one tick --------------------------------------------------------
    def tap(self, key: str, duration: float = TAP_S) -> bool:
        return self._tap(key, duration)

    def _tap(self, key: str, duration: float = TAP_S, *, hp_request: bool = False) -> bool:
        if not key or self._input_stopped():
            return False
        self.backend.down(key)
        if hp_request:
            with self._lock:
                self.hp_tap_time = self.clock()
                self.hp_taps_sent += 1
        try:
            self.sleep(duration)
        finally:
            self.backend.up(key)
        return True

    def _key(self, name: str) -> str:
        return str(self.keys.get(name) or "")

    def apply_once(self, now: float) -> None:
        with self._tick_lock:
            self._apply_once(now)

    def _apply_once(self, now: float) -> None:
        if self._input_stopped():
            return
        with self._lock:
            cmd = self.cmd
            navigation_only = self._navigation_only
            hp_only = self._hp_only
            wall_now = self.clock()
            while self._requests and self._requests[0][1] <= wall_now:
                self._requests.popleft()
            req = self._requests.popleft()[0] if self._requests and not navigation_only else None

        if hp_only:
            self._release_movement()
            if req == "hp_potion":
                self._tap(self._key(req), hp_request=True)
            self._input_stopped()
            return

        if req is not None:
            self._tap(self._key(req), hp_request=req == "hp_potion")
            self._input_stopped()
            return

        buff = None if navigation_only else self._buffs.due(now)
        if buff is not None:
            self._release_movement()
            if self.tap(buff):
                self._buffs.mark(buff, now)
                self._freeze_until = now + self._buffs.freeze_s
            self._input_stopped()
            return
        if now < self._freeze_until:
            self._release_movement()
            return

        self._apply_hold(cmd.move_x, "x", now)
        self._apply_hold(cmd.move_y, "y", now)
        if self._apply_facing(cmd, now):
            self._apply_action(cmd, now)
        if not navigation_only:
            self._apply_pickup(now)
        self._last_cmd = cmd
        self._input_stopped()

    # -- pieces ----------------------------------------------------------
    def _input_stopped(self) -> bool:
        """An independent keyboard watchdog; stale commands never resume later."""
        with self._lock:
            fresh = (self.enabled and self._command_deadline is not None
                     and self.clock() < self._command_deadline and self.is_active())
            if fresh:
                return False
            if not (self.enabled and self.is_active()):
                self._forget_capture()
            # A lease that merely ran out keeps its capture memory: the same
            # capture must not be able to start the spent pulse a second time.
            self.cmd = Command.NONE
            self._command_deadline = None
            self._requests.clear()
        self.release_all()
        return True

    def _release_movement(self) -> None:
        for attr in ("_held_x", "_held_y"):
            k = getattr(self, attr)
            if k:
                self.backend.up(k)
                setattr(self, attr, None)

    def _apply_hold(self, want: str, axis: str, now: float) -> None:
        if self._input_stopped():
            return
        attr = "_held_x" if axis == "x" else "_held_y"
        held = getattr(self, attr)
        target = self._key(want) if want in ("left", "right", "up", "down") else None
        if held == target:
            return
        if held:
            self.backend.up(held)
        if target:
            self.backend.down(target)
            if axis == "x":
                self._facing_x = want
        setattr(self, attr, target)

    def _apply_facing(self, cmd: Command, now: float) -> bool:
        if cmd.face_x == "none":
            return True
        if self._facing_x != cmd.face_x:
            key = self._key(cmd.face_x)
            if not key:
                return False
            if not self.tap(key):
                return False
            self._facing_x = cmd.face_x
            self._face_ready_at = now + TAP_S + FACE_DELAY_S
        return now >= self._face_ready_at

    def _apply_action(self, cmd: Command, now: float) -> None:
        a = cmd.action
        if a in ("attack", "aoe", "normal_attack"):
            kind = "aoe" if (a == "aoe" and self._key("aoe")) else "attack"
            key = self._key("normal_attack" if a == "normal_attack" else kind)
            if key and self._attack.ready(kind, now):
                if self.tap(key):
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
        if not key or not self.enabled or not self.is_active():
            return
        interval = float(self.keys.get("pickup_interval_s", 0.3))
        if self._last_pickup is None or now - self._last_pickup >= interval:
            if self.tap(key):
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
