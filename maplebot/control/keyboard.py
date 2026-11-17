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
# Upper bounds (ms) of the real-hold histogram for leased movement pulses; the
# last bucket counts everything longer.
HOLD_BUCKETS_MS = (20, 40, 60, 80, 120)
# A leased movement pulse that waits for its key-down must go down within its
# own length plus this grace after the grant (and inside its frame limit), or
# it is dropped unpressed.  Two blocking taps: a pickup tap holding the tick
# lock while the pulse is granted, then one HP/MP request tick.
PRESS_GRACE_S = 2 * TAP_S
# Pulses granted with at most this length are the circuit's fine-alignment
# taps; pulse_stats reports them separately from loot/regain walks.
FINE_PULSE_MAX_S = 0.06
_PULSE_COUNTERS = ("granted", "pressed", "expired_before_press", "superseded_before_press", "continued")
log = get_logger()


def _hold_bucket(hold_s: float) -> str:
    ms = hold_s * 1000.0
    for bound in HOLD_BUCKETS_MS:
        if ms <= bound + 1e-6:  # float noise must not move an exact 40 ms hold up a bucket
            return f"le{bound}"
    return f"gt{HOLD_BUCKETS_MS[-1]}"


class _PulseGrant:
    """One leased command for one capture (or one unstamped send).

    ``duration`` is the pulse's own length.  For a movement pulse that still
    waits for its key-down it is not yet running: ``end`` stays ``None`` and
    the grant may wait no later than ``min(press_by, limit)``, where ``limit``
    is the capture-age / watchdog bound of the frame it came from and
    ``press_by`` is the grant time plus its length plus ``PRESS_GRACE_S``.
    Movement keys still held from an earlier command while it waits are
    released by ``others_by``, the deadline the grant had before pulses were
    anchored at key-down.  Once its key goes down, ``end`` is the absolute
    release deadline, ``min(key_down + duration, limit)``.  ``fine`` is fixed
    at the first grant (diagnostics only).
    """

    __slots__ = ("cmd", "duration", "limit", "end", "granted_at", "press_by", "others_by", "fine")

    def __init__(self, cmd: Command, duration: float, limit: float, granted_at: float) -> None:
        self.cmd = cmd
        self.duration = duration
        self.limit = limit
        self.end: float | None = None
        self.granted_at = granted_at
        self.press_by = granted_at + duration + PRESS_GRACE_S
        self.others_by = min(granted_at + duration, limit)
        self.fine = duration <= FINE_PULSE_MAX_S + 1e-9

    def tighten(self, duration: float, limit: float, call_now: float) -> None:
        """A same-capture repeat may only shorten the grant."""
        self.duration = min(self.duration, duration)
        self.limit = min(self.limit, limit)
        self.press_by = min(self.press_by, call_now + duration + PRESS_GRACE_S)
        self.others_by = min(self.others_by, call_now + duration, limit)

    @property
    def wait_deadline(self) -> float:
        return min(self.press_by, self.limit)


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
        self._pulse_deadlines: dict[Command, _PulseGrant] = {}
        # The current command when it is a leased movement pulse whose key has
        # not gone down yet (its own duration starts at the worker's key-down).
        self._pending_pulse: _PulseGrant | None = None
        self._navigation_only = False
        self._hp_only = False
        self.hp_taps_sent = 0
        self.hp_tap_time: float | None = None
        self.mp_taps_sent = 0  # actual MP key-downs, mirroring hp_taps_sent
        # Read-only delivery instrumentation for leased movement pulses; it
        # never feeds back into control.  Guarded by ``_lock``.
        # "all" covers every leased movement pulse; "fine" only those granted
        # with a length up to FINE_PULSE_MAX_S (circuit fine alignment).
        self._pulse_stats = {kind: self._empty_pulse_stats() for kind in ("all", "fine")}
        self._hold_keys: tuple[str, ...] | None = None  # worker-only, under _tick_lock
        self._hold_started = 0.0
        self._hold_fine = False
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
        ``source_time`` never extends it: the shortest length and earliest
        deadline granted for that capture win, also after the worker already
        expired the pulse.  A new capture stamp, a different command, and
        unleased holds renew normally.

        A leased *movement* pulse whose keys are all up starts its own length
        at the worker's key-down, so a tick spent in a blocking tap (pickup, HP)
        or a late tick cannot consume a 40 ms pulse before it is pressed.  Until
        then it may wait only as long as its frame allows (``command_timeout_s``
        from this call and ``source_time + max_frame_age_s``); after key-down
        it is still released by that same bound.  A pulse whose key is already
        held continues from this call, exactly like an unleased renewal.
        """
        # Lease time is read before waiting on the tick, as it always was: a
        # renewal cannot gain the time it spent waiting for the lock.
        call_now = self.clock()
        duration, limit = self._lease_terms(source_time, valid_for_s,
                                            command_valid_for_s=cmd.valid_for_s, now=call_now)
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
                deadline = min(call_now + duration, limit)
                pending = None
                if cmd.valid_for_s is not None:
                    # Command equality ignores valid_for_s, so a repeat with a
                    # different pulse length is still the same pulse.
                    grant = self._pulse_deadlines.get(cmd) if source_time is not None else None
                    repeat = grant is not None
                    if grant is None:
                        grant = _PulseGrant(cmd, duration, limit, call_now)
                        if source_time is not None:
                            self._pulse_deadlines[cmd] = grant
                    else:
                        grant.tighten(duration, limit, call_now)
                        if grant.end is not None:
                            grant.end = min(grant.end, deadline)
                    keys = self._pulse_keys(cmd)
                    if grant.end is not None:
                        deadline = grant.end  # already running (or spent) for this capture
                    elif not keys or any(k in (self._held_x, self._held_y) for k in keys):
                        deadline = min(call_now + grant.duration, grant.limit)
                        grant.end = deadline
                        if keys and not repeat:
                            self._count("continued", grant)
                    else:
                        pending = grant
                        # The latest moment it may still be pressed: its own
                        # length plus a short grace, and never past its frame.
                        deadline = grant.wait_deadline
                if self._pending_pulse is not None and self._pending_pulse is not pending:
                    self._count("superseded_before_press", self._pending_pulse)
                if pending is not None and pending is not self._pending_pulse:
                    self._count("granted", pending)
                self._pending_pulse = pending
                self.cmd = cmd
                self._command_deadline = deadline

    def _pulse_keys(self, cmd: Command) -> tuple[str, ...]:
        """Bound movement keys a command holds."""
        keys = []
        for want in (cmd.move_x, cmd.move_y):
            if want in ("left", "right", "up", "down"):
                key = self._key(want)
                if key:
                    keys.append(key)
        return tuple(keys)

    def _forget_capture(self) -> None:
        """Drop same-capture pulse memory. Caller holds ``_lock``."""
        self._command_source_time = None
        self._pulse_deadlines.clear()

    def _drop_pending_pulse(self, *, superseded: bool = False) -> None:
        """Count a pulse that never went down. Caller holds ``_lock``."""
        if self._pending_pulse is None:
            return
        self._count("superseded_before_press" if superseded else "expired_before_press",
                    self._pending_pulse)
        self._pending_pulse = None

    @staticmethod
    def _empty_pulse_stats() -> dict:
        histogram = {_hold_bucket(b / 1000.0): 0 for b in HOLD_BUCKETS_MS}
        histogram[f"gt{HOLD_BUCKETS_MS[-1]}"] = 0
        stats = {name: 0 for name in _PULSE_COUNTERS}
        stats.update(last_press_delay_s=None, max_press_delay_s=None, last_hold_s=None,
                     hold_histogram_ms=histogram)
        return stats

    def _pulse_kinds(self, fine: bool) -> tuple[dict, ...]:
        return (self._pulse_stats["all"], self._pulse_stats["fine"]) if fine else (self._pulse_stats["all"],)

    def _count(self, name: str, grant: _PulseGrant) -> None:
        """Bump a delivery counter (diagnostics only). Caller holds ``_lock``."""
        for stats in self._pulse_kinds(grant.fine):
            stats[name] += 1

    def _lease_terms(self, source_time: float | None, valid_for_s: float | None = None,
                     *, command_valid_for_s: float | None = None,
                     now: float | None = None) -> tuple[float, float]:
        """Return (own length, frame/watchdog limit) for a grant made at ``now``."""
        if now is None:
            now = self.clock()
        duration = self.command_timeout_s if valid_for_s is None else float(valid_for_s)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("valid_for_s must be finite and positive")
        if command_valid_for_s is not None:
            duration = min(duration, command_valid_for_s)
        duration = min(duration, self.command_timeout_s)
        limit = now + self.command_timeout_s
        if source_time is not None:
            stamp = float(source_time)
            if not math.isfinite(stamp) or stamp > now:
                limit = now  # invalid/future capture stamps cannot extend permission
            else:
                limit = min(limit, stamp + self.max_frame_age_s)
        return duration, limit

    def _lease_deadline(self, source_time: float | None, valid_for_s: float | None = None,
                        *, command_valid_for_s: float | None = None) -> float:
        now = self.clock()
        duration, limit = self._lease_terms(source_time, valid_for_s,
                                            command_valid_for_s=command_valid_for_s, now=now)
        return min(now + duration, limit)

    def pulse_stats(self) -> dict:
        """Delivery counters for leased movement pulses (diagnostics only).

        ``granted`` counts pulses that had to wait for a key-down; each ends as
        ``pressed`` (its first key went down), ``expired_before_press`` (its
        press-by bound -- own length plus ``PRESS_GRACE_S`` -- or its frame
        limit ran out, or input stopped) or ``superseded_before_press``
        (another command or mode replaced it; this includes the normal case
        of the next frame's command arriving before the worker's key-down,
        and every pulse replaced during a buff freeze), or is still ``pending``:
        granted == pressed + expired_before_press + superseded_before_press + pending.
        ``continued`` counts leased renewals of a key that was already held.
        Hold times are real key-down to key-up of a pulse that started with a
        key-down, on the lease clock; a hold extended by later continuations
        is one physical hold.

        The top-level numbers cover every leased movement pulse (circuit,
        loot, regain).  ``fine`` holds the same fields for pulses whose first
        grant was at most ``FINE_PULSE_MAX_S`` long (the circuit's fine taps).
        """
        with self._lock:
            pending = self._pending_pulse
            out = self._pulse_stats_view(self._pulse_stats["all"], pending is not None)
            out["fine"] = self._pulse_stats_view(self._pulse_stats["fine"],
                                                 pending is not None and pending.fine)
            out["press_grace_s"] = PRESS_GRACE_S
            out["mp_taps_sent"] = self.mp_taps_sent
            return out

    @staticmethod
    def _pulse_stats_view(stats: dict, pending: bool) -> dict:
        out = {name: stats[name] for name in _PULSE_COUNTERS}
        out["pending"] = int(pending)
        for name in ("last_press_delay_s", "max_press_delay_s", "last_hold_s"):
            out[name] = stats[name]
        out["holds"] = sum(stats["hold_histogram_ms"].values())
        out["hold_histogram_ms"] = dict(stats["hold_histogram_ms"])
        return out

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
                self._drop_pending_pulse(superseded=True)
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
                    self._drop_pending_pulse()
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
                    self._movement_up(k)
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

    def _tap(self, key: str, duration: float = TAP_S, *, hp_request: bool = False,
             mp_request: bool = False) -> bool:
        if not key or self._input_stopped():
            return False
        self.backend.down(key)
        if hp_request:
            with self._lock:
                self.hp_tap_time = self.clock()
                self.hp_taps_sent += 1
        elif mp_request:
            with self._lock:
                self.mp_taps_sent += 1
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
            self._tap(self._key(req), hp_request=req == "hp_potion", mp_request=req == "mp_potion")
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
            now = self.clock()
            fresh = (self.enabled and self._command_deadline is not None
                     and now < self._command_deadline and self.is_active())
            if fresh:
                # While a pulse waits for its key-down, movement keys left from
                # the command it replaced keep that command's old bound: they
                # must not ride along through request/HP ticks to its press-by.
                # A waiting pulse has none of its own keys down, so every held
                # movement key is such a leftover.
                grant = self._pending_pulse
                leftover = (grant is not None and now >= grant.others_by
                            and (self._held_x is not None or self._held_y is not None))
            else:
                if not (self.enabled and self.is_active()):
                    self._forget_capture()
                # A lease that merely ran out keeps its capture memory: the same
                # capture must not be able to start the spent pulse a second time.
                self._drop_pending_pulse()
                self.cmd = Command.NONE
                self._command_deadline = None
                self._requests.clear()
        if fresh:
            if leftover:
                with self._tick_lock:
                    self._release_movement()
            return False
        self.release_all()
        return True

    def _start_pending_pulse(self, down: float) -> None:
        """Start a waiting pulse at its first key-down (worker tick only).

        Called right after ``backend.down`` of a movement key, so a key that
        physically went down is always counted as pressed and its hold timed,
        even if the watchdog releases it before the tick goes on.
        """
        with self._lock:
            grant = self._pending_pulse
            if grant is None:
                return
            keys = self._pulse_keys(grant.cmd)
            # A waiting pulse has none of its keys down, so any held key of it
            # was just pressed by this tick.
            if not keys or not any(k in (self._held_x, self._held_y) for k in keys):
                return
            # Still bounded by the frame: a pulse never outlives its capture.
            grant.end = min(down + grant.duration, grant.limit)
            self._command_deadline = grant.end
            self._pending_pulse = None
            delay = max(0.0, down - grant.granted_at)
            for stats in self._pulse_kinds(grant.fine):
                stats["pressed"] += 1
                stats["last_press_delay_s"] = delay
                prior = stats["max_press_delay_s"]
                stats["max_press_delay_s"] = delay if prior is None else max(prior, delay)
            self._hold_keys, self._hold_started, self._hold_fine = keys, down, grant.fine

    def _movement_up(self, key: str) -> None:
        """Release a movement key, timing it when it ends a pulse's hold."""
        self.backend.up(key)
        keys = self._hold_keys
        if keys is None or key not in keys:
            return
        hold = max(0.0, self.clock() - self._hold_started)
        with self._lock:
            self._hold_keys = None
            for stats in self._pulse_kinds(self._hold_fine):
                stats["last_hold_s"] = hold
                stats["hold_histogram_ms"][_hold_bucket(hold)] += 1

    def _release_movement(self) -> None:
        for attr in ("_held_x", "_held_y"):
            k = getattr(self, attr)
            if k:
                self._movement_up(k)
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
            self._movement_up(held)
        setattr(self, attr, None)
        if target:
            down = self.clock()
            self.backend.down(target)
            setattr(self, attr, target)
            if axis == "x":
                self._facing_x = want
            self._start_pending_pulse(down)

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
