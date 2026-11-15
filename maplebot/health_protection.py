"""Fresh-reading HP protection driven by confirmed keyboard events, without I/O."""
from __future__ import annotations

import math


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class HealthProtection:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enable", True))
        self.threshold = cfg.get("hp_percent", 60)
        self.minimum_hp_percent = cfg.get("minimum_hp_percent", 50)
        rearm = cfg.get("rearm_delta", 5)
        self.retry_s = cfg.get("protection_retry_s", 1.0)
        self.max_unconfirmed_taps = cfg.get("protection_max_taps", 3)
        if (not _finite(self.minimum_hp_percent) or not _finite(self.threshold)
                or not 0 <= self.minimum_hp_percent < self.threshold <= 100):
            raise ValueError("health requires 0 <= minimum_hp_percent < hp_percent <= 100")
        if not _finite(rearm) or not 0 <= rearm <= 100:
            raise ValueError("health.rearm_delta must be finite and between 0 and 100")
        if not _finite(self.retry_s) or not .5 <= self.retry_s <= 10:
            raise ValueError("health.protection_retry_s must be between 0.5 and 10 seconds")
        if type(self.max_unconfirmed_taps) is not int or not 1 <= self.max_unconfirmed_taps <= 5:
            raise ValueError("health.protection_max_taps must be an integer between 1 and 5")
        self.target = min(100, self.threshold + rearm)
        self.state = "monitoring" if self.enabled else "suspended"
        self.reason = None
        self.latched = False
        self.taps_sent = 0
        self.latest_observed_taps = None
        self.latest_tap_time = None
        self._unconfirmed_taps = 0
        self._episode = False
        self._baseline = None
        self._pending_since = self._response_since = None
        self._last_request = self._last_sent = None
        self._last_now = None
        self._suspended = False
        self._resume_requested = False

    @property
    def unconfirmed_taps(self):
        return self._unconfirmed_taps

    def suspend(self) -> None:
        """Pause timers without erasing failed taps or pending response evidence."""
        self._suspended = True
        if not self.latched:
            self.state = "suspended"

    def request_resume(self) -> None:
        """Arm one fresh healthy-reading check; this never clears an alert itself."""
        self._resume_requested = True

    def _latch(self, state, reason):
        self.state, self.reason, self.latched = state, reason, True
        return False

    def _reset_episode(self):
        self._episode = False
        self._baseline = None
        self._unconfirmed_taps = 0
        self._pending_since = self._response_since = None
        self._last_request = self._last_sent = None
        self.state, self.reason, self.latched = "monitoring", None, False

    def observe(self, hp, now: float, *, taps: int, tap_time: float | None,
                key_available: bool = True) -> bool:
        """Observe a new HP reading and return whether to request the HP key.

        The caller must suspend rather than observe cached/missing frames. ``taps``
        is a cumulative backend counter; only its increases count as sent potions.
        """
        if not _finite(now) or (self._last_now is not None and now < self._last_now):
            raise ValueError("health observation time must be finite and monotonic")
        was_suspended = self._suspended
        if was_suspended and self._last_now is not None:
            pause_s = now - self._last_now
            for attr in ("_pending_since", "_response_since", "_last_request", "_last_sent"):
                value = getattr(self, attr)
                if value is not None:
                    setattr(self, attr, value + pause_s)
        self._last_now, self._suspended = now, False
        rearm = self._resume_requested
        self._resume_requested = False
        if not self.enabled:
            self.state = "suspended"
            return False
        if type(taps) is not int or taps < 0:
            return self._latch("needs_attention", "HP key counter is unavailable")
        delta = 0 if self.latest_observed_taps is None else max(0, taps - self.latest_observed_taps)
        self.latest_observed_taps = taps
        self.taps_sent += delta
        if delta:
            self.latest_tap_time = tap_time if _finite(tap_time) and tap_time <= now else None
        if not _finite(hp) or not 0 <= hp <= 100:
            return False if self.latched else self._latch("needs_attention", "HP is unreadable")
        if hp == 0:
            return self._latch("dead", "HP is zero")
        if not key_available:
            return False if self.latched else self._latch("needs_attention", "HP key is not configured")
        if self.latched:
            if rearm and hp >= self.target:
                self._reset_episode()
            return False
        if not self._episode:
            if hp >= self.threshold:
                self.state = "monitoring"
                return False
            self._episode, self._baseline = True, hp
        if hp >= self.target:
            self._reset_episode()
            return False
        if delta:
            self._unconfirmed_taps += delta
            # A newly observed event after pause gets an active response window;
            # wall-clock time spent paused cannot declare that potion failed.
            sent_at = now
            if not was_suspended and self.latest_tap_time is not None:
                sent_at = max(self._pending_since if self._pending_since is not None else now,
                              self.latest_tap_time)
            self._last_sent = self._response_since = sent_at
            self._pending_since = None
        if self._baseline is not None and hp >= self._baseline + 2:
            self._unconfirmed_taps = 0
            self._baseline = hp
            self._response_since = None
        if self._pending_since is not None:
            self.state = "healing"
            if now - self._pending_since >= 3:
                return self._latch("needs_attention", "HP key was not sent")
            if now - self._last_request >= max(1.0, self.retry_s):
                self._last_request, self._baseline = now, hp
                return True
            return False
        if self._response_since is not None:
            self.state = "awaiting_response"
            if now - self._response_since < self.retry_s:
                return False
            if self._unconfirmed_taps >= self.max_unconfirmed_taps:
                return self._latch("needs_attention", "HP did not improve after confirmed potion presses")
            self._response_since = None
        self.state = "healing"
        if ((self._last_request is not None and now - self._last_request < self.retry_s)
                or (self._last_sent is not None and now - self._last_sent < self.retry_s)):
            return False
        self._baseline = hp
        self._pending_since = self._last_request = now
        return True
