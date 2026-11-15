"""Validate an explicit return request against the latest captured game state."""
import math
import time
from copy import deepcopy

from maplebot.nav.circuit import MinimapCircuit


def valid_health(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 100


def request_hold(bot, request, now=None) -> bool:
    """Turn an explicit fresh request into a farming hold, never ordinary input."""
    bot.pause_all()
    bot._hold_request_error = None
    stamp = time.time() if now is None else now
    created = request.get("created") if isinstance(request, dict) else None
    reason = None
    if (not isinstance(created, (int, float)) or not math.isfinite(created)
            or not 0 <= stamp - created <= 30):
        reason = "Hold request expired; press Hold farming again."
    elif bot.brain.movement_test:
        reason = "Finish the connection test before holding farming."
    elif not bot.cfg["health"].get("enable", True) or not str(bot.cfg["keys"].get("hp_potion") or "").strip():
        reason = "Enable HP protection and configure the HP key first."
    elif not bot.map_ready or bot.map_recognizer is None:
        reason = "The current map must match its saved profile."
    elif (bot.capture_state != "fresh" or bot.capture_time is None
          or not math.isfinite(bot.capture_time)
          or not 0 <= time.monotonic() - bot.capture_time <= .75):
        reason = "A fresh game frame is required before holding farming."
    elif not bot._minimap_seen or bot.minimap_pos is None:
        reason = "The current minimap position is not visible."
    if reason:
        bot._hold_request_error = reason
        return False
    bot.hold_farming()
    return True


def request_return(bot, request, now=None) -> bool:
    """Retire ordinary input first; only fresh, healthy, recognized state may arm."""
    bot.paused = True
    if bot.keyboard is not None:
        bot.keyboard.set_enabled(False)
    bot._return_request_error = None
    stamp = time.time() if now is None else now
    nav = bot.brain.circuit
    reason = None
    created = request.get("created") if isinstance(request, dict) else None
    if (not isinstance(created, (int, float)) or not math.isfinite(created)
            or not 0 <= stamp - created <= 30):
        reason = "Return request expired; press Return to point 1 again."
    elif nav is None:
        reason = "Enable a saved circuit before returning."
    elif bot.brain.movement_test or getattr(nav, "movement_test", False):
        reason = "Finish the connection test before returning."
    elif not bot.map_ready or bot.map_recognizer is None:
        reason = "The current map must match its saved profile."
    elif (bot.capture_state != "fresh" or bot.capture_time is None
          or not math.isfinite(bot.capture_time)
          or not 0 <= time.monotonic() - bot.capture_time <= .75):
        reason = "A fresh game frame is required before returning."
    elif not bot._minimap_seen or bot.minimap_pos is None:
        reason = "The current character position is not visible on the minimap."
    else:
        hp = getattr(bot.last_snapshot, "hp", None)
        if not valid_health(hp):
            reason = "HP is unreadable; return stopped."
        elif hp < nav.return_min_hp_percent:
            reason = f"Restore HP above {nav.return_min_hp_percent:g}% before returning."
        elif not nav.request_return_to_start():
            reason = nav.recovery_reason or "A return is already running or unavailable."
    if reason is not None:
        bot._return_request_error = reason
        return False
    bot.resume()
    return True


class StartupReturn:
    """Start farming only after a fresh, HP-guarded physical return to point 1."""

    def __init__(self, bot):
        self.bot = bot
        self.state = "idle"
        self.reason = None
        self._generation = None
        self._requested = None
        self._nav = None
        self._prior_nav = None

    @property
    def active(self):
        return self.state in ("checking", "returning")

    def request(self, now=None):
        bot = self.bot
        with bot._control_lock:
            if self.active:
                return False
            if bot.brain.movement_test:
                return False
            self.state = "checking"
            self.reason = "Checking HP, map, and position before returning to point 1."
            self._requested = time.monotonic() if now is None else now
            self._nav = None
            self._prior_nav = None
            bot._return_request_error = None
            bot.hold_farming(self.reason)
            if bot.health_protection is not None:
                bot.health_protection.request_resume()
            self._generation = bot._control_generation
            return True

    def cancel(self, reason="Start cancelled."):
        with self.bot._control_lock:
            if not self.active:
                return
            if self._nav is not None and self._nav.returning_to_start:
                self._nav.abort_return_to_start(reason)
            if self._nav is not None and self.bot.brain.circuit is self._nav:
                self.bot.brain.circuit = self._prior_nav
            self.state, self.reason = "cancelled", reason

    def toggle_pause(self):
        """F1 cancels an active start; a paused controller queues a guarded start."""
        with self.bot._control_lock:
            health_hold = (not self.bot.full_pause and (
                self.bot._route_health_hold or self.bot._hp_health_hold))
            if self.active or not self.bot.paused or health_hold:
                self.cancel("Start cancelled with F1.")
                self.bot.pause_all()
                return False
            return self.request()

    def _fail(self, reason):
        if self._nav is not None and self._nav.returning_to_start:
            self._nav.abort_return_to_start(reason)
        self.state, self.reason = "failed", str(reason)
        self.bot.hold_farming("Start stopped: " + str(reason))

    def update(self, now=None):
        """Called after a controller step; never grants keys from a cached sample."""
        bot = self.bot
        stamp = time.monotonic() if now is None else now
        with bot._control_lock:
            if not self.active:
                return
            if bot._control_generation != self._generation or bot.full_pause:
                self.cancel("Start cancelled by a newer pause or hold request.")
                return
            if self.state == "returning":
                if self._nav.recovery_terminal:
                    self._fail(self._nav.recovery_reason or "Return to point 1 failed.")
                elif not self._nav.returning_to_start and self._nav.recovery_state == "completed":
                    # The startup return has its own allowance. A blocked farming
                    # lap earns its normal recovery allowance only after this proof.
                    self._nav.return_count = 0
                    self.state, self.reason = "completed", None
                return
            if stamp - self._requested > 90:
                self._fail("Timed out waiting for a fresh, healthy starting position.")
                return
            cfg = bot.cfg.get("route", {}).get("circuit", {})
            if not cfg.get("enabled") or bot.brain.circuit is None:
                self._fail("Enable a saved circuit before starting.")
                return
            if (not bot.cfg["health"].get("enable", True)
                    or not str(bot.cfg["keys"].get("hp_potion") or "").strip()):
                self._fail("Enable HP protection and configure the HP key before starting.")
                return
            guard = bot.health_protection
            if guard is not None and guard.latched:
                self._fail(guard.reason or "HP protection needs attention.")
                return
            if not bot.map_ready or bot.map_recognizer is None:
                self.reason = "Waiting for the saved map to be recognized."
                return
            if (bot.capture_state != "fresh" or bot.capture_time is None
                    or not math.isfinite(bot.capture_time)
                    or bot.capture_time <= self._requested
                    or not 0 <= stamp - bot.capture_time <= .75):
                self.reason = "Waiting for a fresh game frame."
                return
            if not bot._minimap_seen or bot.minimap_pos is None:
                self.reason = "Waiting for the current minimap position."
                return
            if bot.keyboard is None or not bot.keyboard.is_active():
                self.reason = "Waiting for the game to be in the foreground."
                return
            hp = getattr(bot.last_snapshot, "hp", None)
            if not valid_health(hp) or guard is None:
                self.reason = "Waiting for a fresh HP reading."
                return
            if (guard.state in ("healing", "awaiting_response")
                    or hp < max(guard.threshold, bot.brain.circuit.return_min_hp_percent)):
                self.reason = "Restoring HP before returning to point 1."
                return
            try:
                startup_cfg = deepcopy(cfg)
                startup_cfg["start_idx"] = 0
                nav = MinimapCircuit(startup_cfg)
                if nav.movement_test or not nav.request_return_to_start():
                    raise ValueError("A normal saved circuit is required before starting.")
            except (TypeError, ValueError) as error:
                self._fail(str(error))
                return
            bot.brain.suspend(stamp)
            self._prior_nav = bot.brain.circuit
            bot.brain.circuit = self._nav = nav
            self.state, self.reason = "returning", "Returning to point 1 before farming."
            bot.resume()  # grants no keys; the next fresh Bot step applies all guards
            self._generation = bot._control_generation
