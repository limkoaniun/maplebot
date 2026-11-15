"""Short positional pickup visits after an attack target disappears."""
from __future__ import annotations

import math

from maplebot.nav.circuit import ActiveClock
from maplebot.types import Command


class LootCollector:
    def __init__(self, cfg: dict, pixels_per_minimap_x: float = 10):
        self.enabled = bool(cfg.get("enabled", False))
        self.disappear_s = float(cfg.get("disappear_s", 0.25))
        self.settle_s = float(cfg.get("settle_s", 0.8))
        self.max_approach_s = float(cfg.get("max_approach_s", 2))
        self.tolerance_x = float(cfg.get("tolerance_x", 1.5))
        self.tolerance_y = float(cfg.get("tolerance_y", 2))
        self.scale = float(pixels_per_minimap_x)
        values = (self.disappear_s, self.settle_s, self.max_approach_s,
                  self.tolerance_x, self.tolerance_y, self.scale)
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or min(values[2:]) <= 0:
            raise ValueError("loot timings/tolerances and minimap scale must be finite and valid")
        self._clock = ActiveClock()
        self.cancel()

    def cancel(self):
        self.state = "idle"
        self.goal = None
        self._seen = self._empty_since = self._started = self._arrived = None
        self._visual_since = self._visual_context = self._visual_state = None
        self._platform_bounds = None
        self._reset_approach()

    def _reset_approach(self):
        self._pulse_direction = self._pulse_until = None
        self._pulse_on = False

    def _approach(self, dx, pos, now):
        """Brake near the goal/edge; timed pulses expire without another frame."""
        direction = "left" if dx < 0 else "right"
        edge_distance = math.inf
        if self._platform_bounds is not None:
            edge_distance = (pos[0] - self._platform_bounds[0] if direction == "left"
                             else self._platform_bounds[1] - pos[0])
        precise = abs(dx) <= 5.0 or edge_distance <= 5.0
        if self._pulse_direction != direction:
            reversing = self._pulse_direction is not None
            self._pulse_direction = direction
            self._pulse_on = False
            self._pulse_until = now + .45 if precise or reversing else None
        if not precise:
            if self._pulse_until is not None and not self._pulse_on and now < self._pulse_until:
                return Command.NONE
            self._pulse_until, self._pulse_on = None, False
            return Command(direction)
        if self._pulse_until is None:
            self._pulse_until, self._pulse_on = now + .45, False
        elif now >= self._pulse_until:
            self._pulse_on = not self._pulse_on
            self._pulse_until = now + (.04 if self._pulse_on else .45)
        return (Command(direction, valid_for_s=self._pulse_until - now)
                if self._pulse_on else Command.NONE)

    def visual_lock(self, now, present, context):
        """Preserve a committed visit for at most 0.6s, without granting input.

        The caller supplies a fresh, supported map/floor/route identity or None.
        Tracking a living target alone never earns this stopped reacquisition.
        """
        waiting = self._visual_since is not None
        if present and not waiting:
            return False
        if context is None or (waiting and (context != self._visual_context
                                           or not 0 <= now - self._visual_since <= 0.6)):
            self.cancel()
            return False
        if not waiting:
            if self.goal is None or self.state not in {"settling", "approaching", "dwelling"}:
                return False
            self._visual_since, self._visual_context, self._visual_state = now, context, self.state
        self._clock.update(now, False)  # stopped time cannot consume approach/dwell time
        self._reset_approach()  # a visual interruption never resumes an old movement pulse
        if present:
            self.state = self._visual_state
            self._visual_since = self._visual_context = self._visual_state = None
            return False
        self.state = "waiting_visual"
        return True

    def suspend(self, now):
        self.cancel()
        self._clock.update(now, False)

    def observe(self, minimap_pos, player_pos, target_center, now, minimap_w=None, platform_bounds=None):
        t = self._clock.update(now)
        self.cancel()
        if not self.enabled or minimap_pos is None or player_pos is None:
            return
        x = minimap_pos[0] + (target_center[0] - player_pos[0]) / self.scale
        if (x < 0 or (minimap_w is not None and x >= minimap_w)
                or (platform_bounds is not None and not platform_bounds[0] <= x <= platform_bounds[1])):
            return
        self.goal = (x, float(minimap_pos[1]))
        self._platform_bounds = tuple(platform_bounds) if platform_bounds is not None else None
        self._seen = t
        self.state = "tracking"

    def step(self, pos, now, has_detections=False, has_player=True, platform_bounds=None):
        t = self._clock.update(now, pos is not None and has_player)
        if not self.enabled or self.goal is None:
            return None
        if (pos is None or not has_player or abs(pos[1] - self.goal[1]) > self.tolerance_y
                or (self._platform_bounds is not None
                    and not self._platform_bounds[0] <= pos[0] <= self._platform_bounds[1])
                or (platform_bounds is not None and not platform_bounds[0] <= self.goal[0] <= platform_bounds[1])):
            self.cancel()
            return None
        if has_detections:
            self._empty_since = None
            self._reset_approach()
            if t - self._seen > self.max_approach_s:
                self.cancel()
            return None  # neither a direction tie nor an out-of-range detection is a kill
        if self._empty_since is None:
            self._empty_since = t
        if t - self._empty_since < self.disappear_s:
            self.state = "settling"
            return Command.NONE
        if self._started is None:
            self._started = t
        if t - self._started >= self.max_approach_s:
            self.cancel()
            return None
        dx = self.goal[0] - pos[0]
        if abs(dx) > self.tolerance_x:
            self.state = "approaching"
            self._arrived = None
            return self._approach(dx, pos, t)
        self._reset_approach()
        if self._arrived is None:
            self._arrived = t
        if t - self._arrived < self.settle_s:
            self.state = "dwelling"
            return Command.NONE
        self.cancel()
        return None
