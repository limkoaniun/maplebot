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
        # Optional: size leased approach pulses from the remaining distance
        # (minimap px/s). Unset keeps the fixed 0.04-second taps.
        self.walk_speed_x = float(cfg.get("walk_speed_x") or 0)
        # Optional: remember begun/probable kill sites displaced by a newer target.
        self.max_pending = int(cfg.get("max_pending", 0))
        self.pending_s = float(cfg.get("pending_s", 8))
        self.switch_x = float(cfg.get("switch_x", 2.5))
        values = (self.disappear_s, self.settle_s, self.max_approach_s,
                  self.tolerance_x, self.tolerance_y, self.scale)
        if not all(math.isfinite(v) for v in values) or min(values) < 0 or min(values[2:]) <= 0:
            raise ValueError("loot timings/tolerances and minimap scale must be finite and valid")
        extra = (self.walk_speed_x, self.pending_s, self.switch_x)
        if (not all(math.isfinite(v) for v in extra) or min(extra) < 0 or self.walk_speed_x > 20
                or not 0 <= self.max_pending <= 4 or self.pending_s > 30):
            raise ValueError("loot walk speed and pending-site limits must be finite and bounded")
        if self.max_pending and not self.walk_speed_x:
            # A remembered site can be many pixels away; that walk must be leased.
            raise ValueError("loot.max_pending needs loot.walk_speed_x for leased walk-backs")
        self._raw_now = None
        self._clock = ActiveClock()
        self._pending = []
        self.cancel()

    def cancel(self):
        """Drop the current visit and every remembered site."""
        self._pending = []
        self._clear_current()

    def _clear_current(self):
        self.state = "idle"
        self.goal = None
        self._seen = self._empty_since = self._started = self._arrived = None
        self._visual_since = self._visual_context = self._visual_state = None
        self._platform_bounds = self._platform = self._deadline = None
        self._reset_approach()

    @property
    def pending_count(self):
        return len(self._pending)

    def _defer(self, t):
        """Queue the current goal for one later bounded visit on the same floor."""
        if self.max_pending <= 0 or self.goal is None:
            return
        # A resumed site keeps its first deadline: displacement never renews it.
        deadline = self._deadline if self._deadline is not None else t + self.pending_s
        self._pending = [site for site in self._pending if abs(site[0][0] - self.goal[0]) > self.tolerance_x]
        if t < deadline:
            self._pending.append((self.goal, self._platform_bounds, self._platform, deadline))
        del self._pending[:-self.max_pending]

    def yield_to_target(self, now):
        """A living target takes priority; a visit already begun is kept for later."""
        t = self._clock.update(now)
        if self._started is not None:
            self._defer(t)
        self._clear_current()

    def _next_pending(self, pos, t, platform_bounds):
        """Start the nearest remembered site that is still valid for this floor."""
        self._pending = [site for site in self._pending
                         if t < site[3] and site[1] is not None and platform_bounds is not None
                         and tuple(site[1]) == tuple(platform_bounds)]
        if not self._pending:
            return
        site = min(self._pending, key=lambda item: abs(item[0][0] - pos[0]))
        self._pending.remove(site)
        self.goal, self._platform_bounds, self._platform, self._deadline = site
        self._seen = self._started = t
        self._empty_since = t - self.disappear_s
        self.state = "approaching"

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
            if self.walk_speed_x > 0:  # leased, and short of the brake zone
                room = min(abs(dx), edge_distance) - 4.0
                return Command(direction, valid_for_s=min(.5, max(.04, room / self.walk_speed_x)))
            return Command(direction)
        if self._pulse_until is None:
            self._pulse_until, self._pulse_on = now + .45, False
        elif now >= self._pulse_until:
            self._pulse_on = not self._pulse_on
            self._pulse_until = now + (self._pulse_s(abs(dx), edge_distance) if self._pulse_on else .45)
        elif self._pulse_on and self.walk_speed_x > 0:
            # A fresh observation may only shorten a running pulse, never extend it.
            self._pulse_until = min(self._pulse_until, now + self._pulse_s(abs(dx), edge_distance))
        return (Command(direction, valid_for_s=self._pulse_until - now)
                if self._pulse_on else Command.NONE)

    def _pulse_s(self, distance, edge_distance):
        """Leased pulse length: stops short of the goal and of the outward edge."""
        if self.walk_speed_x <= 0:
            return .04
        room = min(distance, edge_distance - 1.0)
        return min(.5, max(.04, .7 * room / self.walk_speed_x))

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

    def observe(self, minimap_pos, player_pos, target_center, now, minimap_w=None, platform_bounds=None,
                platform=None, visible_x=None):
        """Track the attacked target. `platform` supplies the surface height at
        the goal; `visible_x` lists the minimap X of every current detection so
        a target that vanished while another is fought keeps its drop site."""
        t = self._clock.update(now)
        x = None
        if self.enabled and minimap_pos is not None and player_pos is not None:
            x = minimap_pos[0] + (target_center[0] - player_pos[0]) / self.scale
        if self.goal is not None and (self._started is not None or (
                x is not None and visible_x is not None and self.state == "tracking"
                and abs(x - self.goal[0]) > self.switch_x
                and all(abs(other - self.goal[0]) > self.switch_x for other in visible_x))):
            self._defer(t)
        if visible_x is not None:  # a monster still standing on a remembered site is not a drop
            self._pending = [site for site in self._pending
                             if all(abs(other - site[0][0]) > self.switch_x for other in visible_x)]
        self._clear_current()
        if x is None:
            return
        if (x < 0 or (minimap_w is not None and x >= minimap_w)
                or (platform_bounds is not None and not platform_bounds[0] <= x <= platform_bounds[1])):
            return
        self.goal = (x, float(platform.y_at(x)) if platform is not None else float(minimap_pos[1]))
        self._platform = platform
        self._platform_bounds = tuple(platform_bounds) if platform_bounds is not None else None
        self._seen = t
        self.state = "tracking"

    def step(self, pos, now, has_detections=False, has_player=True, platform_bounds=None):
        t = self._clock.update(now, pos is not None and has_player)
        if self._raw_now is not None and not 0 <= now - self._raw_now <= 1.0:
            self._reset_approach()  # the active clock froze: never resume an old pulse
        self._raw_now = now
        if not self.enabled:
            return None
        if self.goal is None and self._pending and pos is not None and has_player and not has_detections:
            self._next_pending(pos, t, platform_bounds)
        if self.goal is None:
            return None
        off_surface = pos is None or (
            self._platform.support_error(pos) > self.tolerance_y if self._platform is not None
            else abs(pos[1] - self.goal[1]) > self.tolerance_y)
        if (pos is None or not has_player or off_surface
                or (self._platform_bounds is not None
                    and not self._platform_bounds[0] <= pos[0] <= self._platform_bounds[1])
                or (platform_bounds is not None and not platform_bounds[0] <= self.goal[0] <= platform_bounds[1])):
            self.cancel()
            return None
        if has_detections:
            self._empty_since = None
            self._reset_approach()
            if t - self._seen > self.max_approach_s:
                self._clear_current()
            return None  # neither a direction tie nor an out-of-range detection is a kill
        if self._empty_since is None:
            self._empty_since = t
        if t - self._empty_since < self.disappear_s:
            self.state = "settling"
            return Command.NONE
        if self._started is None:
            self._started = t
        if t - self._started >= self.max_approach_s:
            self._clear_current()
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
        self._clear_current()
        return None
