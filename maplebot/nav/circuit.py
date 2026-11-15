"""A bounded, configurable lap over points in the currently visible minimap."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math

from maplebot.types import Command


class ActiveClock:
    """Count observed active time; a long capture/pause gap is not navigation time."""
    def __init__(self):
        self.time = 0.0
        self.last = None
        self.active = False

    def update(self, now: float, active: bool = True) -> float:
        if active and self.active and self.last is not None:
            dt = now - self.last
            if 0 <= dt <= 1.0:
                self.time += dt
        self.last, self.active = now, active
        return self.time


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    action: str = "walk"
    dwell_s: float = 0.5
    from_x: float | None = None
    from_y: float | None = None


@dataclass(frozen=True)
class Platform:
    y: float
    left: float
    right: float
    right_y: float | None = None
    knots: tuple[tuple[float, float], ...] = ()

    def __post_init__(self):
        if not isinstance(self.knots, (list, tuple)):
            raise ValueError("platform knots must be interior [x, y] pairs")
        normalized = []
        previous_x = self.left
        for point in self.knots:
            if (not isinstance(point, (list, tuple)) or len(point) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) for v in point)):
                raise ValueError("platform knots need finite [x, y] coordinates")
            x, y = map(float, point)
            if not previous_x < x < self.right or y < 0:
                raise ValueError("platform knots need strictly increasing interior X and nonnegative Y")
            normalized.append((x, y))
            previous_x = x
        object.__setattr__(self, "knots", tuple(normalized))

    @property
    def sloped(self) -> bool:
        return ((self.right_y is not None and self.right_y != self.y)
                or any(y != self.y for _, y in self.knots))

    def y_at(self, x: float) -> float:
        """Height on this segment; callers check horizontal containment separately."""
        endpoint_y = self.y if self.right_y is None else self.right_y
        if x <= self.left:
            return self.y
        if x >= self.right:
            return endpoint_y
        points = ((self.left, self.y), *self.knots, (self.right, endpoint_y))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x <= x1:
                return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
        return endpoint_y

    def support_range(self, x: float) -> tuple[float, float]:
        """Surface heights consistent with half-pixel minimap X quantization.

        This does not grant horizontal containment; callers still check observed X.
        """
        left = max(self.left, min(self.right, x - 0.5))
        right = max(self.left, min(self.right, x + 0.5))
        heights = [self.y_at(left), self.y_at(right)]
        heights.extend(y for knot_x, y in self.knots if left < knot_x < right)
        return min(heights), max(heights)

    def support_error(self, pos: tuple[float, float]) -> float:
        """Minimum vertical residual over the local quantized surface range."""
        low, high = self.support_range(pos[0])
        return max(low - pos[1], pos[1] - high, 0.0)


class MinimapCircuit:
    def __init__(self, cfg: dict):
        self._cfg = deepcopy(cfg)
        self.return_on_block = bool(cfg.get("return_on_block", False))
        self.movement_test = bool(cfg.get("movement_test", False))
        self.max_returns_per_lap = cfg.get("max_returns_per_lap", 1)
        self.return_timeout_s = float(cfg.get("return_timeout_s", 60))
        self.return_min_hp_percent = float(cfg.get("return_min_hp_percent", 40))
        if (type(self.max_returns_per_lap) is not int or not 1 <= self.max_returns_per_lap <= 3
                or not math.isfinite(self.return_timeout_s) or self.return_timeout_s <= 0):
            raise ValueError("return budget must be an integer from 1 to 3 and return timeout finite and positive")
        if not math.isfinite(self.return_min_hp_percent) or not 0 <= self.return_min_hp_percent <= 100:
            raise ValueError("return_min_hp_percent must be finite and between 0 and 100")
        self.returning_to_start = False
        self.recovery_state = "idle"
        self.recovery_reason = self.recovery_from_index = None
        self.blocked_origin_reason = None
        self.recovery = None
        self.return_count = 0
        self.recovery_terminal = False
        self._return_started = self._return_stable_since = self._return_stable_pos = None
        self._return_platform_id = None
        self._return_stable_frames = 0
        self.waypoints = [Waypoint(float(p["x"]), float(p["y"]), str(p.get("action", "walk")),
                                   float(p.get("dwell_s", 0.5)),
                                   float(p["from_x"]) if p.get("from_x") is not None else None,
                                   float(p["from_y"]) if p.get("from_y") is not None else None)
                          for p in cfg.get("waypoints", [])]
        self.platforms = [Platform(float(p["y"]), float(p["left"]), float(p["right"]),
                                   float(p["right_y"]) if p.get("right_y") is not None else None,
                                   p.get("knots", ()))
                          for p in cfg.get("platforms", [])]
        if any(not all(math.isfinite(v) and v >= 0 for v in (p.y, p.left, p.right))
               or (p.right_y is not None and (not math.isfinite(p.right_y) or p.right_y < 0))
               or p.left >= p.right for p in self.platforms):
            raise ValueError("circuit platforms need finite nonnegative y/left/right/right_y with left < right")
        self.platform_tolerance_y = float(cfg.get("platform_tolerance_y", 5))
        self.floor_rejoin_s = float(cfg.get("floor_rejoin_s", 0.4))
        if len(self.waypoints) < 2:
            raise ValueError("an enabled circuit needs at least two minimap waypoints")
        for point in self.waypoints:
            if (not all(math.isfinite(v) for v in (point.x, point.y, point.dwell_s))
                    or point.x < 0 or point.y < 0 or not 0 <= point.dwell_s <= 10
                    or point.action not in ("walk", "climb", "drop", "jump", "ramp")):
                raise ValueError("circuit waypoints need valid x/y, walk/climb/drop/jump/ramp, and dwell_s between 0 and 10")
            if point.action in ("jump", "ramp") and (point.from_x is None or point.from_y is None
                    or not all(math.isfinite(v) and v >= 0 for v in (point.from_x, point.from_y))
                    or (point.from_x, point.from_y) == (point.x, point.y)):
                raise ValueError(f"{point.action} waypoints require distinct finite takeoff from_x/from_y coordinates")
            if point.action == "ramp" and (point.from_x == point.x or point.from_y == point.y):
                raise ValueError("ramp waypoints need different X coordinates and different heights")
        self.tolerance_x = float(cfg.get("tolerance_x", 2))
        self.tolerance_y = float(cfg.get("tolerance_y", 2))
        self.align_tolerance_x = float(cfg.get("align_tolerance_x", 1))
        self.jump_align_tolerance_x = float(cfg.get("jump_align_tolerance_x", self.align_tolerance_x))
        self.drop_align_tolerance_x = float(cfg.get("drop_align_tolerance_x", self.align_tolerance_x))
        self.precision_walk = bool(cfg.get("precision_walk", False))
        self.approach_radius_x = float(cfg.get("approach_radius_x", 5))
        self.approach_pulse_s = float(cfg.get("approach_pulse_s", 0.1))
        self.approach_release_s = float(cfg.get("approach_release_s", 0.2))
        self.align_settle_s = float(cfg.get("align_settle_s", 0.25))
        self.climb_jitter_x = float(cfg.get("climb_jitter_x", 2))
        self.jump_flight_timeout_s = float(cfg.get("jump_flight_timeout_s", 2))
        self.jump_landing_s = float(cfg.get("jump_landing_s", 0.25))
        self.jump_landing_overshoot_x = float(cfg.get("jump_landing_overshoot_x", 0))
        self.jump_launch_s = float(cfg.get("jump_launch_s", 0.15))
        self.ramp_corridor_y = float(cfg.get("ramp_corridor_y", 4))
        self.ramp_landing_s = float(cfg.get("ramp_landing_s", 0.35))
        self.timeout_s = float(cfg.get("timeout_s", 12))
        self.recovery_s = float(cfg.get("recovery_s", 0.8))
        self.drop_interval_s = float(cfg.get("drop_interval_s", 1))
        self.max_retries = int(cfg.get("max_retries", 2))
        numbers = (self.tolerance_x, self.tolerance_y, self.align_tolerance_x, self.jump_align_tolerance_x,
                   self.drop_align_tolerance_x,
                   self.timeout_s, self.recovery_s, self.drop_interval_s,
                   self.approach_radius_x, self.approach_pulse_s, self.approach_release_s,
                   self.align_settle_s, self.climb_jitter_x, self.platform_tolerance_y, self.floor_rejoin_s,
                   self.jump_flight_timeout_s, self.jump_landing_s, self.jump_launch_s,
                   self.ramp_corridor_y, self.ramp_landing_s)
        if not all(math.isfinite(v) and v > 0 for v in numbers) or not 0 <= self.max_retries <= 5:
            raise ValueError("circuit tolerances/timers must be positive; max_retries must be 0 to 5")
        if self.jump_launch_s >= self.jump_flight_timeout_s:
            raise ValueError("jump_launch_s must be shorter than jump_flight_timeout_s")
        if not math.isfinite(self.jump_landing_overshoot_x) or not 0 <= self.jump_landing_overshoot_x <= 3:
            raise ValueError("jump_landing_overshoot_x must be finite and between 0 and 3")
        start = cfg.get("start_idx")
        if start is not None and (int(start) != start or not 0 <= int(start) < len(self.waypoints)):
            raise ValueError("circuit start_idx is outside the waypoint list")
        self.idx = int(start) if start is not None else 0
        self.loop = bool(cfg.get("loop", True))
        self.completed = False
        self._selected = start is not None
        self.retries = 0
        self.state = "initializing"
        self.reason = None
        self._clock = ActiveClock()
        self._started = None
        self._arrived = None
        self._recovery_until = None
        self._recovery_jumped = False
        self._last_drop = None
        self._climb_last_y = self._climb_progress = self._climb_jump = None
        self._vertical_direction = None
        self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
        self._reset_alignment()
        self._reset_jump()
        self._reset_drop()
        self._reset_ramp()
        if self.platforms:
            for point in self.waypoints:
                if point.action in ("jump", "ramp") and (self._containing_platform((point.from_x, point.from_y)) is None
                                                or self._containing_platform((point.x, point.y)) is None):
                    raise ValueError(f"{point.action} takeoff and landing must lie on configured platforms")

    @property
    def transition_active(self) -> bool:
        if self.returning_to_start:
            return True
        if self._selected and not self.completed and self.waypoints[self.idx].action in ("jump", "ramp"):
            return True
        if (self._selected and not self.completed and self.waypoints[self.idx].action == "drop"
                and self._drop_started):
            return True
        return (self._selected and self.waypoints[self.idx].action in ("climb", "drop")
                and self.state in ("climbing", "dropping", "recovering", "dwelling", "settling"))

    def suspend(self, now: float) -> None:
        self._clock.update(now, False)
        self._reset_return_stability()
        if self.recovery is not None and self.returning_to_start:
            self.recovery.suspend(now)
        self._reset_alignment()
        self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
        self._jump_landing_since = self._jump_landing_y = None
        self._jump_landing_frames = 0
        self._jump_destination_seen = False
        if self._jump_phase == "flight":
            self._jump_launch_ended = True
        self._drop_stable_y = self._drop_stable_since = None
        self._drop_launch_ended = True
        self._ramp_landing_pos = self._ramp_landing_since = self._ramp_source_y = None
        self._ramp_landing_frames = 0
        self._reset_ramp_retry_stability()

    def platform_at(self, pos) -> Platform | None:
        if pos is None or not self.platforms:
            return None
        matches = [p for p in self.platforms if abs(p.y_at(pos[0]) - pos[1]) <= self.platform_tolerance_y]
        return min(matches, key=lambda p: (max(p.left - pos[0], 0, pos[0] - p.right),
                   abs(p.y_at(pos[0]) - pos[1]))) if matches else None

    def _containing_platform(self, pos) -> Platform | None:
        matches = [p for p in self.platforms if p.left <= pos[0] <= p.right
                   and abs(p.y_at(pos[0]) - pos[1]) <= self.tolerance_y]
        return min(matches, key=lambda p: abs(p.y_at(pos[0]) - pos[1])) if matches else None

    def _height_at(self, x: float, y: float, observed_x: float) -> float:
        platform = self._containing_platform((x, y))
        return (platform.y_at(observed_x) if platform is not None
                and platform.left <= observed_x <= platform.right else y)

    def _support_range(self, x: float, y: float, observed_x: float) -> tuple[float, float]:
        platform = self._containing_platform((x, y))
        return (platform.support_range(observed_x) if platform is not None
                and platform.left <= observed_x <= platform.right else (y, y))

    def _support_error(self, x: float, y: float, pos) -> float:
        low, high = self._support_range(x, y, pos[0])
        return max(low - pos[1], pos[1] - high, 0.0)

    def _walk_floor_ready(self, pos, now: float) -> bool:
        """Rejoin only after an observed, stable fall to a known lower floor."""
        point = self.waypoints[self.idx]
        if not self.platforms or point.action != "walk":
            self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
            return True
        platform = self.platform_at(pos)
        destination = self._containing_platform((point.x, point.y))
        same_platform = destination is None or platform == destination
        local_y = self._height_at(point.x, point.y, pos[0])
        on_destination_slope = (destination is not None and destination.sloped
                                and destination.left <= pos[0] <= destination.right
                                and destination.support_error(pos) <= self.tolerance_y)
        on_slope = (platform is not None and platform.sloped
                    and platform.left <= pos[0] <= platform.right
                    and platform.support_error(pos) <= self.tolerance_y)
        if on_destination_slope or same_platform and (on_slope or (not (destination and destination.sloped)
                and (abs(local_y - pos[1]) <= self.tolerance_y
                     or (platform is not None and abs(platform.y - point.y) <= self.tolerance_y)))):
            self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
            return True  # e.g. the ground slope is a few pixels below its level floor
        choices = ([] if platform is None or (platform.sloped and not on_slope)
                   or platform.y_at(pos[0]) <= local_y + self.tolerance_y else
                   [(abs(p.x - pos[0]), i) for i, p in enumerate(self.waypoints)
                    if p.action == "walk" and self._containing_platform((p.x, p.y)) == platform
                    and platform.left <= p.x <= platform.right])
        if not choices:
            self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
            return False
        if self._rejoin_floor != platform or abs(pos[1] - self._rejoin_y) > 1:
            self._rejoin_floor, self._rejoin_y, self._rejoin_since = platform, pos[1], now
        if now - self._rejoin_since < self.floor_rejoin_s:
            return False
        self._set_waypoint(min(choices)[1], now)
        return True

    def _reset_alignment(self) -> None:
        self._aligned_since = None
        self._pulse_until = None
        self._pulse_direction = None
        self._pulse_on = False

    def _precision_approach(self, dx: float, now: float) -> Command:
        """Release before small corrections so momentum and capture lag can settle."""
        direction = "left" if dx < 0 else "right"
        self._aligned_since = None
        if abs(dx) > self.approach_radius_x:
            self._reset_alignment()
            return Command(direction)
        if self._pulse_until is None or self._pulse_direction != direction:
            self._pulse_direction = direction
            self._pulse_on = False
            self._pulse_until = now + self.approach_release_s
        elif now >= self._pulse_until:
            self._pulse_on = not self._pulse_on
            self._pulse_until = now + (self.approach_pulse_s if self._pulse_on else self.approach_release_s)
        # The keyboard must end a short correction even if the next vision frame
        # arrives late. Repeated decisions retain only this pulse's remaining time.
        return (Command(direction, valid_for_s=self._pulse_until - now)
                if self._pulse_on else Command.NONE)

    def _advance(self, now: float) -> None:
        if self.idx == len(self.waypoints) - 1:
            self.return_count = 0  # only completing the original lap replenishes its return budget
        if not self.loop and self.idx == len(self.waypoints) - 1:
            self.completed = True
            self.state = "completed"
            return
        self._set_waypoint((self.idx + 1) % len(self.waypoints), now)

    def _set_waypoint(self, idx: int, now: float) -> None:
        self.idx = idx
        self.retries = 0
        self._started = now
        self._arrived = self._recovery_until = self._last_drop = None
        self._climb_last_y = self._climb_progress = self._climb_jump = None
        self._vertical_direction = None
        self.reason = None
        self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
        self._reset_alignment()
        self._reset_jump()
        self._reset_drop()
        self._reset_ramp()

    def _reset_ramp(self) -> None:
        self._ramp_started = False
        self._ramp_destination_seen = False
        self._ramp_landing_frames = 0
        self._ramp_source_y = self._ramp_landing_pos = self._ramp_landing_since = None
        self._ramp_retry_count = 0
        self._ramp_retry_since = None
        self._reset_ramp_retry_stability()

    def _reset_ramp_retry_stability(self) -> None:
        self._ramp_retry_pos = self._ramp_retry_stable_since = self._ramp_retry_last = None
        self._ramp_retry_frames = 0

    def _wait_ramp_knockback(self, pos, now: float, source: Platform) -> Command:
        """Release during one small backward knockback; only rearm on this stair."""
        point = self.waypoints[self.idx]
        sign = 1 if point.x > point.from_x else -1
        backward = (point.from_x - pos[0]) * sign
        landing_tolerance = min(1.0, self.tolerance_y)
        support_error = source.support_error(pos)
        # An upward hit may briefly lift the character off the stair. A position
        # below its support range is a fall, never permission to walk or retry.
        if (not source.left <= pos[0] <= source.right or backward > 3.0
                or support_error > self.ramp_corridor_y
                or pos[1] > source.support_range(pos[0])[1] + landing_tolerance):
            self.reason, self.state = "left ramp corridor", "blocked"
            return Command.NONE
        if now - self._ramp_retry_since >= 1.2:
            self.reason, self.state = "ramp knockback did not settle", "blocked"
            return Command.NONE
        self.state = "ramp_waiting_knockback"
        if support_error > landing_tolerance:
            self._reset_ramp_retry_stability()
            return Command.NONE
        if (self._ramp_retry_pos is None or abs(pos[0] - self._ramp_retry_pos[0]) > 0.5
                or abs(pos[1] - self._ramp_retry_pos[1]) > 0.5):
            self._reset_ramp_retry_stability()
            self._ramp_retry_pos, self._ramp_retry_stable_since = pos, now
        if self._ramp_retry_last != now:
            self._ramp_retry_frames += 1
            self._ramp_retry_last = now
        if self._ramp_retry_frames >= 3 and now - self._ramp_retry_stable_since >= 0.25:
            self._ramp_started = False
            self._ramp_retry_since = None
            self._ramp_source_y = None
            self._reset_alignment()
            self._reset_ramp_retry_stability()
            self.state = "ramp_realigning"
        # The next fresh observation uses the ordinary precision approach. The
        # original ramp timeout and the one-retry budget are never replenished.
        return Command.NONE

    def _step_ramp(self, pos, now: float) -> Command:
        """Walk a declared slope; never improvise a jump or reverse after a fall."""
        point = self.waypoints[self.idx]
        if self.reason is not None:
            self.state = "blocked"
            return Command.NONE
        if now - self._started >= self.timeout_s:
            self.reason, self.state = "ramp timeout", "blocked"
            return Command.NONE
        landing_tolerance = min(1.0, self.tolerance_y)
        direction = "right" if point.x > point.from_x else "left"
        source = self._containing_platform((point.from_x, point.from_y))
        if self._ramp_retry_since is not None:
            return self._wait_ramp_knockback(pos, now, source)
        if not self._ramp_started:
            if (self._support_error(point.from_x, point.from_y, pos) > landing_tolerance
                    or (source is not None and not source.left <= pos[0] <= source.right)):
                self._reset_alignment()
                self._ramp_source_y = None
                self.state = "ramp_waiting_floor"
                return Command.NONE
            dx = point.from_x - pos[0]
            if abs(dx) > max(self.align_tolerance_x, self.tolerance_x):
                self.state = "ramp_aligning"
                return self._precision_approach(dx, now)
            self._pulse_until = self._pulse_direction = None
            if self._ramp_source_y is None or abs(pos[1] - self._ramp_source_y) > 0.5:
                self._aligned_since = None
                self._ramp_source_y = pos[1]
            if self._aligned_since is None:
                self._aligned_since = now
            if now - self._aligned_since < self.align_settle_s:
                self.state = "ramp_settling"
                return Command.NONE
            self._ramp_started = True

        low_x, high_x = sorted((point.from_x, point.x))
        fraction = (pos[0] - point.from_x) / (point.x - point.from_x)
        expected_y = point.from_y + min(1.0, max(0.0, fraction)) * (point.y - point.from_y)
        destination = self._containing_platform((point.x, point.y))
        corridor_error = abs(pos[1] - expected_y)
        if source is not None and source.sloped:
            # The endpoints' chord can cut across a drawn shelf or a steep final
            # stair. Keep the existing corridor width around the actual surface.
            if source.left <= pos[0] <= source.right:
                corridor_error = source.support_error(pos)
            elif (destination is not None and destination.left <= pos[0] <= destination.right
                  and (pos[0] > source.right if direction == "right" else pos[0] < source.left)):
                corridor_error = destination.support_error(pos)
        supported = (self._support_error(point.x, point.y, pos) <= landing_tolerance
                     and (destination is None or destination.left <= pos[0] <= destination.right))
        forward_overshoot = (pos[0] - point.x) * (1 if direction == "right" else -1)
        landed_beyond = (destination is not None and supported and 0 < forward_overshoot <= 3.0)
        on_reached_floor = self._ramp_destination_seen and destination is not None and supported
        if (pos[0] < low_x - self.tolerance_x or pos[0] > high_x + self.tolerance_x
                or corridor_error > self.ramp_corridor_y) and not (landed_beyond or on_reached_floor):
            backward = (point.from_x - pos[0]) * (1 if direction == "right" else -1)
            if (not self._ramp_destination_seen and self._ramp_retry_count == 0
                    and source is not None and source.sloped
                    and self.tolerance_x < backward <= 3.0
                    and source.left <= pos[0] <= source.right
                    and source.support_error(pos) <= self.ramp_corridor_y
                    and pos[1] <= source.support_range(pos[0])[1] + landing_tolerance):
                self._ramp_retry_count = 1
                self._ramp_retry_since = now
                self._reset_alignment()
                self._reset_ramp_retry_stability()
                return self._wait_ramp_knockback(pos, now, source)
            self.reason, self.state = "left ramp corridor", "blocked"
            return Command.NONE
        # Once arrival was observed, horizontal knockback within the destination
        # floor need not return toward the ramp edge. Fresh floor support and
        # stable height must still persist throughout the landing confirmation.
        arrived = supported and (abs(pos[0] - point.x) <= self.tolerance_x
                                 or landed_beyond or on_reached_floor)
        if arrived:
            self._ramp_destination_seen = True
            if (self._ramp_landing_pos is None
                    or abs(pos[1] - self._ramp_landing_pos[1]) > 0.5):
                self._ramp_landing_pos, self._ramp_landing_since = pos, now
                self._ramp_landing_frames = 1
            else:
                self._ramp_landing_frames += 1
            self.state = "ramp_landing"
            if (self._ramp_landing_frames >= 3
                    and now - self._ramp_landing_since >= max(self.ramp_landing_s, point.dwell_s)):
                self._advance(now)
            return Command.NONE
        self._ramp_destination_seen = False
        self._ramp_landing_frames = 0
        self._ramp_landing_pos = self._ramp_landing_since = None
        self.state = "ramping"
        # Reaching the destination X without its floor is not successful ascent.
        # Wait boundedly rather than walking past the declared endpoint or jumping.
        return Command(direction) if fraction < 1.0 else Command.NONE

    def _reset_drop(self) -> None:
        self._drop_source_y = self._drop_launched_at = None
        self._drop_stable_y = self._drop_stable_since = None
        self._drop_departed = self._drop_started = self._drop_launch_ended = False

    def _block_drop(self, reason: str) -> Command:
        self.reason, self.state = reason, "blocked"
        return Command.NONE

    def _step_drop(self, pos, now: float) -> Command:
        """Drop once per stable ledge, releasing Down throughout every observed fall."""
        point = self.waypoints[self.idx]
        if self.reason is not None:
            self.state = "blocked"
            return Command.NONE
        if now - self._started >= self.timeout_s:
            return self._block_drop("drop timeout")
        landing_tolerance = min(1.0, self.tolerance_y)
        destination_y = self._height_at(point.x, point.y, pos[0])
        destination_low, destination_high = self._support_range(point.x, point.y, pos[0])
        if pos[1] > destination_high + landing_tolerance:
            return self._block_drop("passed destination floor")
        if self._drop_source_y is None:
            previous = self.waypoints[(self.idx - 1) % len(self.waypoints)]
            previous_y = self._height_at(previous.x, previous.y, pos[0])
            self._drop_source_y = (previous_y if previous.action == "walk" and previous_y < destination_y
                                   else pos[1])
        if not self._drop_started and self._drop_launched_at is None:
            previous = self.waypoints[(self.idx - 1) % len(self.waypoints)]
            source = self._containing_platform((previous.x, previous.y))
            if (previous.action == "walk" and source is not None and source.sloped
                    and source.left <= pos[0] <= source.right
                    and source.support_error(pos) <= landing_tolerance):
                self._drop_source_y = pos[1]
        if self._drop_stable_y is None or abs(pos[1] - self._drop_stable_y) > 0.5:
            self._drop_stable_y, self._drop_stable_since = pos[1], now
        stable_for = now - self._drop_stable_since
        dx = point.x - pos[0]
        destination = self._containing_platform((point.x, point.y))
        at_destination = (self._support_error(point.x, point.y, pos) <= landing_tolerance
                          and (destination is None or destination.left <= pos[0] <= destination.right))
        if at_destination:
            self._drop_started = self._drop_launch_ended = True
            self.state = "dwelling"
            if stable_for < max(0.25, point.dwell_s):
                return Command.NONE
            if abs(dx) <= self.tolerance_x:
                self._advance(now)
                return Command.NONE
            # Horizontal correction is allowed only after a stable destination landing.
            return self._precision_approach(dx, now)
        if pos[1] > self._drop_source_y + 1:
            self._drop_departed = self._drop_started = self._drop_launch_ended = True
        elif pos[1] < self._drop_source_y - 1:
            self._drop_launch_ended = True  # Down+Jump may jump upward on a solid ledge
        if self._drop_departed:
            self.state = "dropping"
            if (stable_for >= max(0.4, self.align_settle_s)
                    and self._drop_source_y + 1 < pos[1] < destination_low - landing_tolerance):
                # A stepped platform can catch a fall (e.g. y27 above the y31 floor).
                # A new drop is permitted only after that intermediate ledge is observed stable.
                self._drop_source_y = pos[1]
                self._drop_launched_at = None
                self._drop_departed = False
                self.retries = 0
                self._reset_alignment()
            elif stable_for >= max(0.4, self.align_settle_s) and abs(pos[1] - self._drop_source_y) <= 1:
                self._drop_departed = False  # returning to the same floor does not reset its retry budget
            return Command.NONE
        if self._drop_launched_at is not None:
            self.state = "dropping"
            elapsed = now - self._drop_launched_at
            if not self._drop_launch_ended and elapsed < min(0.15, self.jump_launch_s):
                return Command("none", "down", "jump")
            self._drop_launch_ended = True
            if (abs(pos[1] - self._drop_source_y) <= 1 and stable_for >= max(0.4, self.align_settle_s)
                    and elapsed >= max(0.8, self.drop_interval_s)):
                if self.retries >= self.max_retries:
                    return self._block_drop("drop did not leave source floor")
                self.retries += 1
                self._drop_launched_at = None
                self._reset_alignment()
            return Command.NONE
        if abs(pos[1] - self._drop_source_y) > 1:
            self.state = "dropping"
            return Command.NONE
        if abs(dx) > self.drop_align_tolerance_x:
            self.state = "aligning"
            return self._precision_approach(dx, now)
        self._pulse_until = self._pulse_direction = None
        if self._aligned_since is None:
            self._aligned_since = now
        if now - self._aligned_since < self.align_settle_s or stable_for < self.align_settle_s:
            self.state = "settling"
            return Command.NONE
        self._drop_started = True
        self._drop_launched_at = now
        self._drop_launch_ended = False
        self.state = "dropping"
        return Command("none", "down", "jump")

    def _reset_jump(self) -> None:
        self._jump_phase = "align"
        self._jump_started = self._jump_retry_until = None
        self._jump_takeoff_y = None
        self._jump_departed = self._jump_descending = False
        self._jump_destination_seen = False
        self._jump_launch_ended = False
        self._jump_min_y = self._jump_landing_y = self._jump_landing_since = None
        self._jump_landing_frames = 0

    def _jump_failed(self, now: float) -> Command:
        if self.retries >= self.max_retries:
            self.reason = "jump timeout"
            self.state = "blocked"
        else:
            self.retries += 1
            self._reset_jump()
            self._reset_alignment()
            self._jump_phase = "retry"
            self._jump_retry_until = now + self.recovery_s
            self.state = "jump_retry"
        return Command.NONE

    def _step_jump(self, pos, now: float) -> Command:
        point = self.waypoints[self.idx]
        if self.reason is not None:
            self.state = "blocked"
            return Command.NONE
        if self._jump_phase == "retry":
            if now < self._jump_retry_until:
                self.state = "jump_retry"
                return Command.NONE
            self._jump_phase = "align"
            self._started = now
        if now - self._started >= self.timeout_s:
            return self._jump_failed(now)
        direction = "left" if point.x < point.from_x else "right" if point.x > point.from_x else "none"
        if self._jump_phase == "align":
            source = self._containing_platform((point.from_x, point.from_y))
            if (self._support_error(point.from_x, point.from_y, pos) > min(1.0, self.tolerance_y)
                    or (source is not None and not source.left <= pos[0] <= source.right)):
                self._reset_alignment()
                self.state = "jump_waiting_floor"
                return Command.NONE
            dx = point.from_x - pos[0]
            if abs(dx) > self.jump_align_tolerance_x:
                self.state = "jump_aligning"
                return self._precision_approach(dx, now)
            self._pulse_until = self._pulse_direction = None
            if self._jump_takeoff_y is None or abs(pos[1] - self._jump_takeoff_y) > 0.5:
                self._aligned_since = None
                self._jump_takeoff_y = pos[1]
            if self._aligned_since is None:
                self._aligned_since = now
            if now - self._aligned_since < self.align_settle_s:
                self.state = "jump_settling"
                return Command.NONE
            self._jump_phase = "flight"
            self._jump_started = now
            self._jump_min_y = pos[1]
            self.state = "jumping"
            return Command(direction, "none", "jump")
        flight_s = now - self._jump_started
        self._jump_min_y = min(self._jump_min_y, pos[1])
        self._jump_departed |= abs(pos[1] - self._jump_takeoff_y) >= 2
        self._jump_descending |= self._jump_departed and pos[1] > self._jump_min_y + 0.5
        destination = self._containing_platform((point.x, point.y))
        forward_overshoot = (pos[0] - point.x) * (1 if direction == "right" else -1)
        landed_beyond = (destination is not None and direction != "none"
                         and 0 < forward_overshoot <= self.jump_landing_overshoot_x)
        # A short upward hop can reach a ledge at the minimap's rounded apex,
        # hiding descent entirely. Require longer support confirmation there.
        higher_support = (destination is not None and pos[1] < self._jump_takeoff_y - 1)
        on_reached_floor = self._jump_destination_seen and destination is not None
        at_landing = (self._jump_departed and (self._jump_descending or higher_support)
                      and (abs(pos[0] - point.x) <= self.tolerance_x or landed_beyond or on_reached_floor)
                      and self._support_error(point.x, point.y, pos) <= min(1.0, self.tolerance_y)
                      and (destination is None or destination.left <= pos[0] <= destination.right))
        if at_landing:
            if not self._jump_descending and flight_s >= self.jump_flight_timeout_s:
                return self._jump_failed(now)
            # After observed arrival, inertia or knockback may move horizontally
            # along this floor. Confirm fresh stable support without steering
            # back toward the gap; any lost support invalidates this allowance.
            self._jump_destination_seen = destination is not None
            if self._jump_landing_since is None or abs(pos[1] - self._jump_landing_y) > 0.5:
                self._jump_landing_since, self._jump_landing_y = now, pos[1]
                self._jump_landing_frames = 1
            else:
                self._jump_landing_frames += 1
            self.state = "jump_landing" if self._jump_descending else "airborne"
            required_dwell = max(self.jump_landing_s, point.dwell_s,
                                 0 if self._jump_descending else 0.65)
            if (self._jump_landing_frames >= 3
                    and now - self._jump_landing_since >= required_dwell
                    and (self._jump_descending or flight_s >= 0.8)):
                self._advance(now)
            return Command.NONE
        self._jump_landing_since = self._jump_landing_y = None
        self._jump_landing_frames = 0
        self._jump_destination_seen = False
        if flight_s >= self.jump_flight_timeout_s:
            return self._jump_failed(now)
        self.state = "airborne" if self._jump_departed else "jumping"
        if not self._jump_launch_ended and flight_s < self.jump_launch_s:
            return Command(direction, "none", "jump")
        self._jump_launch_ended = True
        dx = point.x - pos[0]
        direction = "left" if dx < -self.tolerance_x else "right" if dx > self.tolerance_x else "none"
        if direction == "none" and destination is not None:
            # A short hop can be within the target's deadzone while still over
            # the gap. Keep correcting until the destination actually contains X.
            if pos[0] < destination.left:
                direction = "right"
            elif pos[0] > destination.right:
                direction = "left"
        return Command(direction)

    def _reset_return_stability(self) -> None:
        self._return_stable_since = self._return_stable_pos = self._return_platform_id = None
        self._return_stable_frames = 0

    def request_return_to_start(self) -> bool:
        """Queue a guarded return; no original index changes and no immediate input."""
        if self.returning_to_start or self.recovery_terminal or self.movement_test:
            return False
        self.recovery_from_index = self.idx
        self.blocked_origin_reason = self.reason if self.state == "blocked" else None
        if self.return_count >= self.max_returns_per_lap:
            reason = "Return budget exhausted for this lap"
            if self.blocked_origin_reason:
                reason += f"; point {self.idx + 1} blocked: {self.blocked_origin_reason}"
            self.abort_return_to_start(reason)
            return False
        self.return_count += 1
        self.returning_to_start = True
        self.recovery_state = "waiting_source"
        self.recovery_reason = None
        self.recovery = None
        self._return_started = None
        self._reset_return_stability()
        self.state = "return_waiting_source"
        return True

    def abort_return_to_start(self, reason: str) -> None:
        """Make a return failure terminal until an explicit configuration reload."""
        self.returning_to_start = False
        self.recovery_terminal = True
        self.recovery_state = "failed"
        self.recovery_reason = str(reason)
        self.reason = "Return failed: " + str(reason)
        self.state = "blocked"
        self._reset_return_stability()
        if self.recovery is not None:
            self.recovery.suspend(self._clock.last if self._clock.last is not None else 0)

    def _return_position_stable(self, pos, platform_id, now) -> bool:
        if (self._return_stable_pos is None or self._return_platform_id != platform_id
                or max(abs(pos[i] - self._return_stable_pos[i]) for i in (0, 1)) > .5):
            self._return_stable_pos = pos
            self._return_platform_id = platform_id
            self._return_stable_since = now
            self._return_stable_frames = 1
        else:
            self._return_stable_frames += 1
        return self._return_stable_frames >= 3 and now - self._return_stable_since >= .5

    def _step_return(self, pos, now: float, on_ladder: bool) -> Command:
        from maplebot.nav.recovery import plan_return_to_start, supported_platform

        t = self._clock.update(now, pos is not None)
        if self._return_started is None:
            self._return_started = t
        if t - self._return_started >= self.return_timeout_s:
            self.abort_return_to_start("Return route timed out")
            return Command.NONE
        if pos is None:
            self._reset_return_stability()
            if self.recovery is not None:
                self.recovery.suspend(now)
            return Command.NONE
        if self.recovery is None:
            if t - self._return_started >= 3:
                self.abort_return_to_start("Stable source unavailable: " + (self.recovery_reason or "position kept moving"))
                return Command.NONE
            try:
                source = supported_platform(self._cfg, pos)
            except ValueError as exc:
                self.recovery_reason = str(exc)
                self._reset_return_stability()
                return Command.NONE
            self.recovery_reason = None
            if not self._return_position_stable(pos, source["id"], t):
                return Command.NONE
            try:
                self.recovery = MinimapCircuit(plan_return_to_start(self._cfg, pos))
            except ValueError as exc:
                self.abort_return_to_start(str(exc))
                return Command.NONE
            self.recovery_state = "running"
            self._reset_return_stability()
        command = self.recovery.step(pos, now, on_ladder)
        self.state = "return_" + self.recovery.state
        if self.recovery.state == "blocked":
            self.abort_return_to_start(self.recovery.reason or "Return route blocked")
            return Command.NONE
        if not self.recovery.completed:
            return command
        self.recovery_state = "confirming_start"
        self.state = "return_confirming_start"
        try:
            platform = supported_platform(self._cfg, pos)
            final = self.recovery._cfg["waypoints"][-1]
            supported = (platform["id"] == final.get("platform_id")
                         and abs(pos[0] - final["x"]) <= self.recovery.tolerance_x)
        except ValueError:
            supported = False
        if not supported:
            self._reset_return_stability()
            return Command.NONE
        if self._return_position_stable(pos, platform["id"], t):
            self.returning_to_start = False
            self.recovery_state = "completed"
            self.recovery_reason = None
            self.completed = False
            self._selected = True
            self._set_waypoint(0, t)
            self.state = "returned_to_start"
        return Command.NONE

    def step(self, pos: tuple[int, int] | None, now: float, on_ladder: bool = False) -> Command:
        if self.recovery_terminal:
            self.state = "blocked"
            return Command.NONE
        if self.returning_to_start:
            return self._step_return(pos, now, on_ladder)
        command = self._step_route(pos, now, on_ladder)
        if self.state == "blocked" and self.return_on_block and not self.movement_test:
            if self.request_return_to_start():
                return self._step_return(pos, now, on_ladder)
        return command

    def _step_route(self, pos: tuple[int, int] | None, now: float, on_ladder: bool = False) -> Command:
        if self.completed:
            self.state = "completed"
            return Command.NONE
        t = self._clock.update(now, pos is not None)
        if pos is None:
            self._reset_alignment()
            self._rejoin_floor = self._rejoin_y = self._rejoin_since = None
            self._jump_landing_since = self._jump_landing_y = None
            self._jump_landing_frames = 0
            self._jump_destination_seen = False
            if self._jump_phase == "flight":
                self._jump_launch_ended = True
            self._drop_stable_y = self._drop_stable_since = None
            self._drop_launch_ended = True
            self._ramp_landing_pos = self._ramp_landing_since = self._ramp_source_y = None
            self._ramp_landing_frames = 0
            self._reset_ramp_retry_stability()
            self.state = "waiting_position"
            return Command.NONE
        if not self._selected:
            platform = self.platform_at(pos)
            compatible = [(abs(point.x - pos[0]), i) for i, point in enumerate(self.waypoints)
                          if point.action == "walk"
                          and (abs(point.y - pos[1]) <= self.tolerance_y if platform is None else
                               (platform.support_error(pos) <= self.tolerance_y
                                and self._containing_platform((point.x, point.y)) == platform
                                and platform.left <= point.x <= platform.right))]
            if not compatible:
                self.state = "waiting_floor"
                return Command.NONE
            self.idx = min(compatible)[1]
            self._selected = True
        if self._started is None:
            self._started = t
        if self.waypoints[self.idx].action == "jump":
            return self._step_jump(pos, t)
        if self.waypoints[self.idx].action == "drop":
            return self._step_drop(pos, t)
        if self.waypoints[self.idx].action == "ramp":
            return self._step_ramp(pos, t)
        floor_ready = self._walk_floor_ready(pos, t)
        point = self.waypoints[self.idx]
        dx = point.x - pos[0]
        dy = self._height_at(point.x, point.y, pos[0]) - pos[1]
        progressing = False
        if point.action == "climb":
            if self._climb_last_y is not None and pos[1] < self._climb_last_y:
                self._climb_progress = t
            self._climb_last_y = pos[1]
            progressing = self._climb_progress is not None and t - self._climb_progress < 0.6
        if point.action == "climb" and self._vertical_direction is None and abs(dy) > 0.5:
            self._vertical_direction = "up" if dy < 0 else "down"
        reached_y = self._support_error(point.x, point.y, pos) <= self.tolerance_y
        if point.action == "climb":
            reached_y = reached_y and (dy >= -0.5 if self._vertical_direction == "up"
                                      else dy <= 0.5 if self._vertical_direction == "down" else abs(dy) <= 0.5)
        if floor_ready and abs(dx) <= self.tolerance_x and reached_y:
            self.state = "dwelling"
            if self._arrived is None:
                self._arrived = t
            if t - self._arrived >= point.dwell_s:
                self._advance(t)
                return Command.NONE
            return Command("none", self._vertical_direction) if point.action == "climb" and self._vertical_direction else Command.NONE
        self._arrived = None
        if self.reason == "waypoint timeout":
            self.state = "blocked"
            return Command.NONE
        if self._recovery_until is not None:
            if t < self._recovery_until:
                self.state = "recovering"
                if point.action in ("climb", "drop") or not floor_ready:
                    return Command.NONE  # rest, then retry through the same precision approach
                jump = "none" if self._recovery_jumped else "jump"
                self._recovery_jumped = True
                return Command("left" if dx < -self.tolerance_x else
                               "right" if dx > self.tolerance_x else "none", "none", jump)
            self._recovery_until = None
            self._started = t
        if t - self._started >= self.timeout_s:
            if self.retries >= self.max_retries:
                self.reason = "waypoint timeout"
                self.state = "blocked"
            else:
                self.retries += 1
                self._recovery_until = t + self.recovery_s
                self._recovery_jumped = False
                self.state = "recovering"
                self._reset_alignment()
            return Command.NONE  # release old keys before a recovery jump
        if not floor_ready:
            self.state = "waiting_floor"
            return Command.NONE
        entering = self._climb_jump is not None and t - self._climb_jump < 0.6
        if point.action == "climb" and dy < 0 and abs(dx) < self.climb_jitter_x and (progressing or entering):
            self.state = "climbing"
            return Command("none", "up")
        alignment = self.tolerance_x if point.action == "walk" else self.align_tolerance_x
        if abs(dx) > alignment:
            self.state = "walking" if point.action == "walk" else "aligning"
            return (Command("left" if dx < 0 else "right") if point.action == "walk" and not self.precision_walk
                    else self._precision_approach(dx, t))
        if point.action in ("climb", "drop"):
            self._pulse_until = self._pulse_direction = None
            if self._aligned_since is None:
                self._aligned_since = t
            if t - self._aligned_since < self.align_settle_s:
                self.state = "settling"
                return Command("none", "up") if point.action == "climb" and dy < 0 else Command.NONE
        if point.action == "climb":
            self.state = "climbing"
            previous = self.waypoints[(self.idx - 1) % len(self.waypoints)]
            previous_y = self._height_at(previous.x, previous.y, pos[0])
            source_floor = previous_y if previous.action == "walk" and previous_y > point.y else None
            needs_entry = source_floor is None or pos[1] >= source_floor - self.tolerance_y
            # Screen-space ladder guesses can remain latched after landing.
            # Only observed minimap ascent suppresses another rope-entry jump.
            jump = (dy < 0 and needs_entry and not progressing
                    and (self._climb_jump is None or t - self._climb_jump >= self.drop_interval_s))
            if jump:
                self._climb_jump = t
            return Command("none", "up" if dy < 0 else "down", "jump" if jump else "none")
        self.state = "waiting_floor"
        return Command.NONE
