"""A bounded, configurable lap over points in the currently visible minimap."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from copy import deepcopy
import math

from maplebot.types import Command

# A level gap jump still lining up: flight direction, the source's end on the gap side, source.
GapTakeoff = namedtuple("GapTakeoff", ("direction", "edge_x", "source"))


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
    # After a credited resume, the shortest release still served before a pulse.
    RESUME_RELEASE_FLOOR_S = 0.15
    # Key-up slide plus whole-pixel rounding the edge margin must cover beyond the stop margin.
    EDGE_OVER_STOP_MARGIN_X = 0.75
    # Sideways drift of a fall: a drop launched from its zone may land this far beside the zone.
    ZONE_LANDING_DRIFT_X = 2.0

    def __init__(self, cfg: dict, *, far_approach: bool = True, resume_credit: bool | None = None,
                 zone_kinds: tuple = ()):
        """``far_approach=False`` ignores the opt-in far-pulse and resume-credit keys.

        ``resume_credit=False`` ignores ``approach_resume_credit`` alone (a
        return/rejoin child that may use far pulses is never interrupted by
        combat or loot, so it has nothing to credit).

        ``zone_kinds`` are the transition kinds that may use ``takeoff_zones``
        in a movement test. Only a parent whose own zones are in effect passes
        them, to its return/rejoin child; an editor connection test never does.
        """
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
        # Bounded walk-off / rope-descent stage before a return is planned from an
        # unsupported position (maplebot/nav/regain.py). It never proves support itself.
        self.return_regain_support = cfg.get("return_regain_support", True)
        self.regain_timeout_s = cfg.get("regain_timeout_s", 20)
        self.regain_max_walk_x = cfg.get("regain_max_walk_x", 16)
        if type(self.return_regain_support) is not bool:
            raise ValueError("return_regain_support must be true or false")
        for name, low, high in (("regain_timeout_s", 5, 30), ("regain_max_walk_x", 4, 40)):
            value = getattr(self, name)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be finite and between {low} and {high}")
            setattr(self, name, float(value))
        self._regain = None
        self._source_window_started = None
        # Opt-in handling of "stably supported on a drawn platform the waypoint
        # does not expect". Both need the saved editor plan and stay inert in
        # movement tests, so editor connection tests and return/rejoin children
        # never use them.
        self.arrive_on_destination = cfg.get("arrive_on_destination", False)
        self.platform_rejoin_enabled = cfg.get("platform_rejoin", False)
        for name, value in (("arrive_on_destination", self.arrive_on_destination),
                            ("platform_rejoin", self.platform_rejoin_enabled)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false")
        self.max_platform_rejoins_per_lap = cfg.get("max_platform_rejoins_per_lap", 3)
        self.platform_rejoin_max_transitions = cfg.get("platform_rejoin_max_transitions", 1)
        for name, low, high in (("max_platform_rejoins_per_lap", 1, 5),
                                ("platform_rejoin_max_transitions", 1, 4)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        self.platform_rejoin_timeout_s = cfg.get("platform_rejoin_timeout_s", 20)
        if (not isinstance(self.platform_rejoin_timeout_s, (int, float))
                or isinstance(self.platform_rejoin_timeout_s, bool)
                or not math.isfinite(self.platform_rejoin_timeout_s)
                or not 5 <= self.platform_rejoin_timeout_s <= 60):
            raise ValueError("platform_rejoin_timeout_s must be finite and between 5 and 60")
        self.platform_rejoin_timeout_s = float(self.platform_rejoin_timeout_s)
        has_plan = isinstance(cfg.get("editor_plan"), dict)
        if ((self.arrive_on_destination or self.platform_rejoin_enabled)
                and not self.movement_test and not has_plan):
            raise ValueError("arrive_on_destination and platform_rejoin need a saved editor plan")
        self._arrive_enabled = self.arrive_on_destination and not self.movement_test and has_plan
        self._rejoin_enabled = self.platform_rejoin_enabled and not self.movement_test and has_plan
        # Opt-in gap re-entry (OFF when absent): a landing walk knocked back across the
        # level gap it was reached by is re-selected at the walk before that gap jump,
        # after the same 3-frame proof, instead of waiting out its clock or spending a
        # rejoin. Nothing else of it is read while ``_reentry_enabled`` is False.
        gap_reentry = cfg.get("gap_reentry", False)
        if type(gap_reentry) is not bool:
            raise ValueError("gap_reentry must be true or false")
        if gap_reentry and not self.movement_test and not has_plan:
            raise ValueError("gap_reentry needs a saved editor plan")
        self.reentry_count = 0      # re-entries this lap; only a completed lap resets it
        self.reentry_last = None    # {from_index, to_index, platform} of the last one, for status only
        self.gap_reentry_priority_s = None
        self.max_gap_reentries_per_lap = None
        self.rejoining = False
        self.platform_rejoin = None  # child circuit while a rejoin runs
        self.rejoin_count = 0
        self.rejoin_state = "idle"  # idle / running / confirming / completed / failed
        self.rejoin_reason = None
        self.arrival_count = 0  # jump waypoints passed by destination proof, without a takeoff
        self._rejoin_started = self._rejoin_expected = self._rejoin_failed_idx = None
        self._reset_offroute()
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
        self._reentry_enabled = gap_reentry and not self.movement_test and has_plan and bool(self.platforms)
        if self._reentry_enabled:
            budget = cfg.get("max_gap_reentries_per_lap", 2)
            if type(budget) is not int or not 1 <= budget <= 5:
                raise ValueError("max_gap_reentries_per_lap must be an integer from 1 to 5")
            priority = cfg.get("gap_reentry_priority_s", 1.5)
            if (not isinstance(priority, (int, float)) or isinstance(priority, bool)
                    or not math.isfinite(priority) or not 0.5 <= priority <= 3.0):
                raise ValueError("gap_reentry_priority_s must be finite and between 0.5 and 3.0")
            self.max_gap_reentries_per_lap = budget
            self.gap_reentry_priority_s = float(priority)
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
        # Optional far/fine split for walk waypoints. Absent keys keep every
        # correction inside approach_radius_x on the vision-gated fine taps.
        fine = cfg.get("approach_fine_radius_x") if far_approach else None
        self.approach_fine_radius_x = None if fine is None else float(fine)
        self.approach_walk_speed_x = float(cfg.get("approach_walk_speed_x", 0))
        self.approach_stop_margin_x = float(cfg.get("approach_stop_margin_x", 1))
        self.approach_edge_margin_x = float(cfg.get("approach_edge_margin_x", 1))
        self.approach_far_pulse_max_s = float(cfg.get("approach_far_pulse_max_s", 0.4))
        self.approach_far_release_s = float(cfg.get("approach_far_release_s", self.approach_release_s))
        self.approach_resume_credit = (far_approach and resume_credit is not False
                                       and bool(cfg.get("approach_resume_credit", False)))
        # Opt-in transition timing (all OFF when absent; see config/default.yaml).
        #   commit_budget: a transition that is physically under way is bounded by
        #     its own budget (a fall or a rope also by a no-progress rule) instead
        #     of the waypoint clock.
        #   takeoff_far_approach: jump/ramp/drop/climb takeoff alignment may use the
        #     far pulses, bounded by the SOURCE platform's end.
        #   recovery_far_approach: return/rejoin children may use far pulses.
        self.commit_budget = cfg.get("commit_budget", False)
        takeoff_far = cfg.get("takeoff_far_approach", False)
        self.recovery_far_approach = cfg.get("recovery_far_approach", False)
        for name, value in (("commit_budget", self.commit_budget), ("takeoff_far_approach", takeoff_far),
                            ("recovery_far_approach", self.recovery_far_approach)):
            if type(value) is not bool:
                raise ValueError(f"{name} must be true or false")
            if value and name != "commit_budget" and cfg.get("approach_fine_radius_x") is None:
                raise ValueError(f"{name} needs approach_fine_radius_x and the far approach keys")
        # A child built without far pulses keeps fine taps for its takeoffs too.
        self.takeoff_far_approach = takeoff_far and far_approach
        # Opt-in take-off zones (OFF when absent; maplebot/nav/zones.py). The FIRST
        # attempt of a jump, ramp or drop may take off from a stretch of its source
        # platform where the saved geometry gives the same outcome, instead of from
        # one saved pixel. Landing proof, support tests and tolerances are untouched.
        # ``_zone_kinds`` is empty unless zones are in effect, and nothing else of
        # this feature is read while it is empty.
        self.takeoff_zones = cfg.get("takeoff_zones", False)
        if type(self.takeoff_zones) is not bool:
            raise ValueError("takeoff_zones must be true or false")
        self._zone_kinds = ()
        if self.takeoff_zones:
            for name, default, low, high in (("takeoff_zone_max_x", 8, 1, 20), ("takeoff_zone_margin_x", 1.5, 1, 4)):
                value = cfg.get(name, default)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or not low <= value <= high):
                    raise ValueError(f"{name} must be finite and between {low} and {high}")
                setattr(self, name, float(value))
            kinds = cfg.get("takeoff_zone_kinds", ["drop", "jump", "ramp"])
            if (not isinstance(kinds, (list, tuple)) or not kinds or len(set(map(repr, kinds))) != len(kinds)
                    or not all(isinstance(kind, str) and kind in ("drop", "jump", "ramp") for kind in kinds)):
                raise ValueError("takeoff_zone_kinds must be a non-empty list of drop, jump and ramp without repeats")
            if not self.movement_test and not has_plan:
                raise ValueError("takeoff_zones needs a saved editor plan")
            if has_plan:
                # A movement test takes only what its parent hands it, and never a jump.
                self._zone_kinds = (tuple(k for k in ("drop", "ramp") if k in zone_kinds and k in kinds)
                                    if self.movement_test else tuple(k for k in ("jump", "ramp", "drop") if k in kinds))
        self.zone_takeoffs = 0  # take-offs made outside the legacy window, children included
        self._zone_cache = {}
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
        if self.commit_budget:
            # A ramp or rope that starts at once keeps the time the waypoint clock gave
            # it: both defaults follow timeout_s. They are clamped into their own range,
            # so a timeout_s that is valid without the key stays valid with it.
            for name, default, low, high in (("ramp_travel_timeout_s", min(30.0, max(2.0, self.timeout_s)), 2, 30),
                                             ("drop_fall_timeout_s", 6.0, 2, 30),
                                             ("climb_travel_timeout_s", min(60.0, max(2.0, self.timeout_s)), 2, 60),
                                             ("landing_confirm_timeout_s", 3.0, 1, 10),
                                             ("transition_stall_s", 2.0, 1, 5)):
                value = cfg.get(name, default)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or not low <= value <= high):
                    raise ValueError(f"{name} must be finite and between {low} and {high}")
                setattr(self, name, float(value))
        numbers = (self.tolerance_x, self.tolerance_y, self.align_tolerance_x, self.jump_align_tolerance_x,
                   self.drop_align_tolerance_x,
                   self.timeout_s, self.recovery_s, self.drop_interval_s,
                   self.approach_radius_x, self.approach_pulse_s, self.approach_release_s,
                   self.align_settle_s, self.climb_jitter_x, self.platform_tolerance_y, self.floor_rejoin_s,
                   self.jump_flight_timeout_s, self.jump_landing_s, self.jump_launch_s,
                   self.ramp_corridor_y, self.ramp_landing_s)
        if not all(math.isfinite(v) and v > 0 for v in numbers) or not 0 <= self.max_retries <= 5:
            raise ValueError("circuit tolerances/timers must be positive; max_retries must be 0 to 5")
        if self.approach_fine_radius_x is not None:
            far = (self.approach_fine_radius_x, self.approach_walk_speed_x, self.approach_stop_margin_x,
                   self.approach_far_pulse_max_s, self.approach_far_release_s)
            # The shortest far pulse must exceed a fine tap, the longest must stay
            # well inside the 0.75 s command lease, and it always stops short.
            if (not all(math.isfinite(v) and v > 0 for v in far)
                    or not math.isfinite(self.approach_edge_margin_x) or self.approach_edge_margin_x < 0
                    or not self.approach_stop_margin_x < self.approach_fine_radius_x < self.approach_radius_x
                    or not self.approach_pulse_s <= self.approach_far_pulse_max_s <= 0.6
                    or ((self.approach_fine_radius_x - self.approach_stop_margin_x)
                        / self.approach_walk_speed_x < self.approach_pulse_s)):
                raise ValueError("far approach needs stop margin < fine radius < approach radius, a positive "
                                 "walk speed/release, and approach_pulse_s <= far pulse <= 0.6 s")
            # A pulse sized to stop short of the waypoint can still slide and round
            # past it; next to a platform edge the edge bound has to win first.
            if self.approach_edge_margin_x < (self.approach_stop_margin_x
                                              + self.EDGE_OVER_STOP_MARGIN_X - 1e-9):
                raise ValueError("far approach needs approach_edge_margin_x >= approach_stop_margin_x + 0.75")
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
        self._wall = None
        self._suspended = False
        self._approached = False
        self._reset_commit()
        self._reset_zone()
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
        if self.returning_to_start or self.rejoining:
            return True
        if self._selected and not self.completed and self.waypoints[self.idx].action in ("jump", "ramp"):
            return True
        if (self._selected and not self.completed and self.waypoints[self.idx].action == "drop"
                and self._drop_started):
            return True
        return (self._selected and self.waypoints[self.idx].action in ("climb", "drop")
                and self.state in ("climbing", "dropping", "recovering", "dwelling", "settling"))

    @property
    def transition_committed(self) -> bool:
        """True once the current transition can no longer be safely interrupted.

        Committed: a return to start, a blocked/terminal route, every climb state
        that ``transition_active`` covers (settling already holds Up under the
        rope), a jump from its takeoff key until landing proof and during the
        rest after a failed flight, a ramp once it left its takeoff point or
        waits out a knockback, and a drop from its Down+Jump until it is seen
        back on the source floor re-aligning. Only the pre-takeoff phases are
        uncommitted: waiting for the takeoff floor, lining up, settling, and
        re-aligning after a failed attempt. Anything unlisted is committed.
        """
        if self.returning_to_start or self.recovery_terminal or self.rejoining:
            return True
        if not self.transition_active:
            return False
        if self.completed or self.reason is not None:
            return True
        action = self.waypoints[self.idx].action
        if action == "jump":
            return self._jump_phase != "align"
        if action == "ramp":
            return self._ramp_started or self._ramp_retry_since is not None
        if action == "drop":
            # ``state`` is the last stepped frame's: the frame after a failed
            # attempt or a lost position stays committed until it is re-stepped.
            return (self._drop_launched_at is not None or self._drop_departed
                    or self.state not in ("aligning", "settling"))
        return True

    def takeoff_interruptible(self, pos) -> bool:
        """``not transition_committed`` checked against a fresh position.

        The property only knows the last stepped frame. A character knocked off
        a drop ledge since then is already falling, so a drop additionally needs
        the same source-floor height its own takeoff test requires.
        """
        if pos is None or not self.transition_active or self.transition_committed:
            return False
        if self.waypoints[self.idx].action == "drop":
            return self._drop_source_y is not None and abs(pos[1] - self._drop_source_y) <= 1
        return True

    @property
    def gap_reentry_active(self) -> bool:
        """True only with ``gap_reentry`` on a farming circuit with the saved plan and platforms."""
        return self._reentry_enabled

    def reentry_pending(self, pos) -> bool:
        """A gap re-entry would be proven from ``pos`` at the current walk waypoint. Read-only."""
        return self._reentry_enabled and self._reentry_target(pos) is not None

    def level_gap_takeoff(self):
        """``GapTakeoff(direction, edge_x, source)`` while a level gap jump is still lining up.

        Only for a jump waypoint in its uncommitted align phase whose saved take-off
        and landing lie on two different configured platforms that form a level gap
        (``zones.level_gap``). None otherwise, and always None without platforms,
        before selection, once completed, blocked, returning, rejoining or terminal.
        Mutates nothing.
        """
        if (not self.platforms or not self._selected or self.completed or self.returning_to_start
                or self.rejoining or self.recovery_terminal or self.reason is not None):
            return None
        point = self.waypoints[self.idx]
        if point.action != "jump" or self._jump_phase != "align":
            return None
        from maplebot.nav.zones import level_gap

        source = self._containing_platform((point.from_x, point.from_y))
        destination = self._containing_platform((point.x, point.y))
        if source is None or destination is None or source == destination:
            return None
        gap = level_gap(source, destination, point.from_x, point.x)
        return None if gap is None else GapTakeoff(gap[1], gap[0], source)

    def suspend(self, now: float) -> None:
        self._clock.update(now, False)
        self._reset_return_stability()
        if self.recovery is not None and self.returning_to_start:
            self.recovery.suspend(now)
        if self._regain is not None and self.returning_to_start:
            self._regain.interrupt()
        if self.platform_rejoin is not None and self.rejoining:
            self.platform_rejoin.suspend(now)
        self._reset_offroute()
        # Only the first suspend of an uninterrupted run measures the release
        # already served; later ones neither add to it nor restart it.
        credit = self._resume_credit if self._suspended else self._suspend_credit(now)
        self._reset_alignment()
        self._resume_credit, self._suspended = credit, True
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
        self._reset_pulse()
        self._resume_credit = None

    def _reset_pulse(self) -> None:
        self._pulse_until = None
        self._pulse_direction = None
        self._pulse_on = False
        self._pulse_far = False
        self._release_since = None
        self._pulse_dx = None
        self._credit_dx = None

    def _suspend_credit(self, now: float):
        """(wall time, direction, dx, release already served) when keys were last ours."""
        if not self.approach_resume_credit or self._pulse_until is None or self._release_since is None:
            return None
        served = 0.0 if self._pulse_on else max(0.0, self._clock.time - self._release_since)
        return now, self._pulse_direction, self._pulse_dx, served

    def _resume_release(self, release: float, direction: str, dx: float) -> float:
        """Remaining release after an interruption that visibly did not move the player."""
        credit, self._resume_credit = self._resume_credit, None
        if (credit is None or self._wall is None or credit[1] != direction or credit[2] is None
                or abs(dx - credit[2]) > 0.5 or not self._wall >= credit[0]):
            return release
        # One same-pixel observation is not proof of standing still: part of the
        # release is always served again, and _precision_approach restarts it in
        # full if the next observations show the player moving.
        self._credit_dx = dx
        return max(min(release, self.RESUME_RELEASE_FLOOR_S),
                   release - credit[3] - (self._wall - credit[0]))

    def _walk_room(self, point: Waypoint, pos, dx: float) -> float | None:
        """Floor left ahead on the waypoint's platform; None forbids far pulses."""
        if self.approach_fine_radius_x is None or point.action != "walk":
            return None
        if not self.platforms:
            return math.inf
        platform = self._containing_platform((point.x, point.y))
        if platform is None:
            return None
        return pos[0] - platform.left if dx < 0 else platform.right - pos[0]

    def _takeoff_room(self, source: Platform | None, pos, dx: float) -> float | None:
        """Floor beyond the takeoff point on its SOURCE platform; None keeps the fine taps.

        Only with ``takeoff_far_approach`` and a fresh position strictly on the
        known source surface. The takeoff tolerance, the settle on aligned
        frames and every support test stay with the caller, unchanged.

        The room is measured from the takeoff point (``pos[0] + dx``), not from
        the player. A pulse is sized before the next frame can see a knock-back;
        sized this way it is short enough to stop the edge margin before the
        platform end even when the knock puts the character onto the takeoff
        point as the pulse starts. A takeoff closer to that end than the margin
        plus one fine tap (the gap jumps, 2 px from the gap) therefore keeps
        fine taps in that direction.
        """
        if (not self.takeoff_far_approach or not self.platforms or source is None
                or not source.left <= pos[0] <= source.right
                or source.support_error(pos) > min(1.0, self.tolerance_y)):
            return None
        target_x = pos[0] + dx
        return target_x - source.left if dx < 0 else source.right - target_x

    def _previous_walk_platform(self) -> Platform | None:
        previous = self.waypoints[(self.idx - 1) % len(self.waypoints)]
        return self._containing_platform((previous.x, previous.y)) if previous.action == "walk" else None

    # --- commit_budget: a transition under way is not charged to the waypoint clock ---
    def _reset_commit(self) -> None:
        self._commit_started = self._commit_progress_at = self._commit_best = None
        self._climb_commit_spent = 0.0

    def _commit_begin(self, now: float, best: float = 0.0) -> None:
        """``best`` is the progress measure a fall or rope starts from; a ramp has none."""
        if self._commit_started is None:
            self._commit_started = self._commit_progress_at = now
            self._commit_best = best

    def _commit_progress(self, now: float, value: float, proving: bool = False) -> None:
        """One more whole pixel toward the destination, or a running landing proof."""
        if value >= self._commit_best + 1:
            self._commit_best, self._commit_progress_at = value, now
        elif proving:
            self._commit_progress_at = now

    def _commit_end(self, now: float, credit: bool = True) -> None:
        """Back before a takeoff: the committed time is credited, never the time before it.

        ``credit=False`` leaves that time on the waypoint clock: nothing took off after all.
        Every credited end is bounded in number (takeoffs, retries, ever lower ledges).
        """
        if self._commit_started is not None:
            if credit:
                self._started += now - self._commit_started
            self._commit_started = self._commit_progress_at = self._commit_best = None

    # --- takeoff_zones: a first attempt may take off from a stretch of its source platform ---
    def _reset_zone(self) -> None:
        self._zone_open = True     # no attempt of this waypoint selection has launched or failed
        self._zone_x = None        # the X at which the running settle began
        self._zone_start_x = None  # X_t of a take-off made while the zone was open
        self._zone_land_since = None  # since when a drop has been on its destination, frame after frame

    def _zone(self, idx: int):
        """Cached zone ``(lo, hi)`` of waypoint ``idx``; None is the saved take-off alone.

        A jump, ramp or drop gets its own zone from the saved plan. A walk gets
        the zone of the transition that takes off from its platform at its X,
        so the walk in front of a zoned take-off does not demand the saved
        pixel either. A zone lies inside its source platform's span: a position
        inside the zone is inside that span.
        """
        if idx not in self._zone_cache:
            from maplebot.nav.zones import takeoff_zone

            point, meta, zone = self.waypoints[idx], self._wp_meta(idx), None
            if point.action == "walk":
                ahead = idx + 1 if idx + 1 < len(self.waypoints) else 0 if self.loop else None
                if ahead is not None and self.waypoints[ahead].action in self._zone_kinds:
                    target = self.waypoints[ahead]
                    takeoff_x = target.x if target.action == "drop" else target.from_x
                    if (meta.get("platform_id") is not None and abs(point.x - takeoff_x) <= 1e-6
                            and self._wp_meta(ahead).get("from_platform") == meta.get("platform_id")):
                        zone = self._zone(ahead)
            elif point.action in self._zone_kinds:
                before = (idx - 1) % len(self.waypoints)
                # A drop has no take-off of its own: it leaves from the walk waypoint before it.
                if point.action != "drop" or (
                        self.waypoints[before].action == "walk" and abs(self.waypoints[before].x - point.x) <= 1e-6
                        and self._wp_meta(before).get("platform_id") == meta.get("from_platform")):
                    zone = takeoff_zone(
                        self._cfg["editor_plan"], kind=point.action, from_platform=meta.get("from_platform"),
                        to_platform=meta.get("to_platform"), to_x=point.x,
                        from_x=point.x if point.action == "drop" else point.from_x,
                        max_x=self.takeoff_zone_max_x, margin_x=self.takeoff_zone_margin_x,
                        legacy_tolerance_x=self._legacy_tolerance(point.action))
            self._zone_cache[idx] = zone
        return self._zone_cache[idx]

    def _legacy_tolerance(self, action: str) -> float:
        """The take-off alignment tolerance of a transition kind without a zone."""
        return (self.jump_align_tolerance_x if action == "jump" else self.drop_align_tolerance_x
                if action == "drop" else max(self.align_tolerance_x, self.tolerance_x))

    def _open_zone(self):
        """The current transition's zone while no attempt of this selection has launched or failed."""
        return self._zone(self.idx) if self._zone_open else None

    @staticmethod
    def _inside(zone, pos) -> bool:
        return zone is not None and zone[0] <= pos[0] <= zone[1]

    def _zone_settle(self, x: float) -> None:
        """X stillness while a zone is open: a settle belongs to the pixel it began on."""
        if self._aligned_since is None or self._zone_x is None or abs(x - self._zone_x) > 0.5:
            self._aligned_since, self._zone_x = None, x

    def _zone_takeoff(self, zone, x: float) -> None:
        """Every take-off closes the zone; X_t is kept only for one made while it was open."""
        point = self.waypoints[self.idx]
        self._zone_open, self._zone_start_x = False, x if zone is not None else None
        saved_x = point.x if point.action == "drop" else point.from_x
        if zone is not None and abs(x - saved_x) > self._legacy_tolerance(point.action):
            self.zone_takeoffs += 1

    def _zone_spent(self) -> None:
        """A failed attempt or a lower ledge: the saved spot and its own tolerance from here on."""
        self._zone_open, self._zone_start_x = False, None

    def _ramp_from_x(self, point: Waypoint) -> float:
        """Where the running ramp attempt started: X_t inside an open zone, else the saved X."""
        if self._zone_kinds and self._zone_start_x is not None:
            return self._zone_start_x
        return point.from_x

    def _zone_landed(self, point: Waypoint, pos, destination: Platform | None) -> bool:
        """A drop launched from its zone outside the legacy window lands beside the saved X.

        Asked only after the unchanged landing proof. The window is the zone
        plus the sideways drift of a fall, inside the destination platform.
        """
        zone, drift = self._zone(self.idx), self.ZONE_LANDING_DRIFT_X
        return (zone is not None and self._zone_start_x is not None
                and abs(self._zone_start_x - point.x) > self.drop_align_tolerance_x
                and destination is not None and destination.left <= pos[0] <= destination.right
                and zone[0] - drift <= pos[0] <= zone[1] + drift)

    def _child_zone_kinds(self) -> tuple:
        """What a return/rejoin child may take from a zone: our drops and ramps, never a jump."""
        return tuple(kind for kind in ("drop", "ramp") if kind in self._zone_kinds)

    def _step_child(self, child: "MinimapCircuit", pos, now: float, on_ladder: bool) -> Command:
        """Step a return/rejoin child; its zone take-offs count in this circuit's total."""
        if not self._zone_kinds:
            return child.step(pos, now, on_ladder)
        counted = child.zone_takeoffs
        command = child.step(pos, now, on_ladder)
        self.zone_takeoffs += child.zone_takeoffs - counted
        return command

    def _precision_approach(self, dx: float, now: float, room: float | None = None) -> Command:
        """Release before small corrections so momentum and capture lag can settle."""
        direction = "left" if dx < 0 else "right"
        self._aligned_since = None
        self._approached = True
        if abs(dx) > self.approach_radius_x:
            self._reset_alignment()
            return Command(direction)
        release, pulse, far = self.approach_release_s, self.approach_pulse_s, False
        if room is not None and abs(dx) > self.approach_fine_radius_x:
            # One keyboard-enforced pulse sized from the remaining distance at a
            # conservative (upper-bound) speed. It stops short of the target and
            # of the platform edge, so its length does not depend on frame rate.
            planned = min(min(abs(dx) - self.approach_stop_margin_x, room - self.approach_edge_margin_x)
                          / self.approach_walk_speed_x, self.approach_far_pulse_max_s)
            if planned >= self.approach_pulse_s:
                release, pulse, far = self.approach_far_release_s, planned, True
        if self._pulse_until is None or self._pulse_direction != direction:
            self._pulse_direction = direction
            self._pulse_on = False
            self._release_since = now
            self._pulse_until = now + self._resume_release(release, direction, dx)
        elif self._credit_dx is not None and abs(dx - self._credit_dx) > 0.5:
            # Moved after a credited resume: the credit was wrong, release in full.
            self._credit_dx = None
            self._pulse_on = False
            self._release_since = now
            self._pulse_until = now + release
        elif far != self._pulse_far:
            # Crossing the far/fine boundary never carries a pulse across it.
            if self._pulse_on or self._release_since is None:
                self._release_since = now
            self._pulse_on = False
            self._pulse_until = self._release_since + release
        elif now >= self._pulse_until:
            self._pulse_on = not self._pulse_on
            if self._pulse_on:
                self._pulse_until = now + pulse
            else:
                # The keyboard released a far pulse at its deadline, not at this frame.
                self._release_since = self._pulse_until if far else now
                self._pulse_until = self._release_since + release
                if far and now >= self._pulse_until:
                    self._pulse_on = True
                    self._pulse_until = now + pulse
        elif far and self._pulse_on:
            self._pulse_until = min(self._pulse_until, now + pulse)  # fresh frames may only shorten it
        self._pulse_far, self._pulse_dx = far, dx
        if self._pulse_on:
            self._credit_dx = None
        # The keyboard must end a short correction even if the next vision frame
        # arrives late. Repeated decisions retain only this pulse's remaining time.
        return (Command(direction, valid_for_s=self._pulse_until - now)
                if self._pulse_on else Command.NONE)

    def _advance(self, now: float) -> None:
        if self.idx == len(self.waypoints) - 1:
            self.return_count = 0  # only completing the original lap replenishes its return budget
            self.rejoin_count = 0
            if self._reentry_enabled:
                self.reentry_count = 0  # the only reset of the gap re-entry budget
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
        self._rejoin_failed_idx = None  # at most one failed rejoin per waypoint selection
        self._reset_offroute()
        self._reset_commit()
        self._reset_zone()
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
        backward = (self._ramp_from_x(point) - pos[0]) * sign
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
            if self.commit_budget:
                self._commit_end(now)  # the pre-takeoff clock resumes where the takeoff left it
        # The next fresh observation uses the ordinary precision approach. The
        # original ramp timeout and the one-retry budget are never replenished.
        return Command.NONE

    def _step_ramp(self, pos, now: float) -> Command:
        """Walk a declared slope; never improvise a jump or reverse after a fall."""
        point = self.waypoints[self.idx]
        if self.reason is not None:
            self.state = "blocked"
            return Command.NONE
        committed = self.commit_budget and self._commit_started is not None
        if not committed and now - self._started >= self.timeout_s:
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
                meta = self._wp_meta(self.idx)
                if self._platform_rejoin_due(pos, now, meta.get("from_platform"),
                                             (meta.get("to_platform"),)):
                    self._begin_platform_rejoin(pos, now, meta.get("from_platform"))
                return Command.NONE
            self._reset_offroute()
            dx = point.from_x - pos[0]
            zone = self._open_zone() if self._zone_kinds else None
            if abs(dx) > max(self.align_tolerance_x, self.tolerance_x) and not self._inside(zone, pos):
                self.state = "ramp_aligning"
                return self._precision_approach(dx, now, self._takeoff_room(source, pos, dx)
                                                if self.takeoff_far_approach else None)
            self._pulse_until = self._pulse_direction = None
            if self._ramp_source_y is None or abs(pos[1] - self._ramp_source_y) > 0.5:
                self._aligned_since = None
                self._ramp_source_y = pos[1]
            if zone is not None:
                self._zone_settle(pos[0])
            if self._aligned_since is None:
                self._aligned_since = now
            if now - self._aligned_since < self.align_settle_s:
                self.state = "ramp_settling"
                return Command.NONE
            self._ramp_started = True
            if self._zone_kinds:
                self._zone_takeoff(zone, pos[0])
            if self.commit_budget:
                self._commit_begin(now)
                committed = True

        # A ramp that took off inside its zone is judged from where it started.
        from_x = self._ramp_from_x(point)
        low_x, high_x = sorted((from_x, point.x))
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
            backward = (from_x - pos[0]) * (1 if direction == "right" else -1)
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
        if committed:
            # Bounded from the takeoff alone; a running landing proof gets its dwell on
            # top. There is deliberately no no-progress rule here: a verified lap stood
            # 5 s on the first stair pixel with the key held and then walked up
            # (log/claude-circuits-6.jsonl, lap 1, waypoint 22). The corridor and
            # support checks above already end every ramp that leaves its stair.
            budget = self.ramp_travel_timeout_s + (max(self.ramp_landing_s, point.dwell_s) if arrived else 0)
            if now - self._commit_started >= budget:
                self.reason, self.state = "ramp timeout", "blocked"
                return Command.NONE
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
        committed = self.commit_budget and self._commit_started is not None
        if not committed and now - self._started >= self.timeout_s:
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
        if self._zone_kinds:
            # Since when every fresh frame has been on the destination: the dwell of a
            # landing accepted beside the saved X (``stable_for`` follows the height only).
            if not at_destination:
                self._zone_land_since = None
            elif self._zone_land_since is None:
                self._zone_land_since = now
        if committed:
            # From the launch (or an observed departure) the fall has its own budget;
            # the landing proof gets its dwell on top and counts as progress.
            self._commit_progress(now, pos[1], at_destination)
            budget = self.drop_fall_timeout_s + (max(0.25, point.dwell_s) if at_destination else 0)
            if now - self._commit_started >= budget:
                return self._block_drop("drop timeout")
        if at_destination:
            self._drop_started = self._drop_launch_ended = True
            self.state = "dwelling"
            if stable_for < max(0.25, point.dwell_s):
                return Command.NONE
            if committed:
                # Landing proven: the correction on the destination floor runs on
                # the waypoint clock again, credited with the committed time.
                self._commit_end(now)
            if abs(dx) <= self.tolerance_x:
                self._advance(now)
                return Command.NONE
            if self._zone_kinds and self._zone_landed(point, pos, destination):
                # Beside the saved X the whole dwell must have been on the destination,
                # not a height held elsewhere and then one frame here.
                if now - self._zone_land_since >= max(0.25, point.dwell_s):
                    self._advance(now)
                return Command.NONE
            # Horizontal correction is allowed only after a stable destination landing.
            return self._precision_approach(dx, now)
        if pos[1] > self._drop_source_y + 1:
            if self.commit_budget and not self._drop_departed and not committed:
                self._commit_begin(now, pos[1])  # also a fall nobody launched (knocked off)
                committed = True
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
                self._zone_spent()  # the zone was drawn for the saved source floor, not for this ledge
                self._reset_alignment()
                if committed:
                    self._commit_end(now)  # a new takeoff from this lower ledge
            elif stable_for >= max(0.4, self.align_settle_s) and abs(pos[1] - self._drop_source_y) <= 1:
                self._drop_departed = False  # returning to the same floor does not reset its retry budget
                if committed and self._drop_launched_at is None:
                    # Nobody launched and nothing fell: a dip that came back was pre-takeoff
                    # time. Credited, repeated dips would postpone "drop timeout" without a count.
                    self._commit_end(now, credit=False)
            elif committed and now - self._commit_progress_at >= self.transition_stall_s:
                return self._block_drop("drop stalled")
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
                self._zone_spent()
                self._reset_alignment()
                if committed:
                    self._commit_end(now)  # never left the floor: re-aligning is pre-takeoff time
            return Command.NONE
        if abs(pos[1] - self._drop_source_y) > 1:
            self.state = "dropping"
            return Command.NONE
        zone = self._open_zone() if self._zone_kinds else None
        if abs(dx) > self.drop_align_tolerance_x and not self._inside(zone, pos):
            self.state = "aligning"
            return self._precision_approach(dx, now, self._takeoff_room(self._previous_walk_platform(), pos, dx)
                                            if self.takeoff_far_approach else None)
        self._pulse_until = self._pulse_direction = None
        if zone is not None:
            self._zone_settle(pos[0])
        if self._aligned_since is None:
            self._aligned_since = now
        if now - self._aligned_since < self.align_settle_s or stable_for < self.align_settle_s:
            self.state = "settling"
            return Command.NONE
        self._drop_started = True
        self._drop_launched_at = now
        self._drop_launch_ended = False
        self.state = "dropping"
        if self._zone_kinds:
            self._zone_takeoff(zone, pos[0])
        if self.commit_budget:
            self._commit_begin(now, pos[1])
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
        self._zone_spent()
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

    def _off_jump_source(self, pos, point) -> bool:
        source = self._containing_platform((point.from_x, point.from_y))
        return (self._support_error(point.from_x, point.from_y, pos) > min(1.0, self.tolerance_y)
                or (source is not None and not source.left <= pos[0] <= source.right))

    def _step_jump_arrival(self, pos, now: float, point, direction: str) -> Command | None:
        """Accept a jump whose destination already supports the character.

        A knock-back can carry the character across a narrow gap before the
        takeoff. This never presses Jump: it proves strict support on the
        connection's saved destination platform over fresh frames, walks to the
        landing X like a short landing does, then requires the strictest landing
        dwell the flight uses. ``None`` leaves the frame to the ordinary code.
        """
        to_id = self._wp_meta(self.idx).get("to_platform")
        destination = self._containing_platform((point.x, point.y))
        # The entry proof needs the destination to be the ONE strict support. After
        # it passed, a column where a stair toe overlaps the same floor is still
        # the destination: its own saved surface must keep the strict 0.75 support.
        strict = (self._strict_platform_id(pos) == to_id if self._jump_phase == "align"
                  else self._plan_platform_supports(to_id, pos))
        on_destination = (to_id is not None and destination is not None
                          and destination.left <= pos[0] <= destination.right
                          and self._support_error(point.x, point.y, pos) <= min(1.0, self.tolerance_y)
                          and strict)
        if self._jump_phase == "align":
            if not on_destination or not self._off_jump_source(pos, point):
                return None
            self.state = "jump_arrived_checking"
            self._reset_alignment()
            if self._offroute_stable(pos, to_id, now):
                self._reset_offroute()
                self._jump_phase = "arrived"
                self._jump_landing_since = self._jump_landing_y = None
                self._jump_landing_frames = 0
                self._jump_destination_seen = False
            return Command.NONE
        if not on_destination:
            # Support was lost before the proof finished: nothing is latched.
            self._reset_jump()
            self._reset_alignment()
            self._reset_offroute()
            self.state = "jump_waiting_floor"
            return Command.NONE
        forward_overshoot = (pos[0] - point.x) * (1 if direction == "right" else -1)
        in_window = (abs(point.x - pos[0]) <= self.tolerance_x
                     or (direction != "none" and 0 < forward_overshoot <= self.jump_landing_overshoot_x)
                     or self._jump_destination_seen)
        if not in_window:
            self._jump_landing_since = self._jump_landing_y = None
            self._jump_landing_frames = 0
            self.state = "jump_arrived_aligning"
            return self._precision_approach(point.x - pos[0], now)
        self._jump_destination_seen = True
        if self._jump_landing_since is None or abs(pos[1] - self._jump_landing_y) > 0.5:
            self._jump_landing_since, self._jump_landing_y = now, pos[1]
            self._jump_landing_frames = 1
        else:
            self._jump_landing_frames += 1
        self.state = "jump_arrived"
        # No descent was observed, so the longest landing confirmation applies.
        if (self._jump_landing_frames >= 3
                and now - self._jump_landing_since >= max(self.jump_landing_s, point.dwell_s, 0.65)):
            self.arrival_count += 1
            self._advance(now)
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
        # commit_budget: a flight ends by its own flight / landing-confirmation
        # bounds below, never by a waypoint clock that alignment already spent.
        if (not (self.commit_budget and self._jump_phase == "flight")
                and now - self._started >= self.timeout_s):
            return self._jump_failed(now)
        direction = "left" if point.x < point.from_x else "right" if point.x > point.from_x else "none"
        if self._arrive_enabled and self._jump_phase in ("align", "arrived"):
            arrival = self._step_jump_arrival(pos, now, point, direction)
            if arrival is not None:
                return arrival
        if self._jump_phase == "align":
            if self._off_jump_source(pos, point):
                self._reset_alignment()
                self.state = "jump_waiting_floor"
                meta = self._wp_meta(self.idx)
                if self._platform_rejoin_due(pos, now, meta.get("from_platform"),
                                             (meta.get("to_platform"),)):
                    self._begin_platform_rejoin(pos, now, meta.get("from_platform"))
                return Command.NONE
            self._reset_offroute()
            dx = point.from_x - pos[0]
            zone = self._open_zone() if self._zone_kinds else None
            if abs(dx) > self.jump_align_tolerance_x and not self._inside(zone, pos):
                self.state = "jump_aligning"
                return self._precision_approach(dx, now, self._takeoff_room(
                    self._containing_platform((point.from_x, point.from_y)), pos, dx)
                    if self.takeoff_far_approach else None)
            self._pulse_until = self._pulse_direction = None
            if self._jump_takeoff_y is None or abs(pos[1] - self._jump_takeoff_y) > 0.5:
                self._aligned_since = None
                self._jump_takeoff_y = pos[1]
            if zone is not None:
                self._zone_settle(pos[0])
            if self._aligned_since is None:
                self._aligned_since = now
            if now - self._aligned_since < self.align_settle_s:
                self.state = "jump_settling"
                return Command.NONE
            self._jump_phase = "flight"
            self._jump_started = now
            self._jump_min_y = pos[1]
            self.state = "jumping"
            if self._zone_kinds:
                self._zone_takeoff(zone, pos[0])
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
            if (self.commit_budget and flight_s >= self.jump_flight_timeout_s
                    + self.landing_confirm_timeout_s + required_dwell):
                return self._jump_failed(now)  # a landing that never holds still
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

    def _climb_commit(self, point, pos, dx: float, reached_y: bool, t: float) -> tuple[bool, bool]:
        """(on the rope, budget spent or stalled) from minimap height alone.

        On the rope means: under the rope's X, more than ``tolerance_y`` from
        the previous walk waypoint's floor toward the destination, destination
        height not reached. The screen-space ladder flag is never consulted.
        Rope time is cumulative per waypoint selection, so leaving and
        re-entering the rope cannot extend it.
        """
        previous = self.waypoints[(self.idx - 1) % len(self.waypoints)]
        on_rope, travelled = False, 0.0
        if previous.action == "walk":
            floor = self._height_at(previous.x, previous.y, pos[0])
            travelled = (floor - pos[1]) * (1 if point.y < floor else -1)
            on_rope = (floor != point.y and abs(dx) < self.climb_jitter_x
                       and travelled > self.tolerance_y and not reached_y)
        if not on_rope:
            if self._commit_started is not None:
                self._climb_commit_spent += t - self._commit_started
                self._commit_end(t)
            return False, False
        self._commit_begin(t, travelled)
        self._commit_progress(t, travelled)
        if (self._climb_commit_spent + t - self._commit_started >= self.climb_travel_timeout_s
                or t - self._commit_progress_at >= self.transition_stall_s):
            # The ordinary timeout branch (retry rest or "waypoint timeout") takes over.
            self._climb_commit_spent += t - self._commit_started
            self._commit_end(t)
            return True, True
        return True, False

    def _reset_offroute(self) -> None:
        self._off_pos = self._off_id = self._off_since = None
        self._off_frames = 0

    def _offroute_stable(self, pos, platform_id, t: float) -> bool:
        """Three fresh stepped frames, half a second, half a pixel, one strict platform."""
        if (self._off_pos is None or self._off_id != platform_id
                or max(abs(pos[i] - self._off_pos[i]) for i in (0, 1)) > .5):
            self._off_pos, self._off_id, self._off_since = pos, platform_id, t
            self._off_frames = 1
        else:
            self._off_frames += 1
        return self._off_frames >= 3 and t - self._off_since >= .5

    def _strict_platform_id(self, pos):
        """The one drawn platform supporting ``pos``; None when unsupported or ambiguous."""
        from maplebot.nav.recovery import supported_platform

        try:
            return supported_platform(self._cfg, pos)["id"]
        except ValueError:
            return None

    def _plan_platform_supports(self, platform_id, pos) -> bool:
        """The saved platform ``platform_id`` itself supports ``pos`` by the strict rule."""
        for item in (self._cfg.get("editor_plan") or {}).get("platforms", ()):
            if item.get("id") == platform_id:
                surface = Platform(item["y"], item["left"], item["right"],
                                   item.get("right_y"), item.get("knots", ()))
                return surface.left <= pos[0] <= surface.right and surface.support_error(pos) <= .75
        return False

    def _wp_meta(self, idx: int) -> dict:
        return self._cfg["waypoints"][idx]

    def _arrival_jump_index(self, pos, expected_id):
        """The jump this walk leads to, when the character already stands on its destination.

        Only walk waypoints on ``expected_id`` may lie between the current
        waypoint and a jump that takes off from ``expected_id`` and lands on the
        one platform strictly supporting ``pos``. Returns (index, platform id).
        """
        if not self._arrive_enabled or self.reason is not None or expected_id is None:
            return None
        current = self._strict_platform_id(pos)
        if current is None or current == expected_id:
            return None
        for idx in range(self.idx + 1, len(self.waypoints)):
            meta, action = self._wp_meta(idx), self.waypoints[idx].action
            if action == "walk" and meta.get("platform_id") == expected_id:
                continue
            if (action == "jump" and meta.get("from_platform") == expected_id
                    and meta.get("to_platform") == current):
                return idx, current
            break
        return None

    def _reentry_target(self, pos):
        """``(walk index, platform id)`` to re-enter after a knock-back across a level gap.

        Pure: nothing is changed. The current waypoint must be a walk on the
        landing platform of a LEVEL gap jump g (``zones.level_gap``, the jump zone
        rule's predicate), with only walks on that platform between g and here,
        and the character must stand strictly on g's take-off platform, the
        platform of the walk g-1 right before g. Then g-1 is the target. No wrap
        across waypoint 0, no stair jump, drop, ramp or climb, no ambiguous or
        unsupported position (e.g. x=89 on the Construction Site right floors),
        and nothing while blocked, returning, rejoining, terminal, recovering,
        during a legacy lower-floor rejoin or once the lap's budget is spent.
        """
        if (not self._reentry_enabled or not self._selected or self.completed or self.reason is not None
                or self.returning_to_start or self.rejoining or self.recovery_terminal
                or self.reentry_count >= self.max_gap_reentries_per_lap
                or self._recovery_until is not None or self._rejoin_floor is not None
                or pos is None or self.waypoints[self.idx].action != "walk"):
            return None
        expected = self._wp_meta(self.idx).get("platform_id")
        current = self._strict_platform_id(pos)
        if expected is None or current is None or current == expected:
            return None
        j = self.idx - 1
        while j >= 1 and self.waypoints[j].action == "walk" and self._wp_meta(j).get("platform_id") == expected:
            j -= 1
        if j < 1:
            return None
        jump, before = self.waypoints[j], self.waypoints[j - 1]
        meta = self._wp_meta(j)
        if (jump.action != "jump" or meta.get("from_platform") != current or meta.get("to_platform") != expected
                or before.action != "walk" or self._wp_meta(j - 1).get("platform_id") != current):
            return None
        source = self._containing_platform((jump.from_x, jump.from_y))
        destination = self._containing_platform((jump.x, jump.y))
        if source is None or destination is None:
            return None
        from maplebot.nav.zones import level_gap

        if level_gap(source, destination, jump.from_x, jump.x) is None:
            return None
        return j - 1, current

    def _do_reentry(self, target_idx: int, platform_id, t: float) -> Command:
        """Re-select walk ``target_idx`` on ``platform_id``; no input on this frame.

        Like the legacy fall re-selection in ``_walk_floor_ready`` and the
        arrival hand-over, ``_set_waypoint`` resets retries, ``_rejoin_failed_idx``,
        the take-off zone (so the jump's first attempt is zoned again), the commit
        state and the alignment, and starts a fresh ``timeout_s`` window. The
        Brain's pre-takeoff allowance is keyed on the index and is fresh too.
        Bound: at most ``max_gap_reentries_per_lap`` re-entries per lap (only a
        completed lap resets the count); each adds one more selection of walk
        g-1, jump g and walk g+1, each with its own timeout_s and max_retries:
        about (15+0.6+15) s each for walk g-1, jump g and the fresh window of
        the landing walk g+1, plus flight bounds and a frozen pre-fight of up
        to pre_transition_s, roughly 100 s per re-entry and 200 s per lap at
        the default of 2 on the Construction Site settings. rejoin_count,
        return_count, arrival_count and zone_takeoffs are never touched.
        """
        self.reentry_count += 1
        self.reentry_last = {"from_index": self.idx, "to_index": target_idx, "platform": platform_id}
        self._set_waypoint(target_idx, t)
        self.state = "reentered"
        return Command.NONE

    def _platform_rejoin_due(self, pos, t: float, expected_id, exclude=()) -> bool:
        """Stably supported on a drawn platform other than the one this waypoint needs."""
        if (not self._rejoin_enabled or self.returning_to_start or self.reason is not None
                or self.rejoin_count >= self.max_platform_rejoins_per_lap
                or self.idx == self._rejoin_failed_idx or expected_id is None):
            self._reset_offroute()
            return False
        current = self._strict_platform_id(pos)
        if current is None or current == expected_id or current in exclude:
            self._reset_offroute()
            return False
        return self._offroute_stable(pos, current, t)

    def _begin_platform_rejoin(self, pos, t: float, expected_id) -> bool:
        """Plan the saved path back; no input this frame and no index change ever."""
        from maplebot.nav.recovery import plan_rejoin

        self._reset_offroute()
        try:
            child = MinimapCircuit(plan_rejoin(self._cfg, pos, self.idx,
                                               self.platform_rejoin_max_transitions),
                                   far_approach=self.recovery_far_approach, resume_credit=False,
                                   zone_kinds=self._child_zone_kinds())
        except ValueError as exc:
            # Unplannable: latch for this waypoint, spend nothing, keep waiting as before.
            self._rejoin_failed_idx = self.idx
            self.rejoin_state, self.rejoin_reason = "failed", str(exc)
            return False
        self.platform_rejoin = child
        self.rejoining = True
        self.rejoin_count += 1
        self._rejoin_started, self._rejoin_expected = t, expected_id
        self.rejoin_state, self.rejoin_reason = "running", None
        self.state = "rejoin_starting"
        self._reset_alignment()
        return True

    def _finish_platform_rejoin(self, ok: bool, reason=None, t=None) -> Command:
        if self.platform_rejoin is not None:
            self.platform_rejoin.suspend(self._clock.last if self._clock.last is not None else 0)
        self.rejoining = False
        self.platform_rejoin = None
        self._rejoin_started = self._rejoin_expected = None
        self._reset_offroute()
        self._reset_alignment()
        if ok:
            # Physically back on the expected platform: the same waypoint goes on
            # with a fresh timeout window. Index and retries are not touched.
            self.rejoin_state, self.rejoin_reason = "completed", None
            self.state = "rejoined"
            self._started = t
            self._arrived = None
            self._jump_takeoff_y = self._ramp_source_y = None
        else:
            # Not a block: off the expected platform the waypoint keeps its own clock,
            # so the ordinary timeout, block and return/Hold follow as without a rejoin.
            self.rejoin_state, self.rejoin_reason = "failed", str(reason)
            self._rejoin_failed_idx = self.idx
            self.state = "rejoin_failed"
            if t is not None:
                # The last fresh frame stood strictly on the expected platform. The
                # rejoin's own time must not expire the waypoint there (no stuck-recovery
                # jump, no spent retry); retries and the per-waypoint latch are kept.
                self._started = t
                self._arrived = None
                self._jump_takeoff_y = self._ramp_source_y = None
        return Command.NONE

    def cancel_platform_rejoin(self, reason: str) -> None:
        """End an active rejoin without success (map/settings change, superseding return)."""
        if self.rejoining:
            self._finish_platform_rejoin(False, reason)

    def _step_platform_rejoin(self, pos, now: float, on_ladder: bool) -> Command:
        t = self._clock.update(now, pos is not None)
        child = self.platform_rejoin
        if pos is None:
            child.suspend(now)
            self._reset_offroute()
            self.state = "rejoin_waiting_position"
            return Command.NONE
        timed_out = t - self._rejoin_started >= self.platform_rejoin_timeout_s
        current = self._strict_platform_id(pos)
        if current == self._rejoin_expected and (child.completed or not child.transition_committed):
            # The child's index is not proof: confirm the expected platform itself.
            child.suspend(now)
            self.rejoin_state = "confirming"
            self.state = "rejoin_confirming"
            if self._offroute_stable(pos, current, t):
                return self._finish_platform_rejoin(True, t=t)
            if timed_out:
                return self._finish_platform_rejoin(False, "Rejoin timed out", t=t)
            return Command.NONE
        self._reset_offroute()
        self.rejoin_state = "running"
        if child.completed:
            return self._finish_platform_rejoin(False, "Left the waypoint platform before confirmation")
        if timed_out and not child.transition_committed:
            # A committed flight, ramp or rope is never cut: it ends by its own
            # bounded proof or block, and the rejoin fails on the next free frame.
            return self._finish_platform_rejoin(False, "Rejoin timed out")
        command = self._step_child(child, pos, now, on_ladder)
        self.state = "rejoin_" + child.state
        if child.state == "blocked":
            return self._finish_platform_rejoin(
                False, child.reason or "Rejoin route blocked",
                t=t if current == self._rejoin_expected else None)
        return command

    def _reset_return_stability(self) -> None:
        self._return_stable_since = self._return_stable_pos = self._return_platform_id = None
        self._return_stable_frames = 0

    def request_return_to_start(self) -> bool:
        """Queue a guarded return; no original index changes and no immediate input."""
        if self.returning_to_start or self.recovery_terminal or self.movement_test:
            return False
        self.cancel_platform_rejoin("Superseded by return to point 1")
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
        self._regain = None
        self._source_window_started = None
        self._reset_return_stability()
        self.state = "return_waiting_source"
        return True

    def abort_return_to_start(self, reason: str) -> None:
        """Make a return failure terminal until an explicit configuration reload."""
        self.cancel_platform_rejoin(str(reason))
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

    def _step_regain(self, pos, t: float, now: float) -> Command:
        """One unsupported fresh frame of the bounded regain-support stage."""
        from maplebot.nav.regain import SupportRegain

        if self._regain is None:
            self._regain = SupportRegain(self._cfg, timeout_s=self.regain_timeout_s,
                                         max_walk_x=self.regain_max_walk_x)
        command = self._regain.step(pos, t, now)
        # The stable-source window restarts while the stage works; the stage has
        # its own budgets and the whole return keeps return_timeout_s.
        self._source_window_started = t
        if self._regain.failed:
            self.abort_return_to_start(self._regain.reason)
            return Command.NONE
        self.recovery_state = "regaining_support"
        self.state = "return_regain_" + self._regain.phase
        self.recovery_reason = self._regain.detail
        return command

    def _step_return(self, pos, now: float, on_ladder: bool) -> Command:
        from maplebot.nav.recovery import UnsupportedPosition, plan_return_to_start, supported_platform

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
            if self._regain is not None:
                self._regain.interrupt(fresh_deadline=False)
            return Command.NONE
        if self.recovery is None:
            if self._source_window_started is None:
                self._source_window_started = self._return_started
            if t - self._source_window_started >= 3:
                self.abort_return_to_start("Stable source unavailable: " + (self.recovery_reason or "position kept moving"))
                return Command.NONE
            try:
                source = supported_platform(self._cfg, pos)
            except UnsupportedPosition as exc:
                if not self.return_regain_support:
                    self.recovery_reason = str(exc)
                    self._reset_return_stability()
                    return Command.NONE
                self._reset_return_stability()
                return self._step_regain(pos, t, now)
            except ValueError as exc:
                self.recovery_reason = str(exc)
                self._reset_return_stability()
                return Command.NONE
            if self._regain is not None:
                # Supported: the stage goes quiet and proves nothing. A flickering
                # source does not get a new settle deadline each time.
                self._regain.interrupt(fresh_deadline=False)
            if self.recovery_state == "regaining_support":
                self.recovery_state = "waiting_source"
                self.state = "return_waiting_source"
            self.recovery_reason = None
            if not self._return_position_stable(pos, source["id"], t):
                return Command.NONE
            try:
                # The return route ends at connection take-offs and is not the lap the
                # opt-in far pulses were tuned on: it keeps the vision-gated fine taps
                # unless recovery_far_approach opts in. It never uses resume credit.
                self.recovery = MinimapCircuit(plan_return_to_start(self._cfg, pos),
                                               far_approach=self.recovery_far_approach, resume_credit=False,
                                               zone_kinds=self._child_zone_kinds())
            except ValueError as exc:
                self.abort_return_to_start(str(exc))
                return Command.NONE
            self.recovery_state = "running"
            self._reset_return_stability()
        command = self._step_child(self.recovery, pos, now, on_ladder)
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
        self._wall, self._suspended = now, False
        self._approached = False
        try:
            return self._step(pos, now, on_ladder)
        finally:
            self._resume_credit = None  # release credit is for the first step after a suspend only
            if not self._approached and (self.approach_fine_radius_x is not None or self.approach_resume_credit):
                # Dwelling, waiting for a floor, recovering or a lost position: the
                # release/pulse cycle (and any credit built from it) must not outlive
                # the approach it measured. Without the opt-in keys the cycle is left as is.
                self._reset_pulse()

    def _step(self, pos: tuple[int, int] | None, now: float, on_ladder: bool = False) -> Command:
        if self.recovery_terminal:
            self.state = "blocked"
            return Command.NONE
        if self.returning_to_start:
            return self._step_return(pos, now, on_ladder)
        if self.rejoining:
            return self._step_platform_rejoin(pos, now, on_ladder)
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
            self._reset_offroute()
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
        if point.action == "walk":
            # The legacy stable-fall rejoin above keeps precedence while it is pending.
            if not floor_ready and self._rejoin_floor is None and self._recovery_until is None:
                expected = self._wp_meta(self.idx).get("platform_id")
                ahead = self._arrival_jump_index(pos, expected)
                if ahead is not None:
                    # Knocked across the gap before the takeoff walk finished: hand the
                    # frame to that jump waypoint, whose own destination proof decides.
                    if self._offroute_stable(pos, ahead[1], t):
                        self._set_waypoint(ahead[0], t)
                        self.state = "jump_arrived_checking"
                        return Command.NONE
                else:
                    reentry = self._reentry_target(pos) if self._reentry_enabled else None
                    if reentry is not None:
                        # Exactly one _offroute_stable call this frame; the rejoin path is
                        # not evaluated. Until the proof holds, the frame goes on below
                        # (timeout, recovering, waiting_floor) on the unchanged walk clock.
                        if self._offroute_stable(pos, reentry[1], t):
                            return self._do_reentry(reentry[0], reentry[1], t)
                    elif (self._platform_rejoin_due(pos, t, expected)
                            and self._begin_platform_rejoin(pos, t, expected)):
                        return Command.NONE
            else:
                self._reset_offroute()
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
        # In front of a zoned take-off the walk is over anywhere inside that zone; floor,
        # height and dwell are required as at the saved X.
        at_x = abs(dx) <= self.tolerance_x or (self._zone_kinds and self._inside(self._zone(self.idx), pos))
        if floor_ready and at_x and reached_y:
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
        on_rope = expired = False
        if self.commit_budget and point.action == "climb":
            on_rope, expired = self._climb_commit(point, pos, dx, reached_y, t)
        if expired or (not on_rope and t - self._started >= self.timeout_s):
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
            room = (self._takeoff_room(self._previous_walk_platform(), pos, dx)
                    if point.action == "climb" and self.takeoff_far_approach
                    else self._walk_room(point, pos, dx))
            return (Command("left" if dx < 0 else "right") if point.action == "walk" and not self.precision_walk
                    else self._precision_approach(dx, t, room))
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
