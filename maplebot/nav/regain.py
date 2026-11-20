"""Bounded "regain support" stage that runs before a return to point 1 is planned.

The character may stand on something the map does not draw (the entry pipes),
hang on a saved rope, sit just past a drawn end, or be in the air. This module
only decides which of Left, Right or Down may be tried next, each as a short
leased command. It never declares support: the caller stops calling it as soon
as the strict ``supported_platform`` check succeeds and then runs its unchanged
source proof. Up and Jump are never produced.

Last resort, "unsaved rope descent": when every walk option of a perch ended
with silent pulses (Left and Right both moved nothing, so the character is
evidently not standing on a walkable surface) and one drawn floor lies
underneath, Down alone is held in short leases. Why Down is the one key that
may be tried there: on a rope or ladder Down descends and releases at the
bottom; on a solid surface Down only crouches; Down does not enter portals
(Up does). Offline-tested only; not verified in the live game.

Pure: no clock, keyboard or I/O. Times are supplied by the caller.
"""
from __future__ import annotations

from maplebot.nav.recovery import _circuit_config
from maplebot.types import Command

SETTLE_FRAMES = 3
SETTLE_S = 0.5
SETTLE_PX = 0.5
SETTLE_TIMEOUT_S = 4.0
PULSE_S = 0.25
PULSE_MIN_S = 0.1
RELEASE_S = 0.3
SPEED_MAX_X = 10.0
SILENT_PULSES = 2
WALK_LEG_TIMEOUT_S = 6.0
DESCEND_LEASE_S = 0.3
DESCEND_STALL_S = 1.5
DESCEND_LEG_TIMEOUT_S = 8.0
ROPE_X = 2.0
SUPPORT_Y = 0.75
EDGE_MARGIN_X = 3.0
LEVEL_GAP_X = 8.0
LEVEL_INSET_X = 1.0
MAX_MOVES = 8
MAX_PERCHES = 4
STEP_GAP_S = 0.75  # wall time between two steps that still counts as continuous

_SILENT_MIN_PULSE_S = 0.2
_SILENT_PX = 0.5
_RELEASE_FRAMES = 2
# Sideways distance outside the span the options were computed for that still
# counts as the same perch: the drift one short pulse may add past a bound.
_PERCH_SLACK_X = EDGE_MARGIN_X
_OPPOSITE = {"left": "right", "right": "left"}


def _platforms(plan):
    from maplebot.nav.circuit import Platform

    return [(item, Platform(item["y"], item["left"], item["right"],
                            item.get("right_y"), item.get("knots", ())))
            for item in plan["platforms"]]


def _floor_below(platforms, x, y):
    """The unique first drawn floor underneath ``(x, y)`` as ``(item, surface)``, else None.

    None also when two drawn floors lie within SUPPORT_Y of each other (ambiguous).
    """
    below = sorted(((surface.support_range(x)[0], index, item, surface)
                    for index, (item, surface) in enumerate(platforms)
                    if surface.left <= x <= surface.right and surface.support_range(x)[0] - y > SUPPORT_Y),
                   key=lambda entry: entry[:2])
    if not below or (len(below) > 1 and below[1][0] - below[0][0] <= SUPPORT_Y):
        return None
    return below[0][2], below[0][3]


def unsaved_rope_option(cfg, pos) -> dict | None:
    """Down-only descent toward the unique drawn floor underneath ``pos``, or None.

    Not part of ``regain_options``: ``SupportRegain`` asks for it only after every
    walk option of the perch ended with "no horizontal progress". The floor is
    chosen by the same rule as the walk-off class. ``cfg`` is not modified.
    """
    cfg = _circuit_config(cfg)
    plan = cfg.get("editor_plan")
    if not isinstance(plan, dict):
        return None
    x, y = float(pos[0]), float(pos[1])
    found = _floor_below(_platforms(plan), x, y)
    if found is None:
        return None
    item, floor = found
    return {"kind": "descend", "unsaved": True, "x": x, "start_y": y,
            "end_y": floor.support_range(x)[1], "platform_id": item["id"]}


def regain_options(cfg, pos, max_walk_x=16.0, prefer=None) -> list[dict]:
    """Ordered moves that may bring ``pos`` onto a drawn platform; drawn geometry only.

    An empty list means nothing can be done from this position. ``cfg`` is not modified.
    """
    cfg = _circuit_config(cfg)
    plan = cfg.get("editor_plan")
    if not isinstance(plan, dict):
        return []
    x, y = float(pos[0]), float(pos[1])
    platforms = _platforms(plan)
    by_id = {item["id"]: surface for item, surface in platforms}
    points = cfg.get("waypoints") or []
    first_x = float(points[0]["x"]) if points else None
    options = []

    # 1. Hanging on a saved rope: Down is the key the climb executor already uses.
    ropes = {}
    for connection in plan.get("connections", []):
        if connection.get("kind") != "rope":
            continue
        source, target = by_id.get(connection["from_platform"]), by_id.get(connection["to_platform"])
        if source is None or target is None:
            continue
        rx = float(connection["from_x"])
        ends = sorted(((source.y_at(rx), connection["from_platform"]),
                       (target.y_at(rx), connection["to_platform"])), key=lambda end: end[0])
        (hi, _), (lo, lower_id) = ends
        if abs(x - rx) <= ROPE_X and hi + SUPPORT_Y < y < lo - SUPPORT_Y:
            ropes.setdefault((rx, lower_id), {"kind": "descend", "x": rx, "end_y": lo,
                                              "platform_id": lower_id})
    if ropes:
        options.append(min(ropes.values(), key=lambda option: abs(x - option["x"])))

    # 2. Just past a drawn end at the same height: walk back onto that platform.
    level = []
    for item, surface in platforms:
        if x > surface.right:
            gap, end, direction, bound = x - surface.right, surface.right, "left", surface.right - LEVEL_INSET_X
        elif x < surface.left:
            gap, end, direction, bound = surface.left - x, surface.left, "right", surface.left + LEVEL_INSET_X
        else:
            continue
        if 0 < gap <= LEVEL_GAP_X and surface.support_error((end, y)) <= SUPPORT_Y:
            toward_first = first_x is not None and ((first_x < x) == (direction == "left")) and first_x != x
            level.append((gap, 0 if toward_first else 1, len(level),
                          {"kind": "walk", "direction": direction, "bound": bound,
                           "platform_id": item["id"]}))
    if level:
        options.append(min(level, key=lambda entry: entry[:3])[3])

    # 3. On something undrawn above a drawn floor: walk off inside that floor's span.
    found = _floor_below(platforms, x, y)
    if found is not None:
        item, floor = found
        width = float(plan["image_size"][0])
        low = max(floor.left + EDGE_MARGIN_X, x - max_walk_x, 1.0)
        high = min(floor.right - EDGE_MARGIN_X, x + max_walk_x, width - 2.0)
        if prefer in _OPPOSITE:
            first = prefer
        elif first_x is not None and first_x != x:
            first = "left" if first_x < x else "right"
        else:
            first = "left" if (floor.left + floor.right) / 2 < x else "right"
        for direction in (first, _OPPOSITE[first]):
            bound = low if direction == "left" else high
            room = x - bound if direction == "left" else bound - x
            if room >= 1.0:
                options.append({"kind": "walk", "direction": direction, "bound": bound,
                                "platform_id": item["id"]})
    return options


class SupportRegain:
    """Settle, then try the options one at a time; every key is a short lease."""

    def __init__(self, cfg, *, timeout_s, max_walk_x):
        self._cfg = cfg
        self.timeout_s = float(timeout_s)
        self.max_walk_x = float(max_walk_x)
        self.phase = "settling"
        self.failed = False
        self.reason = None
        self.detail = "Waiting for the character to stand still"
        self.moves = 0
        self.perches = 0
        self.elapsed = 0.0
        self._last_t = self._last_now = None
        self._perch = None
        self._span = None
        self._options = []
        self._current = None
        self._entered = False
        self._prefer = None
        self._last_end = None
        self._walk_count = 0      # walk options computed for the current perch
        self._walk_ends = []      # how each of them ended, when it ended by a cause
        self._silent_at = []      # where each walk that ended silent stood at that moment
        self._down_tried = False  # a Down leg (saved or unsaved) already ran on this perch
        self._clear_settle()
        self._clear_leg()

    # -- state helpers ----------------------------------------------------
    def _clear_settle(self) -> None:
        self._settle_started = None
        self._still_pos = self._still_since = None
        self._still_frames = 0

    def _clear_leg(self) -> None:
        self._leg_started = None
        self._pulse_until = self._pulse_length = self._pulse_origin = None
        self._release_frames = 0
        self._silent = 0
        self._leg_origin = None
        self._leg_pulses = 0
        self._leg_moved = False
        self._descend_y = self._descend_progress = None

    def _to_settling(self) -> Command:
        self._current = None
        self._entered = False
        self.phase = "settling"
        self.detail = "Waiting for the character to stand still"
        self._clear_settle()
        self._clear_leg()
        return Command.NONE

    def _end_option(self, cause: str) -> Command:
        if self._current is not None and self._current["kind"] == "walk":
            self._walk_ends.append(cause)
        self._last_end = cause
        return self._to_settling()

    def _end_silent(self, pos) -> Command:
        self._silent_at.append((pos[0], pos[1]))
        return self._end_option("no horizontal progress")

    def _silence_proven(self) -> bool:
        """Every walk option of this perch ended with "no horizontal progress"."""
        return (bool(self._walk_count) and len(self._walk_ends) == self._walk_count
                and all(cause == "no horizontal progress" for cause in self._walk_ends))

    def _silent_here(self, pos) -> bool:
        """Whether ``pos`` is where those walks stood when they moved nothing."""
        return all(abs(pos[0] - x) <= ROPE_X and abs(pos[1] - y) <= 1 for x, y in self._silent_at)

    def _unsaved_rope(self, pos):
        """The last-resort Down option for this perch, or None.

        Only when every walk option of the perch ended with silent pulses: Left
        and Right both moved nothing, so this is not a walkable surface. A walk
        that ended any other way (window exhausted, leg timeout, displaced, left
        the object) says nothing of the kind and gets no Down. Down on a rope
        descends and releases at the bottom, Down on a solid surface only
        crouches, and Down does not enter portals (Up does); Up and Jump stay
        excluded. One Down leg per perch: where Down already did nothing on a
        saved rope corridor it is not held again. The silence only speaks for
        the place where it was observed: ``_step_settling`` opens a new perch,
        with its own walks, when the position is no longer within ROPE_X and
        one pixel of height of where every one of those walks stood.
        """
        if self._down_tried or not self._silence_proven():
            return None
        return unsaved_rope_option(self._cfg, pos)

    def _fail(self, pos, cause: str) -> Command:
        self.failed = True
        self.phase = "failed"
        self._current = None
        self.reason = (f"Could not reach a drawn platform from ({pos[0]:g},{pos[1]:g}): {cause}. "
                       "Move the character onto a drawn platform or draw this surface in the map editor.")
        self.detail = self.reason
        return Command.NONE

    def _inside_perch(self, x: float) -> bool:
        """Whether ``x`` is still where the current options were computed for."""
        low, high = self._span
        return low - _PERCH_SLACK_X <= x <= high + _PERCH_SLACK_X

    def interrupt(self, fresh_deadline: bool = True) -> None:
        """A pause, lost position, supported frame or long gap: restart from settling.

        ``fresh_deadline=False`` is for interruptions that are themselves active
        time (a supported frame, a lost dot): a settle that was already waiting
        keeps its deadline, so a flickering source cannot postpone the settle
        timeout forever. The stage clock is never reset here; the caller's active
        clock already leaves pauses and long gaps out of ``t``.
        """
        if self.failed:
            return
        waiting_since = self._settle_started if self._current is None and not fresh_deadline else None
        if self._current is not None:
            self._options.insert(0, self._current)  # restarting it counts as another move
        self._to_settling()
        self._settle_started = waiting_since

    # -- stepping ---------------------------------------------------------
    def step(self, pos, t: float, now: float) -> Command:
        """Advance on one unsupported fresh frame. ``t`` is active time, ``now`` wall time."""
        if self.failed:
            return Command.NONE
        advanced = self._last_t is not None and t > self._last_t
        if self._last_now is not None and not 0 <= now - self._last_now <= STEP_GAP_S:
            # A gap that was not active time is a pause; one that was is a slow source.
            self.interrupt(fresh_deadline=not advanced)
        if self._last_t is not None and t > self._last_t:
            self.elapsed += t - self._last_t
        self._last_t, self._last_now = t, now
        if self.elapsed >= self.timeout_s:
            return self._fail(pos, "timed out")
        if self.phase == "settling":
            return self._step_settling(pos, t)
        if not self._entered:
            # The option was chosen on the previous frame; this frame only starts
            # its clocks, so at least one more no-key frame precedes any key.
            self._entered = True
            self._leg_started = t
            self._leg_origin = (pos[0], pos[1])
            self._descend_y, self._descend_progress = pos[1], t
            return Command.NONE
        if self.phase == "walking":
            return self._step_walking(pos, t)
        return self._step_descending(pos, t)

    def _step_settling(self, pos, t: float) -> Command:
        if self._settle_started is None:
            self._settle_started = t
        if (self._still_pos is None
                or max(abs(pos[i] - self._still_pos[i]) for i in (0, 1)) > SETTLE_PX):
            self._still_pos, self._still_since, self._still_frames = pos, t, 1
        else:
            self._still_frames += 1
        if self._still_frames < SETTLE_FRAMES or t - self._still_since < SETTLE_S:
            if t - self._settle_started >= SETTLE_TIMEOUT_S:
                return self._fail(pos, "position kept moving")
            return Command.NONE
        # A perch is a place, not only a height: options and their bounds hold for
        # the X they were computed from. Standing elsewhere at the same height
        # (moved during a pause, knocked back, a wrong dot) is a new perch.
        # The same holds for silent walks: Left and Right moving nothing at one
        # spot says nothing about another one. Standing elsewhere once they are
        # used up is a new perch, so both walks are tried there before any Down.
        moved_since_silence = (not self._options and not self._down_tried
                               and self._silence_proven() and not self._silent_here(pos))
        if (self._perch is None or abs(pos[1] - self._perch[1]) > 1
                or not self._inside_perch(pos[0]) or moved_since_silence):
            self.perches += 1
            if self.perches > MAX_PERCHES:
                return self._fail(pos, "too many undrawn surfaces on the way down")
            self._perch = (pos[0], pos[1])
            self._options = regain_options(self._cfg, pos, self.max_walk_x, self._prefer)
            self._walk_count = sum(1 for option in self._options if option["kind"] == "walk")
            self._walk_ends = []
            self._silent_at = []
            self._down_tried = False
            if not self._options:
                return self._fail(pos, "no drawn platform below or beside")
            reach = [float(pos[0])]
            for option in self._options:
                reach += ([option["bound"]] if option["kind"] == "walk"
                          else [option["x"] - ROPE_X, option["x"] + ROPE_X])
            self._span = (min(reach), max(reach))
        if not self._options:
            option = self._unsaved_rope(pos)
            if option is None:
                return self._fail(pos, self._last_end or "no drawn platform below or beside")
            self._options.append(option)
        option = self._options.pop(0)
        self.moves += 1
        if self.moves > MAX_MOVES:
            return self._fail(pos, "too many attempts")
        self._current = option
        self._entered = False
        self._clear_leg()
        if option["kind"] == "descend":
            self._down_tried = True
            self.phase = "descending"
            self.detail = (f"Climbing down an unsaved rope toward {option['platform_id']}"
                           if option.get("unsaved") else
                           f"Climbing down the saved rope toward {option['platform_id']}")
        else:
            self.phase = "walking"
            self.detail = f"Walking {option['direction']} toward drawn platform {option['platform_id']}"
        return Command.NONE

    def _step_walking(self, pos, t: float) -> Command:
        option = self._current
        direction, bound, perch_y = option["direction"], option["bound"], self._perch[1]
        if pos[1] > perch_y + 1:  # left the object: keep going the same way from the next perch
            self._prefer = direction
            return self._to_settling()
        if pos[1] < perch_y - 1:  # moved up
            self._prefer = _OPPOSITE[direction]
            return self._to_settling()
        room = pos[0] - bound if direction == "left" else bound - pos[0]
        if room < -0.5:
            return self._end_option("walk window exhausted")
        if not self._inside_perch(pos[0]):
            # Displaced sideways mid-walk: this direction and bound were not
            # computed for here. The next settle recomputes from where it stands.
            return self._end_option("position moved away from the walk window")
        if t - self._leg_started >= WALK_LEG_TIMEOUT_S:
            # With under 2 px of room every pulse is shorter than _SILENT_MIN_PULSE_S
            # and is not judged on its own. A whole leg of them in which nothing
            # moved at all is the same evidence: no horizontal progress.
            if (self._leg_pulses >= SILENT_PULSES and not self._leg_moved
                    and max(abs(pos[i] - self._leg_origin[i]) for i in (0, 1)) < _SILENT_PX):
                return self._end_silent(pos)
            return self._end_option("walk did not reach a drawn platform in time")
        if self._pulse_until is not None:
            if t < self._pulse_until:
                # A fresh frame only ever carries the pulse's remaining time.
                return Command(direction, valid_for_s=self._pulse_until - t)
            self._release_frames += 1
            if t - self._pulse_until < RELEASE_S or self._release_frames < _RELEASE_FRAMES:
                return Command.NONE
            moved = max(abs(pos[i] - self._pulse_origin[i]) for i in (0, 1))
            self._leg_pulses += 1
            if moved >= _SILENT_PX:
                self._silent = 0
                self._leg_moved = True
            elif self._pulse_length >= _SILENT_MIN_PULSE_S:
                self._silent += 1
                if self._silent >= SILENT_PULSES:
                    return self._end_silent(pos)
            self._pulse_until = None
        if room < 1.0:
            return self._end_option("walk window exhausted")
        self._pulse_length = min(PULSE_S, max(PULSE_MIN_S, room / SPEED_MAX_X))
        self._pulse_until = t + self._pulse_length
        self._pulse_origin = (pos[0], pos[1])
        self._release_frames = 0
        return Command(direction, valid_for_s=self._pulse_length)

    def _step_descending(self, pos, t: float) -> Command:
        option = self._current
        if option.get("unsaved"):
            return self._step_unsaved_descent(option, pos, t)
        if abs(pos[0] - option["x"]) > ROPE_X or pos[1] > option["end_y"] + 1:
            return self._end_option("left the saved rope")
        if pos[1] >= self._descend_y + 1:
            self._descend_y, self._descend_progress = pos[1], t
        if (t - self._descend_progress >= DESCEND_STALL_S
                or t - self._leg_started >= DESCEND_LEG_TIMEOUT_S):
            return self._end_option("no descent on the saved rope")
        return Command("none", "down", valid_for_s=DESCEND_LEASE_S)

    def _step_unsaved_descent(self, option, pos, t: float) -> Command:
        """Down only, one short lease per fresh frame; same lease rule as the saved rope.

        Keys stop as soon as the position is no longer a straight descent from
        where it started. Support is never declared here: the caller's strict
        ``supported_platform`` check ends the stage.
        """
        if (abs(pos[0] - option["x"]) > ROPE_X or pos[1] > option["end_y"] + 1
                or pos[1] < option["start_y"] - 1):
            return self._end_option("left the unsaved rope")
        if pos[1] >= self._descend_y + 1:
            self._descend_y, self._descend_progress = pos[1], t
        if t - self._descend_progress >= DESCEND_STALL_S:
            return self._end_option("no descent with Down")
        if t - self._leg_started >= DESCEND_LEG_TIMEOUT_S:
            return self._end_option("descent with Down did not reach a drawn platform in time")
        return Command("none", "down", valid_for_s=DESCEND_LEASE_S)
