"""Take-off zones: where on its source platform a transition's first attempt may start.

A transition normally waits until the character stands on one saved pixel. A zone
is a short stretch of the SOURCE platform around that pixel from which the saved
geometry gives the same outcome: a fall column that only the saved destination can
catch, the stretch of a stair behind a ramp start, the last pixels before a level
gap. It only widens where "at the take-off" is true before the take-off. Landing
proof, support tests and every tolerance stay with the circuit, unchanged.

Pure geometry over the saved editor plan (y grows downward): no clock, keyboard
or I/O, and the plan is never modified. Zones come from the drawing alone and are
kept conservative; recorded runs took off from the saved pixel almost every time,
so they cannot show how wide a zone works. Offline-tested only.
"""
from __future__ import annotations

import math

from maplebot.nav.editor import validate_plan

BAND_Y = 1.5        # height slack around a fall column and around a stair surface
FALL_DRIFT_X = 1.0  # sideways drift during a fall, kept clear on top of the margin
# Never less than this between a drop column and anything else drawn, or the end of its
# destination: a whole-pixel position hides up to 0.5 px, and a fall may drift 2 px (the
# landing window the circuit allows beside a zone).
FALL_REACH_X = 0.5 + 2.0
ROPE_X = 2.0        # a saved rope can catch a fall this close to its column
LEVEL_Y = 0.5       # a drop zone needs source and destination this level nearby
AMBIGUOUS_X = 0.5   # another platform's end still counts as support this far out
GAP_MAX_X = 1.5     # the jump rule covers a gap between two level floors up to this wide
GAP_INSET_X = 0.75  # a gap jump never starts closer to the platform end than this
GAP_LEVEL_Y = 0.75  # and floors whose heights differ by no more than this
GAP_TAKEOFF_X = (1.0, 4.0)  # saved take-off distance from the gap that the rule covers
STEP_X = 0.0625     # scan step; a drop exclusion is at least twice the clearance wide, a rope column 4 px


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _heights(surface, low, high):
    """Smallest and largest surface height over [low, high] clipped to the platform's span."""
    low, high = max(surface.left, low), min(surface.right, high)
    ys = [surface.y_at(low), surface.y_at(high)]
    ys.extend(y for x, y in surface.knots if low < x < high)
    return min(ys), max(ys)


def _run(allowed, x0, low, high):
    """The maximal run of allowed x around ``x0`` inside [low, high]; None when x0 is not in it.

    Both ends are allowed points: a boundary is approached from inside, never passed.
    """
    if not low <= x0 <= high or not allowed(x0):
        return None
    ends = []
    for bound, step in ((low, -STEP_X), (high, STEP_X)):
        x = x0
        while x != bound:
            ahead = max(bound, x + step) if step < 0 else min(bound, x + step)
            if not allowed(ahead):
                for _ in range(24):
                    middle = (x + ahead) / 2
                    if allowed(middle):
                        x = middle
                    else:
                        ahead = middle
                break
            x = ahead
        ends.append(x)
    return ends[0], ends[1]


def _drop_run(source, destination, others, ropes, x0, max_x, margin):
    clearance = max(margin + FALL_DRIFT_X, FALL_REACH_X)
    # The fall must also end on the destination: its ends keep the same clearance.
    low = max(source.left + margin, destination.left + clearance, x0 - max_x)
    high = min(source.right - margin, destination.right - clearance, x0 + max_x)

    def allowed(x):
        top, bottom = source.y_at(x) - BAND_Y, destination.y_at(x) + BAND_Y
        if destination.y_at(x) <= source.y_at(x):
            return False
        for surface in (source, destination):
            highest, lowest = _heights(surface, x - clearance, x + clearance)
            if lowest - highest > LEVEL_Y:
                return False
        for other in others:
            # Anything drawn between the two floors here, or beside the column within
            # the drift a fall may pick up, could catch the character or read as support.
            if other.left - clearance <= x <= other.right + clearance:
                highest, lowest = _heights(other, x - clearance, x + clearance)
                if highest < bottom and lowest > top:
                    return False
        return not any(abs(x - rope_x) <= ROPE_X and rope_top < bottom and rope_bottom > top
                       for rope_x, rope_top, rope_bottom in ropes)

    return _run(allowed, x0, low, high)


def _ramp_run(source, others, x0, to_x, max_x, margin, tolerance):
    # Only a sloped source: there the ramp's corridor follows the drawn surface, not
    # the chord between its two saved endpoints.
    if not source.sloped or to_x == x0:
        return None
    if to_x > x0:
        low, high = min(x0, max(x0 - max_x, source.left + margin)), min(x0 + tolerance, source.right)
    else:
        low, high = max(x0 - tolerance, source.left), max(x0, min(x0 + max_x, source.right - margin))

    def allowed(x):
        # Where another platform lies at the stair's own height (a stair foot beside
        # its floor) support is ambiguous: no take-off from there.
        y = source.y_at(x)
        return not any(other.left - AMBIGUOUS_X <= x <= other.right + AMBIGUOUS_X
                       and abs(other.y_at(x) - y) <= BAND_Y for other in others)

    return _run(allowed, x0, low, high)


def _jump_run(source, destination, x0, to_x, max_x, tolerance):
    if (source.sloped or source.knots or destination.sloped or destination.knots
            or abs(source.y - destination.y) > GAP_LEVEL_Y or to_x == x0
            or not destination.left <= to_x <= destination.right):
        return None
    leftward = to_x < x0
    edge = source.left if leftward else source.right
    gap = source.left - destination.right if leftward else destination.left - source.right
    if not 0 <= gap <= GAP_MAX_X or not GAP_TAKEOFF_X[0] <= abs(x0 - edge) <= GAP_TAKEOFF_X[1]:
        return None
    # Toward the gap only: a start further from it lands shorter on the destination.
    return ((max(edge + GAP_INSET_X, x0 - max_x), min(x0 + tolerance, source.right)) if leftward
            else (max(x0 - tolerance, source.left), min(edge - GAP_INSET_X, x0 + max_x)))


def _zone(plan, kind, from_platform, to_platform, from_x, to_x, max_x, margin_x, legacy_tolerance_x):
    from maplebot.nav.circuit import Platform

    if (kind not in ("drop", "ramp", "jump")
            or not all(_number(v) for v in (from_x, to_x, max_x, margin_x, legacy_tolerance_x))
            or max_x <= 0 or margin_x <= 0 or legacy_tolerance_x < 0
            or validate_plan(plan, require_route=False)):
        return None
    surfaces = {item["id"]: Platform(item["y"], item["left"], item["right"],
                                     item.get("right_y"), item.get("knots", ()))
                for item in plan["platforms"]}
    source, destination = surfaces.get(from_platform), surfaces.get(to_platform)
    x0 = float(from_x)
    if (source is None or destination is None or from_platform == to_platform
            or not source.left <= x0 <= source.right):
        return None
    if kind == "drop":
        if abs(to_x - x0) > 1e-6:
            return None
        ropes = []
        for connection in plan["connections"]:
            if connection["kind"] == "rope":
                rope_x = connection["from_x"]
                ropes.append((rope_x, *sorted((surfaces[connection["from_platform"]].y_at(rope_x),
                                               surfaces[connection["to_platform"]].y_at(rope_x)))))
        others = [surface for ident, surface in surfaces.items() if ident not in (from_platform, to_platform)]
        run = _drop_run(source, destination, others, ropes, x0, max_x, margin_x)
    elif kind == "ramp":
        others = [surface for ident, surface in surfaces.items() if ident != from_platform]
        run = _ramp_run(source, others, x0, to_x, max_x, margin_x, legacy_tolerance_x)
    else:
        run = _jump_run(source, destination, x0, to_x, max_x, legacy_tolerance_x)
    if run is None:
        return None
    # The legacy window is always part of the zone; a zone that adds nothing to it is none.
    legacy = max(source.left, x0 - legacy_tolerance_x), min(source.right, x0 + legacy_tolerance_x)
    lo, hi = max(source.left, min(run[0], legacy[0])), min(source.right, max(run[1], legacy[1]))
    if lo >= legacy[0] - 1e-9 and hi <= legacy[1] + 1e-9:
        return None
    return lo, hi


def takeoff_zone(plan, *, kind, from_platform, to_platform, from_x, to_x, max_x=8.0, margin_x=1.5,
                 legacy_tolerance_x):
    """Interval ``(lo, hi)`` of take-off X on the source platform, or None for the saved X alone.

    ``plan`` is the saved editor plan. ``from_platform``/``to_platform`` and
    ``from_x``/``to_x`` are the transition's EFFECTIVE direction (a rope or ramp can
    be compiled reversed); a drop passes its landing X as both. The interval lies
    inside the source platform's span, contains ``from_x`` and the legacy window
    ``from_x +/- legacy_tolerance_x`` clipped to that span, and is None when no rule
    applies or nothing wider than the legacy window results. ``max_x`` bounds the
    reach from the saved X and ``margin_x`` the distance kept from platform ends.

    drop  Columns inside both platforms where source and destination are level and
          no other drawn platform or saved rope lies between them or beside the fall;
          the destination's ends are kept as far away as any other platform.
    ramp  Sloped source only: back along the stair, away from the destination, where
          no other platform is at the stair's height. Forward only the legacy window.
    jump  Level gap of at most 1.5 px between two flat platforms, saved take-off 1-4 px
          from the gap: toward the gap up to 0.75 px from the end, never away from it.
    A climb has no zone. Unknown ids, bad numbers or a malformed plan give None.
    """
    try:
        return _zone(plan, kind, from_platform, to_platform, from_x, to_x, max_x, margin_x,
                     legacy_tolerance_x)
    except (ArithmeticError, AttributeError, LookupError, TypeError, ValueError):
        return None  # never raise into the step path: the saved spot stays the only take-off
