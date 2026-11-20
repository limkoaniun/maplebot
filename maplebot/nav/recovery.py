"""Return routes use only the saved editor graph and observed platform support."""
from copy import deepcopy
import math

from maplebot.nav.editor import compile_plan, validate_plan


class UnsupportedPosition(ValueError):
    """The position lies on no drawn platform (as opposed to an ambiguous or invalid one)."""


def _circuit_config(cfg):
    return cfg.get("route", {}).get("circuit", {}) if "route" in cfg else cfg


def supported_platform(cfg, pos):
    """Return one unambiguous drawn platform, never the broad nearest-floor guess."""
    from maplebot.nav.circuit import Platform

    cfg = _circuit_config(cfg)
    plan = cfg.get("editor_plan")
    if not isinstance(plan, dict):
        raise ValueError("Return needs a saved editor plan")
    errors = validate_plan(plan, require_route=False)
    if errors:
        raise ValueError("Invalid return map: " + "; ".join(errors))
    if (pos is None or len(pos) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)):
        raise ValueError("Return needs a current minimap position")
    matches = []
    for item in plan["platforms"]:
        platform = Platform(item["y"], item["left"], item["right"],
                            item.get("right_y"), item.get("knots", ()))
        if platform.left <= pos[0] <= platform.right and platform.support_error(pos) <= .75:
            matches.append(item)
    if not matches:
        raise UnsupportedPosition("Current position is not supported by a drawn platform")
    if len(matches) != 1:
        raise ValueError("Current platform is ambiguous")
    return matches[0]


def _plan_route(cfg, pos, source, target_id, target_x, target_y, *, label, final_dwell_s,
                prefer_non_jump):
    """Compile actual platform/X to one saved platform point over the declared graph."""
    from maplebot.nav.circuit import Platform

    plan = deepcopy(cfg["editor_plan"])
    target = next((p for p in plan["platforms"] if p["id"] == target_id), None)
    if target is None:
        raise ValueError(f"{label} refers to a missing platform")
    surface = Platform(target["y"], target["left"], target["right"],
                       target.get("right_y"), target.get("knots", ()))
    if (not surface.left <= target_x <= surface.right
            or abs(surface.y_at(target_x) - target_y) > .75):
        raise ValueError(f"{label} is not on its saved platform")
    plan.update(loop=False, route=[
        {"platform_id": source["id"], "x": pos[0], "dwell_s": 0},
        {"platform_id": target_id, "x": target_x, "dwell_s": final_dwell_s},
    ])
    if prefer_non_jump:
        # Returning favors declared walks, ropes, ramps, and drops over gap jumps,
        # even when a jump is shorter. Keep jumps when they are the only saved path.
        without_jumps = deepcopy(plan)
        without_jumps["connections"] = [c for c in without_jumps["connections"] if c["kind"] != "jump"]
        if not validate_plan(without_jumps):
            plan = without_jumps
    result = compile_plan(plan, cfg)
    result.update(editor_plan=plan, movement_test=True, return_on_block=False, max_retries=0,
                  tolerance_y=min(.75, float(cfg.get("tolerance_y", 2))))
    return result


def plan_return_to_start(cfg, pos):
    """Compile actual platform/X to original point 1; do not modify saved settings."""
    cfg = _circuit_config(cfg)
    source = supported_platform(cfg, pos)
    points = cfg.get("waypoints") or []
    if not points:
        raise ValueError("Return needs an original first waypoint")
    first = points[0]
    target_id = first.get("platform_id")
    if target_id is None:
        target_id = supported_platform(cfg, (first["x"], first["y"]))["id"]
    return _plan_route(cfg, pos, source, target_id, first["x"], first["y"], label="Point 1",
                       final_dwell_s=max(.6, first.get("dwell_s", 0)), prefer_non_jump=True)


def plan_rejoin(cfg, pos, index, max_transitions=1):
    """Compile actual platform/X to the platform waypoint ``index`` expects.

    A walk waypoint is rejoined at its own point, a jump or ramp at its saved
    takeoff. Only the declared graph is used (shortest saved path, jumps
    included), the path may hold at most ``max_transitions`` connections, and
    the result is a non-looping movement test with both opt-in keys off.
    The saved settings are not modified and no waypoint index is touched.
    """
    cfg = _circuit_config(cfg)
    source = supported_platform(cfg, pos)
    points = cfg.get("waypoints") or []
    if type(index) is not int or not 0 <= index < len(points):
        raise ValueError("Rejoin needs an existing waypoint")
    point = points[index]
    action = point.get("action", "walk")
    if action == "walk":
        target_id, x, y = point.get("platform_id"), point.get("x"), point.get("y")
    elif action in ("jump", "ramp"):
        target_id, x, y = point.get("from_platform"), point.get("from_x"), point.get("from_y")
    else:
        raise ValueError(f"Rejoin does not support {action} waypoints")
    label = f"Point {index + 1}"
    if (not isinstance(target_id, str)
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                       for v in (x, y))):
        raise ValueError(f"{label} has no saved platform to rejoin")
    if source["id"] == target_id:
        raise ValueError("Already on the waypoint platform")
    result = _plan_route(cfg, pos, source, target_id, x, y, label=label, final_dwell_s=0,
                         prefer_non_jump=False)
    if sum(p["action"] != "walk" for p in result["waypoints"]) > max_transitions:
        raise ValueError("Rejoin path is too long")
    result.update(arrive_on_destination=False, platform_rejoin=False)
    return result
