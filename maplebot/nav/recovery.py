"""Return routes use only the saved editor graph and observed platform support."""
from copy import deepcopy
import math

from maplebot.nav.editor import compile_plan, validate_plan


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
        raise ValueError("Current position is not supported by a drawn platform")
    if len(matches) != 1:
        raise ValueError("Current platform is ambiguous")
    return matches[0]


def plan_return_to_start(cfg, pos):
    """Compile actual platform/X to original point 1; do not modify saved settings."""
    from maplebot.nav.circuit import Platform

    cfg = _circuit_config(cfg)
    source = supported_platform(cfg, pos)
    points = cfg.get("waypoints") or []
    if not points:
        raise ValueError("Return needs an original first waypoint")
    first = points[0]
    plan = deepcopy(cfg["editor_plan"])
    target_id = first.get("platform_id")
    if target_id is None:
        target_id = supported_platform(cfg, (first["x"], first["y"]))["id"]
    target = next((p for p in plan["platforms"] if p["id"] == target_id), None)
    if target is None:
        raise ValueError("Point 1 refers to a missing platform")
    surface = Platform(target["y"], target["left"], target["right"],
                       target.get("right_y"), target.get("knots", ()))
    if (not surface.left <= first["x"] <= surface.right
            or abs(surface.y_at(first["x"]) - first["y"]) > .75):
        raise ValueError("Point 1 is not on its saved platform")
    plan.update(loop=False, route=[
        {"platform_id": source["id"], "x": pos[0], "dwell_s": 0},
        {"platform_id": target_id, "x": first["x"], "dwell_s": max(.6, first.get("dwell_s", 0))},
    ])
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
