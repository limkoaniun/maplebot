"""Pure map-editor model and compilation into executable minimap waypoints.

Coordinates are pixels in the saved minimap image. Platforms with different IDs
remain distinct even when they share a height; only drawn connections join them.
This module performs no capture, input, or filesystem operations.
"""
from __future__ import annotations

from copy import deepcopy
import heapq
from itertools import count
import math


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def platform_vertices(platform):
    """Return validated endpoint/interior pairs in walking order from left to right."""
    if not all(_number(platform.get(k)) for k in ("left", "right", "y")) or not _number(platform.get("right_y", platform.get("y"))):
        raise ValueError("Platform endpoints must be finite numbers.")
    knots = platform.get("knots", [])
    if not isinstance(knots, (list, tuple)):
        raise ValueError("Platform knots must be a list of [X, Y] pairs.")
    vertices = [(float(platform["left"]), float(platform["y"]))]
    for knot in knots:
        if not isinstance(knot, (list, tuple)) or len(knot) != 2 or not all(_number(v) for v in knot):
            raise ValueError("Platform knots must contain finite [X, Y] pairs.")
        if not vertices[-1][0] < knot[0] < platform["right"]:
            raise ValueError("Platform knots must run strictly left to right between the endpoints.")
        vertices.append((float(knot[0]), float(knot[1])))
    if platform["right"] <= platform["left"]:
        raise ValueError("A platform must have positive horizontal width.")
    vertices.append((float(platform["right"]), float(platform.get("right_y", platform["y"]))))
    return vertices


def platform_y(platform, x):
    """Piecewise surface height; extrapolate the end segments outside the bounds."""
    vertices = platform_vertices(platform)
    first, second = vertices[-2:]
    for a, b in zip(vertices, vertices[1:]):
        if x <= b[0]:
            first, second = a, b
            break
    return first[1] + (float(x)-first[0])*(second[1]-first[1])/(second[0]-first[0])


def _surface_distance(platform, first_x, second_x):
    left, right = sorted((first_x, second_x))
    points = [(left, platform_y(platform, left))]
    points.extend((x, y) for x, y in platform_vertices(platform) if left < x < right)
    points.append((right, platform_y(platform, right)))
    return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))


def _size(size) -> tuple[float, float]:
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or not all(_number(v) and v > 0 for v in size)):
        raise ValueError("image and viewport sizes must contain two positive finite numbers")
    return float(size[0]), float(size[1])


def fit_transform(image_size, viewport_size, padding=0) -> tuple[float, float, float]:
    """Aspect-preserving scale and centered offsets for drawing an image."""
    iw, ih = _size(image_size)
    vw, vh = _size(viewport_size)
    if not _number(padding) or padding < 0 or 2 * padding >= min(vw, vh):
        raise ValueError("padding must leave positive drawing space")
    scale = min((vw - 2 * padding) / iw, (vh - 2 * padding) / ih)
    return scale, (vw - iw * scale) / 2, (vh - ih * scale) / 2


def image_to_canvas(point, image_size, viewport_size, padding=0) -> tuple[float, float]:
    scale, ox, oy = fit_transform(image_size, viewport_size, padding)
    return ox + float(point[0]) * scale, oy + float(point[1]) * scale


def canvas_to_image(point, image_size, viewport_size, padding=0, clamp=True) -> tuple[float, float]:
    scale, ox, oy = fit_transform(image_size, viewport_size, padding)
    x, y = (float(point[0]) - ox) / scale, (float(point[1]) - oy) / scale
    if clamp:
        width, height = _size(image_size)
        x, y = min(max(x, 0), width - 1), min(max(y, 0), height - 1)
    return x, y


def snap_to_platform(plan, x, y, max_distance=8):
    """Return (platform ID, clamped X, platform Y) for the nearest segment."""
    if not all(_number(v) for v in (x, y, max_distance)) or max_distance < 0:
        return None
    candidates = []
    for index, platform in enumerate(plan.get("platforms", [])):
        if not isinstance(platform, dict) or not all(_number(platform.get(k)) for k in ("left", "right", "y")):
            continue
        try:
            vertices = platform_vertices(platform)
        except (TypeError, ValueError):
            continue
        for a, b in zip(vertices, vertices[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            proportion = min(1., max(0., ((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)))
            sx, sy = a[0]+proportion*dx, a[1]+proportion*dy
            distance = math.hypot(x - sx, y - sy)
            if distance <= max_distance:
                candidates.append((distance, index, platform["id"], sx, sy))
    if not candidates:
        return None
    _, _, ident, sx, sy = min(candidates)
    return ident, sx, sy


def _path(plan, source, destination, source_x=0, destination_x=0):
    """Least pixel-distance path over declared connections, including approaches."""
    if source == destination:
        return []
    edges = {p["id"]: [] for p in plan["platforms"]}
    for connection in plan["connections"]:
        edges[connection["from_platform"]].append((connection["to_platform"], connection, False))
        if connection["kind"] in ("rope", "ramp"):
            edges[connection["to_platform"]].append((connection["from_platform"], connection, True))
    platforms = {p["id"]: p for p in plan["platforms"]}
    sequence = count()
    queue = [(0.0, next(sequence), source, source_x, [])]
    best = {(source, source_x): 0.0}
    result, result_cost = None, math.inf
    while queue:
        cost, _, current, current_x, path = heapq.heappop(queue)
        if cost != best.get((current, current_x)) or cost >= result_cost:
            continue
        if current == destination:
            final_cost = cost + _surface_distance(platforms[current], current_x, destination_x)
            if final_cost < result_cost:
                result, result_cost = path, final_cost
            continue
        for target, connection, reverse in edges[current]:
            fx, tx = connection["from_x"], connection["to_x"]
            if reverse:
                fx, tx = tx, fx
            movement = math.hypot(tx - fx, platform_y(platforms[target], tx) - platform_y(platforms[current], fx))
            next_cost = cost + _surface_distance(platforms[current], current_x, fx) + movement + 1.0
            state = (target, tx)
            if next_cost >= best.get(state, math.inf):
                continue
            best[state] = next_cost
            heapq.heappush(queue, (next_cost, next(sequence), target, tx, path + [(connection, reverse)]))
    return result


def _legs(plan):
    route = plan["route"]
    legs = [(i, route[i], route[i + 1]) for i in range(len(route) - 1)]
    if plan["loop"] and len(route) > 1:
        legs.append((len(route) - 1, route[-1], route[0]))
    return legs


def validate_plan(plan, require_route=True) -> list[str]:
    """Return geometry errors and, unless saving a draft, unreachable route legs."""
    if not isinstance(plan, dict):
        return ["Map plan must be an object."]
    errors = []
    if type(plan.get("version")) is not int or plan["version"] != 1:
        errors.append("Unsupported map plan version; expected 1.")
    if not isinstance(plan.get("name"), str) or not plan["name"].strip():
        errors.append("Give this map a name.")
    size = plan.get("image_size")
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 2 for v in size)):
        errors.append("Image size must be [width, height] with integer dimensions of at least 2.")
        width = height = math.inf
    else:
        width, height = size
    if not isinstance(plan.get("loop"), bool):
        errors.append("Loop must be true or false.")
    for field in ("platforms", "connections", "route"):
        if not isinstance(plan.get(field), list):
            errors.append(f"{field.capitalize()} must be a list.")
    if any(not isinstance(plan.get(field), list) for field in ("platforms", "connections", "route")):
        return errors

    platforms = {}
    if require_route and not plan["platforms"]:
        errors.append("Draw at least one platform.")
    for i, platform in enumerate(plan["platforms"]):
        label = f"Platform {i + 1}"
        if not isinstance(platform, dict):
            errors.append(f"{label} must be an object.")
            continue
        ident = platform.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in platforms:
            errors.append(f"{label} needs a unique, nonempty ID.")
            continue
        if not all(_number(platform.get(k)) for k in ("left", "right", "y")) or not _number(platform.get("right_y", platform.get("y"))):
            errors.append(f"{label} needs finite left, right, and Y coordinates.")
            continue
        if not (0 <= platform["left"] < platform["right"] < width and 0 <= platform["y"] < height
                and 0 <= platform.get("right_y", platform["y"]) < height):
            errors.append(f"{label} must run left to right inside the minimap image.")
            continue
        try:
            vertices = platform_vertices(platform)
            if any(not 0 <= y < height for _, y in vertices):
                raise ValueError("Platform knots must stay inside the minimap image.")
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        platforms[ident] = platform

    identifiers = set()
    for i, connection in enumerate(plan["connections"]):
        label = f"Connection {i + 1}"
        if not isinstance(connection, dict):
            errors.append(f"{label} must be an object.")
            continue
        ident = connection.get("id")
        if not isinstance(ident, str) or not ident.strip() or ident in identifiers:
            errors.append(f"{label} needs a unique, nonempty ID.")
        else:
            identifiers.add(ident)
        kind = connection.get("kind")
        if kind not in ("rope", "jump", "drop", "ramp"):
            errors.append(f"{label} must be a rope, jump, drop, or ramp.")
        source = platforms.get(connection.get("from_platform")) if isinstance(connection.get("from_platform"), str) else None
        target = platforms.get(connection.get("to_platform")) if isinstance(connection.get("to_platform"), str) else None
        if source is None or target is None:
            errors.append(f"{label} refers to a missing platform.")
            continue
        if source["id"] == target["id"]:
            errors.append(f"{label} must join two different platforms.")
        if not all(_number(connection.get(k)) for k in ("from_x", "to_x")):
            errors.append(f"{label} needs finite endpoint X coordinates.")
            continue
        fx, tx = connection["from_x"], connection["to_x"]
        from_y, to_y = platform_y(source, fx), platform_y(target, tx)
        if not source["left"] <= fx <= source["right"] or not target["left"] <= tx <= target["right"]:
            errors.append(f"{label} endpoints must lie on their platforms.")
        if kind in ("rope", "drop") and abs(fx - tx) > 1e-6:
            errors.append(f"{label}: {kind} endpoints must have the same X; use a jump for a horizontal gap.")
        if kind == "rope" and from_y == to_y:
            errors.append(f"{label}: a rope must join different heights.")
        if kind == "drop" and to_y <= from_y:
            errors.append(f"{label}: a drop must land on a lower platform.")
        if kind == "jump" and fx == tx and from_y == to_y:
            errors.append(f"{label}: a jump needs distinct takeoff and landing coordinates.")
        if kind == "ramp" and (fx == tx or from_y == to_y):
            errors.append(f"{label}: a ramp needs different X coordinates and different platform heights.")
        if "dwell_s" in connection and (not _number(connection["dwell_s"]) or not 0 <= connection["dwell_s"] <= 10):
            errors.append(f"{label} dwell must be between 0 and 10 seconds.")

    if require_route and len(plan["route"]) < 2:
        errors.append("Add at least two route stops.")
    for i, stop in enumerate(plan["route"]):
        label = f"Route stop {i + 1}"
        if not isinstance(stop, dict):
            errors.append(f"{label} must be an object.")
            continue
        platform = platforms.get(stop.get("platform_id")) if isinstance(stop.get("platform_id"), str) else None
        if platform is None:
            errors.append(f"{label} refers to a missing platform.")
        elif not _number(stop.get("x")) or not platform["left"] <= stop["x"] <= platform["right"]:
            errors.append(f"{label} must lie on its platform.")
        if "dwell_s" in stop and (not _number(stop["dwell_s"]) or not 0 <= stop["dwell_s"] <= 10):
            errors.append(f"{label} dwell must be between 0 and 10 seconds.")
    if errors or not require_route:
        return errors
    for index, source, target in _legs(plan):
        if _path(plan, source["platform_id"], target["platform_id"], source["x"], target["x"]) is None:
            destination = (index + 1) % len(plan["route"])
            errors.append(f"No connection from route stop {index + 1} ({source['platform_id']}) "
                          f"to stop {destination + 1} ({target['platform_id']}). Draw a rope, jump, drop, or ramp.")
    return errors


def compile_plan(plan, base_circuit=None) -> dict:
    """Compile ordered stops and declared graph edges; preserve navigation tuning."""
    errors = validate_plan(plan)
    if errors:
        raise ValueError("\n".join(errors))
    result = deepcopy(base_circuit or {})
    platforms = {p["id"]: p for p in plan["platforms"]}
    points = []

    def walk(platform_id, x, dwell=0):
        point = {"x": x, "y": platform_y(platforms[platform_id], x), "action": "walk",
                 "dwell_s": dwell, "platform_id": platform_id}
        if (points and points[-1]["action"] == "walk" and points[-1]["x"] == x
                and points[-1].get("platform_id") == platform_id):
            points[-1]["dwell_s"] = max(points[-1]["dwell_s"], dwell)
        else:
            points.append(point)

    first = plan["route"][0]
    walk(first["platform_id"], first["x"], first.get("dwell_s", 0.3))
    for _, source, target in _legs(plan):
        for connection, reverse in _path(plan, source["platform_id"], target["platform_id"], source["x"], target["x"]):
            a, b = connection["from_platform"], connection["to_platform"]
            fx, tx = connection["from_x"], connection["to_x"]
            if reverse:
                a, b, fx, tx = b, a, tx, fx
            walk(a, fx)
            kind = connection["kind"]
            point = {"x": tx, "y": platform_y(platforms[b], tx), "action": "climb" if kind == "rope" else kind,
                     "dwell_s": connection.get("dwell_s", 0.35 if kind == "rope" else 0.25),
                     "from_platform": a, "to_platform": b, "connection_id": connection["id"]}
            if kind in ("jump", "ramp"):
                point.update(from_x=fx, from_y=platform_y(platforms[a], fx))
            points.append(point)
        walk(target["platform_id"], target["x"], target.get("dwell_s", 0.3))
    # Two coincident stops are a valid stationary route, while the executor
    # requires two waypoint entries even when all walks merged above.
    if len(points) == 1:
        points.append(deepcopy(points[0]))
    result.update(enabled=True, start_idx=0, loop=plan["loop"], platforms=deepcopy(plan["platforms"]), waypoints=points)
    return result


def plan_from_circuit(circuit, image_size, name="Map route") -> dict:
    """Turn legacy waypoints into editable stops and explicit connections.

    Existing conservative platform bounds are expanded to include their route
    and connection endpoints, so legacy descent positions remain represented.
    """
    width, height = _size(image_size)
    if width != int(width) or height != int(height):
        raise ValueError("image_size must contain integer pixel dimensions")
    points = deepcopy(circuit.get("waypoints", []))
    platforms = []
    for i, platform in enumerate(circuit.get("platforms", [])):
        platforms.append({"id": f"platform-{i + 1}", "left": platform["left"],
                          "right": platform["right"], "y": platform["y"]})
        if "right_y" in platform:
            platforms[-1]["right_y"] = platform["right_y"]
        if "knots" in platform:
            platforms[-1]["knots"] = deepcopy(platform["knots"])
    if not platforms:
        for y in sorted({p["y"] for p in points}, reverse=True):
            xs = [p["x"] for p in points if p["y"] == y]
            left, right = min(xs), max(xs)
            if left == right:
                left, right = max(0, left - 1), min(width - 1, right + 1)
            platforms.append({"id": f"platform-{len(platforms) + 1}", "left": left, "right": right, "y": y})
    plan = {"version": 1, "name": name, "image_size": [int(width), int(height)],
            "platforms": platforms, "connections": [], "route": [], "loop": bool(circuit.get("loop", True))}
    if not points or not platforms:
        return plan

    def floor(x, y):
        return min(platforms, key=lambda p: (abs(platform_y(p, x) - y), max(p["left"] - x, x - p["right"], 0)))

    def include(platform, x):
        left, right = min(platform["left"], x), max(platform["right"], x)
        left_y, right_y = platform_y(platform, left), platform_y(platform, right)
        if platform.get("knots"):
            platform["knots"] = [[px, py] for px, py in platform_vertices(platform) if left < px < right]
        platform.update(left=left, right=right, y=left_y)
        if "right_y" in platform or platform.get("knots"):
            platform["right_y"] = right_y

    keys = set()
    for i, point in enumerate(points):
        target = floor(point["x"], point["y"])
        include(target, point["x"])
        plan["route"].append({"platform_id": target["id"], "x": point["x"],
                              "dwell_s": point.get("dwell_s", 0.3)})
        action = point.get("action", "walk")
        if action not in ("climb", "drop", "jump", "ramp") or (i == 0 and not plan["loop"]):
            continue
        previous = points[i - 1]
        fx = point.get("from_x", point["x"])
        fy = point.get("from_y", previous["y"])
        source = floor(fx, fy)
        if source["id"] == target["id"]:
            continue
        include(source, fx)
        kind = "rope" if action == "climb" else action
        key = (kind, source["id"], target["id"], fx, point["x"])
        reverse_key = (kind, target["id"], source["id"], point["x"], fx)
        if key in keys or (kind in ("rope", "ramp") and reverse_key in keys):
            continue
        keys.add(key)
        plan["connections"].append({"id": f"connection-{len(keys)}", "kind": kind,
                                    "from_platform": source["id"], "to_platform": target["id"],
                                    "from_x": fx, "to_x": point["x"],
                                    "dwell_s": point.get("dwell_s", 0.35 if kind == "rope" else 0.25)})
    return plan
