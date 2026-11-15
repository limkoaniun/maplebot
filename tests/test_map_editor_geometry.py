"""Exercise editor drag geometry without opening a native window."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from maplebot.map_editor import MapEditor
from maplebot.nav.editor import platform_vertices, validate_plan


def drag_surface(surface, handle, start, destination):
    plan = {"version": 1, "name": "Geometry", "image_size": [100, 100], "loop": False,
            "platforms": [deepcopy(surface)], "route": [], "connections": []}
    editor = SimpleNamespace(
        plan=plan, selected=("platform", surface["id"]),
        drag={"kind": "edit", "handle": handle, "start": start, "before": deepcopy(plan), "changed": True},
        _image=lambda x, y: (x, y), _draw=lambda: None, _pointer=lambda event: None,
        _platform=lambda identifier: plan["platforms"][0],
    )
    MapEditor._motion(editor, SimpleNamespace(x=destination[0], y=destination[1]))
    assert validate_plan(plan, require_route=False) == []
    return plan["platforms"][0]


def test_body_drag_keeps_shape_and_clamps_using_interior_peak():
    original = {"id": "p", "left": 10., "right": 80., "y": 40., "right_y": 50.,
                "knots": [[20.25, 90.], [55.75, 30.]]}
    moved = drag_surface(original, "body", (40, 40), (47, 80))
    assert platform_vertices(moved) == [(x+7, y+9) for x, y in platform_vertices(original)]


def test_interior_handle_moves_only_itself_and_cannot_cross_neighbor():
    original = {"id": "p", "left": 10., "right": 80., "y": 40., "knots": [[20., 60.], [55., 30.]]}
    moved = drag_surface(original, "knot:0", (20, 60), (35, 70))
    assert moved["knots"] == [[35, 70], [55, 30]]
    assert moved["left"] == 10 and moved["right"] == 80 and moved["y"] == 40
    assert drag_surface(original, "knot:0", (20, 60), (55, 70)) == original


def test_endpoint_drag_retains_other_endpoint_with_implicit_right_height():
    original = {"id": "p", "left": 10., "right": 80., "y": 40., "knots": [[30., 60.]]}
    moved = drag_surface(original, "left", (10, 40), (15, 45))
    assert (moved["left"], moved["y"], moved["right"], moved["right_y"]) == (15, 45, 80, 40)
    assert moved["knots"] == original["knots"]
    assert drag_surface(original, "left", (10, 40), (30, 50)) == original
