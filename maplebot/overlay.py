"""Debug drawing: boxes, dots and status text on a copy of the frame."""
from __future__ import annotations

import cv2
import numpy as np

from maplebot.brain.combat import attack_boxes
from maplebot.types import Box

RED = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
WHITE = (255, 255, 255)


def _rect(img, box: Box, color, label: str = "", thickness: int = 1):
    cv2.rectangle(img, (box.x, box.y), (box.x2, box.y2), color, thickness)
    if label:
        cv2.putText(img, label, (box.x, max(12, box.y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def route_preview(route_map, idx: int, global_pos: tuple[int, int] | None, zoom: int = 3,
                  size: int = 80) -> np.ndarray | None:
    """A zoomed crop of the route around the player: route pixels green, up/down grey, player yellow."""
    if route_map is None or not route_map.routes or global_pos is None:
        return None
    code = route_map.routes[idx]
    img = route_map.map_bgr.copy()
    ud = np.array(route_map.is_up_down, bool)
    valid = code >= 0
    is_ud = np.zeros_like(valid)
    if ud.size:
        is_ud[valid] = ud[code[valid]]
    img[valid & ~is_ud] = GREEN
    img[is_ud] = (160, 160, 160)
    gx, gy = int(global_pos[0]), int(global_pos[1])
    h, w = img.shape[:2]
    x0, y0 = max(0, gx - size // 2), max(0, gy - size // 2)
    x1, y1 = min(w, x0 + size), min(h, y0 + size)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1].copy()
    if 0 <= gx - x0 < crop.shape[1] and 0 <= gy - y0 < crop.shape[0]:
        crop[gy - y0, gx - x0] = YELLOW
    return cv2.resize(crop, (crop.shape[1] * zoom, crop.shape[0] * zoom), interpolation=cv2.INTER_NEAREST)


def draw_debug(frame_bgr: np.ndarray, snap, cfg: dict, *, minimap_offset=(0, 0)) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    if snap.minimap_box is not None:
        _rect(img, snap.minimap_box, RED, "minimap")
        if snap.minimap_pos is not None:
            mx, my = snap.minimap_pos
            cv2.circle(img, (round(snap.minimap_box.x + mx - minimap_offset[0]),
                            round(snap.minimap_box.y + my - minimap_offset[1])), 4, GREEN, -1)
        for ox, oy in snap.other_players:
            cv2.circle(img, (round(snap.minimap_box.x + ox - minimap_offset[0]),
                            round(snap.minimap_box.y + oy - minimap_offset[1])), 4, RED, 1)

    if snap.player is not None:
        px, py = snap.player
        cv2.circle(img, (px, py), 4, RED, -1)
        for name, box in attack_boxes(snap.player, cfg["attack"], w, h).items():
            colour = MAGENTA if snap.target == name else RED
            _rect(img, box, colour, f"atk {name}")
        if snap.search_box is not None:
            _rect(img, snap.search_box, CYAN, "search")

    for d in snap.detections:
        colour = YELLOW if d.name == "hp_bar" else GREEN
        _rect(img, d.box, colour, f"{d.name} {d.score:.2f}", 2)

    def pct(v):
        return "--" if v is None else f"{v:.0f}%"

    lines = [
        f"FPS {snap.fps:.1f}  state {snap.state}{'  PAUSED' if snap.paused else ''}",
        f"cmd {snap.command}",
        f"HP {pct(snap.hp)}  MP {pct(snap.mp)}  EXP {pct(snap.exp)}",
        f"player {snap.player}  minimap {snap.minimap_pos}  global {snap.global_pos}",
        f"route {snap.route_idx}  mobs {len(snap.detections)}  target {snap.target}",
        "F1 pause  F2 screenshot  F12 quit",
    ]
    for i, text in enumerate(lines):
        y = 20 + 20 * i
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    prev = route_preview(snap.route_map, snap.route_idx or 0, snap.global_pos, int(cfg.get("debug", {}).get("route_zoom", 3)))
    if prev is not None:
        ph, pw = prev.shape[:2]
        if ph < h and pw < w:
            img[10:10 + ph, w - pw - 10:w - 10] = prev
            cv2.rectangle(img, (w - pw - 10, 10), (w - 10, 10 + ph), WHITE, 1)
    return img
