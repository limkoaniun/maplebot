"""Minimap detection and dot tracking on the in-game minimap."""
from __future__ import annotations

import cv2
import numpy as np

from maplebot.types import Box

WHITE = np.array([255, 255, 255], np.uint8)


def find_minimap(frame_bgr: np.ndarray, min_size: int = 100) -> Box | None:
    """Locate the minimap: a rectangle at least min_size wide/high framed by a 1px pure-white border.

    Returns the content box inside the border (border excluded), or None.
    """
    mask_white = cv2.inRange(frame_bgr, WHITE, WHITE)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_white, connectivity=8)
    for i in range(1, n):
        x0, y0, rw, rh, _area = stats[i]
        if rw < min_size or rh < min_size:
            continue
        x1, y1 = x0 + rw - 1, y0 + rh - 1
        top = np.all(frame_bgr[y0, x0:x1 + 1] == WHITE)
        bottom = np.all(frame_bgr[y1, x0:x1 + 1] == WHITE)
        left = np.all(frame_bgr[y0:y1 + 1, x0] == WHITE)
        right = np.all(frame_bgr[y0:y1 + 1, x1] == WHITE)
        if not (top and bottom and left and right):
            continue
        inner = Box(x0 + 1, y0 + 1, rw - 2, rh - 2)
        if inner.w <= 0 or inner.h <= 0:
            continue
        return inner
    return None


def _color_components(img: np.ndarray, color_bgr, tol: int):
    c = np.array(color_bgr, np.int16)
    lower = np.clip(c - tol, 0, 255).astype(np.uint8)
    upper = np.clip(c + tol, 0, 255).astype(np.uint8)
    mask = cv2.inRange(img, lower, upper)
    return cv2.connectedComponentsWithStats(mask, connectivity=8)


def find_player_dot(
    minimap_bgr: np.ndarray,
    color_bgr,
    tolerance: int = 8,
    last: tuple[int, int] | None = None,
    max_jump_px: int = 30,
    area_range: tuple[int, int] = (2, 50),
) -> tuple[int, int] | None:
    """Find the player's dot: a compact blob of `color_bgr`, nearest to `last` if given.

    Tries widening colour tolerances until any candidate appears. A candidate farther than
    `max_jump_px` (Manhattan) from `last` is rejected (returns None) so static map markers of
    a similar colour cannot steal the lock.
    """
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None
    lo, hi = area_range
    for tol in (tolerance, tolerance + 10, tolerance + 20, tolerance + 30):
        n, _labels, stats, centroids = _color_components(minimap_bgr, color_bgr, tol)
        cands = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if lo <= area <= hi:
                cx, cy = centroids[i]
                pt = (int(round(cx)), int(round(cy)))
                dist = 0 if last is None else abs(pt[0] - last[0]) + abs(pt[1] - last[1])
                cands.append((dist, -area, pt))
        if cands:
            cands.sort()
            dist, _neg_area, pt = cands[0]
            if last is not None and max_jump_px > 0 and dist > max_jump_px:
                return None
            return pt
    return None


def find_other_players(minimap_bgr: np.ndarray, color_bgr, tolerance: int = 20) -> list[tuple[int, int]]:
    """Centres of all blobs of `color_bgr` (other players' dots)."""
    if minimap_bgr is None or minimap_bgr.size == 0:
        return []
    n, _labels, stats, centroids = _color_components(minimap_bgr, color_bgr, tolerance)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 2:
            cx, cy = centroids[i]
            out.append((int(round(cx)), int(round(cy))))
    return out
