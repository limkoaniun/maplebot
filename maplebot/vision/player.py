"""Locate the player's character on screen.

Three strategies, all returning PlayerFix or None:
- locate_by_party_bar: the small red HP bar shown above your character when in a party.
- locate_by_template: a nametag / character-appearance template (optionally split in strips).
"""
from __future__ import annotations

import cv2
import numpy as np

from maplebot.types import Box, PlayerFix
from maplebot.vision.template import match_template


def to_cv_hsv(hsv_std) -> np.ndarray:
    """(H 0-360, S 0-100, V 0-100) -> OpenCV (0-179, 0-255, 0-255)."""
    h, s, v = hsv_std
    return np.array([round(h / 360 * 179), round(s / 100 * 255), round(v / 100 * 255)], np.uint8)


def locate_by_party_bar(frame_bgr: np.ndarray, cfg: dict, exclude: Box | None = None) -> PlayerFix | None:
    """Find the party red bar and return the character position = bar top-left + offset."""
    img = frame_bgr
    if exclude is not None:
        img = frame_bgr.copy()
        e = exclude.clamp(frame_bgr.shape[1], frame_bgr.shape[0])
        img[e.y:e.y2, e.x:e.x2] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, to_cv_hsv(cfg["lower_hsv"]), to_cv_hsv(cfg["upper_hsv"]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_h, max_h, max_w = int(cfg["min_h"]), int(cfg["max_h"]), int(cfg["max_w"])
    best: tuple[int, Box] | None = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (min_h <= h <= max_h and 1 <= w <= max_w):
            continue
        # contourArea under-counts thin shapes, so measure fill on the mask itself
        fill = float(np.count_nonzero(mask[y:y + h, x:x + w])) / float(w * h)
        if fill < 0.7:
            continue
        box = Box(x, y, w, h)
        if best is None or box.area > best[0]:
            best = (box.area, box)
    if best is None:
        return None
    bar = best[1]
    ox, oy = cfg["offset"]
    return PlayerFix(bar.x + int(ox), bar.y + int(oy), score=1.0, source="party_bar")


def locate_by_template(
    frame: np.ndarray,
    tmpl: np.ndarray,
    mask: np.ndarray | None,
    last: tuple[int, int] | None,
    offset: tuple[int, int],
    diff_thres: float,
    split_width: int = 0,
    flip: bool = False,
    method: int = cv2.TM_SQDIFF_NORMED,
    local_radius: int = 60,
    source: str = "template",
) -> PlayerFix | None:
    """Match `tmpl` (BGR or gray; `frame` is converted to match) and anchor at its bottom-centre + offset.

    split_width > 0 matches vertical strips independently and keeps the best one, which survives
    partial occlusion (ladders, other players). flip=True also tries the mirrored template.
    `last` is the previous *fix* position; the local search window is derived from it.
    """
    if tmpl.ndim == 2 and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, tw = tmpl.shape[:2]
    higher = method in (cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED)

    variants = [(tmpl, mask)]
    if flip:
        variants.append((cv2.flip(tmpl, 1), None if mask is None else cv2.flip(mask, 1)))

    last_tl = None
    if last is not None:
        last_tl = (last[0] - offset[0] - tw // 2, last[1] - offset[1] - th)

    best: tuple[float, tuple[int, int]] | None = None
    for t, m in variants:
        n_splits = max(1, tw // split_width) if split_width > 0 else 1
        w_split = tw // n_splits
        for i in range(n_splits):
            xs = i * w_split
            xe = (i + 1) * w_split if i < n_splits - 1 else tw
            sub_t = t[:, xs:xe]
            sub_m = None if m is None else m[:, xs:xe]
            if sub_m is not None and not np.any(sub_m):
                continue
            sub_last = None if last_tl is None else (last_tl[0] + xs, last_tl[1])
            r = match_template(frame, sub_t, sub_m, last_loc=sub_last, local_radius=local_radius,
                               local_thres=diff_thres, method=method)
            score = r.score
            good = score > 1.0 - diff_thres if higher else score < diff_thres
            if not good:
                continue
            tl = (r.loc[0] - xs, r.loc[1])
            better = best is None or (score > best[0] if higher else score < best[0])
            if better:
                best = (score, tl)
    if best is None:
        return None
    score, (x, y) = best
    return PlayerFix(x + tw // 2 + int(offset[0]), y + th + int(offset[1]), score=score, source=source)
