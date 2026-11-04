"""Read HP / MP / EXP percentages from the bottom UI strip."""
from __future__ import annotations

import cv2
import numpy as np

from maplebot.types import Box


def find_status_bars(ui_bgr: np.ndarray, white_thresh: int = 240, aspect: tuple[float, float] = (4, 10),
                     area: tuple[int, int] = (2500, 5000)) -> list[Box]:
    """White-bordered bars in the UI strip, sorted left to right (HP, MP, EXP)."""
    gray = cv2.cvtColor(ui_bgr, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(gray, int(white_thresh), 255)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        if aspect[0] < w / h < aspect[1] and area[0] < w * h < area[1]:
            boxes.append(Box(int(x), int(y), int(w), int(h)))
    return sorted(boxes, key=lambda b: b.x)


def bar_fill_percent(bar_bgr: np.ndarray, grey_tol: int = 10) -> float:
    """Percent filled: sample the centre row, trim white borders, count non-grey pixels."""
    h, w = bar_bgr.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    row = bar_bgr[h // 2].astype(np.int16)
    is_white = np.all(row >= 255, axis=1)
    inner = np.where(~is_white)[0]
    if inner.size == 0:
        return 0.0
    lb, rb = int(inner[0]), int(inner[-1])
    seg = row[lb:rb + 1]
    b, g, r = seg[:, 0], seg[:, 1], seg[:, 2]
    grey = (np.abs(r - g) <= grey_tol) & (np.abs(r - b) <= grey_tol) & (r > 0)
    total = seg.shape[0]
    return 100.0 * (total - int(grey.sum())) / total


def read_bars(ui_bgr: np.ndarray, cfg_health: dict) -> tuple[float | None, float | None, float | None]:
    """(hp%, mp%, exp%) from fixed rects if configured, else detected bars. None when unreadable."""
    rects = cfg_health.get("fixed_rects")
    if rects:
        boxes = [Box(*map(int, r)) for r in rects]
        if len(boxes) not in (2, 3):
            return (None, None, None)
    else:
        boxes = find_status_bars(ui_bgr, int(cfg_health.get("bar_white_thresh", 240)))
        if len(boxes) != 3:  # a missed bar would shift HP/MP/EXP by position
            return (None, None, None)
    vals: list[float | None] = []
    for b in boxes:
        crop = ui_bgr[b.y:b.y2, b.x:b.x2]
        vals.append(bar_fill_percent(crop) if crop.size else None)
    while len(vals) < 3:
        vals.append(None)
    return (vals[0], vals[1], vals[2])
