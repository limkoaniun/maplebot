"""Masked template matching helpers built on cv2.matchTemplate."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from maplebot.types import Detection

HIGHER_IS_BETTER = {cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCOEFF}


@dataclass(frozen=True)
class MatchResult:
    loc: tuple[int, int]
    score: float
    local: bool


def make_mask(img: np.ndarray, key_bgr: tuple[int, int, int] = (0, 255, 0)) -> tuple[np.ndarray, np.ndarray]:
    """Split a template into (bgr, mask). Transparent = alpha < 128 or pixels equal to key_bgr."""
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = np.ascontiguousarray(img[:, :, :3])
        mask = np.where(img[:, :, 3] >= 128, 255, 0).astype(np.uint8)
    else:
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        mask = np.full(bgr.shape[:2], 255, np.uint8)
    keyed = np.all(bgr == np.array(key_bgr, np.uint8), axis=2)
    mask[keyed] = 0
    return bgr, mask


def _pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih >= h and iw >= w:
        return img
    return cv2.copyMakeBorder(img, 0, max(0, h - ih), 0, max(0, w - iw), cv2.BORDER_REPLICATE)


def _best(res: np.ndarray, method: int) -> tuple[tuple[int, int], float]:
    res = np.nan_to_num(res, nan=1.0 if method not in HIGHER_IS_BETTER else 0.0, posinf=1.0, neginf=0.0)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in HIGHER_IS_BETTER:
        return (int(max_loc[0]), int(max_loc[1])), float(max_val)
    return (int(min_loc[0]), int(min_loc[1])), float(min_val)


def match_template(
    img: np.ndarray,
    tmpl: np.ndarray,
    mask: np.ndarray | None = None,
    last_loc: tuple[int, int] | None = None,
    local_radius: int = 50,
    local_thres: float = 0.4,
    method: int = cv2.TM_SQDIFF_NORMED,
) -> MatchResult:
    """Find the best location of `tmpl` in `img`.

    Searches around `last_loc` first and accepts that result when it is good enough
    (score < local_thres for SQDIFF, score > 1 - local_thres for CCORR/CCOEFF).
    `img` is padded if it is smaller than the template so this never raises.
    """
    th, tw = tmpl.shape[:2]
    img = _pad_to(img, th, tw)
    ih, iw = img.shape[:2]
    higher = method in HIGHER_IS_BETTER

    if last_loc is not None and local_radius > 0:
        lx, ly = last_loc
        x0 = max(0, lx - local_radius)
        y0 = max(0, ly - local_radius)
        x1 = min(iw, lx + local_radius + tw)
        y1 = min(ih, ly + local_radius + th)
        if x1 - x0 >= tw and y1 - y0 >= th:
            res = cv2.matchTemplate(img[y0:y1, x0:x1], tmpl, method, mask=mask)
            loc, score = _best(res, method)
            ok = score > 1.0 - local_thres if higher else score < local_thres
            if ok:
                return MatchResult((loc[0] + x0, loc[1] + y0), score, True)

    res = cv2.matchTemplate(img, tmpl, method, mask=mask)
    loc, score = _best(res, method)
    return MatchResult(loc, score, False)


def nms(dets: list[Detection], iou_thres: float = 0.4, higher_is_better: bool = False) -> list[Detection]:
    """Greedy non-maximum suppression; keeps the best-scoring box of each overlapping cluster."""
    order = sorted(dets, key=lambda d: d.score, reverse=higher_is_better)
    kept: list[Detection] = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [d for d in order if best.box.iou(d.box) < iou_thres]
    return kept
