"""On-screen gain evidence: the EXP numeric text and the gain notification lines.

Pure pixel functions; nothing here reads text. They answer "did the EXP number
change" and "did a notification line arrive, and does it look like an EXP or a
meso line". The rects are layout-specific (see `evidence:` in config/default.yaml)
and every result is an observation of the HUD, not a kill or item count.
"""
from __future__ import annotations

import cv2
import numpy as np

_K3 = np.ones((3, 3), np.uint8)
_K5 = np.ones((5, 5), np.uint8)


def crop_rect(frame: np.ndarray | None, rect) -> np.ndarray | None:
    """`frame[y:y+h, x:x+w]`, or None unless the whole rect lies inside a BGR frame."""
    try:
        x, y, w, h = (int(v) for v in rect)
    except (TypeError, ValueError):
        return None
    if frame is None or getattr(frame, "ndim", 0) != 3 or frame.shape[2] != 3:
        return None
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return None
    return frame[y:y + h, x:x + w]


# -- EXP numeric text ---------------------------------------------------------
def changed_pixel_count(prev_crop: np.ndarray, crop: np.ndarray, pixel_delta: int = 24) -> int | None:
    """Pixels whose largest channel difference exceeds `pixel_delta`; None when not comparable."""
    if prev_crop is None or crop is None or prev_crop.shape != crop.shape or crop.size == 0:
        return None
    diff = cv2.absdiff(prev_crop, crop)
    return int(((diff.max(axis=2) if diff.ndim == 3 else diff) > int(pixel_delta)).sum())


def exp_text_changed(prev_crop: np.ndarray, crop: np.ndarray, pixel_delta: int = 24,
                     min_pixels: int = 10) -> bool | None:
    """True when the EXP text crop differs from the previous one. None when not comparable.

    The HUD text is redrawn without noise: an unchanged value gives an identical crop.
    """
    count = changed_pixel_count(prev_crop, crop, pixel_delta)
    return None if count is None else count >= int(min_pixels)


def exp_text_plausible(crop: np.ndarray, dark_fraction: tuple[float, float] = (0.45, 0.80),
                       bright_fraction: tuple[float, float] = (0.04, 0.20)) -> bool:
    """Light text on the dark status panel. Rejects scenery or a window covering the rect."""
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return False
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark, bright = float((grey < 90).mean()), float((grey >= 170).mean())
    return dark_fraction[0] <= dark <= dark_fraction[1] and bright_fraction[0] <= bright <= bright_fraction[1]


# -- gain notification lines --------------------------------------------------
def gain_line_mask(region_img: np.ndarray, white_min: int = 200, max_spread: int = 30,
                   outline_max: int = 70) -> np.ndarray:
    """Near-white, unsaturated pixels that touch a near-black outline pixel (3x3)."""
    if region_img is None or region_img.size == 0 or region_img.ndim != 3:
        return np.zeros((0, 0), bool)
    grey = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
    spread = region_img.max(axis=2).astype(np.int16) - region_img.min(axis=2)
    outline = cv2.dilate((grey < int(outline_max)).astype(np.uint8), _K3)
    return (grey >= int(white_min)) & (spread < int(max_spread)) & (outline > 0)


def gain_line_pixel_count(region_img: np.ndarray, **kw) -> int:
    """Outlined-white pixel count of one line slot: ~0 when empty, tens to hundreds for a line."""
    return int(gain_line_mask(region_img, **kw).sum())


def gain_line_like(mask: np.ndarray, right_gap: int = 24, edge_fraction: float = 0.15) -> bool:
    """A notification line is right-aligned and leaves the slot's top and bottom two rows clear.

    Bright outlined scenery passes the pixel count about four times as often without this gate.
    """
    total = int(mask.sum()) if mask.ndim == 2 and mask.shape[0] >= 5 else 0
    if total == 0:
        return False
    last = int(np.flatnonzero(mask.any(axis=0))[-1])
    return (mask.shape[1] - 1 - last <= right_gap
            and int(mask[:2].sum() + mask[-2:].sum()) <= edge_fraction * total)


def line_detail(img: np.ndarray) -> np.ndarray:
    """5x5 top-hat of the grey image: keeps thin bright strokes, drops broad scenery."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, _K5).astype(np.float32)


def line_similarity(line_img: np.ndarray, other_img: np.ndarray) -> float:
    """Correlation of `line_img`'s text with `other_img` (same-size slots), tolerant of fading.

    Compared only over the columns the line's own outlined-white pixels occupy, with one
    pixel of vertical tolerance. 0.0 when `line_img` holds no line or the shapes differ.
    """
    if line_img is None or other_img is None or line_img.shape != other_img.shape or line_img.shape[0] < 5:
        return 0.0
    cols = np.flatnonzero(gain_line_mask(line_img).any(axis=0))
    if cols.size == 0:
        return 0.0
    x0 = max(0, int(cols[0]) - 2)
    if line_img.shape[1] - x0 < 8:
        return 0.0
    a, b = line_detail(line_img)[1:-1, x0:], line_detail(other_img)[:, x0:]
    if float(a.std()) < 1e-3 or float(b.std()) < 1e-3:
        return 0.0
    score = float(cv2.matchTemplate(b, a, cv2.TM_CCOEFF_NORMED).max())
    return score if np.isfinite(score) else 0.0


def gain_line_arrivals(prev_slots, slots, *, fresh_min: int = 40, present_max: int = 15,
                       shift_min: float = 0.45, shift_margin: float = 0.15) -> int:
    """How many new lines demonstrably arrived between two consecutive captures (0..len-1).

    `slots[0]` is the newest (bottom) line slot, `slots[1]` the one above, and so on;
    all are same-size BGR crops one line pitch apart. Conservative by design:

    * bottom slot was empty (< `present_max` pixels) and now holds a fresh line
      (>= `fresh_min` pixels, shaped like a line) that does not match what the slot showed
      before (similarity < `shift_min`; a fading line can dip under the pixel count on dark
      scenery and return on brighter scenery): one arrival, plus one per further slot that
      filled the same way. Matching text still counts when the old line is now found one
      slot up (>= `shift_min`, and `shift_margin` better than one capture earlier);
    * bottom slot already held a line: an arrival only when that line is now found one
      (or more) slots up (similarity >= `shift_min`), clearly better than the same
      comparison one capture earlier (`shift_margin`), with fresh lines below it.

    Anything else is 0, so an unexplained change is never counted.
    """
    if not slots or not prev_slots or len(slots) != len(prev_slots):
        return 0
    if any(s is None or p is None or s.shape != p.shape for s, p in zip(slots, prev_slots)):
        return 0
    masks = [gain_line_mask(s) for s in slots]
    fresh = [int(m.sum()) >= fresh_min and gain_line_like(m) for m in masks]
    if not fresh[0]:
        return 0
    previous = [gain_line_pixel_count(p) for p in prev_slots]
    if previous[0] < present_max:
        # "Empty" by pixel count is not proof: a half-faded line drops under the count over dark
        # scenery and comes back over brighter scenery. The same text in the same slot is not new.
        if line_similarity(slots[0], prev_slots[0]) >= shift_min:
            # ...unless that old line is now demonstrably one slot up (a repeat of the same text).
            above = line_similarity(slots[0], slots[1])
            if above < shift_min or above - line_similarity(slots[0], prev_slots[1]) < shift_margin:
                return 0
        arrived = 1
        while (arrived < len(slots) - 1 and fresh[arrived] and previous[arrived] < present_max
               and line_similarity(slots[arrived], prev_slots[arrived]) < shift_min):
            arrived += 1
        return arrived
    for k in range(1, len(slots)):
        if not fresh[k - 1]:
            return 0
        now, before = line_similarity(prev_slots[0], slots[k]), line_similarity(prev_slots[0], prev_slots[k])
        if now >= shift_min and now - before >= shift_margin:
            return k
    return 0


def load_gain_templates(paths: dict) -> dict[str, np.ndarray]:
    """{'exp': path, 'meso': path} -> {kind: top-hat template}; unreadable files are skipped."""
    out: dict[str, np.ndarray] = {}
    for kind, path in (paths or {}).items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
        if img is not None and img.size:
            out[str(kind)] = line_detail(img)
    return out


def gain_line_scores(region_img: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, float]:
    """Best TM_CCOEFF_NORMED of each prefix template anywhere in the slot (0.0 if it cannot fit)."""
    scores: dict[str, float] = {}
    if region_img is None or region_img.size == 0 or region_img.ndim != 3:
        return {kind: 0.0 for kind in templates}
    detail = line_detail(region_img)
    for kind, tmpl in templates.items():
        score = 0.0
        if (tmpl.shape[0] <= detail.shape[0] and tmpl.shape[1] <= detail.shape[1]
                and float(tmpl.std()) > 1e-3 and float(detail.std()) > 1e-3):
            score = float(cv2.matchTemplate(detail, tmpl, cv2.TM_CCOEFF_NORMED).max())
        scores[kind] = score if np.isfinite(score) else 0.0
    return scores


def classify_gain_line(region_img: np.ndarray, templates: dict[str, np.ndarray],
                       thresholds: dict | None = None) -> str:
    """'exp' | 'meso' | 'other' for a slot that holds a fresh line.

    'other' is everything that matches neither prefix: an item line, another system
    message, or an EXP/meso line too washed out by a bright background to match.
    """
    limits = {"exp": 0.5, "meso": 0.45, **(thresholds or {})}
    best, margin = "other", 0.0
    for kind, score in gain_line_scores(region_img, templates).items():
        over = score - float(limits.get(kind, 1.0))
        if kind in ("exp", "meso") and over >= margin:
            best, margin = kind, over
    return best
