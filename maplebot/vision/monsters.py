"""Monster template library and detection by masked template matching."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from maplebot.types import Box, Detection
from maplebot.vision.template import make_mask, nms


@dataclass(frozen=True)
class MonsterTemplate:
    name: str
    img_bgr: np.ndarray
    mask: np.ndarray
    scale: float

    @property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2GRAY)


@dataclass
class MonsterLibrary:
    templates: list[MonsterTemplate] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    _gray_cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def min_area(self) -> int:
        if not self.templates:
            return 0
        base = [t for t in self.templates if t.scale == 1.0] or self.templates
        return min(t.img_bgr.shape[0] * t.img_bgr.shape[1] for t in base)

    def gray_of(self, t: MonsterTemplate) -> np.ndarray:
        key = id(t)
        if key not in self._gray_cache:
            self._gray_cache[key] = t.gray
        return self._gray_cache[key]

    @classmethod
    def load(cls, root: str, names: list[str], scales_by_name: dict[str, list[float]],
             default_scales: list[float]) -> "MonsterLibrary":
        """Load `root/<name>/*.png` for each name (all folders when names is empty).

        Every image is keyed (green or alpha), horizontally flipped, and pre-resized per scale.
        """
        if not names:
            names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))) \
                if os.path.isdir(root) else []
        lib = cls(names=list(names))
        for name in names:
            folder = os.path.join(root, name)
            files = sorted(glob.glob(os.path.join(folder, "*.png")))
            if not files:
                raise FileNotFoundError(f"no PNG templates in {folder}")
            scales = [float(s) for s in (scales_by_name.get(name) or default_scales or [1.0])]
            for f in files:
                raw = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    raise ValueError(f"cannot read {f}")
                bgr, mask = make_mask(raw)
                for img, m in ((bgr, mask), (cv2.flip(bgr, 1), cv2.flip(mask, 1))):
                    for s in scales:
                        if s == 1.0:
                            ri, rm = img, m
                        else:
                            w = max(8, int(round(img.shape[1] * s)))
                            h = max(8, int(round(img.shape[0] * s)))
                            ri = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
                            rm = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                        lib.templates.append(MonsterTemplate(name, np.ascontiguousarray(ri),
                                                             np.ascontiguousarray(rm), s))
        return lib


def color_spread(patch: np.ndarray, mask: np.ndarray | None) -> float:
    """Mean (max channel - min channel) over masked pixels: 0 for grey, high for colourful."""
    if patch is None or patch.size == 0:
        return 0.0
    if mask is not None:
        sel = mask[: patch.shape[0], : patch.shape[1]] > 0
        if not np.any(sel):
            return 0.0
        px = patch[sel].astype(np.int16)
    else:
        px = patch.reshape(-1, 3).astype(np.int16)
    return float((px.max(axis=1) - px.min(axis=1)).mean())


def _per_monster(cfg: dict, name: str, key: str, default):
    pm = (cfg.get("per_monster") or {}).get(name) or {}
    return pm.get(key, cfg.get(key, default))


def detect_monsters(roi_bgr: np.ndarray, lib: MonsterLibrary, cfg: dict,
                    exclude: Box | None, roi_origin: tuple[int, int] = (0, 0)) -> list[Detection]:
    """Find monsters in `roi_bgr`. Boxes are returned in frame coordinates (roi_origin added).

    `exclude` is in frame coordinates (the player's own footprint) and is blanked before matching.
    """
    if roi_bgr is None or roi_bgr.size == 0 or not lib.templates:
        return []
    ox, oy = roi_origin
    roi = roi_bgr
    if exclude is not None:
        e = exclude.shifted(-ox, -oy).clamp(roi_bgr.shape[1], roi_bgr.shape[0])
        if e.area > 0:
            roi = roi_bgr.copy()
            roi[e.y:e.y2, e.x:e.x2] = 0
    mode = cfg.get("mode", "color")
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if mode == "grayscale" else None
    max_matches = int(cfg.get("max_matches_per_template", 50))
    rh, rw = roi.shape[:2]

    dets: list[Detection] = []
    for t in lib.templates:
        th, tw = t.img_bgr.shape[:2]
        if th > rh or tw > rw:
            continue
        thres = float(_per_monster(cfg, t.name, "diff_thres", 0.25))
        min_spread = float(_per_monster(cfg, t.name, "min_color_spread", 0))
        if mode == "grayscale":
            res = cv2.matchTemplate(roi_gray, lib.gray_of(t), cv2.TM_SQDIFF_NORMED, mask=t.mask)
        else:
            res = cv2.matchTemplate(roi, t.img_bgr, cv2.TM_SQDIFF_NORMED, mask=t.mask)
        res = np.nan_to_num(res, nan=1.0, posinf=1.0, neginf=1.0)
        ys, xs = np.where(res <= thres)
        if len(xs) == 0:
            continue
        order = np.argsort(res[ys, xs])[:max_matches]
        for i in order:
            x, y = int(xs[i]), int(ys[i])
            if min_spread > 0 and color_spread(roi[y:y + th, x:x + tw], t.mask) < min_spread:
                continue
            dets.append(Detection(t.name, Box(x + ox, y + oy, tw, th), float(res[y, x])))
    return nms(dets, iou_thres=0.4)


def detect_hp_bars(roi_bgr: np.ndarray, color_bgr, tolerance: int, guess_w: int = 70,
                   guess_h: int = 50, roi_origin: tuple[int, int] = (0, 0)) -> list[Detection]:
    """Treat each blob of the enemy HP-bar colour as a monster whose body hangs below the bar."""
    if roi_bgr is None or roi_bgr.size == 0:
        return []
    c = np.array(color_bgr, np.int16)
    mask = cv2.inRange(roi_bgr, np.clip(c - tolerance, 0, 255).astype(np.uint8),
                       np.clip(c + tolerance, 0, 255).astype(np.uint8))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    ox, oy = roi_origin
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 3 or h > 6 or w < 4:
            continue
        out.append(Detection("hp_bar", Box(int(x) + ox, int(y) + oy + 10, guess_w, guess_h), 1.0))
    return out
