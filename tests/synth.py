"""Synthetic frame builders shared by the vision tests."""
from __future__ import annotations

import numpy as np

WHITE = (255, 255, 255)


def make_frame(w: int = 1282, h: int = 693, color=(40, 40, 40)) -> np.ndarray:
    frame = np.empty((h, w, 3), np.uint8)
    frame[:] = color
    return frame


def paste_minimap(frame: np.ndarray, x: int, y: int, w: int, h: int,
                  dot: tuple[int, int] | None = None, dot_color=(136, 255, 255),
                  fill=(20, 20, 60), extra_dots: list[tuple[tuple[int, int], tuple]] | None = None) -> None:
    """Draw a white 1px bordered minimap. `dot` is relative to the inner content (x+1, y+1)."""
    frame[y:y + h, x:x + w] = WHITE
    frame[y + 1:y + h - 1, x + 1:x + w - 1] = fill
    dots = [] if dot is None else [(dot, dot_color)]
    dots += extra_dots or []
    for (dx, dy), c in dots:
        cx, cy = x + 1 + dx, y + 1 + dy
        frame[cy - 1:cy + 2, cx - 1:cx + 2] = c


def paste_sprite(frame: np.ndarray, sprite_bgr: np.ndarray, mask: np.ndarray, x: int, y: int) -> None:
    h, w = sprite_bgr.shape[:2]
    roi = frame[y:y + h, x:x + w]
    sel = mask > 0
    roi[sel] = sprite_bgr[sel]


def make_sprite(w: int = 40, h: int = 30, seed: int = 3, key=(0, 255, 0)) -> np.ndarray:
    """A colourful blob on a pure-green background, BGR."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = key
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((xx - w / 2) ** 2 / (w / 2) ** 2 + (yy - h / 2) ** 2 / (h / 2) ** 2) <= 1.0
    colours = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    colours[:, :, 1] = np.minimum(colours[:, :, 1], 200)  # never equal to the key
    img[inside] = colours[inside]
    return img
