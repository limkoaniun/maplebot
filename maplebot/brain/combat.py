"""Attack ranges, target selection and attack cooldowns."""
from __future__ import annotations

from maplebot.types import Box, Detection


def attack_boxes(player: tuple[int, int], cfg: dict, frame_w: int, frame_h: int) -> dict[str, Box]:
    px, py = player
    if cfg.get("mode", "directional") == "aoe":
        rx, ry = int(cfg["aoe_range_x"]), int(cfg["aoe_range_y"])
        return {"aoe": Box(px - rx // 2, py - ry // 2, rx, ry).clamp(frame_w, frame_h)}
    rx, ry = int(cfg["range_x"]), int(cfg["range_y"])
    y0 = py - ry // 2
    return {
        "left": Box(px - rx, y0, rx, ry).clamp(frame_w, frame_h),
        "right": Box(px, y0, rx, ry).clamp(frame_w, frame_h),
    }


def search_box(player: tuple[int, int], cfg: dict, frame_w: int, frame_h: int) -> Box:
    """Union of the attack boxes grown by `search_margin`: where monsters are looked for."""
    boxes = list(attack_boxes(player, cfg, frame_w, frame_h).values())
    m = int(cfg.get("search_margin", 50))
    x0 = min(b.x for b in boxes) - m
    y0 = min(b.y for b in boxes) - m
    x1 = max(b.x2 for b in boxes) + m
    y1 = max(b.y2 for b in boxes) + m
    return Box(x0, y0, x1 - x0, y1 - y0).clamp(frame_w, frame_h)


def _distance(det: Detection, player: tuple[int, int]) -> int:
    cx, cy = det.box.center
    return abs(cx - player[0]) + abs(cy - player[1])


def choose_target(dets: list[Detection], player: tuple[int, int], cfg: dict, min_area: int,
                  frame_w: int, frame_h: int, prefer: str | None = None) -> str | None:
    """Return "left" / "right" / "aoe" for the side worth attacking this frame, or None.

    A detection counts when its overlap with an attack box is at least
    min(min_area, min_overlap_px) and its centre lies on that box's side of the player.
    When both sides qualify and their nearest targets are within tie_margin_px of each
    other, the previous side (`prefer`) is kept if it still qualifies, otherwise no
    direction is returned, so the character does not flip back and forth.
    """
    if not dets:
        return None
    thres = min(int(min_area), int(cfg.get("min_overlap_px", 1500))) if min_area > 0 \
        else int(cfg.get("min_overlap_px", 1500))
    boxes = attack_boxes(player, cfg, frame_w, frame_h)
    if "aoe" in boxes:
        box = boxes["aoe"]
        return "aoe" if any(box.intersection_area(d.box) >= thres for d in dets) else None

    nearest: dict[str, int] = {}
    for side, box in boxes.items():
        for d in dets:
            if box.intersection_area(d.box) < thres:
                continue
            cx = d.box.center[0]
            if side == "left" and cx >= player[0]:
                continue
            if side == "right" and cx <= player[0]:
                continue
            dist = _distance(d, player)
            if side not in nearest or dist < nearest[side]:
                nearest[side] = dist
    if not nearest:
        return None
    if len(nearest) == 1:
        return next(iter(nearest))
    tie = int(cfg.get("tie_margin_px", 50))
    dl, dr = nearest["left"], nearest["right"]
    if dl < dr - tie:
        return "left"
    if dr < dl - tie:
        return "right"
    return prefer if prefer in nearest else None


class AttackTimer:
    def __init__(self, cfg: dict):
        self.cooldowns = {
            "attack": float(cfg.get("cooldown_s", 0.9)),
            "aoe": float(cfg.get("aoe_cooldown_s", 1.0)),
        }
        self._last: dict[str, float | None] = {k: None for k in self.cooldowns}

    def ready(self, kind: str, now: float) -> bool:
        last = self._last.get(kind)
        return last is None or now - last >= self.cooldowns.get(kind, 0.0)

    def mark(self, kind: str, now: float) -> None:
        self._last[kind] = now
