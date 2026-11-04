"""Left/right sweeping used to find monsters when no route is loaded."""
from __future__ import annotations

from maplebot.types import Command


class SweepSearcher:
    def __init__(self, cfg: dict, minimap_width_bound_pct: float = 0.0):
        self.enable = bool(cfg.get("enable", True))
        self.trigger_frames = max(0, int(cfg.get("trigger_frames", 4)))
        self.flip_frames = max(1, int(cfg.get("flip_frames", 40)))
        self.max_offset_px = float(cfg.get("max_offset_px", 0) or 0)
        self.stall_frames = int(cfg.get("stall_frames", 0) or 0)
        self.attack_while_moving = bool(cfg.get("attack_while_moving", True))
        self.bound_pct = float(minimap_width_bound_pct or 0.0)
        self._dir = "left"
        self.reset()

    def reset(self) -> None:
        self._no_monster = 0
        self._flip_at: int | None = None
        self._anchor_x: float | None = None
        self._last_x: float | None = None
        self._stall = 0

    @property
    def direction(self) -> str:
        return self._dir

    def _flip(self, frame_idx: int, progress_x: float | None) -> None:
        self._dir = "right" if self._dir == "left" else "left"
        self._flip_at = frame_idx + self.flip_frames
        self._stall = 0
        self._anchor_x = self._last_x = progress_x

    def update(self, no_monster: bool, progress_x: float | None, minimap_x: float | None,
               minimap_w: int | None, frame_idx: int) -> Command:
        if not no_monster:
            self.reset()
            return Command.NONE
        if not self.enable:
            return Command.NONE
        self._no_monster += 1
        if self._no_monster <= self.trigger_frames:
            return Command.NONE

        if self._flip_at is None:
            self._flip_at = frame_idx + self.flip_frames
            self._anchor_x = self._last_x = progress_x
        else:
            reason = None
            if self.stall_frames > 0 and progress_x is not None and self._last_x is not None:
                self._stall = self._stall + 1 if abs(progress_x - self._last_x) <= 1.0 else 0
                self._last_x = progress_x
                if self._stall >= self.stall_frames:
                    reason = "stall"
            if reason is None and self.max_offset_px > 0 and progress_x is not None and self._anchor_x is not None:
                off = progress_x - self._anchor_x
                if (self._dir == "left" and off <= -self.max_offset_px) or \
                        (self._dir == "right" and off >= self.max_offset_px):
                    reason = "bound"
            if reason is None and frame_idx >= self._flip_at:
                reason = "timer"
            if reason is not None:
                self._flip(frame_idx, progress_x)

        if 0.0 < self.bound_pct < 1.0 and minimap_x is not None and minimap_w:
            margin = minimap_w * (1.0 - self.bound_pct) / 2.0
            if minimap_x <= margin and self._dir == "left":
                self._flip(frame_idx, progress_x)
            elif minimap_x >= minimap_w - margin and self._dir == "right":
                self._flip(frame_idx, progress_x)

        return Command(self._dir, "none", "attack" if self.attack_while_moving else "none")
