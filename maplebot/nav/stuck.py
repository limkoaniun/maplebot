"""Detect a player that has not moved for too long."""
from __future__ import annotations


class StuckDetector:
    def __init__(self, range_px: int = 10, timeout_s: float = 6.0):
        self.range_px = int(range_px)
        self.timeout_s = float(timeout_s)
        self._anchor: tuple[float, float] | None = None
        self._t: float = 0.0

    def reset(self) -> None:
        self._anchor = None

    def update(self, pos: tuple | None, now: float) -> bool:
        """True exactly once each time the player stays within range_px for longer than timeout_s."""
        if pos is None:
            self._anchor = None
            return False
        if self._anchor is None:
            self._anchor, self._t = (pos[0], pos[1]), now
            return False
        moved = abs(pos[0] - self._anchor[0]) + abs(pos[1] - self._anchor[1])
        if moved > self.range_px:
            self._anchor, self._t = (pos[0], pos[1]), now
            return False
        if now - self._t > self.timeout_s:
            self._anchor, self._t = (pos[0], pos[1]), now
            return True
        return False
