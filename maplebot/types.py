"""Core value types shared by vision, navigation, brain and control."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

MOVE_X = ("none", "left", "right", "stop")
MOVE_Y = ("none", "up", "down", "stop")
ACTIONS = ("none", "jump", "attack", "aoe", "teleport", "goal")


@dataclass(frozen=True)
class Command:
    """One decision per frame: horizontal hold, vertical hold, one-shot action."""

    move_x: str = "none"
    move_y: str = "none"
    action: str = "none"

    NONE: ClassVar["Command"]

    def __post_init__(self) -> None:
        if self.move_x not in MOVE_X:
            raise ValueError(f"bad move_x {self.move_x!r}")
        if self.move_y not in MOVE_Y:
            raise ValueError(f"bad move_y {self.move_y!r}")
        if self.action not in ACTIONS:
            raise ValueError(f"bad action {self.action!r}")

    @classmethod
    def parse(cls, text: str) -> "Command":
        parts = text.split()
        if len(parts) != 3:
            raise ValueError(f"command needs 3 tokens, got {text!r}")
        return cls(*parts)

    def replace(self, **kw) -> "Command":
        d = {"move_x": self.move_x, "move_y": self.move_y, "action": self.action}
        d.update(kw)
        return Command(**d)

    def __str__(self) -> str:
        return f"{self.move_x} {self.move_y} {self.action}"


Command.NONE = Command()


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def intersection_area(self, other: "Box") -> int:
        iw = min(self.x2, other.x2) - max(self.x, other.x)
        ih = min(self.y2, other.y2) - max(self.y, other.y)
        if iw <= 0 or ih <= 0:
            return 0
        return iw * ih

    def iou(self, other: "Box") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, pt: tuple[int, int]) -> bool:
        return self.x <= pt[0] < self.x2 and self.y <= pt[1] < self.y2

    def clamp(self, w: int, h: int) -> "Box":
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(w, self.x2)
        y1 = min(h, self.y2)
        return Box(x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def shifted(self, dx: int, dy: int) -> "Box":
        return Box(self.x + dx, self.y + dy, self.w, self.h)


@dataclass(frozen=True)
class Detection:
    name: str
    box: Box
    score: float


@dataclass(frozen=True)
class PlayerFix:
    x: int
    y: int
    score: float = 0.0
    source: str = ""

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)
