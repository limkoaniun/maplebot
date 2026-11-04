"""Per-frame decision: route or sweep, combat override, stuck recovery."""
from __future__ import annotations

from dataclasses import dataclass, field

from maplebot.brain.combat import choose_target
from maplebot.brain.search import SweepSearcher
from maplebot.nav.stuck import StuckDetector
from maplebot.types import Command, Detection


@dataclass
class Perception:
    frame_w: int
    frame_h: int
    player: tuple[int, int] | None
    global_pos: tuple[int, int] | None
    minimap_pos: tuple[int, int] | None
    minimap_w: int | None
    on_ladder: bool
    detections: list[Detection] = field(default_factory=list)
    now: float = 0.0
    frame_idx: int = 0


class Brain:
    def __init__(self, cfg: dict, route_follower, min_monster_area: int):
        self.cfg = cfg
        self.route = route_follower
        self.min_area = int(min_monster_area)
        self.attack_cfg = cfg["attack"]
        self.aoe_key = str(cfg["keys"].get("aoe") or "")
        stuck_cfg = cfg.get("stuck", {})
        self.stuck_enabled = bool(stuck_cfg.get("enable", True))
        self.stuck = StuckDetector(int(stuck_cfg.get("range_px", 10)), float(stuck_cfg.get("timeout_s", 6.0)))
        self.recover_s = float(stuck_cfg.get("recover_s", 2.0))
        self.search = SweepSearcher(cfg.get("search", {}), float(cfg.get("minimap", {}).get("width_bound_pct", 0.0)))
        self.state = "hunting"
        self.target: str | None = None
        self._last_move_x = "left"
        self._recover_until = 0.0
        self._recover_dir = "right"
        self._recover_jumped = False

    # -- helpers -------------------------------------------------------
    def _choose_target(self, p: Perception) -> str | None:
        if p.player is None or not p.detections:
            self.target = None
            return None
        self.target = choose_target(p.detections, p.player, self.attack_cfg, self.min_area,
                                    p.frame_w, p.frame_h, prefer=self.target)
        return self.target

    def _base_command(self, p: Perception, target: str | None) -> Command:
        if self.route is not None:
            return self.route.step(p.global_pos, p.on_ladder)
        progress = None
        if p.global_pos is not None:
            progress = float(p.global_pos[0])
        elif p.minimap_pos is not None:
            progress = float(p.minimap_pos[0])
        mm_x = float(p.minimap_pos[0]) if p.minimap_pos is not None else None
        # keep sweeping while nothing is actually attackable (a mob only in the search margin,
        # or a tie) instead of freezing in place
        return self.search.update(target is None, progress, mm_x, p.minimap_w, p.frame_idx)

    def _combat(self, p: Perception, cmd: Command, target: str | None) -> Command:
        if target is None:
            return cmd
        crowd = len(p.detections) >= int(self.attack_cfg.get("aoe_monster_count", 3))
        action = "aoe" if self.aoe_key and (target == "aoe" or crowd) else "attack"
        move_x = cmd.move_x
        if target in ("left", "right"):
            move_x = target
            deadzone = int(self.attack_cfg.get("deadzone_px", 0))
            px = p.player[0]
            nearest_dx = min(abs(d.box.center[0] - px) for d in p.detections)
            if nearest_dx <= deadzone:
                move_x = "none"
        move_y = cmd.move_y if p.on_ladder else "none"
        return Command(move_x, move_y, action)

    def _recovering(self, p: Perception) -> Command:
        action = "none"
        if not self._recover_jumped:
            self._recover_jumped = True
            action = "jump"
        return Command(self._recover_dir, "none", action)

    # -- main entry ----------------------------------------------------
    def decide(self, p: Perception) -> Command:
        target = self._choose_target(p)
        pos = p.global_pos if p.global_pos is not None else p.minimap_pos
        stuck_now = self.stuck.update(pos, p.now) if self.stuck_enabled else False

        if self.state == "recovering":
            if p.now < self._recover_until:
                return self._recovering(p)
            self.state = "hunting"
            self.stuck.reset()

        if stuck_now and target is None:
            self.state = "recovering"
            self._recover_until = p.now + self.recover_s
            self._recover_dir = "right" if self._last_move_x == "left" else "left"
            self._recover_jumped = False
            return self._recovering(p)

        cmd = self._base_command(p, target)
        cmd = self._combat(p, cmd, target)
        if cmd.move_x in ("left", "right"):
            self._last_move_x = cmd.move_x
        return cmd
