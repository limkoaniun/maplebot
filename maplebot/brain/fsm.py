"""Per-frame decision: route or sweep, combat override, stuck recovery."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from maplebot.brain.combat import attack_boxes, choose_target
from maplebot.brain.search import SweepSearcher
from maplebot.nav.stuck import StuckDetector
from maplebot.nav.circuit import MinimapCircuit
from maplebot.nav.loot import LootCollector
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
        circuit_cfg = cfg.get("route", {}).get("circuit", {})
        self.circuit = MinimapCircuit(circuit_cfg) if circuit_cfg.get("enabled") else None
        self.movement_test = bool(circuit_cfg.get("movement_test", False))
        self.loot = LootCollector(cfg.get("loot", {}), circuit_cfg.get("pixels_per_minimap_x", 10))
        self.loot.enabled = self.loot.enabled and bool(str(cfg["keys"].get("pickup") or "").strip())
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
        self._combat_memory = None
        self._selected_detection = None
        self._raw_detections: list[Detection] = []
        self._reacquire_s = 0.4
        self._pursuit_s = float(self.attack_cfg.get("pursuit_s", 1.2))
        if not math.isfinite(self._pursuit_s) or not 0 <= self._pursuit_s <= 3:
            raise ValueError("attack.pursuit_s must be between 0 and 3 seconds")
        self._pursuit_key = self._pursuit_started = self._pursuit_seen = None
        # Opt-in: fight a fresh in-range monster during the pre-takeoff part of a
        # transition, for at most this much active time per waypoint selection.
        pre = self.attack_cfg.get("pre_transition_s", 0)
        if (not isinstance(pre, (int, float)) or isinstance(pre, bool)
                or not math.isfinite(pre) or not 0 <= pre <= 10):
            raise ValueError("attack.pre_transition_s must be between 0 and 10 seconds")
        self._pre_transition_s = float(pre)
        self._pre_key = self._decide_last = None
        self._pre_spent = self._frame_dt = 0.0
        # One pre-takeoff fight: the side it faces, when a qualifying detection
        # was last seen, and whether the previous decided frame belonged to it.
        self._pre_side = self._pre_seen = None
        self._pre_engaged = False
        # Optional same-surface test: a target's sprite bottom must rest on the
        # player's own platform surface (slope-aware), not merely overlap the band.
        surface = self.attack_cfg.get("surface") or {}
        self._surface_enabled = bool(surface.get("enabled", False))
        self._feet_offset = float(surface.get("feet_offset_px", 12))
        self._surface_above = float(surface.get("above_px", 32))
        self._surface_below = float(surface.get("below_px", 22))
        self._scale_y = float(circuit_cfg.get("pixels_per_minimap_y", circuit_cfg.get("pixels_per_minimap_x", 10)))
        values = (self._feet_offset, self._surface_above, self._surface_below, self._scale_y)
        if (not all(math.isfinite(v) for v in values) or min(values[1:]) < 0 or self._scale_y <= 0
                or abs(self._feet_offset) > 200 or max(values[1:3]) > 200):
            raise ValueError("attack.surface offsets and the vertical minimap scale must be finite and bounded")

    # -- helpers -------------------------------------------------------
    def suspend(self, now: float) -> None:
        self._clear_combat_memory()
        self._pursuit_seen = None
        self._end_pre_fight()
        if self.circuit is not None:
            self.circuit.suspend(now)
        self.loot.suspend(now)
        self.stuck.reset()

    def _clear_combat_memory(self) -> None:
        self._combat_memory = None
        self._selected_detection = None
        self.target = None

    def _supported_combat_platform(self, p: Perception):
        if self.circuit is None or p.player is None or p.minimap_pos is None or p.on_ladder:
            return None
        platform = self.circuit.platform_at(p.minimap_pos)
        if (platform is None or not platform.left <= p.minimap_pos[0] <= platform.right
                or platform.support_error(p.minimap_pos) > self.circuit.tolerance_y):
            return None
        return platform

    def _minimap_x(self, p: Perception, det: Detection) -> float:
        return p.minimap_pos[0] + (det.box.center[0] - p.player[0]) / self.loot.scale

    def _surface_offset(self, platform, p: Perception, det: Detection) -> float:
        """Sprite bottom relative to the platform surface under it (+ = lower on screen)."""
        rise = platform.y_at(self._minimap_x(p, det)) - platform.y_at(p.minimap_pos[0])
        return det.box.y2 - (p.player[1] + self._feet_offset + rise * self._scale_y)

    def _on_player_surface(self, platform, p: Perception, det: Detection) -> bool:
        return (not self._surface_enabled
                or -self._surface_above <= self._surface_offset(platform, p, det) <= self._surface_below)

    def _remember_target(self, p: Perception, target: str) -> None:
        boxes = attack_boxes(p.player, self.attack_cfg, p.frame_w, p.frame_h)
        box = boxes.get(target)
        minimum = min(self.min_area, int(self.attack_cfg.get("min_overlap_px", 1500))) if self.min_area > 0 else int(self.attack_cfg.get("min_overlap_px", 1500))
        candidates = [d for d in p.detections if box is not None and box.intersection_area(d.box) >= minimum
                      and (target == "aoe" or (d.box.center[0] < p.player[0]) == (target == "left"))]
        if candidates:
            det = (self._selected_detection if self._selected_detection in candidates else
                   min(candidates, key=lambda d: abs(d.box.center[0] - p.player[0]) + abs(d.box.center[1] - p.player[1])))
            self._selected_detection = det
            supported = self._supported_combat_platform(p)
            if supported is not None:
                self._combat_memory = {"detection": det, "platform": supported, "seen": p.now,
                    "relative": (det.box.center[0] - p.player[0], det.box.center[1] - p.player[1])}
            platform = self.circuit.platform_at(p.minimap_pos) if self.circuit is not None else None
            bounds = (platform.left, platform.right) if platform is not None else None
            visible = ([self._minimap_x(p, d) for d in self._raw_detections]
                       if p.minimap_pos is not None else None)
            self.loot.observe(p.minimap_pos, p.player, det.box.center, p.now, p.minimap_w, bounds,
                              platform=platform, visible_x=visible)

    def _choose_target(self, p: Perception) -> str | None:
        self._selected_detection = None
        if p.player is None or not p.detections:
            self.target = None
            return None
        memory = self._combat_memory
        if memory is not None:
            old = memory["detection"]
            candidates = [d for d in p.detections if d.name == old.name
                and abs(d.box.center[0] - p.player[0] - memory["relative"][0]) <= max(32, old.box.w)
                and abs(d.box.center[1] - p.player[1] - memory["relative"][1]) <= max(24, old.box.h)]
            candidates.sort(key=lambda d: abs(d.box.center[0] - p.player[0] - memory["relative"][0])
                            + abs(d.box.center[1] - p.player[1] - memory["relative"][1]))
            for det in candidates:
                side = choose_target([det], p.player, self.attack_cfg, self.min_area, p.frame_w, p.frame_h)
                if side is not None:
                    self._selected_detection, self.target = det, side
                    return side
        self.target = choose_target(p.detections, p.player, self.attack_cfg, self.min_area,
                                    p.frame_w, p.frame_h, prefer=self.target)
        if self.target is None and self._supported_combat_platform(p) is not None:
            # A fresh tie should start one engagement rather than abandon both monsters.
            for det in sorted(p.detections, key=lambda d: (abs(d.box.center[0] - p.player[0])
                              + abs(d.box.center[1] - p.player[1]), d.box.center)):
                side = choose_target([det], p.player, self.attack_cfg, self.min_area, p.frame_w, p.frame_h)
                if side is not None:
                    self._selected_detection, self.target = det, side
                    break
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

    def _pursue_fresh_target(self, p: Perception) -> Command | None:
        """Briefly approach a visible nearby monster before leaving a walk point."""
        nav = self.circuit
        if (not self._pursuit_s or nav is None or nav.waypoints[nav.idx].action != "walk"
                or self.attack_cfg.get("mode", "directional") != "directional"):
            return None
        platform = self._supported_combat_platform(p)
        if platform is None or platform.support_error(p.minimap_pos) > min(1.0, nav.tolerance_y):
            self._pursuit_seen = None
            return None
        key = (id(nav), nav.idx, platform)
        if key != self._pursuit_key:
            self._pursuit_key = key
            self._pursuit_started = self._pursuit_seen = None
        if self._pursuit_started is not None and not 0 <= p.now - self._pursuit_started < self._pursuit_s:
            return None  # the same waypoint cannot repeatedly restart its pursuit allowance
        expanded = dict(self.attack_cfg)
        expanded["range_x"] = int(expanded["range_x"]) + max(0, int(expanded.get("search_margin", 50)))
        candidates = []
        for det in p.detections:
            dx = det.box.center[0] - p.player[0]
            target_x = p.minimap_pos[0] + dx / self.loot.scale
            # Stay one minimap pixel inside the drawn floor, and retain the
            # ordinary vertical attack band so an adjacent ledge is not chased.
            if (abs(dx) <= expanded["range_x"] and platform.left + 1 <= target_x <= platform.right - 1
                    and choose_target([det], p.player, expanded, self.min_area, p.frame_w, p.frame_h) is not None):
                candidates.append(det)
        if candidates:
            det = min(candidates, key=lambda d: abs(d.box.center[0] - p.player[0])
                      + abs(d.box.center[1] - p.player[1]))
            direction = "left" if det.box.center[0] < p.player[0] else "right"
            if ((direction == "left" and p.minimap_pos[0] <= platform.left + 1)
                    or (direction == "right" and p.minimap_pos[0] >= platform.right - 1)):
                return None
            if self._pursuit_started is None:
                self._pursuit_started = p.now
            self._pursuit_seen = p.now
            self.state = "pursuing_target"
            # A visible living target is not a drop; a visit already begun is kept for later.
            self.loot.yield_to_target(p.now)
            return Command(direction)
        if self._pursuit_seen is not None and 0 <= p.now - self._pursuit_seen < self._reacquire_s:
            self.state = "reacquiring_target"
            return Command.NONE  # no stale movement or attacks during a detection gap
        return None

    def _end_pre_fight(self) -> None:
        self._pre_side = self._pre_seen = None
        self._pre_engaged = False

    def _track_frame(self, p: Perception) -> None:
        """Per-frame bookkeeping for the pre-takeoff combat budget."""
        last, self._decide_last = self._decide_last, p.now
        # A long gap is charged as one second, so slow frames stay bounded too.
        self._frame_dt = 0.0 if last is None else min(1.0, max(0.0, p.now - last))
        nav = self.circuit
        key = None if nav is None else (nav.idx, bool(nav.returning_to_start))
        if key != self._pre_key:
            self._pre_key, self._pre_spent = key, 0.0  # next waypoint, or a return began/ended
            self._end_pre_fight()

    def _pre_transition_fight(self, p: Perception) -> Command | None:
        """Command for a frame of a pre-takeoff fight, or None when the circuit steps.

        `p.detections` is already the fresh list ordinary combat uses (platform
        bounds and same-surface filter). No combat memory is consulted, so a frame
        without a qualifying detection never attacks.

        A fight is a window, not isolated frames: once a qualifying detection was
        attacked, frames without one keep the circuit suspended (no keys) for up
        to `_reacquire_s`, because a circuit stepped for a frame or two between
        attacks cannot finish a settle or a release while its timeout clock runs.
        Every frame that follows a fight frame is charged to the allowance, so the
        stall is bounded by it whatever the detection pattern.
        """
        nav = self.circuit
        engaged, self._pre_engaged = self._pre_engaged, False
        if not self._pre_transition_s:
            return None
        exhausted = self._pre_spent >= self._pre_transition_s
        if engaged and not exhausted:
            self._pre_spent += self._frame_dt
        if (exhausted or self._supported_combat_platform(p) is None
                or not nav.takeoff_interruptible(p.minimap_pos)):
            self._end_pre_fight()
            return None
        # Keep the side already being fought while it still has a fresh qualifying
        # detection; with a monster on each side the nearest one changes per frame.
        self.target = self._pre_side
        target = self._choose_target(p)
        if target is not None:
            if not engaged:
                self._pre_spent += self._frame_dt
            self._pre_side, self._pre_seen, self._pre_engaged = target, p.now, True
            # Uncommitted only: suspend drops the alignment/settle proof, so
            # the takeoff needs a fresh one after the fight.
            nav.suspend(p.now)
            self.state = "hunting"
            return self._pre_transition_combat(p, target)
        if self._pre_seen is not None and 0 <= p.now - self._pre_seen < self._reacquire_s:
            self._pre_engaged = True
            self._clear_combat_memory()
            nav.suspend(p.now)  # the circuit clock stays frozen for the whole fight
            self.state = "reacquiring_target"
            return Command.NONE  # no attack and no movement without a fresh detection
        self._end_pre_fight()
        return None

    def _pre_transition_combat(self, p: Perception, target: str) -> Command:
        """Ordinary combat command with every walk turned into facing.

        `_combat` walks toward a target outside the deadzone, which with a wide
        attack box can be many minimap pixels. Before a takeoff the character
        stays where the route put it: it only turns and attacks.
        """
        cmd = self._combat(p, Command.NONE, target)
        if cmd.move_x in ("left", "right"):
            cmd = Command("none", "none", cmd.action, face_x=cmd.move_x)
        return cmd

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
            nearest_dx = (abs(self._selected_detection.box.center[0] - px) if self._selected_detection is not None
                          else min(abs(d.box.center[0] - px) for d in p.detections))
            if nearest_dx <= deadzone:
                move_x = "none"
        platform = self.circuit.platform_at(p.minimap_pos) if self.circuit is not None else None
        if platform is not None and ((move_x == "left" and p.minimap_pos[0] <= platform.left)
                                     or (move_x == "right" and p.minimap_pos[0] >= platform.right)):
            move_x = "none"
        move_y = cmd.move_y if p.on_ladder else "none"
        face_x = target if move_x == "none" and target in ("left", "right") else "none"
        return Command(move_x, move_y, action, face_x=face_x)

    def _recovering(self, p: Perception) -> Command:
        action = "none"
        if not self._recover_jumped:
            self._recover_jumped = True
            action = "jump"
        return Command(self._recover_dir, "none", action)

    # -- main entry ----------------------------------------------------
    def decide(self, p: Perception) -> Command:
        self._track_frame(p)
        if self.circuit is not None and (self.movement_test or self.circuit.completed
                or self.circuit.state == "blocked"
                or getattr(self.circuit, "returning_to_start", False)
                or getattr(self.circuit, "recovery_terminal", False)):
            self._clear_combat_memory()
            self.loot.suspend(p.now)
            cmd = self.circuit.step(p.minimap_pos, p.now, p.on_ladder)
            self.state = "circuit_" + self.circuit.state
            return cmd
        observed_detections = bool(p.detections)
        self._raw_detections = p.detections if p.player is not None else []
        platform = self.circuit.platform_at(p.minimap_pos) if self.circuit is not None else None
        bounds = (platform.left, platform.right) if platform is not None else None
        if self.circuit is not None and self.circuit.platforms:
            if platform is None or p.player is None:
                p = replace(p, detections=[])
                self.loot.cancel()
            else:
                raw = p.detections
                p = replace(p, detections=[d for d in raw
                    if platform.left <= self._minimap_x(p, d) <= platform.right
                    and self._on_player_surface(platform, p, d)])
                if self._surface_enabled:
                    # A detection at this surface's height, in or out of the
                    # platform bounds or mid-hop, is not a kill. A monster on a
                    # lower or upper stair says nothing about the tracked target.
                    observed_detections = any(
                        -self._surface_above - 32 <= self._surface_offset(platform, p, d) <= self._surface_below
                        for d in raw)
        if self._combat_memory is not None and (self._supported_combat_platform(p) != self._combat_memory["platform"]
                or not 0 <= p.now - self._combat_memory["seen"] < self._reacquire_s):
            self._clear_combat_memory()
        if self.circuit is not None and self.circuit.transition_active:
            self._clear_combat_memory()
            self.loot.suspend(p.now)
            fight = self._pre_transition_fight(p)
            if fight is not None:
                return fight
            self._clear_combat_memory()
            cmd = self.circuit.step(p.minimap_pos, p.now, p.on_ladder)
            self.state = "circuit_" + self.circuit.state
            return cmd
        target = self._choose_target(p)
        if target is not None and (self.circuit is not None or self.state != "recovering"):
            self._remember_target(p, target)
        if target is None:
            pursuit = self._pursue_fresh_target(p)
            if pursuit is not None:
                self.circuit.suspend(p.now)
                return pursuit
        if target is None and self._combat_memory is not None:
            # Keep the same route point while vision reacquires; never attack old pixels.
            self.circuit.suspend(p.now)
            self.state = "reacquiring_target"
            return Command.NONE
        loot_cmd = None
        if target is None:
            loot_cmd = self.loot.step(p.minimap_pos, p.now, observed_detections, p.player is not None, bounds)
        if self.circuit is not None:
            if target is not None or loot_cmd is not None:
                self.circuit.suspend(p.now)
                self.state = "hunting" if target is not None else "looting_" + self.loot.state
                return self._combat(p, Command.NONE, target) if target is not None else loot_cmd
            cmd = self.circuit.step(p.minimap_pos, p.now, p.on_ladder)
            self.state = "circuit_" + self.circuit.state
            return cmd
        if loot_cmd is not None and self.state != "recovering":
            self.stuck.reset()
            self.state = "looting_" + self.loot.state
            return loot_cmd
        if self.state.startswith("looting_"):
            self.state = "hunting"
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
