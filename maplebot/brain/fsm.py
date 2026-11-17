"""Per-frame decision: route or sweep, combat override, stuck recovery."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field, replace
import math

from maplebot.brain.combat import attack_boxes, choose_target
from maplebot.brain.search import SweepSearcher
from maplebot.nav.stuck import StuckDetector
from maplebot.nav.circuit import MinimapCircuit
from maplebot.nav.loot import LootCollector
from maplebot.types import Command, Detection

# The landing edge band of one decided frame (attack.gap_landing_band_x): the landing floor,
# its gap-side end, the band's inner line, the direction of the edge, the gap jump's index,
# whether the character is supported on that floor / inside the band, whether the route is
# not walking it toward the edge, and the loot exclusion interval [lo, hi].
EdgeBand = namedtuple("EdgeBand", ("platform", "edge_x", "inner_x", "toward", "jump_index", "on",
                                   "inside", "route_out", "lo", "hi"))
BAND_ENTRY_MARGIN_X = 1.0   # the same 1 px inset pursuit keeps inside a drawn floor


def _finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


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
        # Opt-in take-off-side veto (absent, null or false = off): at a level gap
        # take-off within this many minimap px of the gap edge, a same-platform
        # detection on the side away from the flight ends the pre-takeoff fight for
        # that frame, so the circuit steps as with pre_transition_s 0.
        veto = self.attack_cfg.get("pre_transition_gap_veto_x")
        if veto is None or veto is False:
            self._gap_veto_x = None
        elif (not isinstance(veto, (int, float)) or isinstance(veto, bool)
                or not math.isfinite(veto) or not 0 < veto <= 20):
            raise ValueError("attack.pre_transition_gap_veto_x must be a finite number above 0 and at most 20, "
                             "or false")
        else:
            self._gap_veto_x = float(veto)
        # Opt-in landing edge band (absent, null or false = off): on the walks after a
        # LEVEL gap jump, on its landing floor, the last X minimap px before that floor's
        # gap-side end. Combat and pursuit never step toward the edge there, kill sites in
        # it are not looted, and a bounded window steps the route alone to walk out.
        band = self.attack_cfg.get("gap_landing_band_x")
        self._band_priority_s = self._band_lap_s = None
        if band is None or band is False:
            self._band_x = None
        elif not _finite_number(band) or not 2 <= band <= 12:
            raise ValueError("attack.gap_landing_band_x must be a finite number from 2 to 12 minimap px, or false")
        else:
            self._band_x = float(band)
            if (self.circuit is not None and not self.movement_test
                    and not isinstance(circuit_cfg.get("editor_plan"), dict)):
                raise ValueError("attack.gap_landing_band_x needs a saved editor plan")
            priority = self.attack_cfg.get("gap_landing_band_priority_s", 2.5)
            if not _finite_number(priority) or not (priority == 0 or 0.5 <= priority <= 4.0):
                raise ValueError("attack.gap_landing_band_priority_s must be 0 (off) or finite between 0.5 and 4.0 s")
            self._band_priority_s = float(priority)
            if self._band_priority_s > 0:
                lap = self.attack_cfg.get("gap_landing_band_lap_s", 8.0)
                if not _finite_number(lap) or not self._band_priority_s <= lap <= 20:
                    raise ValueError("attack.gap_landing_band_lap_s must be finite, at least "
                                     "gap_landing_band_priority_s and at most 20 s")
                self._band_lap_s = float(lap)
        self._band = None
        self._band_inside = False
        self._band_spent = 0.0
        self._band_lap_spent = 0.0
        self._band_lap_key = None
        # Status counters: decisions, never kills, items or prevented knocks.
        self.edge_band_frames = 0
        self.edge_band_inside_s = 0.0
        self.edge_band_brain_frames = 0
        self.edge_band_route_dwell_s = 0.0
        self.edge_band_window_frames = 0
        self.edge_band_window_s = 0.0
        self.edge_band_windows_spent = 0
        self.edge_band_move_clamps = 0
        self.edge_band_pursuit_blocks = 0
        self.edge_band_visits_parked = 0
        self.edge_band_last = None
        self.gap_veto_frames = 0
        # route.circuit.gap_reentry: a bounded window per waypoint selection in which a
        # pending re-entry is stepped before combat and loot (see decide()).
        self._reentry_spent = 0.0
        self.reentry_priority_frames = 0
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
        low, high = platform.left + 1, platform.right - 1
        band = self._band
        if band is not None and band.on and band.platform == platform:
            # Landing edge band: never chase a target into it. Only a candidate toward
            # the edge is bounded; a pursuit away from the edge is today's pursuit.
            if band.toward == "right":
                high = min(high, band.inner_x - BAND_ENTRY_MARGIN_X)
            else:
                low = max(low, band.inner_x + BAND_ENTRY_MARGIN_X)
        else:
            band = None
        blocked = False
        for det in p.detections:
            dx = det.box.center[0] - p.player[0]
            target_x = p.minimap_pos[0] + dx / self.loot.scale
            # Stay one minimap pixel inside the drawn floor, and retain the
            # ordinary vertical attack band so an adjacent ledge is not chased.
            if (abs(dx) <= expanded["range_x"] and platform.left + 1 <= target_x <= platform.right - 1
                    and choose_target([det], p.player, expanded, self.min_area, p.frame_w, p.frame_h) is not None):
                # The same side test as the pursuit direction below.
                toward_edge = band is not None and ("left" if dx < 0 else "right") == band.toward
                if not toward_edge or low <= target_x <= high:
                    candidates.append(det)
                else:
                    blocked = True  # excluded only by the band's narrowed bound
        if blocked:
            self.edge_band_pursuit_blocks += 1
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
            self._reentry_spent = 0.0  # the re-entry priority window is per waypoint selection too
            self._band_spent = 0.0  # and so is the landing edge band's walk-out window
            self._end_pre_fight()
        if self._band_x is not None:
            # The band's lap allowance: only a completed lap or another circuit object resets it.
            lap_key = None if nav is None else (id(nav), nav.laps_completed)
            if lap_key != self._band_lap_key:
                self._band_lap_key, self._band_lap_spent = lap_key, 0.0

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
        if self._gap_veto_x is not None and self._gap_veto(p):
            # A threat behind the character at a gap edge: never hold the take-off
            # column in an in-place fight. The circuit steps as without a pre-fight.
            self.gap_veto_frames += 1
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

    def _gap_veto(self, p: Perception) -> bool:
        """At a level gap take-off near its edge, a filtered detection away from the flight.

        ``p.detections`` is the already filtered same-platform list; nothing else
        is a target source here and nothing waits. Only an uncommitted level gap
        jump in its align phase (``level_gap_takeoff``), with the character
        strictly on that jump's source platform (and on no other drawn platform,
        so never at an ambiguous stair-top column) and closer than
        ``pre_transition_gap_veto_x`` to the gap edge, can be vetoed. A detection
        only on the flight side keeps today's fight.
        """
        nav = self.circuit
        gap = nav.level_gap_takeoff()
        if gap is None or p.player is None or p.minimap_pos is None:
            return False
        x = p.minimap_pos[0]
        source = gap.source
        if (not source.left <= x <= source.right or source.support_error(p.minimap_pos) > nav.tolerance_y
                or not abs(x - gap.edge_x) < self._gap_veto_x):
            return False
        # Strictly on the take-off floor means unambiguously: no other drawn platform
        # supports the position too (e.g. a stair's top end at the gap edge column).
        if any(other is not source and other.left <= x <= other.right
               and other.support_error(p.minimap_pos) <= nav.tolerance_y for other in nav.platforms):
            return False
        if gap.direction == "left":
            return any(d.box.center[0] > p.player[0] for d in p.detections)
        return any(d.box.center[0] < p.player[0] for d in p.detections)

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
        band = self._band
        if (band is not None and band.on and move_x == band.toward
                and (p.minimap_pos[0] >= band.inner_x - BAND_ENTRY_MARGIN_X if band.toward == "right"
                     else p.minimap_pos[0] <= band.inner_x + BAND_ENTRY_MARGIN_X)):
            move_x = "none"  # landing edge band: face and attack in place, as in the deadzone
            self.edge_band_move_clamps += 1
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
        cmd = self._decide(p)
        if self._band_inside:
            if not self.state.startswith("circuit_"):
                self.edge_band_brain_frames += 1  # residual Brain-owned exposure in the band
            elif self.state == "circuit_dwelling":
                self.edge_band_route_dwell_s += self._frame_dt  # route-owned exposure (a stop in the band)
        return cmd

    def _edge_band(self, p: Perception):
        """The landing edge band of this frame (``EdgeBand``), or None. Read-only.

        Only on a plain selected walk after a LEVEL gap jump, on its landing floor
        (``MinimapCircuit.landing_band``), and only when that floor is at least twice
        the band width. Membership is the ordinary ``_supported_combat_platform``
        test (tolerance_y), so one-pixel height jitter on the floor is covered.
        ``route_out``: the route's own target is not further toward the edge than
        the character, beyond the walk tolerance (it walks away or dwells).
        """
        land = self.circuit.landing_band()
        if land is None or p.minimap_pos is None:
            return None
        platform = land.platform
        if platform.right - platform.left < 2 * self._band_x:
            return None
        x = p.minimap_pos[0]
        right = land.toward == "right"
        inner = land.edge_x - self._band_x if right else land.edge_x + self._band_x
        tol = self.loot.tolerance_x
        lo, hi = (inner - tol, land.edge_x) if right else (land.edge_x, inner + tol)
        on = self._supported_combat_platform(p) == platform
        inside = on and (x >= inner if right else x <= inner)
        point = self.circuit.waypoints[self.circuit.idx]
        route_out = ((point.x - x <= self.circuit.tolerance_x) if right
                     else (x - point.x <= self.circuit.tolerance_x))
        return EdgeBand(platform, land.edge_x, inner, land.toward, land.jump_index, on, inside, route_out, lo, hi)

    def _decide(self, p: Perception) -> Command:
        self._track_frame(p)
        self._band, self._band_inside = None, False
        if self._band_x is not None:
            self.loot.exclusion = None
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
        nav = self.circuit
        if (nav is not None and nav.gap_reentry_active
                and self._reentry_spent < nav.gap_reentry_priority_s
                and nav.reentry_pending(p.minimap_pos)):
            # Knocked back across a level gap: for a bounded, clock-charged window per
            # waypoint selection the route alone is stepped (never suspended), so its
            # 3-frame proof can form. No attack, no movement of our own, no loot.
            self._reentry_spent += self._frame_dt
            self.reentry_priority_frames += 1
            self._clear_combat_memory()
            self._pursuit_seen = None
            self.loot.suspend(p.now)
            cmd = nav.step(p.minimap_pos, p.now, p.on_ladder)
            self.state = "circuit_" + nav.state
            return cmd
        if self._band_x is not None and nav is not None:
            band = self._band = self._edge_band(p)
            if band is not None:
                self.loot.exclusion = (band.platform, band.lo, band.hi)
                if band.inside:
                    self._band_inside = True
                    self.edge_band_frames += 1
                    self.edge_band_inside_s += self._frame_dt
                    self.edge_band_last = {"index": nav.idx, "jump_index": band.jump_index,
                                           "edge_x": band.edge_x, "inner_x": band.inner_x,
                                           "x": p.minimap_pos[0], "window": False}
                    if (self._band_priority_s and band.route_out
                            and self._band_spent < self._band_priority_s
                            and self._band_lap_spent < self._band_lap_s):
                        # Standing in the band while the route walks away from the edge
                        # (or dwells): for a bounded, clock-charged window per waypoint
                        # selection and per lap the route alone is stepped, never
                        # suspended. No attack, no movement of our own, no loot start; a
                        # begun visit outside the band is remembered for later.
                        self._band_spent += self._frame_dt
                        self._band_lap_spent += self._frame_dt
                        self.edge_band_window_frames += 1
                        self.edge_band_window_s += self._frame_dt
                        if (self._band_spent >= self._band_priority_s
                                or self._band_lap_spent >= self._band_lap_s):
                            self.edge_band_windows_spent += 1
                        self.edge_band_last["window"] = True
                        self._clear_combat_memory()
                        self._pursuit_seen = None
                        self.edge_band_visits_parked += self.loot.park(p.now)
                        cmd = nav.step(p.minimap_pos, p.now, p.on_ladder)
                        self.state = "circuit_" + nav.state
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
