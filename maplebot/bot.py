"""The bot: one frame in, one Command out, every loop tick."""
from __future__ import annotations

import datetime
import math
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from maplebot.brain.combat import search_box
from maplebot.brain.fsm import Brain, Perception
from maplebot.capture.source import ImageSource, VideoSource, prepare_frame
from maplebot.config import parse_color_codes
from maplebot.control.backend import make_backend
from maplebot.control.keyboard import KeyboardController
from maplebot.evidence import EvidenceTracker
from maplebot.health import HealthPolicy
from maplebot.log import get_logger
from maplebot.nav.route import RouteFollower, RouteMap, locate_minimap_on_map
from maplebot.types import Box, Command, Detection, PlayerFix
from maplebot.vision.bars import read_bars
from maplebot.vision.minimap import find_minimap, find_other_players, find_player_dot
from maplebot.vision.monsters import MonsterLibrary, detect_hp_bars, detect_monsters
from maplebot.vision.player import locate_by_party_bar, locate_by_template
from maplebot.vision.template import make_mask

log = get_logger()
DOT_LOST_FRAMES = 15
MAP_MATCH_MAX_SCORE = 0.5  # masked SQDIFF_NORMED above this = the minimap is not on the map (occluded / loading)


@dataclass
class BotSnapshot:
    frame: np.ndarray
    player: tuple[int, int] | None
    minimap_box: Box | None
    minimap_pos: tuple[int, int] | None
    global_pos: tuple[int, int] | None
    detections: list[Detection]
    command: Command
    state: str
    hp: float | None
    mp: float | None
    exp: float | None
    fps: float
    route_idx: int | None
    search_box: Box | None = None
    target: str | None = None
    other_players: list[tuple[int, int]] = field(default_factory=list)
    paused: bool = False
    route_map: RouteMap | None = None
    evidence: dict | None = None  # on-screen gain observations (lower bounds); None when evidence.enabled is false

    def summary(self) -> str:
        return (f"state={self.state} cmd=\"{self.command}\" player={self.player} minimap={self.minimap_pos} "
                f"global={self.global_pos} mobs={len(self.detections)} target={self.target} "
                f"hp={self.hp} mp={self.mp} fps={self.fps:.1f}")


class Bot:
    def __init__(self, cfg: dict, source, keyboard: KeyboardController | None,
                 library: MonsterLibrary, route_map: RouteMap | None):
        self.cfg = cfg
        self.source = source
        self.keyboard = keyboard
        self.library = library
        self.route_map = route_map
        self.follower = RouteFollower(route_map, cfg["route"]) if route_map is not None else None
        self.brain = Brain(cfg, self.follower, library.min_area)
        self.health = HealthPolicy(cfg["health"])
        # Telemetry only: reads pixels of frames the loop already processed, never grants or sends input.
        self.evidence = EvidenceTracker(cfg["evidence"]) if cfg.get("evidence", {}).get("enabled") else None
        self.paused = False
        self._control_lock = threading.RLock()
        self._control_generation = 0
        self.full_pause = False
        self.health_protection = None
        self.health_protection_active = False
        self._route_health_hold = False
        self._manual_health_hold = False
        self.health_hold_reason = None
        self._hp_health_hold = False
        self.frame_idx = 0
        self.fps = 0.0
        self.on_ladder = False
        self.player: PlayerFix | None = None
        self.minimap_box: Box | None = None
        self.minimap_pos: tuple[int, int] | None = None
        self.minimap_raw_pos: tuple[int, int] | None = None
        self.minimap_raw_image: np.ndarray | None = None
        self.minimap_image: np.ndarray | None = None
        self.minimap_offset: tuple[int, int] | None = (0, 0)
        self.global_pos: tuple[int, int] | None = None
        self.last_snapshot: BotSnapshot | None = None
        self._minimap_loc: tuple[int, int] | None = None
        self._dot_lost = 0
        self._minimap_seen = False  # minimap box and player dot both found on the current frame
        self.capture_state = "waiting"
        self.capture_sequence = None
        self.capture_time = None
        self.map_recognizer = None
        self.map_match = None
        self.map_ready = True  # build_bot installs a mandatory guard for live capture
        self.map_reacquiring = False
        self.map_reacquire_reason = None
        self._map_had_lock = False
        self._map_lost_since = None
        self._map_reacquire_failed = False
        self.pending_map_profile = None
        from maplebot.capture.window import WindowCapture
        self._requires_visual_lock = isinstance(source, WindowCapture)
        self._input_blocked = self._requires_visual_lock
        self._missing_frames = 0
        self._max_missing_frames = max(1, int(cfg.get("safety", {}).get("max_missing_frames", 3)))
        self._vertical_minimap_grace_s = max(0.0, min(2.0, float(
            cfg.get("safety", {}).get("vertical_minimap_grace_s", 2.0))))
        self._vertical_minimap_active = False
        self._minimap_route_active = False
        self._return_route_active = False
        self._vertical_minimap_idx = self._vertical_minimap_started = None
        self._vertical_minimap_wait_fix = False
        if self.keyboard is not None and self._input_blocked:
            self.keyboard.set_enabled(False)
        self._t_last: float | None = None
        self._tmpl = None
        self._tmpl_mask = None
        self._nametag_occlusion_mask = None
        self._appearance_tmpl = None
        self._appearance_mask = None
        self._hybrid_primary_pos: tuple[int, int] | None = None
        self._hybrid_missing_frames = 0
        self._hybrid_max_frames = max(0, min(12, int(
            cfg["player"].get("hybrid", {}).get("max_fallback_frames", 12))))
        self._load_player_template()

    # -- setup -----------------------------------------------------------
    def _load_player_template(self) -> None:
        pcfg = self.cfg["player"]
        loc = pcfg.get("locator", "party_bar")
        if loc not in ("nametag", "appearance", "hybrid"):
            return
        primary = "nametag" if loc == "hybrid" else loc
        path = str(pcfg[primary].get("path") or "")
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
        if raw is None:
            if loc == "hybrid":
                log.warning("hybrid locator has no primary nametag at %r; awaiting a valid template", path)
                return
            log.warning("player.locator=%s but no template at %r; falling back to party_bar", loc, path)
            pcfg["locator"] = "party_bar"
            return
        bgr, mask = make_mask(raw)
        if primary == "nametag":
            self._tmpl = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            self._tmpl = bgr
            if not np.any(mask == 0):  # raw screenshot crop: ignore the corners
                mask = np.zeros(bgr.shape[:2], np.uint8)
                hh, ww = mask.shape
                cv2.ellipse(mask, (ww // 2, hh // 2), (int(ww * 0.46), int(hh * 0.46)), 0, 0, 360, 255, -1)
        self._tmpl_mask = mask
        if primary == "nametag":
            width = max(0, min(self._tmpl.shape[1] // 3,
                              int(pcfg["nametag"].get("center_occlusion_px", 0))))
            if width:
                self._nametag_occlusion_mask = mask.copy()
                left = (self._tmpl.shape[1] - width) // 2
                self._nametag_occlusion_mask[:, left:left + width] = 0
        if loc == "hybrid":
            path = str(pcfg["appearance"].get("path") or "")
            raw = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
            if raw is None:
                log.warning("hybrid appearance fallback unavailable at %r; using nametag only", path)
                return
            self._appearance_tmpl, mask = make_mask(raw)
            if not np.any(mask == 0):
                mask = np.zeros(self._appearance_tmpl.shape[:2], np.uint8)
                hh, ww = mask.shape
                cv2.ellipse(mask, (ww // 2, hh // 2), (int(ww * 0.46), int(hh * 0.46)), 0, 0, 360, 255, -1)
            self._appearance_mask = mask

    # -- per-frame pieces --------------------------------------------------
    def _update_input_gate(self, current_lock: bool) -> None:
        if not self._requires_visual_lock:
            return
        was_blocked = self._input_blocked
        if current_lock:
            self._missing_frames = 0
            self._input_blocked = False
        else:
            self._missing_frames += 1
            if self._missing_frames >= self._max_missing_frames:
                self._input_blocked = True
        if self.keyboard is not None and (self.paused or (self._input_blocked
                and not self._vertical_minimap_active and not self._minimap_route_active
                and not self._return_route_active)):
            self.keyboard.set_enabled(False)
        if self._input_blocked != was_blocked:
            log.info("visual lock %s; input %s", "lost" if self._input_blocked else "restored",
                     "blocked" if self._input_blocked else "ready")

    def _locate_player(self, frame: np.ndarray) -> PlayerFix | None:
        pcfg = self.cfg["player"]
        loc = pcfg.get("locator", "party_bar")
        last = self.player.pos if self.player else None
        if loc in ("nametag", "hybrid"):
            if self._tmpl is None:
                return None
            c = pcfg["nametag"]
            fix = locate_by_template(frame, self._tmpl, self._tmpl_mask, last, tuple(c["offset"]),
                                     float(c["diff_thres"]), split_width=int(c.get("split_width", 0)),
                                     source="nametag", local_radius=100,
                                     local_only=loc == "hybrid" and self._hybrid_primary_pos is not None)
            if fix is None and self._nametag_occlusion_mask is not None:
                # A foreground ladder can cover the tag's centre. Retain both sides
                # as one name-shaped match; never broaden the identity search window.
                fix = locate_by_template(
                    frame, self._tmpl, self._nametag_occlusion_mask, last, tuple(c["offset"]),
                    min(float(c["diff_thres"]), float(c.get("center_occlusion_diff_thres", .11))),
                    split_width=0, source="nametag_occlusion", local_radius=100,
                    local_only=loc == "hybrid" and self._hybrid_primary_pos is not None)
            if loc == "nametag":
                return fix
            if fix is not None:
                self._hybrid_primary_pos = fix.pos
                self._hybrid_missing_frames = 0
                return fix
            self._hybrid_missing_frames += 1
            if (self._hybrid_primary_pos is None or self._appearance_tmpl is None
                    or self._hybrid_missing_frames > self._hybrid_max_frames):
                return None
            c = pcfg["appearance"]
            return locate_by_template(
                frame, self._appearance_tmpl, self._appearance_mask, self._hybrid_primary_pos,
                tuple(c["feet_offset"]), 1.0 - float(c["match_thres"]), flip=True,
                method=cv2.TM_CCORR_NORMED, local_radius=max(1, min(100, int(c.get("search_radius", 100)))),
                local_only=True, source="appearance_fallback")
        if loc == "appearance":
            c = pcfg["appearance"]
            return locate_by_template(frame, self._tmpl, self._tmpl_mask, last, tuple(c["feet_offset"]),
                                      1.0 - float(c["match_thres"]), flip=True, method=cv2.TM_CCORR_NORMED,
                                      local_radius=int(c.get("search_radius", 140)), source="appearance")
        return locate_by_party_bar(frame, pcfg["party_bar"], exclude=self.minimap_box)

    def _update_ladder(self, fix: PlayerFix) -> None:
        if self.player is None:
            return
        dx = abs(fix.x - self.player.x)
        dy = abs(fix.y - self.player.y)
        if self.on_ladder:
            if dx > 3:
                self.on_ladder = False
        elif dx < 3 and dy != 0:
            self.on_ladder = True

    def _update_minimap(self, frame: np.ndarray) -> np.ndarray | None:
        self.minimap_raw_pos = None
        self.minimap_raw_image = self.minimap_image = None
        if self.map_recognizer is not None and (not self.map_ready or self.minimap_offset is None
                or getattr(self.map_match, "state", None) != "matched"
                or getattr(self.map_match, "profile", None) is None):
            self._minimap_seen = False
            return None
        if self.map_recognizer is None:
            self.minimap_offset = (0, 0)
        mcfg = self.cfg["minimap"]
        fixed = mcfg.get("fixed_rect")
        box = Box(*map(int, fixed)) if fixed else find_minimap(frame)
        self._minimap_seen = box is not None
        if box is not None:
            self.minimap_box = box
        if self.minimap_box is None:
            self._minimap_seen = False
            return None
        b = self.minimap_box.clamp(frame.shape[1], frame.shape[0])
        if b.area == 0:
            self._minimap_seen = False
            return None
        img = frame[b.y:b.y2, b.x:b.x2]
        self.minimap_raw_image = img
        offset = (0, 0)
        if self.map_recognizer is not None:
            offset = self.minimap_offset
            reference = getattr(self.map_match.profile, "reference_bgr", None)
            size = reference.shape[1::-1] if reference is not None else img.shape[1::-1]
            # Compare current and previous dots in the saved map's coordinates,
            # rather than treating a scrolling crop as character movement.
            img = cv2.warpAffine(img, np.float32([[1, 0, offset[0]], [0, 1, offset[1]]]),
                                 size, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        self.minimap_image = img
        dot = find_player_dot(img, tuple(mcfg["player_color_bgr"]), int(mcfg["player_tolerance"]),
                              last=self.minimap_pos, max_jump_px=int(mcfg["max_jump_px"]))
        if dot is not None:
            self.minimap_pos = dot
            self.minimap_raw_pos = (dot[0] - offset[0], dot[1] - offset[1])
            self._dot_lost = 0
        else:
            self._minimap_seen = False
            self._dot_lost += 1
            if self._dot_lost >= DOT_LOST_FRAMES:
                self.minimap_pos = None  # allow a fresh lock after a teleport / map change
        return img

    def _update_global(self, img_mm: np.ndarray | None) -> None:
        if self.route_map is None or img_mm is None or self.minimap_pos is None or not self._minimap_seen:
            self.global_pos = None  # navigation skips this frame; the route follower's grace handles it
            return
        res = locate_minimap_on_map(self.route_map.map_bgr, img_mm, last=self._minimap_loc)
        if res.score > MAP_MATCH_MAX_SCORE:
            self.global_pos = None
            return
        self._minimap_loc = res.loc
        self.global_pos = (res.loc[0] + self.minimap_pos[0], res.loc[1] + self.minimap_pos[1])

    def _detect(self, frame: np.ndarray) -> tuple[list[Detection], Box | None]:
        if self.player is None or not self.library.templates and not self.cfg["monsters"]["hp_bar"]["enable"]:
            return [], None
        h, w = frame.shape[:2]
        sb = search_box(self.player.pos, self.cfg["attack"], w, h)
        if sb.area == 0:
            return [], sb
        # Match complete sprites whose centres are in the logical search box.
        # Cropping at that box used to discard monsters straddling its edges.
        pad_x = max(((t.img_bgr.shape[1] + 1) // 2 for t in self.library.templates), default=0)
        pad_y = max(((t.img_bgr.shape[0] + 1) // 2 for t in self.library.templates), default=0)
        processing = Box(sb.x - pad_x, sb.y - pad_y,
                         sb.w + 2 * pad_x, sb.h + 2 * pad_y).clamp(w, h)
        roi = frame[processing.y:processing.y2, processing.x:processing.x2]
        pcfg = self.cfg["player"]
        bw, bh = int(pcfg.get("box_w", 100)), int(pcfg.get("box_h", 150))
        exclude = Box(self.player.x - bw // 2, self.player.y - bh // 2, bw, bh)
        mcfg = self.cfg["monsters"]
        dets = detect_monsters(roi, self.library, mcfg, exclude, (processing.x, processing.y))
        dets = [d for d in dets if sb.contains(d.box.center)]
        hb = mcfg.get("hp_bar", {})
        if hb.get("enable"):
            dets += detect_hp_bars(frame[sb.y:sb.y2, sb.x:sb.x2], tuple(hb["color_bgr"]),
                                   int(hb.get("tolerance", 10)), roi_origin=(sb.x, sb.y))
        return dets, sb

    def _read_health(self, frame: np.ndarray, now: float, *, allow_requests: bool = True) -> tuple[float | None, float | None, float | None]:
        hcfg = self.cfg["health"]
        if not hcfg.get("enable", True):
            return (None, None, None)
        if hcfg.get("fixed_rects"):
            hp, mp, exp = read_bars(frame, hcfg)
        else:
            y0 = int(hcfg.get("ui_y_start", 0))
            ui = frame[y0:] if 0 <= y0 < frame.shape[0] else frame
            hp, mp, exp = read_bars(ui, hcfg)
        input_ready = allow_requests and not self.paused and not self._input_blocked and (
            self.keyboard is None or (self.keyboard.enabled and self.keyboard.is_active()))
        requests = self.health.update(hp, mp, now) if input_ready else []
        for req in requests:
            if req == "hp_potion" and self.health_protection is not None:
                continue  # the confirmed-response guard owns HP; ordinary policy still owns MP
            if self.keyboard is not None and not self.paused and not self._input_blocked:
                self.keyboard.request(req)
                log.info("%s (hp=%s mp=%s)", req, hp, mp)
        return hp, mp, exp

    def _observe_evidence(self, frame, now: float, ok: bool, state: str | None = None) -> None:
        """Count only on a fresh capture of the recognised map while not paused; anything else resets its baseline.

        `state` replaces the brain state for the time bucket on a step that sent nothing because input
        was held (suspended brain keeps its last state); pixel counting is unaffected.
        """
        if self.evidence is not None:
            self.evidence.update(frame, now, self.brain.state if state is None else state, fresh=bool(ok))

    def evidence_snapshot(self) -> dict | None:
        return self.evidence.snapshot() if self.evidence is not None else None

    def pause_all(self) -> None:
        """An explicit user pause always revokes healing as well as farming."""
        with self._control_lock:
            self._control_generation += 1
            self.paused = self.full_pause = True
            self._route_health_hold = self._hp_health_hold = False
            self._manual_health_hold = False
            self.health_hold_reason = None
            self.health_protection_active = False
            self._return_route_active = self._minimap_route_active = False
            self.brain.loot.suspend(time.monotonic())
            if self.health_protection is not None:
                self.health_protection.suspend()
            if self.keyboard is not None:
                self.keyboard.set_enabled(False)

    def resume(self) -> None:
        """Fresh health and map checks, not this request, grant input permission."""
        with self._control_lock:
            self._control_generation += 1
            if self._manual_health_hold:
                self._manual_health_hold = self._route_health_hold = False
                self.health_hold_reason = None
            self.full_pause = self.paused = False
            if self.health_protection is not None:
                self.health_protection.request_resume()
            if self.keyboard is not None:
                self.keyboard.set_enabled(False)

    def hold_farming(self, reason="Farming is on hold; HP protection remains on.") -> None:
        """Explicitly hold farming while retaining only guarded healing permission."""
        with self._control_lock:
            self._control_generation += 1
            self.paused, self.full_pause = True, False
            self._manual_health_hold = self._route_health_hold = True
            self.health_hold_reason = str(reason)
            self._return_route_active = self._minimap_route_active = self._vertical_minimap_active = False
            self.brain.suspend(time.monotonic())
            self.brain.target = None
            if self.keyboard is not None:
                self.keyboard.set_enabled(False)

    def _health_guard_step(self, frame, now, foreground):
        """A failed route may heal, but can never borrow ordinary action permission."""
        keyboard = self.keyboard
        generation = self._control_generation
        hcfg = self.cfg["health"]
        self.health_protection_active = False
        if (keyboard is None or self.brain.movement_test or not hcfg.get("enable", True)
                or self.map_recognizer is None or self.capture_time is None):
            return None
        from maplebot.health_protection import HealthProtection
        if self.health_protection is None:
            self.health_protection = HealthProtection(hcfg)
        guard = self.health_protection
        nav = self.brain.circuit
        terminal = nav is not None and (nav.recovery_terminal or (
            nav.state == "blocked" and not nav.return_on_block and not nav.returning_to_start))
        if terminal and not self._manual_health_hold and not hcfg.get("protect_on_route_failure", True):
            self.pause_all()
            return None
        if terminal and not self.full_pause and hcfg.get("protect_on_route_failure", True):
            self._route_health_hold = True
        fresh = (self.capture_state == "fresh" and math.isfinite(self.capture_time)
                 and 0 <= keyboard.clock() - self.capture_time <= keyboard.max_frame_age_s)
        context_ready = fresh and self.map_ready and self._minimap_seen and foreground
        if self.full_pause or not context_ready:
            guard.suspend()
            if self._route_health_hold or self._hp_health_hold or guard.latched:
                self.paused = True
                keyboard.set_enabled(False)
            return None
        if self._route_health_hold or self._hp_health_hold or guard.latched:
            keyboard.set_hp_only(source_time=self.capture_time)
        hp, mp, exp = self._read_health(frame, now, allow_requests=False)
        with self._control_lock:
            return self._finish_health_guard(frame, now, hp, mp, exp, generation)

    def _finish_health_guard(self, frame, now, hp, mp, exp, generation):
        keyboard, guard = self.keyboard, self.health_protection
        # A hotkey may arrive while image work is in progress. Its full stop wins.
        if (generation != self._control_generation or self.full_pause or not keyboard.is_active()
                or not 0 <= keyboard.clock() - self.capture_time <= keyboard.max_frame_age_s):
            guard.suspend()
            keyboard.set_enabled(False)
            return None
        # A new low reading retires attacks before queuing any protection action.
        needs_hold = (self._route_health_hold or self._hp_health_hold or guard.latched
                      or hp is None or not isinstance(hp, (int, float))
                      or not math.isfinite(hp) or hp < guard.threshold)
        if needs_hold:
            keyboard.set_hp_only(source_time=self.capture_time)
        request_hp = guard.observe(hp, now, taps=keyboard.hp_taps_sent,
                                   tap_time=keyboard.hp_tap_time,
                                   key_available=bool(str(self.cfg["keys"].get("hp_potion") or "").strip()))
        if guard.latched:
            self._hp_health_hold = True
        elif guard.state in ("healing", "awaiting_response"):
            self._hp_health_hold = True
        elif self._hp_health_hold and hp is not None and hp >= guard.target:
            self._hp_health_hold = False
            if not self._route_health_hold:
                self.paused = False
        held = self._route_health_hold or self._hp_health_hold or guard.latched
        if not held:
            if keyboard.hp_only:
                keyboard.set_enabled(False)  # cancel a not-yet-sent potion before ordinary perception
            return None
        self.paused = True
        self._return_route_active = self._minimap_route_active = self._vertical_minimap_active = False
        self._input_blocked = True
        self.brain.target = None
        self.brain.loot.suspend(now)
        self.brain.suspend(now)
        self.health_protection_active = not guard.latched
        if guard.latched or guard.state == "monitoring":
            keyboard.set_enabled(False)
        else:
            keyboard.set_hp_only(source_time=self.capture_time)
            keyboard.set_enabled(True)
            if request_hp:
                keyboard.request("hp_potion")
        self.brain.state = "health_protection_" + guard.state
        snap = BotSnapshot(frame, self.player.pos if self.player else None, self.minimap_box,
                           self.minimap_pos, None, [], Command.NONE, self.brain.state,
                           hp, mp, exp, self.fps, None, paused=True)
        self.last_snapshot = snap
        return snap

    def apply_movement_test_permission(self, allow: bool, *, expected_generation: int) -> bool:
        """Apply a test's validated permission without overriding a later user control."""
        if not self.brain.movement_test:
            raise ValueError("Only a movement test may grant test permission")
        with self._control_lock:
            if expected_generation != self._control_generation:
                if self.keyboard is not None:
                    self.keyboard.set_enabled(False)
                return False
            self.paused = self.full_pause = not bool(allow)
            if not allow and self.keyboard is not None:
                self.keyboard.set_enabled(False)
            return bool(allow)  # A fresh Bot.step still supplies the keyboard lease.

    def _stop_vertical_minimap(self) -> None:
        self._vertical_minimap_active = False
        self._vertical_minimap_wait_fix = True
        self._vertical_minimap_idx = self._vertical_minimap_started = None

    def _continue_vertical_minimap(self, now: float, current_minimap: bool, foreground: bool) -> Command | None:
        """Finish only an existing vertical leg; never enter combat using a stale screen fix."""
        nav = self.brain.circuit
        if (self.paused or not foreground or not current_minimap or self.minimap_pos is None
                or self._hybrid_primary_pos is None or nav is None or self._vertical_minimap_wait_fix
                or self._vertical_minimap_grace_s <= 0):
            self._stop_vertical_minimap()
            return None
        if self._vertical_minimap_idx is None:
            if nav.state not in ("climbing", "dropping") or not nav.transition_active:
                self._stop_vertical_minimap()
                return None
            self._vertical_minimap_idx, self._vertical_minimap_started = nav.idx, now
        if (nav.idx != self._vertical_minimap_idx
                or now - self._vertical_minimap_started >= self._vertical_minimap_grace_s
                or abs(self.minimap_pos[0] - nav.waypoints[nav.idx].x) > 2):
            self._stop_vertical_minimap()
            return None
        self.brain.target = None
        self.brain.loot.suspend(now)
        cmd = nav.step(self.minimap_pos, now, self.on_ladder)
        if (nav.idx != self._vertical_minimap_idx or cmd.move_x != "none" or cmd.face_x != "none"
                or cmd.action not in ("none", "jump") or nav.state not in ("climbing", "dropping", "dwelling")):
            self._stop_vertical_minimap()
            return None
        self._vertical_minimap_active = True
        self.brain.state = "minimap_vertical_" + nav.state
        return cmd

    # -- main ----------------------------------------------------------------
    def _invalidate_tracking(self, now):
        self.health_protection_active = False
        if self.health_protection is not None:
            self.health_protection.suspend()
        self.brain.suspend(now)
        self._minimap_route_active = False
        self._return_route_active = False
        self._stop_vertical_minimap()
        self.player = self.minimap_pos = self.global_pos = self._minimap_loc = None
        self.minimap_raw_pos = None
        self.minimap_raw_image = self.minimap_image = None
        self.minimap_offset = None if self.map_recognizer is not None else (0, 0)
        self.minimap_box = None
        self._minimap_seen = False
        self._hybrid_primary_pos = None
        self._hybrid_missing_frames = 0
        self.on_ladder = False
        if self._requires_visual_lock:
            self._input_blocked = True
            self._missing_frames = self._max_missing_frames
        if self.keyboard is not None:
            self.keyboard.set_enabled(False)

    def _hard_stop_map(self, reason):
        nav = self.brain.circuit
        if nav is not None and nav.returning_to_start:
            nav.abort_return_to_start(reason)
        if nav is not None and getattr(nav, "rejoining", False):
            nav.cancel_platform_rejoin(reason)  # not terminal: the waypoint keeps its own timeout
        self.pause_all()
        self.map_reacquiring = False
        self.map_reacquire_reason = str(reason)
        self._map_reacquire_failed = True

    def _wait_for_map(self, now, reason):
        """A single bounded gap may reacquire; every uncertain frame stays input-free."""
        if not self._map_had_lock:
            return
        if self._map_reacquire_failed:
            if not self.full_pause:
                self.pause_all()
            return
        if self._map_lost_since is None:
            self._map_lost_since = now
        if now - self._map_lost_since >= 2.0:
            self._hard_stop_map("Saved map alignment was not confirmed within 2 seconds.")
            return
        self.map_reacquiring = True
        self.map_reacquire_reason = str(reason or "Reacquiring the saved map alignment.")

    def _check_map(self, frame, now):
        if self.map_recognizer is None:
            return True
        from maplebot.map_profiles import profile_config_matches
        match = self.map_recognizer.observe(frame, self.cfg)
        self.map_match = match
        active = self.cfg.get("map_profile", {})
        profile = match.profile
        offset = getattr(match, "offset", (0, 0))
        aligned = (isinstance(offset, (tuple, list)) and len(offset) == 2
                   and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                           and math.isfinite(value) and int(value) == value for value in offset))
        ready = (match.state == "matched" and profile is not None
                 and profile.key == active.get("active")
                 and profile.image_sha256 == active.get("reference_sha256")
                 and profile_config_matches(self.cfg, profile, preserve_test=True))
        if (match.state == "matched" and profile is not None
                and profile.key == active.get("active") and not ready):
            from dataclasses import replace
            self.map_match = replace(match, state="settings_mismatch",
                                     reason="Map settings differ from the saved profile. Reapply it while paused.")
        if ready and not aligned:
            from dataclasses import replace
            self.map_match = replace(match, state="alignment_unknown",
                                     reason="The minimap alignment is not confirmed; movement is paused.")
            ready = False
        if not ready:
            if self.keyboard is not None:
                self.keyboard.set_enabled(False)
            hard_change = (self.map_match.state not in ("unknown", "candidate", "ambiguous")
                           or (match.state == "matched" and profile is not None
                               and profile.key != active.get("active")))
            if hard_change:
                self._hard_stop_map("Map identity or saved settings changed")
            else:
                self._wait_for_map(now, self.map_match.reason)
            self._invalidate_tracking(now)
            if match.state == "matched" and profile is not None and profile.key != active.get("active"):
                self.pending_map_profile = profile
        else:
            if self._map_lost_since is not None and now - self._map_lost_since >= 2.0:
                self._hard_stop_map("Saved map alignment was not confirmed within 2 seconds.")
            self.map_reacquiring = False
            if not self._map_reacquire_failed:
                self.map_reacquire_reason = None
            self._map_lost_since = None
            self._map_reacquire_failed = False
            self._map_had_lock = True
        self.map_ready = ready
        self.minimap_offset = tuple(map(int, offset)) if ready else None
        return ready

    def step(self, now: float | None = None) -> BotSnapshot | None:
        now = time.monotonic() if now is None else now
        with self._control_lock:
            if self.full_pause:
                self.paused = True
                if self.keyboard is not None:
                    self.keyboard.set_enabled(False)
        reader = getattr(self.source, "get_sample", None)
        if callable(reader):
            sample = reader(after_sequence=self.capture_sequence)
            if sample is None:
                latest = reader()
                if latest is not None:
                    self.capture_state = "duplicate"
                    if self.map_reacquiring:
                        self._wait_for_map(now, self.map_reacquire_reason)
                    return None  # never renew a command or advance navigation on the same capture
                raw = None
            else:
                raw = sample.frame
                self.capture_sequence, self.capture_time = sample.sequence, sample.captured_at
        else:
            raw = self.source.get_frame()
        if raw is None:
            self.capture_state = "stale_or_missing"
            self._observe_evidence(None, now, False)
            self._invalidate_tracking(now)
            if self.map_recognizer is not None:
                self.map_recognizer.reset()
                self._wait_for_map(now, "Waiting for a fresh game capture to confirm the saved map.")
                self.map_ready = False
            return None
        self.capture_state = "fresh"
        frame = prepare_frame(raw, self.cfg["game_window"])
        h, w = frame.shape[:2]
        self.frame_idx += 1

        if not self._check_map(frame, now):
            self._minimap_route_active = False
            self._return_route_active = False
            # Keep the relay available for calibration without deriving motion from the wrong crop.
            self.brain.state = "map_" + self.map_match.state
            self._observe_evidence(frame, now, False)
            snap = BotSnapshot(frame, None, None, None, None, [], Command.NONE,
                               self.brain.state, None, None, None, self.fps, None, paused=True)
            self.last_snapshot = snap
            return snap

        img_mm = self._update_minimap(frame)
        was_route_only = self._minimap_route_active or self._vertical_minimap_active or self._return_route_active
        fix = self._locate_player(frame)
        if fix is not None:
            self._update_ladder(fix)
            self.player = fix
            self._vertical_minimap_active = self._vertical_minimap_wait_fix = False
            self._vertical_minimap_idx = self._vertical_minimap_started = None
        foreground = self.keyboard is None or self.keyboard.is_active()
        health_snap = self._health_guard_step(frame, now, foreground)
        if health_snap is not None:
            self._observe_evidence(frame, now, False)  # farming is held: no counting, no time bucket
            return health_snap
        nav = self.brain.circuit
        returning = bool(nav is not None and getattr(nav, "returning_to_start", False))
        blocked = bool(nav is not None and (nav.state == "blocked" or getattr(nav, "recovery_terminal", False)))
        loot_context = None
        if (nav is not None and not self.brain.movement_test and not nav.transition_active
                and not returning and not blocked and not self.paused and foreground
                and self.map_recognizer is not None and self.map_ready
                and self._minimap_seen and img_mm is not None and self.minimap_pos is not None
                and self.brain.loot.goal is not None):
            floor = nav._containing_platform(self.brain.loot.goal)
            if (floor is not None and floor.left <= self.minimap_pos[0] <= floor.right
                    and floor.support_error(self.minimap_pos) <= min(1.0, nav.tolerance_y)):
                loot_context = (id(nav), nav.idx, floor, self.map_match.profile.key,
                                self.map_match.profile.image_sha256)
        loot_visual_hold = self.brain.loot.visual_lock(now, fix is not None, loot_context)
        auto_return = bool(blocked and self.cfg.get("route", {}).get("circuit", {}).get("return_on_block")
                           and not self.brain.movement_test and not getattr(nav, "recovery_terminal", False))
        if (returning or blocked) and not was_route_only and self.keyboard is not None:
            self.keyboard.set_command(Command.NONE, source_time=self.capture_time, navigation_only=True)
        if not self.paused and (returning or auto_return):
            from maplebot.recovery_runtime import valid_health
            hp_check, _, _ = self._read_health(frame, now, allow_requests=False)
            minimum = nav.return_min_hp_percent
            if not valid_health(hp_check) or hp_check < minimum:
                nav.abort_return_to_start("HP is unreadable" if not valid_health(hp_check) else f"HP is below {minimum:g}%")
                self.paused = True
                if self.keyboard is not None:
                    self.keyboard.set_enabled(False)
                returning, blocked = False, True
        self._return_route_active = bool((returning or blocked) and nav is not None
            and self.map_ready and self._minimap_seen and img_mm is not None
            and self.minimap_pos is not None and not self.paused and foreground)
        # A recognized map and current player dot are sufficient for a drawn
        # movement route through scenery that hides the sprite/name. Screen-based
        # combat and potions still require their own current visual lock.
        self._minimap_route_active = bool(fix is None and nav is not None and not loot_visual_hold
            and self.map_recognizer is not None and self.map_ready
            and self._minimap_seen and img_mm is not None and self.minimap_pos is not None
            and not self.paused and foreground)
        self._update_input_gate(fix is not None and self._minimap_seen and img_mm is not None)
        if loot_visual_hold:
            self._input_blocked = True
            nav.suspend(now)
            self.brain._clear_combat_memory()
            if self.keyboard is not None:
                self.keyboard.set_enabled(False)  # movement, attacks, pickup and queued requests all stop
        if (returning or blocked) and not self._return_route_active:
            self._input_blocked = True
            self._minimap_route_active = False
            self._vertical_minimap_active = False
            self.brain.suspend(now)
            if self.keyboard is not None:
                self.keyboard.set_enabled(False)
        route_cmd = None
        if self._minimap_route_active or self._return_route_active:
            self._input_blocked = True
            self.brain.target = None
            self.brain.loot.suspend(now)
            route_cmd = nav.step(self.minimap_pos, now, False)
            self.brain.state = "minimap_route_" + nav.state
        vertical_guard = (self._requires_visual_lock and fix is None
                          and not loot_visual_hold
                          and not self._minimap_route_active and not self._return_route_active
                          and not returning and not blocked
                          and (self._vertical_minimap_wait_fix or self._vertical_minimap_idx is not None
                               or (nav is not None and nav.transition_active)))
        vertical_cmd = None
        if vertical_guard:
            self._input_blocked = True  # health and ordinary combat remain gated
            if self.keyboard is not None and not self._vertical_minimap_active:
                self.keyboard.set_enabled(False)  # clears queued potions and any previous command
            vertical_cmd = self._continue_vertical_minimap(
                now, self._minimap_seen and img_mm is not None, foreground)
            if vertical_cmd is None and self.keyboard is not None:
                self.keyboard.set_enabled(False)
        if self.keyboard is not None:
            with self._control_lock:
                if (not self.keyboard.enabled and not self.paused and not self.full_pause
                        and not self._input_blocked and foreground):
                    self.keyboard.set_command(Command.NONE, source_time=self.capture_time)
                    self.keyboard.set_enabled(True)
        if self.keyboard is not None and (route_cmd is not None or vertical_cmd is not None):
            # Retire ordinary action permission before potentially expensive vision.
            self.keyboard.set_command(route_cmd if route_cmd is not None else vertical_cmd,
                                      source_time=self.capture_time, navigation_only=True)
        self._update_global(img_mm)
        other = find_other_players(img_mm, tuple(self.cfg["minimap"]["other_color_bgr"])) if img_mm is not None else []
        dets, sb = self._detect(frame)
        # On visual reacquisition, the final command below restores ordinary input.
        # Defer requests one frame without consuming the health retry timer.
        hp, mp, exp = self._read_health(frame, now, allow_requests=not was_route_only)

        evidence_state = None  # telemetry only: time bucket override for a step that idles
        if self.full_pause:
            self.paused = True
            self.brain.suspend(now)
            cmd = Command.NONE
            evidence_state = "input_held"
        elif loot_visual_hold:
            self.brain.state = "looting_waiting_visual"
            cmd = Command.NONE
        elif route_cmd is not None:
            cmd = route_cmd
        elif vertical_cmd is not None:
            cmd = vertical_cmd
        elif self.paused or self._input_blocked or not foreground:
            self.brain.suspend(now)
            cmd = Command.NONE
            evidence_state = "input_held"  # the suspended brain keeps a stale state; no key can be sent
        else:
            p = Perception(w, h, self.player.pos if self.player else None, self.global_pos, self.minimap_pos,
                           img_mm.shape[1] if img_mm is not None else None, self.on_ladder, dets, now, self.frame_idx)
            cmd = self.brain.decide(p)
            fallback_mp = self.cfg["attack"].get("mp_fallback_percent")
            if (cmd.action in ("attack", "aoe") and mp is not None and fallback_mp is not None
                    and str(self.cfg["keys"].get("normal_attack") or "").strip()
                    and mp < float(fallback_mp)):
                cmd = cmd.replace(action="normal_attack")
        if nav is not None and getattr(nav, "recovery_terminal", False):
            self.paused = True
            self._return_route_active = self._minimap_route_active = False
        if self.keyboard is not None:
            with self._control_lock:
                if self.full_pause:
                    self.paused = True
                    cmd = Command.NONE
                recovery_input = bool(self._return_route_active or (nav is not None
                                      and getattr(nav, "returning_to_start", False)))
                terminal_block = bool(nav is not None and (getattr(nav, "recovery_terminal", False)
                    or (nav.state == "blocked" and not getattr(nav, "returning_to_start", False))))
                self.keyboard.set_command(cmd, source_time=self.capture_time,
                                          navigation_only=(self._minimap_route_active or self._vertical_minimap_active
                                                           or recovery_input or terminal_block))
                self.keyboard.set_enabled(not self.paused and not self.full_pause and foreground and not terminal_block
                                          and (not self._input_blocked or self._vertical_minimap_active
                                               or self._minimap_route_active or self._return_route_active))

        self._observe_evidence(frame, now, self.capture_state == "fresh" and self.map_ready and not self.paused,
                               evidence_state)
        if self._t_last is not None and now > self._t_last:
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / (now - self._t_last)) if self.fps else 1.0 / (now - self._t_last)
        self._t_last = now

        snap = BotSnapshot(frame, self.player.pos if self.player else None, self.minimap_box, self.minimap_pos,
                           self.global_pos, dets, cmd, self.brain.state, hp, mp, exp, self.fps,
                           self.follower.idx if self.follower else None, sb, self.brain.target, other,
                           self.paused, self.route_map, self.evidence_snapshot())
        self.last_snapshot = snap
        return snap

    def toggle_pause(self) -> None:
        if self.paused and (self.full_pause or not (self._route_health_hold or self._hp_health_hold)):
            self.resume()
        else:
            self.pause_all()
        self._minimap_route_active = False
        self._return_route_active = False
        if self.paused and self._vertical_minimap_active:
            self._stop_vertical_minimap()
        if self.keyboard is not None:
            self.keyboard.set_enabled(False)  # a resumed bot must receive a new capture before input
        log.info("%s", "paused" if self.paused else "resumed")

    def screenshot(self, out_dir: str = "screenshot") -> str | None:
        snap = self.last_snapshot
        if snap is None:
            return None
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(out_dir, f"{stamp}.png")
        cv2.imwrite(path, snap.frame)
        log.info("screenshot saved to %s", path)
        return path

    def close(self) -> None:
        if self.keyboard is not None:
            self.keyboard.stop()
        self.source.stop()

    def run(self, show: bool = True, max_frames: int | None = None) -> BotSnapshot | None:
        from maplebot.overlay import draw_debug

        self.source.start()
        if self.keyboard is not None:
            self.keyboard.start()
        hot = None
        try:
            from maplebot.control.hotkeys import Hotkeys
            hot = Hotkeys({"f1": self.toggle_pause, "f2": self.screenshot, "f12": self.stop_request},
                          float(self.cfg["system"].get("debounce_s", 1.0)))
            hot.start()
        except Exception as e:
            log.warning("hotkeys unavailable: %s", e)
        self._stop = False
        period = 1.0 / max(1.0, float(self.cfg["system"]["fps_main"]))
        snap = None
        try:
            while not self._stop:
                t0 = time.monotonic()
                snap = self.step(t0)
                if snap is None:
                    if self.capture_state != "duplicate":
                        log.warning("no fresh frame from %s", getattr(self.source, "title", "source"))
                    time.sleep(min(0.1, period))
                    continue
                if show:
                    img = draw_debug(snap.frame, snap, self.cfg)
                    scale = float(self.cfg.get("debug", {}).get("scale", 1.0))
                    if scale != 1.0:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("maplebot", img)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                if max_frames is not None and self.frame_idx >= max_frames:
                    break
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            if hot is not None:
                hot.stop()
            self.close()
            if show:
                cv2.destroyAllWindows()
        return snap

    def stop_request(self) -> None:
        self.brain.loot.suspend(time.monotonic())
        self._stop = True


def map_profile_root():
    """The folder whose config/map_profiles holds the saved profiles: the checkout in source
    mode, the folder beside Maplebot.exe when frozen (never _internal)."""
    from maplebot.paths import app_root
    return app_root()


def load_map_recognizer(cfg: dict):
    from maplebot.map_profiles import MapProfileRecognizer, load_profiles
    options = cfg.get("map_profile", {})
    profiles = load_profiles(map_profile_root(), directory=options.get("directory"))
    recognition = options.get("recognition", {})
    return MapProfileRecognizer(profiles, **recognition)


def build_bot(cfg: dict, source_path: str | None, no_input: bool) -> Bot:
    """Wire a Bot from config: source (window / image / video), keyboard, monsters, route."""
    if source_path:
        ext = os.path.splitext(source_path)[1].lower()
        source = VideoSource(source_path) if ext in (".mp4", ".avi", ".mkv", ".mov") else ImageSource(source_path)
    else:
        from maplebot.capture.window import WindowCapture
        source = WindowCapture(cfg)
    backend = make_backend("null" if no_input else "pynput")
    is_active = None
    if not source_path:
        from maplebot.capture.window import is_foreground
        title = source.title
        is_active = lambda: is_foreground(title)  # noqa: E731  keys only go to the game window
    keyboard = KeyboardController(cfg, backend, is_active=is_active)

    mcfg = cfg["monsters"]
    per = mcfg.get("per_monster") or {}
    scales_by_name = {n: v["scales"] for n, v in per.items() if isinstance(v, dict) and v.get("scales")}
    root = str(mcfg.get("dir", "monster"))
    try:
        library = MonsterLibrary.load(root, list(mcfg.get("names") or []), scales_by_name, list(mcfg.get("scales") or [1.0]))
    except (FileNotFoundError, NotADirectoryError) as e:
        log.warning("no monster templates loaded (%s)", e)
        library = MonsterLibrary()
    if not library.templates:
        log.warning("monster library is empty: the bot will not attack anything")
    else:
        log.info("monsters: %s (%d templates)", library.names, len(library.templates))

    route_map = None
    rcfg = cfg["route"]
    if rcfg.get("map"):
        folder = os.path.join(str(rcfg.get("dir", "minimaps")), str(rcfg["map"]))
        route_map = RouteMap.load(folder, parse_color_codes(rcfg["color_code"]), parse_color_codes(rcfg["color_code_up_down"]))
        log.info("route %s: %d route(s)", folder, route_map.n_routes)
    else:
        if cfg.get("route", {}).get("circuit", {}).get("enabled"):
            log.info("minimap farming circuit configured")
        else:
            log.info("no route configured: sweep search mode")
    bot = Bot(cfg, source, keyboard, library, route_map)
    if not source_path:
        bot.map_recognizer = load_map_recognizer(cfg)
        bot.map_ready = False
        bot.pause_all()
    return bot
