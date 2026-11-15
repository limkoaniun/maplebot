"""Bounded, temporary movement tests requested by the local map editor."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import re
import time

import cv2

from maplebot.editor_bridge import capture_signature, map_similarity
from maplebot.nav.circuit import MinimapCircuit, Platform


TERMINAL = {"completed", "blocked", "cancelled", "error"}


class EditorTestSession:
    def __init__(self, request: dict, normal_cfg: dict, log_dir: Path, *, wall_now=None, now=None):
        wall_now = time.time() if wall_now is None else wall_now
        self.started = time.monotonic() if now is None else now
        if not isinstance(request, dict) or request.get("version") != 1:
            raise ValueError("Unsupported connection-test request.")
        self.test_id = request.get("test_id")
        if not isinstance(self.test_id, str) or re.fullmatch(r"[0-9a-f]{32}", self.test_id) is None:
            raise ValueError("The connection-test identifier is invalid.")
        created = float(request.get("created", -1))
        if not math.isfinite(created) or not 0 <= wall_now - created <= 120:
            raise ValueError("The connection-test request expired; queue it again.")
        filename = f"editor-test-{self.test_id}.png"
        if request.get("reference") != filename:
            raise ValueError("The connection-test reference filename is invalid.")
        reference_path = Path(log_dir) / filename
        if reference_path.resolve().parent != Path(log_dir).resolve():
            raise ValueError("The connection-test reference must stay in the log directory.")
        self.reference = cv2.imread(str(reference_path))
        if self.reference is None or min(self.reference.shape[:2]) < 8:
            raise ValueError("The connection-test minimap image is missing or invalid.")
        self.capture = request.get("capture")
        if self.capture != capture_signature(normal_cfg):
            raise ValueError("Capture settings changed; capture the current map before testing.")
        self.source = request.get("source_platform")
        if not isinstance(self.source, dict):
            raise ValueError("The connection-test starting platform is missing.")
        try:
            source_knots = self.source.get("knots", ())
            source_fields = ("left", "right", "y", "right_y") if "right_y" in self.source else ("left", "right", "y")
            self.source = {k: float(self.source[k]) for k in source_fields}
            support = Platform(**self.source, knots=source_knots)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("The starting platform has invalid coordinates.") from error
        height, width = self.reference.shape[:2]
        if (not all(math.isfinite(v) for v in self.source.values())
                or not 0 <= self.source["left"] < self.source["right"] < width
                or not 0 <= self.source["y"] < height
                or not 0 <= self.source.get("right_y", self.source["y"]) < height
                or any(not 0 <= y < height for _, y in support.knots)):
            raise ValueError("The starting platform is outside the reference minimap.")
        if support.knots:
            self.source["knots"] = [list(point) for point in support.knots]
        if not isinstance(request.get("circuit"), dict):
            raise ValueError("The connection-test route is missing.")
        circuit = deepcopy(request["circuit"])
        circuit.update(enabled=True, loop=False, movement_test=True, start_idx=0, max_retries=1)
        circuit["timeout_s"] = min(15.0, max(1.0, float(circuit.get("timeout_s", 15))))
        nav = MinimapCircuit(circuit)
        for point in nav.waypoints:
            if not 0 <= point.x < width or not 0 <= point.y < height:
                raise ValueError("A test waypoint is outside the reference minimap.")
            if point.action == "jump" and not (0 <= point.from_x < width and 0 <= point.from_y < height):
                raise ValueError("A jump takeoff is outside the reference minimap.")
        self.cfg = deepcopy(normal_cfg)
        self.cfg.setdefault("route", {}).update(map="", circuit=circuit)
        self.cfg.setdefault("health", {})["enable"] = False
        self.cfg["buffs"] = {"keys": [], "cooldowns_s": []}
        self.cfg.setdefault("loot", {})["enabled"] = False
        for key in ("attack", "aoe", "normal_attack", "pickup", "hp_potion", "mp_potion"):
            self.cfg.setdefault("keys", {})[key] = ""
        self.state = "queued"
        self.message = "Checking a fresh game image before moving."
        self.preflight_done = False
        self.has_run = False
        self.waiting_since = None

    @property
    def terminal(self):
        return self.state in TERMINAL

    def finish(self, state: str, message: str):
        self.state, self.message = state, message

    def status(self, wall_now=None):
        return {"test_id": self.test_id, "state": self.state, "message": self.message,
                "updated": time.time() if wall_now is None else wall_now}

    def before_step(self, now: float, foreground: bool) -> bool:
        """Return whether this frame may run, before any bot decision can emit input."""
        if self.terminal:
            return False
        if now - self.started >= 90:
            self.finish("blocked", "The connection test reached its 90-second limit.")
            return False
        if not foreground:
            if self.waiting_since is None:
                self.waiting_since = now
            if now - self.waiting_since >= 60:
                self.finish("blocked", "The test waited 60 seconds for MapleStory to be in front.")
                return False
            self.state = "waiting_focus"
            self.message = "Switch to MapleStory to continue the movement-only test."
            return False
        if self.waiting_since is not None and now - self.waiting_since >= 60:
            self.finish("blocked", "The test waited 60 seconds for MapleStory to be in front.")
            return False
        allow = self.state == "running" and self.preflight_done
        if allow:
            self.has_run = True
        return allow

    def _check_frame(self, snap, cfg, minimap_seen: bool, require_source: bool, minimap_offset=None):
        if capture_signature(cfg) != self.capture:
            raise ValueError("Capture settings changed during the test.")
        if snap is None or snap.minimap_box is None or not minimap_seen or snap.minimap_pos is None:
            raise ValueError("The current minimap or character dot is not visible.")
        box = snap.minimap_box.clamp(snap.frame.shape[1], snap.frame.shape[0])
        crop = snap.frame[box.y:box.y2, box.x:box.x2]
        if crop.shape[:2] != self.reference.shape[:2]:
            raise ValueError("The current map does not match the test's reference minimap.")
        if minimap_offset is not None:
            from maplebot.map_profiles import align_map
            if (not isinstance(minimap_offset, (list, tuple)) or len(minimap_offset) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) and int(v) == v for v in minimap_offset)):
                raise ValueError("The minimap alignment is not confirmed for this test.")
            score, offset, _ = align_map(self.reference, crop, cfg.get("minimap", {}))
            threshold = float(cfg.get("map_profile", {}).get("recognition", {}).get("threshold", .78))
            if not math.isfinite(score) or score < threshold or tuple(offset) != tuple(minimap_offset):
                raise ValueError("The current map or its alignment differs from the test's reference minimap.")
        elif cfg.get("map_profile", {}).get("active"):
            raise ValueError("The minimap alignment must be confirmed before testing a saved profile.")
        elif map_similarity(self.reference, crop) < .55:
            raise ValueError("The current map does not match the test's reference minimap.")
        if require_source:
            from maplebot.nav.editor import platform_y
            x, y = snap.minimap_pos
            if (abs(y - platform_y(self.source, x)) > 2
                    or not self.source["left"] - 1 <= x <= self.source["right"] + 1):
                raise ValueError("Move onto the connection's starting platform and queue the test again.")

    def after_step(self, snap, cfg, *, now: float, foreground: bool, minimap_seen: bool, circuit,
                   minimap_offset=None) -> bool:
        if self.terminal:
            return False
        if circuit is not None and circuit.completed:
            self.finish("completed", "Connection test completed. The normal bot is paused.")
            return False
        if circuit is not None and circuit.state == "blocked":
            reason = getattr(circuit, "reason", None) or "Movement could not complete this connection"
            self.finish("blocked", f"{reason}. The normal bot is paused.")
            return False
        if snap is None:
            if self.has_run:
                self.finish("blocked", "Game capture was lost during the test. The normal bot is paused.")
            return False  # capture loss already releases bot input; the overall deadline still applies
        if not self.preflight_done or self.state != "running":
            try:
                self._check_frame(snap, cfg, minimap_seen, require_source=not self.has_run,
                                  minimap_offset=minimap_offset)
            except ValueError as error:
                self.finish("error", str(error))
                return False
            self.preflight_done = True
        if not foreground:
            return False
        self.waiting_since = None
        self.state = "running"
        self.message = "Testing this connection with combat, potions, buffs, and pickup disabled."
        return True
