"""Local profile storage and the visual editor's connection to the paused runner."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

import cv2
import numpy as np
import yaml

from maplebot.config import deep_merge, load_config
from maplebot.map_profiles import (MapProfile, apply_profile, capture_compatible,
    map_settings_from_config, normalize_map_settings, profile_key, read_profile, validate_map_settings)


def capture_signature(cfg: dict) -> dict:
    game = cfg.get("game_window", {})
    return {"window_title": game.get("title", ""),
            "working_size": list(game.get("working_size") or []),
            "minimap_rect": list(cfg.get("minimap", {}).get("fixed_rect") or []),
            "title_bar_px": int(game.get("title_bar_px", 0) or 0)}


def map_similarity(reference: np.ndarray, current: np.ndarray) -> float:
    """Compare map structure, allowing small moving dots and brightness changes."""
    if reference.shape[:2] != current.shape[:2] or min(reference.shape[:2]) < 8:
        return 0.0
    def edges(image):
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        return cv2.magnitude(dx, dy)
    first, second = edges(reference), edges(current)
    if min(float(first.std()), float(second.std())) < 1e-4:
        return float(float(np.mean(cv2.absdiff(reference, current))) < 2.0)
    score = float(cv2.matchTemplate(second, first, cv2.TM_CCOEFF_NORMED)[0, 0])
    return max(0.0, min(1.0, score)) if np.isfinite(score) else 0.0


def _read_json(path: Path, default=None):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return value
    except (OSError, ValueError):
        pass
    return {} if default is None else default


def atomic_bytes(path: Path, value: bytes) -> None:
    """Replace a local file, tolerating brief Windows reader locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(value)
        for attempt in range(4):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(0.01 * 2 ** attempt)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def _fresh(value, seconds=6) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and -1 <= time.time() - value <= seconds)


def _position(value, size):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and math.isfinite(v) for v in value)
            or not 0 <= value[0] < size[0] or not 0 <= value[1] < size[1]):
        return None
    return tuple(value)


class EditorBridge:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.log = self.root / "log"
        self.profiles = self.root / "config/map_profiles"
        self.user_config = self.root / "config/user.yaml"
        self.log.mkdir(parents=True, exist_ok=True)

    def _cfg(self):
        return load_config(str(self.user_config))

    def _status(self):
        return _read_json(self.log / "active-status.json")

    def pause(self):
        (self.log / "resume-active").unlink(missing_ok=True)
        (self.log / "return-to-start-active").unlink(missing_ok=True)
        (self.log / "hold-farming-active").unlink(missing_ok=True)
        (self.log / "pause-active").touch()

    def stop_test(self):
        (self.log / "stop-editor-test").touch()
        (self.log / "editor-test-request.json").unlink(missing_ok=True)
        self.pause()

    def test_status(self):
        return _read_json(self.log / "editor-test-status.json", {"state": "idle", "message": ""})

    def return_to_start(self):
        """Queue one explicit recovery request; only the runner can authorize motion."""
        status = self._status()
        if not status.get("running") or not _fresh(status.get("updated"), seconds=3):
            raise ValueError("Start the controller and wait for a current game image before returning to point 1.")
        if status.get("editor_test") or (self.log / "editor-test-request.json").exists():
            raise ValueError("Stop the movement test before returning to point 1.")
        if status.get("returning_to_start") or (self.log / "return-to-start-active").exists():
            raise ValueError("A return to point 1 is already running or waiting to start.")
        cfg = self._cfg()
        if not (cfg.get("route", {}).get("circuit", {}).get("enabled")
                and status.get("circuit_enabled", True)):
            raise ValueError("Apply an enabled map circuit before returning to point 1.")
        if (not status.get("map_ready") or not status.get("minimap_seen")
                or status.get("capture") != capture_signature(cfg)):
            raise ValueError("Wait for the saved map and your minimap position to be recognized, then try again.")
        atomic_json(self.log / "return-to-start-active", {"created": time.time(), "source": "settings"})
        return "Return requested. The controller will check your position first. Bring MapleStory to the front to move."

    def hold_farming(self):
        """Request stationary HP protection without resuming farming or changing settings."""
        status = self._status()
        if not status.get("running") or not _fresh(status.get("updated"), seconds=3):
            raise ValueError("Start the controller and wait for a current game image before holding farming.")
        potion = _read_json(self.log / "potion-verification.json")
        potion_state = str(potion.get("state") or "").lower()
        potion_running = (_fresh(potion.get("updated"), seconds=30) and potion_state
                          and potion_state not in {"passed", "failed", "completed", "cancelled", "error", "idle", "stopped"})
        if (status.get("editor_test") or status.get("potion_test")
                or potion_running
                or (self.log / "editor-test-request.json").exists()
                or (self.log / "potion-test-request.json").exists()):
            raise ValueError("Finish the movement or potion test before holding farming.")
        cfg = self._cfg()
        if not cfg.get("health", {}).get("enable", True) or not str(cfg.get("keys", {}).get("hp_potion") or "").strip():
            raise ValueError("Enable health readings and assign an HP potion key before holding farming.")
        if (not status.get("map_ready") or not status.get("minimap_seen")
                or status.get("capture") != capture_signature(cfg)):
            raise ValueError("Wait for the saved map and your minimap position to be recognized, then try again.")
        startup_running = status.get("startup_return_state") in {"checking", "returning"}
        circuit = cfg.get("route", {}).get("circuit", {})
        terminal_route = (status.get("recovery_terminal")
                          or (status.get("circuit_state") == "blocked"
                              and not status.get("returning_to_start")
                              and not circuit.get("return_on_block", False)))
        permanent_hold = (status.get("full_pause") is False and not startup_running
                          and (status.get("manual_health_hold")
                               or (terminal_route and cfg.get("health", {}).get("protect_on_route_failure", True))))
        if permanent_hold:
            return "Farming is already held. HP protection remains guarded; no new request was sent."
        if (self.log / "hold-farming-active").exists():
            return "Hold already requested. Waiting for the controller to confirm the farming hold."
        # A normal refill or startup hold can resume by itself. Queue an explicit
        # permanent hold even while its paused/HP-only flags are already set.
        atomic_json(self.log / "hold-farming-active", {"created": time.time(), "source": "settings"})
        return "Hold requested. Movement and combat will stop; guarded HP protection stays on. Pause all stops every input."

    def _raw_snapshot(self, cfg=None):
        cfg = self._cfg() if cfg is None else cfg
        signature = capture_signature(cfg)
        try:
            prepared, metadata = self.full_snapshot(cfg)
            x, y, w, h = map(int, signature["minimap_rect"])
            if x >= 0 and y >= 0 and w >= 8 and h >= 8 and x+w <= prepared.shape[1] and y+h <= prepared.shape[0]:
                frame = prepared[y:y+h, x:x+w].copy()
                from maplebot.vision.minimap import find_player_dot
                minimap = cfg.get("minimap", {})
                player = find_player_dot(frame, minimap.get("player_color_bgr", [0, 255, 255]),
                                         int(minimap.get("player_tolerance", 20)))
                return frame, dict(metadata, capture=signature, size=[w, h], player_pos=player,
                                   coordinate_space="raw")
        except (ValueError, TypeError):
            pass
        for prefix in ("editor-minimap", "relay-editor-minimap"):
            metadata = _read_json(self.log / f"{prefix}.json")
            if not _fresh(metadata.get("updated")) or metadata.get("capture") != signature:
                continue
            if metadata.get("coordinate_space", "raw") != "raw":
                continue  # never align already-transformed pixels/dots a second time
            frame = cv2.imread(str(self.log / f"{prefix}.png"))
            if frame is not None and list(frame.shape[1::-1]) == metadata.get("size"):
                return frame, dict(metadata, coordinate_space="raw")
        raise ValueError("No fresh minimap matches the current capture settings. Keep MapleStory open with its minimap visible.")

    def _snapshot(self, cfg=None):
        """Use the applied profile's coordinates only after a fresh terrain match."""
        cfg = self._cfg() if cfg is None else cfg
        raw = self._raw_snapshot(cfg)
        circuit = cfg.get("route", {}).get("circuit", {})
        key = cfg.get("map_profile", {}).get("active")
        if not key or key != circuit.get("editor_profile"):
            return raw
        try:
            plan, reference, digest = self._load_saved(key)
            if (cfg.get("map_profile", {}).get("reference_sha256") != digest
                    or circuit.get("editor_reference_sha256") != digest):
                return raw
            return self._align_snapshot(plan, reference, cfg, raw, digest=digest)
        except ValueError:
            return raw  # load_current will show a new empty drawing on this raw crop

    def _align_snapshot(self, plan, reference, cfg, snapshot, *, digest=None):
        from maplebot.map_profiles import align_map
        current, metadata = snapshot
        signature = capture_signature(cfg)
        if not _fresh(metadata.get("updated")) or metadata.get("capture") != signature:
            raise ValueError("The minimap capture is stale or uses different game settings.")
        if plan.get("capture") != signature:
            raise ValueError("The game or minimap settings differ from this profile. Capture the current map before applying it.")
        if (list(plan.get("image_size", [])) != list(current.shape[1::-1])
                or reference.shape != current.shape):
            raise ValueError("This profile has a different minimap size.")
        score, offset, margin = align_map(reference, current, cfg.get("minimap", {}))
        threshold = float(cfg.get("map_profile", {}).get("recognition", {}).get("threshold", .78))
        if not np.isfinite(score) or score < threshold:
            raise ValueError("The current minimap differs from this profile, or its terrain alignment is uncertain. Check the map before applying or testing it.")
        raw_player = _position(metadata.get("player_pos"), current.shape[1::-1])
        player = None if raw_player is None else _position(
            (raw_player[0] + offset[0], raw_player[1] + offset[1]), reference.shape[1::-1])
        return reference.copy(), dict(metadata, coordinate_space="profile",
            profile_key=self._profile_key(plan["name"]), reference_sha256=digest,
            minimap_offset=list(offset), raw_player_pos=raw_player, player_pos=player,
            alignment_score=score, alignment_margin=margin)

    def full_snapshot(self, cfg=None):
        cfg = self._cfg() if cfg is None else cfg
        candidates = []
        for prefix in ("controller-prepared", "relay-prepared"):
            metadata = _read_json(self.log / f"{prefix}.json")
            if not _fresh(metadata.get("updated")):
                continue
            candidates.append((metadata["updated"], prefix, metadata))
        for _, prefix, metadata in sorted(candidates, reverse=True):
            frame = cv2.imread(str(self.log / f"{prefix}.png"))
            if frame is not None and capture_compatible(metadata.get("capture", {}), cfg, frame.shape):
                return frame, metadata
        raise ValueError("No fresh full game image is available. Keep MapleStory open, then try again.")

    def settings_for_plan(self, plan):
        path = self.profiles / f"{self._profile_key(plan['name'])}.json"
        if path.exists():
            saved = _read_json(path)
            if isinstance(saved.get("map_settings"), dict):
                return normalize_map_settings(saved["map_settings"])
            # A legacy profile can infer its own crop, never today's monsters.
            return normalize_map_settings({"minimap": {"fixed_rect": list(plan.get("capture", {}).get("minimap_rect", []))},
                    "monsters": {"names": []}, "circuit_settings": {}})
        settings = map_settings_from_config(self._cfg())
        settings["minimap"]["fixed_rect"] = list(plan.get("capture", {}).get("minimap_rect", []))
        settings["monsters"]["names"] = []
        return settings

    def load_current(self):
        cfg = self._cfg()
        frame, metadata = self._snapshot(cfg)
        circuit = cfg.get("route", {}).get("circuit", {})
        stored = circuit.get("editor_plan")
        if stored:
            try:
                if not isinstance(stored, dict):
                    raise ValueError("The applied profile could not be read.")
                plan = deepcopy(stored)
                key = self._profile_key(plan.get("name"))
                if circuit.get("editor_profile") != key:
                    raise ValueError("The applied profile reference is missing.")
                saved, reference, digest = self._load_saved(key)
                expected_digest = circuit.get("editor_reference_sha256")
                if not expected_digest or digest != expected_digest:
                    raise ValueError("The applied profile's reference image changed.")
                if saved.get("capture") != plan.get("capture"):
                    raise ValueError("The saved and applied profiles have different capture settings.")
                frame, metadata = self._check_context(plan, reference, cfg, snapshot=(frame, metadata))
            except ValueError as error:
                frame, metadata = self._raw_snapshot(cfg)
                plan = {"version": 1, "name": "New map", "image_size": list(frame.shape[1::-1]),
                        "platforms": [], "connections": [], "route": [], "loop": True,
                        "capture": capture_signature(cfg),
                        "context_warning": f"{error} A new empty map is shown; the saved route has not changed."}
        else:
            frame, metadata = self._raw_snapshot(cfg)
            plan = {"version": 1, "name": "New map", "image_size": list(frame.shape[1::-1]),
                    "platforms": [], "connections": [], "route": [], "loop": True,
                    "capture": capture_signature(cfg),
                    "context_warning": "No verified map profile is applied. Calibrate the minimap and select this map's monsters before drawing a route."}
        player = _position(metadata.get("player_pos"), frame.shape[1::-1])
        return plan, frame, player

    @staticmethod
    def _profile_key(name):
        return profile_key(name)

    def list_profiles(self):
        names = []
        if self.profiles.exists():
            for path in self.profiles.glob("*.json"):
                data = _read_json(path)
                stored = data.get("plan")
                name = stored.get("name") if isinstance(stored, dict) else None
                if isinstance(name, str) and name:
                    names.append(name)
        return sorted(set(names), key=str.casefold)

    def load_profile(self, name):
        key = self._profile_key(name)
        plan, frame, _ = self._load_saved(key)
        player = None
        try:
            current, metadata = self._check_context(plan, frame, self._cfg())
            player = _position(metadata.get("player_pos"), current.shape[1::-1])
        except ValueError:
            pass
        return deepcopy(plan), frame, player

    def _load_saved(self, key):
        from maplebot.nav.editor import validate_plan
        data = _read_json(self.profiles / f"{key}.json")
        plan = data.get("plan")
        if not isinstance(plan, dict):
            raise ValueError("That map profile could not be read.")
        issues = validate_plan(plan, require_route=False)
        if issues:
            raise ValueError("The saved map profile is invalid: " + "; ".join(issues))
        if self._profile_key(plan.get("name")) != key:
            raise ValueError("The map profile name does not match its saved reference.")
        image_path = self.profiles / f"{key}.png"
        try:
            encoded = image_path.read_bytes()
        except OSError as error:
            raise ValueError("The saved map image is missing.") from error
        digest = hashlib.sha256(encoded).hexdigest()
        if data.get("image_sha256") and data["image_sha256"] != digest:
            raise ValueError("The saved map image changed independently of its profile.")
        frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if frame is None or list(frame.shape[1::-1]) != list(plan.get("image_size", [])):
            raise ValueError("The saved map image is missing or has the wrong size.")
        return plan, frame, digest

    def _check_context(self, plan, reference, cfg, snapshot=None):
        current, metadata = self._snapshot(cfg) if snapshot is None else snapshot
        signature = capture_signature(cfg)
        if not _fresh(metadata.get("updated")) or metadata.get("capture") != signature:
            raise ValueError("The minimap capture is stale or uses different game settings.")
        expected = plan.get("capture")
        if expected != signature:
            raise ValueError("The game or minimap settings differ from this profile. Capture the current map before applying it.")
        if metadata.get("coordinate_space") == "profile":
            if metadata.get("profile_key") == self._profile_key(plan["name"]):
                if not np.array_equal(reference, current):
                    raise ValueError("The drawing's reference image differs from the saved coordinate system. Reopen the saved profile before applying or testing it.")
                return current, metadata
            # A selected different profile needs its own fresh raw alignment.
            current, metadata = self._raw_snapshot(cfg)
        return self._align_snapshot(plan, reference, cfg, (current, metadata))

    def save_profile(self, plan, image_bgr, apply=True, map_settings=None):
        from maplebot.nav.editor import compile_plan, validate_plan, platform_y
        plan = deepcopy(plan)
        issues = validate_plan(plan, require_route=apply)
        if issues:
            raise ValueError("\n".join(issues))
        if list(plan.get("image_size", [])) != list(image_bgr.shape[1::-1]):
            raise ValueError("The drawing and reference minimap have different sizes.")
        cfg = self._cfg()
        key = self._profile_key(plan.get("name"))
        if map_settings is None:
            existing = _read_json(self.profiles / f"{key}.json")
            if existing:
                map_settings = deepcopy(existing.get("map_settings"))
            else:
                map_settings = map_settings_from_config(cfg)
        map_settings = normalize_map_settings(map_settings)
        if apply:
            issues = validate_map_settings(map_settings, plan.get("capture"), plan.get("image_size"))
            if issues:
                raise ValueError("\n".join(issues))
        if apply:
            context_cfg = deepcopy(cfg)
            context_cfg["minimap"] = deepcopy(map_settings["minimap"])
            self._check_context(plan, image_bgr, context_cfg)
        self.profiles.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("Could not save the reference minimap.")
        image_path = self.profiles / f"{key}.png"
        encoded = encoded.tobytes()
        digest = hashlib.sha256(encoded).hexdigest()
        atomic_bytes(image_path, encoded)
        atomic_json(self.profiles / f"{key}.json", {"version": 2, "updated": time.time(),
                    "plan": plan, "map_settings": map_settings, "image_sha256": digest})
        if not apply:
            return "Draft saved. The bot's route has not changed."
        data = yaml.safe_load(self.user_config.read_text(encoding="utf-8")) or {}
        profile = MapProfile(key, plan["name"], plan, image_bgr, map_settings, plan["capture"], digest)
        data = apply_profile(data, profile)
        self.stop_test()
        atomic_bytes(self.user_config, ("# Local Maplebot settings\n" + yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True)).encode("utf-8"))
        if self._status().get("running"):
            (self.log / "reload-active").touch()
        return "Profile saved and route applied. The bot stays paused until you resume it."

    def select_profile(self, name):
        """Apply a saved map's own crop/settings after checking the full frame."""
        key = self._profile_key(name)
        profile = read_profile(self.profiles / f"{key}.json")
        cfg = apply_profile(self._cfg(), profile)
        prepared, metadata = self.full_snapshot(cfg)
        from maplebot.map_profiles import crop_for_profile
        crop = crop_for_profile(prepared, profile, cfg)
        if crop is None:
            raise ValueError("The saved minimap crop does not fit the current game image.")
        self._check_context(profile.plan, profile.reference_bgr, cfg,
                            snapshot=(crop, dict(metadata, capture=capture_signature(cfg))))
        data = yaml.safe_load(self.user_config.read_text(encoding="utf-8")) or {}
        data = apply_profile(data, profile)
        self.stop_test()
        atomic_bytes(self.user_config, ("# Local Maplebot settings\n" + yaml.safe_dump(
            data, sort_keys=False, allow_unicode=True)).encode("utf-8"))
        if self._status().get("running"):
            (self.log / "reload-active").touch()
        return f"Loaded {profile.name}, including its minimap, monsters, and route. The bot stays paused."

    def sync_active_settings(self, user_data):
        """Keep profile-owned tuning consistent with ordinary settings saves."""
        key = user_data.get("map_profile", {}).get("active")
        if not key:
            return
        profile = read_profile(self.profiles / f"{key}.json")
        merged = deep_merge(load_config(), user_data)
        settings = map_settings_from_config(merged)
        issues = validate_map_settings(settings, profile.capture, profile.plan["image_size"])
        if issues:
            raise ValueError("\n".join(issues))
        data = _read_json(self.profiles / f"{key}.json")
        data.update(version=2, map_settings=settings, updated=time.time())
        atomic_json(self.profiles / f"{key}.json", data)

    def calibrate_crop(self, rect, prepared, metadata):
        """Save a user-drawn crop, invalidate the old route, and return a blank map."""
        cfg = self._cfg()
        if not _fresh(metadata.get("updated"), seconds=120) or not capture_compatible(metadata.get("capture", {}), cfg, prepared.shape):
            raise ValueError("The calibration image expired or the game size changed. Capture it again.")
        if (not isinstance(rect, (list, tuple)) or len(rect) != 4
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in rect)):
            raise ValueError("Draw a rectangular minimap crop.")
        x, y, w, h = rect
        if x < 0 or y < 0 or w < 8 or h < 8 or x+w > prepared.shape[1] or y+h > prepared.shape[0]:
            raise ValueError("The minimap crop must stay inside the game image and be at least 8 pixels wide and high.")
        data = yaml.safe_load(self.user_config.read_text(encoding="utf-8")) or {}
        data.setdefault("minimap", {})["fixed_rect"] = list(rect)
        data.setdefault("monsters", {})["names"] = []
        data.setdefault("route", {}).update(map="", circuit={"enabled": False, "waypoints": [], "platforms": []})
        identity = data.setdefault("map_profile", {})
        for key in ("active", "name", "reference_sha256"):
            identity.pop(key, None)
        self.stop_test()
        atomic_bytes(self.user_config, ("# Local Maplebot settings\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)).encode("utf-8"))
        if self._status().get("running"):
            (self.log / "reload-active").touch()
        crop = prepared[y:y+h, x:x+w].copy()
        plan = {"version": 1, "name": "New map", "image_size": [w, h], "platforms": [],
                "connections": [], "route": [], "loop": True, "capture": capture_signature(deep_merge(cfg, data)),
                "context_warning": "New minimap crop calibrated. Select this map's monsters and draw its platforms; the previous route is disabled."}
        return plan, crop, None

    def test_connection(self, plan, connection_id, image_bgr):
        from maplebot.nav.editor import compile_plan, validate_plan, platform_y
        status = self._status()
        if not status.get("running") or not _fresh(status.get("updated")):
            raise ValueError("Start the controller in paused mode before testing a connection.")
        if not status.get("paused"):
            raise ValueError("Pause the controller before testing a connection.")
        if (self.log / "editor-test-request.json").exists():
            raise ValueError("A connection test is already queued. Stop it before queuing another.")
        issues = validate_plan(plan, require_route=False)
        if issues:
            raise ValueError("\n".join(issues))
        cfg = self._cfg()
        _, metadata = self._check_context(plan, image_bgr, cfg)
        if status.get("capture") != capture_signature(cfg):
            raise ValueError("The controller has not loaded the current capture settings. Reload it before testing.")
        connection = next((c for c in plan["connections"] if c["id"] == connection_id), None)
        if connection is None:
            raise ValueError("Select a rope, jump, or drop to test.")
        platforms = {p["id"]: p for p in plan["platforms"]}
        source = platforms[connection["from_platform"]]
        pos = _position(metadata.get("player_pos"), plan["image_size"])
        if (pos is None or abs(pos[1] - platform_y(source, pos[0])) > 2
                or not source["left"] - 1 <= pos[0] <= source["right"] + 1):
            raise ValueError("Move your character onto the connection's starting platform, pause, then test again.")
        destination = platforms[connection["to_platform"]]
        landing_x = connection["to_x"]
        center = (destination["left"] + destination["right"]) / 2
        direction = (1 if landing_x > connection["from_x"] else -1) if connection["kind"] == "ramp" else (1 if landing_x <= center else -1)
        probe_x = max(destination["left"], min(destination["right"], landing_x + 4 * direction))
        if abs(probe_x - landing_x) < 3 and connection["kind"] != "ramp":
            probe_x = max(destination["left"], min(destination["right"], landing_x - 4 * direction))
        probe_length = abs(probe_x - landing_x)
        if probe_length < 3:
            raise ValueError("The destination platform is too narrow to verify landing; at least 3 minimap pixels of walking room are required.")
        single = deepcopy(plan)
        single["connections"] = [deepcopy(connection)]
        single["route"] = [{"platform_id": connection["from_platform"], "x": connection["from_x"], "dwell_s": 0},
                           {"platform_id": connection["to_platform"], "x": landing_x},
                           {"platform_id": connection["to_platform"], "x": probe_x, "dwell_s": 0.6}]
        single["loop"] = False
        circuit = compile_plan(single, cfg.get("route", {}).get("circuit", {}))
        circuit.update(loop=False, movement_test=True, start_idx=0, max_retries=1, timeout_s=15)
        circuit["tolerance_y"] = 0.75
        circuit["tolerance_x"] = min(float(circuit.get("tolerance_x", 2)), probe_length - 2)
        test_id = uuid.uuid4().hex
        reference_path = self.log / f"editor-test-{test_id}.png"
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            raise ValueError("Could not prepare the minimap test.")
        atomic_bytes(reference_path, encoded.tobytes())
        request = {"version": 1, "test_id": test_id, "created": time.time(), "circuit": circuit,
                   "capture": capture_signature(cfg), "reference": reference_path.name,
                   "source_platform": deepcopy(source), "connection_id": connection_id}
        self.pause()
        (self.log / "stop-editor-test").unlink(missing_ok=True)
        atomic_json(self.log / "editor-test-status.json", {"test_id": test_id, "state": "queued",
                    "message": "Switch to MapleStory to test this connection. Combat and pickup are disabled.", "updated": time.time()})
        atomic_json(self.log / "editor-test-request.json", request)
        return "Test queued. Switch to MapleStory; the controller pauses automatically when the test ends."
