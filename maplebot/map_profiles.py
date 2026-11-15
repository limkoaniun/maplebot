"""Complete map profiles and conservative recognition from prepared game frames.

Character controls, combat skills, health policy, and player templates stay in the
base configuration. A map owns its minimap, monsters, navigation calibration,
reference image, and geometry. Recognition never changes configuration or input.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np

from maplebot.nav.editor import compile_plan, validate_plan


PROFILE_VERSION = 2
GEOMETRY_KEYS = {"waypoints", "platforms", "editor_plan", "editor_profile",
                 "editor_reference_sha256", "movement_test", "enabled", "start_idx", "loop"}


def profile_key(name: str) -> str:
    name = str(name).strip()
    if not name or len(name) > 100:
        raise ValueError("Give the map profile a name of 1–100 characters.")
    readable = re.sub(r"[^a-z0-9_-]+", "-", name.casefold()).strip("-")[:50] or "map"
    return f"{readable}-{hashlib.sha256(name.casefold().encode()).hexdigest()[:10]}"


def capture_compatible(capture: dict, cfg: dict, frame_shape=None) -> bool:
    """Compare frame geometry, deliberately excluding the active map's crop."""
    if not isinstance(capture, dict):
        return False
    game = cfg.get("game_window", {})
    size = list(game.get("working_size") or [])
    if (str(capture.get("window_title", "")).casefold() != str(game.get("title", "")).casefold()
            or list(capture.get("working_size") or []) != size
            or capture.get("title_bar_px", 0) != int(game.get("title_bar_px", 0) or 0)):
        return False
    return frame_shape is None or not size or list(frame_shape[1::-1]) == size


def map_settings_from_config(cfg: dict) -> dict:
    circuit = cfg.get("route", {}).get("circuit", {})
    return {"minimap": deepcopy(cfg.get("minimap", {})),
            "monsters": deepcopy(cfg.get("monsters", {})),
            "circuit_settings": {k: deepcopy(v) for k, v in circuit.items() if k not in GEOMETRY_KEYS}}


def normalize_map_settings(settings):
    if not isinstance(settings, dict):
        return None
    from maplebot.config import deep_merge, load_config
    return deep_merge(map_settings_from_config(load_config()), settings)


def validate_map_settings(settings, capture=None, image_size=None) -> list[str]:
    if not isinstance(settings, dict):
        return ["This older profile needs minimap and monster settings. Calibrate and save it before use."]
    errors = []
    minimap, monsters = settings.get("minimap"), settings.get("monsters")
    rect = minimap.get("fixed_rect") if isinstance(minimap, dict) else None
    valid_rect = (isinstance(rect, (list, tuple)) and len(rect) == 4
                  and all(isinstance(v, int) and not isinstance(v, bool) for v in rect)
                  and min(rect[:2]) >= 0 and min(rect[2:]) >= 8)
    if not valid_rect:
        errors.append("Calibrate the minimap crop for this map.")
    elif ((capture is not None and list(capture.get("minimap_rect") or []) != list(rect))
          or (image_size is not None and list(rect[2:]) != list(image_size))):
        errors.append("The minimap crop, reference image, and capture calibration must agree.")
    names = monsters.get("names") if isinstance(monsters, dict) else None
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name.strip() for name in names)):
        errors.append("Select the monsters for this map; an empty list must not load every monster.")
    if not isinstance(settings.get("circuit_settings", {}), dict):
        errors.append("Navigation calibration must contain named settings.")
    return errors


@dataclass
class MapProfile:
    key: str
    name: str
    plan: dict
    reference_bgr: np.ndarray
    map_settings: dict | None
    capture: dict
    image_sha256: str
    version: int = PROFILE_VERSION

    def __post_init__(self):
        self.map_settings = normalize_map_settings(self.map_settings)

    @property
    def reference(self):
        return self.reference_bgr

    @property
    def errors(self) -> list[str]:
        return (validate_map_settings(self.map_settings, self.capture, self.plan.get("image_size"))
                + validate_plan(self.plan))

    @property
    def ready(self) -> bool:
        return not self.errors


def read_profile(path: Path) -> MapProfile:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") not in (1, PROFILE_VERSION):
        raise ValueError("Unsupported map profile version.")
    plan = data.get("plan")
    errors = validate_plan(plan, require_route=False)
    if errors:
        raise ValueError("Invalid map profile: " + "; ".join(errors))
    key = profile_key(plan["name"])
    if path.stem != key:
        raise ValueError("The map profile name does not match its filename.")
    encoded = path.with_suffix(".png").read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if data.get("image_sha256") and data["image_sha256"] != digest:
        raise ValueError("The map reference image changed independently of its profile.")
    reference = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if reference is None or list(reference.shape[1::-1]) != list(plan.get("image_size", [])):
        raise ValueError("The map reference image is missing or has the wrong dimensions.")
    settings = deepcopy(data.get("map_settings")) if data.get("version") == PROFILE_VERSION else None
    return MapProfile(key, plan["name"], deepcopy(plan), reference, settings,
                      deepcopy(plan.get("capture", {})), digest, data.get("version", 1))


def load_profiles(root: Path, directory=None, *, errors=None) -> list[MapProfile]:
    directory = Path(directory or "config/map_profiles")
    folder = directory if directory.is_absolute() else Path(root) / directory
    result = []
    for path in sorted(folder.glob("*.json")):
        try:
            result.append(read_profile(path))
        except (OSError, ValueError, TypeError, KeyError) as error:
            if errors is not None:
                errors.append(f"{path.name}: {error}")
    return result


def apply_profile(base_cfg: dict, profile: MapProfile, *, preserve_test=False) -> dict:
    """Build a complete replacement; the caller must pause before installing it."""
    errors = profile.errors
    if errors:
        raise ValueError("Profile is not ready: " + "; ".join(errors))
    if not capture_compatible(profile.capture, base_cfg):
        raise ValueError("This profile uses different game capture settings.")
    cfg = deepcopy(base_cfg)
    settings = profile.map_settings
    cfg["minimap"] = deepcopy(settings["minimap"])
    cfg["monsters"] = deepcopy(settings["monsters"])
    current_circuit = cfg.get("route", {}).get("circuit", {})
    if not (preserve_test and current_circuit.get("movement_test")):
        circuit = compile_plan(profile.plan, settings.get("circuit_settings", {}))
        circuit.update(editor_plan=deepcopy(profile.plan), editor_profile=profile.key,
                       editor_reference_sha256=profile.image_sha256)
        circuit.pop("movement_test", None)
        cfg.setdefault("route", {}).update(map="", circuit=circuit)
    identity = cfg.setdefault("map_profile", {})
    identity.update(active=profile.key, name=profile.name, reference_sha256=profile.image_sha256)
    identity.setdefault("directory", "config/map_profiles")
    return cfg


def profile_config_matches(cfg: dict, profile: MapProfile, *, preserve_test=True) -> bool:
    """An identity label alone cannot authorize mismatched map calibration."""
    if not profile.ready or not capture_compatible(profile.capture, cfg):
        return False
    identity = cfg.get("map_profile", {})
    if identity.get("active") != profile.key or identity.get("reference_sha256") != profile.image_sha256:
        return False
    settings = profile.map_settings
    if cfg.get("minimap") != settings["minimap"] or cfg.get("monsters") != settings["monsters"]:
        return False
    circuit = cfg.get("route", {}).get("circuit", {})
    if preserve_test and circuit.get("movement_test"):
        return True
    if circuit.get("editor_plan") != profile.plan or circuit.get("editor_reference_sha256") != profile.image_sha256:
        return False
    compiled = compile_plan(profile.plan, settings.get("circuit_settings", {}))
    keys = set(settings.get("circuit_settings", {})) | {"waypoints", "platforms", "enabled", "loop"}
    return all(circuit.get(key) == compiled.get(key) for key in keys)


def crop_for_profile(frame: np.ndarray, profile: MapProfile, cfg: dict):
    if frame is None or frame.ndim != 3 or not capture_compatible(profile.capture, cfg, frame.shape):
        return None
    rect = profile.capture.get("minimap_rect")
    if (not isinstance(rect, (list, tuple)) or len(rect) != 4
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in rect)):
        return None
    x, y, width, height = rect
    if min(x, y) < 0 or min(width, height) < 8 or x+width > frame.shape[1] or y+height > frame.shape[0]:
        return None
    crop = frame[y:y+height, x:x+width]
    return crop if crop.shape == profile.reference_bgr.shape else None


def _marker_mask(image, settings):
    mask = np.zeros(image.shape[:2], np.uint8)
    for field in ("player_color_bgr", "other_color_bgr"):
        color = settings.get(field)
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            continue
        color = np.asarray(color, dtype=np.int16)
        tolerance = min(40, max(5, int(settings.get("player_tolerance", 20))))
        binary = cv2.inRange(image, np.clip(color-tolerance, 0, 255).astype(np.uint8),
                            np.clip(color+tolerance, 0, 255).astype(np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for index in range(1, count):
            if 1 <= stats[index, cv2.CC_STAT_AREA] <= 70 and max(stats[index, 2:4]) <= 14:
                mask[labels == index] = 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def _map_features(image):
    # The minimap panel is translucent: luminance edges can belong to full-size
    # scenery behind it. The miniature terrain retains its own coloured texture.
    pixels = image.astype(np.float32)
    chroma = np.stack((pixels[:, :, 2] - pixels[:, :, 1],
                       pixels[:, :, 0] - pixels[:, :, 1]), axis=-1)
    chroma = cv2.GaussianBlur(chroma, (3, 3), 0)
    dx, dy = cv2.Sobel(chroma, cv2.CV_32F, 1, 0), cv2.Sobel(chroma, cv2.CV_32F, 0, 1)
    return np.sqrt(np.sum(dx * dx + dy * dy, axis=-1))


def _map_evidence(reference, current, settings):
    if (reference is None or current is None or reference.shape != current.shape
            or reference.ndim != 3 or min(reference.shape[:2]) < 8):
        return None
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    terrain = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 35) & (_marker_mask(reference, settings) == 0)
    terrain[:2] = terrain[-2:] = False
    terrain[:, :2] = terrain[:, -2:] = False
    a, b = _map_features(reference), _map_features(current)
    if (np.count_nonzero(terrain) < max(80, terrain.size * .03)
            or np.count_nonzero(a[terrain] > 12) < 40):
        return None
    return a, b, terrain, _marker_mask(current, settings) == 0


def _correlation(first, second):
    first, second = first.astype(np.float64), second.astype(np.float64)
    first, second = first-first.mean(), second-second.mean()
    denominator = np.linalg.norm(first)*np.linalg.norm(second)
    return float(np.dot(first, second)/denominator) if denominator > 1e-8 else 0.0


def _score_alignment(evidence, offset, *, check_bands=True):
    a, b, terrain, current_valid = evidence
    height, width = a.shape
    dx, dy = offset
    x0, x1, y0, y1 = max(0, dx), min(width, width+dx), max(0, dy), min(height, height+dy)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    valid = np.zeros_like(terrain)
    valid[y0:y1, x0:x1] = terrain[y0:y1, x0:x1] & current_valid[y0-dy:y1-dy, x0-dx:x1-dx]
    if np.count_nonzero(valid) < max(80, np.count_nonzero(terrain)*.55):
        return 0.0
    aligned = np.zeros_like(b)
    aligned[y0:y1, x0:x1] = b[y0-dy:y1-dy, x0-dx:x1-dx]
    if np.count_nonzero(aligned[valid] > 12) < 40:
        return 0.0
    score = _correlation(a[valid], aligned[valid])
    if score < .5:
        return 0.0
    if not check_bands:
        return max(0.0, min(1.0, score))
    # A shared strip of brick or grass cannot identify an otherwise different map.
    # Require retained, correlated terrain in distributed horizontal/vertical bands.
    for axis in (0, 1):
        occupied = 0
        for indexes in np.array_split(np.arange(a.shape[axis]), 3):
            selector = np.zeros_like(valid)
            if axis == 0:
                selector[indexes, :] = True
            else:
                selector[:, indexes] = True
            expected = np.count_nonzero(terrain & selector)
            if expected < max(30, np.count_nonzero(terrain)*.04):
                continue
            selected = valid & selector
            if (np.count_nonzero(selected) < max(20, expected*.4)
                    or _correlation(a[selected], aligned[selected]) < .5):
                return 0.0
            occupied += 1
        if occupied < 2:
            return 0.0
    return max(0.0, min(1.0, score))


def stable_map_similarity(reference, current, minimap_settings=None) -> float:
    """Compare terrain at the same coordinates; no implicit translation."""
    evidence = _map_evidence(reference, current, minimap_settings or {})
    return _score_alignment(evidence, (0, 0)) if evidence is not None else 0.0


def align_map(reference, current, minimap_settings=None):
    """Return (score, offset, margin); canonical pixel = current pixel + offset.

    An ambiguous alignment or insufficient distributed terrain returns score 0.
    Callers must still enforce their normal map confidence threshold.
    """
    evidence = _map_evidence(reference, current, minimap_settings or {})
    if evidence is None:
        return 0.0, (0, 0), 0.0
    a, b, terrain, _ = evidence
    height, width = a.shape
    px, py = max(1, round(width*.35)), max(1, round(height*.2))
    padded = cv2.copyMakeBorder(b, py, py, px, px, cv2.BORDER_CONSTANT, value=0)
    scores = cv2.matchTemplate(padded, a, cv2.TM_CCOEFF_NORMED, mask=terrain.astype(np.uint8))
    scores = np.nan_to_num(scores, nan=-1, posinf=-1, neginf=-1)
    candidates = {(0, 0)}
    for index in range(6):
        _, score, _, (x, y) = cv2.minMaxLoc(scores)
        if score < .2:
            break
        offset = (px-x, py-y)
        candidates.add(offset)
        if index == 0:
            candidates.update((offset[0]+dx, offset[1]+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                              if abs(offset[0]+dx) <= px and abs(offset[1]+dy) <= py)
        scores[max(0, y-2):y+3, max(0, x-2):x+3] = -1
    ranked = sorted(((_score_alignment(evidence, offset, check_bands=False), offset) for offset in candidates), reverse=True)
    score, offset = ranked[0]
    # Immediate pixel neighbours describe the width of one interpolation peak.
    # Competing repeated terrain (another floor or scaffold) is a distinct peak.
    distant = [value for value, point in ranked[1:] if max(abs(point[i]-offset[i]) for i in (0, 1)) > 1]
    margin = score-max(distant, default=0.)
    local_margin = score-(ranked[1][0] if len(ranked) > 1 else 0.)
    accepted = margin >= .10 and local_margin >= .02 and _score_alignment(evidence, offset) > 0
    return (score if accepted else 0.0), offset, margin


@dataclass(frozen=True)
class MapMatch:
    state: str
    profile_key: str | None = None
    profile_name: str | None = None
    score: float = 0.0
    margin: float = 0.0
    stable_count: int = 0
    reason: str = ""
    crop: tuple[int, int, int, int] | None = None
    profile: MapProfile | None = None
    offset: tuple[int, int] = (0, 0)


class MapProfileRecognizer:
    def __init__(self, profiles, *, threshold=.78, margin=.10, stable_frames=3):
        self.profiles = list(profiles)
        self.threshold, self.margin = float(threshold), float(margin)
        self.stable_frames = int(stable_frames)
        if (not math.isfinite(self.threshold) or not math.isfinite(self.margin)
                or not 0 < self.threshold <= 1 or not 0 <= self.margin <= 1 or self.stable_frames < 1):
            raise ValueError("Invalid map recognition confidence settings.")
        self.reset()

    def reset(self):
        self._candidate = None
        self._count = 0

    def observe(self, full_prepared_frame, cfg) -> MapMatch:
        ranked = []
        for profile in self.profiles:
            crop = crop_for_profile(full_prepared_frame, profile, cfg)
            if crop is not None:
                settings = (profile.map_settings or {}).get("minimap", {})
                score, offset, _ = align_map(profile.reference_bgr, crop, settings)
                ranked.append((score, profile.key, profile, offset))
        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        if not ranked or ranked[0][0] < self.threshold:
            self.reset()
            return MapMatch("unknown", score=ranked[0][0] if ranked else 0.,
                            reason="No saved map matches this frame. Calibrate a map profile before moving.")
        score, key, profile, offset = ranked[0]
        gap = score-(ranked[1][0] if len(ranked) > 1 else 0.)
        if len(ranked) > 1 and gap < self.margin:
            self.reset()
            return MapMatch("ambiguous", score=score, margin=gap,
                            reason="More than one saved map matches. The bot stays paused.")
        self._count = self._count+1 if self._candidate == key else 1
        self._candidate = key
        crop = tuple(profile.capture["minimap_rect"])
        common = dict(profile_key=key, profile_name=profile.name, score=score, margin=gap,
                      stable_count=self._count, crop=crop, offset=offset)
        if self._count < self.stable_frames:
            return MapMatch("candidate", reason=f"Confirming {profile.name} across fresh frames.", **common)
        errors = profile.errors
        if errors:
            return MapMatch("incomplete", reason=f"{profile.name}: " + errors[0], **common)
        return MapMatch("matched", reason=f"Recognized {profile.name}.", profile=profile, **common)

    def preflight(self, full_prepared_frame, cfg, active_key=None) -> MapMatch:
        result = self.observe(full_prepared_frame, cfg)
        active_key = active_key or cfg.get("map_profile", {}).get("active")
        if result.state == "matched" and active_key and result.profile_key != active_key:
            return MapMatch("wrong_profile", result.profile_key, result.profile_name,
                            result.score, result.margin, result.stable_count,
                            "A different saved map is visible. Load its settings while paused.", result.crop,
                            offset=result.offset)
        return result
