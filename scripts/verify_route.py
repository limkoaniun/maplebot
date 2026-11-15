"""Run selected editor connections once and retain live landing evidence."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from maplebot.editor_bridge import EditorBridge, atomic_json
from maplebot.nav.circuit import Platform
from maplebot.nav.editor import platform_y
from maplebot.vision.bars import bar_fill_percent


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def health(log):
    path = log / "bot-bars.png"
    try:
        if time.time() - path.stat().st_mtime > 3:
            return None
    except OSError:
        return None
    bars = cv2.imread(str(path))
    if bars is None or bars.shape[0] < 48 or bars.shape[1] < 392:
        return None
    value = bar_fill_percent(bars[21:48, 8:392])
    return value if math.isfinite(value) and 0 <= value <= 100 else None


def fresh(value, max_age=3):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and -1 <= time.time() - value <= max_age)


class PinnedContext:
    """One invocation may only use its initial saved config, profile, and map."""
    def __init__(self, bridge):
        self.bridge = bridge
        self.config_hash = hashlib.sha256(bridge.user_config.read_bytes()).hexdigest()
        self.cfg = bridge._cfg()
        self.plan, self.reference, _ = bridge.load_current()
        if self.plan.get("context_warning"):
            raise ValueError(self.plan["context_warning"])
        self.plan = deepcopy(self.plan)
        self.reference = self.reference.copy()
        bridge._check_context(self.plan, self.reference, self.cfg)
        self.files = {}
        profile = self.cfg.get("route", {}).get("circuit", {}).get("editor_profile")
        if profile:
            if Path(profile).name != profile or "/" in profile or "\\" in profile:
                raise ValueError("The saved profile identifier is invalid.")
            for suffix in (".json", ".png"):
                path = bridge.profiles / (profile + suffix)
                self.files[path] = hashlib.sha256(path.read_bytes()).hexdigest()

    def check(self, state=None):
        if hashlib.sha256(self.bridge.user_config.read_bytes()).hexdigest() != self.config_hash:
            raise RuntimeError("Saved settings changed during this test invocation; no further tests will run.")
        for path, expected in self.files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("The saved map profile changed during this test invocation.")
        if state is not None:
            if not state.get("running") or not fresh(state.get("updated"), 6):
                raise RuntimeError("The controller stopped reporting a fresh status.")
            if state.get("capture") != self.plan.get("capture"):
                raise RuntimeError("The controller's capture settings changed during this test.")
            test = state.get("editor_test") or {}
            running_test = isinstance(test, dict) and test.get("state") == "running"
            map_state = state.get("map_state")
            if (map_state in {"unknown", "ambiguous", "incomplete", "settings_mismatch"}
                    or (running_test and state.get("map_ready") is False)):
                raise RuntimeError("The runtime map guard blocked this test: "
                                   + str(state.get("map_reason") or map_state or "map not ready"))
            # A rebuilt runner may briefly report a candidate before preflight.
            # Duplicate captures remain usable within the runner's freshness limit.
            age = state.get("capture_age_s")
            if running_test and (state.get("capture_state") == "stale_or_missing"
                    or (isinstance(age, (int, float)) and (not math.isfinite(age) or age > .75))):
                raise RuntimeError("The runner's game capture became stale during the movement test.")
        _, metadata = self.bridge._check_context(self.plan, self.reference, self.cfg)
        if not fresh(metadata.get("updated"), 3):
            raise RuntimeError("A fresh matching minimap is no longer available.")
        return metadata


def platform_surface(platform):
    """Use the executor's surface model, including minimap rounding uncertainty."""
    return Platform(float(platform["y"]), float(platform["left"]), float(platform["right"]),
                    float(platform["right_y"]) if platform.get("right_y") is not None else None,
                    tuple(tuple(point) for point in platform.get("knots", [])))


class LandingProof:
    """Repeated polls of one snapshot never count as a stable landing."""
    def __init__(self, platform, *, from_x=None, away_px=0, stable_s=.6, stationary=False):
        self.platform = platform
        self.surface = platform_surface(platform)
        self.from_x, self.away_px, self.stable_s = from_x, away_px, stable_s
        self.stationary = stationary
        self.last_updated = None
        self.samples = []
        self.confirmed = False

    def observe(self, pos, updated, visible=True):
        if not fresh(updated) or updated == self.last_updated:
            return self.confirmed
        if self.last_updated is not None and updated < self.last_updated:
            return self.confirmed
        self.last_updated = updated
        valid = (visible and isinstance(pos, (list, tuple)) and len(pos) == 2
                 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)
                 and self.platform["left"] <= pos[0] <= self.platform["right"]
                 and self.surface.support_error(pos) <= .75)
        if not valid:
            self.samples = []
            return self.confirmed
        expected_y = platform_y(self.platform, pos[0])
        low_y, high_y = self.surface.support_range(pos[0])
        error_y = pos[1] - min(high_y, max(low_y, pos[1]))
        if self.samples:
            first = self.samples[0]
            moved = (abs(pos[0] - first["position"][0]) > 1
                     or abs(pos[1] - first["position"][1]) > 1)
            # Moving up a plank changes absolute Y. What must stay stable while
            # walking is the distance from that same support, not its height.
            if (updated - self.samples[-1]["updated"] > 1.5
                    or abs(error_y - first["support_error_y"]) > 1
                    or (self.stationary and moved)):
                self.samples = []
        self.samples.append({"updated": updated, "position": list(pos),
                             "expected_y": expected_y, "support_range_y": [low_y, high_y],
                             "nominal_error_y": pos[1] - expected_y, "support_error_y": error_y})
        moved_away = self.from_x is None or abs(pos[0] - self.from_x) >= self.away_px
        if len(self.samples) >= 3 and updated - self.samples[0]["updated"] >= self.stable_s and moved_away:
            self.confirmed = True
        return self.confirmed


class HealthGuard:
    def __init__(self, minimum=25):
        self.minimum = max(25.0, minimum)
        self.missing_since = None

    def check(self, value, now=None):
        now = time.monotonic() if now is None else now
        if value is None or not math.isfinite(value):
            self.missing_since = now if self.missing_since is None else self.missing_since
            if now - self.missing_since > 2:
                raise RuntimeError("Health diagnostics remained unreadable for more than two seconds.")
        elif value < self.minimum:
            raise RuntimeError(f"Health fell below {self.minimum:g}% ({value:.1f}%).")
        else:
            self.missing_since = None


def evidence_images(log, target, label):
    saved = {}
    for source, suffix in (("relay-frame.png", "game"), ("bot-view.png", "detection"),
                           ("editor-minimap.png", "minimap")):
        image = log / source
        try:
            if image.exists():
                output = target / f"{label}-{suffix}.png"
                shutil.copyfile(image, output)
                saved[suffix] = {"file": output.name, "updated": image.stat().st_mtime}
        except OSError:
            pass
    return saved


def observed_position(state, metadata):
    pos = metadata.get("player_pos")
    if pos is not None:
        return pos, metadata.get("updated"), True
    return state.get("minimap"), state.get("updated"), bool(state.get("minimap_seen"))


def run_connections(bridge, connections, min_hp=25):
    pin = PinnedContext(bridge)
    evidence = bridge.log / ("route-check-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    evidence.mkdir(parents=True)
    results = []
    queued = False
    samples, label, current = [], None, None
    try:
        for number, ident in enumerate(connections, 1):
            label, samples = f"step-{number:02d}", []
            current = {"connection": ident, "result": "preflight"}
            state = read_json(bridge.log / "active-status.json")
            pin.check(state)
            plan, reference = pin.plan, pin.reference
            hp = health(bridge.log)
            if hp is None or hp < max(25.0, min_hp):
                raise RuntimeError(f"Health is too low or unreadable before {ident}: {hp}")
            connection = next((c for c in plan["connections"] if c["id"] == ident), None)
            if connection is None:
                raise ValueError(f"Connection {ident!r} is absent from the pinned map profile.")
            destination = next(p for p in plan["platforms"] if p["id"] == connection["to_platform"])
            proof = LandingProof(destination, from_x=connection["to_x"], away_px=2)
            hp_guard = HealthGuard(min_hp)
            bridge.test_connection(plan, ident, reference)
            queued = True
            requested = bridge.test_status()["test_id"]
            current.update(test_id=requested, destination=destination, result="running")
            print(json.dumps({"connection": ident, "state": "queued", "hp": round(hp, 1)}), flush=True)
            previous = None
            deadline = time.monotonic() + 65
            result = {}
            while time.monotonic() < deadline:
                state = read_json(bridge.log / "active-status.json")
                test = bridge.test_status()
                hp = health(bridge.log)
                sample = {k: state.get(k) for k in ("updated", "minimap", "paused", "foreground",
                          "input_blocked", "circuit_index", "circuit_state", "command", "keys_sent")}
                sample["hp"] = hp
                sample["test"] = test
                samples.append(sample)
                metadata = pin.check(state)
                if state.get("editor_test", {}).get("test_id") == requested:
                    proof.observe(*observed_position(state, metadata))
                marker = (test.get("state"), state.get("circuit_index"), state.get("circuit_state"))
                if marker != previous:
                    print(json.dumps({"connection": ident, "test": marker[0], "step": marker[1],
                          "state": marker[2], "position": sample["minimap"], "hp": hp}), flush=True)
                    previous = marker
                hp_guard.check(hp)
                if test.get("test_id") == requested and test.get("state") in {"completed", "blocked", "cancelled", "error"}:
                    result = test
                    break
                time.sleep(.2)
            if result.get("state") != "completed":
                raise RuntimeError(f"{ident} did not complete: {result or 'test deadline'}")
            movement_landing = deepcopy(proof.samples)
            proof = LandingProof(destination, from_x=connection["to_x"], away_px=2,
                                 stationary=True)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                state = read_json(bridge.log / "active-status.json")
                metadata = pin.check(state)
                if state.get("paused") and "editor_test" not in state:
                    proof.observe(*observed_position(state, metadata))
                if state.get("paused") and "editor_test" not in state and proof.confirmed:
                    break
                time.sleep(.2)
            else:
                raise RuntimeError("The normal paused controller did not provide stable landing and sideways-walk evidence.")
            pos = proof.samples[-1]["position"]
            record = {"connection": ident, "kind": connection["kind"], "destination_y": destination["y"],
                      "expected_y_at_position": platform_y(destination, pos[0]),
                      "position": pos, "result": "landed and walked away", "test_id": requested,
                      "keys_sent": state.get("keys_sent"), "hp": health(bridge.log),
                      "landing_observations": proof.samples,
                      "movement_landing_observations": movement_landing,
                      "images": evidence_images(bridge.log, evidence, label)}
            results.append(record)
            atomic_json(evidence / f"{label}-trace.json", samples)
            atomic_json(evidence / "results.json", results)
            print(json.dumps(record), flush=True)
        print(f"Evidence saved: {evidence}", flush=True)
        return results
    except (Exception, KeyboardInterrupt) as error:
        if current is not None:
            current.update(result="failed", error=str(error) or type(error).__name__,
                           images=evidence_images(bridge.log, evidence, label))
            atomic_json(evidence / f"{label}-failure.json", current)
            atomic_json(evidence / f"{label}-trace.json", samples)
        raise
    finally:
        if queued:
            bridge.stop_test()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("connections", nargs="+", help="Saved connection IDs, in traversal order")
    parser.add_argument("--min-hp", type=float, default=25)
    args = parser.parse_args()
    if not math.isfinite(args.min_hp):
        parser.error("The HP threshold must be finite.")
    run_connections(EditorBridge(ROOT), args.connections, max(25.0, args.min_hp))


if __name__ == "__main__":
    main()
