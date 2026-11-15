"""One actual HP and MP tap, with fresh bar evidence, while farming stays paused."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maplebot.types import Command


def valid_percent(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 100


def potion_config(cfg):
    cfg = deepcopy(cfg)
    for key in ("left", "right", "up", "down", "jump", "teleport", "attack", "aoe", "normal_attack", "pickup"):
        cfg["keys"][key] = ""
    cfg.setdefault("buffs", {})["keys"] = []
    return cfg


class PotionProof:
    def __init__(self, resource, baseline, sequence, tap_time):
        if not valid_percent(baseline) or baseline >= 98:
            raise ValueError(f"{resource.upper()} is unreadable or already full; one tap cannot prove its effect")
        self.resource, self.baseline, self.sequence = resource, float(baseline), sequence
        self.tap_time = tap_time
        self.deadline = tap_time + 3

    def observe(self, value, sequence, captured_at, now):
        if now > self.deadline:
            raise RuntimeError(f"No confirmed {self.resource.upper()} increase within 3 seconds")
        if sequence <= self.sequence or captured_at <= self.tap_time:
            return False
        if not valid_percent(value):
            raise RuntimeError(f"{self.resource.upper()} became unreadable")
        return value >= self.baseline + 2


class OneTapBackend:
    """Reject all keys except the currently armed potion, once per resource."""
    def __init__(self, backend, clock=time.monotonic):
        self.backend, self.clock = backend, clock
        self.armed = None
        self.counts, self.times = {}, {}

    def arm(self, resource, key):
        if not key or self.counts.get(resource, 0):
            raise RuntimeError("Potion key is missing or this resource was already tapped")
        self.armed = (resource, key)

    def down(self, key):
        if self.armed is None or key != self.armed[1] or self.counts.get(self.armed[0], 0):
            raise RuntimeError(f"Unexpected or repeated test key: {key}")
        resource = self.armed[0]
        self.backend.down(key)
        self.counts[resource] = 1
        self.times[resource] = self.clock()

    def up(self, key):
        self.backend.up(key)


class ConfigPin:
    def __init__(self, config_path, cfg):
        active = cfg.get("map_profile", {}).get("active")
        if not active or Path(active).name != active or "/" in active or "\\" in active:
            raise ValueError("Select a saved map profile before testing potions")
        folder = Path(cfg["map_profile"].get("directory", "config/map_profiles"))
        if not folder.is_absolute():
            folder = ROOT / folder
        paths = [Path(config_path), ROOT / "config/default.yaml", folder / (active + ".json"), folder / (active + ".png")]
        self.hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.profile = active
        self.reference = cfg["map_profile"].get("reference_sha256")

    def check(self):
        if any(hashlib.sha256(path.read_bytes()).hexdigest() != expected for path, expected in self.hashes.items()):
            raise RuntimeError("Saved configuration or map profile changed during the potion test")


def verify(config_path, output):
    import cv2
    from maplebot.bot import build_bot
    from maplebot.config import load_config
    from maplebot.control.hotkeys import Hotkeys
    from maplebot.editor_bridge import atomic_json

    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"state": "starting", "started": time.time(), "results": {}, "taps": {}}
    bot = hotkeys = None
    stopped = threading.Event()
    cancel_file = ROOT / "log/stop-potion-test"
    evidence = output.parent / ("potion-evidence-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    evidence.mkdir(exist_ok=True)

    def publish(**fields):
        report.update(fields, updated=time.time())
        atomic_json(output, report)

    def cancel():
        stopped.set()
        if bot is not None:
            bot.pause_all()

    def check_cancel():
        if stopped.is_set() or cancel_file.exists():
            raise RuntimeError("Potion test cancelled")

    try:
        cfg = load_config(str(config_path))
        pin = ConfigPin(config_path, cfg)
        publish(profile=pin.profile, reference_sha256=pin.reference)
        bot = build_bot(potion_config(cfg), None, False)
        bot.pause_all()
        backend = OneTapBackend(bot.keyboard.backend)
        bot.keyboard.backend = backend
        hotkeys = Hotkeys({"f1": cancel, "f12": cancel}, debounce_s=.2)
        hotkeys.start()
        bot.source.start()  # deliberately never start the keyboard worker or Bot.run
        focus_deadline = time.monotonic() + 60
        while not bot.keyboard.is_active():
            check_cancel()
            pin.check()
            publish(state="waiting_focus", reason="Focus the game window; all test input remains off")
            if time.monotonic() >= focus_deadline:
                raise RuntimeError("Game window did not gain focus within 60 seconds")
            time.sleep(.1)

        def next_snapshot(deadline, *, confirming=False):
            while time.monotonic() <= deadline:
                check_cancel()
                pin.check()
                if not bot.keyboard.is_active():
                    raise RuntimeError("Game window lost foreground focus")
                snap = bot.step()
                if snap is None:
                    if bot.capture_state == "stale_or_missing":
                        if not confirming or bot.capture_sequence is not None:
                            raise RuntimeError("Game capture is stale or unavailable")
                    time.sleep(.025)
                    continue
                if not bot.map_ready:
                    if confirming:
                        time.sleep(.025)
                        continue
                    raise RuntimeError("Current map no longer matches the pinned profile")
                if (bot.map_match.profile is None or bot.map_match.profile.key != pin.profile
                        or bot.map_match.profile.image_sha256 != pin.reference):
                    raise RuntimeError("Map reference changed")
                if not bot._minimap_seen or snap.minimap_pos is None:
                    raise RuntimeError("Current minimap player dot is missing")
                if not valid_percent(snap.hp) or snap.hp <= 60:
                    raise RuntimeError("HP is unreadable or at/below 60%; potion test stopped")
                if not valid_percent(snap.mp):
                    raise RuntimeError("MP is unreadable")
                if not 0 <= time.monotonic() - bot.capture_time <= .75:
                    raise RuntimeError("Prepared capture is no longer fresh")
                return snap
            raise RuntimeError("Timed out waiting for a fresh verified game frame")

        for resource in ("hp", "mp"):
            publish(state="preflight", resource=resource)
            snap = next_snapshot(time.monotonic() + 5, confirming=resource == "hp")
            baseline = getattr(snap, resource)
            room_deadline = time.monotonic() + 90
            while baseline >= 98:
                publish(state="waiting_resource", resource=resource, hp=snap.hp, mp=snap.mp,
                        reason=f"{resource.upper()} must be below 98% to prove an increase; all input is off")
                if time.monotonic() >= room_deadline:
                    raise RuntimeError(f"{resource.upper()} stayed full for 90 seconds; no tap sent")
                snap = next_snapshot(room_deadline)
                baseline = getattr(snap, resource)
            publish(state="preflight", resource=resource, reason=None, hp=snap.hp, mp=snap.mp)
            before = evidence / f"{resource}-before.png"
            if not cv2.imwrite(str(before), snap.frame):
                raise RuntimeError("Could not save the baseline image")
            sequence = bot.capture_sequence
            key = str(cfg["keys"].get(resource + "_potion") or "").strip()
            record = report["results"][resource] = {"baseline": baseline, "before_image": str(before), "state": "ready"}
            pin.check()
            with bot._control_lock:
                check_cancel()
                if (not bot.keyboard.is_active() or not bot.full_pause or not bot.map_ready
                        or not 0 <= time.monotonic() - bot.capture_time <= .75):
                    raise RuntimeError("Potion preflight permission expired")
                backend.arm(resource, key)
                try:
                    if resource == "hp":
                        bot.keyboard.set_hp_only(source_time=bot.capture_time)
                    else:
                        bot.keyboard.set_command(Command.NONE, source_time=bot.capture_time)
                    bot.keyboard.set_enabled(True)
                    bot.keyboard.request(resource + "_potion")
                    bot.keyboard.apply_once(time.monotonic())
                finally:
                    bot.keyboard.set_enabled(False)
                    backend.armed = None
            if backend.counts.get(resource) != 1:
                raise RuntimeError(f"{resource.upper()} key was not sent; no automatic retry")
            proof = PotionProof(resource, baseline, sequence, backend.times[resource])
            record.update(state="awaiting_response", tap_time=proof.tap_time, taps=1)
            publish(state="awaiting_response", taps=dict(backend.counts))
            while True:
                snap = next_snapshot(proof.deadline)
                after = evidence / f"{resource}-after.png"
                if not cv2.imwrite(str(after), snap.frame):
                    raise RuntimeError("Could not save the response image")
                value = getattr(snap, resource)
                record.update(after=value, gain=value-baseline, after_image=str(after), sequence=bot.capture_sequence)
                if proof.observe(value, bot.capture_sequence, bot.capture_time, time.monotonic()):
                    record["state"] = "passed"
                    publish(taps=dict(backend.counts))
                    break
        pin.check()
        publish(state="passed", reason=None, finished=time.time(), config_unchanged=True)
        return report
    except (Exception, KeyboardInterrupt) as error:
        publish(state="failed", reason=str(error) or "Cancelled", finished=time.time(),
                taps=dict(backend.counts) if bot is not None and "backend" in locals() else {})
        return report
    finally:
        if bot is not None:
            bot.pause_all()
            bot.close()
        if hotkeys is not None:
            hotkeys.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/user.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "log/potion-verification.json")
    parser.add_argument("--start-controller-paused", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    import win32api
    import win32event
    name = "Local\\MaplebotActive_" + hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:16]
    mutex = win32event.CreateMutex(None, False, name)
    if win32api.GetLastError() == 183:
        win32api.CloseHandle(mutex)
        print("Stop the active controller before running the potion check.")
        return 2
    try:
        report = verify(args.config, args.output)
    finally:
        win32api.CloseHandle(mutex)
    if report["state"] == "passed" and args.start_controller_paused:
        for signal in ("stop-active", "pause-active", "resume-active", "reload-active", "arm-active",
                       "return-to-start-active", "hold-farming-active"):
            (ROOT / "log" / signal).unlink(missing_ok=True)
        subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/run_active.py"), "--wait-for-start"],
                         cwd=ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
    print(json.dumps({"state": report["state"], "reason": report.get("reason"), "output": str(args.output)}))
    return 0 if report["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
