"""Observe the normal farming controller; --start explicitly resumes after preflight.

This never emits keys or changes configuration. At the end of an active watch it
requests Hold farming, preserving guarded HP protection rather than full-pausing.
"""
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
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from maplebot.editor_bridge import EditorBridge, atomic_json, capture_signature
from verify_lap import transitions
from verify_route import LandingProof, platform_surface, read_json


def fresh(status, age=3):
    updated = status.get("updated")
    return (isinstance(updated, (int, float)) and not isinstance(updated, bool)
            and math.isfinite(updated) and -1 <= time.time()-updated <= age)


def at_start(status, first):
    pos = status.get("minimap")
    return (status.get("minimap_seen") and isinstance(pos, (list, tuple)) and len(pos) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)
            and abs(pos[0]-first["x"]) <= 1 and abs(pos[1]-first["y"]) <= .75)


def on_start_platform(status, circuit):
    pos = status.get("minimap")
    ground = next((p for p in circuit.get("platforms", []) if p.get("id") == "ground"), None)
    return bool(ground and status.get("minimap_seen") and isinstance(pos, (list, tuple)) and len(pos) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)
                and ground["left"] <= pos[0] <= ground["right"]
                and platform_surface(ground).support_error(pos) <= .75)


def can_approach_start(status, circuit):
    if on_start_platform(status, circuit):
        return True
    if not status.get('minimap_seen'):
        return False
    from maplebot.nav.recovery import UnsupportedPosition, plan_return_to_start
    pos = status.get('minimap')
    try:
        plan_return_to_start(circuit, pos)
        return True
    except UnsupportedPosition:
        pass  # the controller may first regain support; checked below
    except (ValueError, TypeError, KeyError):
        return False
    # May-attempt check only: the runtime still proves the landing before it plans.
    try:
        if circuit.get('return_regain_support', True) is not True:
            return False
        from maplebot.nav.regain import regain_options
        options = regain_options(circuit, pos)
        if not options:
            return False
        platform = next(p for p in circuit['editor_plan']['platforms']
                        if p['id'] == options[0]['platform_id'])
        px = min(max(pos[0], platform['left']), platform['right'])
        plan_return_to_start(circuit, (px, platform_surface(platform).y_at(px)))
    except (ValueError, TypeError, KeyError, StopIteration):
        return False
    return True


def automatic_hold(status):
    terminal_hold = ((status.get('paused') or status.get('health_protection_active')) and
                     (status.get('recovery_terminal') or status.get('circuit_state') == 'blocked'
                      or status.get('health_protection_state') in {'needs_attention', 'dead'}))
    paused_hold = (status.get('paused') and status.get('manual_health_hold')
                   and status.get('startup_return_state') not in {'checking', 'returning'})
    return bool(status.get('running') and fresh(status) and status.get('full_pause') is False
                and (terminal_hold or paused_hold))


def preflight(bridge, *, require_foreground=False):
    status, cfg = bridge._status(), bridge._cfg()
    potion = read_json(bridge.log / "potion-verification.json")
    if (potion.get("state") != "passed" or not potion.get("config_unchanged")
            or any(potion.get("results", {}).get(name, {}).get("state") != "passed" for name in ("hp", "mp"))):
        raise ValueError("Both live HP and MP potion checks must have passed first.")
    identity = cfg.get("map_profile", {})
    if (potion.get("profile") != identity.get("active")
            or potion.get("reference_sha256") != identity.get("reference_sha256")):
        raise ValueError("Potion evidence does not match the current map profile/reference.")
    manual_monitoring = (status.get("full_pause") is False and status.get("manual_health_hold") is True
                         and status.get("health_protection_state") == "monitoring")
    if (not status.get("running") or not fresh(status) or not status.get("paused")
            or not (status.get("full_pause") is True or manual_monitoring)):
        raise ValueError("Start with a fresh controller fully paused or in a manual HP hold that is monitoring.")
    if require_foreground and not status.get("foreground"):
        raise ValueError("Put MapleStory in front before starting the circuit watch.")
    if (not status.get("map_ready") or status.get("map_state") != "matched"
            or status.get("capture") != capture_signature(cfg)
            or status.get("capture_state") not in ("fresh", "duplicate")
            or not isinstance(status.get("capture_age_s"), (int, float))
            or not 0 <= status["capture_age_s"] <= .75):
        raise ValueError("A fresh capture and matching map are required.")
    hp = status.get("hp")
    if not isinstance(hp, (int, float)) or not math.isfinite(hp) or hp < 65:
        raise ValueError("Preflight HP must be readable and at least 65%.")
    circuit = cfg.get("route", {}).get("circuit", {})
    points = circuit.get("waypoints", [])
    if (not circuit.get("enabled") or not circuit.get("loop") or len(points) != 44
            or status.get("circuit_count") != 44 or status.get("circuit_index") != 0):
        raise ValueError("This watch requires the saved 44-point loop at point 1.")
    if len(transitions(circuit)) != 17:
        raise ValueError("The saved circuit must contain the expected 17 platform transitions.")
    if (points[0].get("platform_id") != "ground" or points[0].get("action", "walk") != "walk"
            or (points[0]["x"], points[0]["y"]) != (49, 105) or not can_approach_start(status, circuit)):
        raise ValueError("The character needs the saved ground platform, a mapped return path, or a position "
                         "the controller can first walk or climb down from onto a drawn platform, "
                         "to reach point 1 (49,105).")
    if (status.get("return_count", 0) != 0 or status.get("returning_to_start")
            or status.get("recovery_terminal") or status.get("editor_test") or status.get("potion_test")):
        raise ValueError("No recovery or movement/potion test may be active at the start.")
    for name in ("editor-test-request.json", "return-to-start-active", "hold-farming-active",
                 "pause-active", "stop-active", "reload-active", "resume-active"):
        if (bridge.log / name).exists():
            raise ValueError(f"A controller request is still pending: {name}")
    if not cfg.get("health", {}).get("enable", True) or not str(cfg.get("keys", {}).get("hp_potion") or "").strip():
        raise ValueError("HP protection must be configured before observing farming.")
    key = identity.get("active")
    if not isinstance(key, str) or Path(key).name != key or "/" in key or "\\" in key:
        raise ValueError("The active profile identifier is invalid.")
    files = [bridge.user_config, bridge.profiles / f"{key}.json", bridge.profiles / f"{key}.png"]
    pinned = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    return status, cfg, deepcopy(circuit), pinned


def usable_capture(status):
    age, pos = status.get("capture_age_s"), status.get("minimap")
    return bool(status.get("map_ready") and status.get("capture_state") == "fresh"
                and isinstance(age, (int, float)) and math.isfinite(age) and 0 <= age <= .75
                and status.get("minimap_seen") and isinstance(pos, (list, tuple)) and len(pos) == 2
                and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos))


def map_guard_error(status, expected_capture):
    if status.get('capture') != expected_capture:
        return 'Capture settings changed.'
    state = status.get('map_state')
    reacquiring = (status.get('map_reacquiring') is True and status.get('full_pause') is False
                   and state in {'unknown', 'candidate', 'ambiguous'})
    if state not in {'matched', 'candidate'} and not reacquiring:
        return 'Map identity is no longer confirmed.'
    return None


def read_status(bridge):
    try:
        value = bridge._status()
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


class StatusContinuity:
    """Retry a locked/partial status file without replaying its last good sample."""
    def __init__(self, initial=None):
        self.last_good = deepcopy(initial) if initial else None
        self.missing_since = None
        self.events = []
        self.change = None

    def observe(self, status, now):
        self.change = None
        if status.get("running") is False:
            return None, "Controller reported that it stopped."
        updated = status.get("updated")
        readable = (status.get("running") is True and isinstance(updated, (int, float))
                    and not isinstance(updated, bool) and math.isfinite(updated))
        if readable:
            if not fresh(status):
                return None, "Controller status became stale."
            if self.missing_since is not None:
                self.events[-1].update(recovered=updated, duration_s=now-self.missing_since)
                self.change = {"event": "status_read_recovered", **self.events[-1]}
            self.missing_since = None
            self.last_good = deepcopy(status)
            return status, None
        if self.missing_since is None:
            self.missing_since = now
            self.events.append({"started": time.time(), "last_good_updated":
                                self.last_good.get("updated") if self.last_good else None})
            self.change = {"event": "status_read_gap", **self.events[-1]}
        self.events[-1]["duration_s"] = now-self.missing_since
        if self.last_good is not None and not fresh(self.last_good):
            return None, "Controller status became stale during an unreadable status-file gap."
        if now-self.missing_since >= 2:
            return None, "Controller status file remained unreadable for two seconds."
        return None, None


class CaptureContinuity:
    """Short missing snapshots are recorded, never used as landing evidence."""
    def __init__(self):
        self.missing_since = None
        self.events = []

    def observe(self, status, now):
        usable = usable_capture(status)
        if usable:
            if self.missing_since is not None:
                self.events[-1].update(recovered=status.get("updated"), duration_s=now-self.missing_since)
            self.missing_since = None
        else:
            if self.missing_since is None:
                self.missing_since = now
                self.events.append({"started": status.get("updated"), "capture_state": status.get("capture_state"),
                                    "minimap_seen": bool(status.get("minimap_seen"))})
            self.events[-1]["duration_s"] = now-self.missing_since
        return bool(usable)

    def expired(self, now):
        return self.missing_since is not None and now-self.missing_since >= 2


class LapEvidence:
    """Forward progress and landing observations; recovery resets never count as wraps."""
    def __init__(self, circuit):
        self.circuit = circuit
        self.first = circuit["waypoints"][0]
        self.last = len(circuit["waypoints"])-1
        self.landings = transitions(circuit)
        self.ground = next(p for p in circuit["platforms"] if p.get("id") == "ground")
        self.resets, self.recovery_events = [], []
        self.startup_events = []
        self.last_startup = None
        self.startup_active = False
        self.max_return_count = 0
        self.max_rejoin_count = 0
        self.arrival_base = None
        self.max_arrival_count = 0
        self.zone_base = None
        self.max_zone_takeoffs = 0
        self.recovery_active = False
        self.last_recovery = None
        self.last_updated = None
        self.wrap = False
        self.reset_progress()

    def reset_progress(self):
        self.wrap = False
        self.previous = None
        self.observed = []
        self.skipped = []
        self.proofs = {t["waypoint_index"]: LandingProof(t["destination"])
                       for t in self.landings if t["destination"] is not None}
        self.ground_proof = LandingProof(self.ground, stationary=True)

    def observe_ground(self, status):
        if at_start(status, self.first):
            self.ground_proof.observe(status["minimap"], status["updated"], True)
        else:
            self.ground_proof.samples.clear()
            self.ground_proof.confirmed = False

    def observe(self, status):
        updated = status.get("updated")
        if updated == self.last_updated or not fresh(status):
            return
        self.last_updated = updated
        startup = str(status.get("startup_return_state") or "idle")
        if startup != self.last_startup:
            if startup != "idle":
                self.startup_events.append({"kind": "startup", "updated": updated, "state": startup,
                    "reason": status.get("startup_return_reason")})
            self.last_startup = startup
        if startup in {"checking", "returning", "failed"}:
            if not self.startup_active:
                self.reset_progress()
            self.startup_active = True
            self.recovery_active = False
            self.last_recovery = (False, status.get("recovery_state"), 0)
            return
        if self.startup_active:
            self.reset_progress()
            self.startup_active = False
            self.last_recovery = (False, status.get("recovery_state"), 0)
        recovery = bool(status.get("returning_to_start"))
        count = int(status.get("return_count") or 0)
        phase = (recovery, status.get("recovery_state"), count)
        previous_count = self.last_recovery[2] if self.last_recovery is not None else 0
        if phase != self.last_recovery:
            self.recovery_events.append({"kind": "recovery", "updated": updated, "active": recovery, "state": phase[1],
                "count": count, "reason": status.get("recovery_reason"), "from_index": status.get("recovery_from_index")})
            self.last_recovery = phase
        if (recovery and not self.recovery_active) or count > previous_count:
            self.resets.append({"updated": updated, "reason": "recovery", "observed_indices": list(self.observed)})
            self.reset_progress()
        self.max_return_count = max(count, self.max_return_count)
        # Reported only: none of these values changes what counts as a verified lap.
        self.max_rejoin_count = max(int(status.get("rejoin_count") or 0), self.max_rejoin_count)
        arrivals = int(status.get("arrival_count") or 0)
        if self.arrival_base is None:
            self.arrival_base = arrivals
        self.max_arrival_count = max(arrivals, self.max_arrival_count)
        zones = int(status.get("zone_takeoffs") or 0)
        if self.zone_base is None:
            self.zone_base = zones
        self.max_zone_takeoffs = max(zones, self.max_zone_takeoffs)
        self.recovery_active = recovery
        if recovery:
            return
        index = status.get("circuit_index")
        if not isinstance(index, int) or not 0 <= index <= self.last:
            return
        if self.previous is None:
            if index != 0:
                return
            self.observed = [0]
        elif index != self.previous:
            if self.previous == self.last and index == 0 and count == 0:
                self.wrap = True
            elif index < self.previous:
                self.resets.append({"updated": updated, "reason": "backward index without complete wrap",
                                    "previous": self.previous, "index": index, "observed_indices": list(self.observed)})
                self.reset_progress()
                if index != 0:
                    return
                self.observed = [0]
            else:
                self.skipped.extend(range(self.previous+1, index))
                self.observed.append(index)
        self.previous = index
        for transition in self.landings:
            waypoint = transition["waypoint_index"]
            # An overlapping stair toe can look supported while still aligning
            # on the launch floor. Observe landings only after the transition.
            if waypoint < index <= waypoint+2 and waypoint in self.observed and waypoint in self.proofs:
                self.proofs[waypoint].observe(status.get("minimap"), updated, bool(status.get("minimap_seen")))
        if index == self.last or self.wrap:
            self.observe_ground(status)

    def summary(self):
        landings = [{**t, "index_observed": t["waypoint_index"] in self.observed,
                     "confirmed": bool(self.proofs.get(t["waypoint_index"]) and self.proofs[t["waypoint_index"]].confirmed),
                     "samples": deepcopy(self.proofs[t["waypoint_index"]].samples) if t["waypoint_index"] in self.proofs else []}
                    for t in self.landings]
        return {"full_wrap_observed": self.wrap, "observed_indices": self.observed,
                "unobserved_forward_indices": sorted(set(self.skipped)), "attempt_resets": self.resets,
                "transitions": landings, "all_transition_indices_observed": all(t["index_observed"] for t in landings),
                "all_landings_observed": all(t["confirmed"] for t in landings),
                "stable_ground_return": self.ground_proof.confirmed,
                "ground_samples": self.ground_proof.samples, "recovery_events": self.recovery_events,
                "startup_events": self.startup_events, "maximum_return_count": self.max_return_count,
                # A jump accepted by destination proof was not flown: say so next to the landings.
                "maximum_rejoin_count": self.max_rejoin_count,
                "arrivals_without_takeoff": self.max_arrival_count - (self.arrival_base or 0),
                # Take-offs made from a zone, outside the saved spot's own tolerance (takeoff_zones).
                "maximum_zone_takeoffs": self.max_zone_takeoffs - (self.zone_base or 0)}


def evidence_images(log, folder):
    folder.mkdir(parents=True, exist_ok=True)
    saved = {}
    for name in ("active-status.json", "bot-view.png", "bot-minimap.png", "bot-bars.png", "relay-frame.png",
                 "controller-prepared.png", "controller-prepared.json", "relay-prepared.png", "relay-prepared.json"):
        source = log / name
        try:
            modified = source.stat().st_mtime
            shutil.copyfile(source, folder / name)
            saved[name] = {"path": str(folder / name), "source_updated": modified,
                           "age_s": time.time()-modified}
        except OSError:
            pass
    return saved


def hold_at_end(bridge, record, tracker=None, last_status=None):
    reader = StatusContinuity(last_status)
    while True:
        status, error = reader.observe(read_status(bridge), time.monotonic())
        if reader.change:
            record(reader.change)
        if error:
            return {"confirmed": False, "needs_attention": True, "reason": error + " No hold request sent."}
        if status is not None:
            break
        time.sleep(.1)
    if not automatic_hold(status):
        try:
            message = bridge.hold_farming()
            record({"event": "hold_requested", "message": message, "updated": time.time()})
        except (OSError, ValueError) as error:
            if not automatic_hold(read_status(bridge)):
                return {"confirmed": False, "needs_attention": True, "reason": str(error)}
    deadline = time.monotonic()+10
    last = None
    while time.monotonic() < deadline:
        current, error = reader.observe(read_status(bridge), time.monotonic())
        if reader.change:
            record(reader.change)
        if error:
            return {"confirmed": False, "needs_attention": True, "reason": error}
        if current is None:
            time.sleep(.1)
            continue
        status = current
        if status.get("updated") != last:
            record({"event": "hold_confirmation", "status": status})
            last = status.get("updated")
        if not status.get("running") or not fresh(status):
            break
        if tracker is not None and tracker.wrap and usable_capture(status):
            tracker.observe_ground(status)
        if automatic_hold(status) and (tracker is None or not tracker.wrap or tracker.ground_proof.confirmed):
            return {"confirmed": True, "manual": bool(status.get("manual_health_hold")),
                    "state": status.get("health_protection_state"), "reason": status.get("health_hold_reason"),
                    "needs_attention": status.get("health_protection_state") in {"needs_attention", "dead"}}
        if status.get("hold_request_error"):
            return {"confirmed": False, "needs_attention": True, "reason": status["hold_request_error"]}
        time.sleep(.1)
    return {"confirmed": automatic_hold(status), "needs_attention": not automatic_hold(status),
            "reason": "Hold confirmation or stable ground observation timed out.", "status": status}


def watch(bridge, output, *, start=False, timeout_s=900):
    summary_path = output.with_suffix(".summary.json")
    summary = {"result": "preflight", "started": time.time(), "trace": str(output), "resumed_by_watch": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    tracker = None
    status_reader = None
    active_seen = False
    preserve_full_pause = False
    with output.open("x", encoding="utf-8") as stream:
        def record(value):
            stream.write(json.dumps(value, ensure_ascii=False)+"\n")
            stream.flush()
        try:
            initial, cfg, circuit, pinned = preflight(bridge, require_foreground=start)
            tracker = LapEvidence(circuit)
            continuity = CaptureContinuity()
            status_reader = StatusContinuity(initial)
            tracker.observe(initial)
            summary.update(result="watching", pid=initial.get("pid"), initial_hp=initial["hp"],
                           minimum_hp=initial["hp"], hp_taps=0, potion_key_presses={},
                           config_hashes={str(k): v for k, v in pinned.items()})
            record({"event": "preflight_passed", "status": initial})
            if start:
                (bridge.log / "resume-active").touch()
                summary["resumed_by_watch"] = True
                record({"event": "resume_requested", "updated": time.time()})
            initial_taps = initial.get("health_protection_taps", 0)
            initial_keys = initial.get("keys_sent") or {}
            last_updated, last_summary = None, 0.
            deadline = time.monotonic()+timeout_s
            while time.monotonic() < deadline:
                state, status_error = status_reader.observe(read_status(bridge), time.monotonic())
                summary["status_read_gaps"] = status_reader.events
                if status_reader.change:
                    record(status_reader.change)
                if status_error:
                    summary.update(result="needs_attention", reason=status_error)
                    break
                if state is None:
                    time.sleep(.1)
                    continue
                if state.get("pid") != initial.get("pid") or any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in pinned.items()):
                    raise RuntimeError("Controller identity, saved settings, or profile changed during observation.")
                if state.get("circuit_count") != 44 or state.get("editor_test") or state.get("potion_test"):
                    raise RuntimeError("The normal 44-point farming circuit was replaced by another session.")
                map_error = map_guard_error(state, initial.get('capture'))
                if map_error:
                    summary.update(result="needs_attention", reason=map_error)
                    break
                gap_count = len(continuity.events)
                usable = continuity.observe(state, time.monotonic())
                if len(continuity.events) > gap_count and not state.get('map_ready'):
                    folder = output.with_name(output.stem + f'-map-gap-{gap_count + 1}')
                    record({'event': 'map_reacquisition_gap', 'status': state,
                            'evidence': evidence_images(bridge.log, folder)})
                summary["capture_gaps"] = continuity.events
                if continuity.expired(time.monotonic()):
                    summary.update(result="needs_attention", reason="Minimap position or fresh game capture was unavailable for two seconds.")
                    break
                if state.get("updated") != last_updated:
                    record({"event": "sample", "elapsed_s": time.time()-summary["started"], "status": state})
                    last_updated = state["updated"]
                    hp = state.get("hp")
                    if isinstance(hp, (int, float)) and math.isfinite(hp):
                        summary["minimum_hp"] = min(summary["minimum_hp"], hp)
                    summary["hp_taps"] = max(0, int(state.get("health_protection_taps") or 0)-initial_taps)
                    summary["potion_key_presses"] = {name: max(0, (state.get("keys_sent") or {}).get(cfg["keys"].get(name), 0)-initial_keys.get(cfg["keys"].get(name), 0)) for name in ("hp_potion", "mp_potion")}
                    active_seen |= not state.get("paused")
                    if usable:
                        tracker.observe(state)
                    if state.get("health_protection_state") in {"needs_attention", "dead"}:
                        summary.update(result="needs_attention", reason=state.get("health_protection_reason"))
                        break
                    if state.get("recovery_terminal") or (state.get("circuit_state") == "blocked" and not state.get("returning_to_start")):
                        summary.update(result="route_failed", reason=state.get("recovery_reason") or state.get("circuit_reason"))
                        break
                    if active_seen and state.get("paused") and state.get("full_pause") is True:
                        summary.update(result="paused_by_user", reason="Full pause preserved; no HP hold will override it.")
                        preserve_full_pause = True
                        break
                    if tracker.wrap:
                        summary.update(result="wrap_observed", reason="Observed final point 44 returning to point 1.")
                        break
                if time.monotonic()-last_summary >= 2:
                    summary.update(tracker.summary())
                    atomic_json(summary_path, summary)
                    last_summary = time.monotonic()
                time.sleep(.1)
            else:
                summary.update(result="timeout", reason="No fully verified loop within the observation limit.")
        except (Exception, KeyboardInterrupt) as error:
            summary.update(result="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           reason=str(error) or "Observation interrupted")
        finally:
            if tracker is not None and (start or active_seen) and not preserve_full_pause:
                summary["hold"] = hold_at_end(bridge, record, tracker,
                    status_reader.last_good if status_reader is not None else None)
            elif preserve_full_pause:
                summary["hold"] = {"confirmed": False, "reason": "Explicit full pause preserved."}
            if tracker is not None:
                summary.update(tracker.summary())
                if summary["result"] == "wrap_observed":
                    complete = all(summary[k] for k in ("all_transition_indices_observed", "all_landings_observed", "stable_ground_return"))
                    summary["result"] = "completed" if complete else "incomplete_evidence"
                if summary["result"] == "completed" and not summary.get("hold", {}).get("confirmed"):
                    summary.update(result="needs_attention", reason="Loop evidence complete, but guarded hold was not confirmed.")
            summary.update(finished=time.time(), evidence=evidence_images(bridge.log, output.with_suffix("").with_name(output.stem+"-evidence")))
            atomic_json(summary_path, summary)
            record({"event": "finished", "summary": str(summary_path), "result": summary["result"]})
    print(json.dumps({"result": summary["result"], "summary": str(summary_path), "trace": str(output),
                      "hold": summary.get("hold"), "reason": summary.get("reason")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Resume a paused or manually held controller after preflight passes")
    parser.add_argument("--timeout", type=float, default=900, help="Observation seconds, at most 900 (15 minutes)")
    parser.add_argument("--output", type=Path, default=ROOT / "log/full-circuit-watch-v2.jsonl")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 900:
        parser.error("--timeout must be greater than 0 and at most 900 seconds")
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".summary.json").exists():
        output = output.with_name(output.stem+"-"+datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]+output.suffix)
    result = watch(EditorBridge(ROOT), output, start=args.start, timeout_s=args.timeout)
    return 0 if result["result"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
