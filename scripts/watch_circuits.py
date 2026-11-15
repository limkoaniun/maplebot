"""Observe complete farming laps (three by default); leave farming active on success.

Supervised observer only. --start sends one normal Resume request after the existing
preflight. Failure requests guarded HP Hold, unless the user explicitly paused.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import watch_full_circuit as one


class ContinuousLaps:
    """Keep both sides of a wrap so boundary proof cannot discard the next point 1."""
    def __init__(self, circuit):
        self.circuit = circuit
        self.current = one.LapEvidence(circuit)
        self.pending = None
        self.pending_since = None
        self.completed = []
        self.error = None
        self.last_updated = None

    def observe(self, status):
        updated = status.get("updated")
        if updated == self.last_updated or not one.fresh(status):
            return None
        self.last_updated = updated
        if self.pending is not None:
            if (status.get("returning_to_start") or status.get("return_count", 0)
                    or status.get("startup_return_state") in {"checking", "returning", "failed"}):
                self.error = "The lap boundary was interrupted before its ground return was verified."
                return None
            self.pending.observe_ground(status)
        self.current.observe(status)
        if self.current.wrap:
            if self.pending is not None:
                self.error = "Another wrap arrived before the preceding ground return was verified."
                return None
            self.pending = self.current
            self.pending_since = updated
            self.current = one.LapEvidence(self.circuit)
            # This same fresh wrap sample is also the first observation of the next lap.
            self.current.observe(status)
            proof = self.pending.summary()
            if not proof["all_transition_indices_observed"] or not proof["all_landings_observed"]:
                self.error = "A full wrap was observed, but not every transition and landing was verified."
                return None
        if self.pending is not None:
            if self.pending.ground_proof.confirmed:
                result = deepcopy(self.pending.summary())
                result.update(lap=len(self.completed)+1, wrap_updated=self.pending_since,
                              confirmed_updated=updated)
                self.completed.append(result)
                self.pending = None
                self.pending_since = None
                return result
            if (updated-self.pending_since > 15
                    or (status.get("circuit_index", 0) > 0 and not one.at_start(status, self.pending.first))):
                self.error = "The normal route left point 1 before a stable ground return was observed."
        return None

    def summary(self):
        return {"laps_completed": len(self.completed), "laps": deepcopy(self.completed),
                "current_lap": self.current.summary(),
                "pending_boundary": self.pending.summary() if self.pending is not None else None}


def finish_hold(bridge, record, last_status):
    """Do not turn an explicit user Pause/Stop back into an HP-only hold."""
    current = one.read_status(bridge)
    if ((one.fresh(current) and current.get("paused") and current.get("full_pause") is True)
            or (bridge.log / "pause-active").exists() or (bridge.log / "stop-active").exists()):
        return {"confirmed": False, "preserved_user_pause": True,
                "reason": "Explicit full pause or Stop preserved; no Hold request sent."}
    return one.hold_at_end(bridge, record, last_status=last_status)


def watch(bridge, output, *, start=False, timeout_s=1800, target_laps=3):
    if type(target_laps) is not int or not 1 <= target_laps <= 3:
        raise ValueError("target_laps must be an integer from 1 to 3")
    summary_path = output.with_suffix(".summary.json")
    summary = {"result": "preflight", "target_laps": target_laps, "started": time.time(),
               "trace": str(output), "resumed_by_watch": False, "success_leaves_farming_active": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    laps = reader = None
    active_seen = preserve_pause = False
    with output.open("x", encoding="utf-8") as stream:
        def record(value):
            stream.write(json.dumps(value, ensure_ascii=False)+"\n")
            stream.flush()
        try:
            initial, cfg, circuit, pinned = one.preflight(bridge, require_foreground=start)
            laps = ContinuousLaps(circuit)
            laps.observe(initial)
            continuity = one.CaptureContinuity()
            reader = one.StatusContinuity(initial)
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
                state, status_error = reader.observe(one.read_status(bridge), time.monotonic())
                summary["status_read_gaps"] = reader.events
                if reader.change:
                    record(reader.change)
                if status_error:
                    summary.update(result="needs_attention", reason=status_error)
                    break
                if state is None:
                    time.sleep(.1)
                    continue
                if (state.get("pid") != initial.get("pid")
                        or any(hashlib.sha256(p.read_bytes()).hexdigest() != digest for p, digest in pinned.items())):
                    raise RuntimeError("Controller identity, saved settings, or profile changed during observation.")
                if state.get("circuit_count") != 44 or state.get("editor_test") or state.get("potion_test"):
                    raise RuntimeError("The normal 44-point farming circuit was replaced by another session.")
                if active_seen and state.get("paused") and state.get("full_pause") is True:
                    summary.update(result="paused_by_user", reason="Full pause preserved; no HP hold will override it.")
                    preserve_pause = True
                    break
                map_error = one.map_guard_error(state, initial.get("capture"))
                if map_error:
                    summary.update(result="needs_attention", reason=map_error)
                    break
                gap_count = len(continuity.events)
                usable = continuity.observe(state, time.monotonic())
                if len(continuity.events) > gap_count and not state.get("map_ready"):
                    folder = output.with_name(output.stem+f"-map-gap-{gap_count+1}")
                    record({"event": "map_reacquisition_gap", "status": state,
                            "evidence": one.evidence_images(bridge.log, folder)})
                summary["capture_gaps"] = continuity.events
                if continuity.expired(time.monotonic()):
                    summary.update(result="needs_attention", reason="Minimap position or fresh game capture was unavailable for two seconds.")
                    break
                if state.get("updated") != last_updated:
                    record({"event": "sample", "lap": len(laps.completed)+1,
                            "elapsed_s": time.time()-summary["started"], "status": state})
                    last_updated = state["updated"]
                    hp = state.get("hp")
                    if isinstance(hp, (int, float)) and math.isfinite(hp):
                        summary["minimum_hp"] = min(summary["minimum_hp"], hp)
                    summary["hp_taps"] = max(0, int(state.get("health_protection_taps") or 0)-initial_taps)
                    summary["potion_key_presses"] = {
                        name: max(0, (state.get("keys_sent") or {}).get(cfg["keys"].get(name), 0)
                                  -initial_keys.get(cfg["keys"].get(name), 0))
                        for name in ("hp_potion", "mp_potion")}
                    active_seen |= not state.get("paused")
                    if state.get("health_protection_state") in {"needs_attention", "dead"}:
                        summary.update(result="needs_attention", reason=state.get("health_protection_reason"))
                        break
                    if (state.get("startup_return_state") == "failed" or state.get("recovery_terminal")
                            or (state.get("circuit_state") == "blocked" and not state.get("returning_to_start"))):
                        summary.update(result="route_failed", reason=state.get("startup_return_reason")
                                       or state.get("recovery_reason") or state.get("circuit_reason"))
                        break
                    if active_seen and state.get("paused") and state.get("manual_health_hold") and state.get("startup_return_state") not in {"checking", "returning"}:
                        summary.update(result="held_by_user", reason="Farming entered a manual HP hold; observation ended.")
                        break
                    completed = laps.observe(state) if usable else None
                    if laps.error:
                        summary.update(result="incomplete_evidence", reason=laps.error)
                        break
                    if completed is not None:
                        folder = output.with_name(output.stem+f"-lap-{completed['lap']}-evidence")
                        completed["evidence"] = one.evidence_images(bridge.log, folder)
                        laps.completed[-1] = completed
                        lap_path = output.with_name(output.stem+f".lap-{completed['lap']}.json")
                        one.atomic_json(lap_path, completed)
                        record({"event": "lap_completed", "lap": completed["lap"], "summary": str(lap_path),
                                "updated": state["updated"], "continuing_without_pause": True})
                        if len(laps.completed) == target_laps:
                            summary.update(result="completed", reason=f"{target_laps} complete farming lap(s) verified; farming remains active.",
                                           farming_active_at_completion=not state.get("paused"), final_status=state)
                            break
                if time.monotonic()-last_summary >= 2:
                    summary.update(laps.summary())
                    one.atomic_json(summary_path, summary)
                    last_summary = time.monotonic()
                time.sleep(.1)
            else:
                summary.update(result="timeout", reason=f"{target_laps} verified lap(s) were not completed within the observation limit.")
        except (Exception, KeyboardInterrupt) as error:
            summary.update(result="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           reason=str(error) or "Observation interrupted")
        finally:
            if laps is not None:
                summary.update(laps.summary())
            if summary["result"] == "completed":
                summary["hold"] = {"requested": False, "reason": "Successful observation leaves farming active."}
            elif preserve_pause:
                summary["hold"] = {"confirmed": False, "preserved_user_pause": True, "reason": "Explicit full pause preserved."}
            elif laps is not None and (summary["resumed_by_watch"] or active_seen):
                summary["hold"] = finish_hold(bridge, record, reader.last_good if reader else None)
            summary.update(finished=time.time(), evidence=one.evidence_images(
                bridge.log, output.with_name(output.stem+"-evidence")))
            one.atomic_json(summary_path, summary)
            record({"event": "finished", "summary": str(summary_path), "result": summary["result"]})
    print(json.dumps({"result": summary["result"], "laps_completed": summary.get("laps_completed", 0),
                      "summary": str(summary_path), "trace": str(output), "hold": summary.get("hold"),
                      "reason": summary.get("reason")}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", action="store_true", help="Request normal Resume once after preflight")
    parser.add_argument("--timeout", type=float, default=1800, help="Maximum seconds, no more than 1800")
    parser.add_argument("--laps", type=int, choices=(1, 2, 3), default=3,
                        help="Number of fully verified laps to observe (default: 3)")
    parser.add_argument("--output", type=Path, default=ROOT / "log/circuits-watch.jsonl")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 0 < args.timeout <= 1800:
        parser.error("--timeout must be greater than zero and at most 1800 seconds")
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".summary.json").exists():
        output = output.with_name(output.stem+"-"+datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:6]+output.suffix)
    result = watch(one.EditorBridge(ROOT), output, start=args.start, timeout_s=args.timeout,
                   target_laps=args.laps)
    return 0 if result["result"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
