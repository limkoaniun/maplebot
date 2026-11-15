"""Run the currently saved route once, retaining every observed floor transition.

This developer harness queues the normal movement-test protocol. It never edits
user.yaml or emits input itself; the runner performs its own paused preflight.
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

import cv2
from maplebot.editor_bridge import EditorBridge, atomic_bytes, atomic_json, capture_signature
from maplebot.nav.circuit import MinimapCircuit
from verify_route import LandingProof, PinnedContext, health, observed_position, platform_surface, read_json


TERMINAL = {"completed", "blocked", "cancelled", "error"}


def fresh(status, age=3):
    value = status.get("updated")
    return (isinstance(value, (int, float)) and math.isfinite(value)
            and -1 <= time.time() - value <= age)


def on_platform(pos, platform, tolerance=2):
    return (isinstance(pos, (list, tuple)) and len(pos) == 2
            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pos)
            and platform["left"] <= pos[0] <= platform["right"]
            and platform_surface(platform).support_error(pos) <= tolerance)


def platform_bottom(platform):
    """Lowest support height includes valleys between the platform endpoints."""
    return max([platform["y"], platform.get("right_y", platform["y"])]
               + [point[1] for point in platform.get("knots", [])])


def prepare_lap(bridge, min_hp=25, *, require_ground=True):
    """Build a request without writes; partial runs still need chunk-source preflight."""
    min_hp = max(25.0, float(min_hp))
    status = read_json(bridge.log / "active-status.json")
    if not status.get("running") or not fresh(status) or not status.get("paused"):
        raise ValueError("The controller must be running, fresh, and paused before a lap test.")
    if (bridge.log / "editor-test-request.json").exists() or status.get("editor_test"):
        raise ValueError("Another movement test is already queued or running.")
    cfg = bridge._cfg()
    signature = capture_signature(cfg)
    if status.get("capture") != signature:
        raise ValueError("The controller has not loaded the current capture settings.")
    plan, reference, pos = bridge.load_current()
    if plan.get("context_warning"):
        raise ValueError(plan["context_warning"])
    bridge._check_context(plan, reference, cfg)
    circuit = deepcopy(cfg.get("route", {}).get("circuit", {}))
    points, platforms = circuit.get("waypoints", []), circuit.get("platforms", [])
    if not points or not platforms or not circuit.get("enabled", False):
        raise ValueError("The saved circuit must contain an enabled route and platforms.")
    first = points[0]
    candidates = [p for p in platforms if on_platform((first["x"], first["y"]), p, .75)]
    source = next((p for p in candidates if p.get("id") == first.get("platform_id")),
                  candidates[0] if len(candidates) == 1 else None)
    if (source is None or first.get("action", "walk") != "walk"
            or (require_ground and platform_bottom(source) != max(platform_bottom(p) for p in platforms))):
        raise ValueError("The saved lap must start with a walk on its ground platform.")
    if require_ground and not on_platform(pos, source):
        raise ValueError("Move onto the saved route's ground platform, then pause before testing.")
    hp = health(bridge.log)
    if hp is None or not math.isfinite(hp) or hp < min_hp:
        raise ValueError(f"Health is below {min_hp:g}% or unreadable before the lap: {hp}")
    circuit.update(enabled=True, loop=False, movement_test=True, start_idx=0, max_retries=1)
    circuit["timeout_s"] = min(22.0, max(1.0, float(circuit.get("timeout_s", 22))))
    MinimapCircuit(circuit)  # validate the saved executor configuration before queuing
    test_id = uuid.uuid4().hex
    request = {"version": 1, "test_id": test_id, "created": time.time(),
               "circuit": circuit, "capture": signature,
               "reference": f"editor-test-{test_id}.png",
               "source_platform": deepcopy(source), "connection_id": "saved-route-lap"}
    return request, reference, hp


def save_images(log, evidence, label):
    saved = {}
    for filename, suffix in (("relay-frame.png", "game"), ("bot-view.png", "detection"),
                             ("editor-minimap.png", "minimap")):
        source = log / filename
        if not source.exists():
            continue
        target = evidence / f"{label}-{suffix}.png"
        try:
            shutil.copyfile(source, target)
            saved[suffix] = {"file": target.name, "updated": source.stat().st_mtime}
        except OSError:
            pass  # the trace still records the transition if a producer replaces this image
    return saved


def transitions(circuit):
    platforms = circuit["platforms"]
    result = []
    for index, point in enumerate(circuit["waypoints"]):
        if point.get("action", "walk") not in {"climb", "drop", "jump", "ramp"}:
            continue
        candidates = [p for p in platforms if on_platform((point["x"], point["y"]), p, .75)]
        destination = next((p for p in candidates if p.get("id") == point.get("to_platform")),
                           candidates[0] if len(candidates) == 1 else None)
        result.append({"waypoint_index": index, "action": point["action"],
                       "connection_id": point.get("connection_id"),
                       "destination": deepcopy(destination), "confirmed": False})
    return result


def split_lap(request, max_points=8):
    """Preserve route order, sharing a terminal walk as each next source anchor."""
    points = request["circuit"]["waypoints"]
    if points[-1].get("action", "walk") != "walk":
        raise ValueError("A chunked lap must end at a saved walk waypoint.")
    chunks, start = [], 0
    while start < len(points) - 1:
        target = min(start + max_points - 1, len(points) - 1)
        candidates = [i for i in range(start + 1, target + 1)
                      if points[i].get("action", "walk") == "walk"]
        if not candidates:
            raise ValueError("Too many transitions without a walk anchor to split the saved route.")
        end = candidates[-1]
        chunk = deepcopy(request)
        chunk["circuit"]["waypoints"] = deepcopy(points[start:end + 1])
        first = points[start]
        matches = [p for p in request["circuit"]["platforms"]
                   if on_platform((first["x"], first["y"]), p, .75)]
        source = next((p for p in matches if p.get("id") == first.get("platform_id")),
                      matches[0] if len(matches) == 1 else None)
        if source is None:
            raise ValueError(f"Waypoint {start} does not identify a unique source platform.")
        chunk["source_platform"] = deepcopy(source)
        chunk["test_id"] = uuid.uuid4().hex
        chunk["reference"] = f"editor-test-{chunk['test_id']}.png"
        chunk["connection_id"] = f"saved-route-lap:chunk-{len(chunks) + 1}"
        chunk["lap_start_index"], chunk["lap_end_index"] = start, end
        chunks.append(chunk)
        start = end
    return chunks


def _prepare_chunk(bridge, request, min_hp):
    status = read_json(bridge.log / "active-status.json")
    if (not status.get("running") or not status.get("paused") or not fresh(status)
            or status.get("editor_test") or (bridge.log / "editor-test-request.json").exists()):
        raise ValueError("The previous test must finish and restore the paused controller first.")
    cfg = bridge._cfg()
    if capture_signature(cfg) != request["capture"] or status.get("capture") != request["capture"]:
        raise ValueError("Capture settings changed before the next lap chunk.")
    plan, reference, pos = bridge.load_current()
    if plan.get("context_warning"):
        raise ValueError(plan["context_warning"])
    bridge._check_context(plan, reference, cfg)
    if not on_platform(pos, request["source_platform"]):
        raise ValueError("The character is not on the next chunk's starting platform.")
    hp = health(bridge.log)
    if hp is None or not math.isfinite(hp) or hp < min_hp:
        raise ValueError(f"Health is below {min_hp:g}% or unreadable before the chunk: {hp}")
    request["created"] = time.time()
    return reference, hp


def _run_chunk(bridge, request, min_hp, evidence, deadline_s, pin=None):
    reference, hp = _prepare_chunk(bridge, request, min_hp)
    if pin is not None:
        pin.check(read_json(bridge.log / "active-status.json"))
    evidence.mkdir(parents=True)
    config_hash = hashlib.sha256(bridge.user_config.read_bytes()).hexdigest()
    summary = {"test_id": request["test_id"], "started": time.time(), "initial_hp": hp,
               "config_sha256_before": config_hash, "result": "preparing",
               "transitions": transitions(request["circuit"]), "floors_observed": [],
               "restored_paused": False}
    proofs = {landing["waypoint_index"]: LandingProof(landing["destination"])
              for landing in summary["transitions"] if landing["destination"] is not None}
    final_point = request["circuit"]["waypoints"][-1]
    final_platform = next((p for p in request["circuit"]["platforms"]
                           if on_platform((final_point["x"], final_point["y"]), p, .75)), None)
    finish_proof = LandingProof(final_platform, stationary=True) if final_platform is not None else None
    samples, queued = [], False
    started, previous, hp_missing_since = time.monotonic(), None, None
    result, error = {}, None
    try:
        ok, encoded = cv2.imencode(".png", reference)
        if not ok:
            raise RuntimeError("Could not encode the current minimap reference.")
        atomic_bytes(bridge.log / request["reference"], encoded.tobytes())
        atomic_json(evidence / "request.json", request)
        save_images(bridge.log, evidence, "start")
        bridge.pause()
        (bridge.log / "stop-editor-test").unlink(missing_ok=True)
        atomic_json(bridge.log / "editor-test-status.json", {"test_id": request["test_id"],
                    "state": "queued", "message": "Testing the saved route once; movement only.",
                    "updated": time.time()})
        atomic_json(bridge.log / "editor-test-request.json", request)
        queued = True
        print(json.dumps({"state": "queued", "test_id": request["test_id"],
                          "waypoints": len(request["circuit"]["waypoints"]),
                          "evidence": str(evidence), "hp": hp}), flush=True)
        while time.monotonic() - started < deadline_s:
            state = read_json(bridge.log / "active-status.json")
            test = bridge.test_status()
            hp = health(bridge.log)
            elapsed = time.monotonic() - started
            sample = {key: state.get(key) for key in ("updated", "minimap", "minimap_seen",
                      "paused", "foreground", "input_blocked", "circuit_index", "circuit_state",
                      "command", "keys_sent")}
            sample.update(elapsed=elapsed, hp=hp, test=deepcopy(test))
            samples.append(sample)
            metadata = pin.check(state) if pin is not None else {}
            matching = state.get("editor_test", {}).get("test_id") == request["test_id"]
            if matching and fresh(state) and state.get("minimap_seen"):
                pos, index = state.get("minimap"), state.get("circuit_index")
                for platform in request["circuit"]["platforms"]:
                    label = platform.get("id", str(platform["y"]))
                    if on_platform(pos, platform, .75) and label not in summary["floors_observed"]:
                        summary["floors_observed"].append(label)
                if isinstance(index, int):
                    for landing in summary["transitions"]:
                        target = landing["destination"]
                        proof = proofs.get(landing["waypoint_index"])
                        if (not landing["confirmed"] and proof is not None
                                and landing["waypoint_index"] < index <= landing["waypoint_index"] + 2
                                and proof.observe(*observed_position(state, metadata))):
                            landing.update(confirmed=True, elapsed=elapsed, position=pos,
                                           observed_index=index, status_updated=state["updated"],
                                           observations=deepcopy(proof.samples))
                            landing["images"] = save_images(bridge.log, evidence,
                                f"landing-{landing['waypoint_index']:02d}")
                            atomic_json(evidence / "summary.json", summary)
            marker = (test.get("state"), state.get("circuit_index"), state.get("circuit_state"))
            if marker != previous:
                print(json.dumps({"test": marker[0], "step": marker[1], "state": marker[2],
                                  "position": sample["minimap"], "hp": hp}), flush=True)
                previous = marker
            if test.get("test_id") == request["test_id"] and test.get("state") in TERMINAL:
                result = test
                break
            if not state.get("running") or not fresh(state, 6):
                raise RuntimeError("The controller stopped reporting a fresh status during the lap.")
            if hp is None or not math.isfinite(hp):
                hp_missing_since = time.monotonic() if hp_missing_since is None else hp_missing_since
                if time.monotonic() - hp_missing_since > 2:
                    raise RuntimeError("Health became unreadable during the lap.")
            elif hp < min_hp:
                raise RuntimeError(f"Health fell below {min_hp:g}% during the lap ({hp:.1f}%).")
            else:
                hp_missing_since = None
            time.sleep(.2)
        if result.get("state") != "completed":
            raise RuntimeError(f"Saved-route lap did not complete: {result or 'harness deadline'}")
        summary["result"] = "completed"
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
        summary.update(result="failed", error=str(caught) or type(caught).__name__)
    finally:
        if queued:
            bridge.stop_test()
            restore_deadline = time.monotonic() + 10
            while time.monotonic() < restore_deadline:
                state = read_json(bridge.log / "active-status.json")
                try:
                    metadata = pin.check(state) if pin is not None else {}
                except (OSError, ValueError, RuntimeError) as context_error:
                    summary["context_error"] = str(context_error)
                    break
                if (state.get("running") and state.get("paused") and fresh(state)
                        and "editor_test" not in state):
                    summary["restored_paused"] = True
                    summary["final_position"] = state.get("minimap")
                    if finish_proof is not None:
                        finish_proof.observe(*observed_position(state, metadata))
                        summary["final_floor_observations"] = deepcopy(finish_proof.samples)
                    # A zero-dwell final walk can complete between status publishes.
                    # Confirm that terminal landing using the fresh restored snapshot.
                    if result.get("state") == "completed" and state.get("minimap_seen"):
                        final_index = len(request["circuit"]["waypoints"]) - 1
                        for landing in summary["transitions"]:
                            proof = proofs.get(landing["waypoint_index"])
                            if (not landing["confirmed"] and landing["destination"] is not None
                                    and 0 < final_index - landing["waypoint_index"] <= 2
                                    and proof.observe(*observed_position(state, metadata))):
                                landing.update(confirmed=True, position=state["minimap"],
                                    status_updated=state["updated"], elapsed=time.monotonic() - started,
                                    observations=deepcopy(proof.samples),
                                    confirmation="completed chunk and stable fresh destination observations")
                                landing["images"] = save_images(bridge.log, evidence,
                                    f"landing-{landing['waypoint_index']:02d}")
                        for platform in request["circuit"]["platforms"]:
                            label = platform.get("id", str(platform["y"]))
                            if on_platform(state.get("minimap"), platform, .75) and label not in summary["floors_observed"]:
                                summary["floors_observed"].append(label)
                    pending_terminal = [t for t in summary["transitions"] if not t["confirmed"]
                                        and 0 < len(request["circuit"]["waypoints"]) - 1 - t["waypoint_index"] <= 2]
                    if (result.get("state") != "completed"
                            or (not pending_terminal and finish_proof is not None and finish_proof.confirmed)):
                        break
                time.sleep(.2)
        summary.update(finished=time.time(), elapsed_s=time.monotonic() - started,
                       runtime_result=result, final_hp=health(bridge.log))
        summary["config_sha256_after"] = hashlib.sha256(bridge.user_config.read_bytes()).hexdigest()
        summary["config_unchanged"] = summary["config_sha256_after"] == config_hash
        summary["images"] = save_images(bridge.log, evidence, "end")
        summary["all_transitions_observed"] = all(t["confirmed"] for t in summary["transitions"])
        summary["final_floor_verified"] = finish_proof is not None and finish_proof.confirmed
        if not all(summary[key] for key in ("restored_paused", "config_unchanged", "all_transitions_observed", "final_floor_verified")):
            summary["result"] = "failed"
        atomic_json(evidence / "trace.json", samples)
        atomic_json(evidence / "summary.json", summary)
        print(json.dumps({"result": summary["result"], "restored_paused": summary["restored_paused"],
                          "all_transitions_observed": summary["all_transitions_observed"],
                          "floors_observed": summary["floors_observed"], "evidence": str(evidence)}), flush=True)
    if error is not None:
        raise error
    if not summary["restored_paused"]:
        raise RuntimeError("The normal paused controller did not return after the lap.")
    if not summary["config_unchanged"]:
        raise RuntimeError("The saved configuration changed while the lap was running.")
    if not summary["all_transitions_observed"]:
        raise RuntimeError("The runner completed, but one or more landings lack observed evidence.")
    if not summary["final_floor_verified"]:
        raise RuntimeError("The runner completed, but its final paused position is not stably on the expected floor.")
    return summary


def run_lap(bridge, min_hp=25, deadline_s=95, *, start_chunk=1):
    if type(start_chunk) is not int or start_chunk < 1:
        raise ValueError("Start chunk must be a positive whole number.")
    min_hp = max(25.0, min_hp)
    # Pin before preparing: a concurrent profile/config edit cannot turn a
    # previously compiled request into a test under a different saved baseline.
    pin = PinnedContext(bridge)
    request, _, hp = prepare_lap(bridge, min_hp, require_ground=start_chunk == 1)
    chunks = split_lap(request)
    if start_chunk > len(chunks):
        raise ValueError(f"Start chunk must be between 1 and {len(chunks)} for this saved route.")
    selected = chunks[start_chunk - 1:]
    pin.check(read_json(bridge.log / "active-status.json"))
    # Validate the actual chosen anchor before writing evidence or test markers.
    _prepare_chunk(bridge, selected[0], min_hp)
    evidence = bridge.log / ("lap-check-" + datetime.now().strftime("%Y%m%d-%H%M%S")
                             + "-" + request["test_id"][:6])
    evidence.mkdir(parents=True)
    initial_hash = hashlib.sha256(bridge.user_config.read_bytes()).hexdigest()
    partial = start_chunk > 1
    summary = {"mode": "partial lap" if partial else "chunked lap", "start_chunk": start_chunk,
               "full_lap_completed": False, "uninterrupted": False,
               "all_requested_chunks_completed": False,
               "verification_scope": (f"Only chunks {start_chunk} through {len(chunks)} of the saved route."
                                      if partial else "All saved chunks, with paused checks between chunks."),
               "started": time.time(), "result": "running",
               "initial_hp": hp, "waypoints": len(request["circuit"]["waypoints"]),
               "chunk_count": len(chunks), "requested_chunk_count": len(selected),
               "requested_chunks": list(range(start_chunk, len(chunks) + 1)),
               "chunks": [], "floors_observed": [],
               "config_sha256_before": initial_hash}
    atomic_json(evidence / "saved-request.json", request)
    atomic_json(evidence / "summary.json", summary)
    error = None
    try:
        for number, chunk in enumerate(selected, start_chunk):
            if hashlib.sha256(bridge.user_config.read_bytes()).hexdigest() != initial_hash:
                raise RuntimeError("The saved configuration changed between lap chunks.")
            child_dir = evidence / f"chunk-{number:02d}"
            print(json.dumps({"chunk": number, "of": len(chunks),
                              "start_index": chunk["lap_start_index"],
                              "end_index": chunk["lap_end_index"]}), flush=True)
            try:
                pin.check(read_json(bridge.log / "active-status.json"))
                result = _run_chunk(bridge, chunk, min_hp, child_dir, deadline_s, pin=pin)
            finally:
                if (child_dir / "summary.json").exists():
                    record = read_json(child_dir / "summary.json")
                    record.update(chunk=number, start_index=chunk["lap_start_index"],
                                  end_index=chunk["lap_end_index"], evidence=child_dir.name)
                    for landing in record.get("transitions", []):
                        landing["global_waypoint_index"] = chunk["lap_start_index"] + landing["waypoint_index"]
                    summary["chunks"].append(record)
                    for floor in record.get("floors_observed", []):
                        if floor not in summary["floors_observed"]:
                            summary["floors_observed"].append(floor)
                    atomic_json(evidence / "summary.json", summary)
        summary["result"] = "completed"
    except (Exception, KeyboardInterrupt) as caught:
        error = caught
        summary.update(result="failed", error=str(caught) or type(caught).__name__)
    finally:
        summary.update(finished=time.time(), final_hp=health(bridge.log),
                       config_unchanged=hashlib.sha256(bridge.user_config.read_bytes()).hexdigest() == initial_hash)
        status = read_json(bridge.log / "active-status.json")
        summary["restored_paused"] = bool(status.get("running") and status.get("paused")
            and fresh(status) and "editor_test" not in status)
        summary["all_requested_chunks_completed"] = (len(summary["chunks"]) == len(selected)
            and all(c.get("result") == "completed" for c in summary["chunks"]))
        if summary["result"] == "completed" and not all(summary[key] for key in (
                "config_unchanged", "restored_paused", "all_requested_chunks_completed")):
            summary["result"] = "failed"
            error = RuntimeError("The requested chunks lack complete evidence or the paused saved context was not restored.")
            summary["error"] = str(error)
        summary["full_lap_completed"] = not partial and summary["result"] == "completed"
        atomic_json(evidence / "summary.json", summary)
        traces = []
        for number, chunk in enumerate(chunks, 1):
            path = evidence / f"chunk-{number:02d}" / "trace.json"
            if path.exists():
                for row in read_json(path):
                    row["chunk"] = number
                    index = row.get("circuit_index")
                    row["global_waypoint_index"] = (chunk["lap_start_index"] + index
                                                     if isinstance(index, int) else None)
                    traces.append(row)
        atomic_json(evidence / "trace.json", traces)
        print(json.dumps({"result": summary["result"], "mode": summary["mode"],
                          "start_chunk": start_chunk, "full_lap_completed": summary["full_lap_completed"],
                          "chunks_completed": sum(c["result"] == "completed" for c in summary["chunks"]),
                          "chunk_count": len(chunks), "requested_chunk_count": len(selected),
                          "floors_observed": summary["floors_observed"],
                          "restored_paused": summary["restored_paused"], "evidence": str(evidence)}), flush=True)
    if error is not None:
        raise error
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-hp", type=float, default=25)
    parser.add_argument("--deadline", type=float, default=95,
                        help="Wall-clock limit per chunk; the runner's 90-second limit still applies")
    parser.add_argument("--start-chunk", type=int, default=1,
                        help="Resume at this saved chunk's platform; values above 1 verify a partial lap only")
    args = parser.parse_args()
    if not math.isfinite(args.min_hp) or not math.isfinite(args.deadline) or args.deadline <= 0:
        parser.error("Health threshold and deadline must be finite positive values.")
    if args.start_chunk < 1:
        parser.error("Start chunk must be a positive whole number.")
    run_lap(EditorBridge(ROOT), max(25.0, args.min_hp), args.deadline, start_chunk=args.start_chunk)


if __name__ == "__main__":
    main()
