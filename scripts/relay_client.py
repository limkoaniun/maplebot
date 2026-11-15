"""Read-only game-window relay for the settings window; never sends input."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2

from maplebot.capture.window import WindowCapture
from maplebot.capture.source import prepare_frame
from maplebot.config import load_config
from maplebot.editor_bridge import capture_signature


def atomic_write(path: Path, data: bytes) -> bool:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    for attempt in range(4):
        try:
            os.replace(temporary, path)
            return True
        except PermissionError:
            # A Windows image reader may briefly hold the destination open.
            # Drop this update after bounded retries rather than kill capture.
            if attempt < 3:
                time.sleep(0.01 * (2 ** attempt))
    temporary.unlink(missing_ok=True)
    return False


def acquire_instance_lock(path: Path):
    """Hold a one-byte OS lock for the lifetime of this process, including crashes."""
    import msvcrt

    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        handle.close()
        return None
    return handle


def main() -> int:
    os.chdir(ROOT)
    log_dir = ROOT / "log"
    log_dir.mkdir(exist_ok=True)
    instance_lock = acquire_instance_lock(log_dir / "relay.lock")
    if instance_lock is None:
        return 0  # Another settings window already owns the relay.
    stop_path = log_dir / "stop-relay"
    status_path = log_dir / "relay-status.json"
    source = None
    last_frame = None
    next_retry = 0.0
    error = None
    config_mtime = None
    last_sequence = None

    def publish(running: bool, available: bool = False) -> None:
        payload = {"running": running, "updated": time.time(), "error": error,
                   "backend": "graphics", "pid": os.getpid(),
                   "frame_available": available, "last_frame": last_frame}
        atomic_write(status_path, json.dumps(payload).encode("utf-8"))

    try:
        publish(True)
        while not stop_path.exists():
            started = time.monotonic()
            user_cfg = ROOT / "config" / "user.yaml"
            modified = user_cfg.stat().st_mtime if user_cfg.exists() else None
            if source is not None and modified != config_mtime:
                source.stop()
                source = None
                next_retry = 0.0
            if source is None and started >= next_retry:
                try:
                    cfg = load_config(str(user_cfg) if user_cfg.exists() else None)
                    config_mtime = modified
                    cfg["game_window"]["capture_backend"] = "graphics"
                    cfg["system"]["fps_capture"] = 4
                    source = WindowCapture(cfg)
                    source.start()
                    last_sequence = None
                    error = None
                except Exception as exc:
                    if source is not None:
                        source.stop()
                    source = None
                    error = str(exc)
                    next_retry = time.monotonic() + 2.0
            sample = source.get_sample(after_sequence=last_sequence) if source is not None else None
            frame = sample.frame if sample is not None else None
            if frame is not None:
                last_sequence = sample.sequence
                capture_stamp = time.time() - max(0, time.monotonic() - sample.captured_at)
                prepared = prepare_frame(frame, cfg["game_window"])
                ok, full_encoded = cv2.imencode(".png", prepared)
                if ok and atomic_write(log_dir / "relay-prepared.png", full_encoded.tobytes()):
                    metadata = {"updated": capture_stamp, "size": list(prepared.shape[1::-1]),
                                "capture": capture_signature(cfg), "sequence": sample.sequence,
                                "captured_at": sample.captured_at, "source": "relay"}
                    atomic_write(log_dir / "relay-prepared.json", json.dumps(metadata).encode("utf-8"))
                rect = cfg.get("minimap", {}).get("fixed_rect")
                if rect:
                    x, y, w, h = map(int, rect)
                    if x >= 0 and y >= 0 and x+w <= prepared.shape[1] and y+h <= prepared.shape[0]:
                        crop = prepared[y:y+h, x:x+w]
                        ok, encoded_map = cv2.imencode(".png", crop)
                        if ok and atomic_write(log_dir / "relay-editor-minimap.png", encoded_map.tobytes()):
                            metadata = {"updated": capture_stamp, "size": [w, h],
                                        "capture": capture_signature(cfg), "player_pos": None,
                                        "sequence": sample.sequence, "captured_at": sample.captured_at,
                                        "source": "relay"}
                            atomic_write(log_dir / "relay-editor-minimap.json", json.dumps(metadata).encode("utf-8"))
                height, width = frame.shape[:2]
                if width > 780:
                    frame = cv2.resize(frame, (780, round(height * 780 / width)),
                                       interpolation=cv2.INTER_AREA)
                success, encoded = cv2.imencode(".png", frame)
                if not success:
                    raise RuntimeError("could not encode the game preview")
                if atomic_write(log_dir / "relay-frame.png", encoded.tobytes()):
                    last_frame = capture_stamp
                    error = None
            elif source is not None and source.get_sample() is None:
                error = source.last_error or "Waiting for the game window; it may be minimized."
                if source.last_error:
                    source.stop()
                    source = None
                    next_retry = time.monotonic() + 2.0
            publish(True, source is not None and source.get_sample() is not None)
            time.sleep(max(0.0, 0.25 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        error = str(exc)
        return 1
    finally:
        if source is not None:
            source.stop()
        publish(False)
        instance_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
