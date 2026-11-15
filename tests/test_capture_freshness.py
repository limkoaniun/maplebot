import sys
from types import SimpleNamespace

import numpy as np

from maplebot.capture import window


def _capture(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(window, "set_dpi_aware", lambda: None)
    monkeypatch.setattr(window, "find_window_title", lambda _: "game")
    monkeypatch.setattr(window, "window_rect", lambda _: (0, 0, 8, 6))
    capture = window.WindowCapture({"game_window": {"title": "game"}}, clock=lambda: now[0])
    return capture, now


def test_reading_cached_frame_never_renews_sequence_or_timestamp(monkeypatch):
    capture, now = _capture(monkeypatch)
    pixels = np.ones((6, 8, 3), np.uint8)
    capture._publish(pixels)
    first = capture.get_sample()
    assert first.sequence == 1 and first.captured_at == 10.0
    first.frame[:] = 99
    now[0] += .4
    assert capture.get_sample(after_sequence=first.sequence) is None
    same = capture.get_sample()
    assert same.sequence == first.sequence and same.captured_at == first.captured_at
    assert np.array_equal(same.frame, pixels)
    now[0] += .36
    assert capture.get_sample() is None and capture.get_frame() is None
    assert capture.latest_sequence == 1 and capture.captured_at == 10.0


def test_new_capture_of_identical_pixels_is_a_fresh_observation(monkeypatch):
    capture, now = _capture(monkeypatch)
    pixels = np.ones((6, 8, 3), np.uint8)
    capture._publish(pixels)
    previous = capture.get_sample()
    now[0] += .8
    capture._publish(pixels.copy())
    current = capture.get_sample(after_sequence=previous.sequence)
    assert current.sequence == 2 and current.captured_at == now[0]
    assert np.array_equal(current.frame, previous.frame)


def test_minimized_or_stopped_capture_clears_cache(monkeypatch):
    capture, _ = _capture(monkeypatch)
    capture._publish(np.ones((6, 8, 3), np.uint8))
    monkeypatch.setattr(window, "window_rect", lambda _: None)
    assert capture.get_sample() is None and capture.captured_at is None
    monkeypatch.setattr(window, "window_rect", lambda _: (0, 0, 8, 6))
    assert capture.get_sample() is None
    capture._publish(np.ones((6, 8, 3), np.uint8))
    capture.stop()
    assert capture.get_frame() is None and capture.captured_at is None


def test_failed_desktop_grab_discards_previous_good_frame(monkeypatch):
    capture, _ = _capture(monkeypatch)
    capture._publish(np.ones((6, 8, 3), np.uint8))

    class FailedGrab:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def grab(self, _):
            capture._stop.set()
            raise RuntimeError("capture unavailable")

    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=FailedGrab))
    monkeypatch.setattr(window.time, "sleep", lambda _: None)
    capture._run()
    assert capture._frame is None and capture.captured_at is None


def test_graphics_processing_does_not_make_old_arrival_look_fresh(monkeypatch):
    capture, now = _capture(monkeypatch)
    observed = []

    def slow_crop(raw, _title):
        now[0] += .8
        return raw[:, :, :3].copy()

    control = SimpleNamespace(stop=lambda: None)

    class Graphics:
        def __init__(self, **_kwargs):
            self.callbacks = {}

        def event(self, callback):
            self.callbacks[callback.__name__] = callback
            return callback

        def start_free_threaded(self):
            self.callbacks["on_frame_arrived"](
                SimpleNamespace(frame_buffer=np.zeros((6, 8, 4), np.uint8)), control)
            observed.append((capture.captured_at, capture.get_sample()))
            capture._stop.set()
            return control

    monkeypatch.setattr(window, "graphics_client_frame", slow_crop)
    monkeypatch.setitem(sys.modules, "win32gui", SimpleNamespace(FindWindow=lambda *_: 1))
    monkeypatch.setitem(sys.modules, "windows_capture", SimpleNamespace(WindowsCapture=Graphics))
    capture._run_graphics()
    assert observed == [(10.0, None)]
    assert capture.latest_sequence == 1
