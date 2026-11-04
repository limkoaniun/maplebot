import sys

import cv2
import numpy as np

from maplebot.capture.source import ImageSource, VideoSource, prepare_frame
from tests.synth import make_frame


def test_image_source_returns_copy(tmp_path):
    p = tmp_path / "f.png"
    cv2.imwrite(str(p), make_frame(100, 50))
    src = ImageSource(str(p))
    src.start()
    a = src.get_frame()
    a[0, 0] = (1, 2, 3)
    b = src.get_frame()
    assert b.shape == (50, 100, 3)
    assert tuple(b[0, 0]) == (40, 40, 40)
    assert src.title == "f.png"
    src.stop()


def test_video_source_loops(tmp_path):
    p = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 32))
    for i in range(3):
        w.write(np.full((32, 64, 3), i * 60, np.uint8))
    w.release()
    src = VideoSource(str(p))
    src.start()
    frames = [src.get_frame() for _ in range(5)]
    assert all(f is not None and f.shape == (32, 64, 3) for f in frames)
    src.stop()


def test_prepare_frame_crops_title_and_resizes():
    f = make_frame(200, 120)
    f[:10] = 255
    out = prepare_frame(f, {"title_bar_px": 10, "working_size": [100, 55]})
    assert out.shape == (55, 100, 3)
    assert tuple(out[0, 0]) == (40, 40, 40)
    same = prepare_frame(f, {"title_bar_px": 0, "working_size": None})
    assert same.shape == f.shape


def test_window_capture_import_is_lazy():
    for m in ("mss", "win32gui", "pynput"):
        sys.modules.pop(m, None)
    import maplebot.capture.window  # noqa: F401
    import maplebot.control.hotkeys  # noqa: F401
    assert "mss" not in sys.modules
    assert "win32gui" not in sys.modules
    assert "pynput" not in sys.modules
