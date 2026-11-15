import numpy as np

from maplebot.capture.window import crop_graphics_client


def test_graphics_crop_preserves_calibrated_client_pixels():
    raw = np.zeros((1112, 1922, 4), dtype=np.uint8)
    raw[31, 1, :3] = [10, 20, 30]
    raw[1110, 1920, :3] = [40, 50, 60]
    client = crop_graphics_client(raw, (2, 64, 1920, 1080), (1, 33, 1923, 1145))
    assert client.shape == (1080, 1920, 3)
    assert client[0, 0].tolist() == [10, 20, 30]
    assert client[-1, -1].tolist() == [40, 50, 60]
    raw[:] = 255
    assert client[0, 0].tolist() == [10, 20, 30]


def test_graphics_crop_rejects_partial_resize_frames():
    raw = np.zeros((600, 800, 4), dtype=np.uint8)
    assert crop_graphics_client(raw, (2, 64, 1920, 1080), (1, 33, 1923, 1145)) is None
    assert crop_graphics_client(raw, (0, 0, 800, 600), (1, 1, 801, 601)) is None


def test_relay_survives_reader_holding_previous_frame(tmp_path, monkeypatch):
    from scripts import relay_client

    destination = tmp_path / "frame.png"
    destination.write_bytes(b"previous")

    def locked(*_args):
        raise PermissionError("viewer is reading")

    monkeypatch.setattr(relay_client.os, "replace", locked)
    monkeypatch.setattr(relay_client.time, "sleep", lambda _seconds: None)
    assert relay_client.atomic_write(destination, b"new frame") is False
    assert destination.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [destination]
