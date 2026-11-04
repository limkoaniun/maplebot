"""Frame sources: a live window (see window.py), a static image, or a video file."""
from __future__ import annotations

import os
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    title: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_frame(self) -> np.ndarray | None: ...


class ImageSource:
    """Serves the same PNG forever (offline debugging)."""

    def __init__(self, path: str):
        self.path = path
        self.title = os.path.basename(path)
        self._frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if self._frame is None:
            raise FileNotFoundError(path)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get_frame(self) -> np.ndarray | None:
        return self._frame.copy()


class VideoSource:
    """Plays a recording in a loop."""

    def __init__(self, path: str):
        self.path = path
        self.title = os.path.basename(path)
        self._cap = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(self.path)

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_frame(self) -> np.ndarray | None:
        if self._cap is None:
            self.start()
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None


def prepare_frame(frame_bgr: np.ndarray, cfg_game_window: dict) -> np.ndarray:
    """Crop the title bar and optionally resize to the configured working size."""
    top = int(cfg_game_window.get("title_bar_px", 0) or 0)
    if top > 0 and top < frame_bgr.shape[0]:
        frame_bgr = frame_bgr[top:]
    size = cfg_game_window.get("working_size")
    if size:
        w, h = int(size[0]), int(size[1])
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (w, h):
            frame_bgr = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame_bgr
