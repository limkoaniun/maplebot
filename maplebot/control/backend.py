"""Key press backends. The real one wraps pynput; RecordingBackend is for tests."""
from __future__ import annotations

from typing import Protocol


class KeyBackend(Protocol):
    def down(self, key: str) -> None: ...
    def up(self, key: str) -> None: ...


class RecordingBackend:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.held: set[str] = set()

    def down(self, key: str) -> None:
        self.events.append(("down", key))
        self.held.add(key)

    def up(self, key: str) -> None:
        self.events.append(("up", key))
        self.held.discard(key)


class NullBackend:
    """Swallows everything; used with --no-input."""

    def down(self, key: str) -> None:
        pass

    def up(self, key: str) -> None:
        pass


def make_backend(name: str = "pynput") -> KeyBackend:
    if name == "null":
        return NullBackend()
    if name == "recording":
        return RecordingBackend()
    if name != "pynput":
        raise ValueError(f"unknown key backend {name!r}")

    from pynput.keyboard import Controller, Key  # imported lazily: Windows runtime only

    class PynputBackend:
        def __init__(self) -> None:
            self._kb = Controller()

        @staticmethod
        def _resolve(key: str):
            k = key.strip().lower()
            if hasattr(Key, k):
                return getattr(Key, k)
            return key

        def down(self, key: str) -> None:
            self._kb.press(self._resolve(key))

        def up(self, key: str) -> None:
            self._kb.release(self._resolve(key))

    return PynputBackend()
