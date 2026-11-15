"""Freshness integration using the real capture buffer and recording-only input."""
from maplebot.bot import Bot
from maplebot.capture import window
from maplebot.config import deep_merge, load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.types import Command
from maplebot.vision.monsters import MonsterLibrary
from tests.test_bot import _full_frame


def _runtime(monkeypatch, **overrides):
    cfg = deep_merge(load_config(), {
        "health": {"enable": False},
        "route": {"circuit": {"enabled": True, "movement_test": True, "start_idx": 0,
                              "waypoints": [{"x": 80, "y": 30}, {"x": 40, "y": 30}]}}
    })
    cfg = deep_merge(cfg, overrides)
    clock = [10.0]
    monkeypatch.setattr(window, "set_dpi_aware", lambda: None)
    monkeypatch.setattr(window, "find_window_title", lambda _: "game")
    monkeypatch.setattr(window, "window_rect", lambda _: (0, 0, 1282, 693))
    source = window.WindowCapture(cfg, clock=lambda: clock[0])
    source._publish(_full_frame())
    backend = RecordingBackend()
    keyboard = KeyboardController(cfg, backend, clock=lambda: clock[0], sleep=lambda _: None)
    bot = Bot(cfg, source, keyboard, MonsterLibrary(), None)
    return bot, backend, clock


def test_duplicate_capture_does_not_advance_navigation_or_renew_command(monkeypatch):
    bot, backend, clock = _runtime(monkeypatch)
    snap = bot.step(clock[0])
    assert snap.command == Command("right")
    bot.keyboard.apply_once(clock[0])
    assert backend.held == {"right"}
    deadline = bot.keyboard._command_deadline
    nav_clock = bot.brain.circuit._clock.time
    for elapsed in (.2, .4, .6):
        clock[0] = 10.0 + elapsed
        assert bot.step(clock[0]) is None
        assert bot.capture_state == "duplicate"
        assert bot.keyboard._command_deadline == deadline
        assert bot.brain.circuit._clock.time == nav_clock
        assert bot.frame_idx == 1
    # Identical pixels from a real second capture are nevertheless a new observation.
    bot.source._publish(_full_frame())
    assert bot.step(clock[0]) is not None
    assert bot.capture_sequence == 2 and bot.frame_idx == 2
    assert bot.keyboard._command_deadline > deadline


def test_stale_capture_clears_tracking_movement_and_pending_potion(monkeypatch):
    bot, backend, clock = _runtime(monkeypatch)
    bot.step(clock[0])
    bot.keyboard.apply_once(clock[0])
    bot.keyboard.request("hp_potion")
    bot._hybrid_primary_pos = (641, 346)
    clock[0] += .8
    assert bot.step(clock[0]) is None
    assert bot.capture_state == "stale_or_missing"
    assert not backend.held and not bot.keyboard.enabled and not bot.keyboard._requests
    assert bot.player is None and bot.minimap_pos is None and bot.minimap_box is None
    assert bot.global_pos is None and bot._hybrid_primary_pos is None
    bot.keyboard.apply_once(clock[0])
    assert ("down", "1") not in backend.events


def test_slow_perception_cannot_grant_input_past_capture_expiry(monkeypatch):
    bot, backend, clock = _runtime(monkeypatch, keys={"pickup": "x"},
                                    buffs={"keys": ["a"], "cooldowns_s": [60]})
    clock[0] += .6  # fresh at acquisition, but only .15 seconds of validity remain

    def slow_detection(_frame):
        clock[0] += .2
        bot.keyboard.apply_once(clock[0])  # independent keyboard worker during perception
        return [], None

    monkeypatch.setattr(bot, "_detect", slow_detection)
    bot.step(clock[0])
    bot.keyboard.apply_once(clock[0])
    assert not backend.events  # no buff, pickup, movement, or attack


def test_resume_does_not_mint_a_lease_from_cached_capture(monkeypatch):
    bot, backend, clock = _runtime(monkeypatch, keys={"pickup": "x"})
    bot.step(clock[0])
    bot.toggle_pause()
    clock[0] += .8
    bot.toggle_pause()
    bot.keyboard.apply_once(clock[0])
    assert not backend.events
