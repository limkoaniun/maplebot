"""Deterministic interleavings with recording input; no game or live key backend."""
import threading

from maplebot.types import Command
from tests.test_health_protection_integration import _Game


def test_pause_during_low_hp_image_read_cannot_reenable_healing(monkeypatch):
    game = _Game(monkeypatch)

    def read_while_pausing(*args):
        game.bot.pause_all()
        return 55, 80, 1

    monkeypatch.setattr("maplebot.bot.read_bars", read_while_pausing)
    game.step(55)
    assert game.bot.full_pause and game.bot.paused
    assert not game.bot.keyboard.enabled and not game.backend.events
    assert game.bot.health_protection.latest_observed_taps is None


def test_healthy_exit_revokes_queued_hp_before_ordinary_detection(monkeypatch):
    game = _Game(monkeypatch)
    game.step(55, apply=False)
    assert game.bot.keyboard._requests

    def detection_with_keyboard_tick(frame):
        game.bot.keyboard.apply_once(game.wall)
        return [], None

    monkeypatch.setattr(game.bot, "_detect", detection_with_keyboard_tick)
    game.step(65, apply=False)
    assert ("down", "q") not in game.backend.events
    assert game.bot.keyboard.hp_taps_sent == 0
    assert not game.bot.paused and game.bot.health_protection.state == "monitoring"


def test_pause_cannot_be_overwritten_by_early_ordinary_permission_grant(monkeypatch):
    game = _Game(monkeypatch)
    game.step(55, apply=False)
    original = game.bot.keyboard.set_command
    paused = threading.Event()
    worker = []

    def pause_worker():
        game.bot.pause_all()
        paused.set()

    def pause_before_new_ordinary_lease(cmd, **kwargs):
        if cmd == Command.NONE and not kwargs.get("navigation_only", False) and not worker:
            thread = threading.Thread(target=pause_worker, daemon=True)
            worker.append(thread)
            thread.start()
            # Without a control lock the pause completes here, before the old
            # decision remints permission. A serialized grant holds pause until
            # it finishes; detection then waits for that pause to revoke input.
            thread.join(.05)
        return original(cmd, **kwargs)

    def detection_after_pause(frame):
        if not worker:  # removing the early grant entirely also closes the window
            pause_worker()
        assert paused.wait(2), "pause was not released after ordinary input setup"
        game.bot.keyboard.apply_once(game.wall)
        return [], None

    monkeypatch.setattr(game.bot.keyboard, "set_command", pause_before_new_ordinary_lease)
    monkeypatch.setattr(game.bot, "_detect", detection_after_pause)
    try:
        game.step(65, apply=False)
    finally:
        for thread in worker:
            thread.join(2)
    assert game.bot.full_pause and game.bot.paused
    assert not game.bot.keyboard.enabled and not game.backend.events


def test_pause_after_combat_decision_cannot_enable_ordinary_input_at_step_end(monkeypatch):
    game = _Game(monkeypatch)

    def decide_then_pause(perception):
        game.bot.pause_all()
        return Command("right", action="attack")

    monkeypatch.setattr(game.bot.brain, "decide", decide_then_pause)
    game.step(75)
    assert game.bot.full_pause and game.bot.paused
    assert not game.bot.keyboard.enabled and not game.backend.events
