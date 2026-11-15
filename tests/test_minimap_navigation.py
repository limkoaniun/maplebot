"""Live-source route gating with synthetic frames and a recording keyboard only."""
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from maplebot.map_profiles import MapProfileRecognizer, apply_profile
from maplebot.types import Command, PlayerFix
from tests.test_bot import _live_bot
from tests.test_map_profiles import example


@pytest.fixture
def routed(monkeypatch):
    cfg, profile, frame = example()
    cfg.setdefault("safety", {})["max_missing_frames"] = 1
    cfg["keys"].update(attack="z", normal_attack="alt", jump="c", hp_potion="q", mp_potion="a", pickup="x")
    cfg["health"].update(enable=True, warmup_reads=0)
    cfg = apply_profile(cfg, profile)
    x, y, _, _ = cfg["minimap"]["fixed_rect"]
    cv2.circle(frame, (x+30, y+50), 2, (0, 255, 255), -1)
    bot, backend = _live_bot(frame, cfg)
    bot.map_recognizer = MapProfileRecognizer([profile], stable_frames=1)
    bot.map_ready = False
    monkeypatch.setattr(bot, "_locate_player", lambda image: None)
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (1., 1., None))
    monkeypatch.setattr(bot.health, "update", Mock(side_effect=AssertionError("potions used without screen lock")))
    monkeypatch.setattr(bot.brain, "decide", Mock(side_effect=AssertionError("combat brain used without screen lock")))
    monkeypatch.setattr(bot.brain.circuit, "step", Mock(wraps=bot.brain.circuit.step))
    monkeypatch.setattr(bot.brain.loot, "suspend", Mock(wraps=bot.brain.loot.suspend))
    return bot, backend, profile, frame


def test_route_can_start_behind_obstruction_without_any_previous_character_fix(routed):
    bot, backend, _, _ = routed
    assert bot.player is None and bot._hybrid_primary_pos is None
    bot.brain.target = "old combat target"
    snap = bot.step(0.)
    assert snap.command == Command("left")
    assert snap.player is None and snap.minimap_pos == (30, 50)
    assert snap.hp == snap.mp == 1. and bot.map_ready
    assert bot._minimap_route_active and bot._input_blocked and bot.keyboard.enabled
    assert bot.brain.state.startswith("minimap_route_") and bot.brain.target is None
    bot.brain.circuit.step.assert_called_once_with((30, 50), 0., False)
    bot.brain.loot.suspend.assert_called_once_with(0.)
    bot.brain.decide.assert_not_called()
    bot.health.update.assert_not_called()
    bot.keyboard.apply_once(0.)
    assert backend.held == {"left"}
    assert backend.events == [("down", "left")]


@pytest.mark.parametrize("missing", ["map_guard", "circuit"])
def test_missing_character_does_not_allow_ordinary_or_unguarded_navigation(routed, missing):
    bot, backend, _, _ = routed
    if missing == "map_guard":
        bot.map_recognizer = None
    else:
        bot.brain.circuit = None
    snap = bot.step(0.)
    assert snap.command == Command.NONE and not bot._minimap_route_active
    assert bot._input_blocked and not bot.keyboard.enabled
    bot.keyboard.apply_once(0.)
    assert not backend.events
    bot.brain.decide.assert_not_called()
    bot.health.update.assert_not_called()


@pytest.mark.parametrize("loss", ["dot", "focus", "map", "settings", "capture", "pause"])
def test_minimap_route_stops_immediately_when_any_required_observation_is_lost(routed, monkeypatch, loss):
    bot, backend, profile, frame = routed
    assert bot.step(0.).command == Command("left")
    bot.keyboard.apply_once(0.)
    assert backend.held == {"left"}
    if loss == "dot":
        x, y, w, h = bot.cfg["minimap"]["fixed_rect"]
        bot.source.frame[y:y+h, x:x+w] = profile.reference_bgr
    elif loss == "focus":
        monkeypatch.setattr(bot.keyboard, "is_active", lambda: False)
    elif loss == "map":
        bot.source.frame = np.zeros_like(frame)
    elif loss == "settings":
        bot.cfg["route"]["circuit"]["platforms"][0]["y"] += 1
    elif loss == "capture":
        bot.source.frame = None
    else:
        bot.toggle_pause()
        assert not backend.held  # pause releases before the next perception tick
    snap = bot.step(.1)
    assert snap is None or snap.command == Command.NONE
    assert not bot._minimap_route_active and not bot.keyboard.enabled
    assert not backend.held and backend.events == [("down", "left"), ("up", "left")]
    assert bot.brain.circuit.step.call_count == 1
    bot.brain.decide.assert_not_called()
    bot.health.update.assert_not_called()
    if loss in ("map", "settings", "capture"):
        assert not bot.map_ready


def test_map_confirmation_frames_do_not_move_and_pause_is_preserved(routed):
    bot, backend, profile, _ = routed
    bot.map_recognizer = MapProfileRecognizer([profile], stable_frames=3)
    bot.paused = True
    for now in (0., .1, .2, .3):
        snap = bot.step(now)
        assert snap.command == Command.NONE and not bot._minimap_route_active
        bot.keyboard.apply_once(now)
    assert bot.map_ready and bot.paused and not backend.events
    bot.brain.circuit.step.assert_not_called()


def test_current_screen_fix_returns_control_to_combat_policy(routed, monkeypatch):
    bot, _, _, _ = routed
    bot.step(0.)
    monkeypatch.setattr(bot, "_locate_player", lambda frame: PlayerFix(200, 160, 1., "test"))
    monkeypatch.setattr(bot.brain, "decide", Mock(return_value=Command.NONE))
    monkeypatch.setattr(bot.health, "update", Mock(return_value=[]))
    snap = bot.step(.1)
    assert snap.player == (200, 160)
    assert not bot._minimap_route_active and not bot._input_blocked
    bot.brain.decide.assert_called_once()
    bot.health.update.assert_not_called()


def test_switch_to_route_mode_clears_previous_combat_and_potions_before_detection(routed, monkeypatch):
    bot, backend, _, _ = routed
    bot.keyboard.set_enabled(True)
    bot.keyboard.set_command(Command("right", "none", "attack"))
    bot.keyboard.request("hp_potion")

    def perception_tick(frame):
        # Simulate the keyboard thread ticking while screen perception is busy.
        bot.keyboard.apply_once(0.)
        return [], None

    monkeypatch.setattr(bot, "_detect", perception_tick)
    bot.step(0.)
    bot.keyboard.apply_once(.1)
    assert backend.held == {"left"}
    assert {key for event, key in backend.events if event == "down"} == {"left"}
    assert not bot.keyboard._requests


def test_reacquired_fix_defers_health_without_consuming_retry_then_requests_next_frame(routed, monkeypatch):
    bot, backend, _, _ = routed
    bot.step(0.)
    bot.keyboard.apply_once(0.)
    monkeypatch.setattr(bot, "_locate_player", lambda frame: PlayerFix(200, 160, 1., "test"))
    monkeypatch.setattr(bot.brain, "decide", Mock(return_value=Command.NONE))
    monkeypatch.setattr(bot.health, "update", Mock(return_value=["hp_potion"]))
    bot.step(.1)
    bot.health.update.assert_not_called()  # retry state is untouched while keyboard mode is changing
    bot.keyboard.apply_once(.1)
    assert ("down", "q") not in backend.events
    bot.step(.2)
    bot.health.update.assert_called_once()
    bot.keyboard.apply_once(.2)
    assert ("down", "q") in backend.events


def test_duplicate_capture_does_not_advance_route_or_renew_keyboard_permission(routed, monkeypatch):
    from maplebot.capture.window import FrameSample
    bot, backend, _, frame = routed
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (80, 80, 0))
    clock = [100.]
    monkeypatch.setattr(bot.keyboard, "clock", lambda: clock[0])
    sample = FrameSample(frame, 1, clock[0])
    monkeypatch.setattr(bot.source, "get_sample", lambda after_sequence=None: sample if after_sequence != 1 else None)
    bot.step(0.)
    bot.keyboard.apply_once(0.)
    assert backend.held == {"left"}
    deadline = bot.keyboard._command_deadline
    clock[0] += .1
    assert bot.step(.1) is None and bot.capture_state == "duplicate"
    assert bot.keyboard._command_deadline == deadline
    assert bot.brain.circuit.step.call_count == 1
    clock[0] = deadline + .001
    bot.keyboard.apply_once(.2)
    assert not backend.held and bot.keyboard.cmd == Command.NONE
