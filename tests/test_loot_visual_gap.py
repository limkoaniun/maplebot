from unittest.mock import Mock

import numpy as np
import pytest

from maplebot.brain.fsm import Brain
from maplebot.nav.loot import LootCollector
from maplebot.types import Command, PlayerFix
from tests.test_minimap_navigation import routed


def committed_loot(bot):
    loot = bot.brain.loot
    loot.enabled = True
    loot.observe((30, 50), (200, 160), (240, 160), -.5)
    loot.step((30, 50), -.4)
    assert loot.step((30, 50), -.1) == Command.NONE
    assert loot.state == "approaching"
    return loot


def test_missing_sprite_stops_every_input_and_route_but_preserves_committed_goal(routed, monkeypatch):
    bot, backend, _, _ = routed
    loot = committed_loot(bot)
    goal = loot.goal
    bot.keyboard.set_enabled(True)
    bot.keyboard.set_command(Command("left", "none", "attack"))
    bot.keyboard.request("hp_potion")
    def detection_tick(frame):
        bot.keyboard.apply_once(0.)
        return [], None
    monkeypatch.setattr(bot, "_detect", detection_tick)
    snap = bot.step(0.)
    assert snap.command == Command.NONE and snap.state == "looting_waiting_visual"
    assert loot.goal == goal and loot.state == "waiting_visual"
    assert not bot.keyboard.enabled and not bot._minimap_route_active
    assert not backend.events and not bot.keyboard._requests
    bot.brain.circuit.step.assert_not_called()
    bot.brain.loot.suspend.assert_not_called()


def test_fresh_sprite_resumes_same_floor_visit_before_route_progress(routed, monkeypatch):
    bot, backend, _, _ = routed
    loot = committed_loot(bot)
    goal, started = loot.goal, loot._started
    bot.step(0.)
    monkeypatch.setattr(bot, "_locate_player", lambda _: PlayerFix(200, 160, 1., "test"))
    monkeypatch.setattr(bot, "_detect", lambda _: ([], None))
    monkeypatch.setattr(bot.health, "update", Mock(return_value=[]))
    monkeypatch.setattr(bot.brain, "decide", Mock(wraps=lambda p: Brain.decide(bot.brain, p)))
    snap = bot.step(.4)
    assert snap.command == Command.NONE and snap.state == "looting_approaching"
    assert loot.goal == goal and loot._started == started
    assert loot._clock.time == pytest.approx(.4)  # stopped interval did not consume the visit
    bot.brain.circuit.step.assert_not_called()
    bot.keyboard.apply_once(.4)
    assert not backend.held
    assert bot.step(.9).command == Command("right")
    bot.keyboard.apply_once(.9)
    assert "right" in backend.held


def test_visual_gap_timeout_discards_goal_and_returns_to_safe_minimap_navigation(routed):
    bot, _, _, _ = routed
    loot = committed_loot(bot)
    bot.step(0.)
    snap = bot.step(.61)
    assert loot.goal is None and loot.state == "idle"
    assert bot._minimap_route_active and snap.command == Command("left")
    bot.brain.circuit.step.assert_called_once()


@pytest.mark.parametrize("loss", ["pause", "stop", "health_hold", "focus", "map", "capture", "floor", "route_point", "return", "transition"])
def test_invalid_context_never_resurrects_a_preserved_visit(routed, monkeypatch, loss):
    bot, _, _, frame = routed
    loot = committed_loot(bot)
    bot.step(0.)
    assert loot.goal is not None
    if loss == "pause":
        bot.pause_all()
        assert loot.goal is None
        return
    if loss == "stop":
        bot.stop_request()
        assert loot.goal is None
        return
    if loss == "health_hold":
        bot.hold_farming()
        assert loot.goal is None
        return
    if loss == "focus":
        monkeypatch.setattr(bot.keyboard, "is_active", lambda: False)
    elif loss == "map":
        bot.source.frame = np.zeros_like(frame)
    elif loss == "capture":
        bot.source.frame = None
    elif loss == "floor":
        original = bot._update_minimap
        def moved_floor(image):
            result = original(image)
            bot.minimap_pos = (30, 40)
            return result
        monkeypatch.setattr(bot, "_update_minimap", moved_floor)
    elif loss == "route_point":
        bot.brain.circuit.idx = 1
    elif loss == "return":
        bot.brain.circuit.returning_to_start = True
        monkeypatch.setattr(bot.brain.circuit, "step", Mock(return_value=Command.NONE))
    else:
        monkeypatch.setattr(type(bot.brain.circuit), "transition_active", property(lambda _: True))
    bot.step(.2)
    assert loot.goal is None and loot._visual_since is None


def test_tracking_a_living_target_does_not_delay_route_on_visual_loss(routed):
    bot, _, _, _ = routed
    loot = bot.brain.loot
    loot.enabled = True
    loot.observe((30, 50), (200, 160), (240, 160), -.1)
    assert loot.state == "tracking"
    assert bot.step(0.).command == Command("left")
    assert loot.goal is None


def test_reacquisition_after_deadline_cancels_even_with_same_context():
    loot = LootCollector({"enabled": True, "disappear_s": 0})
    loot.observe((30, 50), (200, 160), (240, 160), 0)
    loot.step((30, 50), .1)
    assert loot.visual_lock(.2, False, ("map", "floor", 0))
    assert not loot.visual_lock(.81, True, ("map", "floor", 0))
    assert loot.goal is None
