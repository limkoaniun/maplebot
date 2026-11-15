"""Uncertain map frames release keys immediately, with one bounded fresh recheck."""
from types import SimpleNamespace

import numpy as np
import pytest

from maplebot.map_profiles import MapMatch
from maplebot.types import Command
from tests.test_runtime_map_guard import profile, scene, bot_for


def locked_bot():
    first, cfg = profile("first", [5, 5, 50, 40], 10)
    bot, backend = bot_for(first, cfg, [first])
    for now in (0, .1, .2):
        bot._check_map(scene(first), now)
    assert bot.map_ready
    bot.paused = False
    bot.keyboard.set_command(Command("right"))
    bot.keyboard.set_enabled(True)
    bot.keyboard.apply_once(0)
    assert backend.held == {"right"}
    assert bot.brain.circuit.request_return_to_start()
    return bot, backend, first


def test_transient_unknown_keeps_return_but_releases_keys_until_three_new_matches():
    bot, backend, first = locked_bot()
    nav, generation = bot.brain.circuit, bot._control_generation
    assert not bot._check_map(np.zeros_like(scene(first)), 1)
    assert bot.map_reacquiring and not bot.map_ready and not bot.paused
    assert not backend.held and not bot.keyboard.enabled and bot.minimap_pos is None
    assert nav.returning_to_start and not nav.recovery_terminal
    for now in (1.1, 1.2):
        assert not bot._check_map(scene(first), now)
        assert bot.map_reacquiring and not bot.keyboard.enabled
    assert bot._check_map(scene(first), 1.3)
    assert bot.map_ready and not bot.map_reacquiring and not bot.paused
    assert nav.returning_to_start and not nav.recovery_terminal
    assert bot._control_generation == generation and not bot.keyboard.enabled


def test_candidate_and_missing_frame_cycles_cannot_restart_reacquisition_deadline():
    bot, backend, first = locked_bot()
    assert not bot._check_map(np.zeros_like(scene(first)), 1)
    assert not bot._check_map(scene(first), 1.4)
    bot.source = SimpleNamespace(get_frame=lambda: None)
    assert bot.step(1.7) is None
    assert bot._map_lost_since == 1 and bot.map_reacquiring
    assert not bot._check_map(scene(first), 2.2)
    assert not bot._check_map(scene(first), 2.5)
    assert bot.step(3.01) is None
    assert bot.full_pause and bot.paused and not bot.map_reacquiring
    assert bot.brain.circuit.recovery_terminal and "2 seconds" in bot.map_reacquire_reason
    assert not backend.held and not bot.keyboard.enabled


def test_confident_match_after_deadline_still_requires_explicit_resume():
    bot, backend, first = locked_bot()
    bot._check_map(np.zeros_like(scene(first)), 1)
    bot._check_map(scene(first), 1.8)
    bot._check_map(scene(first), 2.8)
    assert bot._check_map(scene(first), 3.01)
    assert bot.map_ready and bot.full_pause and bot.paused and not bot.map_reacquiring
    assert bot.brain.circuit.recovery_terminal and not bot.keyboard.enabled and not backend.held


def test_reused_but_fresh_capture_cannot_extend_the_map_reacquisition_deadline():
    bot, backend, first = locked_bot()
    bot._check_map(np.zeros_like(scene(first)), 1)
    bot.capture_sequence, bot.capture_time = 5, 2.8
    sample = SimpleNamespace(frame=scene(first), sequence=5, captured_at=2.8)
    bot.source = SimpleNamespace(get_sample=lambda after_sequence=None: sample if after_sequence is None else None)
    assert bot.step(3.01) is None and bot.capture_state == "duplicate"
    assert bot.full_pause and bot.brain.circuit.recovery_terminal
    assert not bot.map_reacquiring and not backend.held


def test_explicit_pause_wins_even_when_map_recovers_before_deadline():
    bot, backend, first = locked_bot()
    bot._check_map(np.zeros_like(scene(first)), 1)
    bot.pause_all()
    for now in (1.1, 1.2, 1.3):
        bot._check_map(scene(first), now)
    assert bot.map_ready and bot.full_pause and bot.paused
    assert not bot.keyboard.enabled and not backend.held


@pytest.mark.parametrize("bad", ["settings", "wrong_profile", "incomplete", "matched_without_profile"])
def test_confirmed_identity_or_configuration_failure_never_receives_grace(bad):
    bot, backend, first = locked_bot()
    if bad == "settings":
        bot.cfg["minimap"]["fixed_rect"][0] += 1
    else:
        second, _ = profile("second", [5, 5, 50, 40], 11)
        match = (MapMatch("matched", profile=second) if bad == "wrong_profile" else
                 MapMatch("incomplete") if bad == "incomplete" else MapMatch("matched"))
        bot.map_recognizer = SimpleNamespace(observe=lambda *args: match)
    assert not bot._check_map(scene(first), 1)
    assert bot.full_pause and bot.paused and not bot.map_reacquiring
    assert bot.brain.circuit.recovery_terminal and not backend.held
