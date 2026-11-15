"""An explicit farming hold keeps only fresh, guarded HP protection available."""
import threading

import pytest

from maplebot.recovery_runtime import request_hold
from maplebot.types import Command
from tests.test_health_protection_integration import _Game


@pytest.mark.parametrize("terminal", [False, True])
def test_manual_hold_keeps_route_stopped_and_allows_only_hp_even_with_failure_opt_out(monkeypatch, terminal):
    game = _Game(monkeypatch, terminal=terminal, protect=False)
    bot, nav = game.bot, game.bot.brain.circuit
    initial_index = nav.idx
    bot.hold_farming("Finished the observed circuit")
    assert bot.paused and not bot.full_pause and not bot.keyboard.enabled
    assert bot.health_hold_reason == "Finished the observed circuit"
    game.mp = 0
    for _ in range(3):
        snap = game.step(75)
        assert snap.command == Command.NONE and not bot.keyboard.enabled
    assert not game.backend.events
    game.step(55)
    assert bot.keyboard.hp_taps_sent == 1 and bot.keyboard.hp_only
    game.step(55)
    for hp in (65, 75, 75):
        snap = game.step(hp)
        assert bot.paused and bot._manual_health_hold and not bot.full_pause
        assert snap.command == Command.NONE and not bot.keyboard.enabled
        assert nav.idx == initial_index
    assert bot.health_protection.state == "monitoring"
    game.only_hp_pressed()


def test_pause_all_clears_manual_hold_and_cancels_queued_hp_permanently(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.hold_farming()
    game.step(55, apply=False)
    assert bot.keyboard.enabled
    bot.pause_all()
    assert bot.full_pause and bot.paused and not bot._manual_health_hold
    assert bot.health_hold_reason is None and not bot.keyboard.enabled
    for hp in (55, 45, 75, 55):
        game.step(hp)
        assert bot.full_pause and bot.paused and not bot.keyboard.enabled
    assert bot.keyboard.hp_taps_sent == 0 and not game.backend.events


def test_explicit_resume_after_healthy_manual_hold_restores_farming_on_fresh_frame(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.hold_farming()
    game.step(75)
    bot.resume()
    assert not bot._manual_health_hold and not bot._route_health_hold
    assert bot.health_hold_reason is None and not bot.keyboard.enabled
    snap = game.step(75)
    assert not bot.paused and not bot.full_pause and bot.keyboard.enabled
    assert not bot.keyboard.hp_only and snap.command != Command.NONE


def _ready_request_game(monkeypatch):
    game = _Game(monkeypatch)
    game.step(75, apply=False)
    monkeypatch.setattr("maplebot.recovery_runtime.time.monotonic", lambda: game.wall)
    game.bot.keyboard.request("mp_potion")
    return game


def test_fresh_hold_request_retires_existing_action_permission_before_arming_healing(monkeypatch):
    game = _ready_request_game(monkeypatch)
    bot = game.bot
    bot.keyboard.set_command(Command("right", action="attack"), source_time=game.wall)
    assert request_hold(bot, {"created": 1000}, now=1000)
    bot.keyboard.apply_once(game.wall)
    assert bot.paused and not bot.full_pause and bot._manual_health_hold
    assert not bot.keyboard.enabled and not game.backend.events
    game.step(55)
    assert bot.keyboard.hp_taps_sent == 1
    game.only_hp_pressed()


@pytest.mark.parametrize("bad", ["stale", "duplicate", "unknown_map", "missing_dot", "missing_key",
                                 "disabled_health", "expired_request", "movement_test"])
def test_rejected_hold_request_revokes_existing_input_without_arming_healing(monkeypatch, bad):
    game = _ready_request_game(monkeypatch)
    bot = game.bot
    request = {"created": 1000}
    if bad == "stale":
        bot.capture_time -= 1
    elif bad == "duplicate":
        bot.capture_state = "duplicate"
    elif bad == "unknown_map":
        bot.map_ready = False
    elif bad == "missing_dot":
        bot._minimap_seen = False
    elif bad == "missing_key":
        bot.cfg["keys"]["hp_potion"] = ""
    elif bad == "disabled_health":
        bot.cfg["health"]["enable"] = False
    elif bad == "expired_request":
        request["created"] = 969
    else:
        bot.brain.movement_test = True
    assert not request_hold(bot, request, now=1000)
    bot.keyboard.apply_once(game.wall)
    assert bot.full_pause and bot.paused and not bot._manual_health_hold
    assert not bot.keyboard.enabled and not game.backend.events
    assert bot._hold_request_error


def test_full_pause_during_held_health_image_read_cannot_be_overridden(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.hold_farming()

    def pause_during_read(*args):
        bot.pause_all()
        return 55, 80, 1

    monkeypatch.setattr("maplebot.bot.read_bars", pause_during_read)
    game.step(55)
    assert bot.full_pause and bot.paused and not bot._manual_health_hold
    assert not bot.keyboard.enabled and not game.backend.events


def test_concurrent_full_pause_wins_after_held_health_guard_finishes_its_locked_update(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.hold_farming()
    enabling, continue_guard = threading.Event(), threading.Event()
    pause_started, pause_finished = threading.Event(), threading.Event()
    failures = []
    original_set_enabled = bot.keyboard.set_enabled

    def delayed_enable(enabled):
        if enabled:
            enabling.set()
            assert continue_guard.wait(2)
        original_set_enabled(enabled)

    def guarded_frame():
        try:
            game.step(55, apply=False)
        except BaseException as error:
            failures.append(error)

    def pause():
        pause_started.set()
        bot.pause_all()
        pause_finished.set()

    monkeypatch.setattr(bot.keyboard, "set_enabled", delayed_enable)
    worker = threading.Thread(target=guarded_frame)
    stopper = threading.Thread(target=pause)
    worker.start()
    try:
        assert enabling.wait(2)
        stopper.start()
        assert pause_started.wait(2)
        assert not pause_finished.wait(.05)
    finally:
        continue_guard.set()
        worker.join(2)
        if stopper.ident is not None:
            stopper.join(2)
    assert not failures and not worker.is_alive() and not stopper.is_alive()
    assert pause_finished.is_set() and bot.full_pause and not bot._manual_health_hold
    bot.keyboard.apply_once(game.wall)
    assert not bot.keyboard.enabled and not game.backend.events
    game.step(55)
    assert not bot.keyboard.enabled and not game.backend.events


def test_inconsistent_full_pause_revokes_old_permission_before_reading_capture(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.keyboard.set_enabled(True)
    bot.keyboard.set_command(Command("right", action="attack"), source_time=game.wall)
    bot.keyboard.request("mp_potion")
    bot.full_pause, bot.paused = True, False  # Reproduce a stale caller's partial resume.
    read_sample = bot.source.get_sample
    observed = []

    def check_disabled_before_capture(*args, **kwargs):
        observed.append(bot.keyboard.enabled)
        bot.keyboard.apply_once(game.wall)
        return read_sample(*args, **kwargs)

    monkeypatch.setattr(bot.source, "get_sample", check_disabled_before_capture)
    snap = game.step(75)
    assert observed and not any(observed)
    assert bot.full_pause and bot.paused and not bot.keyboard.enabled
    assert snap.command == Command.NONE and not game.backend.events


@pytest.mark.parametrize("stage", ["perception", "decision"])
def test_full_pause_is_authoritative_even_if_partial_resume_happens_mid_frame(monkeypatch, stage):
    game = _Game(monkeypatch)
    bot = game.bot

    def inconsistent_pause(*args):
        bot.full_pause, bot.paused = True, False
        return ([], None) if stage == "perception" else Command("right", action="attack")

    monkeypatch.setattr(bot if stage == "perception" else bot.brain,
                        "_detect" if stage == "perception" else "decide", inconsistent_pause)
    snap = game.step(75)
    assert snap.command == Command.NONE and bot.full_pause and bot.paused
    assert not bot.keyboard.enabled and not game.backend.events


def test_movement_test_permission_clears_full_pause_only_after_current_validation(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.brain.movement_test = bot.brain.circuit.movement_test = True
    bot.pause_all()
    generation = bot._control_generation
    assert not bot.apply_movement_test_permission(False, expected_generation=generation)
    assert bot.full_pause and bot.paused and not bot.keyboard.enabled
    assert bot.apply_movement_test_permission(True, expected_generation=generation)
    assert not bot.full_pause and not bot.paused and not bot.keyboard.enabled
    snap = game.step(75, apply=False)
    assert snap.command != Command.NONE and bot.keyboard.enabled


@pytest.mark.parametrize("control", ["pause_all", "hold_farming"])
def test_movement_test_cannot_apply_permission_validated_before_later_user_control(monkeypatch, control):
    game = _Game(monkeypatch)
    bot = game.bot
    bot.brain.movement_test = bot.brain.circuit.movement_test = True
    generation = bot._control_generation
    getattr(bot, control)()
    assert not bot.apply_movement_test_permission(True, expected_generation=generation)
    assert bot.paused and not bot.keyboard.enabled
    assert bot.full_pause if control == "pause_all" else bot._manual_health_hold


def test_normal_bot_cannot_bypass_full_pause_through_test_permission_api(monkeypatch):
    game = _Game(monkeypatch)
    game.bot.pause_all()
    with pytest.raises(ValueError, match="movement test"):
        game.bot.apply_movement_test_permission(True, expected_generation=game.bot._control_generation)
    assert game.bot.full_pause and not game.bot.keyboard.enabled
