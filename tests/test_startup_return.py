"""Normal Start/Resume waits for guarded physical arrival before farming."""
from copy import deepcopy

import pytest

from maplebot.recovery_runtime import StartupReturn, request_return
from maplebot.types import Command
from tests.test_health_protection_integration import _Game
from tests.test_recovery import start_child, finish_child


def _startup(monkeypatch, **kwargs):
    game = _Game(monkeypatch, **kwargs)
    monkeypatch.setattr("maplebot.recovery_runtime.time.monotonic", lambda: game.wall)
    startup = StartupReturn(game.bot)
    return game, startup


def test_start_heals_first_then_arms_return_without_any_farming_action(monkeypatch):
    game, startup = _startup(monkeypatch)
    bot, original = game.bot, game.bot.brain.circuit
    saved = deepcopy(bot.cfg)
    assert startup.request()
    assert bot.paused and bot._manual_health_hold and not bot.keyboard.enabled
    for hp in (55, 62):
        assert game.step(hp).command == Command.NONE
        startup.update()
        assert startup.state == "checking" and bot.brain.circuit is original
    game.only_hp_pressed()
    game.step(65)
    startup.update()
    nav = bot.brain.circuit
    assert startup.state == "returning" and nav is not original
    assert nav.idx == 0 and nav.returning_to_start and nav.return_count == 1
    assert not bot.paused and not bot._manual_health_hold and not bot.keyboard.enabled
    assert bot.cfg == saved
    game.step(75)
    assert bot.brain.circuit.recovery is None  # source still needs stable fresh observations
    game.only_hp_pressed()


@pytest.mark.parametrize("missing", ["duplicate", "stale", "map", "dot", "focus"])
def test_start_waits_without_inputs_for_fresh_recognized_foreground_position(monkeypatch, missing):
    game, startup = _startup(monkeypatch)
    bot = game.bot
    original = bot.brain.circuit
    startup.request()
    if missing == "focus":
        game.focused = False
    elif missing == "map":
        game.map_state = "unknown"
        bot.map_ready = False  # initial recognition has not completed yet
    game.step(75, dot=missing != "dot")
    if missing == "duplicate":
        game.step(75, fresh=False)
    elif missing == "stale":
        game.wall += 1
    startup.update()
    assert startup.state == "checking" and bot.brain.circuit is original
    assert not bot.keyboard.enabled and not game.backend.events


def test_brief_map_uncertainty_keeps_start_waiting_without_requiring_another_resume(monkeypatch):
    game, startup = _startup(monkeypatch)
    game.bot.pause_all()
    game.step(75)
    startup.request()
    game.map_state = "unknown"
    game.step(75)
    startup.update()
    assert startup.state == "checking" and game.bot.map_reacquiring
    assert not game.bot.keyboard.enabled
    game.map_state = "matched"
    game.step(75)
    startup.update()
    assert startup.state == "returning" and not game.backend.events


def test_sustained_map_uncertainty_cancels_start_and_does_not_auto_rearm(monkeypatch):
    game, startup = _startup(monkeypatch)
    game.bot.pause_all()
    game.step(75)
    startup.request()
    game.map_state = "unknown"
    game.step(75)
    startup.update()
    game.step(75, dt=2.1)
    startup.update()
    assert startup.state == "cancelled" and game.bot.full_pause
    game.map_state = "matched"
    game.step(75)
    startup.update()
    assert startup.state == "cancelled" and not game.backend.events


def _armed(monkeypatch, **kwargs):
    game, startup = _startup(monkeypatch, **kwargs)
    startup.request()
    game.step(75)
    startup.update()
    assert startup.state == "returning"
    return game, startup


def test_only_verified_startup_arrival_replenishes_normal_lap_budget(monkeypatch):
    game, startup = _startup(monkeypatch, terminal=True)
    bot, original = game.bot, game.bot.brain.circuit
    original.return_count = 1
    startup.request()
    game.step(75)
    startup.update()
    nav = bot.brain.circuit
    assert original.recovery_terminal and original.return_count == 1
    assert nav.return_count == 1 and nav.idx == 0
    start_child(nav)
    assert nav.return_count == 1
    finish_child(nav)
    assert nav.recovery_state == "completed" and nav.idx == 0
    startup.update()
    assert startup.state == "completed" and nav.return_count == 0
    assert original.recovery_terminal and original.return_count == 1
    # A later blocked farming return still consumes its ordinary allowance.
    nav.state, nav.reason = "blocked", "waypoint timeout"
    assert nav.request_return_to_start()
    nav.abort_return_to_start("jump timeout")
    startup.update()
    assert nav.return_count == 1 and nav.recovery_terminal


def test_already_at_start_still_requires_stable_physical_confirmation(monkeypatch):
    game, startup = _armed(monkeypatch)
    nav = game.bot.brain.circuit
    assert nav.step((10, 80), 0) == Command.NONE
    startup.update()
    assert startup.state == "returning" and nav.return_count == 1
    for i in range(1, 35):
        assert nav.step((10, 80), i / 10) == Command.NONE
        startup.update()
        if startup.state == "completed":
            break
    assert startup.state == "completed" and i >= 10 and nav.idx == 0 and nav.return_count == 0


def test_start_from_unsupported_position_completes_only_after_verified_arrival(monkeypatch):
    game, startup = _armed(monkeypatch)
    nav = game.bot.brain.circuit
    clock = [0.0]

    def feed(pos, frames=1, dt=.125):
        commands = []
        for _ in range(frames):
            commands.append(nav.step(pos, clock[0]))
            clock[0] += dt
            startup.update()
        return commands

    # (65,50) is above "middle" with nothing drawn under the feet: settle, then one leased pulse.
    commands = feed((65, 50), 7)
    assert commands[:6] == [Command.NONE] * 6
    assert commands[6] == Command("left") and commands[6].valid_for_s == .25
    assert nav.recovery_state == "regaining_support" and startup.state == "returning"
    # The character falls onto "middle"; the stage sends nothing more and proves nothing.
    assert feed((63, 55)) == [Command.NONE]
    assert feed((63, 60), 4) == [Command.NONE] * 4
    assert nav.recovery is None and nav.recovery_state == "waiting_source" and startup.state == "returning"
    feed((63, 60))
    assert nav.recovery is not None and nav.recovery_state == "running" and nav.idx == 0
    for pos, frames in (((40, 60), 6), ((40, 70), 1), ((40, 80), 5), ((30, 80), 1)):
        feed(pos, frames)
        assert startup.state == "returning" and nav.returning_to_start
    feed((10, 80), 6)
    assert nav.recovery.completed and nav.recovery_state == "confirming_start"
    assert startup.state == "returning" and nav.return_count == 1
    for confirmations in range(1, 8):
        feed((10, 80))
        if startup.state != "returning":
            break
    assert confirmations >= 2  # stable point-1 proof, not the route index
    assert nav.recovery_state == "completed" and not nav.returning_to_start
    assert startup.state == "completed" and nav.idx == 0 and nav.return_count == 0
    assert nav._regain.moves == 1


def test_failed_regain_stage_fails_the_start_and_leaves_hp_only_input(monkeypatch):
    game, startup = _armed(monkeypatch)
    bot, nav = game.bot, game.bot.brain.circuit
    for i in range(5):  # (110,50): no drawn platform below, none within 8 px beside
        assert nav.step((110, 50), i * .125) == Command.NONE
        startup.update()
    assert nav.recovery_terminal and nav.recovery is None and nav.return_count == 1
    assert startup.state == "failed"
    assert "Could not reach a drawn platform from (110,50)" in startup.reason
    assert "no drawn platform below or beside" in startup.reason
    assert bot.paused and bot._manual_health_hold
    game.step(55)
    startup.update()
    assert startup.state == "failed" and bot.keyboard.hp_only
    game.step(75)
    startup.update()
    assert startup.state == "failed" and nav.recovery_terminal
    game.only_hp_pressed()


def test_failed_start_stays_in_hp_hold_and_never_auto_rearms(monkeypatch):
    game, startup = _armed(monkeypatch)
    bot, nav = game.bot, game.bot.brain.circuit
    nav.abort_return_to_start("No safe return path")
    startup.update()
    assert startup.state == "failed" and "No safe return path" in startup.reason
    assert bot.paused and bot._manual_health_hold and nav.return_count == 1
    game.step(55)
    startup.update()
    assert startup.state == "failed" and nav.recovery_terminal and nav.return_count == 1
    game.only_hp_pressed()


@pytest.mark.parametrize("stage", ["checking", "returning"])
@pytest.mark.parametrize("action", ["pause", "hold"])
def test_newer_pause_or_manual_hold_cancels_start_and_preserves_old_budget(monkeypatch, stage, action):
    game, startup = _startup(monkeypatch)
    bot, original = game.bot, game.bot.brain.circuit
    original.return_count = 1
    startup.request()
    if stage == "returning":
        game.step(75)
        startup.update()
    bot.pause_all() if action == "pause" else bot.hold_farming("User hold")
    startup.update()
    assert startup.state == "cancelled" and bot.brain.circuit is original
    assert original.return_count == 1 and bot.paused and not bot.keyboard.enabled
    assert bot.full_pause if action == "pause" else bot._manual_health_hold
    game.step(75)
    startup.update()
    assert startup.state == "cancelled" and not game.backend.events


def test_explicit_return_after_cancel_uses_original_recovery_allowance(monkeypatch):
    game, startup = _armed(monkeypatch)
    bot, original = game.bot, startup._prior_nav
    startup.cancel("Replaced by manual return")
    bot.pause_all()
    game.step(75)
    assert bot.brain.circuit is original and original.return_count == 0
    assert request_return(bot, {"created": 1000}, now=1000)
    assert original.returning_to_start and original.return_count == 1


def test_start_wait_is_bounded_and_movement_tests_bypass_this_workflow(monkeypatch):
    game, startup = _startup(monkeypatch)
    startup.request()
    startup.update(now=game.wall + 91)
    assert startup.state == "failed" and game.bot.paused
    assert game.bot._manual_health_hold and not game.backend.events
    game.bot.brain.movement_test = True
    assert not StartupReturn(game.bot).request()


def test_start_cannot_override_a_latched_low_hp_failure(monkeypatch):
    game, startup = _startup(monkeypatch, terminal=True)
    game.step(0)
    assert game.bot.health_protection.latched
    startup.request()
    game.step(55)
    startup.update()
    assert startup.state == "failed" and game.bot.health_protection.latched
    assert not game.bot.keyboard.enabled and not game.backend.events


def test_f1_resumes_through_return_and_cancels_active_start_including_hp_hold(monkeypatch):
    game, startup = _startup(monkeypatch)
    bot = game.bot
    assert not startup.toggle_pause()  # initially active: first F1 is Pause all
    assert bot.full_pause and bot.paused
    assert startup.toggle_pause()  # second F1 queues the physical return
    assert startup.state == "checking" and bot._manual_health_hold
    game.step(55)
    startup.update()
    assert startup.state == "checking" and bot.keyboard.hp_only
    assert not startup.toggle_pause()  # a healing start is still an active start
    assert startup.state == "cancelled" and bot.full_pause and not bot.keyboard.enabled
    game.step(75)
    startup.update()
    assert startup.state == "cancelled" and bot.full_pause


def test_f1_during_ordinary_hp_protection_still_means_full_pause(monkeypatch):
    game, startup = _startup(monkeypatch)
    game.step(55)
    assert game.bot.paused and game.bot._hp_health_hold
    assert not startup.toggle_pause()
    assert game.bot.full_pause and not game.bot.keyboard.enabled and startup.state == "idle"
    game.step(75)
    assert startup.toggle_pause()
    assert startup.state == "checking" and not game.bot.keyboard.enabled
