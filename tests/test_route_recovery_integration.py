"""Recovery boundaries exercised without a real window or keyboard backend."""
import time
from types import SimpleNamespace

import pytest

from maplebot.brain.fsm import Brain, Perception
from maplebot.map_profiles import MapMatch
from maplebot.recovery_runtime import request_return
from maplebot.types import Box, Command, Detection
from tests.test_bot import _cfg, _live_bot, _vertical_frame
from tests.test_recovery import recovery_config


def _config(**circuit):
    return _cfg(
        route={"circuit": recovery_config(**{"return_on_block": False, **circuit})},
        keys={"attack": "z", "normal_attack": "v", "aoe": "a", "pickup": "x",
              "hp_potion": "q", "mp_potion": "w", "jump": "c"},
        buffs={"keys": ["b"], "cooldowns_s": [60]},
        health={"warmup_reads": 0},
    )


def _bot(monkeypatch, hp=80, **circuit):
    bot, backend = _live_bot(_vertical_frame((65, 40)), _config(**circuit))
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (hp, 80, 1))
    monkeypatch.setattr(bot, "_detect", lambda frame: (
        [Detection("nearby mob", Box(680, 330, 40, 30), .01)], None))
    return bot, backend


def _seed_old_actions(bot):
    bot.keyboard.set_enabled(True)
    bot.keyboard.set_command(Command("right", action="attack"))
    bot.keyboard.request("hp_potion")
    bot.keyboard.request("mp_potion")


def _start_return(bot):
    assert bot.brain.circuit.request_return_to_start()
    origin = bot.brain.circuit.idx
    for now in (0, .3, .6, .7, .8):
        bot.step(now)
        bot.keyboard.apply_once(now)
    assert bot.brain.circuit.recovery is not None
    assert bot.brain.circuit.returning_to_start
    return origin


def _assert_no_extras(backend):
    assert not any(event == "down" and key in {"z", "v", "a", "x", "q", "w", "b"}
                   for event, key in backend.events)


@pytest.mark.parametrize("state", ["returning", "blocked", "failed"])
def test_brain_return_and_blocked_states_preempt_combat_and_loot(monkeypatch, state):
    brain = Brain(_config(), None, 0)
    if state == "returning":
        assert brain.circuit.request_return_to_start()
    elif state == "failed":
        brain.circuit.abort_return_to_start("no safe path")
    else:
        brain.circuit.state, brain.circuit.reason = "blocked", "waypoint timeout"
    monkeypatch.setattr(brain, "_choose_target", lambda p: pytest.fail("combat ran during return/block"))
    monkeypatch.setattr(brain.loot, "step", lambda *args: pytest.fail("loot ran during return/block"))
    p = Perception(1282, 693, (641, 346), None, (65, 40), 150, False,
                   [Detection("mob", Box(680, 330, 40, 30), .01)], now=0)
    assert brain.decide(p) == Command.NONE
    assert brain.target is None


def test_return_with_visible_player_suppresses_old_and_automatic_actions(monkeypatch):
    bot, backend = _bot(monkeypatch)
    _seed_old_actions(bot)
    monkeypatch.setattr(bot.brain, "decide", lambda p: pytest.fail("ordinary brain ran during return"))
    origin = _start_return(bot)
    assert bot.player is not None  # A visible sprite must not restore combat permission.
    assert bot.keyboard.enabled and backend.held == {"right"}
    assert bot.brain.circuit.idx == origin and bot.brain.circuit.return_count == 1
    assert bot.keyboard._buffs.due(1) == "b"
    _assert_no_extras(backend)


def test_action_permission_is_retired_before_health_reader_runs(monkeypatch):
    bot, backend = _bot(monkeypatch)
    _seed_old_actions(bot)
    assert bot.brain.circuit.request_return_to_start()
    reads = []

    def read_health(*args):
        # Simulate the independent keyboard tick while image work is underway.
        bot.keyboard.apply_once(0)
        reads.append(True)
        return 80, 80, 1

    monkeypatch.setattr("maplebot.bot.read_bars", read_health)
    bot.step(0)
    assert reads
    _assert_no_extras(backend)


@pytest.mark.parametrize("terminal", [False, True])
def test_blocked_route_disables_all_input_even_with_current_visual_lock(monkeypatch, terminal):
    bot, backend = _bot(monkeypatch)
    _seed_old_actions(bot)
    if terminal:
        bot.brain.circuit.abort_return_to_start("no safe path")
    else:
        bot.brain.circuit.state, bot.brain.circuit.reason = "blocked", "waypoint timeout"
    snap = bot.step(0)
    bot.keyboard.apply_once(0)
    assert snap.command == Command.NONE and snap.player is not None
    assert not bot.keyboard.enabled and not backend.held and not backend.events


@pytest.mark.parametrize("hp", [None, 37.5, float("nan"), float("inf"), -1, 101])
def test_return_health_guard_pauses_before_any_movement(monkeypatch, hp):
    bot, backend = _bot(monkeypatch, hp=hp)
    _seed_old_actions(bot)
    assert bot.brain.circuit.request_return_to_start()
    origin = bot.brain.circuit.idx
    snap = bot.step(0)
    bot.keyboard.apply_once(0)
    nav = bot.brain.circuit
    assert bot.paused and snap.command == Command.NONE and not bot.keyboard.enabled
    assert nav.recovery_terminal and nav.recovery_state == "failed"
    assert nav.idx == origin and nav.return_count == 1 and nav.recovery is None
    assert not backend.events


def test_health_dropping_during_return_releases_held_movement(monkeypatch):
    bot, backend = _bot(monkeypatch)
    origin = _start_return(bot)
    assert backend.held
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (39, 80, 1))
    bot.step(.9)
    bot.keyboard.apply_once(.9)
    assert not backend.held and not bot.keyboard.enabled and bot.paused
    assert bot.brain.circuit.recovery_terminal and bot.brain.circuit.idx == origin
    _assert_no_extras(backend)


@pytest.mark.parametrize("hp, allowed", [(40, True), (39, False)])
def test_automatic_return_checks_health_before_arming_and_suppresses_extras(monkeypatch, hp, allowed):
    bot, backend = _bot(monkeypatch, hp=hp, return_on_block=True)
    _seed_old_actions(bot)
    nav = bot.brain.circuit
    nav.state, nav.reason = "blocked", "waypoint timeout"
    origin = nav.idx
    snap = bot.step(0)
    bot.keyboard.apply_once(0)
    assert snap.command == Command.NONE and nav.idx == origin
    assert nav.returning_to_start is allowed
    assert nav.return_count == int(allowed)
    assert nav.recovery_terminal is not allowed
    assert bot.paused is not allowed
    assert not backend.events


def test_duplicate_capture_cannot_supply_stability_evidence_or_renew_return_permission(monkeypatch):
    bot, backend = _bot(monkeypatch)
    wall = [100.0]
    bot.keyboard.clock = lambda: wall[0]
    sample = SimpleNamespace(frame=bot.source.frame, sequence=1, captured_at=wall[0])

    def get_sample(after_sequence=None):
        return sample if wall[0] - sample.captured_at <= .75 and after_sequence != sample.sequence else None

    bot.source.get_sample = get_sample
    assert bot.brain.circuit.request_return_to_start()
    nav, origin = bot.brain.circuit, bot.brain.circuit.idx
    bot.step(0)
    lease = bot.keyboard._command_deadline
    for now in (.3, .6):
        wall[0] = 100 + now
        assert bot.step(now) is None and bot.capture_state == "duplicate"
        assert bot.keyboard._command_deadline == lease
        assert nav.recovery is None and nav.return_count == 1 and nav.idx == origin
    wall[0] = 100.8
    assert bot.step(.8) is None
    bot.keyboard.apply_once(.8)
    assert not bot.keyboard.enabled and not backend.events
    assert nav.recovery is None and nav.return_count == 1 and nav.idx == origin


@pytest.mark.parametrize("loss", ["capture", "dot", "focus", "map", "pause"])
def test_return_interruption_releases_input_without_resetting_index_or_budget(monkeypatch, loss):
    bot, backend = _bot(monkeypatch)
    origin = _start_return(bot)
    child = bot.brain.circuit.recovery
    assert backend.held
    if loss == "capture":
        bot.source.frame = None
    elif loss == "dot":
        bot.source.frame[61:181, 11:161] = 40
    elif loss == "focus":
        monkeypatch.setattr(bot.keyboard, "is_active", lambda: False)
    elif loss == "map":
        bot.map_recognizer = SimpleNamespace(
            observe=lambda *args: MapMatch("unknown"), reset=lambda: None)
    else:
        bot.toggle_pause()
    snap = bot.step(.9)
    bot.keyboard.apply_once(.9)
    assert snap is None or snap.command == Command.NONE
    assert not bot.keyboard.enabled and not backend.held
    nav = bot.brain.circuit
    assert nav.idx == origin and nav.return_count == 1 and nav.recovery is child
    assert nav.recovery_state != "completed"
    _assert_no_extras(backend)


def _regain_bot(monkeypatch, hp=80):
    """Unsupported at (65,50): 'middle' (y=60) is the first drawn floor below, point 1 is left."""
    bot, backend = _bot(monkeypatch, hp=hp)
    bot.source.frame = _vertical_frame((65, 50))
    wall = [100.0]
    bot.keyboard.clock = lambda: wall[0]
    sequence = [0]
    sample = [None]

    def get_sample(after_sequence=None):
        current = sample[0]
        if current is None or wall[0] - current.captured_at > .75 or after_sequence == current.sequence:
            return None
        return current

    def step(now, fresh=True):
        wall[0] = 100 + now
        if fresh:
            sequence[0] += 1
            sample[0] = SimpleNamespace(frame=bot.source.frame, sequence=sequence[0], captured_at=wall[0])
        snap = bot.step(now)
        bot.keyboard.apply_once(now)
        return snap

    bot.source.get_sample = get_sample
    return bot, backend, step, wall


def _regain_to_pulse(bot, step):
    assert bot.brain.circuit.request_return_to_start()
    nav = bot.brain.circuit
    for i in range(6):
        snap = step(i * .125)
        assert snap.command == Command.NONE and nav.recovery_state == "regaining_support"
        assert not bot.keyboard._held_x
    snap = step(.75)
    assert snap.command == Command("left") and snap.command.valid_for_s == pytest.approx(.25)
    assert nav.state == "return_regain_walking" and nav.recovery is None
    return nav


def test_regain_pulse_is_a_leased_navigation_only_command(monkeypatch):
    bot, backend, step, wall = _regain_bot(monkeypatch)
    _seed_old_actions(bot)
    monkeypatch.setattr(bot.brain, "decide", lambda p: pytest.fail("ordinary brain ran during regain"))
    nav = _regain_to_pulse(bot, step)
    assert bot.keyboard.enabled and bot.keyboard._navigation_only and backend.held == {"left"}
    assert bot.keyboard._command_deadline == pytest.approx(wall[0] + .25)
    assert bot.brain.state == "minimap_route_return_regain_walking"
    assert nav.return_count == 1 and nav.returning_to_start
    assert not any(event == "down" and key in {"up", "c"} for event, key in backend.events)
    _assert_no_extras(backend)


def test_duplicate_capture_during_regain_pulse_sends_nothing_new_and_cannot_extend_it(monkeypatch):
    bot, backend, step, wall = _regain_bot(monkeypatch)
    _regain_to_pulse(bot, step)
    lease, events = bot.keyboard._command_deadline, list(backend.events)
    assert backend.held == {"left"}
    assert step(.8, fresh=False) is None and bot.capture_state == "duplicate"
    assert bot.keyboard._command_deadline == lease and backend.events == events
    # No fresh frame arrives: the keyboard ends the pulse at its own deadline.
    assert step(1.05, fresh=False) is None
    assert bot.keyboard._command_deadline in (None, lease)  # expired, never renewed
    assert not backend.held and backend.events == events + [("up", "left")]
    assert bot.brain.circuit._regain.moves == 1


def test_fresh_frames_during_a_regain_pulse_cannot_extend_it(monkeypatch):
    bot, backend, step, wall = _regain_bot(monkeypatch)
    _regain_to_pulse(bot, step)
    lease = bot.keyboard._command_deadline
    assert lease == pytest.approx(100.75 + .25) and backend.held == {"left"}
    snap = step(.875)  # a new capture inside the pulse: only the remaining time
    assert snap.command == Command("left") and snap.command.valid_for_s == pytest.approx(.125)
    assert bot.keyboard._command_deadline == pytest.approx(lease) and backend.held == {"left"}
    for now in (1.0, 1.125, 1.25):  # the pulse is over: release, however fresh the frames
        assert step(now).command == Command.NONE
        assert not backend.held
    assert backend.events.count(("down", "left")) == 1 and backend.events.count(("up", "left")) == 1


@pytest.mark.parametrize("hp", [39, None])
def test_low_or_unreadable_hp_during_regain_aborts_and_releases_movement(monkeypatch, hp):
    bot, backend, step, _ = _regain_bot(monkeypatch)
    nav = _regain_to_pulse(bot, step)
    origin = nav.idx
    assert backend.held == {"left"}
    monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (hp, 80, 1))
    snap = step(.875)
    assert snap.command == Command.NONE
    assert not backend.held and not bot.keyboard.enabled and bot.paused
    assert nav.recovery_terminal and nav.recovery_state == "failed" and nav.recovery is None
    assert nav.idx == origin and nav.return_count == 1
    assert ("unreadable" if hp is None else "below 40%") in nav.recovery_reason
    step(1.0)
    assert not backend.held and not any(event == "down" and key in {"left", "right", "down", "up", "c"}
                                        for event, key in backend.events[2:])
    _assert_no_extras(backend)


@pytest.mark.parametrize("loss", ["focus", "map", "pause", "dot"])
def test_regain_interruption_disables_input_and_requires_a_new_settle(monkeypatch, loss):
    bot, backend, step, _ = _regain_bot(monkeypatch)
    nav = _regain_to_pulse(bot, step)
    origin = nav.idx
    assert backend.held == {"left"}
    if loss == "focus":
        monkeypatch.setattr(bot.keyboard, "is_active", lambda: False)
    elif loss == "map":
        bot.map_recognizer = SimpleNamespace(
            observe=lambda *args: MapMatch("unknown"), reset=lambda: None)
    elif loss == "dot":
        bot.source.frame = bot.source.frame.copy()
        bot.source.frame[61:181, 11:161] = 40
    else:
        bot.toggle_pause()  # F1
    snap = step(.875)
    assert snap is None or snap.command == Command.NONE
    assert not bot.keyboard.enabled and not backend.held
    assert nav.returning_to_start and not nav.recovery_terminal and nav.recovery is None
    assert nav.idx == origin and nav.return_count == 1
    assert nav._regain.phase == "settling" and nav._regain.moves == 1
    assert [key for event, key in backend.events if event == "down"] == ["left"]
    _assert_no_extras(backend)


def _request_ready_bot(monkeypatch):
    bot, backend = _bot(monkeypatch)
    bot.paused = True
    bot.step(0)
    bot.map_recognizer = object()  # Request validation uses the preceding recognized snapshot.
    bot.capture_time = time.monotonic()
    assert bot.capture_state == "fresh" and bot._minimap_seen
    return bot, backend


def test_manual_request_releases_existing_input_then_waits_for_next_guarded_frame(monkeypatch):
    bot, backend = _request_ready_bot(monkeypatch)
    _seed_old_actions(bot)
    # Simulate already held movement without running the deliberately queued actions.
    bot.keyboard._held_x = "right"
    backend.down("right")
    assert request_return(bot, {"created": 100}, now=100)
    assert not bot.paused and not bot.keyboard.enabled and not backend.held
    nav = bot.brain.circuit
    assert nav.returning_to_start and nav.return_count == 1 and nav.recovery is None
    bot.keyboard.apply_once(0)
    _assert_no_extras(backend)


@pytest.mark.parametrize("bad", ["expired", "future", "missing_time", "map", "no_recognizer",
                                 "stale", "duplicate", "no_time", "dot", "no_position",
                                 "low_hp", "unreadable_hp", "nonfinite_hp", "movement_test"])
def test_invalid_manual_request_stays_paused_without_reset_or_budget_spend(monkeypatch, bad):
    bot, backend = _request_ready_bot(monkeypatch)
    request = {"created": 100}
    if bad == "expired":
        request["created"] = 69
    elif bad == "future":
        request["created"] = 101
    elif bad == "missing_time":
        request = {}
    elif bad == "map":
        bot.map_ready = False
    elif bad == "no_recognizer":
        bot.map_recognizer = None
    elif bad == "stale":
        bot.capture_time -= 1
    elif bad == "duplicate":
        bot.capture_state = "duplicate"
    elif bad == "no_time":
        bot.capture_time = None
    elif bad == "dot":
        bot._minimap_seen = False
    elif bad == "no_position":
        bot.minimap_pos = None
    elif bad == "low_hp":
        bot.last_snapshot.hp = 37.5
    elif bad == "unreadable_hp":
        bot.last_snapshot.hp = None
    elif bad == "nonfinite_hp":
        bot.last_snapshot.hp = float("nan")
    else:
        bot.brain.movement_test = True
    nav = bot.brain.circuit
    origin = nav.idx
    _seed_old_actions(bot)
    assert not request_return(bot, request, now=100)
    bot.keyboard.apply_once(0)
    assert bot.paused and not bot.keyboard.enabled and not backend.events
    assert nav.idx == origin and nav.return_count == 0 and not nav.returning_to_start
    assert bot._return_request_error


def test_duplicate_manual_request_does_not_restart_child_or_replenish_budget(monkeypatch):
    bot, _ = _request_ready_bot(monkeypatch)
    assert request_return(bot, {"created": 100}, now=100)
    nav, origin = bot.brain.circuit, bot.brain.circuit.idx
    assert not request_return(bot, {"created": 100}, now=100)
    assert nav.idx == origin and nav.return_count == 1 and nav.returning_to_start
    assert bot.paused and not bot.keyboard.enabled
