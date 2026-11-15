"""Captured-frame health protection through the real Bot and recorded keyboard."""
from types import SimpleNamespace

import pytest

from maplebot.capture.window import FrameSample
from maplebot.map_profiles import MapMatch
from maplebot.types import Box, Command, Detection
from tests.test_bot import _cfg, _live_bot, _vertical_frame
from tests.test_recovery import recovery_config


class _Game:
    def __init__(self, monkeypatch, *, terminal=False, protect=True):
        self.wall = 100.0
        self.hp, self.mp = 75.0, 80.0
        self.focused = True
        self.map_state = "matched"
        self.sample = None
        self.sequence = 0
        cfg = _cfg(
            route={"circuit": recovery_config(return_on_block=False)},
            map_profile={"active": "test-map", "reference_sha256": "test-reference"},
            keys={"attack": "z", "normal_attack": "v", "aoe": "a", "pickup": "x",
                  "hp_potion": "q", "mp_potion": "w", "jump": "c"},
            buffs={"keys": ["b"], "cooldowns_s": [60]},
            health={"warmup_reads": 0, "hp_percent": 60, "minimum_hp_percent": 50,
                    "rearm_delta": 5, "protection_retry_s": 1,
                    "protection_max_taps": 3, "protect_on_route_failure": protect},
        )
        self.bot, self.backend = _live_bot(_vertical_frame((65, 40)), cfg)
        self.bot.keyboard.clock = lambda: self.wall
        self.bot.keyboard.is_active = lambda: self.focused
        self.bot.source.get_sample = self.get_sample
        profile = SimpleNamespace(key="test-map", image_sha256="test-reference")
        self.bot.map_recognizer = SimpleNamespace(
            observe=lambda *args: MapMatch(self.map_state, profile=profile if self.map_state == "matched" else None),
            reset=lambda: None,
        )
        monkeypatch.setattr("maplebot.map_profiles.profile_config_matches", lambda *args, **kwargs: True)
        monkeypatch.setattr("maplebot.bot.read_bars", lambda *args: (self.hp, self.mp, 1))
        monkeypatch.setattr(self.bot, "_detect", lambda frame: (
            [Detection("nearby mob", Box(680, 330, 40, 30), .01)], None))
        if terminal:
            self.bot.brain.circuit.abort_return_to_start("Return budget exhausted for this lap")

    def get_sample(self, after_sequence=None):
        if (self.sample is None or self.wall - self.sample.captured_at > .75
                or after_sequence == self.sample.sequence):
            return None
        return self.sample

    def step(self, hp, dt=.1, *, fresh=True, dot=True, apply=True):
        self.wall += dt
        self.hp = hp
        if fresh:
            self.sequence += 1
            frame = _vertical_frame((65, 40))
            if not dot:
                frame[61:181, 11:161] = 40
            self.sample = FrameSample(frame, self.sequence, self.wall)
        snap = self.bot.step(self.wall)
        if apply:
            self.bot.keyboard.apply_once(self.wall)
        return snap

    def only_hp_pressed(self):
        assert all(key == "q" for action, key in self.backend.events if action == "down")
        assert not self.backend.held


def test_failed_route_keeps_monitoring_and_heals_without_ever_resuming_route(monkeypatch):
    game = _Game(monkeypatch, terminal=True)
    bot, nav = game.bot, game.bot.brain.circuit
    original_index = nav.idx
    snap = game.step(75)
    assert bot.paused and not bot.keyboard.enabled and not game.backend.events
    assert bot.health_protection.state == "monitoring"
    assert bot.health_protection_active and snap.command == Command.NONE
    game.mp = 0  # MP, buffs, and ordinary combat must not borrow the healing permission.
    snap = game.step(55)
    assert bot.paused and bot.keyboard.enabled and bot.keyboard.hp_only
    assert bot.keyboard.hp_taps_sent == 1 and snap.command == Command.NONE
    game.step(55)
    game.step(65)
    for _ in range(4):
        game.step(75)
    assert bot.paused and not bot.keyboard.enabled
    assert bot.health_protection.state == "monitoring" and not bot.health_protection.latched
    assert nav.idx == original_index and nav.recovery_terminal
    assert bot.keyboard.hp_taps_sent == 1
    game.only_hp_pressed()


def test_normal_low_hp_holds_until_target_then_resumes_farming(monkeypatch):
    game = _Game(monkeypatch)
    bot = game.bot
    snap = game.step(59)
    assert bot.paused and snap.command == Command.NONE and bot.keyboard.hp_only
    assert bot.keyboard.hp_taps_sent == 1
    game.step(62)
    assert bot.paused and bot.keyboard.hp_only
    game.only_hp_pressed()
    snap = game.step(65)
    assert not bot.paused and bot.keyboard.enabled and not bot.keyboard.hp_only
    assert snap.command != Command.NONE
    assert not snap.state.startswith("health_protection_")
    assert bot.health_protection.state == "monitoring"


@pytest.mark.parametrize("hp, state", [(0, "dead"), (None, "needs_attention"),
                                      (float("nan"), "needs_attention"),
                                      (101, "needs_attention")])
def test_dead_or_unreadable_hp_latches_without_any_input(monkeypatch, hp, state):
    game = _Game(monkeypatch)
    game.step(hp)
    assert game.bot.health_protection.latched and game.bot.health_protection.state == state
    assert game.bot.paused and not game.bot.keyboard.enabled
    assert not game.backend.events and game.bot.keyboard.hp_taps_sent == 0
    game.step(75)
    assert game.bot.health_protection.latched and not game.backend.events


def test_three_actual_ineffective_hp_taps_latch_attention_and_stop_retrying(monkeypatch):
    game = _Game(monkeypatch, terminal=True)
    for _ in range(20):
        game.step(55, dt=.25)
    guard = game.bot.health_protection
    assert game.bot.keyboard.hp_taps_sent == 3 and guard.unconfirmed_taps == 3
    assert guard.latched and guard.state == "needs_attention"
    assert "improve" in guard.reason
    assert not game.bot.keyboard.enabled and game.bot.paused
    assert len([event for event in game.backend.events if event == ("down", "q")]) == 3
    game.only_hp_pressed()


@pytest.mark.parametrize("loss", ["focus", "map", "dot", "stale"])
def test_protection_context_loss_discards_queued_hp_and_releases_all_permission(monkeypatch, loss):
    game = _Game(monkeypatch, terminal=True)
    game.step(55, apply=False)
    assert game.bot.keyboard.enabled and game.bot.keyboard.hp_only
    if loss == "focus":
        game.focused = False
    elif loss == "map":
        game.map_state = "unknown"
    snap = game.step(55, dt=.8 if loss == "stale" else .1,
                     dot=loss != "dot", fresh=loss != "stale")
    assert snap is None or snap.command == Command.NONE
    assert not game.bot.keyboard.enabled and not game.backend.held
    assert not game.backend.events and game.bot.keyboard.hp_taps_sent == 0
    assert not game.bot.health_protection_active


def test_duplicate_frame_never_renews_hp_lease_or_counts_as_response(monkeypatch):
    game = _Game(monkeypatch, terminal=True)
    game.step(55)
    lease = game.bot.keyboard._command_deadline
    for _ in range(3):
        assert game.step(55, dt=.2, fresh=False) is None
        assert game.bot.keyboard._command_deadline == lease
    assert game.bot.keyboard.hp_taps_sent == 1
    assert game.bot.health_protection.latest_observed_taps == 0
    game.step(55, dt=.2, fresh=False)
    assert not game.bot.keyboard.enabled and game.bot.keyboard._command_deadline is None
    game.only_hp_pressed()


def test_manual_full_pause_cancels_hp_and_never_rearms_on_later_frames(monkeypatch):
    game = _Game(monkeypatch, terminal=True)
    game.step(55, apply=False)
    game.bot.pause_all()
    assert game.bot.full_pause and not game.bot.keyboard.enabled
    for hp in (55, 45, 65, 75, 55):
        game.step(hp)
        assert game.bot.paused and game.bot.full_pause and not game.bot.keyboard.enabled
    assert not game.backend.events and game.bot.keyboard.hp_taps_sent == 0


def test_healthy_reading_does_not_clear_latch_without_explicit_resume(monkeypatch):
    game = _Game(monkeypatch)
    game.step(0)
    game.step(75)
    assert game.bot.health_protection.latched and not game.backend.events
    game.bot.resume()
    assert not game.bot.keyboard.enabled  # Resume itself grants no input permission.
    game.step(75)
    assert not game.bot.health_protection.latched and not game.bot.paused
    assert game.bot.keyboard.enabled and not game.bot.keyboard.hp_only


def test_opt_out_disables_post_route_healing_but_preserves_normal_low_hp_guard(monkeypatch):
    stopped = _Game(monkeypatch, terminal=True, protect=False)
    for hp in (75, 55, 45):
        stopped.step(hp)
    assert not stopped.backend.events and not stopped.bot.keyboard.enabled
    assert stopped.bot.keyboard.hp_taps_sent == 0
    ordinary = _Game(monkeypatch, protect=False)
    ordinary.step(55)
    assert ordinary.bot.keyboard.hp_taps_sent == 1 and ordinary.bot.paused
    ordinary.only_hp_pressed()


def test_ordinary_mp_potion_still_works_when_hp_guard_is_only_monitoring(monkeypatch):
    game = _Game(monkeypatch)
    game.mp = 10
    game.step(75)
    assert game.bot.health_protection.state == "monitoring"
    assert not game.bot.paused and not game.bot.keyboard.hp_only
    assert ("down", "w") in game.backend.events
    assert game.bot.keyboard.hp_taps_sent == 0 and ("down", "q") not in game.backend.events


def test_missing_hp_binding_latches_attention_instead_of_claiming_healing(monkeypatch):
    game = _Game(monkeypatch, terminal=True)
    game.bot.cfg["keys"]["hp_potion"] = ""
    game.step(55)
    assert game.bot.health_protection.latched
    assert game.bot.health_protection.state == "needs_attention"
    assert "configured" in game.bot.health_protection.reason
    assert not game.backend.events and game.bot.keyboard.hp_taps_sent == 0
    assert not game.bot.keyboard.enabled


def test_known_route_failure_retires_old_attack_before_health_image_work(monkeypatch):
    game = _Game(monkeypatch, terminal=True)
    keyboard = game.bot.keyboard
    keyboard.set_enabled(True)
    keyboard.set_command(Command("right", action="attack"), source_time=game.wall)
    keyboard.request("mp_potion")

    def read_health(*args):
        keyboard.apply_once(game.wall)
        return game.hp, game.mp, 1

    monkeypatch.setattr("maplebot.bot.read_bars", read_health)
    game.step(75)
    assert not game.backend.events and not game.backend.held
    assert game.bot.paused and not keyboard.enabled
