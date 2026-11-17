import threading

import pytest

from maplebot.config import deep_merge, load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.types import Command


def _kb(**over):
    cfg = deep_merge(load_config(), over)
    be = RecordingBackend()
    # Action timing is simulated by apply_once; lease time is a separate clock.
    kb = KeyboardController(cfg, be, clock=lambda: 0.0, sleep=lambda s: None)
    return kb, be


def _taps(be, key):
    return [e for e in be.events if e[1] == key and e[0] == "down"]


def test_left_hold_then_switch_right_releases_left():
    kb, be = _kb()
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    assert be.events == [("down", "left")]
    kb.set_command(Command("right"))
    kb.apply_once(0.1)
    assert be.events[1:] == [("up", "left"), ("down", "right")]
    assert be.held == {"right"}


def test_repeated_command_does_not_repress():
    kb, be = _kb()
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    kb.apply_once(0.1)
    assert be.events == [("down", "left")]


def test_none_releases_held_keys():
    kb, be = _kb()
    kb.set_command(Command("left", "up"))
    kb.apply_once(0.0)
    assert be.held == {"left", "up"}
    kb.set_command(Command.NONE)
    kb.apply_once(0.1)
    assert be.held == set()


def test_jump_tapped_once_per_command_change():
    kb, be = _kb()
    kb.set_command(Command(action="jump"))
    kb.apply_once(0.0)
    kb.apply_once(0.1)
    assert _taps(be, "space") == [("down", "space")]
    assert be.held == set()
    kb.set_command(Command("left", action="jump"))
    kb.apply_once(0.2)
    assert len(_taps(be, "space")) == 2


def test_attack_repeats_on_cooldown():
    kb, be = _kb(attack={"cooldown_s": 0.9})
    kb.set_command(Command(action="attack"))
    kb.apply_once(0.0)
    kb.apply_once(0.5)
    assert len(_taps(be, "w")) == 1
    kb.apply_once(1.0)
    assert len(_taps(be, "w")) == 2


def test_aoe_uses_aoe_key_and_falls_back():
    kb, be = _kb()
    kb.set_command(Command(action="aoe"))
    kb.apply_once(0.0)
    assert _taps(be, "q") and not _taps(be, "w")
    kb2, be2 = _kb(keys={"aoe": ""})
    kb2.set_command(Command(action="aoe"))
    kb2.apply_once(0.0)
    assert _taps(be2, "w")


def test_potion_request_takes_priority():
    kb, be = _kb()
    kb.set_command(Command(action="attack"))
    kb.request("hp_potion")
    kb.apply_once(0.0)
    assert [e for e in be.events if e[0] == "down"] == [("down", "1")]
    kb.apply_once(0.1)
    assert _taps(be, "w")


@pytest.mark.parametrize("action", ["attack", "aoe", "normal_attack", "teleport"])
def test_navigation_only_clears_queued_actions_and_suppresses_automatic_extras(action):
    kb, be = _kb(keys={"pickup": "x", "normal_attack": "alt", "teleport": "e"},
                 buffs={"keys": ["a"], "cooldowns_s": [60]})
    kb.set_command(Command(action="attack"))
    kb.request("hp_potion")
    kb.set_command(Command("right", "up", action, face_x="left"), navigation_only=True)
    kb.request("mp_potion")
    for now in (0, .1, .4):
        kb.apply_once(now)
    assert not kb._requests
    assert be.events == [("down", "right"), ("down", "up")]
    assert kb._buffs._last["a"] is None


def test_navigation_only_preserves_jump_and_resets_on_fresh_ordinary_command():
    kb, be = _kb(keys={"pickup": "x"},
                 buffs={"keys": ["a"], "cooldowns_s": [60], "freeze_s": 0})
    kb.set_command(Command("right", action="jump"), navigation_only=True)
    kb.apply_once(0)
    kb.apply_once(.1)
    assert _taps(be, "space") == [("down", "space")]
    assert not _taps(be, "x") and not _taps(be, "a")
    kb.set_command(Command(action="attack"))
    kb.request("hp_potion")
    kb.apply_once(.2)
    kb.apply_once(.3)
    kb.apply_once(.4)
    assert _taps(be, "1") and _taps(be, "a") and _taps(be, "w") and _taps(be, "x")


def test_navigation_only_still_expires_independently_and_releases_movement():
    kb, be = _kb(keys={"pickup": "x"})
    clock = [10.0]
    kb.clock = lambda: clock[0]
    kb.set_command(Command("right"), source_time=clock[0], navigation_only=True)
    kb.apply_once(clock[0])
    assert be.held == {"right"}
    clock[0] += .8
    kb.apply_once(clock[0])
    assert not be.held and kb._command_deadline is None
    assert not _taps(be, "x")


def test_disable_releases_all():
    kb, be = _kb()
    kb.set_command(Command("left", "up", "attack"))
    kb.apply_once(0.0)
    kb.set_enabled(False)
    assert be.held == set()
    n = len(be.events)
    kb.apply_once(0.1)
    assert len(be.events) == n


def test_buff_cast_and_freeze():
    kb, be = _kb(buffs={"keys": ["a"], "cooldowns_s": [60], "freeze_s": 1.0})
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    assert _taps(be, "a") and "left" not in be.held
    kb.apply_once(0.5)
    assert "left" not in be.held
    kb.apply_once(1.1)
    assert "left" in be.held
    kb.apply_once(30.0)
    assert len(_taps(be, "a")) == 1


def test_teleport_falls_back_to_jump_when_unbound():
    kb, be = _kb(keys={"teleport": ""})
    kb.set_command(Command(action="teleport"))
    kb.apply_once(0.0)
    assert _taps(be, "space")
    kb2, be2 = _kb(keys={"teleport": "e"})
    kb2.set_command(Command(action="teleport"))
    kb2.apply_once(0.0)
    assert _taps(be2, "e")


def test_pickup_tapped_while_moving():
    kb, be = _kb(keys={"pickup": "z", "pickup_interval_s": 0.3})
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    kb.apply_once(0.1)
    assert len(_taps(be, "z")) == 1
    kb.apply_once(0.4)
    assert len(_taps(be, "z")) == 2
    kb.set_command(Command.NONE)
    kb.apply_once(1.0)
    assert len(_taps(be, "z")) == 3


def test_pickup_while_fighting_stationary_respects_interval_pause_and_focus():
    kb, be = _kb(keys={"attack": "z", "pickup": "x", "pickup_interval_s": 0.5})
    active = {"on": True}
    kb.is_active = lambda: active["on"]
    kb.set_command(Command(action="attack"))
    kb.apply_once(0.0)
    kb.apply_once(0.2)
    assert len(_taps(be, "z")) == 1 and len(_taps(be, "x")) == 1
    assert not be.held
    kb.apply_once(0.5)
    assert len(_taps(be, "x")) == 2
    kb.set_enabled(False)  # both manual pause and the visual-lock gate disable input
    kb.apply_once(1.0)
    assert len(_taps(be, "x")) == 2
    kb.set_enabled(True)
    active["on"] = False
    kb.apply_once(2.0)
    assert len(_taps(be, "x")) == 2
    active["on"] = True
    kb.set_command(Command(action="attack"))
    kb.apply_once(3.0)
    assert len(_taps(be, "x")) == 3


def test_release_all_clears_state():
    kb, be = _kb()
    kb.set_command(Command("right", "down"))
    kb.apply_once(0.0)
    kb.release_all()
    assert be.held == set()
    kb.apply_once(0.1)
    assert be.held == {"right", "down"}


# --- review fix pass -------------------------------------------------------

def test_inactive_window_releases_and_skips():
    cfg = deep_merge(load_config(), {})
    be = RecordingBackend()
    active = {"on": True}
    kb = KeyboardController(cfg, be, clock=lambda: 0.0, sleep=lambda s: None,
                            is_active=lambda: active["on"])
    kb.set_command(Command("left", "none", "attack"))
    kb.apply_once(0.0)
    assert "left" in be.held
    active["on"] = False
    kb.apply_once(0.1)
    assert be.held == set()
    n = len(be.events)
    kb.apply_once(0.2)
    assert len(be.events) == n
    active["on"] = True
    kb.apply_once(5.0)
    assert not be.held  # focus return alone cannot replay an old producer command
    kb.set_command(Command("left", "none", "attack"))
    kb.apply_once(5.1)
    assert "left" in be.held


def test_buff_without_cooldown_is_skipped():
    kb, be = _kb(buffs={"keys": ["a"], "cooldowns_s": [], "freeze_s": 1.0})
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    kb.apply_once(0.5)
    assert not _taps(be, "a")
    assert "left" in be.held


def test_disable_discards_queued_commands_and_potions():
    kb, be = _kb()
    kb.set_command(Command("left"))
    kb.request("hp_potion")
    kb.set_enabled(False)
    kb.request("mp_potion")
    kb.set_enabled(True)
    kb.apply_once(1.0)
    assert not be.events


def test_focus_loss_discards_potions_without_replaying_them():
    kb, be = _kb()
    active = {"on": True}
    kb.is_active = lambda: active["on"]
    kb.request("hp_potion")
    active["on"] = False
    kb.apply_once(0.1)
    kb.request("mp_potion")
    active["on"] = True
    kb.apply_once(1.0)
    assert not be.events


def test_disabling_during_inflight_tick_leaves_no_held_keys():
    kb, be = _kb()
    kb.set_command(Command("left"))
    tick_ready = threading.Event()
    continue_tick = threading.Event()
    disabling = threading.Event()
    disabled = threading.Event()
    original_hold = kb._apply_hold

    def delayed_hold(*args):
        tick_ready.set()
        assert continue_tick.wait(2)
        original_hold(*args)

    def disable():
        disabling.set()
        kb.set_enabled(False)
        disabled.set()

    kb._apply_hold = delayed_hold
    worker = threading.Thread(target=lambda: kb.apply_once(0.0))
    stopper = threading.Thread(target=disable)
    worker.start()
    try:
        assert tick_ready.wait(2)
        stopper.start()
        assert disabling.wait(2)
        disabled.wait(0.05)
    finally:
        continue_tick.set()
        worker.join(2)
        if stopper.ident is not None:
            stopper.join(2)
    assert not worker.is_alive() and not stopper.is_alive()
    assert disabled.is_set() and not kb.enabled and not be.held
    events = list(be.events)
    kb.apply_once(1.0)
    assert be.events == events


def test_normal_attack_emits_alt_and_shares_skill_attack_cooldown():
    kb, be = _kb(keys={"normal_attack": "alt"}, attack={"cooldown_s": 0.9})
    kb.set_command(Command(action="normal_attack"))
    kb.apply_once(0.0)
    kb.apply_once(0.3)
    assert _taps(be, "alt") == [("down", "alt")]
    assert not _taps(be, "w")
    kb.set_command(Command(action="attack"))
    kb.apply_once(0.5)
    assert not _taps(be, "w")
    kb.apply_once(1.0)
    assert _taps(be, "w") == [("down", "w")]
    kb.set_command(Command(action="normal_attack"))
    kb.apply_once(1.1)
    assert len(_taps(be, "alt")) == 1
    kb.apply_once(2.0)
    assert len(_taps(be, "alt")) == 2 and not be.held


@pytest.mark.parametrize("action,key", [("attack", "w"), ("normal_attack", "alt")])
def test_close_target_behind_turns_once_before_attacking(action, key):
    kb, be = _kb(keys={"normal_attack": "alt"}, attack={"cooldown_s": 0.9})
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    kb.set_command(Command(action=action, face_x="right"))
    kb.apply_once(1.0)
    assert be.events[-3:] == [("up", "left"), ("down", "right"), ("up", "right")]
    assert not be.held and not _taps(be, key)
    kb.apply_once(1.05)
    assert not _taps(be, key)
    kb.apply_once(1.14)
    assert _taps(be, key) == [("down", key)]
    kb.apply_once(1.2)
    kb.apply_once(2.1)
    assert len(_taps(be, key)) == 2
    assert _taps(be, "right") == [("down", "right")] and not be.held
    kb.is_active = lambda: False
    kb.apply_once(2.2)
    kb.is_active = lambda: True
    kb.set_command(Command(action=action, face_x="right"))
    kb.apply_once(3.0)
    assert len(_taps(be, "right")) == 2  # facing is unknown after possible manual input
    assert len(_taps(be, key)) == 2  # wait for the new turn before attacking


def _leased_keyboard(**over):
    cfg = deep_merge(load_config(), over)
    wall = [10.0]
    backend = RecordingBackend()
    keyboard = KeyboardController(cfg, backend, clock=lambda: wall[0], sleep=lambda _: None)
    return keyboard, backend, wall


def test_stalled_producer_releases_movement_and_cancels_all_discrete_actions():
    kb, be, wall = _leased_keyboard(keys={"pickup": "x"})
    kb.set_command(Command("right", "up", "attack"), source_time=wall[0])
    kb.apply_once(0.0)
    assert be.held == {"right", "up"}
    kb.request("hp_potion")
    kb.request("mp_potion")
    wall[0] += .76
    kb.apply_once(1.0)
    assert not be.held and kb.cmd == Command.NONE and not kb._requests
    assert not _taps(be, "1") and not _taps(be, "2")
    events = list(be.events)
    kb.apply_once(100.0)
    assert be.events == events  # no attack, pickup, or movement after expiry
    kb.set_command(Command("left"), source_time=wall[0])
    kb.apply_once(100.1)
    assert be.held == {"left"}  # only a new producer command resumes
    assert not _taps(be, "1") and not _taps(be, "2")


def test_old_capture_timestamp_cannot_be_renewed_by_new_commands():
    kb, be, wall = _leased_keyboard(buffs={"keys": ["a"], "cooldowns_s": [1]})
    old_frame_time = wall[0]
    wall[0] += .8
    kb.set_command(Command("left", action="jump"), source_time=old_frame_time)
    kb.request("hp_potion")
    kb.apply_once(0.0)
    assert not be.events and not kb._requests  # also suppress the initially due buff
    kb.set_command(Command("left", action="jump"), source_time=old_frame_time)
    kb.apply_once(1.0)
    assert not be.events


def test_command_deadline_is_bounded_by_source_age_not_delivery_time():
    kb, be, wall = _leased_keyboard()
    captured_at = wall[0]
    wall[0] += .6
    kb.set_command(Command("left"), source_time=captured_at, valid_for_s=10)
    kb.apply_once(0.0)
    assert be.held == {"left"}
    wall[0] += .16
    kb.apply_once(.1)
    assert not be.held


def test_expired_potion_is_not_revived_by_fresh_command():
    kb, be, wall = _leased_keyboard()
    kb.set_command(Command.NONE)
    kb.request("hp_potion")
    wall[0] += .8
    kb.set_command(Command("left"))
    kb.apply_once(0.0)
    assert be.held == {"left"} and not _taps(be, "1")


def test_no_automatic_buff_or_pickup_before_first_fresh_command():
    kb, be, _ = _leased_keyboard(keys={"pickup": "x"},
                                buffs={"keys": ["a"], "cooldowns_s": [1]})
    kb.apply_once(0.0)
    assert not be.events


def test_expiry_during_action_tap_releases_movement_and_skips_pickup():
    kb, be, wall = _leased_keyboard(keys={"pickup": "x"})
    kb.sleep = lambda _: wall.__setitem__(0, wall[0] + .8)
    kb.set_command(Command("right", action="attack"))
    kb.apply_once(0.0)
    assert _taps(be, "w") and not _taps(be, "x")
    assert not be.held and kb.cmd == Command.NONE


def test_hp_only_immediately_releases_movement_and_sends_only_new_hp_request():
    kb, be, wall = _leased_keyboard(keys={"pickup": "x"},
                                    buffs={"keys": ["a"], "cooldowns_s": [60]})
    kb.set_command(Command("right", "up"), navigation_only=True)
    kb.apply_once(0)
    assert be.held == {"right", "up"}
    kb.set_command(Command("right", action="attack", face_x="left"))
    kb.request("hp_potion")
    kb.request("mp_potion")
    kb.set_hp_only(source_time=wall[0])
    assert kb.hp_only and kb.cmd == Command.NONE and not be.held
    before = list(be.events)
    for request in ("mp_potion", "attack", "jump", "pickup"):
        kb.request(request)
    kb.apply_once(0)
    assert be.events == before and kb.hp_taps_sent == 0 and kb.hp_tap_time is None
    kb.request("hp_potion")
    kb.apply_once(.1)
    kb.apply_once(100)
    assert be.events[len(before):] == [("down", "1"), ("up", "1")]
    assert kb.hp_taps_sent == 1 and kb.hp_tap_time == wall[0]
    assert kb._buffs.due(100) == "a"


def test_hp_only_same_mode_renewal_preserves_one_pending_request_without_accumulating():
    kb, be, wall = _leased_keyboard()
    kb.set_hp_only(source_time=wall[0])
    for _ in range(50):
        kb.request("hp_potion")
    assert kb.hp_taps_sent == 0
    wall[0] += .1
    kb.set_hp_only(source_time=wall[0])
    for _ in range(50):
        kb.request("hp_potion")
    kb.apply_once(1000)  # Action time is unrelated to the capture/lease clock.
    kb.apply_once(1001)
    assert be.events == [("down", "1"), ("up", "1")]
    assert kb.hp_taps_sent == 1 and kb.hp_tap_time == wall[0]


def test_hp_only_does_not_enable_input_until_guard_approves():
    kb, be, wall = _leased_keyboard()
    kb.set_enabled(False)
    kb.set_hp_only(source_time=wall[0])
    kb.request("hp_potion")
    assert not kb.enabled
    kb.apply_once(0)
    assert not be.events and kb.hp_taps_sent == 0
    kb.set_hp_only(source_time=wall[0])
    kb.set_enabled(True)
    kb.request("hp_potion")
    kb.apply_once(.1)
    assert kb.hp_taps_sent == 1 and not be.held


@pytest.mark.parametrize("navigation_only", [False, True])
def test_leaving_hp_only_discards_pending_hp_and_restores_requested_mode(navigation_only):
    kb, be, wall = _leased_keyboard(keys={"pickup": "x"})
    kb.set_hp_only(source_time=wall[0])
    kb.request("hp_potion")
    kb.set_command(Command("left", action="attack"), source_time=wall[0],
                   navigation_only=navigation_only)
    kb.apply_once(0)
    assert not kb.hp_only and not _taps(be, "1") and kb.hp_taps_sent == 0
    assert be.held == {"left"}
    assert bool(_taps(be, "w")) is not navigation_only
    assert bool(_taps(be, "x")) is not navigation_only


@pytest.mark.parametrize("loss", ["expiry", "focus", "disabled", "stop"])
def test_hp_only_interruption_discards_pending_request_without_later_replay(loss):
    kb, be, wall = _leased_keyboard()
    focused = [True]
    kb.is_active = lambda: focused[0]
    kb.set_hp_only(source_time=wall[0])
    kb.request("hp_potion")
    if loss == "expiry":
        wall[0] += .8
    elif loss == "focus":
        focused[0] = False
    elif loss == "disabled":
        kb.set_enabled(False)
    else:
        kb.stop()
    kb.apply_once(0)
    kb.request("hp_potion")
    assert not be.events and not be.held and kb.hp_taps_sent == 0
    focused[0] = True
    kb.set_hp_only(source_time=wall[0])
    kb.set_enabled(True)
    kb.apply_once(.1)
    assert not be.events  # Fresh permission does not resurrect the cancelled tap.
    kb.request("hp_potion")
    kb.apply_once(.2)
    assert kb.hp_taps_sent == 1


@pytest.mark.parametrize("stamp", [9.0, 11.0, float("nan"), float("inf")])
def test_hp_only_invalid_or_old_capture_never_grants_input(stamp):
    kb, be, _ = _leased_keyboard()
    kb.set_hp_only(source_time=stamp)
    kb.request("hp_potion")
    kb.apply_once(0)
    assert not be.events and kb.hp_taps_sent == 0 and kb.hp_tap_time is None


def test_hp_only_renewal_after_expired_capture_drops_request_even_without_keyboard_tick():
    kb, be, wall = _leased_keyboard()
    captured = wall[0]
    wall[0] += .6
    kb.set_hp_only(source_time=captured)
    kb.request("hp_potion")
    wall[0] += .2
    kb.set_hp_only(source_time=wall[0])
    kb.apply_once(0)
    assert not be.events and kb.hp_taps_sent == 0


def test_hp_counter_counts_actual_normal_and_hp_only_taps_and_ignores_mp():
    kb, be, wall = _leased_keyboard()
    kb.set_command(Command.NONE, source_time=wall[0])
    kb.request("mp_potion")
    kb.apply_once(0)
    assert kb.hp_taps_sent == 0 and kb.hp_tap_time is None
    kb.request("hp_potion")
    assert kb.hp_taps_sent == 0
    kb.apply_once(.1)
    assert kb.hp_taps_sent == 1 and kb.hp_tap_time == 10.0
    wall[0] += .2
    kb.set_hp_only(source_time=wall[0])
    kb.request("hp_potion")
    kb.apply_once(.2)
    assert kb.hp_taps_sent == 2 and kb.hp_tap_time == 10.2
    assert len(_taps(be, "1")) == 2


@pytest.mark.parametrize("blank_key", ["", None])
def test_unbound_hp_key_does_not_claim_a_sent_tap(blank_key):
    kb, be, wall = _leased_keyboard(keys={"hp_potion": blank_key})
    kb.set_hp_only(source_time=wall[0])
    kb.request("hp_potion")
    kb.apply_once(0)
    assert not be.events and kb.hp_taps_sent == 0 and kb.hp_tap_time is None


def test_hp_tap_counter_records_keydown_even_if_focus_is_lost_during_tap():
    kb, be, wall = _leased_keyboard()
    focused = [True]
    kb.is_active = lambda: focused[0]
    kb.sleep = lambda _: focused.__setitem__(0, False)
    kb.set_hp_only(source_time=wall[0])
    kb.request("hp_potion")
    kb.apply_once(0)
    assert be.events == [("down", "1"), ("up", "1")]
    assert kb.hp_taps_sent == 1 and kb.hp_tap_time == 10.0 and not be.held
    focused[0] = True
    kb.apply_once(.1)
    assert kb.hp_taps_sent == 1


def test_hp_only_transition_waits_for_inflight_tick_and_releases_its_movement():
    kb, be, wall = _leased_keyboard(keys={"pickup": "x"})
    kb.set_command(Command("left", action="attack"))
    tick_ready, continue_tick = threading.Event(), threading.Event()
    switching, switched = threading.Event(), threading.Event()
    original_hold = kb._apply_hold

    def delayed_hold(*args):
        tick_ready.set()
        assert continue_tick.wait(2)
        original_hold(*args)

    def switch():
        switching.set()
        kb.set_hp_only(source_time=wall[0])
        switched.set()

    kb._apply_hold = delayed_hold
    worker = threading.Thread(target=lambda: kb.apply_once(0))
    setter = threading.Thread(target=switch)
    worker.start()
    try:
        assert tick_ready.wait(2)
        setter.start()
        assert switching.wait(2)
        assert not switched.wait(.05)
    finally:
        continue_tick.set()
        worker.join(2)
        if setter.ident is not None:
            setter.join(2)
    assert not worker.is_alive() and not setter.is_alive()
    assert switched.is_set() and kb.hp_only and not be.held
    before = list(be.events)
    kb.apply_once(.1)
    assert be.events == before  # The old tick cannot append actions after the mode switch returned.
    kb.request("hp_potion")
    kb.apply_once(.2)
    assert be.events[len(before):] == [("down", "1"), ("up", "1")]


@pytest.mark.parametrize("value", [0, -.1, float("nan"), float("inf"), -float("inf"), True, ".1"])
def test_command_rejects_invalid_precision_lease(value):
    with pytest.raises(ValueError, match="finite and positive"):
        Command("right", valid_for_s=value)


def test_precision_lease_is_metadata_and_survives_command_replacement():
    pulse = Command("right", "up", "jump", face_x="left", valid_for_s=.1)
    assert pulse.replace(face_x="none").valid_for_s == .1
    assert pulse.replace(valid_for_s=None).valid_for_s is None
    assert pulse == pulse.replace(valid_for_s=.05)  # not a new jump edge
    assert str(pulse) == "right up jump"
    assert Command.parse(str(pulse)).valid_for_s is None


def test_precision_movement_expires_on_worker_clock_without_a_new_frame():
    kb, be, wall = _leased_keyboard(keys={"pickup": "x"})
    kb.set_command(Command("right", "up", valid_for_s=.1),
                   source_time=wall[0], navigation_only=True)
    kb.apply_once(0)
    deadline = kb._command_deadline  # the pulse's own length runs from its key-down
    assert deadline == pytest.approx(10.1)
    assert be.held == {"right", "up"}
    for tick in (10.025, 10.05, 10.075, 10.099):
        wall[0] = tick
        kb.apply_once(500)  # action/cooldown time must not renew the lease
        assert be.held == {"right", "up"} and kb._command_deadline == deadline
    wall[0] = 10.1
    kb.apply_once(500)
    assert not be.held and kb.cmd == Command.NONE and kb._command_deadline is None
    events = list(be.events)
    wall[0] = 10.3
    kb.apply_once(501)
    assert be.events == events and not _taps(be, "x")


@pytest.mark.parametrize("command_limit,explicit_limit,watchdog,frame_age,expected", [
    (.1, None, .75, 0, .1),
    (.1, .4, .75, 0, .1),
    (.1, .04, .75, 0, .04),
    (.1, None, .06, 0, .06),
    (.1, .4, .75, .72, .03),
    (None, None, .75, 0, .75),
])
def test_precision_lease_uses_shortest_command_argument_watchdog_and_source_limit(
        command_limit, explicit_limit, watchdog, frame_age, expected):
    kb, be, wall = _leased_keyboard(safety={"command_timeout_s": watchdog})
    kb.set_command(Command("left", valid_for_s=command_limit),
                   source_time=wall[0]-frame_age, valid_for_s=explicit_limit)
    kb.apply_once(0)  # key-down at the grant time
    assert kb._command_deadline == pytest.approx(10+expected)
    assert be.held == {"left"}
    wall[0] = 10+expected+.00001
    kb.apply_once(0)
    assert not be.held


@pytest.mark.parametrize("explicit_limit", [0, -.1, float("nan"), float("inf")])
def test_short_command_lease_does_not_hide_invalid_explicit_limit(explicit_limit):
    kb, _, _ = _leased_keyboard()
    with pytest.raises(ValueError, match="finite and positive"):
        kb.set_command(Command("right", valid_for_s=.1), valid_for_s=explicit_limit)


def test_navigation_sanitization_preserves_precision_deadline():
    kb, be, wall = _leased_keyboard()
    kb.set_command(Command("right", action="attack", face_x="left", valid_for_s=.1),
                   navigation_only=True)
    assert kb.cmd.valid_for_s == .1 and kb.cmd.action == kb.cmd.face_x == "none"
    kb.apply_once(0)
    wall[0] = 10.1
    kb.apply_once(.1)
    assert not be.held and not _taps(be, "w")


def test_changing_only_precision_lifetime_does_not_repeat_jump():
    kb, be, wall = _leased_keyboard()
    command = Command("right", action="jump", valid_for_s=.1)
    kb.set_command(command, navigation_only=True)
    kb.apply_once(0)
    wall[0] = 10.02
    kb.set_command(command.replace(valid_for_s=.07), navigation_only=True)
    kb.apply_once(.02)
    assert len(_taps(be, kb.keys["jump"])) == 1


@pytest.mark.parametrize("loss", ["focus", "disable", "stop"])
def test_precision_lease_cannot_outlive_focus_or_explicit_stop(loss):
    kb, be, wall = _leased_keyboard()
    kb.set_command(Command("right", valid_for_s=.1), navigation_only=True)
    kb.apply_once(0)
    wall[0] = 10.03
    if loss == "focus":
        kb.is_active = lambda: False
    elif loss == "disable":
        kb.set_enabled(False)
    else:
        kb.stop()
    kb.apply_once(.03)
    assert not be.held
    kb.is_active = lambda: True
    kb.set_enabled(True)
    kb.apply_once(.04)
    assert not be.held  # restored focus alone cannot resurrect the old pulse


def test_hp_mode_gets_fresh_watchdog_lease_after_precision_movement():
    kb, be, wall = _leased_keyboard()
    kb.set_command(Command("right", valid_for_s=.1), navigation_only=True)
    kb.apply_once(0)
    wall[0] = 10.05
    kb.set_hp_only(source_time=wall[0])
    assert not be.held and kb._command_deadline == pytest.approx(10.8)
    wall[0] = 10.2
    kb.request("hp_potion")
    kb.apply_once(.2)
    assert kb.hp_taps_sent == 1 and _taps(be, kb.keys["hp_potion"])


# -- same-capture repeat sends (Bot.step sends a route command twice per frame) --

def test_same_capture_repeat_of_leased_command_does_not_extend_pulse():
    kb, be, wall = _leased_keyboard()
    captured = wall[0]
    pulse = Command("right", valid_for_s=.119)
    kb.set_command(pulse, source_time=captured, navigation_only=True)
    kb.apply_once(0)
    assert kb._command_deadline == pytest.approx(10.119)
    assert be.held == {"right"}
    wall[0] = 10.09  # vision work between the two sends of one Bot.step
    kb.set_command(pulse, source_time=captured, navigation_only=True)
    assert kb._command_deadline == pytest.approx(10.119)
    kb.apply_once(0)
    assert be.held == {"right"}
    wall[0] = 10.12
    kb.apply_once(0)
    assert not be.held and kb.cmd == Command.NONE


@pytest.mark.parametrize("pressed_first", [True, False])
@pytest.mark.parametrize("second,length", [(.5, .119), (.01, .01)])
def test_same_capture_repeat_with_other_lifetime_only_shortens(second, length, pressed_first):
    kb, _, wall = _leased_keyboard()
    captured = wall[0]
    kb.set_command(Command("right", valid_for_s=.119), source_time=captured)
    if pressed_first:
        kb.apply_once(0)  # running since 10.0: a repeat may only pull its end in
    wall[0] = 10.09
    kb.set_command(Command("right", valid_for_s=second), source_time=captured)
    kb.apply_once(0)
    # Not yet pressed: the shorter length runs from the key-down at 10.09.
    expected = min(10.119, 10.09 + second) if pressed_first else 10.09 + length
    assert kb._command_deadline == pytest.approx(expected)
    assert kb.cmd.valid_for_s == second


def test_same_capture_repeat_cannot_restart_a_pulse_the_worker_already_expired():
    kb, be, wall = _leased_keyboard()
    captured = wall[0]
    pulse = Command("right", valid_for_s=.04)
    kb.set_command(pulse, source_time=captured, navigation_only=True)
    kb.apply_once(0)
    wall[0] = 10.05
    kb.apply_once(0)  # the 30 Hz worker ends the tap by itself
    assert not be.held and kb._command_deadline is None
    events = list(be.events)
    wall[0] = 10.09
    kb.set_command(pulse, source_time=captured, navigation_only=True)
    kb.apply_once(0)
    assert be.events == events and not be.held


@pytest.mark.parametrize("first_pressed", [True, False])
def test_same_capture_other_command_takes_effect_and_cannot_rearm_first_pulse(first_pressed):
    kb, be, wall = _leased_keyboard()
    captured = wall[0]
    kb.set_command(Command("right", valid_for_s=.119), source_time=captured)
    if first_pressed:
        kb.apply_once(0)
        assert be.held == {"right"}
    wall[0] = 10.05
    kb.set_command(Command("left", valid_for_s=.1), source_time=captured)
    kb.apply_once(0)
    assert kb._command_deadline == pytest.approx(10.15)
    assert be.held == {"left"}
    wall[0] = 10.06
    kb.set_command(Command.NONE, source_time=captured)  # unleased: full watchdog lease
    assert kb._command_deadline == pytest.approx(10.75)
    kb.set_command(Command("right", valid_for_s=.119), source_time=captured)
    kb.apply_once(0)
    if first_pressed:
        # The pulse already ran from 10.0 for this capture: never re-armed.
        assert kb._command_deadline == pytest.approx(10.119)
    else:
        # It never went down, so it spent nothing; it still gets only its own length.
        assert kb._command_deadline == pytest.approx(10.06 + .119)
    assert be.held == {"right"}


def test_new_capture_renews_equal_leased_command():
    kb, be, wall = _leased_keyboard()
    pulse = Command("right", valid_for_s=.119)
    kb.set_command(pulse, source_time=wall[0], navigation_only=True)
    kb.apply_once(0)
    wall[0] = 10.09
    kb.set_command(pulse, source_time=wall[0], navigation_only=True)
    assert kb._command_deadline == pytest.approx(10.209)
    wall[0] = 10.2
    kb.apply_once(0)
    assert be.held == {"right"}
    wall[0] = 10.21
    kb.apply_once(0)
    assert not be.held


def test_unleased_hold_renewal_is_unchanged_for_same_and_new_capture():
    kb, be, wall = _leased_keyboard()
    captured = wall[0]
    kb.set_command(Command("right"), source_time=captured)
    assert kb._command_deadline == pytest.approx(10.75)
    wall[0] = 10.09
    kb.set_command(Command("right"), source_time=captured)
    assert kb._command_deadline == pytest.approx(10.75)  # capture age bound, as before
    kb.set_command(Command("right"))  # unstamped renewal
    assert kb._command_deadline == pytest.approx(10.84)
    wall[0] = 10.2
    kb.set_command(Command("right"), source_time=wall[0])
    assert kb._command_deadline == pytest.approx(10.95)
    kb.apply_once(0)
    assert be.held == {"right"} and be.events == [("down", "right")]


def test_unstamped_leased_repeat_still_renews():
    kb, _, wall = _leased_keyboard()
    pulse = Command("right", valid_for_s=.1)
    kb.set_command(pulse)
    kb.apply_once(0)
    wall[0] = 10.05
    kb.set_command(pulse)  # no capture identity, so nothing to deduplicate
    assert kb._command_deadline == pytest.approx(10.15)


@pytest.mark.parametrize("clear", ["hp_only", "disable", "focus"])
def test_mode_change_clears_stored_capture_stamp(clear):
    kb, be, wall = _leased_keyboard()
    captured = wall[0]
    pulse = Command("right", valid_for_s=.119)
    kb.set_command(pulse, source_time=captured, navigation_only=True)
    assert kb._command_source_time == captured and kb._pulse_deadlines
    wall[0] = 10.05
    if clear == "hp_only":
        kb.set_hp_only(source_time=captured)
    elif clear == "disable":
        kb.set_enabled(False)
        kb.set_enabled(True)
    else:
        kb.is_active = lambda: False
        kb.apply_once(0)
        kb.is_active = lambda: True
    assert kb._command_source_time is None and not kb._pulse_deadlines
    kb.set_command(pulse, source_time=captured, navigation_only=True)
    kb.apply_once(0)
    assert kb._command_deadline == pytest.approx(10.169)  # an ordinary fresh lease
