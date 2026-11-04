from maplebot.config import deep_merge, load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.types import Command


def _kb(**over):
    cfg = deep_merge(load_config(), over)
    be = RecordingBackend()
    kb = KeyboardController(cfg, be, sleep=lambda s: None)
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
    assert len(_taps(be, "z")) == 2


def test_release_all_clears_state():
    kb, be = _kb()
    kb.set_command(Command("right", "down"))
    kb.apply_once(0.0)
    kb.release_all()
    assert be.held == set()
    kb.apply_once(0.1)
    assert be.held == {"right", "down"}
