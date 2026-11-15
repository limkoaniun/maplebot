from copy import deepcopy

import pytest

from maplebot.config import load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.types import Command
from scripts.verify_potions import OneTapBackend, PotionProof, potion_config


def test_potion_config_keeps_farming_geometry_without_mutating_saved_settings():
    original = load_config()
    before = deepcopy(original)
    result = potion_config(original)
    assert original == before
    assert result["route"] == original["route"] and result["minimap"] == original["minimap"]
    assert result["keys"]["hp_potion"] == original["keys"]["hp_potion"]
    assert result["keys"]["mp_potion"] == original["keys"]["mp_potion"]
    assert not result["buffs"]["keys"]
    assert all(result["keys"][key] == "" for key in ("left", "right", "up", "down", "jump", "teleport",
                                                     "attack", "aoe", "normal_attack", "pickup"))


@pytest.mark.parametrize("resource", ["hp", "mp"])
def test_proof_requires_post_tap_fresh_bar_increase_not_key_count(resource):
    proof = PotionProof(resource, 95, sequence=10, tap_time=100)
    assert not proof.observe(100, 10, 100.2, 100.2)
    assert not proof.observe(100, 11, 99.9, 100.2)
    assert not proof.observe(96.9, 12, 100.3, 100.3)
    assert proof.observe(97, 13, 100.5, 100.5)
    with pytest.raises(RuntimeError, match="3 seconds"):
        proof.observe(100, 14, 103.1, 103.1)


@pytest.mark.parametrize("baseline", [None, float("nan"), 98, 100])
def test_full_or_unreadable_baseline_cannot_be_claimed_as_a_pass(baseline):
    with pytest.raises(ValueError):
        PotionProof("hp", baseline, sequence=1, tap_time=0)


def test_exactly_one_actual_tap_per_resource_and_no_other_keys():
    original = RecordingBackend()
    backend = OneTapBackend(original, clock=lambda: 1.)
    backend.arm("hp", "q")
    backend.down("q")
    backend.up("q")
    with pytest.raises(RuntimeError):
        backend.down("q")
    with pytest.raises(RuntimeError):
        backend.down("right")
    backend.arm("mp", "a")
    backend.down("a")
    backend.up("a")
    assert backend.counts == {"hp": 1, "mp": 1}
    assert original.events == [("down", "q"), ("up", "q"), ("down", "a"), ("up", "a")]


@pytest.mark.parametrize("resource", ["hp", "mp"])
def test_request_cancelled_by_lease_expiry_cannot_fall_through_to_other_actions(resource):
    cfg = potion_config(load_config())
    clock = [1.]
    record = RecordingBackend()
    backend = OneTapBackend(record, clock=lambda: clock[0])
    keyboard = KeyboardController(cfg, backend, clock=lambda: clock[0], sleep=lambda _: None)
    key = cfg["keys"][resource + "_potion"]
    backend.arm(resource, key)
    if resource == "hp":
        keyboard.set_hp_only(source_time=clock[0])
    else:
        keyboard.set_command(Command.NONE, source_time=clock[0])
    keyboard.request(resource + "_potion")
    clock[0] += .8
    keyboard.apply_once(clock[0])
    assert not record.events and not backend.counts
