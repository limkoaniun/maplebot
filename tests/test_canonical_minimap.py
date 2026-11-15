"""Confirmed scrolling offsets preserve saved navigation coordinates."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from maplebot.map_profiles import MapMatch
from maplebot.types import Command
from tests.test_bot import _cfg, _live_bot, _vertical_frame


@dataclass(frozen=True)
class ShiftedMatch(MapMatch):
    offset: tuple | None = (0, 0)


def _bot(monkeypatch):
    cfg = _cfg(minimap={"fixed_rect": [11, 61, 150, 120], "max_jump_px": 5},
               route={"circuit": {"enabled": True, "start_idx": 0,
                   "waypoints": [{"x": 40, "y": 30}, {"x": 80, "y": 30}]}},
               map_profile={"active": "scrolling-map", "reference_sha256": "reference"},
               keys={"pickup": "", "hp_potion": "", "mp_potion": ""},
               buffs={"keys": []}, health={"enable": False})
    bot, backend = _live_bot(_vertical_frame((50, 30)), cfg)
    profile = SimpleNamespace(key="scrolling-map", image_sha256="reference",
                              reference_bgr=np.zeros((120, 150, 3), np.uint8))
    match = {"value": ShiftedMatch("matched", profile=profile)}
    bot.map_recognizer = SimpleNamespace(observe=lambda *args: match["value"], reset=lambda: None)
    bot.map_ready = False
    monkeypatch.setattr("maplebot.map_profiles.profile_config_matches", lambda *args, **kwargs: True)
    return bot, backend, match, profile


@pytest.mark.parametrize("offset,raw", [((23, 0), (27, 30)), ((23, 10), (27, 20)), ((-20, 0), (70, 30))])
def test_confirmed_scroll_keeps_dot_and_walk_command_in_canonical_coordinates(monkeypatch, offset, raw):
    bot, backend, match, profile = _bot(monkeypatch)
    initial = bot.step(0)
    assert initial.minimap_pos == (50, 30) and initial.command == Command("left")
    match["value"] = ShiftedMatch("matched", profile=profile, offset=offset)
    bot.source.frame = _vertical_frame(raw)
    shifted = bot.step(.1)
    # With the +23 scroll, a raw-coordinate route would wrongly turn right to40.
    assert shifted.minimap_pos == (50, 30) and shifted.command == Command("left")
    assert bot.minimap_raw_pos == raw and bot.minimap_offset == offset and bot._minimap_seen
    assert bot.minimap_image.shape == profile.reference_bgr.shape
    assert tuple(bot.minimap_image[30, 50]) == tuple(bot.cfg["minimap"]["player_color_bgr"])
    assert tuple(bot.minimap_raw_image[raw[1], raw[0]]) == tuple(bot.cfg["minimap"]["player_color_bgr"])
    assert not bot.paused and not backend.events  # no real keyboard thread runs in this test


@pytest.mark.parametrize("state,offset", [("ambiguous", (23, 0)), ("candidate", (23, 0)),
    ("unknown", (0, 0)), ("matched", None), ("matched", (float("nan"), 0)),
    ("matched", (23.5, 0))])
def test_unknown_or_invalid_alignment_releases_input_and_clears_coordinate_evidence(monkeypatch, state, offset):
    bot, backend, match, profile = _bot(monkeypatch)
    bot.step(0)
    bot.keyboard.apply_once(0)
    assert backend.held == {"left"}
    match["value"] = ShiftedMatch(state, profile=profile, offset=offset)
    snap = bot.step(.1)
    assert snap.command == Command.NONE and not bot.map_ready
    assert bot.map_reacquiring if state in ("unknown", "candidate", "ambiguous") else bot.full_pause
    assert not backend.held and not bot.keyboard.enabled
    assert bot.minimap_offset is None and bot.minimap_pos is None and bot.minimap_raw_pos is None
    assert bot.minimap_image is None and bot.minimap_raw_image is None
    assert bot._update_minimap(bot.source.frame) is None and not bot._minimap_seen


def test_lost_capture_discards_previous_canonical_image_and_offset(monkeypatch):
    bot, _, _, _ = _bot(monkeypatch)
    bot.step(0)
    assert bot.minimap_image is not None
    bot.source.frame = None
    assert bot.step(.1) is None
    assert bot.minimap_image is None and bot.minimap_raw_pos is None and bot.minimap_offset is None


def test_legacy_unprofiled_global_locator_receives_unwarped_crop(monkeypatch):
    bot, _ = _live_bot(_vertical_frame((50, 30)), _cfg(minimap={"fixed_rect": [11, 61, 150, 120]}))
    image = bot._update_minimap(bot.source.frame)
    assert np.array_equal(image, bot.source.frame[61:181, 11:161])
    bot.route_map = SimpleNamespace(map_bgr=np.zeros((250, 250, 3), np.uint8))
    observed = []

    def locate(reference, crop, last):
        observed.append(crop)
        return SimpleNamespace(score=0, loc=(10, 20))

    monkeypatch.setattr("maplebot.bot.locate_minimap_on_map", locate)
    bot._update_global(image)
    assert observed[0] is image and bot.global_pos == (60, 50)
    assert bot.minimap_raw_pos == bot.minimap_pos == (50, 30) and bot.minimap_offset == (0, 0)
