from maplebot.nav.loot import LootCollector
from maplebot.types import Command


def collector(**cfg):
    return LootCollector({"enabled": True, **cfg}, pixels_per_minimap_x=16)


def test_after_target_disappears_walks_to_projected_drop_then_dwells():
    loot = collector()
    loot.observe((40, 61), (500, 400), (564, 400), 0)
    assert loot.goal == (44, 61)
    assert loot.step((40, 61), .1) == Command.NONE and loot.state == "settling"
    assert loot.step((40, 61), .4) == Command.NONE  # brake before the short final approach
    assert loot.step((40, 61), .9) == Command("right")
    assert loot.step((44, 61), 1.) == Command.NONE and loot.state == "dwelling"
    assert loot.step((44, 61), 1.5) == Command.NONE
    assert loot.step((44, 61), 1.9) is None and loot.state == "idle"


def test_detection_ties_do_not_trigger_loot_and_new_combat_replaces_goal():
    loot = collector()
    loot.observe((40, 61), (500, 400), (564, 400), 0)
    assert loot.step((40, 61), .5, has_detections=True) is None
    assert loot.state == "tracking"
    loot.step((40, 61), .6)
    loot.observe((40, 61), (500, 400), (468, 400), .7)
    assert loot.goal == (38, 61) and loot.state == "tracking"
    assert loot.step((40, 61), .8) == Command.NONE
    assert loot.step((40, 61), 1.1) == Command.NONE
    assert loot.step((40, 61), 1.6) == Command("left")


def test_loot_stops_on_different_floor_missing_player_or_pause():
    loot = collector()
    for reason in ("floor", "player", "pause"):
        loot.observe((40, 61), (500, 400), (564, 400), 0)
        if reason == "pause":
            loot.suspend(.1)
            result = loot.step((40, 61), .2)
        else:
            result = loot.step((40, 76) if reason == "floor" else (40, 61), .1,
                               has_player=reason != "player")
        assert result is None and loot.goal is None


def test_unreachable_loot_expires_without_blind_walking_forever():
    loot = collector(max_approach_s=.5, disappear_s=0)
    loot.observe((40, 61), (500, 400), (564, 400), 0)
    assert loot.step((40, 61), .1) == Command.NONE
    loot.step((40, 61), .4)
    assert loot.step((40, 61), .7) is None and loot.state == "idle"


def test_loot_ignores_projected_drop_outside_platform_and_cancels_existing_unsafe_goal():
    loot = collector()
    loot.observe((75, 61), (500, 400), (660, 400), 0, platform_bounds=(20, 80))
    assert loot.goal is None and loot.step((75, 61), .1) is None
    loot.observe((75, 61), (500, 400), (564, 400), .2, platform_bounds=(20, 80))
    assert loot.goal == (79, 61)
    assert loot.step((75, 61), .3, platform_bounds=(20, 78)) is None
    assert loot.goal is None
