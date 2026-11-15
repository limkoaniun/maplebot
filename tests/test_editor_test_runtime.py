from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from maplebot.config import load_config
from maplebot.editor_bridge import capture_signature
from maplebot.editor_test_runtime import EditorTestSession
from maplebot.types import Box


def prepared(tmp_path, **over):
    cfg = load_config()
    cfg["health"]["enable"] = True
    cfg["buffs"] = {"keys": ["f4"], "cooldowns_s": [60]}
    cfg["loot"] = {"enabled": True}
    cfg["keys"].update(attack="z", aoe="v", normal_attack="alt", pickup="x", hp_potion="1", mp_potion="2")
    rng = np.random.default_rng(81)
    reference = rng.integers(0, 255, (80, 120, 3), dtype=np.uint8)
    test_id = "a" * 32
    cv2.imwrite(str(tmp_path / f"editor-test-{test_id}.png"), reference)
    request = {"version": 1, "test_id": test_id, "created": 100,
               "reference": f"editor-test-{test_id}.png", "capture": capture_signature(cfg),
               "source_platform": {"left": 10, "right": 60, "y": 50},
               "circuit": {"enabled": True, "waypoints": [{"x": 20, "y": 50}, {"x": 40, "y": 50}]},
               **over}
    snap = SimpleNamespace(frame=reference.copy(), minimap_box=Box(0, 0, 120, 80), minimap_pos=(20, 50))
    return cfg, request, snap


def session(tmp_path, **over):
    cfg, request, snap = prepared(tmp_path, **over)
    return EditorTestSession(request, cfg, tmp_path, wall_now=101, now=0), cfg, snap


def after(test, snap, *, now=0, foreground=True, minimap_seen=True, circuit=None):
    return test.after_step(snap, test.cfg, now=now, foreground=foreground,
                           minimap_seen=minimap_seen, circuit=circuit)


def test_ephemeral_config_disables_nonmovement_actions_without_mutating_saved_config(tmp_path):
    cfg, request, _ = prepared(tmp_path)
    original = deepcopy(cfg)
    test = EditorTestSession(request, cfg, tmp_path, wall_now=101, now=0)
    assert cfg == original
    assert not test.cfg["health"]["enable"] and test.cfg["buffs"]["keys"] == []
    assert not test.cfg["loot"]["enabled"]
    assert all(test.cfg["keys"][k] == "" for k in ("attack", "aoe", "normal_attack", "pickup", "hp_potion", "mp_potion"))
    assert test.cfg["keys"]["jump"] == cfg["keys"]["jump"]
    circuit = test.cfg["route"]["circuit"]
    assert circuit["movement_test"] and not circuit["loop"] and circuit["max_retries"] == 1


@pytest.mark.parametrize("change", [
    {"created": -20}, {"created": 102}, {"version": 2}, {"test_id": "../bad"},
    {"reference": "../editor-test-" + "a" * 32 + ".png"}, {"capture": {}},
])
def test_request_rejects_expiry_future_time_wrong_version_identity_path_or_capture(tmp_path, change):
    cfg, request, _ = prepared(tmp_path, **change)
    with pytest.raises(ValueError):
        EditorTestSession(request, cfg, tmp_path, wall_now=101, now=0)


def test_preflight_requires_paused_fresh_snapshot_then_foreground_before_motion(tmp_path):
    test, _, snap = session(tmp_path)
    assert not test.before_step(0, False)
    assert not after(test, snap, foreground=False)
    assert test.state == "waiting_focus" and test.preflight_done and not test.has_run
    assert not test.before_step(1, True)  # still paused for a fresh foreground preflight
    assert after(test, snap, now=1)
    assert test.state == "running" and not test.has_run
    assert test.before_step(1.1, True) and test.has_run


@pytest.mark.parametrize("wrong", ["map", "shape", "source", "dot", "capture"])
def test_preflight_failure_never_permits_motion(tmp_path, wrong):
    test, _, snap = session(tmp_path)
    assert not test.before_step(0, True)
    if wrong == "map":
        snap.frame[:] = 0
    elif wrong == "shape":
        snap.minimap_box = Box(0, 0, 100, 80)
    elif wrong == "source":
        snap.minimap_pos = (90, 50)
    elif wrong == "dot":
        snap.minimap_pos = None
    else:
        test.cfg["game_window"]["working_size"] = [1, 1]
    assert not after(test, snap)
    assert test.state == "error" and not test.has_run
    assert not test.before_step(.1, True)


def test_source_position_is_rechecked_when_focus_arrives(tmp_path):
    test, _, snap = session(tmp_path)
    test.before_step(0, False)
    after(test, snap, foreground=False)
    snap.minimap_pos = (90, 50)
    assert not test.before_step(1, True)
    assert not after(test, snap, now=1) and test.state == "error"


def test_shifted_profile_preflight_uses_confirmed_alignment_and_canonical_source_position(tmp_path):
    test, _, snap = session(tmp_path, source_platform={"left": 40, "right": 60, "y": 50})
    test.cfg["map_profile"] = {"active": "scrolling-map"}
    snap.frame = cv2.warpAffine(snap.frame, np.float32([[1, 0, -23], [0, 1, 0]]), (120, 80))
    snap.minimap_pos = (50, 50)  # raw dot would be27, outside this starting platform
    assert not test.before_step(0, True)
    assert test.after_step(snap, test.cfg, now=0, foreground=True, minimap_seen=True,
                           circuit=None, minimap_offset=(23, 0))
    assert test.state == "running" and test.preflight_done and not test.has_run


@pytest.mark.parametrize("offset", [None, (0, 0), (24, 0), (float("nan"), 0)])
def test_profile_preflight_refuses_missing_or_inconsistent_alignment(tmp_path, offset):
    test, _, snap = session(tmp_path)
    test.cfg["map_profile"] = {"active": "scrolling-map"}
    snap.frame = cv2.warpAffine(snap.frame, np.float32([[1, 0, -23], [0, 1, 0]]), (120, 80))
    assert not test.after_step(snap, test.cfg, now=0, foreground=True, minimap_seen=True,
                               circuit=None, minimap_offset=offset)
    assert test.state == "error" and not test.has_run
    assert not test.before_step(.1, True)


@pytest.mark.parametrize("position,allowed", [((40, 30), True), ((40, 50), False), ((70, 0), False)])
def test_sloped_source_preflight_uses_observed_x_not_left_endpoint_height(tmp_path, position, allowed):
    test, _, snap = session(tmp_path, source_platform={"left": 20, "right": 60, "y": 50, "right_y": 10})
    snap.minimap_pos = position
    assert after(test, snap) is allowed
    assert test.source["right_y"] == 10


def test_sloped_source_rejects_out_of_image_endpoint(tmp_path):
    cfg, request, _ = prepared(tmp_path, source_platform={"left": 20, "right": 60, "y": 50, "right_y": 90})
    with pytest.raises(ValueError, match="outside"):
        EditorTestSession(request, cfg, tmp_path, wall_now=101, now=0)


def test_piecewise_source_preflight_uses_interior_shelf(tmp_path):
    source = {"left": 10, "right": 70, "y": 65, "right_y": 20,
              "knots": [[20, 45], [55, 45]]}
    test, _, snap = session(tmp_path, source_platform=source)
    snap.minimap_pos = (50, 45)
    assert after(test, snap)
    assert test.source["knots"] == source["knots"]


@pytest.mark.parametrize("knots", [[[25, 81]], [[25, float("nan")]], [[35, 30], [25, 35]], [[60, 30]]])
def test_piecewise_source_rejects_invalid_interior_points(tmp_path, knots):
    cfg, request, _ = prepared(tmp_path, source_platform={"left": 20, "right": 60, "y": 50, "knots": knots})
    with pytest.raises(ValueError):
        EditorTestSession(request, cfg, tmp_path, wall_now=101, now=0)


def test_focus_wait_and_total_test_deadlines_are_checked_before_movement(tmp_path):
    test, _, snap = session(tmp_path)
    test.before_step(0, False)
    after(test, snap, foreground=False)
    assert not test.before_step(60, True) and test.state == "blocked"
    test, _, snap = session(tmp_path)
    test.before_step(0, True)
    after(test, snap)
    assert test.before_step(89, True)
    assert not test.before_step(90, True) and test.state == "blocked"


@pytest.mark.parametrize("timeout_s", [.2, 1.0, 1.5, 15, 99])
def test_connection_test_with_commit_budget_accepts_its_own_clamped_timeout(tmp_path, timeout_s):
    # The session clamps timeout_s into 1-15 s. The commit_budget defaults derived from it must
    # not refuse the lower end under another key's name ("climb_travel_timeout_s must be ...").
    circuit = {"enabled": True, "commit_budget": True, "timeout_s": timeout_s,
               "waypoints": [{"x": 20, "y": 50}, {"x": 40, "y": 50}]}
    test, _, _ = session(tmp_path, circuit=circuit)
    built = test.cfg["route"]["circuit"]
    assert built["commit_budget"] and built["timeout_s"] == min(15.0, max(1.0, timeout_s))


@pytest.mark.parametrize("result", ["completed", "blocked", "cancelled", "capture"])
def test_test_terminal_states_never_reenable_movement(tmp_path, result):
    test, _, snap = session(tmp_path)
    test.before_step(0, True)
    after(test, snap)
    assert test.before_step(.1, True)
    if result == "cancelled":
        test.finish("cancelled", "Stopped by the user.")
        assert not after(test, snap, now=.2)
    elif result == "capture":
        assert not after(test, None, now=.2)
        assert test.state == "blocked"
    else:
        nav = SimpleNamespace(completed=result == "completed", state=result)
        assert not after(test, snap, now=.2, circuit=nav)
        assert test.state == result
    assert test.terminal and not test.before_step(.3, True)
