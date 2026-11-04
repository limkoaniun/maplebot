import cv2
import numpy as np
import pytest

from maplebot.brain.fsm import Brain, Perception
from maplebot.config import deep_merge, load_config, parse_color_codes
from maplebot.nav.route import RouteFollower, RouteMap
from maplebot.types import Box, Command, Detection

W, H = 1282, 693
P = (641, 346)


class _Route:
    """Stand-in for RouteFollower with a fixed answer."""

    def __init__(self, cmd):
        self.cmd = cmd
        self.calls = []

    def step(self, pos, on_ladder):
        self.calls.append((pos, on_ladder))
        return self.cmd


def _cfg(**over):
    cfg = load_config()
    return deep_merge(cfg, over)


def _p(dets=(), now=0.0, frame_idx=0, player=P, global_pos=(50, 50), minimap_pos=(50, 50),
       minimap_w=150, on_ladder=False):
    return Perception(W, H, player, global_pos, minimap_pos, minimap_w, on_ladder, list(dets), now, frame_idx)


def _det(x, y, w=40, h=30):
    return Detection("m", Box(x, y, w, h), 0.1)


def test_brain_follows_route_without_monsters():
    route = _Route(Command("right"))
    b = Brain(_cfg(), route, 1200)
    assert b.decide(_p()) == Command("right")
    assert route.calls == [((50, 50), False)]
    assert b.state == "hunting"


def test_brain_attacks_left_when_monster_left():
    b = Brain(_cfg(), _Route(Command("right")), 1200)
    cmd = b.decide(_p([_det(500, 330)]))
    assert cmd == Command("left", "none", "attack")


def test_brain_stops_walking_inside_deadzone():
    b = Brain(_cfg(attack={"deadzone_px": 30}), _Route(Command("right")), 1200)
    cmd = b.decide(_p([_det(600, 330)]))  # centre 620, 21px from the player
    assert cmd == Command("none", "none", "attack")


def test_brain_keeps_vertical_route_command_on_ladder():
    b = Brain(_cfg(), _Route(Command("none", "up")), 1200)
    cmd = b.decide(_p([_det(500, 330)], on_ladder=True))
    assert cmd.move_y == "up" and cmd.action == "attack"


def test_brain_uses_aoe_with_crowd():
    b = Brain(_cfg(attack={"aoe_monster_count": 2}), _Route(Command.NONE), 1200)
    cmd = b.decide(_p([_det(500, 330), _det(540, 330)]))
    assert cmd.action == "aoe"


def test_brain_no_aoe_without_aoe_key():
    b = Brain(_cfg(attack={"aoe_monster_count": 2}, keys={"aoe": ""}), _Route(Command.NONE), 1200)
    cmd = b.decide(_p([_det(500, 330), _det(540, 330)]))
    assert cmd.action == "attack"


def test_brain_no_attack_on_tie():
    b = Brain(_cfg(), _Route(Command("right")), 1200)
    assert b.decide(_p([_det(521, 330), _det(721, 330)])) == Command("right")


def test_brain_sweeps_without_route():
    b = Brain(_cfg(search={"trigger_frames": 1}), None, 1200)
    assert b.decide(_p(frame_idx=0)) == Command.NONE
    cmd = b.decide(_p(frame_idx=1, now=0.1))
    assert cmd.move_x in ("left", "right") and cmd.action == "none"
    b2 = Brain(_cfg(search={"trigger_frames": 1, "attack_while_moving": True}), None, 1200)
    b2.decide(_p(frame_idx=0))
    assert b2.decide(_p(frame_idx=1, now=0.1)).action == "attack"


def test_brain_enters_recovering_when_stuck_and_exits_after_timeout():
    b = Brain(_cfg(stuck={"timeout_s": 2.0, "recover_s": 1.0}), _Route(Command("left")), 1200)
    b.decide(_p(now=0.0))
    b.decide(_p(now=1.0))
    cmd = b.decide(_p(now=2.5))
    assert b.state == "recovering"
    assert cmd == Command("right", "none", "jump")       # opposite of the last move, jump once
    assert b.decide(_p(now=2.8)) == Command("right")
    assert b.decide(_p(now=3.6)) == Command("left")      # back to the route
    assert b.state == "hunting"


def test_brain_not_stuck_while_fighting():
    b = Brain(_cfg(stuck={"timeout_s": 1.0}), _Route(Command("left")), 1200)
    for t in (0.0, 0.6, 1.2, 1.8):
        b.decide(_p([_det(500, 330)], now=t))
    assert b.state == "hunting"


def test_brain_stuck_disabled():
    b = Brain(_cfg(stuck={"enable": False, "timeout_s": 1.0}), _Route(Command("left")), 1200)
    for t in (0.0, 0.6, 1.2, 1.8, 5.0):
        b.decide(_p(now=t))
    assert b.state == "hunting"


def test_brain_without_player_fix_still_moves():
    b = Brain(_cfg(), _Route(Command("left")), 1200)
    assert b.decide(_p([_det(500, 330)], player=None)) == Command("left")


def test_hunting_stops_after_grace(tmp_path):
    cfg = _cfg(route={"search_range": 5, "miss_grace_frames": 2})
    d = tmp_path / "r"
    d.mkdir()
    m = np.zeros((50, 50, 3), np.uint8)
    r = m.copy()
    r[10, 10] = (0, 0, 255)  # red in BGR = "left none none"
    cv2.imwrite(str(d / "map.png"), m)
    cv2.imwrite(str(d / "route1.png"), r)
    rm = RouteMap.load(str(d), parse_color_codes(cfg["route"]["color_code"]),
                       parse_color_codes(cfg["route"]["color_code_up_down"]))
    b = Brain(cfg, RouteFollower(rm, cfg["route"]), 1200)
    assert b.decide(_p(global_pos=(10, 10))) == Command("left")
    out = [b.decide(_p(global_pos=(40, 40), now=0.1 * i)) for i in range(1, 4)]
    assert out[0] == Command("left") and out[1] == Command("left") and out[2] == Command.NONE


# --- review fix pass -------------------------------------------------------

def test_brain_keeps_sweeping_when_detection_has_no_target():
    # a monster just outside the attack box (margin only) must not freeze the sweep
    b = Brain(_cfg(search={"trigger_frames": 1, "stall_frames": 0}), None, 1200)
    for i in range(3):
        cmd = b.decide(_p([_det(1000, 330)], frame_idx=i, now=0.1 * i, player=P))
    assert cmd.move_x in ("left", "right")


def test_brain_tie_keeps_previous_target():
    b = Brain(_cfg(), _Route(Command("right")), 1200)
    assert b.decide(_p([_det(500, 330)], now=0.0)).move_x == "left"
    cmd = b.decide(_p([_det(521, 330), _det(721, 330)], now=0.1))
    assert cmd.move_x == "left" and cmd.action == "attack"


def test_brain_stuck_fires_with_untargetable_detection():
    b = Brain(_cfg(stuck={"timeout_s": 1.0, "recover_s": 1.0}), _Route(Command("left")), 1200)
    for t in (0.0, 0.5, 1.2):
        b.decide(_p([_det(1000, 330)], now=t))
    assert b.state == "recovering"
