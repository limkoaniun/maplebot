from maplebot.brain.search import SweepSearcher
from maplebot.types import Command

CFG = {
    "enable": True,
    "trigger_frames": 2,
    "flip_frames": 5,
    "max_offset_px": 0,
    "stall_frames": 3,
    "attack_while_moving": True,
}


def _run(s, frames, progress=None, mm_x=None, mm_w=None, start=0):
    out = []
    for i in range(start, start + frames):
        px = progress(i) if callable(progress) else progress
        out.append(s.update(True, px, mm_x, mm_w, i))
    return out


def test_search_waits_trigger_frames():
    s = SweepSearcher(CFG)
    cmds = _run(s, 3, progress=lambda i: 100.0 + i)
    assert cmds[0] == Command.NONE
    assert cmds[1] == Command.NONE
    assert cmds[2].move_x in ("left", "right") and cmds[2].action == "attack"


def test_search_disabled():
    s = SweepSearcher(dict(CFG, enable=False))
    assert all(c == Command.NONE for c in _run(s, 5, progress=lambda i: 100.0 + i))


def test_search_no_attack_while_moving():
    s = SweepSearcher(dict(CFG, attack_while_moving=False))
    assert _run(s, 3, progress=lambda i: 100.0 + i)[-1].action == "none"


def test_search_flips_on_timer():
    s = SweepSearcher(dict(CFG, stall_frames=0))
    cmds = _run(s, 14, progress=lambda i: 100.0 + i)
    first = cmds[2].move_x
    assert all(c.move_x == first for c in cmds[2:7])
    assert cmds[7].move_x != first
    assert cmds[12].move_x == first


def test_search_flips_on_stall():
    s = SweepSearcher(dict(CFG, flip_frames=1000))
    cmds = _run(s, 8, progress=100.0)  # never moves
    first = cmds[2].move_x
    assert cmds[2].move_x == cmds[3].move_x == cmds[4].move_x == first
    assert cmds[5].move_x != first


def test_search_flips_on_max_offset():
    s = SweepSearcher(dict(CFG, flip_frames=1000, stall_frames=0, max_offset_px=10))
    # moves 4 px per frame in the current direction
    s.update(True, 0.0, None, None, 0)
    s.update(True, 0.0, None, None, 1)
    c = s.update(True, 0.0, None, None, 2)
    sign = -1 if c.move_x == "left" else 1
    x = 0.0
    seen = [c.move_x]
    for i in range(3, 9):
        x += 4 * sign
        seen.append(s.update(True, x, None, None, i).move_x)
    assert seen[0] == seen[1] == seen[2]
    assert seen[3] != seen[0]  # offset 12 >= 10 -> flipped


def test_search_respects_minimap_bounds():
    s = SweepSearcher(dict(CFG, flip_frames=1000, stall_frames=0), minimap_width_bound_pct=0.6)
    s.update(True, 50.0, 50.0, 100, 0)
    s.update(True, 50.0, 50.0, 100, 1)
    s._dir = "left"
    c = s.update(True, 15.0, 15.0, 100, 2)   # left bound is 20
    assert c.move_x == "right"
    s._dir = "right"
    c = s.update(True, 85.0, 85.0, 100, 3)   # right bound is 80
    assert c.move_x == "left"


def test_search_reset_on_monster():
    s = SweepSearcher(CFG)
    _run(s, 3, progress=lambda i: 100.0 + i)
    assert s.update(False, 110.0, None, None, 3) == Command.NONE
    assert s.update(True, 110.0, None, None, 4) == Command.NONE  # trigger count restarted


def test_search_direction_property():
    s = SweepSearcher(CFG)
    assert s.direction in ("left", "right")
