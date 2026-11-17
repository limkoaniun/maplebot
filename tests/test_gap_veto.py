"""attack.pre_transition_gap_veto_x: no in-place pre-takeoff fight against a threat behind a gap edge.

Offline only. A real Brain and MinimapCircuit get scripted minimap positions through the
harness of tests/test_pre_transition_combat.py. The level gap copies the Construction
Site floor-3 gap: take-off floor 89-111 and landing floor 42-88.475, both at y 55, jump
from x 91 to x 85, so the gap edge is x 89 and the flight goes left. The 20:17:31 shape
of the 2026-11-17 run is a monster behind the character while it lines up at 91-95.
Nothing here is a live verification.
"""
import pytest

from maplebot.types import Command
from tests.test_pre_transition_combat import ATTACKS, JUMP, brain, mob, see

SOURCE = {"left": 89, "right": 111, "y": 55}
LANDING = {"left": 42, "right": 88.475, "y": 55}
STAIR = {"left": 80, "right": 88, "y": 64, "right_y": 57}  # a stair below the gap, off the source
GAP = {"start_idx": 1, "loop": False, "platforms": [SOURCE, LANDING, STAIR], "waypoints": [
    {"x": 96, "y": 55, "dwell_s": 0},
    {"x": 85, "y": 55, "action": "jump", "from_x": 91, "from_y": 55, "dwell_s": 0},
    {"x": 80, "y": 55, "dwell_s": 0}]}
DT = .1


def gap_brain(pre=6.0, veto=None, circuit=GAP):
    over = {} if veto is None else {"attack": {"pre_transition_gap_veto_x": veto}}
    return brain(circuit, pre, **over)


def drive(b, positions, dets_for):
    out = []
    for i, pos in enumerate(positions):
        dets = dets_for(i)
        cmd = b.decide(see(pos, round(i * DT, 6), dets))
        out.append((cmd, b.state, b.circuit.idx, b.circuit.state, b._pre_spent))
    return out


APPROACH = [(94, 55)] * 4 + [(93, 55)] * 4 + [(92, 55)] * 4 + [(91, 55)] * 4
BEHIND = lambda i: [mob(60 + 10 * (i % 6))]  # noqa: E731  +60..+110 screen px: the non-flight side


def stream(out):
    return [(cmd, state, idx, nav_state) for cmd, state, idx, nav_state, _ in out]


def test_level_gap_takeoff_is_reported_only_while_the_gap_jump_lines_up():
    b = gap_brain(veto=6.0)
    nav = b.circuit
    assert nav.level_gap_takeoff() is not None  # selected by start_idx, align phase
    gap = nav.level_gap_takeoff()
    assert (gap.direction, gap.edge_x, (gap.source.left, gap.source.right)) == ("left", 89.0, (89, 111))
    assert b.circuit.level_gap_takeoff() == gap  # read-only: asking twice changes nothing
    for _ in range(3):
        nav.step((91, 55), _ * DT)
    nav.step((91, 55), .3)
    assert nav._jump_phase == "flight" and nav.level_gap_takeoff() is None
    assert gap_brain(circuit=JUMP).circuit.level_gap_takeoff() is None  # 10 px height step: not level


def test_takeoff_side_threat_never_fights_at_the_edge():
    """Frames 94 down to 91 (|x - 89| < 6) with a same-platform monster behind the character."""
    vetoed = gap_brain(veto=6.0)
    out = drive(vetoed, APPROACH, BEHIND)
    assert not any(state in ("hunting", "reacquiring_target") for _, state, _, _, _ in out)
    assert all(spent == 0 for *_, spent in out) and vetoed.gap_veto_frames >= 12
    reference = drive(gap_brain(pre=0), APPROACH, BEHIND)
    assert stream(out) == stream(reference)
    assert not any(cmd.action in ATTACKS or cmd.face_x != "none" for cmd, *_ in out)
    # Key off: today's in-place fight at the take-off column (contrast).
    today = gap_brain()
    states = [state for _, state, *_ in drive(today, APPROACH, BEHIND)]
    assert "hunting" in states and today.gap_veto_frames == 0 and today._pre_spent > 0


def test_flight_side_only_detection_keeps_todays_fight():
    between = lambda i: [mob(-30)]  # noqa: E731  between the character and the gap edge
    vetoed, today = gap_brain(veto=6.0), gap_brain()
    a, b = drive(vetoed, APPROACH, between), drive(today, APPROACH, between)
    assert a == b and vetoed.gap_veto_frames == 0
    assert any(state == "hunting" and cmd.action in ATTACKS for cmd, state, *_ in a)


@pytest.mark.parametrize("case", ["stair", "at the limit", "beyond a small key", "not level"])
def test_veto_not_applied_off_source_or_beyond_x_or_non_level(case):
    if case == "stair":  # off the take-off floor: jump_waiting_floor on a stair below the gap
        positions, dets, veto, circuit = [(84, 60)] * 10, lambda i: [mob(48)], 6.0, GAP
    elif case == "at the limit":  # |95 - 89| = 6 is not closer than 6
        positions, dets, veto, circuit = [(95, 55)] * 10, BEHIND, 6.0, GAP
    elif case == "beyond a small key":
        positions, dets, veto, circuit = [(93, 55)] * 4 + [(92, 55)] * 6, BEHIND, 3.0, GAP
    else:  # the JUMP geometry of tests/test_pre_transition_combat.py: a 10 px height step
        positions, dets, veto, circuit = [(27, 50)] * 10, lambda i: [mob(-60)], 6.0, JUMP
    vetoed, today = gap_brain(veto=veto, circuit=circuit), gap_brain(circuit=circuit)
    a, b = drive(vetoed, positions, dets), drive(today, positions, dets)
    assert a == b and vetoed.gap_veto_frames == 0
    assert any(state == "hunting" for _, state, *_ in a)
    if case == "stair":
        assert {nav_state for *_, nav_state, _ in a} <= {"initializing", "jump_waiting_floor"}


def test_veto_frames_never_attack_and_never_charge_beyond_today():
    # The monster changes side every few frames: fights and vetoes interleave.
    mixed = lambda i: [mob(-30)] if (i // 3) % 2 else [mob(70)]  # noqa: E731
    positions = [(92, 55)] * 40
    vetoed, today = gap_brain(pre=2.0, veto=6.0), gap_brain(pre=2.0)
    a, vetoes, previous = [], [], 0
    for i, pos in enumerate(positions):
        cmd = vetoed.decide(see(pos, round(i * DT, 6), mixed(i)))
        a.append((cmd, vetoed.state, vetoed.circuit.idx, vetoed.circuit.state, vetoed._pre_spent))
        vetoes.append(vetoed.gap_veto_frames > previous)
        previous = vetoed.gap_veto_frames
    b = drive(today, positions, mixed)
    assert sum(vetoes) > 0 and "hunting" in {row[1] for row in b}
    fighting = ("hunting", "reacquiring_target")
    for i, (cmd, state, _, _, spent) in enumerate(a):
        before = a[i - 1][4] if i else 0.0
        if vetoes[i]:
            assert cmd.action not in ATTACKS and cmd.face_x == "none" and state.startswith("circuit_")
            # A vetoed frame is charged only as the frame after a fight frame, which today's
            # fight window charges too; otherwise it costs the allowance nothing.
            if not (i and a[i - 1][1] in fighting):
                assert spent == before
        assert spent - before <= DT + 1e-9  # at most one frame's time per frame, as today
        assert spent <= vetoed._pre_transition_s + DT + 1e-9  # the allowance's own bound
        if cmd.action in ATTACKS:
            assert state == "hunting"
    assert vetoes[:3] == [True] * 3 and not any(vetoes[3:6])  # behind, then flight side
    assert any(cmd == Command("none", "none", "attack", face_x="left") for cmd, *_ in a)


# --- review findings: pins for rules no earlier test exercised --------------------------------

R_SOURCE = {"left": 69, "right": 91, "y": 55}
R_LANDING = {"left": 91.525, "right": 138, "y": 55}
R_GAP = {"start_idx": 1, "loop": False, "platforms": [R_SOURCE, R_LANDING], "waypoints": [
    {"x": 84, "y": 55, "dwell_s": 0},
    {"x": 95, "y": 55, "action": "jump", "from_x": 89, "from_y": 55, "dwell_s": 0},
    {"x": 100, "y": 55, "dwell_s": 0}]}  # the mirror image: flight to the right, edge x 91
R_APPROACH = [(86, 55)] * 4 + [(87, 55)] * 4 + [(88, 55)] * 4 + [(89, 55)] * 4


def test_rightward_gap_veto_fires_only_for_a_threat_on_the_left():
    gap = gap_brain(veto=6.0, circuit=R_GAP).circuit.level_gap_takeoff()
    assert gap is not None and (gap.direction, gap.edge_x) == ("right", 91.0)
    behind = lambda i: [mob(-60 - 10 * (i % 4))]  # noqa: E731  left of the player: away from the flight
    vetoed = gap_brain(veto=6.0, circuit=R_GAP)
    out = drive(vetoed, R_APPROACH, behind)
    assert vetoed.gap_veto_frames >= 12
    assert not any(state in ("hunting", "reacquiring_target") for _, state, *_ in out)
    assert "hunting" in {state for _, state, *_ in drive(gap_brain(circuit=R_GAP), R_APPROACH, behind)}
    flight = lambda i: [mob(30)]  # noqa: E731  right of the player: between it and the edge
    a, b = gap_brain(veto=6.0, circuit=R_GAP), gap_brain(circuit=R_GAP)
    assert drive(a, R_APPROACH, flight) == drive(b, R_APPROACH, flight) and a.gap_veto_frames == 0


def test_veto_ignores_a_raw_detection_that_the_platform_filter_removed():
    # A flight-side same-platform monster (fought today) and one on the non-flight side beyond
    # the take-off floor's right end (minimap x ~113 > 111), which the platform filter removes.
    dets = lambda i: [mob(-30), mob(340)]  # noqa: E731
    vetoed, today = gap_brain(veto=6.0), gap_brain()
    a, b = drive(vetoed, [(92, 55)] * 10, dets), drive(today, [(92, 55)] * 10, dets)
    assert vetoed.gap_veto_frames == 0 and a == b
    assert any(state == "hunting" for _, state, *_ in a)


LOWER = {"left": 90, "right": 100, "y": 61}  # a floor 6 px under the take-off floor, inside its x range
GAP_LOWER = {**GAP, "platforms": [SOURCE, LANDING, LOWER]}


def test_veto_needs_strict_support_on_the_source_floor():
    dets = lambda i: [mob(48)]  # noqa: E731  behind the player, on the lower floor with it
    vetoed, today = gap_brain(veto=6.0, circuit=GAP_LOWER), gap_brain(circuit=GAP_LOWER)
    a, b = drive(vetoed, [(92, 61)] * 10, dets), drive(today, [(92, 61)] * 10, dets)
    assert any(state == "hunting" for _, state, *_ in b)
    assert a == b and vetoed.gap_veto_frames == 0


def test_veto_needs_the_character_on_the_source_x_range():
    # x 87 is past the gap edge (89), on the landing floor at the same height.
    dets = lambda i: [mob(20)]  # noqa: E731
    vetoed, today = gap_brain(veto=6.0), gap_brain()
    a, b = drive(vetoed, [(87, 55)] * 8, dets), drive(today, [(87, 55)] * 8, dets)
    assert any(state == "hunting" for _, state, *_ in b)
    assert a == b and vetoed.gap_veto_frames == 0


def test_vetoed_frame_ends_the_fight_so_no_reacquire_window_follows():
    # Fight a flight-side monster, then one behind (vetoed), then nothing: the empty frame
    # after a veto steps the circuit, exactly as after a frame with no pre-fight.
    seq = lambda i: [mob(-30)] if i < 3 else [mob(70)] if i == 3 else []  # noqa: E731
    b = gap_brain(veto=6.0)
    out = drive(b, [(92, 55)] * 6, seq)
    assert [s for _, s, *_ in out[:3]] == ["hunting"] * 3 and b.gap_veto_frames == 1
    assert out[4][1].startswith("circuit_"), out[4][1]


def _site_veto_frames(pos, veto, frames=12):
    from tests.test_gap_reentry import brain_at, mob as site_mob, see as site_see
    b = brain_at(17, pre=6.0, veto=veto)
    rows = [(b.decide(site_see(pos, round(i * DT, 6), [site_mob(70)])), b.state, b.circuit.state)
            for i in range(frames)]
    return b, rows


@pytest.mark.parametrize("pos", [(89, 55), (89, 56)])
def test_veto_never_acts_at_the_ambiguous_stair_top_column(pos):
    """Review finding: at x=89 the stair-2 top overlaps floor-3-right; that is not strictly the take-off floor."""
    from maplebot.nav.recovery import supported_platform
    from tests.test_wrong_floor_rejoin import site
    try:
        strict = supported_platform(site(), pos)["id"]
    except ValueError:
        strict = None
    assert strict != "floor-3-right"  # ambiguous at (89,55), stair-2 at (89,56)
    vetoed, a = _site_veto_frames(pos, 6.0)
    _, b = _site_veto_frames(pos, None)
    assert vetoed.gap_veto_frames == 0 and a == b


def test_veto_still_acts_strictly_on_floor_3_right():
    vetoed, rows = _site_veto_frames((93, 55), 6.0)  # still lining up toward 91
    assert vetoed.gap_veto_frames == len(rows)
    assert not any(cmd.action in ATTACKS or cmd.face_x != "none" for cmd, *_ in rows)
    _, today = _site_veto_frames((93, 55), None)
    assert "hunting" in {state for _, state, _ in today}
