import json
import os

import cv2
import numpy as np
import pytest

from maplebot.bot import Bot, BotSnapshot
from maplebot.config import PACKAGE_ROOT, deep_merge, load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
from maplebot.evidence import (BUCKETS, EvidenceTracker, activity_bucket, carry_over_tracker, record_snapshot,
                               session_summary_lines, summary_line)
from maplebot.vision.monsters import MonsterLibrary
from tests.synth import make_frame

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gains")
EXP_XY, LINES_XY = (738, 737), (1150, 576)  # where the fixture crops sat in the 1366x768 prepared frame


def _img(name):
    img = cv2.imread(os.path.join(FIX, name + ".png"))
    assert img is not None, name
    return img


def _frame(exp="exp-text-4975-a", lines="lines-empty", size=(1366, 768)):
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    for name, (x, y) in ((exp, EXP_XY), (lines, LINES_XY)):
        if name is not None:
            crop = _img(name)
            frame[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return frame


def _tracker(**over):
    cfg = dict(load_config()["evidence"], **over)
    return EvidenceTracker(cfg, wall_clock=lambda: 1234.5)


def _run(tracker, steps, state="hunting", dt=0.1, start=10.0):
    for i, step in enumerate(steps):
        tracker.update(_frame(**step), start + i * dt, state)
    return tracker.snapshot()


def _lines(*names):
    return [dict(lines=name) for name in names]


def test_default_config_is_disabled_and_templates_exist():
    cfg = load_config()["evidence"]
    assert cfg["enabled"] is False
    assert cfg["exp_text_rect"] == [738, 737, 107, 14] and cfg["gain_line_rect"] == [1150, 604, 214, 16]
    assert all(os.path.isfile(os.path.join(PACKAGE_ROOT, path)) for path in cfg["templates"].values())


def test_unchanged_exp_text_is_never_counted():
    snap = _run(_tracker(), [dict(exp="exp-text-4975-a"), dict(exp="exp-text-4975-b")] * 5)
    assert snap["exp_text_changes"] == 0 and snap["last_exp_text_change_at"] is None
    assert snap["frames_observed"] == 10 and snap["exp_text_unreadable_frames"] == 0


def test_changed_exp_text_counts_once_when_it_repeats():
    steps = [dict(exp=n) for n in ("exp-text-4975-a", "exp-text-4975-b", "exp-text-4990", "exp-text-4990",
                                   "exp-text-4990", "exp-text-next", "exp-text-next")]
    tracker = _tracker()
    assert _run(tracker, steps[:3])["exp_text_changes"] == 0  # seen once: not yet confirmed
    snap = _run(tracker, steps[3:], start=10.3)
    assert snap["exp_text_changes"] == 2 and snap["last_exp_text_change_at"] == 1234.5


def test_one_frame_overlay_on_the_exp_text_is_not_a_change():
    steps = [dict(exp=n) for n in ("exp-text-4975-a", "exp-text-4990", "exp-text-4975-b", "exp-text-next",
                                   "exp-text-4975-a")]
    assert _run(_tracker(), steps)["exp_text_changes"] == 0


def test_scenery_under_the_exp_rect_is_unreadable_not_counted():
    tracker = _tracker()
    for i in range(6):  # a flickering scene where this layout has no EXP text
        frame = _frame(exp=None)
        frame[737:751, 738:845] = 255 if i % 2 else 0
        tracker.update(frame, 1.0 + i * 0.1, "hunting")
    snap = tracker.snapshot()
    assert snap["exp_text_changes"] == 0 and snap["exp_text_unreadable_frames"] == 6


def test_fading_line_is_counted_once_and_classified():
    names = ["lines-empty"] + [f"lines-decay-{i}" for i in range(5) for _ in range(2)] + ["lines-empty"]
    snap = _run(_tracker(), _lines(*names))
    assert (snap["gain_lines_total"], snap["exp_lines"], snap["meso_lines"], snap["other_lines"]) == (1, 1, 0, 0)
    assert snap["last_gain_line_kind"] == "exp" and snap["last_gain_line_at"] == 1234.5


def test_meso_exp_and_item_lines_are_told_apart():
    tracker = _tracker()
    assert _run(tracker, _lines("lines-empty", "lines-exp", "lines-exp"))["exp_lines"] == 1
    snap = _run(tracker, _lines("lines-empty", "lines-item-over-meso", "lines-empty"), start=11.0)
    assert (snap["gain_lines_total"], snap["other_lines"]) == (2, 1)
    # A fresh meso line above an already visible EXP line: both arrived since the empty frame.
    snap = _run(tracker, _lines("lines-meso-exp-item", "lines-meso-exp-item"), start=12.0)
    assert (snap["gain_lines_total"], snap["exp_lines"], snap["meso_lines"], snap["other_lines"]) == (4, 2, 1, 1)
    assert snap["last_gain_line_kind"] == "exp"  # slots are read newest first


def test_new_line_over_a_visible_one_and_repeated_text_are_counted():
    tracker = _tracker()
    snap = _run(tracker, _lines("lines-shift-before", "lines-shift-before", "lines-shift-after", "lines-shift-after"))
    assert (snap["gain_lines_total"], snap["exp_lines"]) == (1, 1)  # the first frame is only a baseline
    tracker.reset_baseline()
    snap = _run(tracker, _lines("lines-repeat-before", "lines-repeat-after", "lines-repeat-after"), start=20.0)
    assert (snap["gain_lines_total"], snap["exp_lines"]) == (2, 2)
    tracker.reset_baseline()
    snap = _run(tracker, _lines("lines-double-before", "lines-double-after"), start=30.0)
    assert snap["gain_lines_total"] == 4 and snap["other_lines"] >= 1


def test_gap_resets_the_baseline_so_nothing_is_fabricated():
    tracker = _tracker()
    tracker.update(_frame("exp-text-4975-a", "lines-empty"), 1.0, "hunting")
    for t in (5.0, 5.1, 5.2):  # > max_gap_s later the screen differs: unknown, not an event
        tracker.update(_frame("exp-text-4990", "lines-exp"), t, "hunting")
    snap = tracker.snapshot()
    assert snap["exp_text_changes"] == 0 and snap["gain_lines_total"] == 0
    assert snap["baseline_resets"] == 1 and snap["seconds_total"] == pytest.approx(0.2)


def test_non_fresh_frames_never_count_and_reset_the_baseline():
    tracker = _tracker()
    tracker.update(_frame("exp-text-4975-a", "lines-empty"), 1.0, "hunting")
    tracker.update(_frame("exp-text-4990", "lines-exp"), 1.1, "hunting", fresh=False)  # paused / map lost / stale
    tracker.update(None, 1.2, "hunting", fresh=False)
    tracker.update(_frame("exp-text-4990", "lines-exp"), 1.3, "hunting")
    tracker.update(_frame("exp-text-4990", "lines-exp"), 1.4, "hunting")
    snap = tracker.snapshot()
    assert snap["exp_text_changes"] == 0 and snap["gain_lines_total"] == 0
    assert snap["frames_observed"] == 3 and snap["seconds_total"] == pytest.approx(0.1)


def test_same_instant_twice_is_ignored():
    tracker = _tracker()
    tracker.update(_frame("exp-text-4975-a", "lines-empty"), 1.0, "hunting")
    for _ in range(3):  # a repeated call for one capture must not confirm or count anything
        tracker.update(_frame("exp-text-4990", "lines-exp"), 1.0, "hunting")
    tracker.update(_frame("exp-text-4990", "lines-exp"), 0.5, "hunting")  # clock going backwards
    snap = tracker.snapshot()
    assert snap["exp_text_changes"] == 0 and snap["gain_lines_total"] == 0 and snap["frames_observed"] == 1


@pytest.mark.parametrize("frame", [
    np.zeros((100, 100, 3), np.uint8), make_frame(), np.zeros((768, 1366), np.uint8),
    np.zeros((0, 0, 3), np.uint8), np.zeros((768, 1366, 4), np.uint8), "not a frame"])
def test_odd_frames_report_unavailable_instead_of_zero(frame):
    tracker = _tracker()
    for i in range(3):
        tracker.update(frame, 1.0 + i * 0.1, "hunting")
    snap = tracker.snapshot()
    assert snap["exp_text_changes"] is None and snap["gain_lines_total"] is None
    assert snap["exp_lines"] is None and snap["meso_lines"] is None and snap["other_lines"] is None
    assert snap["error"] is None
    json.dumps(snap)


def test_partly_fitting_layout_keeps_the_available_counter():
    tracker = _tracker(gain_line_rect=[1150, 604, 400, 16])  # runs off the right edge
    snap = _run(tracker, [dict(exp=n) for n in ("exp-text-4975-a", "exp-text-4990", "exp-text-4990")])
    assert snap["exp_text_changes"] == 1 and snap["gain_lines_total"] is None
    tracker = _tracker(gain_line_rect=[1150, 20, 214, 16])  # the upper slots would be above the frame
    assert _run(tracker, [dict(), dict()])["gain_lines_total"] is None
    tracker = _tracker(exp_text_rect=None, gain_line_rect="bad")
    snap = _run(tracker, [dict(), dict()])
    assert snap["exp_text_changes"] is None and snap["gain_lines_total"] is None and snap["error"] is None


def test_before_any_fresh_frame_everything_is_unavailable():
    snap = _tracker().snapshot()
    assert snap["exp_text_changes"] is None and snap["gain_lines_total"] is None
    assert snap["seconds"] == dict.fromkeys(BUCKETS, 0.0) and "lower bounds" in snap["note"]


def test_missing_templates_keep_the_total_but_not_the_kinds():
    tracker = _tracker(templates={"exp": "assets/does-not-exist.png"})
    snap = _run(tracker, _lines("lines-empty", "lines-exp"))
    assert snap["gain_lines_total"] == 1 and snap["last_gain_line_kind"] is None
    assert snap["exp_lines"] is None and snap["meso_lines"] is None and snap["other_lines"] is None


def test_internal_error_is_reported_not_raised():
    tracker = _tracker()
    tracker._observe_exp = lambda frame: 1 / 0
    tracker.update(_frame(), 1.0, "hunting")
    assert "ZeroDivisionError" in tracker.snapshot()["error"]


@pytest.mark.parametrize("state,bucket", [
    ("hunting", "fighting"), ("pursuing_target", "fighting"), ("reacquiring_target", "fighting"),
    ("looting_walking", "looting"), ("looting_waiting_visual", "looting"),
    ("circuit_walking", "traversing"), ("circuit_climbing", "traversing"),
    ("minimap_route_walking", "traversing"), ("minimap_vertical_climbing", "traversing"),
    ("circuit_blocked", "recovering"), ("circuit_return_walking", "recovering"),
    ("minimap_route_return_confirming_start", "recovering"), ("circuit_returned_to_start", "recovering"),
    ("circuit_recovering", "recovering"), ("recovering", "recovering"),
    ("health_protection_healing", "other"), ("map_unknown", "other"), ("", "other"), (None, "other")])
def test_activity_bucket(state, bucket):
    assert activity_bucket(state) == bucket


def test_time_buckets_cap_each_step_and_skip_pauses():
    tracker = _tracker()
    frame = _frame()
    timeline = [(0.0, "hunting", True), (0.1, "hunting", True), (0.3, "looting_walking", True),
                (1.8, "circuit_walking", True),            # 1.5 s step: capped at max_dt_s
                (2.0, "circuit_return_walking", True),
                (2.5, "hunting", False),                   # paused: nothing accrues, baseline dropped
                (3.0, "hunting", True),                    # first frame after the pause: no interval yet
                (3.5, "map_unknown", True),
                (9.0, "hunting", True), (9.25, "hunting", True)]  # the 5.5 s gap is not time in any state
    for now, state, fresh in timeline:
        tracker.update(frame, now, state, fresh=fresh)
    seconds = tracker.snapshot()["seconds"]
    assert seconds == {"fighting": pytest.approx(0.35), "looting": pytest.approx(0.2),
                       "traversing": pytest.approx(1.0), "recovering": pytest.approx(0.2),
                       "other": pytest.approx(0.5)}
    assert tracker.snapshot()["seconds_total"] == pytest.approx(2.25)


def test_snapshot_is_plain_json_and_summary_is_honest():
    tracker = _tracker()
    snap = _run(tracker, [dict(exp="exp-text-4975-a", lines="lines-empty"),
                          dict(exp="exp-text-4990", lines="lines-exp"),
                          dict(exp="exp-text-4990", lines="lines-exp")])
    assert json.loads(json.dumps(snap)) == snap
    line = summary_line(snap)
    assert "lower bounds" in line and "exp_text_changes=1" in line and "gain_lines=1 (exp=1 meso=0 other=0)" in line
    assert "kill" not in line and "\n" not in line
    assert summary_line(None) == "evidence: disabled"
    assert "exp_text_changes=n/a" in summary_line(_tracker().snapshot())


# -- bot wiring ----------------------------------------------------------------
class _Src:
    title = "synthetic"

    def __init__(self, frame):
        self.frame = frame

    def start(self):
        pass

    def stop(self):
        pass

    def get_frame(self):
        return None if self.frame is None else self.frame.copy()


def _bot(frame, **evidence):
    cfg = deep_merge(load_config(), {"evidence": evidence})
    backend = RecordingBackend()
    keyboard = KeyboardController(cfg, backend, sleep=lambda s: None)
    return Bot(cfg, _Src(frame), keyboard, MonsterLibrary(), None), backend


def test_bot_has_no_tracker_by_default():
    bot, _ = _bot(_frame())
    snap = bot.step(now=0.0)
    assert bot.evidence is None and bot.evidence_snapshot() is None
    assert isinstance(snap, BotSnapshot) and snap.evidence is None


def test_bot_counts_once_per_fresh_frame_and_exposes_the_snapshot():
    bot, backend = _bot(_frame("exp-text-4975-a", "lines-empty"), enabled=True)
    plain, plain_backend = _bot(_frame("exp-text-4975-a", "lines-empty"))
    commands = []
    for now, exp, lines in ((0.0, "exp-text-4975-a", "lines-empty"), (0.1, "exp-text-4990", "lines-exp"),
                            (0.2, "exp-text-4990", "lines-exp")):
        bot.source.frame = plain.source.frame = _frame(exp, lines)
        snap = bot.step(now=now)
        commands.append((snap.command, plain.step(now=now).command))
    assert snap.evidence == bot.evidence_snapshot()
    assert snap.evidence["exp_text_changes"] == 1 and snap.evidence["exp_lines"] == 1
    assert snap.evidence["frames_observed"] == 3
    assert snap.evidence["seconds_total"] == pytest.approx(0.2)
    assert all(a == b for a, b in commands)  # telemetry changes no decision
    assert vars(backend) == vars(plain_backend)


def test_bot_pause_missing_frame_and_small_frame_do_not_count():
    bot, _ = _bot(_frame("exp-text-4975-a", "lines-empty"), enabled=True)
    bot.step(now=0.0)
    bot.paused = True
    bot.source.frame = _frame("exp-text-4990", "lines-exp")
    bot.step(now=0.1)
    bot.step(now=0.2)
    bot.source.frame = None
    assert bot.step(now=0.3) is None
    bot.paused = False
    bot.source.frame = _frame("exp-text-4990", "lines-exp")
    bot.step(now=0.4)
    snap = bot.step(now=0.5)
    assert snap.evidence["exp_text_changes"] == 0 and snap.evidence["gain_lines_total"] == 0
    assert snap.evidence["seconds_total"] == pytest.approx(0.1)
    small, _ = _bot(make_frame(), enabled=True)  # 1282x693: the configured rects do not fit
    snap = small.step(now=0.0)
    assert snap.evidence["exp_text_changes"] is None and snap.evidence["gain_lines_total"] is None


# -- rules the review found unpinned ---------------------------------------------
def test_overlay_seen_twice_on_non_consecutive_frames_is_not_a_change():
    steps = [dict(exp=n) for n in ("exp-text-4975-a", "exp-text-4990", "exp-text-4975-b", "exp-text-4990",
                                   "exp-text-4975-a", "exp-text-4990", "exp-text-4975-b")]
    assert _run(_tracker(), steps)["exp_text_changes"] == 0


def test_unreadable_exp_frame_drops_the_exp_baseline():
    tracker = _tracker()
    tracker.update(_frame("exp-text-4975-a"), 1.0, "hunting")
    covered = _frame("exp-text-4975-a")
    covered[737:751, 738:845] = 255  # something covers the text for one frame
    tracker.update(covered, 1.1, "hunting")
    for now in (1.2, 1.3, 1.4):      # what changed behind the cover is unknown, not an event
        tracker.update(_frame("exp-text-4990"), now, "hunting")
    snap = tracker.snapshot()
    assert snap["exp_text_changes"] == 0 and snap["exp_text_unreadable_frames"] == 1
    tracker.update(_frame("exp-text-next"), 1.5, "hunting")
    tracker.update(_frame("exp-text-next"), 1.6, "hunting")
    assert tracker.snapshot()["exp_text_changes"] == 1  # the baseline restarted from the readable frame


def test_unavailable_line_rect_drops_the_line_baseline():
    tracker = _tracker()
    tracker.update(_frame(lines="lines-empty"), 1.0, "hunting")
    tracker.update(np.zeros((400, 600, 3), np.uint8), 1.1, "hunting")  # the rects do not fit this frame
    assert tracker.snapshot()["gain_lines_total"] is None
    tracker.update(_frame(lines="lines-exp"), 1.2, "hunting")
    tracker.update(_frame(lines="lines-exp"), 1.3, "hunting")
    snap = tracker.snapshot()
    assert snap["gain_lines_total"] == 0 and snap["exp_text_changes"] == 0


def test_internal_error_drops_every_baseline():
    tracker = _tracker()
    tracker.update(_frame("exp-text-4975-a", "lines-empty"), 1.0, "hunting")
    observe, tracker._observe_exp = tracker._observe_exp, lambda frame: 1 / 0
    tracker.update(_frame("exp-text-4975-a", "lines-empty"), 1.1, "hunting")
    tracker._observe_exp = observe
    for now in (1.2, 1.3, 1.4):
        tracker.update(_frame("exp-text-4990", "lines-exp"), now, "hunting")
    snap = tracker.snapshot()
    assert "ZeroDivisionError" in snap["error"]
    assert snap["exp_text_changes"] == 0 and snap["gain_lines_total"] == 0
    assert snap["seconds_total"] == pytest.approx(0.3)  # 1.0-1.1, then 1.2-1.4; never across the error


def _faded_frame(background, alpha):
    """The real EXP line alpha-blended over flat scenery in the newest slot (a model of the fade)."""
    from tests.test_gains import _faded
    frame = np.full((768, 1366, 3), background, np.uint8)
    frame[737:751, 738:845] = _img("exp-text-4975-a")
    line = _img("lines-exp")[-16:]
    frame[604:620, 1150:1364] = _faded(line, background, alpha)
    return frame


def test_one_fading_line_over_changing_scenery_is_counted_once():
    tracker = _tracker()
    tracker.update(np.full((768, 1366, 3), 110, np.uint8), 0.9, "hunting")
    tracker.update(_faded_frame(110, 1.0), 1.0, "hunting")  # the line arrives
    for i, (background, alpha) in enumerate([(30, .74), (110, .72), (30, .70), (110, .68), (30, .66), (110, .64)]):
        tracker.update(_faded_frame(background, alpha), 1.1 + 0.12 * i, "hunting")
    snap = tracker.snapshot()
    assert (snap["gain_lines_total"], snap["exp_lines"]) == (1, 1)


# -- keeping counts across a rebuilt bot (scripts/run_active.py) ------------------
def _seen(tracker, start=1.0):
    _run(tracker, [dict(exp="exp-text-4975-a", lines="lines-empty"), dict(exp="exp-text-4990", lines="lines-exp"),
                   dict(exp="exp-text-4990", lines="lines-exp")], start=start)
    return tracker


def test_carry_over_reuses_the_tracker_when_the_settings_are_unchanged():
    final, said = {}, []
    first = carry_over_tracker(None, _tracker(), final, said.append)
    assert final["evidence"]["frames_observed"] == 0 and not said
    record_snapshot(final, _seen(first).snapshot())
    kept = carry_over_tracker(first, _tracker(), final, said.append)
    assert kept is first and not said and "evidence_previous" not in final
    assert kept.snapshot()["exp_text_changes"] == 1 and kept.snapshot()["baseline_resets"] == 1
    # The carried tracker starts from a clean baseline: the first frame after a reload is not an event.
    snap = _run(kept, [dict(exp="exp-text-next", lines="lines-meso-over-exp")] * 3, start=1.3)
    assert snap["exp_text_changes"] == 1 and snap["gain_lines_total"] == 1


def test_carry_over_keeps_counts_made_under_different_settings():
    final, said = {}, []
    first = _seen(carry_over_tracker(None, _tracker(), final, said.append))
    changed = carry_over_tracker(first, _tracker(exp_text_min_pixels=12), final, said.append)
    assert changed is not first and final["evidence"]["frames_observed"] == 0
    assert [snap["exp_text_changes"] for snap in final["evidence_previous"]] == [1]
    assert len(said) == 1 and "settings changed" in said[0] and "exp_text_changes=1" in said[0]
    _seen(changed)
    changed.reset_baseline()
    record_snapshot(final, _seen(changed, start=5.0).snapshot())
    third = carry_over_tracker(changed, _tracker(exp_text_min_pixels=14), final, said.append)
    assert [snap["exp_text_changes"] for snap in final["evidence_previous"]] == [1, 2]
    assert len(said) == 2 and "exp_text_changes=2" in said[1]
    # A tracker that never saw a fresh frame leaves no empty entry behind.
    carry_over_tracker(third, _tracker(exp_text_min_pixels=16), final, said.append)
    assert len(final["evidence_previous"]) == 2
    lines = session_summary_lines(final)
    assert len(lines) == 3 and "exp_text_changes=n/a" in lines[0]
    assert "earlier" in lines[1] and "exp_text_changes=1" in lines[1] and "exp_text_changes=2" in lines[2]
    json.dumps(final)


def test_disabling_evidence_never_replaces_the_counts_with_none():
    final, said = {}, []
    first = _seen(carry_over_tracker(None, _tracker(), final, said.append))
    assert record_snapshot(final, first.snapshot())["exp_text_changes"] == 1
    assert carry_over_tracker(first, None, final, said.append) is None  # evidence.enabled false on reload
    assert len(said) == 1 and "disabled" in said[0] and "exp_text_changes=1" in said[0]
    assert record_snapshot(final, None) is None                         # the live status shows it is off
    assert final["evidence"]["exp_text_changes"] == 1 and "evidence_previous" not in final
    assert "exp_text_changes=1" in session_summary_lines(final)[0]
    # Enabled again later: the old counts become an earlier segment, the new tracker starts at zero.
    again = carry_over_tracker(None, _tracker(), final, said.append)
    assert again is not first and [snap["exp_text_changes"] for snap in final["evidence_previous"]] == [1]
    assert final["evidence"]["frames_observed"] == 0


def test_record_snapshot_and_summary_when_evidence_was_never_enabled():
    final = {}
    assert carry_over_tracker(None, None, final) is None
    assert record_snapshot(final, None) is None and final == {"evidence": None}
    assert session_summary_lines(final) == ["evidence: disabled"] == session_summary_lines({})


# -- bot gating through the real step, health guard and map guard -----------------
GAME_EXP_XY, GAME_LINES_XY = (738, 672), (1050, 576)  # free areas of the 1282x693 synthetic frame
BEFORE, AFTER = ("exp-text-4975-a", "lines-empty"), ("exp-text-4990", "lines-exp")


def _game(monkeypatch):
    from tests.test_health_protection_integration import _Game
    game = _Game(monkeypatch)
    cfg = dict(load_config()["evidence"], exp_text_rect=[*GAME_EXP_XY, 107, 14],
               gain_line_rect=[GAME_LINES_XY[0], GAME_LINES_XY[1] + 28, 214, 16])
    game.bot.evidence = EvidenceTracker(cfg)
    return game


def _game_step(game, exp, lines, hp=75.0, dt=0.1, player=True):
    from maplebot.capture.window import FrameSample
    from tests.test_bot import _vertical_frame
    game.wall += dt
    game.hp = hp
    game.sequence += 1
    frame = _vertical_frame((65, 40), player=player)
    for name, (x, y) in ((exp, GAME_EXP_XY), (lines, GAME_LINES_XY)):
        crop = _img(name)
        frame[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    game.sample = FrameSample(frame, game.sequence, game.wall)
    game.bot.step(game.wall)
    game.bot.keyboard.apply_once(game.wall)
    return game.bot.evidence_snapshot()


def _farming(game, steps=3):
    for _ in range(steps):
        snap = _game_step(game, *BEFORE)
    assert not game.bot.paused and game.bot.brain.state == "circuit_walking"
    assert snap["exp_text_changes"] == 0 and snap["gain_lines_total"] == 0 and snap["baseline_resets"] == 0
    assert snap["seconds"]["traversing"] == pytest.approx(0.1 * (steps - 1)) == snap["seconds_total"]
    return snap


def _nothing_counted(snap, seconds, resets=1):
    assert snap["exp_text_changes"] == 0 and snap["gain_lines_total"] == 0
    assert snap["seconds_total"] == pytest.approx(seconds) and snap["baseline_resets"] == resets


def test_bot_counts_through_the_guarded_step(monkeypatch):
    game = _game(monkeypatch)  # the control for the gating tests below: the same change does count
    _farming(game)
    for _ in range(3):
        snap = _game_step(game, *AFTER)
    assert snap["exp_text_changes"] == 1 and snap["gain_lines_total"] == 1 and snap["baseline_resets"] == 0
    assert snap["seconds_total"] == pytest.approx(0.5)


def test_bot_health_hold_frames_neither_count_nor_keep_the_baseline(monkeypatch):
    game = _game(monkeypatch)
    _farming(game)
    for hp in (55.0, 55.0):  # low HP: farming is held while the guard heals
        snap = _game_step(game, *AFTER, hp=hp)
        assert game.bot.paused and game.bot.brain.state.startswith("health_protection_")
        _nothing_counted(snap, 0.2)
    for i in range(3):       # healed 0.3 s after the last farming frame: what changed meanwhile is unknown
        snap = _game_step(game, *AFTER)
        assert not game.bot.paused
        _nothing_counted(snap, 0.2 + 0.1 * i)


def test_bot_short_unrecognised_map_interval_resets_the_baseline(monkeypatch):
    game = _game(monkeypatch)
    _farming(game)
    game.map_state = "unknown"
    _nothing_counted(_game_step(game, *AFTER), 0.2)
    assert game.bot.brain.state == "map_unknown"
    game.map_state = "matched"
    for i in range(3):       # recognised again after 0.2 s, well inside max_gap_s
        snap = _game_step(game, *AFTER)
        assert game.bot.map_ready and not game.bot.paused
        _nothing_counted(snap, 0.2 + 0.1 * i)


def test_bot_short_missing_frame_interval_resets_the_baseline(monkeypatch):
    game = _game(monkeypatch)
    _farming(game)
    game.wall += 0.1
    game.sample = None       # the capture went stale for one step; the bot is not paused
    assert game.bot.step(game.wall) is None and game.bot.capture_state == "stale_or_missing"
    _nothing_counted(game.bot.evidence_snapshot(), 0.2)
    for i in range(3):
        snap = _game_step(game, *AFTER)
        assert not game.bot.paused
        _nothing_counted(snap, 0.2 + 0.1 * i)


@pytest.mark.parametrize("attribute,value", [("capture_state", "stale_or_missing"), ("map_ready", False)])
def test_bot_final_evidence_call_requires_fresh_capture_and_ready_map(monkeypatch, attribute, value):
    game = _game(monkeypatch)
    _farming(game)
    bot = game.bot

    def lose_context(frame):  # the context is lost while this frame is being processed
        setattr(bot, attribute, value)
        return [], None

    bot._detect = lose_context
    for _ in range(3):
        snap = _game_step(game, *AFTER)
        assert getattr(bot, attribute) == value and not bot.paused
        _nothing_counted(snap, 0.2)


def test_bot_unfocused_game_credits_no_activity_but_keeps_observing(monkeypatch):
    game = _game(monkeypatch)
    _farming(game)
    game.focused = False     # alt-tabbed: capture stays fresh, the map stays recognised, no key can be sent
    _game_step(game, *BEFORE)
    for _ in range(3):
        snap = _game_step(game, *AFTER)
    assert not game.bot.paused and game.bot.brain.state == "circuit_walking"  # the suspended brain's stale state
    assert not game.bot.keyboard.enabled and not game.backend.held
    assert snap["seconds"]["traversing"] == pytest.approx(0.2)                # nothing added after focus was lost
    assert snap["seconds"]["other"] == pytest.approx(0.4) and snap["seconds_total"] == pytest.approx(0.6)
    assert snap["exp_text_changes"] == 1 and snap["gain_lines_total"] == 1    # pixels are still observations
    game.focused = True
    for _ in range(2):
        snap = _game_step(game, *AFTER)
    assert snap["seconds"]["traversing"] == pytest.approx(0.4) and snap["seconds"]["other"] == pytest.approx(0.4)


def test_bot_minimap_route_movement_without_a_player_fix_still_counts_as_traversing(monkeypatch):
    game = _game(monkeypatch)
    _farming(game)
    for _ in range(3):       # the sprite is hidden; the drawn route continues on the minimap dot
        snap = _game_step(game, *BEFORE, player=False)
    bot = game.bot
    assert bot._minimap_route_active and bot._input_blocked and bot.brain.state.startswith("minimap_route_")
    assert snap["seconds"]["traversing"] == pytest.approx(0.5) == snap["seconds_total"]


def test_bot_vertical_minimap_movement_still_counts_as_traversing(monkeypatch):
    from tests.test_bot import _started_vertical_bot
    bot, _ = _started_vertical_bot(monkeypatch)
    bot.evidence = EvidenceTracker(load_config()["evidence"])
    bot.step(.4)
    bot.step(.5)
    assert bot._vertical_minimap_active and bot._input_blocked
    assert bot.brain.state.startswith(("circuit_", "minimap_vertical_"))
    assert bot.evidence_snapshot()["seconds"] == dict.fromkeys(BUCKETS, 0.0) | {"traversing": pytest.approx(0.1)}
