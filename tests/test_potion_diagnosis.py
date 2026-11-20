"""Diagnostics for HP/MP potion failures (live run 7): observation only, no control change.

Run 7 latched "HP did not improve after confirmed potion presses" after the HP item
stack ran out: three sent presses left HP at exactly 63.5 on fresh frames. The MP
item slot was empty from the start while the MP gate kept pressing. These tests
pin the diagnosis wording and the response counter without changing the latch.
"""
from collections import Counter

import pytest

import maplebot.gui as gui
from maplebot.config import deep_merge, load_config
from maplebot.control.backend import RecordingBackend
from maplebot.control.keyboard import KeyboardController
import maplebot.health as health
from maplebot.health import HealthPolicy
from maplebot.health_protection import HealthProtection
from maplebot.types import Command

REASON = "HP did not improve after confirmed potion presses"
RUN7 = {"hp_percent": 70, "minimum_hp_percent": 40, "rearm_delta": 5}


def _guard():
    guard = HealthProtection(RUN7)
    assert guard.target == 75
    # Prime the cumulative counter at 31 earlier taps, above threshold.
    assert not guard.observe(73.1, 0.0, taps=31, tap_time=None)
    return guard


def _run_three_flat_or_falling(guard, readings):
    """The run 7 cadence: request, tap seen, 1 s window, retry; returns observe() results."""
    out = []
    taps, t = 31, 0.1
    out.append(guard.observe(readings[0], t, taps=taps, tap_time=None))
    for i in range(3):
        taps += 1
        sent = t + .1
        t = sent + .1
        out.append(guard.observe(readings[1 + 2 * i], t, taps=taps, tap_time=sent))
        t = sent + 1.05
        out.append(guard.observe(readings[2 + 2 * i], t, taps=taps, tap_time=sent))
    return out


def test_constant_hp_across_three_sent_presses_is_no_change_with_unchanged_reason():
    guard = _guard()
    results = _run_three_flat_or_falling(guard, [63.5] * 7)
    # Control timeline is the pre-existing one: request, wait, retry, wait, retry, wait, latch.
    assert results == [True, False, True, False, True, False, False]
    assert guard.latched and guard.state == "needs_attention" and guard.reason == REASON
    assert guard.taps_sent == 3 and guard.unconfirmed_taps == 3
    assert guard.diagnosis == "no_change"
    assert guard.diagnosis_detail == (
        "HP stayed at 64% with no new damage during 3 sent presses; the HP item slot may be "
        "empty or the key unbound. Restock or check the hotkey, then Resume.")
    assert "out of potions" not in guard.diagnosis_detail.lower()
    assert guard.responses_confirmed == 0
    assert guard.response_window == (63.5, 63.5)


def test_falling_hp_across_three_sent_presses_is_falling_with_unchanged_reason():
    guard = _guard()
    results = _run_three_flat_or_falling(guard, [63.5, 63.0, 61.0, 59.5, 58.0, 56.0, 55.0])
    assert results == [True, False, True, False, True, False, False]
    assert guard.latched and guard.reason == REASON and guard.diagnosis == "falling"
    detail = guard.diagnosis_detail
    assert "3 sent presses" in detail and "now 55%" in detail
    assert "may be empty or the key unbound" in detail and "out of potions" not in detail.lower()
    assert "no new damage" not in detail
    assert guard.responses_confirmed == 0


def test_direct_rise_to_target_after_one_sent_press_counts_exactly_one_response():
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
    assert not guard.observe(63.5, 0.3, taps=32, tap_time=0.2)
    assert not guard.observe(86.5, 0.5, taps=32, tap_time=0.2)
    assert guard.state == "monitoring" and not guard.latched
    assert guard.responses_confirmed == 1
    assert not guard.observe(86.5, 0.7, taps=32, tap_time=0.2)
    assert guard.responses_confirmed == 1


def test_rise_seen_in_the_same_frame_as_the_tap_counts_one_response():
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
    assert not guard.observe(86.5, 0.3, taps=32, tap_time=0.2)
    assert guard.responses_confirmed == 1 and guard.state == "monitoring"


def test_stepwise_rise_after_one_sent_press_counts_one_response():
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
    assert not guard.observe(63.5, 0.3, taps=32, tap_time=0.2)
    for t, hp in ((0.5, 66.0), (0.7, 70.0), (0.9, 76.0)):
        guard.observe(hp, t, taps=32, tap_time=0.2)
    assert guard.state == "monitoring" and guard.responses_confirmed == 1
    assert guard.responses_confirmed <= guard.taps_sent


@pytest.mark.parametrize("rise_hp", [80.0, 66.0])  # reset branch, and the baseline+2 branch
def test_rise_in_a_frame_captured_before_the_tap_is_not_credited(rise_hp):
    # Review F1: the frame (step start now=0.3) predates the tap (tap_time=0.31), so the
    # guard stores latest_tap_time=None; a rise in that frame cannot be that tap's effect.
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
    assert not guard.observe(rise_hp, 0.3, taps=32, tap_time=0.31)
    assert guard.latest_tap_time is None
    assert guard.responses_confirmed == 0


def test_tap_after_the_frame_leaves_control_unchanged():
    # The F1 fix is diagnostic only: control state matches a frame where the tap came first.
    late, early = _guard(), _guard()
    for guard, tap_time in ((late, 0.31), (early, 0.2)):
        assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
        assert not guard.observe(66.0, 0.3, taps=32, tap_time=tap_time)
    for attr in ("state", "reason", "latched", "unconfirmed_taps", "taps_sent"):
        assert getattr(late, attr) == getattr(early, attr)
    assert (late.responses_confirmed, early.responses_confirmed) == (0, 1)


def test_weak_potion_that_raises_hp_every_press_is_small_rise_not_falling():
    # Review F2: +1.5 per press never reaches baseline + 2, HP only rises 60 -> 64.5.
    guard = HealthProtection({**RUN7, "protection_retry_s": 1.0})
    guard.observe(73.0, 0.0, taps=0, tap_time=None)
    t, hp, taps = 0.1, 60.0, 0
    results = [guard.observe(hp, t, taps=taps, tap_time=None)]
    for _ in range(3):
        taps += 1
        sent = t + .05
        t = sent + .1
        results.append(guard.observe(hp, t, taps=taps, tap_time=sent))
        hp += 1.5
        t = sent + 1.05
        results.append(guard.observe(hp, t, taps=taps, tap_time=sent))
    assert results == [True, False, True, False, True, False, False]
    assert guard.latched and guard.reason == REASON
    assert guard.diagnosis == "small_rise"
    detail = guard.diagnosis_detail
    assert detail.startswith("HP rose from 60% to 64% during 3 sent presses")
    assert "damage may be outpacing" not in detail and "out of potions" not in detail.lower()


def test_dip_that_returns_near_its_start_is_mixed():
    guard = _guard()
    # Each re-request moves the baseline, so no reading reaches baseline + 2.
    _run_three_flat_or_falling(guard, [63.5, 63.0, 58.0, 59.0, 59.9, 61.5, 61.8])
    assert guard.latched and guard.reason == REASON and guard.diagnosis == "mixed"
    assert "moved between 58% and 63% and is now 62%" in guard.diagnosis_detail


def test_regeneration_without_a_sent_press_counts_no_response():
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)  # request; never sent
    for t, hp in ((0.3, 66.0), (0.5, 70.0), (0.7, 76.0)):
        guard.observe(hp, t, taps=31, tap_time=None)
    assert guard.state == "monitoring" and guard.taps_sent == 0
    assert guard.responses_confirmed == 0


def test_two_sent_presses_answered_by_one_rise_count_one_response():
    # Run 7 taps 13 and 14: two sends, one rise.
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
    assert not guard.observe(63.5, 0.3, taps=32, tap_time=0.2)
    assert guard.observe(63.5, 1.25, taps=32, tap_time=0.2)
    assert not guard.observe(63.5, 1.4, taps=33, tap_time=1.3)
    assert not guard.observe(86.5, 1.6, taps=33, tap_time=1.3)
    assert guard.taps_sent == 2 and guard.responses_confirmed == 1


def test_diagnosis_clears_after_rearm_through_request_resume():
    guard = _guard()
    _run_three_flat_or_falling(guard, [63.5] * 7)
    assert guard.diagnosis == "no_change"
    assert not guard.observe(80.0, 5.0, taps=34, tap_time=2.7)  # no rearm requested
    assert guard.latched and guard.diagnosis == "no_change"
    guard.request_resume()
    assert not guard.observe(80.0, 5.2, taps=34, tap_time=2.7)
    assert not guard.latched and guard.reason is None
    assert guard.diagnosis is None and guard.diagnosis_detail is None
    assert guard.response_window is None
    assert guard.responses_confirmed == 0


def test_diagnosis_clears_after_episode_reset_and_is_not_carried_to_a_new_run():
    guard = _guard()
    _run_three_flat_or_falling(guard, [63.5] * 7)
    guard.request_resume()
    guard.observe(80.0, 5.0, taps=34, tap_time=2.7)
    # A later successful episode leaves no stale diagnosis behind.
    assert guard.observe(65.0, 6.0, taps=34, tap_time=2.7)
    assert not guard.observe(65.0, 6.2, taps=35, tap_time=6.1)
    assert guard.response_window == (65.0, 65.0)
    assert not guard.observe(80.0, 6.4, taps=35, tap_time=6.1)
    assert guard.diagnosis is None and guard.response_window is None
    assert guard.responses_confirmed == 1


@pytest.mark.parametrize("hp,taps,reason", [(0, 34, "HP is zero"), (63.5, None, "HP key counter is unavailable")])
def test_every_other_latch_clears_a_previous_diagnosis(hp, taps, reason):
    guard = _guard()
    _run_three_flat_or_falling(guard, [63.5] * 7)
    assert guard.diagnosis == "no_change"
    guard.observe(hp, 5.0, taps=taps, tap_time=None)
    assert guard.reason == reason and guard.diagnosis is None and guard.diagnosis_detail is None


def test_not_sent_latch_has_no_diagnosis():
    guard = HealthProtection({"protection_retry_s": .5})
    guard.observe(40, 0, taps=0, tap_time=None)
    for t in (.5, 1, 2, 3):
        guard.observe(40, t, taps=0, tap_time=None)
    assert guard.reason == "HP key was not sent" and guard.diagnosis is None


def test_single_tap_budget_uses_singular_wording():
    guard = HealthProtection({**RUN7, "protection_max_taps": 1})
    guard.observe(73.1, 0.0, taps=0, tap_time=None)
    assert guard.observe(63.5, 0.1, taps=0, tap_time=None)
    guard.observe(63.5, 0.3, taps=1, tap_time=0.2)
    guard.observe(63.5, 1.3, taps=1, tap_time=0.2)
    assert guard.reason == REASON and "during 1 sent press;" in guard.diagnosis_detail


# -- GUI -------------------------------------------------------------------

def test_gui_needs_attention_appends_diagnosis_and_response_ratio():
    label, detail = gui._health_protection_display({
        "health_protection_state": "needs_attention", "health_protection_active": False,
        "health_protection_reason": REASON, "health_protection_taps": 34,
        "health_protection_responses": 30,
        "health_protection_diagnosis": "no_change",
        "health_protection_diagnosis_detail": "HP stayed at 64% with no new damage during 3 sent "
                                              "presses; the HP item slot may be empty or the key unbound."})
    assert label == "Health needs attention"
    assert detail.startswith(REASON)
    assert "HP stayed at 64% with no new damage during 3 sent presses" in detail
    assert "30 HP rises were observed after 34 sent presses." in detail
    assert "does not confirm HP restoration" in detail


def test_gui_does_not_show_a_stale_diagnosis_outside_needs_attention():
    _, detail = gui._health_protection_display({
        "health_protection_state": "healing", "health_protection_active": True,
        "health_protection_taps": 2, "health_protection_responses": 1,
        "health_protection_diagnosis_detail": "HP stayed at 64% with no new damage"})
    assert "no new damage" not in detail
    assert "1 HP rise was observed after 2 sent presses." in detail
    assert "does not confirm HP restoration" in detail


def test_gui_omits_ratio_without_a_response_count():
    _, detail = gui._health_protection_display({
        "health_protection_state": "needs_attention", "health_protection_reason": REASON,
        "health_protection_taps": 3})
    assert "observed after" not in detail and "does not confirm HP restoration" in detail


def test_gui_rise_count_is_not_worded_as_answered_presses():
    # Review F4: run 7 taps 13/14 were both followed by one rise; the guard counts one
    # rise, so the text must not say only 1 of the 2 presses was followed by a rise.
    guard = _guard()
    assert guard.observe(63.5, 0.1, taps=31, tap_time=None)
    assert not guard.observe(63.5, 0.3, taps=32, tap_time=0.2)
    assert guard.observe(63.5, 1.25, taps=32, tap_time=0.2)
    assert not guard.observe(63.5, 1.4, taps=33, tap_time=1.3)
    assert not guard.observe(86.5, 1.6, taps=33, tap_time=1.3)
    _, detail = gui._health_protection_display({
        "health_protection_state": "healing", "health_protection_active": True,
        "health_protection_taps": guard.taps_sent,
        "health_protection_responses": guard.responses_confirmed})
    assert "of 2 sent presses were followed" not in detail
    assert "1 HP rise was observed after 2 sent presses." in detail


def test_gui_hides_a_zero_rise_count_while_the_first_window_is_open():
    _, detail = gui._health_protection_display({
        "health_protection_state": "awaiting_response", "health_protection_active": True,
        "health_protection_taps": 1, "health_protection_responses": 0})
    assert "observed after" not in detail and "does not confirm HP restoration" in detail
    _, latched = gui._health_protection_display({
        "health_protection_state": "needs_attention", "health_protection_reason": REASON,
        "health_protection_taps": 3, "health_protection_responses": 0})
    assert "0 HP rises were observed after 3 sent presses." in latched


# -- MP observation ----------------------------------------------------------

def test_mp_observation_needs_a_sent_counter():
    policy = HealthPolicy({"warmup_reads": 0})
    policy.update(80, 2, 0.0)
    assert policy.mp_presses_unanswered is None


def test_mp_observation_counts_sent_presses_without_a_rise_and_ignores_regen():
    obs = health.MpPressObservation()
    obs.observe(16.5, 0.0, 0)
    obs.observe(16.5, 1.0, 1)        # first press sent
    obs.observe(30.1, 2.0, 1)        # answered within 3 s
    assert obs.presses_unanswered == 0 and obs.presses_answered == 1
    obs.observe(4.0, 10.0, 1)
    for i, t in enumerate((40.0, 70.0, 100.0), start=2):
        obs.observe(4.0, t - 0.5, i - 1)  # a fresh reading just before the press
        obs.observe(4.0, t, i)
        obs.observe(4.5, t + 3.0, i)   # no rise of 2 within the window
    assert obs.presses_unanswered == 3 and obs.presses_sent == 4
    obs.observe(7.0, 110.0, 4)        # regeneration with no pending press
    assert obs.presses_unanswered == 3
    obs.observe(7.0, 129.5, 4)
    obs.observe(7.0, 130.0, 5)
    obs.observe(7.5, 131.0, 5)        # still pending: not yet counted
    assert obs.presses_unanswered == 3
    obs.observe(7.5, 133.5, 5)
    assert obs.presses_unanswered == 4 and obs.presses_unknown == 0


def _five_unanswered():
    obs = health.MpPressObservation()
    obs.observe(4.0, 0.0, 0)
    for i in range(1, 6):
        t = i * 30.0
        obs.observe(4.0, t - 0.3, i - 1)
        obs.observe(4.0, t, i)
        obs.observe(4.0, t + 3.0, i)
    assert obs.presses_unanswered == 5 and obs.presses_unknown == 0
    return obs


def test_mp_press_first_seen_after_an_update_gap_is_unknown_not_answered():
    # Review F3: a press sent before a hold/pause is first seen 25 s later, after two
    # 1.9-point regen ticks (run 7's tick size); the pre-gap baseline is stale.
    obs = _five_unanswered()
    obs.observe(4.0 + 3.8, 205.0, 6)
    assert obs.presses_answered == 0 and obs.presses_unanswered == 5
    assert obs.presses_unknown == 1 and obs.presses_sent == 6


def test_mp_press_pending_across_an_update_gap_is_unknown_not_answered():
    obs = _five_unanswered()
    obs.observe(4.0, 179.7, 5)
    obs.observe(4.0, 180.0, 6)        # press seen with a fresh reading
    obs.observe(4.0 + 3.8, 200.0, 6)  # the observer stopped; regen ticked meanwhile
    assert obs.presses_answered == 0 and obs.presses_unanswered == 5
    assert obs.presses_unknown == 1


def test_mp_rise_read_after_the_window_closed_does_not_answer_it():
    obs = _five_unanswered()
    obs.observe(4.0, 179.7, 5)
    obs.observe(4.0, 180.0, 6)
    obs.observe(4.0, 180.5, 6)        # in the window, no rise
    obs.observe(4.0 + 2.5, 183.4, 6)  # gap 2.9 s, but the 3 s window closed at 183.0
    assert obs.presses_answered == 0 and obs.presses_unanswered == 6
    assert obs.presses_unknown == 0


def _counted_keyboard():
    cfg = deep_merge(load_config(), {"keys": {"mp_potion": "a"}})
    wall = [10.0]
    backend = RecordingBackend()
    counts = Counter()

    class Counted:
        def down(self, key):
            backend.down(key)
            counts[key] += 1

        def up(self, key):
            backend.up(key)

    keyboard = KeyboardController(cfg, Counted(), clock=lambda: wall[0], sleep=lambda _: None)
    return keyboard, counts, wall


def test_mp_observation_ignores_requests_the_keyboard_dropped():
    keyboard, counts, wall = _counted_keyboard()
    policy = HealthPolicy({"warmup_reads": 0, "mp_percent": 30, "retry_s": 1, "max_retries": 5})
    policy.mp_sent_counter = lambda: counts["a"]
    keyboard.set_hp_only(source_time=wall[0])  # HP-only drops every MP request
    for t in (0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0):
        wall[0] = 10.0 + t
        keyboard.set_hp_only(source_time=wall[0])
        for req in policy.update(80, 5, t):
            keyboard.request(req)
        keyboard.apply_once(t)
    assert counts["a"] == 0 and policy.mp_presses.presses_sent == 0
    assert policy.mp_presses_unanswered == 0


def test_mp_observation_counts_presses_the_keyboard_actually_sent():
    keyboard, counts, wall = _counted_keyboard()
    policy = HealthPolicy({"warmup_reads": 0, "mp_percent": 30, "retry_s": 1, "max_retries": 5})
    policy.mp_sent_counter = lambda: counts["a"]
    # Fresh readings every 0.5 s, as while input is ready; the gate backs off after
    # five retries, so every pending window closes before the loop ends.
    for t in [i * 0.5 for i in range(25)]:
        wall[0] = 10.0 + t
        keyboard.set_command(Command.NONE, source_time=wall[0])
        for req in policy.update(80, 5, t):
            keyboard.request(req)
        keyboard.apply_once(t)
    sent = counts["a"]
    assert sent >= 3
    assert policy.mp_presses.presses_sent == sent
    assert policy.mp_presses_unanswered == sent


def test_mp_observation_has_no_effect_on_potion_decisions():
    readings = [(80, 50), (80, 5), (80, 5), (55, 5), (40, 5), (40, 30), (80, 2), (80, 2)]
    plain = HealthPolicy({"warmup_reads": 0, "retry_s": 1})
    observed = HealthPolicy({"warmup_reads": 0, "retry_s": 1})
    sent = [0]
    observed.mp_sent_counter = lambda: sent[0]
    flaky = HealthPolicy({"warmup_reads": 0, "retry_s": 1})
    flaky.mp_sent_counter = lambda: 1 / 0
    for i, (hp, mp) in enumerate(readings):
        t = i * 1.5
        expected = plain.update(hp, mp, t)
        assert observed.update(hp, mp, t) == expected
        assert flaky.update(hp, mp, t) == expected
        sent[0] += len(expected)
