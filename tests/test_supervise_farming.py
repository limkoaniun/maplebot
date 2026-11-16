"""Offline tests of the session supervisor (scripts/supervise_farming.py).

Part 1 classifies recorded controller statuses, part 2 and 3 test the pure decision code one condition at a time,
part 4 runs main() on two fake clocks against a fake controller that publishes and reads the request files the way
scripts/run_active.py does, part 5 covers the arguments and the rules that can be checked statically.
Nothing sleeps, nothing can hang (every loop is bounded by the fake clock) and nothing leaves tmp_path.
"""
import ast
import builtins
import inspect
import io
import json
import math
import os
import pathlib
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import supervise_farming as supervisor

# ---------------------------------------------------------------------------------------------------------------
# Recorded statuses (fixtures/supervisor/README.md names the trace and line of each) and the statuses that
# maplebot/bot.py and maplebot/recovery_runtime.py derive from them.
# ---------------------------------------------------------------------------------------------------------------
RECORDED = json.loads((Path(__file__).resolve().parent / "fixtures" / "supervisor" / "recorded_statuses.json")
                      .read_text(encoding="utf-8"))
USER_HOLD = "Farming is on hold; HP protection remains on."               # bot.hold_farming(), default reason
CHECKING = "Checking HP, map, and position before returning to point 1."  # StartupReturn.request()


def recorded(name, **changes):
    """A copy of a recorded status with the given keys replaced."""
    return dict(RECORDED[name]["status"], **changes)


def full_pause(status, **changes):
    """bot.pause_all() applied to a status: Pause all, F1, a map hard stop. The circuit fields stay as they are."""
    return {**status, "paused": True, "full_pause": True, "manual_health_hold": False, "health_hold_reason": None,
            "health_protection_active": False, "health_protection_state": "suspended", **changes}


def user_hold(status, **changes):
    """bot.hold_farming() applied to a status: an accepted Hold farming request."""
    return {**status, "paused": True, "full_pause": False, "manual_health_hold": True,
            "health_hold_reason": USER_HOLD, **changes}


def failed_start(reason):
    """StartupReturn._fail(reason): the recorded failed start with another reason in the four places it appears."""
    return recorded("failed_start", health_hold_reason="Start stopped: " + reason, startup_return_reason=reason,
                    recovery_reason=reason, circuit_reason="Return failed: " + reason)


ROUTE_HOLD, FARMING, RETURNING = recorded("route_hold"), recorded("farming"), recorded("returning")
# A route blocked with return_on_block off, as the real Bot publishes it (paused by the health guard, not terminal).
BLOCKED_WITHOUT_RETURN = recorded("route_hold", recovery_terminal=False, recovery_state="idle", recovery_reason=None,
                                  circuit_reason="waypoint timeout")
# A start cancelled by Hold farming after a route failure: StartupReturn.cancel() puts the failed circuit back.
HOLD_DURING_START = user_hold(ROUTE_HOLD, startup_return_state="cancelled",
                              startup_return_reason="Start cancelled by Hold farming.")
TRANSIENT = ("waypoint timeout", "jump timeout", "ramp timeout", "drop timeout", "drop stalled", "Return route timed out",
             "Stable source unavailable: position kept moving", "left ramp corridor", "ramp knockback did not settle",
             "drop did not leave source floor")
NEEDS_USER = (RECORDED["failed_start"]["status"]["startup_return_reason"],  # "Could not reach a drawn platform ..."
              "HP is below 40%", "HP is unreadable", "HP did not improve after confirmed potion presses",
              "HP key was not sent", "HP protection needs attention.",
              "Timed out waiting for a fresh, healthy starting position.", "Enable a saved circuit before starting.",
              "Enable HP protection and configure the HP key before starting.",
              "A normal saved circuit is required before starting.", "Return needs a saved editor plan",
              "Invalid return map: connection 3 refers to a missing platform", "Return to point 1 failed.",
              "Return route blocked", "passed destination floor")
BLOCKING = ("pause-active", "stop-active", "hold-farming-active", "return-to-start-active", "reload-active",
            "resume-active", "editor-test-request.json", "stop-editor-test")


# ---------------------------------------------------------------------------------------------------------------
# 1. classify_hold
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name, kind", [
    ("waiting", "user"), ("checking", "starting"), ("returning", "starting"), ("farming", "farming"),
    ("farming_background", "farming"), ("farming_map_lost", "farming"), ("hp_hold", "healing"),
    ("hp_hold_waiting", "healing"), ("automatic_return", "starting"), ("hp_hold_during_return", "starting"),
    ("route_hold", "route"), ("failed_start", "start_failed"), ("watcher_hold", "user"),
    ("full_pause_no_snapshot", "user")])
def test_every_recorded_status_has_its_kind(name, kind):
    assert supervisor.classify_hold(recorded(name))[0] == kind


def test_details_name_the_route_failure_the_failed_start_and_the_lost_map():
    assert supervisor.classify_hold(ROUTE_HOLD) == ("route", "Return failed: waypoint timeout")
    assert supervisor.classify_hold(recorded("route_hold", circuit_reason=None)) == ("route", "waypoint timeout")
    kind, detail = supervisor.classify_hold(recorded("failed_start"))
    assert detail.startswith("Could not reach a drawn platform from (96,96)") and "Start stopped" not in detail
    kind, detail = supervisor.classify_hold(recorded("full_pause_no_snapshot"))
    assert detail == ("full pause (user, or the controller after losing the map or capture): "
                      "Saved map alignment was not confirmed within 2 seconds.")
    assert supervisor.classify_hold(recorded("waiting"))[1] == (
        "full pause (user, or the controller after losing the map or capture)")


@pytest.mark.parametrize("status", [
    {"running": False, "frames": 19180, "keys_sent": {"left": 178}, "hp": 79.8, "mp": 7.7, "exp": 52.6, "pid": 18480,
     "updated": 1794758100.0},                                    # the row scripts/run_active.py publishes at its end
    {"running": False, "error": "Could not load recognized map: x", "pid": 18480, "updated": 1794758100.0},
    recorded("route_hold", running=None), {key: value for key, value in ROUTE_HOLD.items() if key != "running"}])
def test_a_controller_that_does_not_say_it_runs_is_stopped(status):
    assert supervisor.classify_hold(status)[0] == "stopped"


@pytest.mark.parametrize("base", ["automatic_return", "hp_hold_during_return", "returning", "checking", "route_hold",
                                  "failed_start", "farming", "hp_hold", "watcher_hold"])
def test_a_full_pause_is_a_pause_whatever_else_the_status_says(base):
    """pause_all() does not end the controller's own return, so returning_to_start stays true under a pause."""
    assert supervisor.classify_hold(full_pause(recorded(base)))[0] == "user"


@pytest.mark.parametrize("value", [None, "no", 0, 1])
def test_only_an_explicit_false_means_no_full_pause(value):
    assert supervisor.classify_hold(recorded("route_hold", full_pause=value))[0] == "user"
    assert supervisor.classify_hold({k: v for k, v in ROUTE_HOLD.items() if k != "full_pause"})[0] == "user"


def test_a_connection_test_is_not_farming():
    """scripts/run_active.py publishes editor_test instead of startup_return_state while a test moves the character."""
    moving = {key: value for key, value in FARMING.items() if not key.startswith("startup_return_")}
    assert supervisor.classify_hold(moving)[0] == "farming"
    assert supervisor.classify_hold(dict(moving, editor_test={"test_id": "x", "state": "running"}))[0] == "user"


@pytest.mark.parametrize("status, kind", [
    (recorded("checking"), "starting"),                                       # the start's own hold
    (recorded("checking", startup_return_state="idle"), "user"),              # the same hold without a start
    (recorded("failed_start"), "start_failed"),
    (recorded("failed_start", health_hold_reason=USER_HOLD), "user"),         # Pause all, then Hold, after a failed start
    (recorded("failed_start", health_hold_reason=None), "user"),
    (recorded("failed_start", startup_return_state="cancelled"), "user"),     # not a failed start any more
    (recorded("failed_start", startup_return_state="completed"), "user"),
    (recorded("watcher_hold"), "user"),                                       # Hold farming during farming
    (user_hold(recorded("automatic_return")), "user"),                        # ... during the controller's own return
    (user_hold(recorded("hp_hold_during_return")), "user"),
    (HOLD_DURING_START, "user"),                                              # ... during a start after a route failure
    (user_hold(ROUTE_HOLD), "user"),                                          # Pause all, then Hold, after a route failure
    (user_hold(recorded("returning")), "user"),
])
def test_a_manual_hold_is_the_users_unless_it_is_the_starts_own_or_a_failed_start(status, kind):
    assert supervisor.classify_hold(status)[0] == kind


@pytest.mark.parametrize("status, kind", [
    (recorded("route_hold", startup_return_state="checking"), "starting"),
    (recorded("route_hold", startup_return_state="returning"), "starting"),
    (recorded("route_hold", returning_to_start=True), "starting"),
    (recorded("hp_hold", returning_to_start=True), "starting"),
    (recorded("route_hold", paused=False), "farming"),
    (recorded("route_hold", paused=None), "farming"),                         # not said to be paused: it only waits
    (ROUTE_HOLD, "route"),
    (recorded("route_hold", recovery_terminal=None), "unsupported"),          # only an explicit true is a route failure
    (recorded("route_hold", recovery_terminal=1), "unsupported"),
    (recorded("route_hold", recovery_terminal="true"), "unsupported"),
    (BLOCKED_WITHOUT_RETURN, "unsupported"),
    (dict(BLOCKED_WITHOUT_RETURN, circuit_state="walking"), "healing"),
    (recorded("hp_hold"), "healing"),
    (recorded("route_hold", manual_health_hold=None), "user"),
    (recorded("hp_hold", manual_health_hold=None), "user"),
    (recorded("route_hold", manual_health_hold="yes"), "user"),
])
def test_one_field_at_a_time_decides_between_the_remaining_kinds(status, kind):
    assert supervisor.classify_hold(status)[0] == kind


def test_start_visible_is_what_the_design_calls_consumed():
    assert supervisor.start_visible(recorded("checking")) and supervisor.start_visible(RETURNING)
    assert supervisor.start_visible(recorded("automatic_return"))
    assert supervisor.start_visible(recorded("route_hold", startup_return_state="returning"))
    for status in (ROUTE_HOLD, FARMING, recorded("failed_start"), full_pause(ROUTE_HOLD), HOLD_DURING_START,
                   recorded("route_hold", returning_to_start=None)):
        assert not supervisor.start_visible(status)


# ---------------------------------------------------------------------------------------------------------------
# 2. blocker: what makes one status of a hold ineligible
# ---------------------------------------------------------------------------------------------------------------
def blocked(status=ROUTE_HOLD, *, age_s=0.2, min_hp=75.0, pending=()):
    return supervisor.blocker(status, age_s=age_s, min_hp=min_hp, pending=pending)


def test_the_recorded_route_hold_is_fully_eligible():
    assert blocked() is None
    assert blocked(age_s=1.0) is None and blocked(age_s=-1.0) is None and blocked(age_s=0.0) is None
    assert blocked(recorded("route_hold", hp=75.0)) is None and blocked(recorded("route_hold", hp=100)) is None
    assert blocked(recorded("route_hold", capture_age_s=1.0)) is None
    assert blocked(recorded("route_hold", capture_age_s=0)) is None
    assert blocked(recorded("route_hold", minimap=(107.5, 55))) is None


@pytest.mark.parametrize("age_s", [1.01, 3.0, 10914.2, -1.01, -30.0, float("nan"), float("inf")])
def test_a_status_read_more_than_a_second_from_its_timestamp_is_stale(age_s):
    assert blocked(age_s=age_s) == "The status is not fresh."


@pytest.mark.parametrize("changes, text", [
    ({"health_protection_state": "suspended"}, "HP protection is suspended"),
    ({"health_protection_state": "healing"}, "HP protection is healing"),
    ({"health_protection_state": "awaiting_response"}, "HP protection is awaiting_response"),
    ({"health_protection_state": None}, "HP protection is None"),
    ({"hp": 74.99}, "HP must read 75% to 100%."), ({"hp": 100.01}, "HP must read"), ({"hp": 0}, "HP must read"),
    ({"hp": None}, "HP must read"), ({"hp": True}, "HP must read"), ({"hp": "80"}, "HP must read"),
    ({"hp": float("nan")}, "HP must read"), ({"hp": float("inf")}, "HP must read"),
    ({"foreground": False}, "not in front"), ({"foreground": None}, "not in front"), ({"foreground": 1}, "not in front"),
    ({"map_ready": False}, "map is not recognised"), ({"map_ready": None}, "map is not recognised"),
    ({"map_state": "candidate"}, "map is not recognised"), ({"map_state": "unknown"}, "map is not recognised"),
    ({"capture_state": "duplicate"}, "capture is not fresh"), ({"capture_state": "stale_or_missing"}, "capture is not fresh"),
    ({"capture_age_s": 1.01}, "capture is not fresh"), ({"capture_age_s": -0.01}, "capture is not fresh"),
    ({"capture_age_s": None}, "capture is not fresh"), ({"capture_age_s": True}, "capture is not fresh"),
    ({"capture_age_s": float("nan")}, "capture is not fresh"),
    ({"minimap_seen": False}, "minimap position"), ({"minimap_seen": None}, "minimap position"),
    ({"minimap": None}, "minimap position"), ({"minimap": [107]}, "minimap position"),
    ({"minimap": [107, 55, 1]}, "minimap position"), ({"minimap": [107, None]}, "minimap position"),
    ({"minimap": "ab"}, "minimap position"), ({"minimap": [True, 55]}, "minimap position"),
    ({"minimap": [float("inf"), 55]}, "minimap position"), ({"minimap": [107, float("nan")]}, "minimap position"),
    ({"capture_age_s": float("inf")}, "capture is not fresh"),
])
def test_one_changed_field_blocks(changes, text):
    assert text in blocked(recorded("route_hold", **changes))


@pytest.mark.parametrize("missing", ["health_protection_state", "hp", "foreground", "map_ready", "map_state",
                                     "capture_state", "capture_age_s", "minimap_seen", "minimap"])
def test_a_missing_field_blocks(missing):
    """The live status of fixtures full_pause_no_snapshot has no hp and no minimap key at all."""
    assert blocked({key: value for key, value in ROUTE_HOLD.items() if key != missing}) is not None


def test_min_hp_is_the_lower_bound():
    assert blocked(recorded("route_hold", hp=66.0), min_hp=65.0) is None
    assert blocked(recorded("route_hold", hp=66.0), min_hp=66.0) is None
    assert blocked(recorded("route_hold", hp=66.0), min_hp=66.5) == "HP must read 66.5% to 100%."


@pytest.mark.parametrize("name", BLOCKING)
def test_each_pending_request_blocks(name):
    assert blocked(pending=(name,)) == "Another request is pending: " + name


def test_the_blocking_request_files_are_exactly_these_and_arm_active_is_not_one():
    assert supervisor.BLOCKING_REQUESTS == ("pause-active", "stop-active", "hold-farming-active",
                                            "return-to-start-active", "reload-active", "resume-active",
                                            "editor-test-request.json", "stop-editor-test")
    assert "arm-active" not in supervisor.BLOCKING_REQUESTS
    assert (supervisor.STATUS_FILE, supervisor.RESUME_FILE, supervisor.OFF_SWITCH) == (
        "active-status.json", "resume-active", "supervise-off")


# ---------------------------------------------------------------------------------------------------------------
# 3. ResumePolicy: the session's memory. Feed shows it statuses the way main() does.
# ---------------------------------------------------------------------------------------------------------------
WALL0, MONO0 = 1_794_758_000.0, 5_000.0  # the wall clock and the monotonic clock are deliberately unrelated


class Feed:
    """One read every 0.25 s; every status is new unless the test says otherwise."""

    def __init__(self, **options):
        self.policy = supervisor.ResumePolicy(**{"per_hour": 4, "max_total": 12, **options})
        self.t, self.serial = 0.0, 0

    def read(self, status, *, new=True, age_s=0.1, **conditions):
        """One poll now: (decision, reason). new=False is a second read of the previous status."""
        self.serial += new
        decision = self.policy.decide(dict(status, updated=WALL0 + self.serial), age_s=age_s, now=MONO0 + self.t,
                                      **conditions)
        self.t += 0.25
        return decision

    def show(self, status, seconds, **conditions):
        """New statuses for that long: [(time of the poll, decision), ...]."""
        return [(self.t, self.read(status, **conditions)[0]) for _ in range(round(seconds / 0.25))]

    def request(self):
        """main() created the request file at the poll that just said "resume"."""
        self.policy.requested(MONO0 + self.t - 0.25)

    def resumed_route_failure(self, start=RETURNING, seconds=6.0):
        """A route hold that the tool resumes at 20.0 s, followed by the start that the controller runs."""
        assert self.show(ROUTE_HOLD, 20.25)[-1] == (20.0, "resume")
        self.request()
        if start is not None:
            assert {decision for _, decision in self.show(start, seconds)} == {"wait"}
        return self


def first(polls, decision="resume"):
    return next((t for t, value in polls if value == decision), None)


def test_a_route_hold_is_resumed_after_20_s_and_not_one_poll_earlier():
    feed = Feed()
    polls = feed.show(ROUTE_HOLD, 20.25)
    assert [decision for _, decision in polls] == ["wait"] * 80 + ["resume"]
    assert feed.read(ROUTE_HOLD) == ("resume", "route: Return failed: waypoint timeout")


def test_the_settle_time_starts_with_the_hold_not_with_the_session():
    feed = Feed()
    assert {decision for _, decision in feed.show(FARMING, 60.0)} == {"wait"}
    assert first(feed.show(ROUTE_HOLD, 25.0)) == 80.0


def test_a_hold_that_was_interrupted_must_settle_again():
    """Continuously present: one status of another kind and the 20 s start again."""
    feed = Feed()
    feed.show(ROUTE_HOLD, 19.0)
    assert feed.read(recorded("automatic_return"))[0] == "wait"            # t = 19.0
    assert first(feed.show(ROUTE_HOLD, 25.0)) == 39.25


@pytest.mark.parametrize("status, conditions", [
    (recorded("route_hold", foreground=False), {}),
    (recorded("route_hold", health_protection_state="suspended"), {}),
    (recorded("route_hold", health_protection_state="healing"), {}),
    (recorded("route_hold", hp=70.0), {}),
    (recorded("route_hold", map_ready=False), {}),
    (recorded("route_hold", capture_state="duplicate"), {}),
    (recorded("route_hold", minimap_seen=False, minimap=None), {}),
    (ROUTE_HOLD, {"age_s": 1.5}),
    (ROUTE_HOLD, {"age_s": -1.5}),
    (ROUTE_HOLD, {"pending": ("pause-active",)}),
    (ROUTE_HOLD, {"pending": ("resume-active",)}),
])
def test_one_ineligible_or_stale_status_restarts_the_5_s(status, conditions):
    feed = Feed()
    assert {decision for _, decision in feed.show(ROUTE_HOLD, 18.0)} == {"wait"}
    assert feed.read(status, **conditions)[0] == "wait"                     # t = 18.0
    polls = feed.show(ROUTE_HOLD, 6.0)                                      # t = 18.25 ... 24.0
    assert first(polls) == 23.25                                            # 5 s after the first eligible status


def test_a_second_read_of_the_same_status_that_is_stale_by_then_restarts_the_5_s():
    feed = Feed()
    feed.show(ROUTE_HOLD, 18.0)
    assert feed.read(ROUTE_HOLD, new=False, age_s=1.2) == ("wait", "The status is not fresh.")   # t = 18.0
    assert first(feed.show(ROUTE_HOLD, 6.0)) == 23.25


def test_the_5_s_need_three_statuses_and_a_status_counts_once():
    feed = Feed()
    feed.show(ROUTE_HOLD, 19.0)
    feed.read(ROUTE_HOLD, age_s=2.0)                                        # t = 19.0: stale, the run is over
    assert feed.read(ROUTE_HOLD)[0] == "wait"                               # t = 19.25: first status of a new run
    again = [feed.read(ROUTE_HOLD, new=False, age_s=0.5) for _ in range(20)]   # t = 19.5 ... 24.25: read again
    assert {decision for decision, _ in again} == {"wait"}                  # 5 s are over at 24.25, with one status
    assert again[-1][1] == "Waiting for 5 s of unbroken, fully eligible statuses."
    assert feed.read(ROUTE_HOLD)[0] == "wait"                               # t = 24.5: the second status
    assert feed.read(ROUTE_HOLD)[0] == "resume"                             # t = 24.75: the third


def test_a_change_of_kind_restarts_the_5_s_even_without_a_settle_time():
    """Rule 7: every status of the 5 s is of the same kind. The 20 s settle hides this rule, so it is switched off."""
    feed = Feed(settle_s=0.0)
    assert {decision for _, decision in feed.show(ROUTE_HOLD, 4.0)} == {"wait"}
    assert feed.read(recorded("hp_hold"))[0] == "wait"                      # t = 4.0: the controller's own HP hold
    assert first(feed.show(ROUTE_HOLD, 6.0)) == 9.25                        # 5 s after the hold came back at 4.25


def test_every_decision_can_be_taken_again_on_the_same_status():
    """main() decides a second time immediately before it creates the file; that must not count anything twice."""
    feed = Feed()
    feed.show(ROUTE_HOLD, 20.0)
    assert feed.read(ROUTE_HOLD)[0] == "resume"
    feed.t -= 0.25
    assert feed.read(ROUTE_HOLD, new=False)[0] == "resume"
    feed.t -= 0.25
    assert feed.read(ROUTE_HOLD, new=False, pending=("pause-active",))[0] == "wait"
    assert feed.read(ROUTE_HOLD)[0] == "wait"                               # and the 5 s have started again


@pytest.mark.parametrize("active", ["farming", "farming_background", "checking", "returning", "automatic_return",
                                    "route_hold", "hp_hold"])
def test_a_pause_only_waits_until_the_controller_has_been_active_in_this_session(active):
    feed = Feed()
    assert {decision for _, decision in feed.show(recorded("waiting"), 30.0)} == {"wait"}
    assert feed.read(recorded("full_pause_no_snapshot")) == (
        "wait", "full pause (user, or the controller after losing the map or capture): "
                "Saved map alignment was not confirmed within 2 seconds.")
    assert feed.read(recorded("watcher_hold"))[0] == "wait"
    assert feed.read(recorded(active))[0] == "wait"
    assert feed.read(recorded("waiting")) == (
        "never", "full pause (user, or the controller after losing the map or capture)")


@pytest.mark.parametrize("status", [full_pause(ROUTE_HOLD), full_pause(recorded("automatic_return")),
                                    recorded("watcher_hold"), HOLD_DURING_START, user_hold(recorded("automatic_return")),
                                    recorded("full_pause_no_snapshot"), recorded("route_hold", manual_health_hold=None)])
def test_a_user_pause_or_hold_after_farming_ends_resuming(status):
    feed = Feed()
    feed.show(FARMING, 5.0)
    assert feed.read(status)[0] == "never"


def test_a_stopped_controller_and_an_unsupported_block_end_resuming_even_at_the_very_start():
    assert Feed().read({"running": False, "pid": 1}) == ("never", "The controller is not running.")
    decision, reason = Feed().read(BLOCKED_WITHOUT_RETURN)
    assert decision == "never" and "return_on_block" in reason


@pytest.mark.parametrize("changes, text", [
    ({"health_protection_state": "needs_attention",
      "health_protection_reason": "HP did not improve after confirmed potion presses"},
     "HP did not improve after confirmed potion presses"),
    ({"health_protection_state": "needs_attention"}, "needs_attention"),      # the state alone
    ({"health_protection_reason": "HP key was not sent"}, "HP key was not sent"),  # the reason alone
    ({"health_protection_state": "dead", "health_protection_reason": "HP is zero"}, "HP is zero"),
    ({"health_protection_state": "dead"}, "dead"),
    ({"health_protection_state": "idle"}, "idle"),                            # HP protection is not running at all
    ({"health_protection_state": None}, "None"),
])
def test_an_hp_protection_alert_in_a_hold_ends_resuming(changes, text):
    for base in ("route_hold", "hp_hold"):
        decision, reason = Feed().read(recorded(base, **changes))
        assert decision == "never" and reason == "HP protection needs attention: " + text
    feed = Feed().resumed_route_failure()
    assert feed.read(dict(failed_start("waypoint timeout"), **changes))[0] == "never"


@pytest.mark.parametrize("state", ["healing", "awaiting_response", "suspended"])
def test_a_potion_in_progress_or_a_suspended_guard_only_waits(state):
    feed = Feed()
    assert {decision for _, decision in feed.show(recorded("route_hold", health_protection_state=state), 30.0)} == {"wait"}
    assert first(feed.show(ROUTE_HOLD, 6.0)) == 35.0                         # eligible from 30.0 on


def test_an_alert_outside_a_hold_does_not_end_resuming():
    """A rebuilt bot publishes "idle" until its first matched frame; a paused controller keeps a latched alert."""
    feed = Feed()
    for status in (recorded("checking", health_protection_state="idle"), recorded("farming", health_protection_state="idle"),
                   recorded("returning", health_protection_state="idle")):
        assert feed.read(status)[0] == "wait"
    alert = full_pause(ROUTE_HOLD, health_protection_state="needs_attention", health_protection_reason="HP is unreadable")
    assert Feed().read(alert)[0] == "wait"                                  # before any activity: a pause only waits


def test_the_controllers_own_hp_hold_only_waits():
    feed = Feed()
    for name in ("hp_hold", "hp_hold_waiting", "hp_hold_during_return"):
        assert {decision for _, decision in feed.show(recorded(name), 30.0)} == {"wait"}


def test_the_off_switch_ends_resuming_in_every_state():
    for status in (ROUTE_HOLD, FARMING, recorded("waiting"), RETURNING):
        assert Feed().read(status, off_switch=True) == ("never", "The off switch log/supervise-off exists.")


# -- failed starts ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("reason", TRANSIENT)
def test_a_failed_start_of_the_tools_own_request_is_retried_for_each_transient_reason(reason):
    feed = Feed().resumed_route_failure()
    polls = feed.show(failed_start(reason), 115.0)                          # the failed-start hold from 26.25 s on
    assert first(polls) == 140.0 and first(polls, "never") is None           # 120 s after the request at 20.0
    assert feed.read(failed_start(reason)) == ("resume", "start_failed: " + reason)


def test_the_transient_reasons_are_exactly_the_ten_of_the_design():
    assert supervisor.TRANSIENT_START_FAILURES == (
        "waypoint timeout", "jump timeout", "ramp timeout", "drop timeout", "drop stalled", "Return route timed out",
        "Stable source unavailable", "left ramp corridor", "ramp knockback did not settle",
        "drop did not leave source floor")


@pytest.mark.parametrize("reason", NEEDS_USER)
def test_a_failed_start_that_needs_the_user_is_never_retried(reason):
    feed = Feed().resumed_route_failure()
    assert feed.read(failed_start(reason)) == ("never", "The failed start needs the user: " + reason)


def test_the_recorded_failed_start_is_not_resumed_whoever_looks_at_it():
    assert Feed().read(recorded("failed_start"))[0] == "never"
    feed = Feed()
    feed.show(FARMING, 5.0)
    assert feed.read(recorded("failed_start"))[0] == "never"


@pytest.mark.parametrize("reason", TRANSIENT)
def test_a_failed_start_that_the_tool_did_not_request_is_never_retried(reason):
    decision, text = Feed().read(failed_start(reason))
    assert decision == "never" and text == "A start that this tool cannot confirm as its own has failed: " + reason
    feed = Feed()
    feed.show(recorded("waiting"), 5.0)                                     # the user's own first start ...
    feed.show(recorded("checking"), 1.0)
    feed.show(RETURNING, 6.0)
    assert feed.read(failed_start(reason))[0] == "never"                    # ... fails


def test_a_start_by_someone_else_after_the_tools_own_failed_start_is_not_the_tools():
    feed = Feed().resumed_route_failure()
    assert {decision for _, decision in feed.show(failed_start("waypoint timeout"), 30.0)} == {"wait"}
    feed.show(recorded("checking"), 0.5)                                    # e.g. the settings app's Resume farming
    feed.show(RETURNING, 6.0)
    assert feed.read(failed_start("waypoint timeout"))[0] == "never"


def test_a_request_that_was_never_followed_by_a_start_does_not_make_a_later_start_the_tools():
    feed = Feed().resumed_route_failure(start=None)
    feed.show(FARMING, 5.0)                                                 # farming again without a visible start
    feed.show(RETURNING, 6.0)                                               # a start nobody asked the tool for
    assert feed.read(failed_start("waypoint timeout"))[0] == "never"


def test_farming_ends_the_claim_on_a_start_that_the_tool_saw_begin():
    """After farming, a start that fails before any status showed it running is nobody's the tool knows."""
    feed = Feed().resumed_route_failure()                                   # the tool's own start was visible
    feed.show(FARMING, 5.0)
    decision, reason = feed.read(failed_start("waypoint timeout"))          # straight from farming to a failed start
    assert decision == "never" and reason.endswith(": waypoint timeout")


def test_a_failed_start_that_was_never_seen_running_is_not_claimed_as_the_tools():
    """The tool's request was made, but the start failed within one status interval: no retry, and the reason
    says what the tool knows (finding 5 of the second review)."""
    feed = Feed()
    assert feed.show(ROUTE_HOLD, 20.25)[-1] == (20.0, "resume")
    feed.request()
    assert feed.read(failed_start("Stable source unavailable: position kept moving")) == (
        "never", "A start that this tool cannot confirm as its own has failed: "
                 "Stable source unavailable: position kept moving")


def test_a_request_the_controller_did_not_act_on_does_not_make_a_later_start_the_tools():
    for on_disk, outcome in ((True, "not_consumed"), (False, "unconfirmed")):
        feed = Feed().resumed_route_failure(start=None)
        assert feed.policy.watch(ROUTE_HOLD, now=MONO0 + 25.0, on_disk=on_disk) == outcome
        feed.show(RETURNING, 6.0)
        assert feed.read(failed_start("waypoint timeout"))[0] == "never"


def test_only_one_retry_per_route_failure():
    feed = Feed().resumed_route_failure()
    assert first(feed.show(failed_start("waypoint timeout"), 114.0)) == 140.0
    feed.request()
    feed.show(RETURNING, 6.0)
    assert feed.read(failed_start("waypoint timeout")) == ("never", "The retried start failed again: waypoint timeout")


def test_farming_in_between_allows_the_next_route_failure_its_own_retry():
    feed = Feed().resumed_route_failure()
    assert first(feed.show(failed_start("jump timeout"), 114.0)) == 140.0   # the one retry of the first failure
    feed.request()
    feed.show(RETURNING, 6.0)
    feed.show(FARMING, 10.0)                                                # the retry worked
    assert first(feed.show(ROUTE_HOLD, 104.0)) == 260.0                     # the next route failure
    feed.request()
    feed.show(RETURNING, 6.0)
    polls = feed.show(failed_start("jump timeout"), 114.0)
    assert first(polls, "never") is None and first(polls) == 380.0


# -- limits ---------------------------------------------------------------------------------------------------------
def requests_over(feed, seconds, status=ROUTE_HOLD):
    """Poll times at which a hold that never ends is resumed, and the last decision."""
    times, last = [], None
    for _ in range(round(seconds / 0.25)):
        t = feed.t
        last = feed.read(status)
        if last[0] == "resume":
            times.append(t)
            feed.request()
            assert feed.policy.watch(RETURNING, now=MONO0 + feed.t, on_disk=False) == "confirmed"   # main() watches
        elif last[0] == "never":
            break
    return times, last


def test_two_requests_are_at_least_120_s_apart():
    times, last = requests_over(Feed(per_hour=6), 300.0)
    assert times == [20.0, 140.0, 260.0]


def test_the_hourly_allowance_waits_and_comes_back_after_exactly_an_hour():
    times, last = requests_over(Feed(per_hour=2), 3900.0)
    assert times == [20.0, 140.0, 3620.0, 3740.0]
    assert last == ("wait", "2 Resume request(s) were made in the last hour; the allowance comes back.")
    assert requests_over(Feed(per_hour=1), 3700.0)[0] == [20.0, 3620.0]


def test_the_session_limit_ends_resuming():
    times, last = requests_over(Feed(per_hour=6, max_total=3), 600.0)
    assert times == [20.0, 140.0, 260.0]
    assert last == ("never", "The session limit of 3 Resume request(s) is used up.")
    assert Feed(max_total=0).read(FARMING)[0] == "never"


def test_the_session_limit_ends_resuming_only_after_its_last_request_was_watched():
    """Switching off withdraws an untaken own request; the limit must not withdraw the request that reached it."""
    feed = Feed(max_total=1, min_interval_s=0.0)
    assert first(feed.show(ROUTE_HOLD, 20.25)) == 20.0
    feed.request()
    polls = feed.show(ROUTE_HOLD, 2.0)                                      # still eligible, still watched
    assert {decision for _, decision in polls} == {"wait"}
    assert feed.read(ROUTE_HOLD) == ("wait", "The last permitted Resume request is still being watched.")
    assert feed.policy.watch(RETURNING, now=MONO0 + feed.t, on_disk=False) == "confirmed"
    assert feed.read(ROUTE_HOLD) == ("never", "The session limit of 1 Resume request(s) is used up.")


# -- the tool's own request -------------------------------------------------------------------------------------------
def test_without_a_request_there_is_nothing_to_watch():
    policy = supervisor.ResumePolicy(per_hour=4, max_total=12)
    assert policy.watch(RETURNING, now=MONO0, on_disk=False) is None


def test_a_request_that_stays_on_disk_for_5_s_was_not_consumed():
    policy = supervisor.ResumePolicy(per_hour=4, max_total=12)
    policy.requested(MONO0)
    assert policy.resume_times == [MONO0]
    assert policy.watch(ROUTE_HOLD, now=MONO0 + 4.75, on_disk=True) is None
    assert policy.watch(recorded("checking"), now=MONO0 + 4.9, on_disk=True) is None   # not taken: not consumed
    assert policy.watch(ROUTE_HOLD, now=MONO0 + 5.0, on_disk=True) == "not_consumed"
    assert policy.sent_at is None and policy.watch(ROUTE_HOLD, now=MONO0 + 6.0, on_disk=True) is None


@pytest.mark.parametrize("status", [recorded("checking"), RETURNING, recorded("automatic_return")])
def test_a_request_is_confirmed_when_its_file_is_gone_and_a_start_is_visible(status):
    policy = supervisor.ResumePolicy(per_hour=4, max_total=12)
    policy.requested(MONO0)
    assert policy.watch(ROUTE_HOLD, now=MONO0 + 0.25, on_disk=False) is None             # taken, nothing to see yet
    assert policy.watch(status, now=MONO0 + 0.5, on_disk=False) == "confirmed"
    assert policy.sent_at is None


def test_a_request_that_went_without_a_visible_start_is_unconfirmed():
    policy = supervisor.ResumePolicy(per_hour=4, max_total=12)
    policy.requested(MONO0)
    assert policy.watch(ROUTE_HOLD, now=MONO0 + 0.25, on_disk=False) is None
    assert policy.watch(full_pause(ROUTE_HOLD), now=MONO0 + 4.75, on_disk=False) is None
    assert policy.watch(full_pause(ROUTE_HOLD), now=MONO0 + 5.0, on_disk=False) == "unconfirmed"


def test_a_file_that_comes_back_after_it_was_taken_is_somebody_elses():
    policy = supervisor.ResumePolicy(per_hour=4, max_total=12)
    policy.requested(MONO0)
    assert policy.watch(ROUTE_HOLD, now=MONO0 + 0.25, on_disk=False) is None
    assert policy.taken is True
    assert policy.watch(ROUTE_HOLD, now=MONO0 + 5.0, on_disk=True) == "unconfirmed"       # never "not_consumed"
    policy.requested(MONO0 + 200.0)
    assert policy.taken is False                                                         # a new request, a new watch


# ---------------------------------------------------------------------------------------------------------------
# 4. main() against a fake controller
# ---------------------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 0.0             # seconds since the session began
        self.wall_step = 0.0     # a wall clock adjustment; the monotonic clock never sees it

    def wall(self, t=None):
        return WALL0 + self.wall_step + (self.t if t is None else t)

    def mono(self):
        return MONO0 + self.t


CHECK_S = 0.5                                                   # how long a start shows "checking" (1-3 recorded rows)
START_THEN_FARMING = ((RETURNING, 6.0), (FARMING, None))        # a guarded start that works
START_THEN_ROUTE_FAILS = ((RETURNING, 4.0), (FARMING, 2.0), (ROUTE_HOLD, None))


class FakeController:
    """The controller's side of the signal files, as scripts/run_active.py implements it in normal mode.

    A loop turn is 0.14-0.2 s long (about six a second). It first reads the request files in the controller's order:
    Stop wins and consumes nothing else; a connection test deletes a Resume request unread; Pause all wins over
    Resume and deletes it; Resume starts a guarded start unless one is running; Hold farming cancels a start.
    Then the state moves on, then a status is published if 0.15 s have passed since the last one
    (`stamp - last_status >= 0.15`), which gives the recorded 0.2-0.33 s between statuses. Published statuses are
    recorded ones, changed the way maplebot/bot.py and maplebot/recovery_runtime.py change them.
    The F1 hotkey and the settings app's buttons are methods; a test schedules them with at() or when().
    """
    TURNS = (0.14, 0.19, 0.14, 0.19, 0.20)

    def __init__(self, clock, log, script, *, start=START_THEN_FARMING, reads_requests=True):
        self.clock, self.log = clock, log
        self.state = dict(script[0][0])   # the status that is published, without pid and updated
        self.plan = []                    # what the controller does by itself: [(time, action), ...]
        self.starts = [start] if isinstance(start[0][0], dict) else list(start)   # one plan, or one per start
        self.before_start = None          # the state a cancelled start goes back to
        self.reads_requests = reads_requests
        self.user, self.triggers = [], []
        self.now, self.next_turn, self.turns, self.last_publish = 0.0, 0.0, 0, None
        self.consumed = []                # when Resume requests were consumed
        self.pid, self.lag, self.garbage_until, self.frozen_from, self.stopped = 18480, 0.0, None, None, False
        self.follow(0.0, script)

    # -- scripting -------------------------------------------------------------------------------------------------
    def follow(self, t, steps):
        """From t on: each (status, seconds) in turn; the last one, with seconds None, stays."""
        self.plan = []
        for status, seconds in steps:
            self.plan.append((t, lambda status=status: self.state.clear() or self.state.update(status)))
            if seconds is None:
                break
            t += seconds

    def at(self, t, action):
        self.user.append((t, action))
        self.user.sort(key=lambda entry: entry[0])

    def when(self, condition, action):
        self.triggers.append((condition, action))

    def starting(self):
        return self.state.get("startup_return_state") in ("checking", "returning")

    # -- what the user can do ------------------------------------------------------------------------------------------
    def press_f1(self):
        """scripts/run_active.py toggle_normal_pause: a hotkey thread, no file. Pauses unless fully paused."""
        if self.state["paused"] and self.state["full_pause"] and not self.starting():
            self.request_start()
        else:
            self.cancel_start("Start cancelled with F1.")
            self.pause_all()

    def click_pause_all(self):
        """maplebot/gui.py _pause_resume while the button reads "Pause all" or "Pause"."""
        (self.log / "resume-active").unlink(missing_ok=True)
        (self.log / "pause-active").touch()

    def click_resume(self):
        """maplebot/gui.py _resume_farming: it leaves arm-active behind for the controller's lifetime."""
        for name in ("pause-active", "hold-farming-active", "return-to-start-active"):
            (self.log / name).unlink(missing_ok=True)
        (self.log / "resume-active").touch()
        (self.log / "arm-active").touch()

    def click_hold(self):
        """EditorBridge.hold_farming(), also behind the settings app's button. False: "Farming is already held"."""
        state = self.state
        if state["full_pause"] is False and not self.starting() and (state["manual_health_hold"]
                                                                    or state["recovery_terminal"]):
            return False
        (self.log / "hold-farming-active").write_text(
            json.dumps({"created": self.clock.wall(self.now), "source": "settings"}), encoding="utf-8")
        return True

    # -- the controller ------------------------------------------------------------------------------------------------
    def pause_all(self):
        self.plan = []
        self.state.update(full_pause(self.state))

    def request_start(self):
        """StartupReturn.request(): ignored while a start runs; otherwise the start's own hold, then the plan."""
        if self.starting():
            return
        self.before_start = dict(self.state)
        self.state.update(paused=True, full_pause=False, waiting_to_start=False, manual_health_hold=True,
                          health_hold_reason=CHECKING, startup_return_state="checking", startup_return_reason=CHECKING,
                          health_protection_active=True, health_protection_state="monitoring")
        self.follow(self.now + CHECK_S, self.starts.pop(0) if len(self.starts) > 1 else self.starts[0])

    def cancel_start(self, reason):
        """StartupReturn.cancel(): the circuit the start replaced comes back."""
        if self.starting():
            self.state = dict(self.before_start, startup_return_state="cancelled", startup_return_reason=reason)
            self.plan = []

    def read_requests(self):
        def present(name):
            return (self.log / name).exists()

        def drop(*names):
            for name in names:
                (self.log / name).unlink(missing_ok=True)

        if present("stop-active"):   # run_active.py:403-415, 661: the runner ends; resume-active is not consumed
            drop("hold-farming-active", "return-to-start-active")
            self.state = {"running": False, "frames": self.state.get("frames"), "keys_sent": {},
                          "hp": self.state.get("hp")}
            self.publish()
            self.stopped = True
            return
        assert not present("reload-active") and not present("return-to-start-active"), "not modelled"
        if present("stop-editor-test") or present("editor-test-request.json"):   # :439-463
            self.cancel_start("Start cancelled by a connection test.")
            drop("stop-editor-test", "editor-test-request.json", "hold-farming-active", "pause-active", "resume-active")
            self.pause_all()
            return
        pause = present("pause-active")
        if pause:                    # :465-472: Pause all wins and deletes the Resume request
            self.cancel_start("Start cancelled by Pause all.")
            drop("hold-farming-active", "pause-active", "resume-active")
            self.pause_all()
        elif present("resume-active"):   # :473-478
            drop("resume-active")
            self.consumed.append(self.now)
            self.request_start()
        if present("hold-farming-active") and not pause:   # :486-490 and 500-510, recovery_runtime.request_hold
            self.cancel_start("Start cancelled by Hold farming.")
            drop("hold-farming-active")
            self.pause_all()
            self.state.update(user_hold(self.state))
            self.plan = [(self.now + 0.3, lambda: self.state.update(health_protection_state="monitoring"))]

    def publish(self):
        path = self.log / "active-status.json"
        if self.garbage_until is not None and self.now < self.garbage_until:
            path.write_text('{"running": tr', encoding="utf-8")   # a reader caught the file half written
        else:
            status = dict(self.state, pid=self.pid, updated=self.clock.wall(self.now) - self.lag)
            path.write_text(json.dumps(status), encoding="utf-8")
        self.last_publish = self.now

    def turn(self):
        if self.stopped or (self.frozen_from is not None and self.now >= self.frozen_from):
            return
        for trigger in list(self.triggers):
            if trigger[0](self):
                self.triggers.remove(trigger)
                trigger[1]()
        if self.reads_requests:
            self.read_requests()
        while self.plan and self.plan[0][0] <= self.now + 1e-9:
            self.plan.pop(0)[1]()
        if not self.stopped and (self.last_publish is None or self.now - self.last_publish >= 0.15 - 1e-9):
            self.publish()

    def advance(self, to):
        """Everything that happens up to that time, in time order."""
        while True:
            user = self.user[0][0] if self.user else math.inf
            when = min(user, self.next_turn)
            if when > to + 1e-9:
                return
            self.now = when
            if user <= self.next_turn:
                self.user.pop(0)[1]()
            else:
                self.turn()
                self.next_turn = when + self.TURNS[self.turns % len(self.TURNS)]
                self.turns += 1


class Guard:
    """Fails the test when the tool does anything to the file system that it may not do.

    Allowed: read anything; write its --output file; create log/resume-active; delete that file while it is still
    the tool's own (nobody else has taken, replaced or touched it). Not allowed: every other delete, rename or
    replace; every other write (in log/ or elsewhere); starting a process. The fake controller and the simulated
    user are not the tool; what they do is not checked, only noted.
    """
    WRAPPED = (
        (os, "os", ("unlink", "remove", "rmdir", "removedirs", "rename", "renames", "replace"), "delete"),
        (pathlib.Path, "Path", ("unlink", "rmdir", "rename", "replace"), "delete"),
        (shutil, "shutil", ("rmtree", "move"), "delete"),
        (os, "os", ("open", "utime", "truncate", "mkdir", "makedirs", "link", "symlink"), "write"),
        (builtins, "builtins", ("open",), "write"),
        (io, "io", ("open",), "write"),
        (pathlib.Path, "Path", ("touch", "write_text", "write_bytes", "open", "mkdir"), "write"),
        (shutil, "shutil", ("copy", "copy2", "copyfile"), "write"),
        (subprocess, "subprocess", ("Popen",), "process"),
        (os, "os", ("system", "startfile", "popen", "spawnl", "spawnle", "spawnv", "spawnve", "execv", "execve",
                    "execl", "execle"), "process"),
    )

    def __init__(self, log, output):
        self.request, self.output = self.norm(log / "resume-active"), self.norm(output)
        self.tool = False            # True while supervisor.main() itself runs
        self.own_request = False     # log/resume-active was created by the tool and nobody else has touched it since
        self.created = self.deleted = self.depth = 0
        self.violations = []

    @staticmethod
    def norm(path):
        return os.path.normcase(os.path.abspath(os.fspath(path)))

    @staticmethod
    def writes(label, args, kwargs):
        if label in ("builtins.open", "io.open", "Path.open"):
            return any(letter in str(args[1] if len(args) > 1 else kwargs.get("mode", "r")) for letter in "wax+")
        if label == "os.open":
            return bool(args[1] & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
        return True

    def install(self, patch):
        for owner, prefix, names, kind in self.WRAPPED:
            for name in names:
                if hasattr(owner, name):
                    patch.setattr(owner, name, self.wrap(getattr(owner, name), f"{prefix}.{name}", kind))

    def wrap(self, real, label, kind):
        def call(*args, **kwargs):
            try:
                target = None if not args or isinstance(args[0], int) else self.norm(args[0])
            except TypeError:   # not a path (subprocess arguments)
                target = None
            changes = kind == "delete" or (kind == "write" and self.writes(label, args, kwargs))
            if self.tool:
                allowed = not changes and kind != "process"
                if kind == "delete":
                    allowed = (target == self.request and self.own_request
                               and label in ("os.unlink", "os.remove", "Path.unlink"))
                elif changes:
                    allowed = target == self.output or (target == self.request
                                                        and label in ("Path.touch", "os.open", "os.utime"))
                if not allowed:
                    self.violations.append(f"the tool called {label} on {args[:1]}")
                    raise AssertionError(self.violations[-1])
            self.depth += 1
            try:
                result = real(*args, **kwargs)
            finally:
                self.depth -= 1
            if changes and target == self.request:
                self.own_request = self.tool and kind == "write"
                if self.tool and not self.depth:   # Path.unlink calls os.unlink, Path.touch calls os.open: count once
                    self.created += kind == "write"
                    self.deleted += kind == "delete"
            return result
        return call


class Session:
    """Runs supervisor.main() in tmp_path on the fake clocks; every sleep moves the clocks and the fake controller."""

    def __init__(self, monkeypatch, tmp_path):
        self.monkeypatch, self.root = monkeypatch, tmp_path
        self.log, self.output = tmp_path / "log", tmp_path / "session.jsonl"
        self.log.mkdir()
        self.clock, self.controller, self.guard = Clock(), None, None
        self.before_recheck = None   # runs between the tool's decision and its second read of the status
        self.before_create = None    # runs after the tool's last check, immediately before it creates the file
        self.interrupt_at = None     # Ctrl+C during the sleep that reaches this time
        self.sleeps = 0

    def control(self, *script, **options):
        """script: (status, seconds) steps; the last one stays. Returns the fake controller."""
        self.controller = FakeController(self.clock, self.log, script, **options)
        return self.controller

    def files(self):
        return sorted(path.name for path in self.log.iterdir() if path.name != "active-status.json")

    def run(self, *arguments, minutes):
        guard = self.guard = Guard(self.log, self.output)
        limit = minutes * 60 / supervisor.RETRY_S + 200
        counts = {"clock": 0, "reads": 0}
        real_read = supervisor.read_status

        def others(action):   # the fake controller and the simulated user are not the tool
            guard.tool = False
            try:
                action()
            finally:
                guard.tool = True

        def sleep(seconds):
            assert seconds > 0, "time.sleep(0): the loop would spin"
            self.sleeps += 1
            assert self.sleeps <= limit, "the loop does not end"
            counts["clock"] = counts["reads"] = 0
            self.clock.t += seconds
            if self.controller is not None:
                others(lambda: self.controller.advance(self.clock.t))
            if self.interrupt_at is not None and self.clock.t >= self.interrupt_at:
                self.interrupt_at = None
                raise KeyboardInterrupt

        def monotonic():
            counts["clock"] += 1
            assert counts["clock"] < 500, "the loop spins without sleeping"
            return self.clock.mono()

        def read_status(path):
            counts["reads"] += 1
            if counts["reads"] == 2 and self.before_recheck is not None:   # the read immediately before a request
                hook, self.before_recheck = self.before_recheck, None
                others(hook)
            return real_read(path)

        if self.controller is not None:
            self.controller.advance(0.0)
        with self.monkeypatch.context() as patch:
            guard.install(patch)
            guarded_touch = pathlib.Path.touch

            def touch(path, *args, **kwargs):
                if guard.tool and self.before_create is not None:
                    hook, self.before_create = self.before_create, None
                    others(hook)
                return guarded_touch(path, *args, **kwargs)

            patch.setattr(pathlib.Path, "touch", touch)
            patch.setattr(supervisor, "read_status", read_status)
            patch.setattr(supervisor.time, "time", self.clock.wall)
            patch.setattr(supervisor.time, "monotonic", monotonic)
            patch.setattr(supervisor.time, "sleep", sleep)
            guard.tool = True
            try:
                code = supervisor.main(["--root", str(self.root), "--output", str(self.output),
                                        "--minutes", str(minutes), *arguments])
            finally:
                guard.tool = False
        assert code == 0 and guard.violations == []
        assert "scripts.run_active" not in sys.modules and "run_active" not in sys.modules
        return [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def session(monkeypatch, tmp_path):
    return Session(monkeypatch, tmp_path)


def of(events, name):
    return [event for event in events if event["event"] == name]


def times(events, name="resume_requested"):
    return [event["elapsed_s"] for event in of(events, name)]


def off_reasons(events):
    return [event["reason"] for event in of(events, "auto_resume_off")]


def first_sample(events, kind):
    return next(event["elapsed_s"] for event in of(events, "sample")
                if supervisor.classify_hold(event["status"])[0] == kind)


AUTO = ("--auto-resume", "4")


# -- the recorder ----------------------------------------------------------------------------------------------------
def test_without_the_option_it_only_records(session, capsys):
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    for name in ("pause-active", "supervise-off"):
        (session.log / name).write_text("x", encoding="utf-8")
    events = session.run(minutes=1)
    assert [event["event"] for event in events[:2]] == ["supervisor_started", "sample"]
    assert {event["event"] for event in events} == {"supervisor_started", "sample", "supervisor_finished"}
    samples = of(events, "sample")
    assert 195 <= len(samples) <= 215 and events[-1]["rows"] == len(samples)   # 3.5 statuses a second for 60 s
    assert samples[0]["status"] == dict(ROUTE_HOLD, pid=controller.pid, updated=WALL0)
    assert all(0 <= b["elapsed_s"] - a["elapsed_s"] <= 0.5 for a, b in zip(samples, samples[1:]))
    assert events[-1]["reason"] == "time limit" and events[-1]["resumes"] == 0
    assert events[-1]["own_request_withdrawn"] is False
    assert session.files() == ["pause-active", "supervise-off"] and session.guard.created == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f"supervisor finished: time limit; {len(samples)} samples, 0 Resume request(s), "
                   f"output {session.output}"]


def test_the_start_row_names_the_settings(session):
    session.control((FARMING, None))
    events = session.run("--auto-resume", "3", "--max-resumes", "7", "--min-hp", "80", minutes=0.05)
    assert events[0] == {"event": "supervisor_started", "started": WALL0, "minutes": 0.05,
                         "auto_resume_per_hour": 3, "max_resumes": 7, "min_hp": 80.0}


# -- a route failure is resumed ---------------------------------------------------------------------------------------
def test_a_route_failure_is_resumed_20_s_after_its_first_status(session, capsys):
    controller = session.control((FARMING, 45.0), (ROUTE_HOLD, None))
    events = session.run(*AUTO, minutes=1.5)
    hold = first_sample(events, "route")
    assert 45.0 <= hold <= 45.6
    assert times(events) == [hold + 20.0]                                   # not on the first eligible poll
    request = of(events, "resume_requested")[0]
    assert request["count"] == 1 and request["reason"] == "route: Return failed: waypoint timeout"
    assert request["time"] == WALL0 + request["elapsed_s"] and request["position"] == [107, 55]
    assert len(controller.consumed) == 1 and 0 < controller.consumed[0] - request["elapsed_s"] <= 0.2
    assert 0 < times(events, "resume_confirmed")[0] - request["elapsed_s"] <= 0.75
    assert off_reasons(events) == [] and of(events, "resume_not_consumed") == []
    assert events[-1]["reason"] == "time limit" and events[-1]["resumes"] == 1
    assert session.files() == [] and (session.guard.created, session.guard.deleted) == (1, 0)
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("auto-resume on: at most 4 Resume request(s) per hour, 12 in this session.")
    assert str((session.log / "supervise-off").resolve()) in out[0] and "Hold farming does not stop it" in out[0]
    assert out[1] == "Resume requested (1): route: Return failed: waypoint timeout"
    assert out[2].startswith("supervisor finished: time limit;") and "1 Resume request(s)" in out[2] and len(out) == 3


def test_potion_pauses_and_the_controllers_own_return_only_make_it_wait(session):
    """18-30 s of automatic return and a potion pause on the way to the route failure; neither ends resuming."""
    session.control((FARMING, 5.0), (recorded("hp_hold"), 0.4), (recorded("hp_hold_waiting"), 1.2), (FARMING, 5.0),
                    (recorded("automatic_return"), 15.0), (recorded("hp_hold_during_return"), 1.0),
                    (recorded("automatic_return"), 4.0), (ROUTE_HOLD, None))
    events = session.run(*AUTO, minutes=1.0)
    assert off_reasons(events) == []
    assert times(events) == [first_sample(events, "route") + 20.0]
    notes = [event["reason"] for event in of(events, "resume_waiting")]
    assert "The controller's own low-HP hold." in notes


def test_a_session_may_begin_while_the_controller_still_waits_for_its_first_start(session):
    """The app's Resume starts the controller and leaves arm-active behind; neither switches the tool off."""
    controller = session.control((recorded("waiting"), None), start=START_THEN_ROUTE_FAILS)
    controller.at(10.0, controller.click_resume)
    events = session.run(*AUTO, minutes=1.0)
    assert off_reasons(events) == []
    assert times(events) == [first_sample(events, "route") + 20.0]
    assert len(controller.consumed) == 2                                    # the app's request, then the tool's
    assert session.files() == ["arm-active"]


def test_arm_active_does_not_block(session):
    session.control((ROUTE_HOLD, None))
    (session.log / "arm-active").touch()
    before = (session.log / "arm-active").stat().st_mtime_ns
    assert times(session.run(*AUTO, minutes=0.45)) == [20.0]
    assert session.files() == ["arm-active"] and (session.log / "arm-active").stat().st_mtime_ns == before


def test_focusing_the_game_never_fires_at_once(session):
    """The route failed while another window was in front; the 5 s begin when the game comes to the front."""
    background = recorded("route_hold", foreground=False, health_protection_state="suspended")
    session.control((background, 30.0), (ROUTE_HOLD, None))
    events = session.run(*AUTO, minutes=0.75)
    front = next(event["elapsed_s"] for event in of(events, "sample") if event["status"]["foreground"])
    assert 30.0 <= front <= 30.6 and times(events) == [front + 5.0]


def test_a_wall_clock_step_changes_no_duration(session):
    """Durations run on the monotonic clock; the controller's timestamps and the tool's freshness on the wall clock."""
    controller = session.control((ROUTE_HOLD, None), start=START_THEN_ROUTE_FAILS)
    controller.at(5.0, lambda: setattr(session.clock, "wall_step", 7200.0))   # the clock is set two hours ahead
    events = session.run("--auto-resume", "1", minutes=1.0)
    assert times(events) == [20.0] and "The status is not fresh." in [e["reason"] for e in of(events, "resume_waiting")]
    assert events[-1]["reason"] == "time limit" and 60.0 <= session.clock.t <= 60.25
    assert max(event["elapsed_s"] for event in of(events, "sample")) < 60.25


# -- limits ---------------------------------------------------------------------------------------------------------
def test_at_the_hourly_limit_it_waits_and_stays_on(session):
    controller = session.control((ROUTE_HOLD, None), start=START_THEN_ROUTE_FAILS)
    events = session.run("--auto-resume", "1", minutes=2.6)
    assert times(events) == [20.0] and controller.consumed and off_reasons(events) == []
    assert of(events, "resume_waiting")[-1]["reason"] == (
        "1 Resume request(s) were made in the last hour; the allowance comes back.")


def test_requests_are_120_s_apart_and_the_session_limit_ends_resuming(session, capsys):
    controller = session.control((ROUTE_HOLD, None), start=START_THEN_ROUTE_FAILS)
    events = session.run("--auto-resume", "6", "--max-resumes", "2", minutes=4.6)
    assert times(events) == [20.0, 140.0] and len(controller.consumed) == 2
    assert [event["count"] for event in of(events, "resume_requested")] == [1, 2]
    assert off_reasons(events) == ["The session limit of 2 Resume request(s) is used up."]
    assert times(events, "auto_resume_off") == [140.25]
    assert len(of(events, "resume_confirmed")) == 2                         # the last request is still watched
    assert events[-1]["resumes"] == 2 and session.files() == []
    assert ("auto-resume off for the rest of this session: The session limit of 2 Resume request(s) is used up."
            in capsys.readouterr().out.splitlines())


def test_two_an_hour_means_two(session):
    session.control((ROUTE_HOLD, None), start=START_THEN_ROUTE_FAILS)
    events = session.run("--auto-resume", "2", minutes=4.6)
    assert times(events) == [20.0, 140.0] and off_reasons(events) == []


# -- failed starts ----------------------------------------------------------------------------------------------------
def test_the_tools_own_failed_start_is_retried_once(session):
    failing = ((RETURNING, 6.0), (failed_start("waypoint timeout"), None))
    controller = session.control((ROUTE_HOLD, None), start=failing)
    events = session.run(*AUTO, minutes=4.6)
    assert times(events) == [20.0, 140.0]
    assert [event["reason"] for event in of(events, "resume_requested")] == [
        "route: Return failed: waypoint timeout", "start_failed: waypoint timeout"]
    assert off_reasons(events) == ["The retried start failed again: waypoint timeout"]
    assert len(controller.consumed) == 2 and session.files() == []


def test_a_failed_start_that_asks_for_the_user_is_not_retried(session):
    """The only recorded failed start: 'Could not reach a drawn platform ... Move the character ...'."""
    controller = session.control((ROUTE_HOLD, None), start=((RETURNING, 6.0), (recorded("failed_start"), None)))
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and len(controller.consumed) == 1
    assert off_reasons(events) == ["The failed start needs the user: "
                                   + RECORDED["failed_start"]["status"]["startup_return_reason"]]


def test_a_failed_start_found_at_the_beginning_is_left_alone(session):
    session.control((recorded("failed_start"), None))
    events = session.run(*AUTO, minutes=0.5)
    assert times(events) == [] and times(events, "auto_resume_off") == [0.0]
    assert off_reasons(events)[0].startswith("A start that this tool cannot confirm as its own has failed: Could not reach")
    assert session.files() == []


def test_the_users_own_failed_first_start_is_not_retried(session):
    controller = session.control((recorded("waiting"), None), start=((RETURNING, 6.0), (failed_start("waypoint timeout"), None)))
    controller.at(3.0, controller.press_f1)
    events = session.run(*AUTO, minutes=0.5)
    assert times(events) == [] and session.files() == []
    assert off_reasons(events) == ["A start that this tool cannot confirm as its own has failed: waypoint timeout"]


# -- HP and freshness -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("hp, arguments, expected", [
    (74.9, (), []), (75.0, (), [20.0]), (70.0, ("--min-hp", "65"), [20.0]), (79.8, ("--min-hp", "80"), []),
    (100.0, ("--min-hp", "100"), [20.0])])
def test_hp_must_reach_min_hp(session, hp, arguments, expected):
    session.control((recorded("route_hold", hp=hp), None), reads_requests=False)
    events = session.run(*AUTO, *arguments, minutes=0.4)
    assert times(events) == expected
    if not expected:
        assert of(events, "resume_waiting")[-1]["reason"].startswith("HP must read") and off_reasons(events) == []


def test_a_status_that_lags_behind_the_wall_clock_is_never_acted_on(session):
    controller = session.control((ROUTE_HOLD, None))
    controller.lag = 10.0
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and off_reasons(events) == [] and session.files() == []
    assert {event["reason"] for event in of(events, "resume_waiting")} == {"The status is not fresh."}


def test_a_status_from_the_future_is_never_acted_on(session):
    controller = session.control((ROUTE_HOLD, None))
    controller.lag = -10.0
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and session.files() == []


def test_a_controller_that_stalls_for_two_seconds_restarts_the_5_s(session):
    controller = session.control((ROUTE_HOLD, None))
    controller.frozen_from = 17.0
    controller.at(19.0, lambda: setattr(controller, "frozen_from", None))
    events = session.run(*AUTO, minutes=0.5)
    fresh_again = next(event["elapsed_s"] for event in of(events, "sample") if event["elapsed_s"] > 18.0)
    assert 19.0 <= fresh_again <= 19.6 and times(events) == [fresh_again + 5.0]


# -- other people's requests ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["pause-active", "stop-active", "hold-farming-active", "return-to-start-active",
                                  "reload-active", "resume-active", "editor-test-request.json", "stop-editor-test"])
def test_a_pending_request_file_blocks_and_is_left_untouched(session, name):
    """The controller has not read the file yet (or never will): the tool adds nothing and changes nothing."""
    session.control((ROUTE_HOLD, None), reads_requests=False)
    path = session.log / name
    path.write_text("somebody else's request", encoding="utf-8")
    os.utime(path, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
    before = path.stat().st_mtime_ns
    events = session.run(*AUTO, minutes=0.45)
    assert times(events) == [] and off_reasons(events) == []
    assert of(events, "resume_waiting")[-1]["reason"] == "Another request is pending: " + name
    assert session.files() == [name]
    assert path.read_text(encoding="utf-8") == "somebody else's request" and path.stat().st_mtime_ns == before
    assert (session.guard.created, session.guard.deleted) == (0, 0)


def test_a_request_file_that_comes_and_goes_restarts_the_5_s(session):
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    controller.at(18.1, lambda: (session.log / "hold-farming-active").write_text("{}", encoding="utf-8"))
    controller.at(19.1, lambda: (session.log / "hold-farming-active").unlink())
    assert times(session.run(*AUTO, minutes=0.45)) == [24.25]                # 5 s after the first poll without it


# -- the off switch ----------------------------------------------------------------------------------------------------
def test_the_off_switch_is_read_only_and_ends_resuming_at_once(session, capsys):
    session.control((ROUTE_HOLD, None))
    switch = session.log / "supervise-off"
    switch.write_text("", encoding="utf-8")
    before = switch.stat().st_mtime_ns
    events = session.run(*AUTO, minutes=0.5)
    assert times(events) == [] and times(events, "auto_resume_off") == [0.0]
    assert off_reasons(events) == ["The off switch log/supervise-off exists."]
    assert session.files() == ["supervise-off"] and switch.stat().st_mtime_ns == before
    out = capsys.readouterr().out.splitlines()
    assert str(switch.resolve()) in out[0]
    assert out[1] == "auto-resume off for the rest of this session: The off switch log/supervise-off exists."


@pytest.mark.parametrize("appears", [10.0, 19.9])
def test_the_off_switch_latches_even_if_it_is_removed_again(session, appears):
    controller = session.control((ROUTE_HOLD, None))
    switch = session.log / "supervise-off"
    controller.at(appears, switch.touch)
    controller.at(appears + 0.6, switch.unlink)
    events = session.run(*AUTO, minutes=1.0)
    assert times(events) == [] and len(off_reasons(events)) == 1 and session.files() == []
    assert appears <= times(events, "auto_resume_off")[0] <= appears + 0.25


def test_the_off_switch_is_seen_while_farming(session):
    controller = session.control((FARMING, 30.0), (ROUTE_HOLD, None))
    controller.at(3.0, (session.log / "supervise-off").touch)
    events = session.run(*AUTO, minutes=1.0)
    assert times(events) == [] and times(events, "auto_resume_off") == [3.0]


# -- the user steps in ---------------------------------------------------------------------------------------------------
def nothing_after(session, events, action_time):
    """After the user's action: one switch-off, no request, nothing created; the route hold later changes nothing."""
    assert times(events) == [] and len(off_reasons(events)) == 1
    assert action_time <= times(events, "auto_resume_off")[0] <= action_time + 0.75
    assert (session.guard.created, session.guard.deleted) == (0, 0) and session.files() == []
    assert first_sample(events, "route") is not None


@pytest.mark.parametrize("moment", [5.0, 17.0], ids=["during the settle", "during the 5 s"])
@pytest.mark.parametrize("action", ["press_f1", "click_pause_all"])
def test_a_pause_in_the_route_hold_ends_resuming_and_a_later_route_failure_is_not_resumed(session, moment, action):
    """F1 or Pause all, later F1 again: the user's own start, farming, the next route failure. The tool stays out."""
    controller = session.control((ROUTE_HOLD, None), start=START_THEN_ROUTE_FAILS)
    controller.at(moment, getattr(controller, action))
    controller.at(30.0, controller.press_f1)
    events = session.run(*AUTO, minutes=1.1)
    nothing_after(session, events, moment)
    assert off_reasons(events) == ["full pause (user, or the controller after losing the map or capture)"]
    assert first_sample(events, "starting") > 30.0 and events[-1]["reason"] == "time limit"
    last = of(events, "sample")[-1]["status"]
    assert supervisor.classify_hold(last)[0] == "route"                     # held for 25 s at the end, untouched


def test_hold_farming_in_a_route_hold_changes_nothing_and_cannot_stop_the_tool(session):
    """The documented limit: EditorBridge.hold_farming() answers 'already held' and writes no request."""
    controller = session.control((ROUTE_HOLD, None))
    answers = []
    controller.at(5.0, lambda: answers.append(controller.click_hold()))
    events = session.run(*AUTO, minutes=0.45)
    assert answers == [False] and times(events) == [20.0]


def test_the_status_changes_between_the_decision_and_the_request(session):
    """F1 is pressed and published after the tool read the status: the second read sees a newer status."""
    controller = session.control((ROUTE_HOLD, None))
    session.before_recheck = lambda: (controller.press_f1(), controller.publish())
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and session.guard.created == 0 and session.files() == []
    assert of(events, "resume_waiting")[-1] == {"event": "resume_waiting", "elapsed_s": 20.0,
                                                "reason": "The status changed while deciding."}
    assert off_reasons(events) == ["full pause (user, or the controller after losing the map or capture)"]
    assert times(events, "auto_resume_off") == [20.25]


@pytest.mark.parametrize("name", ["pause-active", "hold-farming-active", "stop-editor-test"])
def test_a_request_file_appears_between_the_decision_and_the_request(session, name):
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    session.before_recheck = lambda: (session.log / name).write_text("{}", encoding="utf-8")
    events = session.run(*AUTO, minutes=0.45)
    assert times(events) == [] and session.guard.created == 0 and session.files() == [name]
    assert of(events, "resume_waiting")[-1]["reason"] == "Another request is pending: " + name


def test_the_off_switch_appears_between_the_decision_and_the_request(session):
    session.control((ROUTE_HOLD, None))
    session.before_recheck = (session.log / "supervise-off").touch
    events = session.run(*AUTO, minutes=0.45)
    assert times(events) == [] and times(events, "auto_resume_off") == [20.0] and session.guard.created == 0


def test_pause_all_clicked_in_the_moment_of_the_request_wins(session, capsys):
    """After the tool's last check: the app deletes the Resume request and asks for a pause; nothing starts."""
    controller = session.control((ROUTE_HOLD, None))
    session.before_create = controller.click_pause_all                      # too late for the tool to see
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and controller.consumed == [] and session.files() == []
    assert off_reasons(events) == ["full pause (user, or the controller after losing the map or capture)"]
    assert times(events, "resume_unconfirmed") == [25.0] and of(events, "resume_unconfirmed")[0]["removed"] is False
    assert session.guard.deleted == 0
    assert all(supervisor.classify_hold(event["status"])[0] in ("route", "user") for event in of(events, "sample"))
    assert ("WARNING: resume-active was taken, but no start became visible within 5 s."
            in capsys.readouterr().out.splitlines())


@pytest.mark.parametrize("hook", ["before_recheck", "before_create"])
def test_a_pause_the_controller_has_not_published_yet_cannot_be_seen(session, hook):
    """The limit the docstring states. F1 acts without a file, and a Pause all request is gone once the controller
    has read it; either shows in the next status only (up to a third of a second later). A request created in
    between is consumed after the pause and starts a guarded start out of it (scripts/run_active.py:473-478).
    Only the controller could close this."""
    controller = session.control((ROUTE_HOLD, None))
    setattr(session, hook, controller.press_f1)                             # paused, not published yet
    events = session.run(*AUTO, minutes=0.45)
    assert times(events) == [20.0] and len(controller.consumed) == 1 and len(of(events, "resume_confirmed")) == 1


def test_a_status_older_than_the_act_limit_is_never_acted_on(session):
    """Finding 1 of the second review: a status may be up to 1 s old for the 5 s, but the one the request is made
    on must be at most 0.35 s old, so that a pause the controller has not published yet is at most that old."""
    controller = session.control((ROUTE_HOLD, None))
    controller.lag = 0.5                                                    # 0.5-0.83 s old at every read
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and off_reasons(events) == [] and session.guard.created == 0
    assert of(events, "resume_waiting")[-1]["reason"] == (
        "The status is more than 0.35 s old; the request waits for a newer one.")


@pytest.mark.parametrize("stall", [0.45, 0.5, 0.7, 0.9])
def test_an_f1_hidden_by_a_controller_stall_is_seen_before_any_request(session, stall):
    """Harness E1 of the second review: the controller's loop stalls, F1 is pressed in the stall (no file), and the
    status the tool reads at 20.0 s is still the route hold. Before the fix the tool requested at 20.0 s and the
    controller started out of the F1 pause at 20.11 s, with resuming still on."""
    controller = session.control((ROUTE_HOLD, None))
    begin = 20.0 - stall + 0.05
    controller.frozen_from = begin
    controller.at(begin + 0.01, controller.press_f1)
    controller.at(20.1, lambda: setattr(controller, "frozen_from", None))
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and controller.consumed == [] and session.guard.created == 0
    assert off_reasons(events) == ["full pause (user, or the controller after losing the map or capture)"]
    assert of(events, "sample")[-1]["status"]["full_pause"] is True


def test_the_off_switch_withdraws_the_tools_request_that_was_not_taken_yet(session, capsys):
    """Harness E3 of the second review: the controller reads no requests until 22 s; the off switch at 20.1 s used to
    leave the request on disk, and the controller started at 22.16 s."""
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    controller.at(20.1, (session.log / "supervise-off").touch)
    controller.at(22.0, lambda: setattr(controller, "reads_requests", True))
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and controller.consumed == []
    assert off_reasons(events) == ["The off switch log/supervise-off exists."]
    assert of(events, "resume_withdrawn") == [{"event": "resume_withdrawn", "elapsed_s": 20.25, "removed": True}]
    assert not [event for event in events if event["event"] in ("resume_confirmed", "resume_unconfirmed",
                                                                 "resume_not_consumed")]
    assert session.files() == ["supervise-off"] and (session.guard.created, session.guard.deleted) == (1, 1)
    assert events[-1]["own_request_withdrawn"] is False
    assert ("auto-resume off for the rest of this session: The off switch log/supervise-off exists. "
            "Its own Resume request, not taken yet, was withdrawn." in capsys.readouterr().out.splitlines())


def test_a_pause_seen_before_the_controller_took_the_request_withdraws_it(session):
    """F1 is published while the tool's request still waits: the request must not start a start out of that pause."""
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    controller.at(20.1, controller.press_f1)
    controller.at(22.0, lambda: setattr(controller, "reads_requests", True))
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and controller.consumed == [] and session.files() == []
    assert off_reasons(events) == ["full pause (user, or the controller after losing the map or capture)"]
    assert [event["removed"] for event in of(events, "resume_withdrawn")] == [True]
    assert of(events, "sample")[-1]["status"]["full_pause"] is True


def test_a_request_that_somebody_else_touched_is_not_withdrawn_at_a_switch_off(session):
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    path = session.log / "resume-active"
    controller.at(20.05, lambda: os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000)))
    controller.at(20.1, (session.log / "supervise-off").touch)
    events = session.run(*AUTO, minutes=0.6)
    assert [event["removed"] for event in of(events, "resume_withdrawn")] == [False]
    assert session.files() == ["resume-active", "supervise-off"] and session.guard.deleted == 0
    assert of(events, "resume_not_consumed")[0]["removed"] is False


def test_the_session_limit_does_not_withdraw_the_request_that_reached_it(session):
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    controller.at(20.4, lambda: setattr(controller, "reads_requests", True))   # slower than one poll
    events = session.run(*AUTO, "--max-resumes", "1", minutes=0.6)
    assert times(events) == [20.0] and len(controller.consumed) == 1 and 20.4 <= controller.consumed[0] <= 20.6
    assert len(of(events, "resume_confirmed")) == 1 and of(events, "resume_withdrawn") == []
    assert off_reasons(events) == ["The session limit of 1 Resume request(s) is used up."]
    assert times(events, "auto_resume_off")[0] >= times(events, "resume_confirmed")[0]
    assert session.guard.deleted == 0


def test_a_long_wait_in_a_route_hold_is_printed_once_per_reason(session, capsys):
    """Finding 4 of the second review: claude-circuits-4.jsonl ended in a route hold at 70.19 % HP, above the potion
    trigger (70) and below --min-hp 75. The tool waits; an unattended run must show why."""
    low = recorded("route_hold", hp=70.1923076923077)
    session.control((low, 30.0), (recorded("route_hold", foreground=False), 4.0), (low, None))   # HP is checked first
    events = session.run(*AUTO, minutes=1.5)
    assert times(events) == [] and off_reasons(events) == []
    waiting = [line for line in capsys.readouterr().out.splitlines() if line.startswith("auto-resume waiting")]
    assert waiting == ["auto-resume waiting (route hold for 20 s): HP must read 75% to 100%.",
                       "auto-resume waiting (route hold for 30 s): The game is not in front."]


def test_each_new_hold_prints_its_reasons_again(session, capsys):
    low = recorded("route_hold", hp=70.1923076923077)
    session.control((low, 25.0), (FARMING, 5.0), (low, None))
    events = session.run(*AUTO, minutes=1.0)
    assert times(events) == [] and off_reasons(events) == []
    waiting = [line for line in capsys.readouterr().out.splitlines() if line.startswith("auto-resume waiting")]
    assert waiting == ["auto-resume waiting (route hold for 20 s): HP must read 75% to 100%."] * 2


def test_a_tools_start_that_fails_before_it_was_seen_is_not_retried(session):
    """Harness E5 of the second review: the start fails within one status interval of taking the request."""
    controller = session.control((ROUTE_HOLD, None))

    def request_start():
        controller.before_start = dict(controller.state)
        controller.state.update(failed_start("Stable source unavailable: position kept moving"))
        controller.plan = []

    controller.request_start = request_start
    events = session.run(*AUTO, minutes=2.6)
    assert times(events) == [20.0] and len(controller.consumed) == 1
    assert off_reasons(events) == ["A start that this tool cannot confirm as its own has failed: "
                                   "Stable source unavailable: position kept moving"]


@pytest.mark.parametrize("action, reason", [
    ("press_f1", "full pause (user, or the controller after losing the map or capture)"),
    ("click_pause_all", "full pause (user, or the controller after losing the map or capture)"),
    ("click_hold", "manual hold (Hold farming): " + USER_HOLD)])
@pytest.mark.parametrize("phase", ["checking", "returning"])
def test_a_pause_or_hold_during_the_tools_own_start_ends_resuming(session, action, reason, phase):
    controller = session.control((ROUTE_HOLD, None))
    controller.when(lambda c: c.state["startup_return_state"] == phase and c.now >= {"checking": 20.0, "returning": 23.0}[phase],
                    getattr(controller, action))
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and off_reasons(events) == [reason]
    assert len(controller.consumed) == 1 and session.files() == [] and session.guard.deleted == 0
    last = of(events, "sample")[-1]["status"]
    assert last["recovery_terminal"] is True and last["startup_return_state"] == "cancelled"
    assert last["manual_health_hold"] is (action == "click_hold")


@pytest.mark.parametrize("action, reason", [
    ("press_f1", "full pause (user, or the controller after losing the map or capture)"),
    ("click_pause_all", "full pause (user, or the controller after losing the map or capture)"),
    ("click_hold", "manual hold (Hold farming): " + USER_HOLD)])
def test_a_pause_or_hold_during_the_controllers_own_return_ends_resuming(session, action, reason):
    """returning_to_start stays true under the pause; this used to be reported as 'starting' for ever."""
    controller = session.control((FARMING, 5.0), (recorded("automatic_return"), 25.0), (ROUTE_HOLD, None))
    controller.at(8.0, getattr(controller, action))
    events = session.run(*AUTO, minutes=1.0)
    assert times(events) == [] and off_reasons(events) == [reason] and session.files() == []
    assert 8.0 < times(events, "auto_resume_off")[0] <= 8.6
    assert of(events, "sample")[-1]["status"]["returning_to_start"] is True


def test_hold_farming_during_the_tools_own_start_is_never_resumed_again(session):
    """The --watcher-hold defect: this hold (manual, recovery_terminal true, start cancelled) was resumed again
    120 s after the tool's first request. The option is gone and the hold is the user's."""
    controller = session.control((ROUTE_HOLD, None))
    controller.when(lambda c: c.state["startup_return_state"] == "returning" and c.now >= 23.0, controller.click_hold)
    events = session.run(*AUTO, minutes=2.6)
    assert times(events) == [20.0] and off_reasons(events) == ["manual hold (Hold farming): " + USER_HOLD]
    held = [event["status"] for event in of(events, "sample") if event["elapsed_s"] > 30.0]
    assert held and all(status == dict(HOLD_DURING_START, pid=status["pid"], updated=status["updated"],
                                       health_protection_state="monitoring") for status in held)
    assert session.files() == [] and session.guard.created == 1


def test_the_watchers_hold_request_during_farming_ends_resuming(session):
    controller = session.control((FARMING, None))
    controller.at(6.0, controller.click_hold)
    events = session.run(*AUTO, minutes=0.6)
    assert off_reasons(events) == ["manual hold (Hold farming): " + USER_HOLD] and times(events) == []


def test_a_connection_test_request_wins_over_the_tools_request(session):
    """run_active.py:439-463 deletes the Resume request unread and pauses."""
    controller = session.control((ROUTE_HOLD, None))
    session.before_create = lambda: (session.log / "editor-test-request.json").write_text("{}", encoding="utf-8")
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and controller.consumed == [] and session.files() == []
    assert times(events, "resume_unconfirmed") == [25.0] and session.guard.deleted == 0


# -- what happens to the tool's own request ----------------------------------------------------------------------------
def test_a_request_the_controller_does_not_take_is_removed_after_5_s(session, capsys):
    session.control((ROUTE_HOLD, None), reads_requests=False)
    events = session.run(*AUTO, minutes=1.0)
    assert times(events) == [20.0] and times(events, "resume_not_consumed") == [25.0]
    assert of(events, "resume_not_consumed")[0]["removed"] is True
    reason = "The controller did not take resume-active within 5 s; the request was removed again."
    assert off_reasons(events) == [reason] and times(events, "auto_resume_off") == [25.0]
    assert session.files() == [] and (session.guard.created, session.guard.deleted) == (1, 1)
    assert of(events, "resume_waiting")[-1]["reason"] == "Another request is pending: resume-active"
    assert events[-1]["own_request_withdrawn"] is False and events[-1]["resumes"] == 1
    out = capsys.readouterr().out.splitlines()
    assert "WARNING: " + reason in out and "auto-resume off for the rest of this session: " + reason in out


def test_a_request_that_somebody_else_touched_is_not_the_tools_any_more(session):
    """The settings app's Resume touches the same file: the tool must not delete it."""
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    path = session.log / "resume-active"
    controller.at(22.0, lambda: os.utime(path, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000)))
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and of(events, "resume_not_consumed")[0]["removed"] is False
    assert off_reasons(events) == ["The controller did not take resume-active within 5 s; the file was left alone."]
    assert session.files() == ["resume-active"] and session.guard.deleted == 0
    assert events[-1]["own_request_withdrawn"] is False


def test_a_request_taken_without_a_visible_start_ends_resuming(session):
    controller = session.control((ROUTE_HOLD, None), reads_requests=False)
    controller.at(20.1, (session.log / "resume-active").unlink)             # taken, and nothing happens
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [20.0] and times(events, "resume_unconfirmed") == [25.0]
    assert off_reasons(events) == ["resume-active was taken, but no start became visible within 5 s."]
    assert session.guard.deleted == 0 and session.files() == []


@pytest.mark.parametrize("minutes, reason", [(20.4 / 60, "time limit"), (1.0, "interrupted")])
def test_a_request_that_is_still_open_when_the_tool_ends_is_withdrawn(session, minutes, reason, capsys):
    session.control((ROUTE_HOLD, None), reads_requests=False)
    session.interrupt_at = 22.0                                             # Ctrl+C, unless the time limit comes first
    events = session.run(*AUTO, minutes=minutes)
    assert times(events) == [20.0] and events[-1]["reason"] == reason
    assert events[-1]["own_request_withdrawn"] is True and session.files() == []
    assert (session.guard.created, session.guard.deleted) == (1, 1)
    assert capsys.readouterr().out.splitlines()[-1].startswith(f"supervisor finished: {reason};")


def test_ctrl_c_ends_the_session_cleanly_also_without_a_request(session):
    session.control((FARMING, None))
    session.interrupt_at = 3.0
    events = session.run(minutes=1.0)
    assert events[-1]["reason"] == "interrupted" and 3.0 <= session.clock.t <= 3.25


def test_stop_wins_over_the_tools_request_which_is_withdrawn(session):
    """Stop does not consume resume-active (run_active.py:403-415); left behind it would start the next controller."""
    controller = session.control((ROUTE_HOLD, None))
    session.before_create = lambda: (session.log / "stop-active").touch()
    events = session.run(*AUTO, minutes=1.0)
    assert times(events) == [20.0] and controller.consumed == []
    assert events[-1]["reason"] == "controller stopped" and events[-1]["own_request_withdrawn"] is True
    assert of(events, "sample")[-1]["status"]["running"] is False
    assert session.files() == ["stop-active"] and session.clock.t <= 20.5


def test_another_resume_request_that_appears_first_is_not_adopted(session):
    """Created by somebody else after the tool's last look: the tool counts nothing and never deletes that file."""
    session.control((ROUTE_HOLD, None), reads_requests=False)
    session.before_create = lambda: (session.log / "resume-active").write_text("theirs", encoding="utf-8")
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and off_reasons(events) == [] and events[-1]["resumes"] == 0
    assert [event["reason"] for event in of(events, "resume_waiting")][-2:] == [
        "Another Resume request appeared first.", "Another request is pending: resume-active"]
    assert (session.log / "resume-active").read_text(encoding="utf-8") == "theirs"
    assert (session.guard.created, session.guard.deleted) == (0, 0)


def test_a_request_file_that_cannot_be_created_ends_resuming(session):
    session.control((ROUTE_HOLD, None))

    def denied():
        raise PermissionError(13, "Access is denied")

    session.before_create = denied
    events = session.run(*AUTO, minutes=0.6)
    assert times(events) == [] and session.files() == [] and events[-1]["resumes"] == 0
    assert off_reasons(events) == ["resume-active could not be created: [Errno 13] Access is denied"]


# -- when the session ends ---------------------------------------------------------------------------------------------
def test_it_ends_at_the_time_limit(session):
    session.control((FARMING, None))
    events = session.run(minutes=1.0)
    assert events[-1]["reason"] == "time limit" and session.clock.t == 60.0
    assert 195 <= events[-1]["rows"] <= 215 and session.sleeps == 240


def test_it_ends_when_the_controller_reports_that_it_stopped(session):
    controller = session.control((ROUTE_HOLD, None))
    controller.at(10.0, (session.log / "stop-active").touch)
    events = session.run(*AUTO, minutes=1.0)
    assert events[-1]["reason"] == "controller stopped" and 10.0 < session.clock.t <= 10.5
    assert of(events, "sample")[-1]["status"]["running"] is False and times(events) == []
    assert session.files() == ["stop-active"]


def test_it_ends_when_the_controller_was_restarted(session):
    controller = session.control((ROUTE_HOLD, None))
    controller.at(10.0, lambda: setattr(controller, "pid", 4242))
    events = session.run(*AUTO, minutes=1.0)
    assert events[-1]["reason"] == "controller restarted" and 10.0 < session.clock.t <= 10.6
    assert of(events, "sample")[-1]["status"]["pid"] == 4242 and times(events) == [] and session.files() == []


@pytest.mark.parametrize("content", [None, '{"running": tr', "[1, 2]", '{"running": true}', '{"running": true, "updated": "x"}'],
                         ids=["missing", "half written", "not an object", "no timestamp", "bad timestamp"])
def test_it_ends_after_30_s_without_a_readable_status(session, content):
    if content is not None:
        (session.log / "active-status.json").write_text(content, encoding="utf-8")
    events = session.run(*AUTO, minutes=10)
    assert events[-1]["reason"] == "no new status for 30 s" and events[-1]["rows"] == 0
    assert 30.0 < session.clock.t <= 30.11 and 600 <= session.sleeps <= 603       # it retries 20 times a second
    assert [event["event"] for event in events] == ["supervisor_started", "supervisor_finished"]


def test_it_ends_30_s_after_the_last_new_status(session):
    controller = session.control((FARMING, None))
    controller.frozen_from = 40.0
    events = session.run(minutes=10)
    last = of(events, "sample")[-1]["elapsed_s"]
    assert events[-1]["reason"] == "no new status for 30 s" and 39.5 <= last <= 40.0
    assert last + 30.0 < session.clock.t <= last + 30.5


def test_a_publisher_slower_than_the_poll_does_not_end_it(session):
    controller = session.control((FARMING, None))
    controller.TURNS = (0.6,)
    events = session.run(minutes=1.5)
    assert events[-1]["reason"] == "time limit" and 148 <= events[-1]["rows"] <= 151


def test_a_status_file_that_is_unreadable_for_a_while_is_followed_by_normal_samples(session):
    controller = session.control((ROUTE_HOLD, None))
    controller.garbage_until = 5.0
    events = session.run(*AUTO, minutes=0.6)
    samples = of(events, "sample")
    assert 5.0 <= samples[0]["elapsed_s"] <= 5.4 and events[-1]["reason"] == "time limit"
    assert times(events) == [pytest.approx(samples[0]["elapsed_s"] + 20.0, abs=1e-6)]


def test_a_single_unreadable_read_inside_the_5_s_changes_nothing(session):
    """The controller replaces the file several times a second; one failed read is not a stale status."""
    controller = session.control((ROUTE_HOLD, None))
    path = session.log / "active-status.json"
    controller.at(18.2, lambda: path.write_text('{"running": tr', encoding="utf-8"))
    requested = times(session.run(*AUTO, minutes=0.4))
    assert len(requested) == 1 and 20.0 <= requested[0] <= 20.25            # a few short retries, no restart


# -- the output file -----------------------------------------------------------------------------------------------------
def test_an_existing_output_file_is_refused_and_left_alone(session, capsys):
    session.control((ROUTE_HOLD, None))
    session.output.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        session.run(*AUTO, minutes=0.1)
    assert error.value.code == 2 and session.output.read_text(encoding="utf-8") == "keep me\n"
    assert session.sleeps == 0 and session.files() == []
    assert "an existing file is never overwritten" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["resume-active", "pause-active", "stop-active", "arm-active", "session.json",
                                  "session.jsonl.tmp", "supervise-off", "active-status.json"])
def test_the_output_can_never_be_a_request_file(session, name):
    """--output log/resume-active would otherwise create the Resume request itself."""
    session.control((ROUTE_HOLD, None))
    session.output = session.log / name
    with pytest.raises(SystemExit):
        session.run(*AUTO, minutes=0.1)
    assert session.files() == [] and session.sleeps == 0


def test_the_recording_may_live_in_the_log_folder(session):
    session.control((FARMING, None))
    session.output = session.log / "Session-1.JSONL"
    events = session.run(minutes=0.05)
    assert events[-1]["reason"] == "time limit" and session.files() == ["Session-1.JSONL"]


def test_an_output_folder_that_does_not_exist_is_an_argument_error(session):
    session.output = session.root / "missing" / "session.jsonl"
    with pytest.raises(SystemExit):
        session.run(minutes=0.1)
    assert session.sleeps == 0


# ---------------------------------------------------------------------------------------------------------------
# 5. arguments and static rules
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arguments", [
    ["--minutes", "0"], ["--minutes", "-1"], ["--minutes", "600.5"], ["--minutes", "nan"], ["--minutes", "inf"],
    ["--minutes", "5", "--auto-resume", "7"], ["--minutes", "5", "--auto-resume", "-1"],
    ["--minutes", "5", "--auto-resume", "1.5"], ["--minutes", "5", "--max-resumes", "31"],
    ["--minutes", "5", "--max-resumes", "-1"], ["--minutes", "5", "--min-hp", "0"],
    ["--minutes", "5", "--min-hp", "0.99"], ["--minutes", "5", "--min-hp", "100.5"], ["--minutes", "5", "--min-hp", "-5"],
    ["--minutes", "5", "--min-hp", "nan"], ["--minutes", "5", "--min-hp", "inf"],
    ["--minutes", "5", "--watcher-hold"], []])
def test_bad_arguments_fail_at_once_and_never_reach_the_loop(monkeypatch, tmp_path, arguments):
    def entered(_):
        raise AssertionError("main() accepted the arguments and entered its loop")

    monkeypatch.setattr(supervisor.time, "sleep", entered)
    with pytest.raises(SystemExit) as error:
        supervisor.main(["--root", str(tmp_path), "--output", str(tmp_path / "x.jsonl"), *arguments])
    assert error.value.code == 2 and not (tmp_path / "x.jsonl").exists()


@pytest.mark.parametrize("arguments", [
    ["--minutes", "600"], ["--minutes", "0.01", "--auto-resume", "6", "--max-resumes", "30", "--min-hp", "1"],
    ["--minutes", "0.01", "--auto-resume", "0", "--max-resumes", "0", "--min-hp", "100"]])
def test_the_bounds_themselves_are_accepted(monkeypatch, tmp_path, arguments):
    class Entered(Exception):
        pass

    def entered(_):
        raise Entered

    monkeypatch.setattr(supervisor.time, "sleep", entered)
    with pytest.raises(Entered):
        supervisor.main(["--root", str(tmp_path), "--output", str(tmp_path / "x.jsonl"), *arguments])


def test_the_defaults_are_no_resume_twelve_per_session_and_75_percent(session):
    session.control((FARMING, None))
    started = session.run(minutes=0.01)[0]
    assert (started["auto_resume_per_hour"], started["max_resumes"], started["min_hp"]) == (0, 12, 75.0)
    policy = supervisor.ResumePolicy(per_hour=4, max_total=12)
    assert (policy.min_hp, policy.settle_s, policy.window_s, policy.window_samples, policy.min_interval_s,
            policy.confirm_s) == (75.0, 20.0, 5.0, 3, 120.0, 5.0)
    assert (supervisor.POLL_S, supervisor.RETRY_S, supervisor.NO_STATUS_S, supervisor.ACT_AGE_S) == (0.25, 0.05, 30.0,
                                                                                                     0.35)


def test_the_watcher_hold_option_is_gone():
    assert list(inspect.signature(supervisor.classify_hold).parameters) == ["status"]
    assert "watcher" not in inspect.getsource(supervisor).lower()


def test_the_module_can_neither_send_keys_nor_start_anything():
    """Only these standard modules: no subprocess, ctypes, keyboard library, and nothing of the controller."""
    tree = ast.parse(inspect.getsource(supervisor))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported == {"__future__", "argparse", "json", "math", "pathlib", "time"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names & {"__import__", "exec", "eval", "importlib", "os", "subprocess"}


def test_the_docstring_says_what_the_design_requires():
    text = " ".join(supervisor.__doc__.split())
    for needed in ("log/supervise-off", "Hold farming cannot stop this tool",
                   "answers \"already held\" without writing a request", "end the tool with Ctrl+C",
                   "they switch HP protection off", "not what the settings app's Resume button does",
                   "arm-active", "offers no direct Resume at all", "tests/test_startup_return.py",
                   "has not been verified in the live game", "is the only file this tool ever deletes",
                   "The controller can publish faster than that, so some statuses are never read or recorded.",
                   "at most 0.35 s old", "What it cannot see:", "resuming stays on",
                   "a connection test, once the controller has been active",
                   "withdraws its own request if the controller has not taken it yet",
                   "delete it yourself before a later session", "Neither stops a start the controller has already taken"):
        assert needed in text, needed
    assert "exactly what" not in text and "writes every new status" not in text and "in that moment" not in text
    assert f"at most {supervisor.ACT_AGE_S:g} s old" in text
