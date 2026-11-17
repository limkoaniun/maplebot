"""Delivery of short leased movement pulses (run 7: ~25% of 40 ms fine taps never went down).

All timing runs on fake clocks; blocking taps advance the fake lease clock
instead of sleeping, and the threaded cases are sequenced with events.
"""
import random
import threading

import pytest

from maplebot.config import deep_merge, load_config
from maplebot.control.keyboard import TAP_S, KeyboardController
from maplebot.types import Command

PERIOD_S = 1.0 / 30  # default system.fps_keyboard
PULSE_S = 0.04  # circuit fine-alignment pulse (route.circuit.approach_pulse_s on the live map)


class TimedBackend:
    def __init__(self, wall):
        self.wall = wall
        self.events: list[tuple[str, str, float]] = []
        self.held: set[str] = set()

    def down(self, key):
        self.events.append(("down", key, self.wall[0]))
        self.held.add(key)

    def up(self, key):
        self.events.append(("up", key, self.wall[0]))
        self.held.discard(key)

    def downs(self, key):
        return [e[2] for e in self.events if e[0] == "down" and e[1] == key]

    def holds(self, key):
        out, start = [], None
        for kind, k, t in self.events:
            if k != key:
                continue
            if kind == "down":
                start = t
            elif start is not None:
                out.append(t - start)
                start = None
        return out


def _keyboard(start=10.0, **over):
    cfg = deep_merge(load_config(), deep_merge({"keys": {"pickup": "x", "pickup_interval_s": 0.15}}, over))
    wall = [start]
    backend = TimedBackend(wall)

    def fake_sleep(seconds):  # a blocking tap: the lease clock moves, nothing really sleeps
        wall[0] += seconds

    kb = KeyboardController(cfg, backend, clock=lambda: wall[0], sleep=fake_sleep)
    return kb, backend, wall


def _tick(kb, wall, at):
    wall[0] = max(wall[0], at)
    kb.apply_once(wall[0])


def _assert_counters_add_up(kb):
    s = kb.pulse_stats()
    for part in (s, s.get("fine", s)):
        assert part["granted"] == (part["pressed"] + part["expired_before_press"]
                                   + part["superseded_before_press"] + part["pending"])
        assert part["holds"] == sum(part["hold_histogram_ms"].values())
    return s


class _WatchedLock:
    """Re-entrant tick lock that reports when a watched thread starts waiting for it."""

    def __init__(self):
        self._inner = threading.RLock()
        self.watch_ident = None
        self.contended = threading.Event()

    def __enter__(self):
        if threading.get_ident() == self.watch_ident:
            self.contended.set()
        self._inner.acquire()
        return self

    def __exit__(self, *exc):
        self._inner.release()
        return False


def _grant_during_pickup_tap(kb, wall, command, *, grant_at, source_time):
    """Run one worker tick whose pickup tap holds the tick lock while ``command`` is sent.

    Mirrors the live race: Bot.step calls set_command while the 30 Hz worker is
    inside the 50 ms X tap.  Returns the lease-clock time the tick ended.
    """
    lock = _WatchedLock()
    kb._tick_lock = lock
    tap_started, tap_continue = threading.Event(), threading.Event()

    def blocking_sleep(seconds):
        tap_started.set()
        assert tap_continue.wait(2)
        wall[0] += seconds

    kb.sleep = blocking_sleep
    tap_start = wall[0]
    worker = threading.Thread(target=lambda: kb.apply_once(wall[0]))
    worker.start()
    assert tap_started.wait(2)
    wall[0] = grant_at

    def send():
        lock.watch_ident = threading.get_ident()
        kb.set_command(command, source_time=source_time)

    setter = threading.Thread(target=send)
    setter.start()
    try:
        assert lock.contended.wait(2)  # the setter read its lease clock and waits on the tick
        wall[0] = tap_start  # the tap's own sleep advances from its start
    finally:
        tap_continue.set()
        worker.join(2)
        setter.join(2)
    assert not worker.is_alive() and not setter.is_alive()
    kb.sleep = lambda seconds: wall.__setitem__(0, wall[0] + seconds)
    return wall[0]


# -- lost delivery (fails on the pre-fix keyboard) -----------------------------

def test_pulse_granted_while_pickup_tap_holds_the_lock_is_pressed_after_the_tap():
    kb, be, wall = _keyboard()
    kb.set_command(Command.NONE, source_time=wall[0])  # circuit release frame: pickup active
    end = _grant_during_pickup_tap(kb, wall, Command("left", valid_for_s=PULSE_S),
                                   grant_at=10.01, source_time=10.0)
    assert be.downs("x") == [10.0] and end == pytest.approx(10.05)
    # The tick overran its period, so the worker ticks again at once.
    _tick(kb, wall, end)
    assert be.downs("left") == [pytest.approx(10.05)]
    assert kb._command_deadline == pytest.approx(10.05 + PULSE_S)  # its own length from key-down
    _tick(kb, wall, end + PERIOD_S)
    assert be.held == {"left"}
    _tick(kb, wall, end + 2 * PERIOD_S)
    assert not be.held
    assert be.holds("left") == [pytest.approx(2 * PERIOD_S)]


def test_pulse_queued_behind_an_hp_tap_tick_is_pressed_on_the_next_tick():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])
    kb.request("hp_potion")
    _tick(kb, wall, 10.0)  # HP request has priority and blocks 50 ms
    assert be.downs("1") == [10.0] and kb.hp_taps_sent == 1 and not be.downs("left")
    _tick(kb, wall, wall[0])
    assert be.downs("left") == [pytest.approx(10.05)]
    assert kb._command_deadline == pytest.approx(10.09)


def test_pulse_survives_a_late_worker_tick():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("right", valid_for_s=PULSE_S), source_time=wall[0])
    _tick(kb, wall, 10.045)  # e.g. a Windows sleep overshooting a 33 ms period
    assert be.downs("right") == [pytest.approx(10.045)]
    _tick(kb, wall, 10.08)
    assert be.held == {"right"}
    _tick(kb, wall, 10.09)
    assert not be.held and be.holds("right") == [pytest.approx(0.045)]


# -- the grant still dies with its frame ----------------------------------------

@pytest.mark.parametrize("blocker", ["hp_tap", "late_tick", "lock_wait"])
def test_pulse_whose_frame_goes_stale_before_it_could_be_pressed_is_dropped_and_counted(blocker):
    kb, be, wall = _keyboard()
    captured = 10.0 - 0.72  # 30 ms of frame age left at the grant
    if blocker == "lock_wait":
        kb.set_command(Command.NONE, source_time=wall[0])
        end = _grant_during_pickup_tap(kb, wall, Command("left", valid_for_s=PULSE_S),
                                       grant_at=10.01, source_time=captured)
        _tick(kb, wall, end)
    else:
        kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=captured)
        if blocker == "hp_tap":
            kb.request("hp_potion")
            _tick(kb, wall, 10.0)
            _tick(kb, wall, wall[0])
        else:
            _tick(kb, wall, 10.031)
    assert not be.downs("left") and not be.held and kb.cmd == Command.NONE
    s = _assert_counters_add_up(kb)
    assert (s["granted"], s["pressed"], s["expired_before_press"], s["pending"]) == (1, 0, 1, 0)
    _tick(kb, wall, wall[0] + 1)
    assert not be.downs("left")  # a spent capture never comes back


def test_pressed_pulse_is_still_released_by_its_frame_limit():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    captured = 10.0 - 0.68  # frame limit 10.07
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=captured)
    kb.request("hp_potion")
    _tick(kb, wall, 10.0)
    _tick(kb, wall, wall[0])  # key-down at 10.05; its own length would end at 10.09
    assert be.downs("left") == [pytest.approx(10.05)]
    assert kb._command_deadline == pytest.approx(10.07)
    _tick(kb, wall, 10.07)
    assert not be.held


@pytest.mark.parametrize("stamp", [10.5, float("nan")])
def test_future_or_invalid_capture_never_presses_a_pulse(stamp):
    kb, be, wall = _keyboard()
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=stamp)
    _tick(kb, wall, 10.0)
    assert not be.events
    assert _assert_counters_add_up(kb)["expired_before_press"] == 1


@pytest.mark.parametrize("loss", ["disable", "focus", "stop"])
def test_input_stop_while_pending_drops_the_pulse_without_later_replay(loss):
    kb, be, wall = _keyboard()
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])
    if loss == "disable":
        kb.set_enabled(False)
    elif loss == "focus":
        kb.is_active = lambda: False
        _tick(kb, wall, 10.0)
    else:
        kb.stop()
    kb.is_active = lambda: True
    kb.set_enabled(True)
    _tick(kb, wall, 10.01)
    assert not be.events
    s = _assert_counters_add_up(kb)
    assert (s["granted"], s["expired_before_press"], s["pending"]) == (1, 1, 0)


def test_mode_or_command_change_supersedes_a_waiting_pulse():
    kb, be, wall = _keyboard()
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])
    kb.set_command(Command("right", valid_for_s=PULSE_S), source_time=wall[0])
    kb.set_hp_only(source_time=wall[0])
    _tick(kb, wall, 10.0)
    assert not be.downs("left") and not be.downs("right")
    s = _assert_counters_add_up(kb)
    assert (s["granted"], s["superseded_before_press"], s["pending"]) == (2, 2, 0)


# -- same-capture rules ----------------------------------------------------------

def test_same_capture_repeat_while_waiting_is_one_pulse_and_never_longer():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    pulse = Command("left", valid_for_s=PULSE_S)
    kb.set_command(pulse, source_time=10.0)
    kb.request("hp_potion")
    _tick(kb, wall, 10.0)  # HP tap: pulse still waiting at 10.05
    kb.set_command(pulse.replace(valid_for_s=0.5), source_time=10.0)  # Bot.step's second send
    assert _assert_counters_add_up(kb)["granted"] == 1
    _tick(kb, wall, wall[0])
    assert kb._command_deadline == pytest.approx(10.05 + PULSE_S)
    kb.set_command(pulse.replace(valid_for_s=0.5), source_time=10.0)
    assert kb._command_deadline == pytest.approx(10.09)  # running: a repeat cannot extend it
    _tick(kb, wall, 10.09)
    assert not be.held
    kb.set_command(pulse, source_time=10.0)
    _tick(kb, wall, 10.1)
    assert be.downs("left") == [pytest.approx(10.05)]  # spent for this capture
    s = _assert_counters_add_up(kb)
    assert (s["granted"], s["pressed"], s["holds"]) == (1, 1, 1)


# -- unchanged: unleased commands, continued holds, HP taps ----------------------

def test_unleased_command_sent_during_a_pickup_tap_keeps_its_call_time_lease():
    kb, be, wall = _keyboard()
    kb.set_command(Command.NONE, source_time=wall[0])
    end = _grant_during_pickup_tap(kb, wall, Command("left"), grant_at=10.01, source_time=10.01)
    assert kb._command_deadline == pytest.approx(10.76)  # not re-anchored at the lock (10.05)
    _tick(kb, wall, end)
    assert be.downs("left") == [pytest.approx(10.05)]
    assert kb._command_deadline == pytest.approx(10.76)
    assert kb.pulse_stats()["granted"] == 0


def test_leased_renewal_of_an_already_held_key_runs_from_the_call_like_before():
    kb, be, wall = _keyboard(start=9.9)
    kb.set_command(Command("left"), source_time=wall[0])
    _tick(kb, wall, 9.9)  # left down, plus the first pickup tap
    assert be.held == {"left"}
    wall[0] = 10.05  # next pickup due
    end = _grant_during_pickup_tap(kb, wall, Command("left", valid_for_s=0.1),
                                   grant_at=10.06, source_time=10.06)
    assert be.downs("x") == [9.9, 10.05] and end == pytest.approx(10.1)
    assert kb._command_deadline == pytest.approx(10.16)  # call time, not lock acquisition
    _tick(kb, wall, end)
    assert be.downs("left") == [9.9] and kb._command_deadline == pytest.approx(10.16)
    s = _assert_counters_add_up(kb)
    assert (s["granted"], s["continued"]) == (0, 1)


def test_hp_and_mp_taps_count_actual_key_downs_and_keep_priority_over_a_waiting_pulse():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])
    kb.request("hp_potion")
    kb.request("mp_potion")
    _tick(kb, wall, 10.0)
    _tick(kb, wall, wall[0])
    _tick(kb, wall, wall[0])
    assert [e[1] for e in be.events if e[0] == "down"] == ["1", "2", "left"]
    assert kb.hp_taps_sent == 1 and kb.mp_taps_sent == 1
    kb.set_command(Command.NONE, source_time=wall[0], navigation_only=True)
    kb.request("mp_potion")  # dropped by navigation-only: not a sent tap
    _tick(kb, wall, wall[0] + 0.1)
    assert kb.mp_taps_sent == 1 and kb.pulse_stats()["mp_taps_sent"] == 1


# -- hold duration bounds ----------------------------------------------------------

def _worst_case_hold(blocker, delta, *, press_delay=False, d=PULSE_S):
    """Hold of one pulse: key-down tick, then a tick ``delta`` later carrying ``blocker``.

    Ticks never start sooner than one period after the previous tick start;
    ``delta`` may be longer (late ticks).  ``press_delay`` puts a blocking HP
    tap in front of the key-down (only the fixed keyboard still presses then).
    """
    kb, be, wall = _keyboard()
    kb.set_command(Command("left", valid_for_s=d), source_time=wall[0])
    t = wall[0]
    if press_delay:
        kb.request("hp_potion")
        _tick(kb, wall, t)
        t = wall[0]
    kb._last_pickup = t + delta - 0.1501 if blocker == "pickup" else t  # due only at the blocker tick
    _tick(kb, wall, t)  # key-down
    assert be.held == {"left"}
    if blocker == "hp":
        kb.request("hp_potion")
    tick_start = t + delta
    while be.held and tick_start < t + 1:
        start = max(wall[0], tick_start)
        _tick(kb, wall, start)
        tick_start = max(start + PERIOD_S, wall[0])
    [hold] = be.holds("left")
    return hold


def _delta_grid():
    # Tick spacing from one period up to (excluding) the pulse length: the last
    # tick that can still start a blocking tap before the pulse ends.
    steps = 40
    return [PERIOD_S + (PULSE_S - PERIOD_S) * i / steps for i in range(steps)]


@pytest.mark.parametrize("press_delay", [False, True])
@pytest.mark.parametrize("blocker", ["pickup", "hp"])
def test_worst_case_pulse_hold_is_its_length_plus_one_blocking_tap(blocker, press_delay):
    holds = [_worst_case_hold(blocker, delta, press_delay=press_delay) for delta in _delta_grid()]
    worst = max(holds)
    assert worst < PULSE_S + TAP_S + 1e-9  # 90 ms: same bound as the pre-fix keyboard
    assert worst > PULSE_S + TAP_S - 0.001  # and the bound is reached, not just stated
    assert min(holds) == pytest.approx(PERIOD_S + TAP_S)  # regular 30 Hz ticks: 83.3 ms


def test_pulse_without_a_blocking_tap_is_held_between_its_length_and_one_more_period():
    for phase in (0.0, 0.004, 0.0066):
        kb, be, wall = _keyboard(keys={"pickup": ""})
        kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])
        t = 10.0 + phase
        while not be.holds("left") and t < 11:
            _tick(kb, wall, t)
            t += PERIOD_S
        [hold] = be.holds("left")
        assert PULSE_S <= hold + 1e-9 and hold < PULSE_S + PERIOD_S
        assert kb.pulse_stats()["last_hold_s"] == pytest.approx(hold)


# -- instrumentation ------------------------------------------------------------------

def test_hold_histogram_and_press_delay_are_recorded():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])
    kb.request("hp_potion")
    _tick(kb, wall, 10.0)
    _tick(kb, wall, wall[0])  # press 50 ms after the grant
    _tick(kb, wall, 10.05 + PERIOD_S)
    _tick(kb, wall, 10.05 + 2 * PERIOD_S)
    s = _assert_counters_add_up(kb)
    assert s["last_press_delay_s"] == pytest.approx(0.05) == s["max_press_delay_s"]
    assert s["last_hold_s"] == pytest.approx(2 * PERIOD_S)
    assert s["hold_histogram_ms"] == {"le20": 0, "le40": 0, "le60": 0, "le80": 1, "le120": 0, "gt120": 0}
    # Exactly 40 ms belongs to the 40 ms bucket despite float noise.
    kb.set_command(Command("right", valid_for_s=PULSE_S), source_time=wall[0])
    _tick(kb, wall, 10.16)
    _tick(kb, wall, 10.2)
    assert kb.pulse_stats()["hold_histogram_ms"]["le40"] == 1


def test_counters_add_up_over_a_mixed_command_stream():
    rng = random.Random(7)
    kb, be, wall = _keyboard()
    source = wall[0]
    for _ in range(3000):
        op = rng.random()
        if op < 0.35:
            if rng.random() < 0.6:
                source = wall[0]
            kb.set_command(Command(rng.choice(["left", "right"]),
                                   valid_for_s=rng.choice([0.04, 0.04, 0.09, 0.3])),
                           source_time=source, navigation_only=rng.random() < 0.3)
        elif op < 0.45:
            kb.set_command(Command.NONE, source_time=wall[0])
        elif op < 0.5:
            kb.set_command(Command("left"), source_time=wall[0] - rng.random())
        elif op < 0.55:
            kb.request(rng.choice(["hp_potion", "mp_potion"]))
        elif op < 0.57:
            kb.set_enabled(rng.random() < 0.5)
        elif op < 0.58:
            kb.set_hp_only(source_time=wall[0])
        else:
            _tick(kb, wall, wall[0] + rng.choice([0.0, PERIOD_S, PERIOD_S, 0.05, 0.2]))
        _assert_counters_add_up(kb)
    s = kb.pulse_stats()
    assert s["pressed"] > 100 and s["expired_before_press"] > 0 and s["superseded_before_press"] > 0
    assert s["holds"] <= s["pressed"]
    assert len(be.downs("left")) + len(be.downs("right")) >= s["pressed"]


# -- review follow-ups: press-by grace, leftover keys, key-down accounting --------

# A waiting pulse may go down no later than its own length plus a 0.1 s grace
# (two blocking taps) after its grant; keyboard.PRESS_GRACE_S.  Written as a
# literal so this file still imports against older keyboards.
PRESS_WINDOW_S = PULSE_S + 2 * TAP_S


class _CostlyBackend(TimedBackend):
    """Each key-down takes time on the lease clock (SendInput latency)."""

    def __init__(self, wall, down_cost):
        super().__init__(wall)
        self.down_cost = down_cost

    def down(self, key):
        super().down(key)
        self.wall[0] += self.down_cost


def _run_ticks(kb, wall, until, *, first):
    """30 Hz worker: next tick one period after the previous start, or at once if it overran."""
    t = first
    while t < until:
        start = max(wall[0], t)
        _tick(kb, wall, start)
        t = max(start + PERIOD_S, wall[0])


@pytest.mark.parametrize("stall, pressed", [(PRESS_WINDOW_S - 0.001, True),
                                            (PRESS_WINDOW_S + 0.001, False),
                                            (0.2, False), (0.5, False), (0.74, False)])
def test_waiting_pulse_must_go_down_within_its_length_plus_the_press_grace(stall, pressed):
    kb, be, wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=10.0)  # frame limit 10.75
    _tick(kb, wall, 10.0 + stall)  # worker stalled; the producer sends no newer frame
    s = _assert_counters_add_up(kb)
    if pressed:
        assert be.downs("left") == [pytest.approx(10.0 + stall)]
        assert kb._command_deadline == pytest.approx(10.0 + stall + PULSE_S)
        assert (s["pressed"], s["expired_before_press"]) == (1, 0)
    else:
        assert not be.events and kb.cmd == Command.NONE
        assert (s["granted"], s["pressed"], s["expired_before_press"], s["pending"]) == (1, 0, 1, 0)
        _tick(kb, wall, wall[0] + PERIOD_S)
        assert not be.events  # dropped for good, not pressed late


def test_waiting_pulse_deadline_is_its_press_by_not_its_frame_limit():
    kb, _be, _wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=10.0)
    assert kb._command_deadline == pytest.approx(10.0 + PRESS_WINDOW_S)
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=10.0 - 0.72)
    assert kb._command_deadline == pytest.approx(10.03)  # an older frame still wins


def test_pulse_granted_during_a_pickup_tap_then_queued_behind_an_hp_tick_still_goes_down():
    kb, be, wall = _keyboard()
    kb.set_command(Command.NONE, source_time=wall[0])
    end = _grant_during_pickup_tap(kb, wall, Command("left", valid_for_s=PULSE_S),
                                   grant_at=10.01, source_time=10.0)
    kb.request("hp_potion")  # Bot.step's request lands before the worker's next tick
    _tick(kb, wall, end)  # HP tap 10.05-10.10
    assert be.downs("1") == [pytest.approx(10.05)] and not be.downs("left")
    _tick(kb, wall, wall[0])
    assert be.downs("left") == [pytest.approx(10.10)]  # 90 ms after its grant: inside the grace
    assert kb._command_deadline == pytest.approx(10.10 + PULSE_S)


def test_buff_freeze_drops_a_waiting_pulse_instead_of_pressing_it_at_the_freeze_end():
    kb, be, wall = _keyboard(keys={"pickup": ""},
                             buffs={"keys": ["a"], "cooldowns_s": [100], "freeze_s": 1.0})
    kb.set_command(Command.NONE, source_time=10.0)
    _tick(kb, wall, 10.0)  # buff cast: movement frozen until 11.0
    assert be.downs("a") == [10.0]
    wall[0] = 10.4
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=10.4)  # last fresh frame
    _run_ticks(kb, wall, 11.2, first=10.4)
    assert not be.downs("left")
    assert _assert_counters_add_up(kb)["expired_before_press"] == 1


def _leftover_hold(n_requests):
    kb, be, wall = _keyboard(keys={"pickup": ""})
    kb.set_command(Command("right", valid_for_s=PULSE_S), source_time=10.0)
    _tick(kb, wall, 10.0)  # pulse A: right down at 10.00, its own end 10.04
    assert be.held == {"right"}
    wall[0] = 10.03
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=10.02)  # next frame's pulse B
    for i in range(n_requests):
        kb.request("mp_potion" if i % 2 else "hp_potion")
    _run_ticks(kb, wall, 11.0, first=10.0 + PERIOD_S)
    return kb, be


@pytest.mark.parametrize("n_requests", [2, 4, 8])
def test_key_of_a_replaced_command_keeps_its_old_bound_through_request_ticks(n_requests):
    kb, be = _leftover_hold(n_requests)
    # Released at the first check after B's old grant deadline (10.07): the end
    # of the first request tap, 83.3 ms -- as before pulses waited for key-down.
    assert be.holds("right") == [pytest.approx(PERIOD_S + TAP_S)]
    s = _assert_counters_add_up(kb)
    if n_requests == 2:  # B still goes down after the two taps, inside its grace
        assert be.downs("left") == [pytest.approx(10.0 + PERIOD_S + 2 * TAP_S)]
        assert s["pressed"] == 2  # A and B
    else:  # more queued taps than the grace allows: B is dropped unpressed
        assert not be.downs("left") and s["expired_before_press"] == 1


def test_key_of_a_replaced_command_is_swapped_directly_on_an_ordinary_tick():
    kb, be = _leftover_hold(0)
    assert be.holds("right") == [pytest.approx(PERIOD_S)]
    assert be.downs("left") == [pytest.approx(10.0 + PERIOD_S)]


@pytest.mark.parametrize("cmd", [Command("left", valid_for_s=PULSE_S),
                                 Command("left", "up", valid_for_s=PULSE_S)],
                         ids=["one_axis", "two_axis"])
def test_a_key_that_went_down_is_counted_pressed_even_if_the_watchdog_lifts_it_at_once(cmd):
    cfg_kb, _be, _wall = _keyboard(keys={"pickup": ""})
    wall = [10.0]
    be = _CostlyBackend(wall, down_cost=0.002)
    kb = KeyboardController(cfg_kb.cfg, be, clock=lambda: wall[0],
                            sleep=lambda s: wall.__setitem__(0, wall[0] + s))
    kb.set_command(cmd, source_time=10.0 - 0.749)  # frame limit 10.001
    kb.apply_once(10.0)
    assert be.downs("left") == [10.0] and not be.held  # really down, lifted by the watchdog
    assert not be.downs("up")
    s = _assert_counters_add_up(kb)
    assert (s["pressed"], s["expired_before_press"], s["holds"]) == (1, 0, 1)
    assert s["last_hold_s"] == pytest.approx(0.002)


def test_fine_pulse_stats_are_reported_apart_from_continued_walks():
    kb, be, wall = _keyboard(keys={"pickup": ""})
    for i in range(8):  # loot non-precise walk: leased 0.3 s, renewed every 0.12 s frame
        t = 10.0 + 0.12 * i
        wall[0] = max(wall[0], t)
        kb.set_command(Command("right", valid_for_s=0.3), source_time=t)
        _run_ticks(kb, wall, t + 0.12, first=wall[0])
    kb.set_command(Command.NONE, source_time=wall[0])
    _tick(kb, wall, wall[0])
    kb.set_command(Command("left", valid_for_s=PULSE_S), source_time=wall[0])  # one fine tap
    t = wall[0]
    _run_ticks(kb, wall, t + 0.2, first=t)
    s = _assert_counters_add_up(kb)
    assert (s["granted"], s["continued"], s["holds"]) == (2, 7, 2)
    assert s["hold_histogram_ms"]["gt120"] == 1  # the walk: one long physical hold
    fine = s["fine"]
    assert (fine["granted"], fine["pressed"], fine["continued"], fine["holds"]) == (1, 1, 0, 1)
    assert fine["hold_histogram_ms"]["gt120"] == 0
    assert fine["last_hold_s"] == pytest.approx(be.holds("left")[0])
    assert s["press_grace_s"] == pytest.approx(PRESS_WINDOW_S - PULSE_S)
