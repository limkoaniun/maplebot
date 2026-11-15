import pytest

from maplebot.health_protection import HealthProtection


def observe(policy, hp, now, taps=0, tap_time=None, **kw):
    return policy.observe(hp, now, taps=taps, tap_time=tap_time, **kw)


def test_initial_threshold_monitoring_then_refills_to_target_after_episode_begins():
    hp = HealthProtection({})
    assert (hp.minimum_hp_percent, hp.threshold, hp.target) == (50, 60, 65)
    assert not observe(hp, 60, 0) and hp.state == "monitoring"
    assert observe(hp, 49, .1) and hp.state == "healing"
    assert not observe(hp, 61, .3, 1, .2)
    assert hp.state == "healing" and hp.unconfirmed_taps == 0
    assert observe(hp, 61, 1.3, 1, .2)
    assert not observe(hp, 65, 1.5, 2, 1.4)
    assert hp.state == "monitoring" and not hp.latched
    assert not observe(hp, 62, 2, 2, 1.4)  # no new episode above threshold


def test_target_is_capped_at_one_hundred():
    hp = HealthProtection({"hp_percent": 98, "minimum_hp_percent": 80, "rearm_delta": 5})
    assert hp.target == 100
    assert observe(hp, 97, 0)
    assert not observe(hp, 99, .2, 1, .1) and hp.state != "monitoring"
    assert not observe(hp, 100, .3, 1, .1) and hp.state == "monitoring"


def test_unsent_queue_requeues_without_counting_failures_then_latches_at_three_seconds():
    hp = HealthProtection({"protection_retry_s": .5})
    assert observe(hp, 40, 0)
    assert not observe(hp, 39, .5)
    assert observe(hp, 38, 1)
    assert observe(hp, 37, 2)
    assert hp.taps_sent == 0 and hp.unconfirmed_taps == 0
    assert not observe(hp, 36, 3)
    assert hp.latched and hp.state == "needs_attention" and hp.reason == "HP key was not sent"


def test_three_actual_failed_taps_get_response_windows_then_stop_requests():
    hp = HealthProtection({})
    assert observe(hp, 45, 0)
    assert not observe(hp, 44, .2, 1, .1)
    assert not observe(hp, 43, 1.0, 1, .1)
    assert observe(hp, 42, 1.2, 1, .1)
    assert not observe(hp, 41, 1.4, 2, 1.3)
    assert observe(hp, 40, 2.4, 2, 1.3)
    assert not observe(hp, 39, 2.6, 3, 2.5)
    assert hp.state == "awaiting_response" and not hp.latched
    assert not observe(hp, 38, 3.4, 3, 2.5)
    assert not hp.latched
    assert not observe(hp, 37, 3.6, 3, 2.5)
    assert hp.latched and hp.state == "needs_attention" and hp.taps_sent == 3
    assert not observe(hp, 35, 10, 3, 2.5)


def test_observed_gain_resets_failed_tap_budget_and_last_potion_can_succeed():
    hp = HealthProtection({"protection_max_taps": 1})
    assert observe(hp, 40, 0)
    observe(hp, 40, .2, 1, .1)
    assert not observe(hp, 42, .9, 1, .1)
    assert hp.unconfirmed_taps == 0 and not hp.latched
    assert observe(hp, 42, 1.2, 1, .1)
    assert not observe(hp, 65, 1.4, 2, 1.3)
    assert hp.state == "monitoring" and hp.taps_sent == 2


def test_one_percent_change_does_not_claim_potion_improvement():
    hp = HealthProtection({"protection_max_taps": 1})
    observe(hp, 40, 0)
    observe(hp, 41, .2, 1, .1)
    assert hp.unconfirmed_taps == 1
    assert not observe(hp, 41, 1.2, 1, .1) and hp.latched


def test_confirmation_uses_actual_tap_time_and_cumulative_counter():
    hp = HealthProtection({})
    assert observe(hp, 40, 10, 200)  # old presses before policy start are not current failures
    assert observe(hp, 40, 11.2, 201, 10.1)  # response interval already elapsed
    assert hp.taps_sent == 1 and hp.unconfirmed_taps == 1
    assert not observe(hp, 40, 11.3, 201, 10.1)
    assert hp.unconfirmed_taps == 1  # rereading a counter never creates another attempt


def test_pause_keeps_failure_budget_and_pending_response_without_aging_it():
    hp = HealthProtection({"protection_max_taps": 2})
    observe(hp, 45, 0)
    observe(hp, 44, .2, 1, .1)
    hp.suspend()
    assert hp.state == "suspended" and hp.unconfirmed_taps == 1
    assert not observe(hp, 35, 100, 1, .1)  # damage while paused does not erase the earlier failure
    assert not hp.latched and hp.unconfirmed_taps == 1
    assert observe(hp, 34, 101, 1, .1)
    assert not observe(hp, 33, 101.2, 2, 101.1)
    assert not observe(hp, 32, 102.2, 2, 101.1)
    assert hp.latched and hp.taps_sent == 2
    hp.suspend()
    assert hp.latched and hp.state == "needs_attention"


def test_pause_preserves_unsent_queue_timeout_and_unobserved_actual_tap_evidence():
    hp = HealthProtection({})
    observe(hp, 40, 0)
    observe(hp, 40, .5)
    hp.suspend()
    assert not observe(hp, 40, 100, 1, .6)
    assert hp.unconfirmed_taps == 1 and hp.state == "awaiting_response" and not hp.latched
    assert observe(hp, 40, 101.1, 1, .6)
    hp.suspend()
    assert not observe(hp, 40, 200, 1, .6)
    assert not hp.latched


@pytest.mark.parametrize("reading", [None, float("nan"), float("inf"), -1, 101, True])
def test_invalid_hp_latches_without_requesting(reading):
    hp = HealthProtection({})
    assert not observe(hp, reading, 0)
    assert hp.latched and hp.state == "needs_attention"


def test_missing_key_and_death_latch_and_death_is_not_downgraded_by_missing_reading():
    hp = HealthProtection({})
    assert not observe(hp, 50, 0, key_available=False)
    assert hp.latched and hp.state == "needs_attention"
    assert not observe(hp, 0, 1, key_available=False)
    assert hp.state == "dead"
    hp.suspend()
    assert not observe(hp, None, 100) and hp.state == "dead"
    assert not observe(hp, 100, 101) and hp.state == "dead"
    hp.request_resume()
    assert not observe(hp, 65, 102) and hp.state == "monitoring" and not hp.latched


def test_explicit_rearm_needs_next_fresh_reading_at_target_and_does_not_remain_armed():
    hp = HealthProtection({})
    observe(hp, None, 0)
    assert not observe(hp, 100, 1) and hp.latched
    hp.request_resume()
    assert hp.latched  # the button itself cannot claim recovery
    assert not observe(hp, 64, 2) and hp.latched
    assert not observe(hp, 100, 3) and hp.latched
    hp.request_resume()
    assert not observe(hp, 65, 4) and not hp.latched
    assert hp.state == "monitoring"


def test_disabled_protection_never_requests_or_latches():
    hp = HealthProtection({"enable": False})
    assert not observe(hp, 0, 0, key_available=False)
    assert hp.state == "suspended" and not hp.latched


@pytest.mark.parametrize("cfg", [
    {"minimum_hp_percent": 60}, {"minimum_hp_percent": -1}, {"hp_percent": 101},
    {"hp_percent": float("nan")}, {"rearm_delta": -1}, {"rearm_delta": float("inf")},
    {"protection_retry_s": .49}, {"protection_retry_s": 10.1},
    {"protection_retry_s": float("nan")}, {"protection_max_taps": 0},
    {"protection_max_taps": 6}, {"protection_max_taps": 1.5}, {"protection_max_taps": True},
])
def test_invalid_protection_settings_are_rejected(cfg):
    with pytest.raises(ValueError):
        HealthProtection(cfg)


@pytest.mark.parametrize("retry", [.5, 10])
def test_retry_interval_bounds_are_accepted(retry):
    hp = HealthProtection({"protection_retry_s": retry})
    assert hp.retry_s == retry
    assert observe(hp, 40, 0)
    assert not observe(hp, 40, .2, 1, .1)
    assert not observe(hp, 40, retry, 1, .1)
    assert observe(hp, 40, retry + .2, 1, .1)
