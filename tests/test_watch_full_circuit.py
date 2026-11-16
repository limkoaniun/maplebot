from types import SimpleNamespace

import pytest

from scripts import watch_full_circuit as watcher


def test_terminal_route_health_hold_is_confirmed_but_temporary_refill_is_not(monkeypatch):
    monkeypatch.setattr(watcher.time, 'time', lambda: 1000.)
    status = {'running': True, 'updated': 1000., 'paused': False, 'full_pause': False,
              'health_protection_active': True, 'health_protection_state': 'monitoring',
              'recovery_terminal': True, 'circuit_state': 'blocked'}
    assert watcher.automatic_hold(status)
    assert not watcher.automatic_hold({**status, 'recovery_terminal': False, 'circuit_state': 'walking'})
    temporary = {**status, 'paused': True, 'recovery_terminal': False, 'circuit_state': 'walking'}
    assert not watcher.automatic_hold(temporary)
    assert not watcher.automatic_hold({**temporary, 'manual_health_hold': True, 'startup_return_state': 'checking'})
    assert not watcher.automatic_hold({**status, 'full_pause': True})


def test_map_reacquisition_observes_no_progress_and_still_has_two_second_limit():
    status = {'map_state': 'unknown', 'map_ready': False, 'map_reacquiring': True,
              'full_pause': False, 'capture': {}, 'capture_state': 'fresh',
              'capture_age_s': .1, 'minimap_seen': False, 'minimap': None}
    assert watcher.map_guard_error(status, {}) is None
    continuity = watcher.CaptureContinuity()
    assert not continuity.observe(status, 10)
    assert not continuity.expired(11.9)
    assert continuity.expired(12)
    assert watcher.map_guard_error({**status, 'map_reacquiring': False}, {})
    assert watcher.map_guard_error({**status, 'full_pause': True}, {})
    assert watcher.map_guard_error({**status, 'map_state': 'settings_mismatch'}, {})
    assert watcher.map_guard_error({**status, 'map_state': 'wrong_profile'}, {})
    assert watcher.map_guard_error(status, {'changed': True})


def test_start_approach_accepts_saved_upper_entry_but_rejects_air_and_missing_connection():
    from maplebot.nav.editor import compile_plan
    plan = {'version': 1, 'name': 'Entry map', 'image_size': [210, 128], 'loop': False,
            'platforms': [{'id': 'ground', 'left': 0, 'right': 100, 'y': 105},
                          {'id': 'entry', 'left': 30, 'right': 60, 'y': 88}],
            'connections': [{'id': 'drop', 'kind': 'drop', 'from_platform': 'entry',
                             'to_platform': 'ground', 'from_x': 49, 'to_x': 49}],
            'route': [{'platform_id': 'ground', 'x': 49}, {'platform_id': 'ground', 'x': 55}]}
    circuit = compile_plan(plan, {'editor_plan': plan})
    state = {'minimap_seen': True, 'minimap': [49, 88]}
    assert watcher.can_approach_start(state, circuit)
    # Between the floors with drawn ground below: the controller may first regain
    # support there, and ground has a path to point 1.
    assert watcher.can_approach_start({**state, 'minimap': [49, 90]}, circuit)
    assert not watcher.can_approach_start({**state, 'minimap': [49, 90]},
                                          {**circuit, 'return_regain_support': False})
    assert not watcher.can_approach_start({**state, 'minimap': [150, 90]}, circuit)  # nothing drawn below
    assert not watcher.can_approach_start({**state, 'minimap_seen': False}, circuit)
    plan['connections'].clear()
    disconnected = compile_plan(plan, {'editor_plan': plan})
    assert not watcher.can_approach_start(state, disconnected)
    # Above "entry": the first drawn floor underneath has no saved path to point 1.
    assert not watcher.can_approach_start({**state, 'minimap': [49, 80]}, disconnected)
    assert watcher.can_approach_start({**state, 'minimap': [49, 95]}, disconnected)  # lands on ground itself
    # The first option decides: just past the end of a platform with no saved path,
    # although the connected ground lies below as a later option.
    plan['platforms'].append({'id': 'island', 'left': 64, 'right': 80, 'y': 95})
    island = compile_plan(plan, {'editor_plan': plan})
    from maplebot.nav.regain import regain_options
    assert [o['platform_id'] for o in regain_options(island, (62, 95))] == ['island', 'ground', 'ground']
    assert not watcher.can_approach_start({**state, 'minimap': [62, 95]}, island)
    assert watcher.can_approach_start({**state, 'minimap': [62, 99]}, island)  # only ground below


@pytest.fixture
def evidence(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(watcher.time, "time", lambda: clock[0])
    circuit = {"platforms": [{"id": "ground", "left": 40, "right": 58, "y": 105}],
               "waypoints": [{"x": 49, "y": 105, "action": "jump" if i in range(1, 35, 2) else "walk",
                              "platform_id": "ground", "to_platform": "ground", "connection_id": f"edge-{i}"}
                             for i in range(44)]}
    tracker = watcher.LapEvidence(circuit)

    def sample(index, **extra):
        clock[0] += .31
        status = {"updated": clock[0], "minimap": [49, 105], "minimap_seen": True, "circuit_index": index,
                  "return_count": 0, "returning_to_start": False, "recovery_state": "idle", **extra}
        tracker.observe(status)
        return status
    return tracker, sample, clock


def test_ordered_wrap_requires_separate_transition_and_stable_landing_evidence(evidence):
    tracker, sample, _ = evidence
    for index in range(44):
        for _ in range(3):
            sample(index)
    sample(0)
    summary = tracker.summary()
    assert summary["full_wrap_observed"] and summary["stable_ground_return"]
    assert summary["all_transition_indices_observed"] and summary["all_landings_observed"]
    assert summary["observed_indices"] == list(range(44))
    assert len(summary["transitions"]) == 17


def test_launch_floor_overlapping_destination_cannot_confirm_landing(evidence):
    tracker, sample, _ = evidence
    sample(0)
    for _ in range(6):
        sample(1, circuit_state='jump_aligning')
    assert not tracker.proofs[1].confirmed
    assert not tracker.proofs[1].samples
    for _ in range(3):
        sample(2, circuit_state='walking')
    assert tracker.proofs[1].confirmed


def test_unobserved_transition_is_not_claimed_as_verified_even_after_wrap(evidence):
    tracker, sample, _ = evidence
    for index in range(44):
        if index == 3:
            continue
        for _ in range(3):
            sample(index)
    sample(0)
    summary = tracker.summary()
    assert summary["full_wrap_observed"]
    assert not summary["all_transition_indices_observed"] and not summary["all_landings_observed"]
    assert summary["unobserved_forward_indices"] == [3]


def test_recovery_reset_from_final_index_never_counts_as_a_completed_loop(evidence):
    tracker, sample, _ = evidence
    sample(0)
    sample(43)
    sample(0, return_count=1, recovery_state="completed")  # active recovery occurred between polls
    assert not tracker.wrap and tracker.observed == [0]
    assert tracker.resets[-1]["reason"] == "recovery"


def test_ordinary_backward_reset_invalidates_partial_progress(evidence):
    tracker, sample, _ = evidence
    sample(0)
    sample(30)
    sample(0)
    sample(43)
    sample(0)
    assert tracker.wrap
    assert not tracker.summary()["all_transition_indices_observed"]
    assert tracker.resets[-1]["reason"] == "backward index without complete wrap"


def test_leaving_start_clears_latched_ground_proof(evidence):
    tracker, sample, _ = evidence
    sample(0)
    for _ in range(3):
        sample(43)
    assert tracker.ground_proof.confirmed
    sample(0, minimap=[49, 102])
    assert tracker.wrap and not tracker.ground_proof.confirmed


def test_duplicate_status_does_not_count_as_multiple_stable_ground_samples(evidence):
    tracker, sample, _ = evidence
    sample(0)
    status = sample(43)
    for _ in range(10):
        tracker.observe(status)
    assert len(tracker.ground_proof.samples) == 1 and not tracker.ground_proof.confirmed


def test_end_requests_guarded_hold_and_never_full_pause(monkeypatch):
    monkeypatch.setattr(watcher.time, "time", lambda: 1000.)
    state = {"running": True, "updated": 1000., "paused": False, "full_pause": False}
    calls = []
    def hold():
        calls.append("hold")
        state.update(paused=True, manual_health_hold=True, health_protection_state="monitoring")
        return "queued"
    bridge = SimpleNamespace(_status=lambda: state, hold_farming=hold)
    result = watcher.hold_at_end(bridge, lambda row: None)
    assert calls == ["hold"] and result["confirmed"] and result["manual"]


def test_stale_controller_never_receives_deferred_hold_request(monkeypatch):
    monkeypatch.setattr(watcher.time, "time", lambda: 1000.)
    bridge = SimpleNamespace(_status=lambda: {"running": True, "updated": 990.},
                             hold_farming=lambda: pytest.fail("stale hold request"))
    result = watcher.hold_at_end(bridge, lambda row: None)
    assert result["needs_attention"] and not result["confirmed"]


def test_short_capture_gap_recovers_without_counting_duplicate_as_usable():
    continuity = watcher.CaptureContinuity()
    good = {"map_ready": True, "capture_state": "fresh", "capture_age_s": .1,
            "minimap_seen": True, "minimap": [49, 105], "updated": 1000.}
    assert continuity.observe(good, 0)
    assert not continuity.observe(dict(good, capture_state="duplicate"), .1)
    assert not continuity.expired(1.9)
    assert continuity.observe(dict(good, updated=1001.), 1.9)
    assert not continuity.expired(5)
    assert continuity.events[0]["duration_s"] == pytest.approx(1.8)
    assert continuity.events[0]["recovered"] == 1001.


def test_missing_minimap_expires_after_two_seconds():
    continuity = watcher.CaptureContinuity()
    missing = {"map_ready": True, "capture_state": "fresh", "capture_age_s": .1,
               "minimap_seen": False, "minimap": None, "updated": 1000.}
    assert not continuity.observe(missing, 10)
    assert not continuity.expired(11.99)
    assert continuity.expired(12)


def test_missing_status_read_retries_without_replaying_previous_sample(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(watcher.time, "time", lambda: clock[0])
    initial = {"running": True, "updated": 1000., "circuit_index": 42}
    reader = watcher.StatusContinuity(initial)
    assert reader.observe({}, 10) == (None, None)
    assert reader.change["event"] == "status_read_gap"
    clock[0] = 1001.5
    assert reader.observe({}, 11.5) == (None, None)
    recovered = {**initial, "updated": 1001.5, "circuit_index": 43}
    assert reader.observe(recovered, 11.6) == (recovered, None)
    assert reader.change["event"] == "status_read_recovered"
    assert reader.events[0]["duration_s"] == pytest.approx(1.6)
    assert reader.last_good["circuit_index"] == 43


def test_status_read_gap_is_bounded_even_with_recent_previous_status(monkeypatch):
    monkeypatch.setattr(watcher.time, "time", lambda: 1000.)
    reader = watcher.StatusContinuity({"running": True, "updated": 1000.})
    assert reader.observe({}, 0) == (None, None)
    assert "two seconds" in reader.observe({}, 2)[1]
    assert "stopped" in watcher.StatusContinuity().observe({"running": False}, 0)[1]


def test_last_good_status_becoming_stale_ends_read_retry_before_two_seconds(monkeypatch):
    clock = [1002.9]
    monkeypatch.setattr(watcher.time, "time", lambda: clock[0])
    reader = watcher.StatusContinuity({"running": True, "updated": 1000.})
    assert reader.observe({}, 10) == (None, None)
    clock[0] = 1003.01
    assert "stale" in reader.observe({}, 10.11)[1]
    assert "stale" in reader.observe({"running": True, "updated": 1000.}, 10.12)[1]


def test_locked_status_file_is_a_retryable_unavailable_read():
    def locked():
        raise PermissionError("Windows reader lock")
    assert watcher.read_status(SimpleNamespace(_status=locked)) == {}


def test_startup_return_does_not_use_failure_recovery_budget_or_count_as_wrap(evidence):
    tracker, sample, _ = evidence
    sample(0)
    sample(43)
    sample(0, startup_return_state="checking", startup_return_reason="Checking HP")
    sample(0, startup_return_state="returning", returning_to_start=True, return_count=1, recovery_state="running")
    sample(0, startup_return_state="completed", return_count=0, recovery_state="completed")
    summary = tracker.summary()
    assert not summary["full_wrap_observed"] and summary["observed_indices"] == [0]
    assert summary["maximum_return_count"] == 0 and not summary["attempt_resets"]
    assert [event["state"] for event in summary["startup_events"]] == ["checking", "returning", "completed"]
    assert not any(event["active"] for event in summary["recovery_events"])
    sample(10, startup_return_state="completed", returning_to_start=True, return_count=1, recovery_state="running")
    assert tracker.summary()["maximum_return_count"] == 1
    assert tracker.resets[-1]["reason"] == "recovery"


def test_watch_survives_single_unavailable_status_without_requesting_hold(evidence, tmp_path, monkeypatch):
    tracker, _, clock = evidence
    initial = {"running": True, "updated": clock[0], "pid": 1, "paused": True, "full_pause": True,
               "hp": 80, "map_ready": True, "map_state": "matched", "capture": {}, "capture_state": "fresh",
               "capture_age_s": .1, "minimap": [49, 105], "minimap_seen": True, "circuit_count": 44, "circuit_index": 0}
    states = iter([{**initial, "paused": False, "full_pause": False}, {},
                   {**initial, "updated": 1000.1, "paused": False, "full_pause": False, "circuit_index": 1},
                   {**initial, "updated": 1000.2, "circuit_index": 1}])
    cfg = {"keys": {"hp_potion": "q", "mp_potion": "a"}}
    monkeypatch.setattr(watcher, "preflight", lambda bridge, **kwargs: (initial, cfg, tracker.circuit, {}))
    monkeypatch.setattr(watcher.time, "sleep", lambda seconds: None)
    bridge = SimpleNamespace(log=tmp_path, _status=lambda: next(states),
                             hold_farming=lambda: pytest.fail("Transient read must not end the watch"))
    summary = watcher.watch(bridge, tmp_path / "watch.jsonl", timeout_s=1)
    assert summary["result"] == "paused_by_user"
    assert len(summary["status_read_gaps"]) == 1 and summary["status_read_gaps"][0]["recovered"] == 1000.1
    assert summary["observed_indices"] == [0, 1]


def test_start_preflight_requires_foreground_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(watcher.time, "time", lambda: 1000.)
    identity = {"active": "map", "reference_sha256": "reference"}
    potion = {"state": "passed", "config_unchanged": True, "profile": "map", "reference_sha256": "reference",
              "results": {name: {"state": "passed"} for name in ("hp", "mp")}}
    monkeypatch.setattr(watcher, "read_json", lambda path: potion)
    bridge = SimpleNamespace(log=tmp_path, _cfg=lambda: {"map_profile": identity},
        _status=lambda: {"running": True, "updated": 1000., "paused": True, "full_pause": True, "foreground": False})
    result = watcher.watch(bridge, tmp_path / "watch.jsonl", start=True, timeout_s=.1)
    assert result["result"] == "failed" and "in front" in result["reason"]
    assert not (tmp_path / "resume-active").exists() and not (tmp_path / "hold-farming-active").exists()


@pytest.mark.parametrize("manual_hold", [False, True])
def test_preflight_accepts_ground_approach_but_completion_stays_at_point_one(evidence, tmp_path, monkeypatch, manual_hold):
    tracker, _, clock = evidence
    circuit = {**tracker.circuit, "enabled": True, "loop": True}
    identity = {"active": "map", "reference_sha256": "reference"}
    cfg = {"map_profile": identity, "route": {"circuit": circuit},
           "health": {"enable": True}, "keys": {"hp_potion": "q"}}
    state = {"running": True, "updated": clock[0], "paused": True, "full_pause": True, "foreground": True,
             "map_ready": True, "map_state": "matched", "capture": watcher.capture_signature(cfg),
             "capture_state": "fresh", "capture_age_s": .1, "hp": 80, "circuit_count": 44, "circuit_index": 0,
             "minimap_seen": True, "minimap": [43, 105]}
    if manual_hold:
        state.update(full_pause=False, manual_health_hold=True, health_protection_state="monitoring")
    potion = {"state": "passed", "config_unchanged": True, "profile": "map", "reference_sha256": "reference",
              "results": {name: {"state": "passed"} for name in ("hp", "mp")}}
    monkeypatch.setattr(watcher, "read_json", lambda path: potion)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    for path in (tmp_path / "user.yaml", profiles / "map.json", profiles / "map.png"):
        path.write_bytes(b"test")
    bridge = SimpleNamespace(log=tmp_path, user_config=tmp_path / "user.yaml", profiles=profiles,
                             _cfg=lambda: cfg, _status=lambda: state)
    assert watcher.preflight(bridge, require_foreground=True)[0] is state
    assert not watcher.at_start(state, circuit["waypoints"][0])
    if manual_hold:
        state["health_protection_state"] = "healing"
        with pytest.raises(ValueError, match="monitoring"):
            watcher.preflight(bridge, require_foreground=True)
        state["health_protection_state"] = "monitoring"
        state["manual_health_hold"] = False
        with pytest.raises(ValueError, match="manual HP hold"):
            watcher.preflight(bridge, require_foreground=True)
        state["manual_health_hold"] = True
        state["recovery_terminal"] = True
        with pytest.raises(ValueError, match="recovery"):
            watcher.preflight(bridge, require_foreground=True)
        state["recovery_terminal"] = False
    state["minimap"] = [43, 104]
    with pytest.raises(ValueError, match="ground platform"):
        watcher.preflight(bridge, require_foreground=True)


def test_rejoin_states_at_a_constant_index_record_no_reset_and_the_landing_still_confirms(evidence):
    tracker, sample, _ = evidence
    sample(0)
    for index in range(1, 10):
        sample(index)
    sample(10, circuit_state="waiting_floor", minimap=[93, 105])  # knocked back off the landing floor
    for state in ("rejoin_starting", "rejoin_walking", "rejoin_jumping", "rejoin_airborne"):
        sample(10, circuit_state=state, minimap=[93, 105], rejoin_active=True, rejoin_count=1,
               rejoin_state="running")
    assert not tracker.proofs[9].confirmed and not tracker.proofs[9].samples
    for state in ("rejoin_jump_landing", "rejoin_confirming", "rejoined", "dwelling"):
        sample(10, circuit_state=state, rejoin_count=1)
    assert tracker.proofs[9].confirmed and not tracker.resets and tracker.observed == list(range(11))
    summary = tracker.summary()
    assert not summary["full_wrap_observed"]  # a rejoin is never a wrap
    assert summary["maximum_rejoin_count"] == 1 and summary["maximum_return_count"] == 0
    assert summary["arrivals_without_takeoff"] == 0


def test_jump_accepted_by_arrival_confirms_its_landing_positionally_and_is_reported(evidence):
    tracker, sample, _ = evidence
    sample(0, arrival_count=4)  # counted since the controller started, not since this watch
    for index in range(1, 17):
        sample(index, arrival_count=4)
    for state in ("jump_arrived_checking", "jump_arrived_aligning", "jump_arrived"):
        sample(17, circuit_state=state, arrival_count=4)
    assert not tracker.proofs[17].samples  # nothing before the index passes
    for _ in range(3):
        sample(18, circuit_state="walking", arrival_count=5)
    assert tracker.proofs[17].confirmed
    summary = tracker.summary()
    assert summary["arrivals_without_takeoff"] == 1 and summary["maximum_rejoin_count"] == 0


def test_rejoin_and_arrival_reporting_does_not_change_a_verified_lap(evidence):
    tracker, sample, _ = evidence
    for index in range(44):
        for _ in range(3):
            sample(index, rejoin_count=2 if index > 20 else 0, arrival_count=1 if index > 30 else 0)
    sample(0)
    summary = tracker.summary()
    assert summary["full_wrap_observed"] and summary["stable_ground_return"]
    assert summary["all_transition_indices_observed"] and summary["all_landings_observed"]
    assert summary["maximum_rejoin_count"] == 2 and summary["arrivals_without_takeoff"] == 1
    assert not summary["attempt_resets"]


def test_zone_takeoffs_are_reported_since_the_watch_began_and_change_no_verdict(evidence):
    first, _, clock = evidence

    def lap(count):
        tracker, clock[0] = watcher.LapEvidence(first.circuit), 1000.
        for index in [i for i in range(44) for _ in range(3)] + [0]:
            clock[0] += .31
            tracker.observe({"updated": clock[0], "minimap": [49, 105], "minimap_seen": True, "circuit_index": index,
                             "return_count": 0, "returning_to_start": False, "recovery_state": "idle", **count(index)})
        return tracker.summary()
    # Counted since the controller's circuit was built (5 before this watch), then 7 more in the lap.
    zoned = lap(lambda index: {"zone_takeoffs": 5 + min(7, index // 6)})
    assert zoned["maximum_zone_takeoffs"] == 7
    assert zoned["full_wrap_observed"] and zoned["stable_ground_return"]
    assert zoned["all_transition_indices_observed"] and zoned["all_landings_observed"] and not zoned["attempt_resets"]
    plain = lap(lambda index: {})  # a controller that does not publish the counter
    assert plain["maximum_zone_takeoffs"] == 0
    assert {**zoned, "maximum_zone_takeoffs": 0} == plain  # nothing else in the summary depends on it
