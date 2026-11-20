"""Offline boundary regressions for the supervised circuit observer. Never sends requests."""
from types import SimpleNamespace
import pytest
from scripts import watch_circuits as watch


@pytest.fixture
def samples(monkeypatch):
    clock = [1000.]
    monkeypatch.setattr(watch.one.time, "time", lambda: clock[0])
    circuit = {"platforms": [{"id": "ground", "left": 40, "right": 58, "y": 105}],
               "waypoints": [{"x": 49, "y": 105, "action": "jump" if i in range(1, 35, 2) else "walk",
                              "platform_id": "ground", "to_platform": "ground", "connection_id": f"edge-{i}"}
                             for i in range(44)]}
    laps = watch.ContinuousLaps(circuit)
    def sample(index, **extra):
        clock[0] += .31
        state = {"updated": clock[0], "minimap": [49, 105], "minimap_seen": True, "circuit_index": index,
                 "return_count": 0, "returning_to_start": False, "recovery_state": "idle", **extra}
        return laps.observe(state)
    return laps, sample


def fill_lap(sample, *, last_x=49):
    for index in range(44):
        for _ in range(3):
            sample(index, minimap=[last_x if index == 43 else 49, 105])


def test_three_continuous_wraps_seed_next_lap_point_zero(samples):
    laps, sample = samples
    for number in range(1, 4):
        fill_lap(sample)
        completed = sample(0)
        assert completed["lap"] == number
        assert completed["all_landings_observed"] and completed["stable_ground_return"]
        assert laps.current.observed == [0]
        assert laps.current.previous == 0
        assert not laps.current.wrap
    assert len(laps.completed) == 3 and laps.error is None


def test_postwrap_natural_ground_dwell_confirms_without_losing_next_lap(samples):
    laps, sample = samples
    fill_lap(sample, last_x=54)
    assert sample(0, minimap=[54, 105]) is None
    assert laps.pending is not None and laps.current.observed == [0]
    assert sample(0) is None
    assert sample(0) is None
    assert sample(0)["lap"] == 1
    sample(1)
    assert laps.current.observed == [0, 1]


def test_recovery_index_reset_never_completes_a_lap(samples):
    laps, sample = samples
    fill_lap(sample)
    sample(0, return_count=1, recovery_state="completed")
    assert not laps.completed and laps.pending is None and not laps.current.wrap
    assert laps.current.resets[-1]["reason"] == "recovery"


def test_missing_transition_or_ground_proof_is_not_success(samples):
    laps, sample = samples
    for index in range(44):
        if index != 3:
            for _ in range(3):
                sample(index)
    sample(0)
    assert laps.error and not laps.completed


def test_departure_before_stable_ground_proof_is_incomplete(samples):
    laps, sample = samples
    fill_lap(sample, last_x=54)
    sample(0)
    sample(1, minimap=[49, 103])
    assert laps.error and not laps.completed


def test_explicit_user_pause_is_never_overridden(monkeypatch, tmp_path):
    monkeypatch.setattr(watch.one.time, "time", lambda: 1000.)
    bridge = SimpleNamespace(log=tmp_path, _status=lambda: {
        "running": True, "updated": 1000., "paused": True, "full_pause": True},
        hold_farming=lambda: pytest.fail("Must not replace explicit Pause"))
    assert watch.finish_hold(bridge, lambda _: None, None)["preserved_user_pause"]


@pytest.mark.parametrize("target", [1, 3])
def test_observer_stops_only_after_requested_full_laps_and_leaves_active(monkeypatch, tmp_path, target):
    clock = [1000.]
    monkeypatch.setattr(watch.time, "time", lambda: clock[0])
    monkeypatch.setattr(watch.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(watch.time, "sleep", lambda _: None)
    circuit = {"platforms": [{"id": "ground", "left": 40, "right": 58, "y": 105}],
               "waypoints": [{"x": 49, "y": 105, "action": "jump" if i in range(1, 35, 2) else "walk",
                              "platform_id": "ground", "to_platform": "ground", "connection_id": f"edge-{i}"}
                             for i in range(44)]}

    def status(index):
        return {"running": True, "updated": clock[0], "pid": 123, "hp": 90,
                "paused": False, "full_pause": False, "minimap": [49, 105],
                "minimap_seen": True, "circuit_index": index, "circuit_count": 44,
                "return_count": 0, "returning_to_start": False, "recovery_state": "idle"}

    initial = status(0)
    indices = iter([i for _ in range(target) for i in ([j for j in range(44) for _ in range(3)] + [0])])

    def read_status(_):
        clock[0] += .31
        return status(next(indices))  # asking for more frames would make the observer fail

    class Reader:
        events = []
        change = None

        def __init__(self, initial):
            self.last_good = initial

        def observe(self, state, now):
            self.last_good = state
            return state, None

    monkeypatch.setattr(watch.one, "preflight", lambda *a, **k: (
        initial, {"keys": {"hp_potion": "q", "mp_potion": "a"}}, circuit, {}))
    monkeypatch.setattr(watch.one, "read_status", read_status)
    monkeypatch.setattr(watch.one, "StatusContinuity", Reader)
    monkeypatch.setattr(watch.one, "CaptureContinuity", lambda: SimpleNamespace(
        events=[], observe=lambda *a: True, expired=lambda *a: False))
    monkeypatch.setattr(watch.one, "map_guard_error", lambda *a: None)
    monkeypatch.setattr(watch.one, "evidence_images", lambda *a: {})
    monkeypatch.setattr(watch.one, "atomic_json", lambda *a: None)
    monkeypatch.setattr(watch, "finish_hold", lambda *a: pytest.fail("Success must leave farming active"))
    bridge = SimpleNamespace(log=tmp_path)
    # Leaving the argument absent exercises the existing default-three behavior.
    options = {"target_laps": 1} if target == 1 else {}
    result = watch.watch(bridge, tmp_path / "watch.jsonl", **options)
    assert result["result"] == "completed"
    assert result["target_laps"] == result["laps_completed"] == target
    assert result["farming_active_at_completion"]
    assert result["hold"]["requested"] is False
    assert all(lap["all_landings_observed"] and lap["full_wrap_observed"]
               and lap["stable_ground_return"] for lap in result["laps"])
    assert not (tmp_path / "resume-active").exists()


@pytest.mark.parametrize("arguments,target", [([], 3), (["--laps", "1"], 1)])
def test_lap_option_defaults_to_three_and_accepts_one(monkeypatch, tmp_path, arguments, target):
    seen = {}

    def observe(*args, **kwargs):
        seen.update(kwargs)
        return {"result": "completed"}

    monkeypatch.setattr(watch, "watch", observe)
    monkeypatch.setattr(watch.one, "EditorBridge", lambda _: object())
    monkeypatch.setattr(watch.sys, "argv", ["watch", "--output", str(tmp_path / "out.jsonl"), *arguments])
    assert watch.main() == 0
    assert seen["target_laps"] == target


@pytest.mark.parametrize("target", [0, 4, True, 1.5])
def test_invalid_lap_target_rejected_before_preflight_or_requests(tmp_path, target):
    with pytest.raises(ValueError, match="target_laps"):
        watch.watch(object(), tmp_path / "watch.jsonl", target_laps=target)
    assert not (tmp_path / "watch.jsonl").exists()


def test_lap_with_published_reentry_completes_and_next_lap_starts_clean(samples):
    laps, sample = samples
    step = {"from_index": 18, "to_index": 16, "platform": "floor-3-right"}
    for number in (1, 2):
        for index in range(19):
            for _ in range(3):
                sample(index, reentry_count=0)
        sample(16, reentry_count=1, reentry_last=step)
        for index in range(16, 44):
            for _ in range(3):
                sample(index, reentry_count=1, reentry_last=step)
        completed = sample(0, reentry_count=0, reentry_last=step)
        assert completed["lap"] == number and laps.error is None
        assert completed["all_landings_observed"] and completed["stable_ground_return"]
        assert not completed["attempt_resets"]
        assert [(r["from"], r["to"], r["platform"]) for r in completed["reentries"]] == [(18, 16, "floor-3-right")]
        assert laps.current.observed == [0] and laps.current.reentries == []


def test_backward_step_without_reentry_fields_still_resets_the_lap(samples):
    laps, sample = samples
    for index in range(19):
        sample(index)
    sample(16)
    fill_lap(sample)
    sample(0)
    assert laps.error is None and len(laps.completed) == 1
    resets = laps.completed[0]["attempt_resets"]
    assert [(r["reason"], r["previous"], r["index"]) for r in resets] == [
        ("backward index without complete wrap", 18, 16)]
    assert laps.completed[0]["reentries"] == []
