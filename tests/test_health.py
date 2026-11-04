from maplebot.health import BuffScheduler, HealthPolicy

CFG = {
    "enable": True,
    "hp_percent": 50,
    "mp_percent": 30,
    "rearm_delta": 5,
    "retry_s": 2.0,
    "warmup_reads": 3,
}


def _warm(policy, t=0.0):
    for i in range(CFG["warmup_reads"]):
        policy.update(100, 100, t + i * 0.01)
    return t + 1


def test_policy_ignores_warmup_reads():
    p = HealthPolicy(CFG)
    for i in range(CFG["warmup_reads"]):
        assert p.update(0, 0, i * 0.1) == []


def test_policy_drinks_below_threshold():
    p = HealthPolicy(CFG)
    t = _warm(p)
    assert p.update(40, 100, t) == ["hp_potion"]
    assert p.update(100, 20, t + 0.1) == ["mp_potion"]


def test_policy_drinks_both_when_both_low():
    p = HealthPolicy(CFG)
    t = _warm(p)
    assert p.update(40, 20, t) == ["hp_potion", "mp_potion"]


def test_policy_requires_rearm():
    p = HealthPolicy(CFG)
    t = _warm(p)
    assert p.update(40, 100, t) == ["hp_potion"]
    assert p.update(52, 100, t + 0.2) == []        # not yet above 50 + 5
    assert p.update(56, 100, t + 0.3) == []        # re-armed, but healthy
    assert p.update(40, 100, t + 0.4) == ["hp_potion"]


def test_policy_retries_after_timeout():
    p = HealthPolicy(CFG)
    t = _warm(p)
    assert p.update(40, 100, t) == ["hp_potion"]
    assert p.update(40, 100, t + 1.0) == []
    assert p.update(40, 100, t + 2.1) == ["hp_potion"]


def test_policy_tracks_hp_drop_time():
    p = HealthPolicy(CFG)
    t = _warm(p)
    p.update(90, 100, t)
    p.update(80, 100, t + 1)
    assert p.last_hp_drop_t == t + 1
    assert p.hp == 80 and p.mp == 100


def test_policy_none_reads_keep_last_values():
    p = HealthPolicy(CFG)
    t = _warm(p)
    p.update(70, 60, t)
    assert p.update(None, None, t + 1) == []
    assert (p.hp, p.mp) == (70, 60)


def test_policy_disabled_never_drinks():
    p = HealthPolicy(dict(CFG, enable=False))
    t = _warm(p)
    assert p.update(1, 1, t) == []


def test_buff_scheduler_due_once_per_cooldown():
    b = BuffScheduler({"keys": ["a", "s"], "cooldowns_s": [10, 20], "freeze_s": 1.0})
    assert b.due(0.0) == "a"
    b.mark("a", 0.0)
    assert b.due(0.1) == "s"
    b.mark("s", 0.1)
    assert b.due(5.0) is None
    assert b.due(10.5) == "a"


def test_buff_scheduler_empty():
    assert BuffScheduler({"keys": [], "cooldowns_s": []}).due(0.0) is None


def test_policy_backs_off_after_repeated_failed_retries():
    cfg = dict(CFG, retry_s=2.0, max_retries=3, backoff_s=30.0)
    p = HealthPolicy(cfg)
    t = _warm(p)
    drinks = 0
    for i in range(30):
        if p.update(10, 100, t + i):
            drinks += 1
    # initial drink + 3 retries (t, t+2, t+4, t+6), then back off for 30 s
    assert drinks == 4
    assert p.update(10, 100, t + 6 + 30.5) == ["hp_potion"]
