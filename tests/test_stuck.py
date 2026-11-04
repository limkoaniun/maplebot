from maplebot.nav.stuck import StuckDetector


def test_stuck_fires_after_timeout_and_resets():
    d = StuckDetector(range_px=10, timeout_s=3.0)
    assert d.update((100, 100), 0.0) is False
    assert d.update((102, 101), 2.9) is False
    assert d.update((103, 100), 3.1) is True
    assert d.update((103, 100), 3.2) is False      # timer restarted
    assert d.update((103, 100), 6.3) is True


def test_stuck_resets_on_movement():
    d = StuckDetector(range_px=10, timeout_s=3.0)
    d.update((100, 100), 0.0)
    d.update((130, 100), 2.0)
    assert d.update((130, 100), 4.0) is False
    assert d.update((130, 100), 5.1) is True


def test_stuck_ignores_none_positions():
    d = StuckDetector(range_px=10, timeout_s=3.0)
    d.update((100, 100), 0.0)
    assert d.update(None, 10.0) is False
    assert d.update((100, 100), 10.1) is False   # None does not accumulate, timer restarts
