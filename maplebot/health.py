"""Potion and buff decisions. Pure state machines fed with readings and the current time."""
from __future__ import annotations

import math


class _PotionGate:
    """One resource: drink below `low`, re-arm above `low + rearm`, retry after `retry_s`.

    After `max_retries` retries without the bar recovering, slow down to one try per
    `backoff_s` so a misread bar cannot drain the potion stock.
    """

    def __init__(self, low: float, rearm: float, retry_s: float, max_retries: int = 5, backoff_s: float = 30.0):
        self.low = low
        self.rearm = rearm
        self.retry_s = retry_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.armed = True
        self.last_drink_t: float | None = None
        self.retries = 0

    def update(self, value: float, now: float) -> bool:
        if value >= self.low + self.rearm:
            self.armed = True
            self.retries = 0
        if value >= self.low:
            return False
        if self.armed:
            self.armed = False
            self.last_drink_t = now
            self.retries = 0
            return True
        wait = self.retry_s if self.retries < self.max_retries else self.backoff_s
        if self.last_drink_t is not None and now - self.last_drink_t >= wait:
            self.last_drink_t = now
            self.retries += 1
            return True
        return False


class MpPressObservation:
    """Observation only: MP-key presses actually sent without a following MP rise.

    Fed a cumulative counter of MP-key presses the input backend really emitted
    (for example the controller's ``keys_sent`` count for ``keys.mp_potion``).
    It must never be fed ``_PotionGate`` requests: ``KeyboardController.request``
    silently drops requests while HP-only, disabled, navigation-only or out of
    focus, so a request is not a press.

    A sent press is answered when a fresh MP reading reaches the reading before
    that press + ``rise`` within ``response_s``; a press whose window ends
    without that is unanswered. ``presses_unanswered`` counts unanswered sent
    presses since an answered one. A rise with no sent press pending
    (regeneration) neither answers nor resets anything, so a slowly
    regenerating bar cannot hide an empty MP slot; regeneration inside a
    response window can still look like an answer. The observer runs only
    while input is ready, so a press pending across a gap of more than
    ``response_s`` without a valid reading is counted in ``presses_unknown``:
    regeneration during the gap must not answer it against a stale baseline.

    This has no effect on control: the MP gate keeps its retry and backoff.
    Below ``attack.mp_fallback_percent`` combat uses the normal attack, so an
    empty MP item slot shows up here rather than as a stop.
    """

    def __init__(self, response_s: float = 3.0, rise: float = 2.0):
        self.response_s = float(response_s)
        self.rise = float(rise)
        self.presses_sent = 0
        self.presses_unanswered = 0
        self.presses_answered = 0
        # Sent presses that could not be judged: no valid reading for longer than
        # ``response_s`` around them (pause, hold, latch), so neither answered nor not.
        self.presses_unknown = 0
        self._last_total: int | None = None
        self._last_mp: float | None = None
        self._last_mp_t: float | None = None
        self._pending: list | None = None  # [baseline mp, last sent time, presses]

    def observe(self, mp, now: float, sent_total) -> None:
        if (type(sent_total) is not int or sent_total < 0 or not isinstance(now, (int, float))
                or not math.isfinite(now)):
            return
        valid = (isinstance(mp, (int, float)) and not isinstance(mp, bool)
                 and math.isfinite(mp) and 0 <= mp <= 100)
        if self._last_total is None or sent_total < self._last_total:
            self._last_total, self._pending = sent_total, None
            if valid:
                self._last_mp, self._last_mp_t = float(mp), now
            return
        delta = sent_total - self._last_total
        self._last_total = sent_total
        if delta:
            self.presses_sent += delta
            before = self._last_mp if self._last_mp is not None else (float(mp) if valid else None)
            if self._pending is None:
                self._pending = [before, now, delta]
            else:
                if before is not None:
                    self._pending[0] = before if self._pending[0] is None else min(self._pending[0], before)
                self._pending[1] = now
                self._pending[2] += delta
        if not valid:
            return
        if self._pending is not None:
            baseline, sent_at, count = self._pending
            if self._last_mp_t is not None and now - self._last_mp_t > self.response_s:
                # Observation gap: the press baseline predates the gap and the window
                # may have closed inside it, so regen during the gap cannot answer it.
                self.presses_unknown += count
                self._pending = None
            elif baseline is None:
                self._pending[0] = float(mp)
            elif now - sent_at <= self.response_s and mp >= baseline + self.rise:
                self.presses_answered += 1
                self.presses_unanswered = 0
                self._pending = None
            elif now - sent_at >= self.response_s:
                self.presses_unanswered += count
                self._pending = None
        self._last_mp, self._last_mp_t = float(mp), now


class HealthPolicy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enable = bool(cfg.get("enable", True))
        self.warmup_reads = int(cfg.get("warmup_reads", 10))
        rearm = float(cfg.get("rearm_delta", 5))
        retry = float(cfg.get("retry_s", 2.0))
        max_retries = int(cfg.get("max_retries", 5))
        backoff = float(cfg.get("backoff_s", 30.0))
        self._hp_gate = _PotionGate(float(cfg.get("hp_percent", 50)), rearm, retry, max_retries, backoff)
        self._mp_gate = _PotionGate(float(cfg.get("mp_percent", 30)), rearm, retry, max_retries, backoff)
        self.hp: float | None = None
        self.mp: float | None = None
        self.last_hp_drop_t: float = 0.0
        self._reads = 0
        # Observation only. The integrator may set ``mp_sent_counter`` to a
        # callable returning the cumulative count of MP-key presses actually sent.
        self.mp_sent_counter = None
        self.mp_presses = MpPressObservation()

    @property
    def mp_presses_unanswered(self) -> int | None:
        """Sent MP-key presses not followed by an MP rise; None without a sent counter.

        An observation with no effect on control. Below attack.mp_fallback_percent
        the normal attack is used, so an empty MP slot degrades combat only.
        """
        return None if self.mp_sent_counter is None else self.mp_presses.presses_unanswered

    def _observe_mp_presses(self, mp, now) -> None:
        counter = self.mp_sent_counter
        if counter is None or mp is None:
            return
        try:
            sent = counter()
        except Exception:
            return  # diagnostics must never interrupt potion decisions
        self.mp_presses.observe(mp, now, sent)

    def update(self, hp: float | None, mp: float | None, now: float) -> list[str]:
        if hp is not None:
            if self.hp is not None and hp < self.hp:
                self.last_hp_drop_t = now
            self.hp = hp
        if mp is not None:
            self.mp = mp
            self._observe_mp_presses(mp, now)
        if hp is None and mp is None:
            return []
        self._reads += 1
        if not self.enable or self._reads <= self.warmup_reads:
            return []
        out = []
        if hp is not None and self._hp_gate.update(hp, now):
            out.append("hp_potion")
        if mp is not None and self._mp_gate.update(mp, now):
            out.append("mp_potion")
        return out


class BuffScheduler:
    def __init__(self, cfg: dict):
        keys = [str(k) for k in (cfg.get("keys") or [])]
        cds = cfg.get("cooldowns_s") or []
        self.keys: list[str] = []
        self.cooldowns: list[float] = []
        for i, k in enumerate(keys):
            cd = float(cds[i]) if i < len(cds) else 0.0
            if cd <= 0:
                # a buff with no cooldown would be cast every tick and freeze movement forever
                import logging
                logging.getLogger("maplebot").warning("buff key %r has no cooldown in buffs.cooldowns_s; skipped", k)
                continue
            self.keys.append(k)
            self.cooldowns.append(cd)
        self.freeze_s = float(cfg.get("freeze_s", 1.0))
        self._last = {k: None for k in self.keys}

    def due(self, now: float) -> str | None:
        for k, cd in zip(self.keys, self.cooldowns):
            last = self._last[k]
            if last is None or now - last >= cd:
                return k
        return None

    def mark(self, key: str, now: float) -> None:
        self._last[key] = now
