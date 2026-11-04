"""Potion and buff decisions. Pure state machines fed with readings and the current time."""
from __future__ import annotations


class _PotionGate:
    """One resource: drink below `low`, re-arm above `low + rearm`, retry after `retry_s`."""

    def __init__(self, low: float, rearm: float, retry_s: float):
        self.low = low
        self.rearm = rearm
        self.retry_s = retry_s
        self.armed = True
        self.last_drink_t: float | None = None

    def update(self, value: float, now: float) -> bool:
        if value >= self.low + self.rearm:
            self.armed = True
        if value >= self.low:
            return False
        if self.armed:
            self.armed = False
            self.last_drink_t = now
            return True
        if self.last_drink_t is not None and now - self.last_drink_t >= self.retry_s:
            self.last_drink_t = now
            return True
        return False


class HealthPolicy:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enable = bool(cfg.get("enable", True))
        self.warmup_reads = int(cfg.get("warmup_reads", 10))
        rearm = float(cfg.get("rearm_delta", 5))
        retry = float(cfg.get("retry_s", 2.0))
        self._hp_gate = _PotionGate(float(cfg.get("hp_percent", 50)), rearm, retry)
        self._mp_gate = _PotionGate(float(cfg.get("mp_percent", 30)), rearm, retry)
        self.hp: float | None = None
        self.mp: float | None = None
        self.last_hp_drop_t: float = 0.0
        self._reads = 0

    def update(self, hp: float | None, mp: float | None, now: float) -> list[str]:
        if hp is not None:
            if self.hp is not None and hp < self.hp:
                self.last_hp_drop_t = now
            self.hp = hp
        if mp is not None:
            self.mp = mp
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
        self.keys = [str(k) for k in (cfg.get("keys") or [])]
        cds = cfg.get("cooldowns_s") or []
        self.cooldowns = [float(cds[i]) if i < len(cds) else 0.0 for i in range(len(self.keys))]
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
