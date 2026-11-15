"""Evidence counters read from screen pixels, plus time-in-state accounting.

Everything here is an observation of the HUD on fresh frames: "the EXP number was
redrawn", "a gain notification line arrived". They are lower bounds on gain events,
never exact kill, meso or item counts, and they are independent of key presses.
"""
from __future__ import annotations

import os
import time

from maplebot.config import PACKAGE_ROOT
from maplebot.log import get_logger
from maplebot.vision.gains import (changed_pixel_count, classify_gain_line, crop_rect, exp_text_plausible,
                                   gain_line_arrivals, load_gain_templates)

log = get_logger()
BUCKETS = ("fighting", "looting", "traversing", "recovering", "other")
NOTE = "On-screen gain observations on fresh frames; lower bounds, not kill, meso or item counts."


def activity_bucket(brain_state) -> str:
    """Map a brain state string to one of BUCKETS. Return/recovery/blocked states win."""
    state = str(brain_state or "")
    if state.startswith(("circuit_rejoin", "minimap_route_rejoin")):
        return "traversing"  # a platform rejoin walks the saved route; it is not a return
    if any(word in state for word in ("return", "recover", "blocked")):
        return "recovering"
    if state in ("hunting", "pursuing_target", "reacquiring_target"):
        return "fighting"  # "hunting" is also the brain's idle default while it searches
    if state.startswith("looting_"):
        return "looting"
    if state.startswith(("circuit_", "minimap_route_", "minimap_vertical_")):
        return "traversing"
    # Includes "input_held": Bot.step passes it for a step that idled (unfocused, paused, input
    # blocked with no route running), when the suspended brain still shows its last state.
    return "other"


def summary_line(snapshot: dict | None) -> str:
    """One log line for a session. States plainly what the numbers are."""
    if not snapshot:
        return "evidence: disabled"
    show = lambda key: "n/a" if snapshot.get(key) is None else snapshot[key]  # noqa: E731
    seconds = snapshot.get("seconds") or {}
    return ("evidence (on-screen observations, lower bounds): "
            f"exp_text_changes={show('exp_text_changes')} gain_lines={show('gain_lines_total')} "
            f"(exp={show('exp_lines')} meso={show('meso_lines')} other={show('other_lines')}) "
            + "seconds " + " ".join(f"{name}={float(seconds.get(name, 0.0)):.0f}" for name in BUCKETS))


def carry_over_tracker(previous, current, final: dict, report=log.info):
    """Pick the tracker for a rebuilt bot (settings reload, editor test) and keep earlier counts.

    `previous` is the tracker of the bot being replaced, `current` the new bot's (None when
    evidence is disabled). Same settings: `previous` continues, with its baselines dropped.
    Different settings: counts are not comparable and never summed; the old tracker's last
    snapshot is reported and moved to `final['evidence_previous']`. Disabled: its last
    snapshot stays in `final['evidence']`. Returns the tracker the new bot must use.
    """
    if previous is not None and current is not None and current.cfg == previous.cfg:
        previous.reset_baseline()
        return previous
    if previous is not None:
        final["evidence"] = previous.snapshot()
        report(("evidence settings changed" if current is not None else "evidence disabled")
               + "; counts so far are kept: " + summary_line(final["evidence"]))
    if current is not None:
        if (final.get("evidence") or {}).get("frames_observed"):
            final.setdefault("evidence_previous", []).append(final["evidence"])
        final["evidence"] = current.snapshot()
    return current


def record_snapshot(final: dict, snapshot: dict | None) -> dict | None:
    """Store the live snapshot in `final`; a None (disabled) never replaces counts already seen."""
    if snapshot is not None or final.get("evidence") is None:
        final["evidence"] = snapshot
    return snapshot


def session_summary_lines(final: dict) -> list[str]:
    """Summary lines for the stop message: the latest snapshot first, then each earlier one."""
    earlier = list(final.get("evidence_previous") or [])
    return [summary_line(final.get("evidence"))] + [
        f"earlier, under different evidence settings ({i} of {len(earlier)}): " + summary_line(snap)
        for i, snap in enumerate(earlier, 1)]


class EvidenceTracker:
    """Counts HUD gain observations and seconds per activity bucket. `cfg` is the `evidence:` section.

    `update` must be called once per processed capture, with `fresh=False` whenever the
    frame is not a new capture of the recognised map or the bot is paused. Every such
    call, a time gap above `max_gap_s`, and a non-advancing clock drop the comparison
    baselines, so a pause or a map change cannot fabricate an event. A counter whose
    rect does not fit the frame is reported as None (unavailable), never as 0.
    """

    def __init__(self, cfg: dict | None, wall_clock=time.time):
        cfg = dict(cfg or {})
        self.cfg = cfg
        self._wall = wall_clock
        self.exp_rect = cfg.get("exp_text_rect")
        self.pixel_delta = int(cfg.get("exp_text_pixel_delta", 24))
        self.min_pixels = int(cfg.get("exp_text_min_pixels", 10))
        self.line_rect = cfg.get("gain_line_rect")
        self.pitch = int(cfg.get("gain_line_pitch", 14))
        self.slots = max(2, min(4, int(cfg.get("gain_line_slots", 3))))
        self.line_args = dict(fresh_min=int(cfg.get("gain_line_fresh_pixels", 40)),
                              present_max=int(cfg.get("gain_line_empty_pixels", 15)),
                              shift_min=float(cfg.get("gain_line_shift_score", 0.45)),
                              shift_margin=float(cfg.get("gain_line_shift_margin", 0.15)))
        self.thresholds = {"exp": float(cfg.get("exp_line_score", 0.5)),
                           "meso": float(cfg.get("meso_line_score", 0.45))}
        self.max_gap_s = float(cfg.get("max_gap_s", 2.0))
        self.max_dt_s = float(cfg.get("max_dt_s", 1.0))
        paths = {kind: path if os.path.isabs(str(path)) else os.path.join(PACKAGE_ROOT, str(path))
                 for kind, path in (cfg.get("templates") or {}).items() if path}
        self.templates = load_gain_templates(paths)
        self.classifies = "exp" in self.templates and "meso" in self.templates
        if not self.classifies:
            log.warning("evidence: gain-line templates missing (%s); line kinds are unavailable", paths)
        self.exp_text_changes = self.gain_lines_total = 0
        self.kinds = {"exp": 0, "meso": 0, "other": 0}
        self.exp_available: bool | None = None   # None until a fresh frame has been examined
        self.lines_available: bool | None = None
        self.last_exp_text_change_at = self.last_gain_line_at = self.last_gain_line_kind = None
        self.frames_observed = self.baseline_resets = self.exp_text_unreadable_frames = 0
        self.seconds = dict.fromkeys(BUCKETS, 0.0)
        self.error: str | None = None
        self._t: float | None = None
        self._exp_base = self._exp_candidate = self._line_base = None

    def reset_baseline(self) -> None:
        """Forget the previous crops and time; the next fresh frame only re-establishes them."""
        if self._t is not None or self._exp_base is not None or self._line_base is not None:
            self.baseline_resets += 1
        self._t = self._exp_base = self._exp_candidate = self._line_base = None

    def update(self, frame, now: float, brain_state, fresh: bool = True) -> None:
        try:
            self._update(frame, float(now), brain_state, bool(fresh))
        except Exception as error:  # telemetry must never stop the control loop
            if self.error is None:
                log.warning("evidence tracker error: %r", error)
            self.error = repr(error)
            self._t = self._exp_base = self._exp_candidate = self._line_base = None

    def _update(self, frame, now: float, brain_state, fresh: bool) -> None:
        if not fresh or frame is None:
            self.reset_baseline()
            return
        if self._t is not None:
            dt = now - self._t
            if dt <= 0:
                return  # the same instant twice is not a new observation
            if dt > self.max_gap_s:
                self.reset_baseline()
            else:
                self.seconds[activity_bucket(brain_state)] += min(dt, self.max_dt_s)
        self._t = now
        self.frames_observed += 1
        self._observe_exp(frame)
        self._observe_lines(frame)

    def _observe_exp(self, frame) -> None:
        crop = crop_rect(frame, self.exp_rect)
        self.exp_available = crop is not None
        if crop is None or not exp_text_plausible(crop):
            self.exp_text_unreadable_frames += crop is not None
            self._exp_base = self._exp_candidate = None
            return
        if self._exp_base is None:
            self._exp_base = crop.copy()
        elif changed_pixel_count(self._exp_base, crop, self.pixel_delta) < self.min_pixels:
            self._exp_candidate = None  # unchanged, or a one-frame overlay went away
        elif (self._exp_candidate is not None
                and changed_pixel_count(self._exp_candidate, crop, self.pixel_delta) < self.min_pixels):
            # The new text was drawn identically on two consecutive fresh frames.
            self.exp_text_changes += 1
            self.last_exp_text_change_at = self._wall()
            self._exp_base, self._exp_candidate = crop.copy(), None
        else:
            self._exp_candidate = crop.copy()

    def _observe_lines(self, frame) -> None:
        slots = None
        try:
            x, y, w, h = (int(v) for v in self.line_rect)
            slots = [crop_rect(frame, (x, y - self.pitch * k, w, h)) for k in range(self.slots)]
        except (TypeError, ValueError):
            pass
        self.lines_available = slots is not None and all(s is not None for s in slots)
        if not self.lines_available:
            self._line_base = None
            return
        if self._line_base is not None:
            arrived = gain_line_arrivals(self._line_base, slots, **self.line_args)
            for k in range(arrived):
                kind = classify_gain_line(slots[k], self.templates, self.thresholds) if self.classifies else None
                self.gain_lines_total += 1
                if kind is not None:
                    self.kinds[kind] += 1
                self.last_gain_line_at, self.last_gain_line_kind = self._wall(), kind
        self._line_base = [s.copy() for s in slots]

    def snapshot(self) -> dict:
        """Plain JSON values. None means "unavailable on this layout", not zero."""
        lines = self.gain_lines_total if self.lines_available else None
        kind = lambda name: self.kinds[name] if self.lines_available and self.classifies else None  # noqa: E731
        return {
            "note": NOTE,
            "exp_text_changes": self.exp_text_changes if self.exp_available else None,
            "gain_lines_total": lines,
            "exp_lines": kind("exp"), "meso_lines": kind("meso"), "other_lines": kind("other"),
            "last_exp_text_change_at": self.last_exp_text_change_at,
            "last_gain_line_at": self.last_gain_line_at,
            "last_gain_line_kind": self.last_gain_line_kind,
            "frames_observed": self.frames_observed,
            "baseline_resets": self.baseline_resets,
            "exp_text_unreadable_frames": self.exp_text_unreadable_frames,
            "seconds": {name: round(value, 2) for name, value in self.seconds.items()},
            "seconds_total": round(sum(self.seconds.values()), 2),
            "error": self.error,
        }
