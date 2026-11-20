"""Record a farming session; with --auto-resume, also ask the controller to start again after a route failure.

    python scripts/supervise_farming.py --minutes 60 --output log/session.jsonl
    python scripts/supervise_farming.py --minutes 60 --output log/session.jsonl --auto-resume 4

Without ``--auto-resume`` this is a passive recorder. It reads ``log/active-status.json`` every 0.25 s (every
0.05 s while the file cannot be read) and writes each status it has not read before to its ``--output`` file (rows ``{"event": "sample", "elapsed_s": ...,
"status": ...}``). The controller can publish faster than that, so some statuses are never read or recorded. It
creates no other file and sends nothing. ``--output`` must be a new file whose name ends in ``.jsonl``; nothing is
ever overwritten, and the name can therefore never be one of the controller's request files.

With ``--auto-resume N`` it may also create ``log/resume-active``, the controller's Resume request: at most N
times in any hour, at most ``--max-resumes`` times per session, never twice within 120 s, and only in two states.

* The controller itself stopped farming because its route failed for good: paused, no full pause, no manual
  hold, ``recovery_terminal`` true.
* A guarded start that this tool had requested, and that a status showed running, failed for a transient route
  reason (a waypoint, jump, ramp or drop timeout and the like, see TRANSIENT_START_FAILURES). It is retried once
  per route failure.

The controller then runs its ordinary guarded start (fresh frame, recognised map, game in front, HP check,
physical return to point 1). That is not what the settings app's Resume button does: the button also removes
pending pause, hold and return requests and creates ``arm-active``, and in an automatic route-failure hold the
app offers no direct Resume at all (a user would press Pause all, then Resume). Resuming straight out of that
automatic hold is covered by offline tests (``tests/test_startup_return.py``); it has not been verified in the
live game.

Before a request the hold must have lasted 20 s, and every status the tool read during the last 5 s (at least
three) must have been at most 1 s old when it was read and must have shown the same kind of hold, HP protection
``monitoring`` without a reason, HP from ``--min-hp`` to 100, the game in front, the map matched, a fresh
capture and a minimap position, while none of the request files in BLOCKING_REQUESTS existed. One status that
fails this starts the 5 s again. Immediately before the request the tool reads the status file, the request
files and the off switch once more, and it creates the request only if the status is unchanged, at most 0.35 s
old and still eligible.

What it cannot see: a Pause all, an F1 or any other change that the controller has already applied but not yet
published. Since the status must be at most 0.35 s old, that is a change made at most 0.35 s before the request.
The controller then takes the request and runs a guarded start out of that pause, and because the pause never
appeared in a status, resuming stays on. Pause again once the start is visible; that pause is seen and ends
resuming. Only the controller could close this gap.

After a request the tool watches for 5 s. If the file went and a start became visible, the request is confirmed.
If the file went but no start became visible, it stops resuming. If the file is still there after 5 s, the
controller did not take it: the tool deletes it if it is unchanged (a file somebody else touched is left alone),
warns and stops resuming. When resuming stops for any other reason, or the tool ends (time limit, Ctrl+C, error,
controller stopped or restarted, no status for 30 s), while its own request is still there untaken and unchanged,
the tool deletes it as well. Its own, unchanged ``log/resume-active`` is the only file this tool ever deletes. A
request the controller has already taken cannot be withdrawn, and a tool that is killed outright deletes nothing.

Resuming ends for the rest of the session, with a printed reason, as soon as the tool reads any of these: the off
switch; a full pause (Pause all, F1, or the controller's own stop after it lost the map or the capture), a
manual hold (Hold farming) or a connection test, once the controller has been active in this session (farming,
a start, a route-failure hold or the controller's own low-HP hold was seen; before that these only make the tool
wait, so a session may begin while the controller is still paused); a failed start the tool may not retry; an HP
protection alert, or a protection state other than monitoring, healing, awaiting_response or suspended, in a
route-failure, low-HP or failed-start hold; a route blocked without a return to point 1 (``return_on_block`` off
is not supported); a status that does not say the controller runs; a request that was not taken or not followed
by a visible start; the session limit, once its last request has been watched. Everything else only makes it
wait: low HP, a potion, a start or a return in progress, a game that is not in front, a status that is stale or
not eligible, another pending request, the 120 s spacing and the hourly allowance. Once a route-failure or
failed-start hold has lasted 20 s, each reason for which the tool still waits is printed once for that hold.

How to stop it. Hold farming cannot stop this tool in the two states in which it acts. In a route-failure hold
and in a failed-start hold the settings app disables that button, and ``EditorBridge.hold_farming()`` answers
"already held" without writing a request, so nothing changes that the tool could see. The ways to stop it that
keep HP protection are:

* Create the file ``log/supervise-off`` (any content; its full path is printed at the start). The tool only reads
  it. Within 0.25 s it ends resuming for the session, withdraws its own request if the controller has not taken
  it yet, and goes on recording. Seeing it once is enough; delete it yourself before a later session that should
  resume.
* Or end the tool with Ctrl+C. It withdraws its own untaken request in the same way.

Neither stops a start the controller has already taken; ``EditorBridge.hold_farming()`` while that start is
running cancels it, keeps HP protection and also ends resuming. Pause all, F1 and Stop stop the tool as well, but
they switch HP protection off.

The session ends at the time limit, when the controller reports that it stopped, when the controller's pid
changes, or when no new readable status has arrived for 30 s. The tool never sends a key, never starts a
process and writes nothing in ``log/`` except ``resume-active`` (and its own ``--output`` file if that is put
there).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

STATUS_FILE, RESUME_FILE, OFF_SWITCH = "active-status.json", "resume-active", "supervise-off"
# Requests somebody else has queued for the controller: while one exists this tool adds nothing. arm-active is
# left out on purpose: the settings app's Resume leaves it behind for the controller's whole lifetime.
BLOCKING_REQUESTS = ("pause-active", "stop-active", "hold-farming-active", "return-to-start-active",
                     "reload-active", "resume-active", "editor-test-request.json", "stop-editor-test")
START_STOPPED = "Start stopped: "
# The only failed guarded starts that may be retried: route trouble that a second attempt can get past.
TRANSIENT_START_FAILURES = ("waypoint timeout", "jump timeout", "ramp timeout", "drop timeout", "drop stalled",
                            "Return route timed out", "Stable source unavailable", "left ramp corridor",
                            "ramp knockback did not settle", "drop did not leave source floor")
POLL_S, RETRY_S, NO_STATUS_S = 0.25, 0.05, 30.0
ACT_AGE_S = 0.35  # the status a request is made on is at most this old (about one publish interval)
_UNSEEN = object()


def _real(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def classify_hold(status: dict) -> tuple[str, str]:
    """(kind, detail). Kinds: stopped, user, starting, start_failed, farming, route, unsupported, healing."""
    if status.get("running") is not True:
        return "stopped", "The controller is not running."
    if status.get("full_pause") is not False:
        extra = status.get("map_reacquire_reason")
        return "user", ("full pause (user, or the controller after losing the map or capture)"
                        + (f": {extra}" if extra else ""))
    if status.get("editor_test"):
        return "user", "A connection test is running."
    manual, startup = status.get("manual_health_hold"), status.get("startup_return_state")
    reason = str(status.get("health_hold_reason") or "")
    if manual is True:
        if startup == "checking":
            return "starting", "A guarded start is checking HP, map and position."  # the start's own hold
        if startup == "failed" and reason.startswith(START_STOPPED):
            return "start_failed", reason[len(START_STOPPED):]
        return "user", "manual hold (Hold farming): " + reason
    if startup in ("checking", "returning") or status.get("returning_to_start"):
        return "starting", "A guarded start or a return to point 1 is in progress."
    if not status.get("paused"):
        return "farming", ""
    if manual is not False:
        return "user", "The kind of hold is unknown; it is treated as the user's."
    if status.get("recovery_terminal") is True:
        return "route", str(status.get("circuit_reason") or status.get("recovery_reason") or "route failure")
    if status.get("circuit_state") == "blocked":
        return "unsupported", "The route is blocked without a return to point 1 (return_on_block off); not supported."
    return "healing", "The controller's own low-HP hold."


def start_visible(status: dict) -> bool:
    """A guarded start or a return to point 1 is running."""
    return status.get("startup_return_state") in ("checking", "returning") or status.get("returning_to_start") is True


def blocker(status: dict, *, age_s: float, min_hp: float, pending: tuple[str, ...] = ()) -> str | None:
    """Why this status of a hold must not lead to a Resume request now; None when it is fully eligible.

    age_s is the wall clock at the read minus the status' "updated".
    """
    if not -1.0 <= age_s <= 1.0:
        return "The status is not fresh."
    guard, hp, age, position = (status.get(key) for key in ("health_protection_state", "hp", "capture_age_s", "minimap"))
    if guard != "monitoring":
        return f"HP protection is {guard}, not monitoring."
    if not _real(hp) or not min_hp <= hp <= 100:
        return f"HP must read {min_hp:g}% to 100%."
    if status.get("foreground") is not True:
        return "The game is not in front."
    if status.get("map_ready") is not True or status.get("map_state") != "matched":
        return "The map is not recognised."
    if status.get("capture_state") != "fresh" or not _real(age) or not 0 <= age <= 1.0:
        return "The capture is not fresh."
    if (status.get("minimap_seen") is not True or not isinstance(position, (list, tuple)) or len(position) != 2
            or not all(_real(value) for value in position)):
        return "The minimap position is not visible."
    if pending:
        return "Another request is pending: " + ", ".join(pending)
    return None


class ResumePolicy:
    """What one session remembers between polls. Pure: main() tells it what it read and when (monotonic seconds)."""

    def __init__(self, *, per_hour: int, max_total: int, min_hp: float = 75.0, settle_s: float = 20.0,
                 window_s: float = 5.0, window_samples: int = 3, min_interval_s: float = 120.0,
                 confirm_s: float = 5.0):
        self.per_hour, self.max_total, self.min_hp = per_hour, max_total, min_hp
        self.settle_s, self.window_s, self.window_samples = settle_s, window_s, window_samples
        self.min_interval_s, self.confirm_s = min_interval_s, confirm_s
        self.resume_times: list[float] = []  # when this session's Resume requests were created
        self.active_seen = False             # farming, a start, a route hold or a potion pause has been seen
        self.kind = None                     # kind of the newest status
        self.updated = _UNSEEN               # its "updated": a status counts once, however often it is read
        self.hold_since = None               # first status of the route or failed-start hold that is on now
        self.window_since, self.window_count = None, 0  # the unbroken run of fully eligible statuses
        self.expect_start = False            # a request of this tool has not shown up as a start yet
        self.own_start = False               # the start that runs or failed now was requested by this tool
        self.retried = False                 # a failed start was already retried since farming last ran
        self.sent_at, self.taken = None, False  # the request being watched; whether its file was seen gone

    def decide(self, status: dict, *, age_s: float, now: float, pending: tuple[str, ...] = (),
               off_switch: bool = False) -> tuple[str, str]:
        """("resume" | "wait" | "never", reason) for the newest readable status; call it on every poll.

        "never" means: stop resuming for the rest of the session.
        """
        kind, detail = classify_hold(status)
        new = status.get("updated") != self.updated
        if new:
            self.updated = status.get("updated")
            if kind != self.kind:
                self.hold_since = now if kind in ("route", "start_failed") else None
                self.window_since, self.window_count = None, 0
                if kind == "starting":  # a new start: the tool's own only if it follows the tool's request
                    self.own_start, self.expect_start = self.expect_start, False
            if kind == "farming":
                self.own_start = self.retried = self.expect_start = False
            self.active_seen = self.active_seen or kind in ("farming", "starting", "route", "healing")
            self.kind = kind
        if off_switch:
            return "never", f"The off switch log/{OFF_SWITCH} exists."
        used_up = len(self.resume_times) >= self.max_total
        if used_up and self.sent_at is None:  # never while the last request is watched: that would withdraw it
            return "never", f"The session limit of {self.max_total} Resume request(s) is used up."
        if kind in ("stopped", "unsupported") or (kind == "user" and self.active_seen):
            return "never", detail
        if kind in ("user", "farming", "starting"):
            return "wait", detail or "The controller is farming."
        guard, alert = status.get("health_protection_state"), status.get("health_protection_reason")
        if alert or guard not in ("monitoring", "healing", "awaiting_response", "suspended"):
            return "never", f"HP protection needs attention: {alert or guard}"
        if kind == "healing":
            return "wait", detail
        if kind == "start_failed":
            if not self.own_start:  # also the tool's own start when it failed before a status showed it running
                return "never", "A start that this tool cannot confirm as its own has failed: " + detail
            if not any(text in detail for text in TRANSIENT_START_FAILURES):
                return "never", "The failed start needs the user: " + detail
            if self.retried:
                return "never", "The retried start failed again: " + detail
        problem = blocker(status, age_s=age_s, min_hp=self.min_hp, pending=pending)
        if problem:
            self.window_since, self.window_count = None, 0
            return "wait", problem
        if self.window_since is None:
            self.window_since = now
        if new:
            self.window_count += 1
        if now - self.hold_since < self.settle_s:
            return "wait", f"The hold must last {self.settle_s:g} s first."
        if now - self.window_since < self.window_s or self.window_count < self.window_samples:
            return "wait", f"Waiting for {self.window_s:g} s of unbroken, fully eligible statuses."
        if self.resume_times and now - self.resume_times[-1] < self.min_interval_s:
            return "wait", f"Less than {self.min_interval_s:g} s since the previous Resume request."
        if sum(now - sent < 3600.0 for sent in self.resume_times) >= self.per_hour:
            return "wait", f"{self.per_hour} Resume request(s) were made in the last hour; the allowance comes back."
        if used_up:
            return "wait", "The last permitted Resume request is still being watched."
        return "resume", f"{kind}: {detail}"

    def requested(self, now: float) -> None:
        """The request file was just created: count it and watch what the controller does with it."""
        self.resume_times.append(now)
        self.retried = self.retried or self.kind == "start_failed"
        self.expect_start, self.sent_at, self.taken = True, now, False

    def watch(self, status: dict, *, now: float, on_disk: bool) -> str | None:
        """Follow the tool's own request; None while it is open or when there is none, else how it ended.

        "confirmed": the file went and a start is visible. "not_consumed": the file is still there after
        confirm_s; the caller removes it. "unconfirmed": the file went but no start became visible. The last
        two end resuming for the session.
        """
        if self.sent_at is None:
            return None
        self.taken = self.taken or not on_disk
        if self.taken and start_visible(status):
            outcome = "confirmed"
        elif now - self.sent_at < self.confirm_s:
            return None
        else:
            outcome, self.expect_start = "unconfirmed" if self.taken else "not_consumed", False
        self.sent_at = None
        return outcome

    def withdrawn(self) -> None:
        """main() deleted the tool's own request before the controller took it: nothing is left to watch."""
        self.sent_at, self.expect_start = None, False


def read_status(path: Path) -> dict | None:
    """The controller's status, or None while the file is missing, half written or not a status."""
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return status if isinstance(status, dict) and _real(status.get("updated")) else None


def _mtime(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def remove_own_request(path: Path, created_mtime: int | None) -> bool:
    """Delete log/resume-active only while it is still the file this tool created (same modification time)."""
    if created_mtime is None or _mtime(path) != created_mtime:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=".", help="repository root (holds log/)")
    parser.add_argument("--output", required=True, help="new .jsonl file for the recording; never overwritten")
    parser.add_argument("--minutes", type=float, required=True, help="length of the session, at most 600")
    parser.add_argument("--auto-resume", type=int, default=0, metavar="N",
                        help="create log/resume-active at most N times per hour after a route failure "
                             "(0..6; 0, the default, only records)")
    parser.add_argument("--max-resumes", type=int, default=12, help="Resume requests per session (0..30)")
    parser.add_argument("--min-hp", type=float, default=75.0, help="lowest HP reading for a Resume request (1..100)")
    args = parser.parse_args(argv)
    if (not 0 < args.minutes <= 600 or not 0 <= args.auto_resume <= 6 or not 0 <= args.max_resumes <= 30
            or not 1 <= args.min_hp <= 100):
        parser.error("--minutes must be in (0, 600], --auto-resume in 0..6, --max-resumes in 0..30, --min-hp in 1..100")
    if not args.output.lower().endswith(".jsonl"):
        parser.error("--output must end in .jsonl, so that it can never be one of the controller's request files")
    log = Path(args.root) / "log"
    status_path, resume_path, off_path = log / STATUS_FILE, log / RESUME_FILE, (log / OFF_SWITCH).resolve()
    try:
        stream = open(args.output, "x", encoding="utf-8")
    except OSError as error:
        parser.error(f"--output cannot be created; an existing file is never overwritten ({error})")
    policy = ResumePolicy(per_hour=args.auto_resume, max_total=args.max_resumes, min_hp=args.min_hp)
    resuming = args.auto_resume > 0
    started = time.monotonic()  # durations use the monotonic clock, status freshness the wall clock
    deadline, last_new = started + args.minutes * 60, started
    last = pid = own_mtime = note = told_hold = None
    rows, reason, withdrawn, told = 0, "time limit", False, set()
    with stream:
        def record(event, **fields):
            stream.write(json.dumps({"event": event, **fields}, ensure_ascii=False) + "\n")
            stream.flush()

        def switch_off(why):
            nonlocal resuming
            if resuming:
                resuming = False
                elapsed = time.monotonic() - started
                record("auto_resume_off", elapsed_s=elapsed, reason=why)
                extra = ""
                if policy.sent_at is not None and not policy.taken:  # its own request still waits: take it back
                    removed = remove_own_request(resume_path, own_mtime)
                    record("resume_withdrawn", elapsed_s=elapsed, removed=removed)
                    if removed:
                        policy.withdrawn()
                        extra = " Its own Resume request, not taken yet, was withdrawn."
                print(f"auto-resume off for the rest of this session: {why}{extra}", flush=True)

        def decide():
            pending = tuple(name for name in BLOCKING_REQUESTS if (log / name).exists())
            return policy.decide(last, age_s=time.time() - last["updated"], now=time.monotonic(), pending=pending,
                                 off_switch=off_path.exists())

        record("supervisor_started", started=time.time(), minutes=args.minutes, auto_resume_per_hour=args.auto_resume,
               max_resumes=args.max_resumes, min_hp=args.min_hp)
        if resuming:
            print(f"auto-resume on: at most {args.auto_resume} Resume request(s) per hour, {args.max_resumes} in this "
                  f"session. To stop it and keep HP protection, create {off_path} or end this tool (Ctrl+C); "
                  "Hold farming does not stop it in a route-failure or failed-start hold.", flush=True)
        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                status = read_status(status_path)
                if status is not None and (last is None or status["updated"] != last["updated"]):
                    last, last_new, rows = status, now, rows + 1
                    record("sample", elapsed_s=now - started, status=status)
                    if status.get("running") is False:
                        reason = "controller stopped"
                        break
                    pid = status.get("pid") if rows == 1 else pid
                    if status.get("pid") != pid:
                        reason = "controller restarted"
                        break
                elif now - last_new > NO_STATUS_S:
                    reason = f"no new status for {NO_STATUS_S:g} s"
                    break
                if last is not None and policy.sent_at is not None:  # the tool's own request is still open
                    outcome = policy.watch(last, now=now, on_disk=resume_path.exists())
                    if outcome == "confirmed":
                        record("resume_confirmed", elapsed_s=now - started)
                    elif outcome is not None:
                        removed = outcome == "not_consumed" and remove_own_request(resume_path, own_mtime)
                        record("resume_" + outcome, elapsed_s=now - started, removed=removed)
                        why = (f"The controller did not take {RESUME_FILE} within {policy.confirm_s:g} s"
                               + ("; the request was removed again." if removed else "; the file was left alone.")
                               if outcome == "not_consumed" else
                               f"{RESUME_FILE} was taken, but no start became visible within {policy.confirm_s:g} s.")
                        print("WARNING: " + why, flush=True)
                        switch_off(why)
                if last is not None and resuming:
                    decision, why = decide()
                    if decision == "resume":  # read everything once more immediately before the request
                        again = read_status(status_path)
                        if again is None or again["updated"] != last["updated"]:
                            decision, why = "wait", "The status changed while deciding."
                        elif time.time() - again["updated"] > ACT_AGE_S:
                            decision, why = "wait", (f"The status is more than {ACT_AGE_S:g} s old; "
                                                     "the request waits for a newer one.")
                        else:
                            decision, why = decide()
                    if decision == "resume":
                        try:
                            resume_path.touch(exist_ok=False)
                        except FileExistsError:
                            decision, why = "wait", "Another Resume request appeared first."
                        except OSError as error:
                            decision, why = "never", f"{RESUME_FILE} could not be created: {error}"
                        else:
                            own_mtime = _mtime(resume_path)
                            policy.requested(now)
                            record("resume_requested", elapsed_s=now - started, time=time.time(),
                                   count=len(policy.resume_times), reason=why, position=last.get("minimap"),
                                   hp=last.get("hp"))
                            print(f"Resume requested ({len(policy.resume_times)}): {why}", flush=True)
                    if decision == "never":
                        switch_off(why)
                    elif decision == "wait":
                        if last.get("paused") and why != note:
                            note = why
                            record("resume_waiting", elapsed_s=now - started, reason=why)
                        hold = policy.hold_since
                        if hold is not None and now - hold >= policy.settle_s:  # a long wait: say why, once a hold
                            if hold != told_hold:
                                told, told_hold = set(), hold
                            if why not in told:
                                told.add(why)
                                print(f"auto-resume waiting ({policy.kind} hold for {now - hold:.0f} s): {why}",
                                      flush=True)
                time.sleep(POLL_S if status is not None else RETRY_S)
        except KeyboardInterrupt:
            reason = "interrupted"
        except BaseException as error:
            reason = f"error: {error!r}"
            raise
        finally:
            if policy.sent_at is not None and not policy.taken:  # never leave an unwatched request behind
                withdrawn = remove_own_request(resume_path, own_mtime)
            record("supervisor_finished", reason=reason, rows=rows, resumes=len(policy.resume_times),
                   own_request_withdrawn=withdrawn, finished=time.time())
    print(f"supervisor finished: {reason}; {rows} samples, {len(policy.resume_times)} Resume request(s), "
          f"output {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
