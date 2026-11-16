# Recorded controller statuses for `tests/test_supervise_farming.py`

`recorded_statuses.json` holds fourteen status objects exactly as `scripts/run_active.py` published them
(`log/active-status.json`). Nothing in them was edited; every entry names its source. They are the base of the
supervisor's decision tests and of the statuses its fake controller publishes, so that the tests classify what the
controller really writes. All of them come from the supervised Kerning City Construction Site runs of
2026-11-15 and 2026-11-16 on the development machine.

| Entry | Source (file under `log/` : line) | What it shows |
| --- | --- | --- |
| `waiting` | `claude-circuits-6.jsonl:3` | fully paused, waiting for the first start |
| `checking` | `claude-circuits-6.jsonl:4` | guarded start, first phase: the start's own hold |
| `returning` | `claude-circuits-6.jsonl:5` | guarded start, second phase: returning to point 1 |
| `farming` | `claude-circuits-6.jsonl:64` | farming |
| `farming_background` | `claude-circuits-6.jsonl:98` | farming while the game is not in front |
| `farming_map_lost` | `claude-circuits-6.jsonl:655` | farming during a short map reacquisition (`hp` and `minimap` null) |
| `hp_hold` | `claude-circuits-6.jsonl:1040` | the controller's own low-HP hold during farming (`healing`) |
| `hp_hold_waiting` | `claude-circuits-6.jsonl:1041` | the same hold one status later (`awaiting_response`) |
| `automatic_return` | `claude-circuits-6.jsonl:1589` | the controller's own return to point 1 after a blocked waypoint |
| `hp_hold_during_return` | `claude-circuits-6.jsonl:1628` | low-HP hold while that automatic return is on |
| `route_hold` | `claude-circuits-6.jsonl:1651` (last sample) | the return failed: the controller's own route-failure hold |
| `failed_start` | `claude-circuits-5.jsonl:22` (last sample) | a guarded start failed and asks for the user; it must not be resumed |
| `watcher_hold` | `full-circuit-watch-v4.jsonl:663` | first status after the lap watcher's Hold farming request |
| `full_pause_no_snapshot` | `log/active-status.json`, read (read-only) on 2026-11-16 | full pause with a stale capture: no `hp`, `minimap`, `player`, `command` or `fps` key |

The traces themselves stay in the untracked `log/` folder. `full_pause_no_snapshot` is the only entry that is not a
row of a trace: it is the live status file of the same controller process (pid 18480) as it stood several hours after
`claude-circuits-6.jsonl` ended, fully paused, with the route failure still in `recovery_terminal`.

Statuses the tests need but no trace contains (a user's F1 pause in the middle of a session, a start cancelled by
Hold farming, a failed start with a transient reason) are built in the test from these rows by changing exactly
the fields that `maplebot/bot.py` (`pause_all`, `hold_farming`) and `maplebot/recovery_runtime.py`
(`StartupReturn.cancel`, `_fail`) change; the test says so where it does.
