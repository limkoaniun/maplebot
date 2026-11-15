"""Active runtime shared by the desktop shortcuts and settings window."""
from collections import Counter
from pathlib import Path
import json
import os
import sys
import time
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
LOG = ROOT / 'log'
LOG.mkdir(exist_ok=True)
output = open(LOG / 'active.log', 'a', encoding='utf-8', buffering=1)
sys.stdout = sys.stderr = output

from maplebot.bot import build_bot
from maplebot.config import load_config
from maplebot.editor_bridge import EditorBridge, capture_signature
from maplebot.editor_test_runtime import EditorTestSession
from maplebot.evidence import carry_over_tracker, record_snapshot, session_summary_lines
from maplebot.recovery_runtime import request_return, request_hold, StartupReturn
from maplebot.overlay import draw_debug, route_preview
from maplebot.vision.bars import find_status_bars

STOP = LOG / 'stop-active'
PAUSE = LOG / 'pause-active'
RESUME = LOG / 'resume-active'
RELOAD = LOG / 'reload-active'
ARM = LOG / 'arm-active'
RETURN_START = LOG / 'return-to-start-active'
HOLD = LOG / 'hold-farming-active'
STATUS = LOG / 'active-status.json'
TEST_REQUEST = LOG / 'editor-test-request.json'
TEST_STOP = LOG / 'stop-editor-test'
TEST_STATUS = LOG / 'editor-test-status.json'

def publish(status):
    status.update(pid=os.getpid(), updated=time.time())
    temporary = STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(status), encoding='utf-8')
    try:
        temporary.replace(STATUS)
    except PermissionError:
        pass  # A Windows reader can briefly hold the destination; retry next tick.


def atomic_diagnostic(path, data):
    """Skip a busy viewer file instead of interrupting the control loop."""
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        temporary.write_bytes(data)
        for attempt in range(3):
            try:
                temporary.replace(path)
                return True
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.005 * (attempt + 1))
        return False
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def publish_png(name, frame):
    ok, encoded = cv2.imencode('.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return ok and atomic_diagnostic(LOG / name, encoded.tobytes())


def box_values(box):
    return None if box is None else [int(box.x), int(box.y), int(box.w), int(box.h)]


def snapshot_details(bot, snap):
    fix = bot.player if snap is not None else None
    circuit = getattr(bot.brain, 'circuit', None)
    loot = getattr(bot.brain, 'loot', None)
    return {
        'capture_state': bot.capture_state,
        'capture_sequence': bot.capture_sequence,
        'capture_age_s': (None if bot.capture_time is None else max(0, time.monotonic() - bot.capture_time)),
        'map_ready': bot.map_ready,
        'map_state': bot.map_match.state if bot.map_match is not None else 'not_checked',
        'map_name': bot.map_match.profile_name if bot.map_match is not None else None,
        'map_score': bot.map_match.score if bot.map_match is not None else None,
        'map_reason': bot.map_match.reason if bot.map_match is not None else None,
        'map_reacquiring': getattr(bot, 'map_reacquiring', False),
        'map_reacquire_reason': getattr(bot, 'map_reacquire_reason', None),
        'minimap_offset': getattr(bot, 'minimap_offset', (0, 0)),
        'minimap_raw': getattr(bot, 'minimap_raw_pos', None),
        'vertical_minimap_active': bool(getattr(bot, '_vertical_minimap_active', False)),
        'minimap_route_active': bool(getattr(bot, '_minimap_route_active', False)),
        'circuit_enabled': circuit is not None,
        'circuit_index': circuit.idx if circuit is not None else None,
        'circuit_count': len(circuit.waypoints) if circuit is not None else 0,
        'circuit_state': circuit.state if circuit is not None else None,
        'brain_state': bot.brain.state,
        'circuit_reason': circuit.reason if circuit is not None else None,
        'blocked_origin_reason': circuit.blocked_origin_reason if circuit is not None else None,
        'full_pause': bot.full_pause,
        'manual_health_hold': bot._manual_health_hold,
        'health_hold_reason': bot.health_hold_reason,
        'hold_request_error': getattr(bot, '_hold_request_error', None),
        'health_protection_active': bot.health_protection_active,
        'health_protection_state': bot.health_protection.state if bot.health_protection is not None else 'idle',
        'health_protection_reason': bot.health_protection.reason if bot.health_protection is not None else None,
        'health_protection_taps': bot.keyboard.hp_taps_sent if bot.keyboard is not None else 0,
        'returning_to_start': bool(circuit is not None and circuit.returning_to_start),
        'recovery_state': circuit.recovery_state if circuit is not None else 'idle',
        'recovery_reason': getattr(bot, '_return_request_error', None) or (circuit.recovery_reason if circuit is not None else None),
        'recovery_from_index': circuit.recovery_from_index if circuit is not None else None,
        'return_count': circuit.return_count if circuit is not None else 0,
        'recovery_terminal': bool(circuit is not None and circuit.recovery_terminal),
        'recovery_index': circuit.recovery.idx if circuit is not None and circuit.recovery is not None else None,
        'recovery_count': len(circuit.recovery.waypoints) if circuit is not None and circuit.recovery is not None else 0,
        'rejoin_active': bool(getattr(circuit, 'rejoining', False)),
        'rejoin_count': getattr(circuit, 'rejoin_count', 0),
        'rejoin_state': getattr(circuit, 'rejoin_state', 'idle'),
        'rejoin_reason': getattr(circuit, 'rejoin_reason', None),
        'arrival_count': getattr(circuit, 'arrival_count', 0),
        'loot_state': loot.state if loot is not None else None,
        'player_score': float(fix.score) if fix is not None else None,
        'player_source': fix.source if fix is not None else None,
        'missing_frames': bot._missing_frames,
        'minimap_seen': bool(bot._minimap_seen and snap is not None),
        'target': snap.target if snap is not None else None,
        'route_loaded': bot.route_map is not None,
        'minimap_box': box_values(snap.minimap_box) if snap is not None else None,
        'search_box': box_values(snap.search_box) if snap is not None else None,
        'face_x': snap.command.face_x if snap is not None else 'none',
        'detections': [
            {'name': det.name, 'score': float(det.score), 'box': box_values(det.box)}
            for det in snap.detections
        ] if snap is not None else [],
    }


def canonical_minimap(snap, bot):
    if snap.minimap_box is None:
        return None
    aligned = getattr(bot, 'minimap_image', None)
    if aligned is not None:
        return aligned.copy()
    box = snap.minimap_box.clamp(snap.frame.shape[1], snap.frame.shape[0])
    return snap.frame[box.y:box.y2, box.x:box.x2].copy()


def diagnostic_minimap(snap, bot):
    preview = np.full((294, 333, 3), (40, 31, 24), dtype=np.uint8)
    if snap.minimap_box is not None:
        crop = canonical_minimap(snap, bot)
        if crop.size:
            if snap.minimap_pos is not None:
                point = (int(snap.minimap_pos[0]), int(snap.minimap_pos[1]))
                cv2.circle(crop, point, 3, (0, 0, 255), 1)
            scale = min(333 / crop.shape[1], 294 / crop.shape[0])
            width, height = max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))
            ox, oy = (333 - width) // 2, (294 - height) // 2
            preview[oy:oy+height, ox:ox+width] = cv2.resize(crop, (width, height), interpolation=cv2.INTER_NEAREST)
            circuit = getattr(bot.brain, 'circuit', None)
            if circuit is not None and circuit.returning_to_start and circuit.recovery is not None:
                circuit = circuit.recovery
            elif circuit is not None and getattr(circuit, 'rejoining', False) and circuit.platform_rejoin is not None:
                circuit = circuit.platform_rejoin
            if circuit is not None:
                scale_x, scale_y = width / crop.shape[1], height / crop.shape[0]
                for platform in circuit.platforms:
                    vertices = [(platform.left, platform.y), *platform.knots,
                                (platform.right, platform.y_at(platform.right))]
                    line = [(ox + round(x * scale_x), oy + round(y * scale_y)) for x, y in vertices]
                    cv2.polylines(preview, [np.array(line, dtype=np.int32)], False, (130, 220, 130), 1)
                points = [(ox + round(w.x * scale_x), oy + round(w.y * scale_y)) for w in circuit.waypoints]
                if len(points) > 1:
                    cv2.polylines(preview, [np.array(points, dtype=np.int32)], circuit.loop, (180, 180, 80), 1)
                for i, point in enumerate(points):
                    cv2.circle(preview, point, 4 if i == circuit.idx else 2,
                               (0, 120, 255) if i == circuit.idx else (230, 230, 230), 1)
                if points:
                    cv2.putText(preview, f'Point {circuit.idx + 1}/{len(points)}  {circuit.state}',
                                (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)
    else:
        cv2.putText(preview, 'Minimap not located', (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if snap.route_map is not None:
        route = route_preview(snap.route_map, snap.route_idx or 0, snap.global_pos)
        panel = np.full_like(preview, (40, 31, 24))
        if route is not None:
            scale = min(333 / route.shape[1], 294 / route.shape[0])
            route = cv2.resize(route, (max(1, round(route.shape[1] * scale)),
                                       max(1, round(route.shape[0] * scale))),
                               interpolation=cv2.INTER_NEAREST)
            panel[:route.shape[0], :route.shape[1]] = route
        else:
            cv2.putText(panel, 'Route position not located', (8, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        preview = np.hstack((preview, panel))
    return preview


def diagnostic_bars(snap, cfg):
    """Show the exact pixels supplied to the runtime bar-fill reader."""
    panel = np.full((150, 400, 3), (40, 31, 24), dtype=np.uint8)
    health = cfg['health']
    rects = health.get('fixed_rects')
    if not rects:
        y0 = int(health.get('ui_y_start', 0))
        y0 = y0 if 0 <= y0 < snap.frame.shape[0] else 0
        found = find_status_bars(snap.frame[y0:], int(health.get('bar_white_thresh', 240)))
        rects = [(b.x, b.y + y0, b.w, b.h) for b in found] if len(found) == 3 else []
    for i, (name, value) in enumerate(zip(('HP', 'MP', 'EXP'), (snap.hp, snap.mp, snap.exp))):
        text = f'{name}: {value:.1f}%' if value is not None else f'{name}: unreadable'
        crop = None
        if i < len(rects):
            x, y, width, height = map(int, rects[i])
            text += f'  [{x}, {y}, {width}, {height}]'
            crop = snap.frame[y:y + height, x:x + width]
        cv2.putText(panel, text, (8, 16 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (255, 255, 255), 1, cv2.LINE_AA)
        if crop is not None and crop.size:
            scale = min(384 / crop.shape[1], 27 / crop.shape[0])
            enlarged = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                         max(1, round(crop.shape[0] * scale))),
                                 interpolation=cv2.INTER_NEAREST)
            panel[21 + 50 * i:21 + 50 * i + enlarged.shape[0], 8:8 + enlarged.shape[1]] = enlarged
    return panel


def export_diagnostics(bot, snap, cfg):
    capture_stamp = time.time() - (max(0, time.monotonic() - bot.capture_time)
                                  if bot.capture_time is not None else 0)
    if publish_png('controller-prepared.png', snap.frame):
        metadata = {'updated': capture_stamp, 'size': list(snap.frame.shape[1::-1]),
                    'capture': capture_signature(cfg), 'sequence': bot.capture_sequence,
                    'captured_at': bot.capture_time, 'source': 'controller'}
        atomic_diagnostic(LOG / 'controller-prepared.json', json.dumps(metadata).encode('utf-8'))
    if snap.minimap_box is not None:
        crop = canonical_minimap(snap, bot)
        if crop.size and publish_png('editor-minimap.png', crop):
            metadata = {'updated': capture_stamp, 'size': [crop.shape[1], crop.shape[0]],
                        'capture': capture_signature(cfg),
                        'player_pos': snap.minimap_pos if bot._minimap_seen else None, 'source': 'controller',
                        'coordinate_space': 'profile' if bot.map_ready and cfg.get('map_profile', {}).get('active') else 'raw',
                        'profile_key': cfg.get('map_profile', {}).get('active'),
                        'reference_sha256': cfg.get('map_profile', {}).get('reference_sha256'),
                        'minimap_offset': getattr(bot, 'minimap_offset', (0, 0)),
                        'raw_player_pos': getattr(bot, 'minimap_raw_pos', None)}
            atomic_diagnostic(LOG / 'editor-minimap.json', json.dumps(metadata).encode('utf-8'))
    view = draw_debug(snap.frame, snap, cfg, minimap_offset=getattr(bot, 'minimap_offset', None) or (0, 0))
    score = '--' if bot.player is None else f'{bot.player.score:.3f}'
    info = (f'Lock missing: {bot._missing_frames}  player score: {score}  '
            f'face: {snap.command.face_x}  input blocked: {bot._input_blocked}')
    if getattr(bot, '_vertical_minimap_active', False):
        info = 'MINIMAP NAVIGATION ONLY - finishing vertical transition; combat disabled'
    if not bot.map_ready:
        info = 'PAUSED - ' + (bot.map_match.reason if bot.map_match is not None else 'Checking map identity')
    cv2.putText(view, info, (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, info, (400, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    if view.shape[1] > 1000:
        view = cv2.resize(view, (1000, round(view.shape[0] * 1000 / view.shape[1])),
                          interpolation=cv2.INTER_AREA)
    publish_png('bot-minimap.png', diagnostic_minimap(snap, bot))
    publish_png('bot-bars.png', diagnostic_bars(snap, cfg))
    return publish_png('bot-view.png', view)


class CountedBackend:
    """Record only keys emitted by this bot, never the user's keyboard input."""
    def __init__(self, backend, counts):
        self.backend = backend
        self.counts = counts

    def down(self, key):
        self.backend.down(key)
        self.counts[key] += 1

    def up(self, key):
        self.backend.up(key)


def publish_test(session=None, *, test_id=None, state='error', message=''):
    value = (session.status() if session is not None else
             {'test_id': test_id, 'state': state, 'message': message, 'updated': time.time()})
    atomic_diagnostic(TEST_STATUS, json.dumps(value).encode('utf-8'))


def consume_test(cfg):
    """Claim one atomic editor request; never follow a caller-provided image path."""
    claimed = TEST_REQUEST.with_name(f'.editor-test-request.{os.getpid()}.json')
    try:
        TEST_REQUEST.replace(claimed)
    except FileNotFoundError:
        return None
    request = {}
    try:
        request = json.loads(claimed.read_text(encoding='utf-8'))
        session = EditorTestSession(request, cfg, LOG)
        publish_test(session)
        return session
    except (OSError, ValueError, TypeError, KeyError) as error:
        test_id = request.get('test_id') if isinstance(request, dict) else None
        publish_test(test_id=test_id, message=str(error))
        return None
    finally:
        claimed.unlink(missing_ok=True)

def run():
    counts = Counter()
    waiting = '--wait-for-start' in sys.argv
    keep_paused = PAUSE.exists()
    total_frames = 0
    diagnostic_updated = None
    diagnostic_error = None
    last_trace = 0.0
    trace_index = 0
    session = None
    evidence_tracker = None
    final = {}  # last on-screen evidence and hp/mp/exp readings, kept for the stop publish
    print('Starting active Maplebot for player. F1 pause/resume, F12 stop.', flush=True)
    while not STOP.exists():
        normal_cfg = load_config(str(ROOT / 'config/user.yaml'))
        cfg = session.cfg if session is not None else normal_cfg
        try:
            bot = build_bot(cfg, None, False)
        except Exception as error:
            if session is None:
                raise
            session.finish('error', f'Could not prepare the movement test: {error}')
            publish_test(session)
            session, keep_paused, waiting = None, True, False
            continue
        bot.keyboard.backend = CountedBackend(bot.keyboard.backend, counts)
        # One session's observations survive a reload or test; counts made under different evidence
        # settings are kept in final['evidence_previous'] instead of being dropped.
        bot.evidence = evidence_tracker = carry_over_tracker(
            evidence_tracker, bot.evidence, final, lambda text: print(text, flush=True))
        bot.paused = session is not None or keep_paused or waiting
        if bot.paused:
            bot.pause_all()
        startup = StartupReturn(bot) if session is None else None
        if startup is not None and not bot.paused:
            startup.request()
        reload_requested = False
        pending_test = None
        pending_profile = None
        restore_normal = False
        external_stop = False
        original_step = bot.step
        original_stop = bot.stop_request
        last_status = 0.0
        last_diagnostic = 0.0
        last_test_status = 0.0
        last_test_state = None

        def request_runner_stop():
            nonlocal external_stop
            external_stop = True
            if startup is not None:
                startup.cancel("Start cancelled by Stop.")
            bot.pause_all()
            original_stop()

        bot.stop_request = request_runner_stop

        def stop_current_test(state=None, message=None, restore=True):
            nonlocal restore_normal
            if state is not None:
                session.finish(state, message)
            publish_test(session)
            restore_normal = restore
            bot.pause_all()
            original_stop()

        if session is not None:
            # F1 is a cancellation during a temporary test, never a preflight bypass.
            bot.toggle_pause = lambda: stop_current_test('cancelled', 'Connection test cancelled with F1.')
        else:
            def toggle_normal_pause():
                nonlocal waiting, keep_paused
                if startup.toggle_pause():
                    waiting, keep_paused = False, False
                else:
                    keep_paused = True
            bot.toggle_pause = toggle_normal_pause

        def observed_step(now=None):
            nonlocal last_status, waiting, reload_requested, pending_test, keep_paused
            nonlocal last_diagnostic, diagnostic_updated, diagnostic_error, last_trace, trace_index
            nonlocal last_test_status, last_test_state
            nonlocal pending_profile
            requested_return = False
            requested_hold = False
            test_generation = bot._control_generation
            if STOP.exists() or RELOAD.exists():
                if startup is not None:
                    startup.cancel("Start cancelled by stop or reload.")
                HOLD.unlink(missing_ok=True)
                RETURN_START.unlink(missing_ok=True)
                reload_requested = RELOAD.exists() and not STOP.exists()
                if session is not None:
                    stop_current_test('cancelled', 'Connection test cancelled by stop or reload.',
                                      restore=reload_requested)
                    return None
                bot.keyboard.set_enabled(False)
                original_stop()
                return None
            if session is not None:
                HOLD.unlink(missing_ok=True)
                RETURN_START.unlink(missing_ok=True)
                if TEST_STOP.exists() or PAUSE.exists() or TEST_REQUEST.exists():
                    TEST_STOP.unlink(missing_ok=True)
                    PAUSE.unlink(missing_ok=True)
                    RESUME.unlink(missing_ok=True)
                    stop_current_test('cancelled', 'Connection test cancelled. The normal bot is paused.')
                    return None
                RESUME.unlink(missing_ok=True)  # preflight and focus decide when a test may move
                foreground = bot.keyboard.is_active()
                allow = session.before_step(time.monotonic(), foreground)
                cancelled = (TEST_STOP.exists() or PAUSE.exists() or STOP.exists()
                             or RELOAD.exists() or TEST_REQUEST.exists())
                if cancelled:
                    stop_current_test('cancelled', 'Connection test cancelled before movement.')
                    return None
                if session.terminal:
                    stop_current_test()
                    return None
                bot.apply_movement_test_permission(allow, expected_generation=test_generation)
            else:
                handled_test = False
                if TEST_STOP.exists():
                    startup.cancel("Start cancelled by a connection test.")
                    bot.pause_all()
                    TEST_STOP.unlink(missing_ok=True)
                    if TEST_REQUEST.exists():
                        cancelled = consume_test(normal_cfg)
                        if cancelled is not None:
                            cancelled.finish('cancelled', 'Queued connection test cancelled.')
                            publish_test(cancelled)
                    handled_test = True
                elif TEST_REQUEST.exists():
                    startup.cancel("Start cancelled by a connection test.")
                    bot.pause_all()
                    pending_test = consume_test(normal_cfg)
                    handled_test = True
                if handled_test:
                    HOLD.unlink(missing_ok=True)
                    RETURN_START.unlink(missing_ok=True)
                    PAUSE.unlink(missing_ok=True)  # the editor pauses before posting its request
                    RESUME.unlink(missing_ok=True)
                    bot.paused, keep_paused, waiting = True, True, False
                    bot.pause_all()
                    if pending_test is not None:
                        original_stop()
                        return None
                else:
                    requested_pause = PAUSE.exists()
                    if requested_pause:
                        startup.cancel("Start cancelled by Pause all.")
                        HOLD.unlink(missing_ok=True)
                        RETURN_START.unlink(missing_ok=True)
                        PAUSE.unlink(missing_ok=True)
                        RESUME.unlink(missing_ok=True)
                        bot.pause_all()
                    elif RESUME.exists():
                        PAUSE.unlink(missing_ok=True)
                        RESUME.unlink(missing_ok=True)
                        requested_pause = False
                        startup.request()
                        waiting = False
                    if waiting:
                        if ARM.exists():
                            waiting = False
                            ARM.unlink(missing_ok=True)
                            bot.pause_all() if requested_pause else startup.request()
                        else:
                            bot.pause_all()
                    if HOLD.exists() and not requested_pause:
                        startup.cancel("Start cancelled by Hold farming.")
                        requested_hold = True
                        RETURN_START.unlink(missing_ok=True)
                        bot.pause_all()
                    elif RETURN_START.exists() and not requested_pause:
                        startup.cancel("Start replaced by an explicit return request.")
                        requested_return = True
                        bot.pause_all()
            if bot.paused and bot.full_pause:
                bot.keyboard.set_enabled(False)
            snap = original_step(now)
            stamp = time.time()
            foreground = bot.keyboard.is_active()
            if requested_hold and bot.capture_state != 'duplicate':
                cancelled = PAUSE.exists() or STOP.exists() or RELOAD.exists() or TEST_REQUEST.exists()
                try:
                    request = json.loads(HOLD.read_text(encoding='utf-8'))
                except FileNotFoundError:
                    cancelled, request = True, None
                except (ValueError, OSError):
                    request = None
                HOLD.unlink(missing_ok=True)
                if not cancelled and request_hold(bot, request, now=stamp):
                    waiting, keep_paused = False, True
            if requested_return and bot.capture_state != 'duplicate':
                # Take a new paused capture before validating a queued UI action.
                # A later Pause/Stop wins over the return, even while capturing.
                cancelled = PAUSE.exists() or STOP.exists() or RELOAD.exists() or TEST_REQUEST.exists()
                try:
                    request = json.loads(RETURN_START.read_text(encoding='utf-8'))
                except FileNotFoundError:
                    cancelled = True
                    request = None
                except (ValueError, OSError):
                    request = None
                RETURN_START.unlink(missing_ok=True)
                if not cancelled and request_return(bot, request, now=stamp):
                    waiting, keep_paused = False, False
            if startup is not None and startup.active:
                cancelled = PAUSE.exists() or STOP.exists() or RELOAD.exists() or TEST_REQUEST.exists() or HOLD.exists()
                if cancelled:
                    startup.cancel("Start cancelled by a newer controller request.")
                    bot.pause_all()
                else:
                    startup.update()
            if bot.pending_map_profile is not None:
                pending_profile = bot.pending_map_profile
                bot.paused = True
                bot.keyboard.set_enabled(False)
                if session is not None:
                    session.finish('cancelled', 'Map changed during the test. The controller is paused.')
                    publish_test(session)
                original_stop()
            if session is not None and pending_profile is None and bot.capture_state != 'duplicate':
                if not bot.map_ready:
                    bot.keyboard.set_enabled(False)
                    if session.has_run:
                        stop_current_test('blocked', 'Map identity was lost during the test.')
                        return snap
                    # Initial recognition needs several fresh frames before test preflight.
                    return snap
                allow = session.after_step(snap, cfg, now=time.monotonic(), foreground=foreground,
                                           minimap_seen=bot._minimap_seen, circuit=bot.brain.circuit,
                                           minimap_offset=bot.minimap_offset)
                cancelled = (TEST_STOP.exists() or PAUSE.exists() or STOP.exists()
                             or RELOAD.exists() or TEST_REQUEST.exists())
                if cancelled:
                    stop_current_test('cancelled', 'Connection test cancelled during verification.')
                elif session.terminal:
                    stop_current_test()
                else:
                    bot.apply_movement_test_permission(allow, expected_generation=test_generation)
                if stamp - last_test_status >= .3 or session.state != last_test_state:
                    publish_test(session)
                    last_test_status, last_test_state = stamp, session.state
            if snap is not None:
                monotonic = time.monotonic()
                if monotonic - last_diagnostic >= 0.3:
                    last_diagnostic = monotonic
                    try:
                        if export_diagnostics(bot, snap, cfg):
                            diagnostic_updated = stamp
                        diagnostic_error = None
                    except Exception as error:
                        diagnostic_error = str(error)
                if (snap.command.action in ('attack', 'aoe', 'normal_attack')
                        and not snap.paused and not bot.paused and not bot._input_blocked
                        and foreground and monotonic - last_trace >= 1.0):
                    last_trace = monotonic
                    name = f'combat-frame-{trace_index % 30:02d}'
                    trace_index += 1
                    trace = snapshot_details(bot, snap)
                    trace.update(pid=os.getpid(), updated=stamp, frame=total_frames + bot.frame_idx,
                                 sequence=trace_index, player=snap.player, minimap=snap.minimap_pos,
                                 global_pos=snap.global_pos, command=str(snap.command),
                                 hp=snap.hp, mp=snap.mp, exp=snap.exp, keys_sent=dict(counts),
                                 frame_size=[snap.frame.shape[1], snap.frame.shape[0]])
                    try:
                        if publish_png(name + '.png', snap.frame):
                            atomic_diagnostic(LOG / (name + '.json'), json.dumps(trace).encode('utf-8'))
                    except Exception as error:
                        diagnostic_error = str(error)
            if stamp - last_status >= 0.15:
                status = {'running': True, 'frames': total_frames + bot.frame_idx,
                          'capture': capture_signature(cfg),
                          'input_blocked': bot._input_blocked,
                          'foreground': foreground, 'paused': bot.paused,
                          'waiting_to_start': waiting, 'keys_sent': dict(counts),
                          'diagnostic_updated': diagnostic_updated, 'diagnostic_error': diagnostic_error}
                if session is not None:
                    status['editor_test'] = session.status()
                elif startup is not None:
                    status.update(startup_return_state=startup.state, startup_return_reason=startup.reason)
                status.update(snapshot_details(bot, snap))
                # Pixel observations (lower bounds), not kill or item counts; None when disabled.
                status['evidence'] = record_snapshot(final, bot.evidence_snapshot())
                if snap is not None:
                    final.update({key: value for key, value in
                                  (('hp', snap.hp), ('mp', snap.mp), ('exp', snap.exp)) if value is not None})
                    status.update(player=snap.player, minimap=snap.minimap_pos,
                                  monsters=[d.name for d in snap.detections],
                                  command=str(snap.command), hp=snap.hp, mp=snap.mp,
                                  exp=snap.exp, fps=snap.fps)
                publish(status)
                last_status = stamp
            return snap

        bot.step = observed_step
        try:
            bot.run(show=False)
        except Exception as error:
            if session is None:
                raise
            session.finish('error', f'Connection test stopped: {error}')
            publish_test(session)
            restore_normal = True
        total_frames += bot.frame_idx
        record_snapshot(final, bot.evidence_snapshot())
        keep_paused = bot.paused
        if external_stop:
            cancelled = session if session is not None else pending_test
            if cancelled is not None:
                cancelled.finish('cancelled', 'Connection test stopped with F12.')
                publish_test(cancelled)
            break
        if pending_profile is not None and not STOP.exists():
            try:
                EditorBridge(ROOT).select_profile(pending_profile.name)
                print(f'Loaded recognized map {pending_profile.name}; controller remains paused.', flush=True)
            except Exception as error:
                publish({'running': False, 'error': f'Could not load recognized map: {error}'})
                raise
            session, keep_paused, waiting = None, True, False
            RELOAD.unlink(missing_ok=True)
            continue
        if session is not None:
            if restore_normal:
                session, keep_paused, waiting = None, True, False
                RESUME.unlink(missing_ok=True)
                PAUSE.unlink(missing_ok=True)
                if reload_requested:
                    RELOAD.unlink(missing_ok=True)
                continue
            if not session.terminal:
                session.finish('cancelled', 'Connection test stopped with the controller.')
                publish_test(session)
            break  # F12 or Stop ends the runner, including during a test
        if pending_test is not None:
            session, keep_paused, waiting = pending_test, True, False
            continue
        if not reload_requested:
            break
        RELOAD.unlink(missing_ok=True)
        print('Applying updated settings.', flush=True)
    publish({'running': False, 'frames': total_frames, 'keys_sent': dict(counts), **final})
    lines = session_summary_lines(final)
    print('Session ' + lines[0] + ' | last readings '
          + ' '.join(f'{key}={final.get(key)}' for key in ('hp', 'mp', 'exp')), flush=True)
    for line in lines[1:]:
        print('Session ' + line, flush=True)
    print('Active Maplebot stopped; keys released.', flush=True)

if __name__ == '__main__':
    import hashlib
    import win32api
    import win32event
    mutex_name = 'Local\\MaplebotActive_' + hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:16]
    instance_mutex = win32event.CreateMutex(None, False, mutex_name)
    if win32api.GetLastError() == 183:
        win32api.CloseHandle(instance_mutex)
        print('Maplebot is already running; duplicate launch ignored.', flush=True)
        raise SystemExit(0)
    try:
        run()
    except Exception as error:
        traceback.print_exc()
        publish({'running': False, 'error': str(error)})
        raise
    finally:
        win32api.CloseHandle(instance_mutex)
