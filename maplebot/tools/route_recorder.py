"""Record a map and routes: stitch the live minimap into map.png and paint colour-coded routes
from the keys you press. Pure parts (MapStitcher, RouteCanvas, command_from_keys) are tested;
run_recorder is the Windows-only interactive loop."""
from __future__ import annotations

import os
import time

import cv2
import numpy as np

from maplebot.config import parse_color_codes
from maplebot.log import get_logger
from maplebot.types import Command
from maplebot.vision.template import match_template

log = get_logger()


class MapStitcher:
    """Grows a canvas by template-matching each new minimap crop into it."""

    def __init__(self, padding: int = 30, dot_radius: int = 5):
        self.padding = int(padding)
        self.dot_radius = int(dot_radius)
        self.map: np.ndarray | None = None
        self.last_loc: tuple[int, int] | None = None
        self.last_shift: tuple[int, int] = (0, 0)  # how far existing content moved on the last add()

    def _ensure_capacity(self, x: int, y: int, h: int, w: int) -> tuple[int, int]:
        mh, mw = self.map.shape[:2]
        p = self.padding
        top = max(0, p - y)
        left = max(0, p - x)
        bottom = max(0, y + h + p - mh)
        right = max(0, x + w + p - mw)
        self.last_shift = (left, top)
        if top or left or bottom or right:
            canvas = np.zeros((mh + top + bottom, mw + left + right, 3), np.uint8)
            canvas[top:top + mh, left:left + mw] = self.map
            self.map = canvas
            if self.last_loc is not None:
                self.last_loc = (self.last_loc[0] + left, self.last_loc[1] + top)
        return x + left, y + top

    def add(self, minimap_bgr: np.ndarray, player_dot: tuple[int, int] | None,
            loc_hint: tuple[int, int] | None = None) -> tuple[int, int]:
        """Place the crop and fill only unknown (black) canvas pixels. Returns the crop's top-left."""
        mm = minimap_bgr.copy()
        if player_dot is not None:
            r = self.dot_radius
            px, py = player_dot
            mm[max(0, py - r):py + r + 1, max(0, px - r):px + r + 1] = 0
        h, w = mm.shape[:2]
        if self.map is None:
            p = self.padding
            self.map = np.zeros((h + 2 * p, w + 2 * p, 3), np.uint8)
            self.map[p:p + h, p:p + w] = mm
            self.last_loc = (p, p)
            self.last_shift = (0, 0)
            return (p, p)
        mask = (np.any(mm != 0, axis=2).astype(np.uint8) * 255)
        res = match_template(self.map, mm, mask=mask, last_loc=loc_hint or self.last_loc,
                             local_radius=40, local_thres=0.3)
        x, y = self._ensure_capacity(res.loc[0], res.loc[1], h, w)
        target = self.map[y:y + h, x:x + w]
        fill = np.all(target == 0, axis=2) & (mask > 0)
        target[fill] = mm[fill]
        self.last_loc = (x, y)
        return (x, y)


class RouteCanvas:
    """Paints route colours on a copy of the map."""

    def __init__(self, map_bgr: np.ndarray, color_code_rgb: dict[tuple, str]):
        self.by_command: dict[str, tuple[int, int, int]] = {}
        for rgb, cmd in color_code_rgb.items():
            self.by_command.setdefault(str(Command.parse(cmd)), (rgb[2], rgb[1], rgb[0]))
        self.img: np.ndarray | None = None
        self._last: tuple[int, int] | None = None
        self.reset(map_bgr)

    def reset(self, map_bgr: np.ndarray) -> None:
        img = map_bgr.copy()
        for bgr in self.by_command.values():
            img[np.all(img == np.array(bgr, np.uint8), axis=2)] = 0
        self.img = img
        self._last = None

    def grow(self, new_map: np.ndarray, shift: tuple[int, int]) -> None:
        """The stitched map grew (and its content moved by `shift`): move the painting with it."""
        old = self.img
        img = np.zeros_like(new_map)
        dx, dy = int(shift[0]), int(shift[1])
        h = min(old.shape[0], img.shape[0] - dy)
        w = min(old.shape[1], img.shape[1] - dx)
        if h > 0 and w > 0:
            img[dy:dy + h, dx:dx + w] = old[:h, :w]
        self.img = img
        if self._last is not None:
            self._last = (self._last[0] + dx, self._last[1] + dy)

    def mark(self, pos: tuple[int, int], command: Command, blob: bool) -> None:
        bgr = self.by_command.get(str(command))
        if bgr is None:
            return
        pos = (int(pos[0]), int(pos[1]))
        if blob:
            cv2.circle(self.img, pos, 2, bgr, -1)
            self._last = None
        else:
            cv2.line(self.img, self._last or pos, pos, bgr, 1)
            self._last = pos

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cv2.imwrite(path, self.img)


def command_from_keys(keys: set[str], jump_key: str, teleport_key: str) -> Command | None:
    """Translate held keys into the route command to paint (same priorities as the reference recorder)."""
    k = {s.lower() for s in keys}
    if jump_key and jump_key.lower() in k:
        if "left" in k:
            return Command("left", "none", "jump")
        if "right" in k:
            return Command("right", "none", "jump")
        if "down" in k:
            return Command("none", "down", "jump")
        return Command("none", "none", "jump")
    if teleport_key and teleport_key.lower() in k:
        for d, cmd in (("left", Command("left", "none", "teleport")), ("right", Command("right", "none", "teleport")),
                       ("down", Command("none", "down", "teleport")), ("up", Command("none", "up", "teleport"))):
            if d in k:
                return cmd
        return None
    for d, cmd in (("up", Command("none", "up")), ("down", Command("none", "down")),
                   ("left", Command("left")), ("right", Command("right"))):
        if d in k:
            return cmd
    return None


class _HeldKeys:
    """Tracks which keys are currently held, via a pynput listener (Windows runtime)."""

    def __init__(self):
        from pynput import keyboard

        self.held: set[str] = set()
        self.func: dict[str, float] = {}
        self._l = keyboard.Listener(on_press=self._press, on_release=self._release)
        self._l.daemon = True
        self._l.start()

    @staticmethod
    def _name(key) -> str | None:
        name = getattr(key, "name", None)
        if name:
            return name.lower()
        ch = getattr(key, "char", None)
        return ch.lower() if ch else None

    def _press(self, key):
        n = self._name(key)
        if n:
            self.held.add(n)
            if n.startswith("f") and n[1:].isdigit():
                self.func[n] = time.monotonic()

    def _release(self, key):
        n = self._name(key)
        if n:
            self.held.discard(n)

    def pop_func(self, name: str, debounce: float = 0.8) -> bool:
        t = self.func.get(name)
        if t is None:
            return False
        del self.func[name]
        return True


def run_recorder(cfg: dict, source, out_dir: str, existing_map: str | None = None) -> None:
    """Interactive loop. F1 start/pause painting, F3 save route (goal), F4 save map, q quit."""
    from maplebot.capture.source import prepare_frame
    from maplebot.vision.minimap import find_minimap, find_player_dot

    os.makedirs(out_dir, exist_ok=True)
    codes = parse_color_codes(cfg["route"]["color_code"])
    codes.update(parse_color_codes(cfg["route"]["color_code_up_down"]))
    stitcher = MapStitcher(padding=30)
    if existing_map:
        stitcher.map = cv2.imread(existing_map, cv2.IMREAD_COLOR)
    canvas: RouteCanvas | None = None
    keys = _HeldKeys()
    mcfg = cfg["minimap"]
    enabled = False
    route_idx = 1
    last_blob_t = 0.0
    last_dot = None
    minimap_box = None
    source.start()
    period = 1.0 / max(1.0, float(cfg["system"].get("fps_main", 12)))
    log.info("route recorder: walk the map; F1 toggles painting, F3 saves a route, F4 saves map.png, q quits")
    try:
        while True:
            t0 = time.monotonic()
            raw = source.get_frame()
            if raw is None:
                time.sleep(0.2)
                continue
            frame = prepare_frame(raw, cfg["game_window"])
            fixed = mcfg.get("fixed_rect")
            box = find_minimap(frame) if not fixed else None
            if fixed:
                from maplebot.types import Box
                box = Box(*map(int, fixed))
            if box is not None:
                minimap_box = box
            if minimap_box is None:
                cv2.imshow("recorder", frame)
                cv2.waitKey(1)
                continue
            mm = frame[minimap_box.y:minimap_box.y2, minimap_box.x:minimap_box.x2]
            dot = find_player_dot(mm, tuple(mcfg["player_color_bgr"]), int(mcfg["player_tolerance"]),
                                  last=last_dot, max_jump_px=int(mcfg["max_jump_px"]))
            if dot is not None:
                last_dot = dot
            loc = stitcher.add(mm, last_dot)
            if canvas is None:
                canvas = RouteCanvas(stitcher.map, codes)
            elif canvas.img.shape[:2] != stitcher.map.shape[:2]:
                canvas.grow(stitcher.map, stitcher.last_shift)
            gpos = None if last_dot is None else (loc[0] + last_dot[0], loc[1] + last_dot[1])

            if keys.pop_func("f1"):
                enabled = not enabled
                log.info("painting %s", "ON" if enabled else "OFF")
            if keys.pop_func("f4"):
                p = os.path.join(out_dir, "map.png")
                cv2.imwrite(p, stitcher.map)
                log.info("saved %s", p)
            if keys.pop_func("f3") and gpos is not None:
                canvas.mark(gpos, Command("none", "none", "goal"), blob=True)
                p = os.path.join(out_dir, f"route{route_idx}.png")
                canvas.save(p)
                log.info("saved %s", p)
                route_idx += 1
                canvas.reset(stitcher.map)
            if enabled and gpos is not None:
                cmd = command_from_keys(keys.held, str(cfg["keys"].get("jump", "space")), str(cfg["keys"].get("teleport") or ""))
                if cmd is not None:
                    blob = cmd.action in ("jump", "teleport")
                    if blob:
                        if time.monotonic() - last_blob_t > 0.7:
                            canvas.mark(gpos, cmd, blob=True)
                            last_blob_t = time.monotonic()
                    else:
                        canvas.mark(gpos, cmd, blob=False)

            view = canvas.img.copy()
            if gpos is not None:
                cv2.circle(view, gpos, 2, (0, 255, 255), -1)
            view = cv2.resize(view, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            cv2.putText(view, f"paint {'ON' if enabled else 'OFF'}  route{route_idx}  F1 paint F3 route F4 map q quit",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("recorder", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    finally:
        source.stop()
        cv2.destroyAllWindows()
