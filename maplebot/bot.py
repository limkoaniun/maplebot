"""The bot: one frame in, one Command out, every loop tick."""
from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from maplebot.brain.combat import search_box
from maplebot.brain.fsm import Brain, Perception
from maplebot.capture.source import ImageSource, VideoSource, prepare_frame
from maplebot.config import parse_color_codes
from maplebot.control.backend import make_backend
from maplebot.control.keyboard import KeyboardController
from maplebot.health import HealthPolicy
from maplebot.log import get_logger
from maplebot.nav.route import RouteFollower, RouteMap, locate_minimap_on_map
from maplebot.types import Box, Command, Detection, PlayerFix
from maplebot.vision.bars import read_bars
from maplebot.vision.minimap import find_minimap, find_other_players, find_player_dot
from maplebot.vision.monsters import MonsterLibrary, detect_hp_bars, detect_monsters
from maplebot.vision.player import locate_by_party_bar, locate_by_template
from maplebot.vision.template import make_mask

log = get_logger()
DOT_LOST_FRAMES = 15


@dataclass
class BotSnapshot:
    frame: np.ndarray
    player: tuple[int, int] | None
    minimap_box: Box | None
    minimap_pos: tuple[int, int] | None
    global_pos: tuple[int, int] | None
    detections: list[Detection]
    command: Command
    state: str
    hp: float | None
    mp: float | None
    exp: float | None
    fps: float
    route_idx: int | None
    search_box: Box | None = None
    target: str | None = None
    other_players: list[tuple[int, int]] = field(default_factory=list)
    paused: bool = False
    route_map: RouteMap | None = None

    def summary(self) -> str:
        return (f"state={self.state} cmd=\"{self.command}\" player={self.player} minimap={self.minimap_pos} "
                f"global={self.global_pos} mobs={len(self.detections)} target={self.target} "
                f"hp={self.hp} mp={self.mp} fps={self.fps:.1f}")


class Bot:
    def __init__(self, cfg: dict, source, keyboard: KeyboardController | None,
                 library: MonsterLibrary, route_map: RouteMap | None):
        self.cfg = cfg
        self.source = source
        self.keyboard = keyboard
        self.library = library
        self.route_map = route_map
        self.follower = RouteFollower(route_map, cfg["route"]) if route_map is not None else None
        self.brain = Brain(cfg, self.follower, library.min_area)
        self.health = HealthPolicy(cfg["health"])
        self.paused = False
        self.frame_idx = 0
        self.fps = 0.0
        self.on_ladder = False
        self.player: PlayerFix | None = None
        self.minimap_box: Box | None = None
        self.minimap_pos: tuple[int, int] | None = None
        self.global_pos: tuple[int, int] | None = None
        self.last_snapshot: BotSnapshot | None = None
        self._minimap_loc: tuple[int, int] | None = None
        self._dot_lost = 0
        self._t_last: float | None = None
        self._tmpl = None
        self._tmpl_mask = None
        self._load_player_template()

    # -- setup -----------------------------------------------------------
    def _load_player_template(self) -> None:
        pcfg = self.cfg["player"]
        loc = pcfg.get("locator", "party_bar")
        if loc not in ("nametag", "appearance"):
            return
        path = str(pcfg[loc].get("path") or "")
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path else None
        if raw is None:
            log.warning("player.locator=%s but no template at %r; falling back to party_bar", loc, path)
            pcfg["locator"] = "party_bar"
            return
        bgr, mask = make_mask(raw)
        if loc == "nametag":
            self._tmpl = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            self._tmpl = bgr
            if not np.any(mask == 0):  # raw screenshot crop: ignore the corners
                mask = np.zeros(bgr.shape[:2], np.uint8)
                hh, ww = mask.shape
                cv2.ellipse(mask, (ww // 2, hh // 2), (int(ww * 0.46), int(hh * 0.46)), 0, 0, 360, 255, -1)
        self._tmpl_mask = mask

    # -- per-frame pieces --------------------------------------------------
    def _locate_player(self, frame: np.ndarray) -> PlayerFix | None:
        pcfg = self.cfg["player"]
        loc = pcfg.get("locator", "party_bar")
        last = self.player.pos if self.player else None
        if loc == "nametag":
            c = pcfg["nametag"]
            return locate_by_template(frame, self._tmpl, self._tmpl_mask, last, tuple(c["offset"]),
                                      float(c["diff_thres"]), split_width=int(c.get("split_width", 0)),
                                      source="nametag")
        if loc == "appearance":
            c = pcfg["appearance"]
            return locate_by_template(frame, self._tmpl, self._tmpl_mask, last, tuple(c["feet_offset"]),
                                      1.0 - float(c["match_thres"]), flip=True, method=cv2.TM_CCORR_NORMED,
                                      local_radius=int(c.get("search_radius", 140)), source="appearance")
        return locate_by_party_bar(frame, pcfg["party_bar"], exclude=self.minimap_box)

    def _update_ladder(self, fix: PlayerFix) -> None:
        if self.player is None:
            return
        dx = abs(fix.x - self.player.x)
        dy = abs(fix.y - self.player.y)
        if self.on_ladder:
            if dx > 3:
                self.on_ladder = False
        elif dx < 3 and dy != 0:
            self.on_ladder = True

    def _update_minimap(self, frame: np.ndarray) -> np.ndarray | None:
        mcfg = self.cfg["minimap"]
        fixed = mcfg.get("fixed_rect")
        box = Box(*map(int, fixed)) if fixed else find_minimap(frame)
        if box is not None:
            self.minimap_box = box
        if self.minimap_box is None:
            return None
        b = self.minimap_box.clamp(frame.shape[1], frame.shape[0])
        if b.area == 0:
            return None
        img = frame[b.y:b.y2, b.x:b.x2]
        dot = find_player_dot(img, tuple(mcfg["player_color_bgr"]), int(mcfg["player_tolerance"]),
                              last=self.minimap_pos, max_jump_px=int(mcfg["max_jump_px"]))
        if dot is not None:
            self.minimap_pos = dot
            self._dot_lost = 0
        else:
            self._dot_lost += 1
            if self._dot_lost >= DOT_LOST_FRAMES:
                self.minimap_pos = None  # allow a fresh lock after a teleport / map change
        return img

    def _update_global(self, img_mm: np.ndarray | None) -> None:
        if self.route_map is None or img_mm is None or self.minimap_pos is None:
            self.global_pos = None
            return
        res = locate_minimap_on_map(self.route_map.map_bgr, img_mm, last=self._minimap_loc)
        self._minimap_loc = res.loc
        self.global_pos = (res.loc[0] + self.minimap_pos[0], res.loc[1] + self.minimap_pos[1])

    def _detect(self, frame: np.ndarray) -> tuple[list[Detection], Box | None]:
        if self.player is None or not self.library.templates and not self.cfg["monsters"]["hp_bar"]["enable"]:
            return [], None
        h, w = frame.shape[:2]
        sb = search_box(self.player.pos, self.cfg["attack"], w, h)
        if sb.area == 0:
            return [], sb
        roi = frame[sb.y:sb.y2, sb.x:sb.x2]
        pcfg = self.cfg["player"]
        bw, bh = int(pcfg.get("box_w", 100)), int(pcfg.get("box_h", 150))
        exclude = Box(self.player.x - bw // 2, self.player.y - bh // 2, bw, bh)
        mcfg = self.cfg["monsters"]
        dets = detect_monsters(roi, self.library, mcfg, exclude, (sb.x, sb.y))
        hb = mcfg.get("hp_bar", {})
        if hb.get("enable"):
            dets += detect_hp_bars(roi, tuple(hb["color_bgr"]), int(hb.get("tolerance", 10)), roi_origin=(sb.x, sb.y))
        return dets, sb

    def _read_health(self, frame: np.ndarray, now: float) -> tuple[float | None, float | None, float | None]:
        hcfg = self.cfg["health"]
        if not hcfg.get("enable", True):
            return (None, None, None)
        if hcfg.get("fixed_rects"):
            hp, mp, exp = read_bars(frame, hcfg)
        else:
            y0 = int(hcfg.get("ui_y_start", 0))
            ui = frame[y0:] if 0 <= y0 < frame.shape[0] else frame
            hp, mp, exp = read_bars(ui, hcfg)
        for req in self.health.update(hp, mp, now):
            if self.keyboard is not None and not self.paused:
                self.keyboard.request(req)
                log.info("%s (hp=%s mp=%s)", req, hp, mp)
        return hp, mp, exp

    # -- main ----------------------------------------------------------------
    def step(self, now: float | None = None) -> BotSnapshot | None:
        now = time.monotonic() if now is None else now
        raw = self.source.get_frame()
        if raw is None:
            return None
        frame = prepare_frame(raw, self.cfg["game_window"])
        h, w = frame.shape[:2]
        self.frame_idx += 1

        img_mm = self._update_minimap(frame)
        fix = self._locate_player(frame)
        if fix is not None:
            self._update_ladder(fix)
            self.player = fix
        self._update_global(img_mm)
        other = find_other_players(img_mm, tuple(self.cfg["minimap"]["other_color_bgr"])) if img_mm is not None else []
        dets, sb = self._detect(frame)
        hp, mp, exp = self._read_health(frame, now)

        if self.paused:
            cmd = Command.NONE
        else:
            p = Perception(w, h, self.player.pos if self.player else None, self.global_pos, self.minimap_pos,
                           img_mm.shape[1] if img_mm is not None else None, self.on_ladder, dets, now, self.frame_idx)
            cmd = self.brain.decide(p)
        if self.keyboard is not None:
            self.keyboard.set_command(cmd)

        if self._t_last is not None and now > self._t_last:
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / (now - self._t_last)) if self.fps else 1.0 / (now - self._t_last)
        self._t_last = now

        snap = BotSnapshot(frame, self.player.pos if self.player else None, self.minimap_box, self.minimap_pos,
                           self.global_pos, dets, cmd, self.brain.state, hp, mp, exp, self.fps,
                           self.follower.idx if self.follower else None, sb, self.brain.target, other,
                           self.paused, self.route_map)
        self.last_snapshot = snap
        return snap

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if self.keyboard is not None:
            self.keyboard.set_command(Command.NONE)
            self.keyboard.set_enabled(not self.paused)
        log.info("%s", "paused" if self.paused else "resumed")

    def screenshot(self, out_dir: str = "screenshot") -> str | None:
        snap = self.last_snapshot
        if snap is None:
            return None
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(out_dir, f"{stamp}.png")
        cv2.imwrite(path, snap.frame)
        log.info("screenshot saved to %s", path)
        return path

    def close(self) -> None:
        if self.keyboard is not None:
            self.keyboard.stop()
        self.source.stop()

    def run(self, show: bool = True, max_frames: int | None = None) -> BotSnapshot | None:
        from maplebot.overlay import draw_debug

        self.source.start()
        if self.keyboard is not None:
            self.keyboard.start()
        hot = None
        try:
            from maplebot.control.hotkeys import Hotkeys
            hot = Hotkeys({"f1": self.toggle_pause, "f2": self.screenshot, "f12": self.stop_request},
                          float(self.cfg["system"].get("debounce_s", 1.0)))
            hot.start()
        except Exception as e:
            log.warning("hotkeys unavailable: %s", e)
        self._stop = False
        period = 1.0 / max(1.0, float(self.cfg["system"]["fps_main"]))
        snap = None
        try:
            while not self._stop:
                t0 = time.monotonic()
                snap = self.step(t0)
                if snap is None:
                    log.warning("no frame from %s", getattr(self.source, "title", "source"))
                    time.sleep(0.5)
                    continue
                if show:
                    img = draw_debug(snap.frame, snap, self.cfg)
                    scale = float(self.cfg.get("debug", {}).get("scale", 1.0))
                    if scale != 1.0:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("maplebot", img)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
                if max_frames is not None and self.frame_idx >= max_frames:
                    break
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)
        finally:
            if hot is not None:
                hot.stop()
            self.close()
            if show:
                cv2.destroyAllWindows()
        return snap

    def stop_request(self) -> None:
        self._stop = True


def build_bot(cfg: dict, source_path: str | None, no_input: bool) -> Bot:
    """Wire a Bot from config: source (window / image / video), keyboard, monsters, route."""
    if source_path:
        ext = os.path.splitext(source_path)[1].lower()
        source = VideoSource(source_path) if ext in (".mp4", ".avi", ".mkv", ".mov") else ImageSource(source_path)
    else:
        from maplebot.capture.window import WindowCapture
        source = WindowCapture(cfg)
    backend = make_backend("null" if no_input else "pynput")
    keyboard = KeyboardController(cfg, backend)

    mcfg = cfg["monsters"]
    per = mcfg.get("per_monster") or {}
    scales_by_name = {n: v["scales"] for n, v in per.items() if isinstance(v, dict) and v.get("scales")}
    root = str(mcfg.get("dir", "monster"))
    try:
        library = MonsterLibrary.load(root, list(mcfg.get("names") or []), scales_by_name, list(mcfg.get("scales") or [1.0]))
    except (FileNotFoundError, NotADirectoryError) as e:
        log.warning("no monster templates loaded (%s)", e)
        library = MonsterLibrary()
    if not library.templates:
        log.warning("monster library is empty: the bot will not attack anything")
    else:
        log.info("monsters: %s (%d templates)", library.names, len(library.templates))

    route_map = None
    rcfg = cfg["route"]
    if rcfg.get("map"):
        folder = os.path.join(str(rcfg.get("dir", "minimaps")), str(rcfg["map"]))
        route_map = RouteMap.load(folder, parse_color_codes(rcfg["color_code"]), parse_color_codes(rcfg["color_code_up_down"]))
        log.info("route %s: %d route(s)", folder, route_map.n_routes)
    else:
        log.info("no route configured: sweep search mode")
    return Bot(cfg, source, keyboard, library, route_map)
