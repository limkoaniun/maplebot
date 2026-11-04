"""Route maps: colour-coded PNG routes over a stitched minimap, and following them."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from maplebot.types import Command
from maplebot.vision.template import MatchResult, match_template


@dataclass
class RouteMap:
    map_bgr: np.ndarray
    routes: list[np.ndarray] = field(default_factory=list)   # int16 code index per pixel, -1 = none
    codes: list[str] = field(default_factory=list)            # index -> "move_x move_y action"
    is_up_down: list[bool] = field(default_factory=list)      # index -> belongs to the up/down set
    commands: list[Command] = field(default_factory=list)     # index -> parsed Command

    @classmethod
    def load(cls, dir_path: str, color_code: dict[tuple, str], color_code_up_down: dict[tuple, str]) -> "RouteMap":
        """Read map.png and route*.png (route_rest.png excluded). Colour keys are RGB; files are BGR.

        A route pixel whose colour also appears at the same position in map.png is treated as map
        content, not as a route instruction.
        """
        map_path = os.path.join(dir_path, "map.png")
        map_bgr = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if map_bgr is None:
            raise FileNotFoundError(map_path)
        rm = cls(map_bgr=map_bgr)
        entries: list[tuple[tuple[int, int, int], str, bool]] = []
        for rgb, cmd in color_code.items():
            entries.append(((rgb[2], rgb[1], rgb[0]), cmd, False))
        for rgb, cmd in color_code_up_down.items():
            entries.append(((rgb[2], rgb[1], rgb[0]), cmd, True))
        for _bgr, cmd, ud in entries:
            rm.codes.append(cmd)
            rm.is_up_down.append(ud)
            rm.commands.append(Command.parse(cmd))

        files = sorted(p for p in glob.glob(os.path.join(dir_path, "route*.png"))
                       if not os.path.basename(p).startswith("route_rest"))
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"cannot read {f}")
            if img.shape[:2] != map_bgr.shape[:2]:
                img = cv2.resize(img, (map_bgr.shape[1], map_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            code = np.full(img.shape[:2], -1, np.int16)
            for idx, (bgr, _cmd, _ud) in enumerate(entries):
                colour = np.array(bgr, np.uint8)
                sel = np.all(img == colour, axis=2) & ~np.all(map_bgr == colour, axis=2)
                code[sel] = idx
            rm.routes.append(code)
        return rm

    @property
    def n_routes(self) -> int:
        return len(self.routes)


@dataclass(frozen=True)
class RouteHit:
    command: Command
    pixel: tuple[int, int]
    distance: int


def nearest_command(route: RouteMap, idx: int, pos: tuple[int, int], radius: int) -> tuple[RouteHit | None, RouteHit | None]:
    """Nearest main-code pixel and nearest up/down-code pixel within `radius` (Manhattan) of pos."""
    if not route.routes:
        return None, None
    code = route.routes[idx]
    h, w = code.shape
    x0, y0 = int(pos[0]), int(pos[1])
    xs0, xs1 = max(0, x0 - radius), min(w, x0 + radius + 1)
    ys0, ys1 = max(0, y0 - radius), min(h, y0 + radius + 1)
    if xs1 <= xs0 or ys1 <= ys0:
        return None, None
    win = code[ys0:ys1, xs0:xs1]
    yy, xx = np.mgrid[ys0:ys1, xs0:xs1]
    dist = np.abs(xx - x0) + np.abs(yy - y0)
    ud = np.array(route.is_up_down, bool)

    def pick(mask: np.ndarray) -> RouteHit | None:
        if not mask.any():
            return None
        d = np.where(mask, dist, np.iinfo(np.int32).max)
        i = int(np.argmin(d))
        py, px = divmod(i, d.shape[1])
        c = int(win[py, px])
        return RouteHit(route.commands[c], (int(xx[py, px]), int(yy[py, px])), int(dist[py, px]))

    valid = win >= 0
    is_ud = np.zeros_like(valid)
    if ud.size:
        is_ud[valid] = ud[win[valid]]
    return pick(valid & ~is_ud), pick(valid & is_ud)


def locate_minimap_on_map(map_bgr: np.ndarray, minimap_bgr: np.ndarray, last: tuple[int, int] | None,
                          local_radius: int = 50, local_thres: float = 0.4) -> MatchResult:
    """Where the live minimap crop sits on the stitched map (black minimap pixels ignored)."""
    mask = (np.any(minimap_bgr != 0, axis=2).astype(np.uint8) * 255)
    return match_template(map_bgr, minimap_bgr, mask=mask, last_loc=last,
                          local_radius=local_radius, local_thres=local_thres)


class RouteFollower:
    """Turns the player's global position into the route's movement command."""

    def __init__(self, route_map: RouteMap, cfg_route: dict):
        self.route = route_map
        self.search_range = int(cfg_route.get("search_range", 10))
        self.miss_grace = int(cfg_route.get("miss_grace_frames", 15))
        self.comp_dist = int(cfg_route.get("ladder_complement_dist", 3))
        self.idx = 0
        self.last = Command.NONE
        self.miss_frames = 0
        self.last_hit: RouteHit | None = None

    def step(self, pos: tuple[int, int] | None, on_ladder: bool) -> Command:
        if self.route.n_routes == 0:
            return Command.NONE
        if pos is None:
            hit = hit_ud = None
        else:
            hit, hit_ud = nearest_command(self.route, self.idx, pos, self.search_range)
        self.last_hit = hit or hit_ud
        if hit is None and hit_ud is None:
            self.miss_frames += 1
            if self.miss_frames > self.miss_grace:
                self.last = Command.NONE
            return self.last
        self.miss_frames = 0
        if hit is not None and hit_ud is not None:
            if hit.distance < hit_ud.distance:
                cmd = hit.command
                ud_y = hit_ud.command.move_y
                if cmd.move_y == "none" and hit_ud.distance <= self.comp_dist and \
                        (on_ladder or (cmd.action == "jump" and ud_y == "up")):
                    cmd = cmd.replace(move_y=ud_y)
            else:
                cmd = hit_ud.command
                if cmd.move_x == "none" and on_ladder and hit.distance <= self.comp_dist:
                    cmd = cmd.replace(move_x=hit.command.move_x)
        elif hit is not None:
            cmd = hit.command
        else:
            cmd = hit_ud.command
        if cmd.action == "goal":
            self.idx = (self.idx + 1) % self.route.n_routes
        self.last = cmd
        return cmd
