"""Command line entry: python -m maplebot run [...]"""
from __future__ import annotations

import argparse
import sys

from maplebot.config import load_config
from maplebot.log import get_logger

log = get_logger()


def _run(args) -> int:
    from maplebot.bot import build_bot

    cfg = load_config(args.cfg)
    if args.map is not None:
        cfg["route"]["map"] = args.map
    if args.monsters:
        cfg["monsters"]["names"] = [m.strip() for m in args.monsters.split(",") if m.strip()]
    bot = build_bot(cfg, args.source, args.no_input)
    snap = bot.run(show=not args.no_viz, max_frames=args.frames)
    if snap is not None:
        print(f"frames={bot.frame_idx} {snap.summary()}")
    return 0


def _record_route(args) -> int:
    from maplebot.capture.source import ImageSource, VideoSource
    from maplebot.tools.route_recorder import run_recorder

    cfg = load_config(args.cfg)
    if args.source:
        source = VideoSource(args.source) if args.source.lower().endswith((".mp4", ".avi", ".mkv")) else ImageSource(args.source)
    else:
        from maplebot.capture.window import WindowCapture
        source = WindowCapture(cfg)
    out_dir = f"{cfg['route'].get('dir', 'minimaps')}/{args.name}"
    run_recorder(cfg, source, out_dir, existing_map=args.map)
    return 0


def _fetch_mob(args) -> int:
    from maplebot.tools.fetch_mob import fetch_mob

    n = fetch_mob(args.name, args.out, args.region, args.version)
    print(f"saved {n} sprite(s) to {args.out}/{args.name.lower().replace(' ', '_')}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maplebot", description="Pure-vision MapleStory auto-leveling bot")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the bot")
    r.add_argument("--cfg", default=None, help="user YAML merged over config/default.yaml")
    r.add_argument("--source", default=None, help="PNG or video to use instead of the live window")
    r.add_argument("--no-input", action="store_true", help="never press keys (watch only)")
    r.add_argument("--no-viz", action="store_true", help="no debug window")
    r.add_argument("--map", default=None, help="override route.map")
    r.add_argument("--monsters", default=None, help="comma-separated monster folder names")
    r.add_argument("--frames", type=int, default=None, help="stop after N frames (debugging)")
    r.set_defaults(func=_run)

    rr = sub.add_parser("record-route", help="stitch map.png and paint route*.png while you walk")
    rr.add_argument("name", help="map folder name under route.dir (e.g. my_map)")
    rr.add_argument("--cfg", default=None)
    rr.add_argument("--source", default=None, help="video/PNG instead of the live window")
    rr.add_argument("--map", default=None, help="continue from an existing map.png")
    rr.set_defaults(func=_record_route)

    fm = sub.add_parser("fetch-mob", help="download monster sprites from maplestory.io")
    fm.add_argument("name", help='mob name as listed on maplestory.wiki, e.g. "Orange Mushroom"')
    fm.add_argument("--out", default="monster")
    fm.add_argument("--region", default="GMS")
    fm.add_argument("--version", default="83")
    fm.set_defaults(func=_fetch_mob)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
