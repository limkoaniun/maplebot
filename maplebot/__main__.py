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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
