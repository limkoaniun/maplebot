"""Configuration loading: default YAML deep-merged with an optional user YAML."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PACKAGE_ROOT, "config", "default.yaml")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with `override` recursively merged over `base`."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(user_path: str | None = None) -> dict:
    """Load config/default.yaml and merge the user YAML on top if given."""
    cfg = load_yaml(DEFAULT_CONFIG_PATH)
    if user_path:
        cfg = deep_merge(cfg, load_yaml(user_path))
    return cfg


def parse_color_codes(d: dict[str, str]) -> dict[tuple[int, int, int], str]:
    """Turn {"255,0,0": "left none none"} into {(255, 0, 0): "left none none"}."""
    out: dict[tuple[int, int, int], str] = {}
    for key, cmd in d.items():
        parts = tuple(int(p.strip()) for p in str(key).split(","))
        if len(parts) != 3:
            raise ValueError(f"colour code key must be 'R,G,B', got {key!r}")
        out[parts] = cmd  # type: ignore[index]
    return out


def cfg_get(cfg: dict, path: str, default: Any = None) -> Any:
    """Dotted lookup: cfg_get(cfg, "attack.range_x", 300)."""
    node: Any = cfg
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
