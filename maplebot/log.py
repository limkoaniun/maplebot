"""Shared logger: console + log/maplebot_<timestamp>.log."""
from __future__ import annotations

import datetime
import logging
import os

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "maplebot", log_dir: str = "log") -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)
        try:
            os.makedirs(log_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            fh = logging.FileHandler(os.path.join(log_dir, f"{name}_{stamp}.log"), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except OSError:
            pass
    _LOGGERS[name] = logger
    return logger
