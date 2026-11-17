"""PyInstaller entry point for Maplebot.exe and maplebot-cli.exe (keeps maplebot.app a module)."""
from maplebot.app import main

raise SystemExit(main())
