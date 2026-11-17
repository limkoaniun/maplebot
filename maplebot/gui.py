"""Native desktop settings and status panel for the local Maplebot runner."""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk

import yaml
import cv2

from maplebot.config import load_config

ROOT = Path(__file__).resolve().parents[1]
USER_CONFIG = ROOT / "config/user.yaml"
LOG = ROOT / "log"


def _read_user() -> dict:
    if not USER_CONFIG.exists():
        return {}
    value = yaml.safe_load(USER_CONFIG.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("The settings file must contain named settings.")
    return value


def _set(data: dict, path: str, value) -> None:
    pieces = path.split(".")
    for piece in pieces[:-1]:
        data = data.setdefault(piece, {})
    data[pieces[-1]] = value


def _startup_return_display(status):
    state = str(status.get("startup_return_state") or "idle").lower()
    reason = str(status.get("startup_return_reason") or "").strip()
    if state == "failed":
        return "Return to point 1 stopped", (reason or "No verified return is available.") + " Farming stays held; Resume farming checks a new attempt."
    if state == "checking":
        detail = reason or "Checking HP and the current platform before returning to point 1."
        if not status.get("foreground"):
            detail += " Bring MapleStory to the front."
        return "Preparing return to point 1", detail
    if state == "returning" and not status.get("returning_to_start"):
        return "Returning to point 1", reason or "Farming begins only after arrival at point 1 is confirmed."
    return None


def _recovery_display(status):
    """Describe recovery separately from the original blocked circuit point."""
    startup = _startup_return_display(status)
    if startup:
        return startup
    reason = str(status.get("recovery_reason") or "").strip()
    if not status.get("returning_to_start") and not status.get("recovery_terminal"):
        if reason and str(status.get("recovery_state") or "idle").lower() == "idle":
            return "Return unavailable", reason
        return None
    origin = status.get("recovery_from_index")
    origin_text = f"point {origin+1}" if isinstance(origin, int) and not isinstance(origin, bool) else "the current point"
    if status.get("recovery_terminal"):
        return "Recovery stopped", f"Return from {origin_text} stopped. {reason or 'No verified route is available.'}"
    index, count = status.get("recovery_index"), status.get("recovery_count")
    progress = (f" · return step {index+1}/{count}" if isinstance(index, int) and isinstance(count, int) and count > 0 else "")
    attempts = status.get("return_count")
    attempt = f" · attempt {attempts}" if isinstance(attempts, int) and attempts > 0 else ""
    detail = f"Returning from {origin_text}{progress}{attempt}."
    state = str(status.get("recovery_state") or "").replace("_", " ")
    if reason or state:
        detail += " " + (reason or state.capitalize())
    if status.get("paused"):
        detail += " Paused."
    elif not status.get("foreground"):
        detail += " Bring MapleStory to the front to continue."
    return "Returning to point 1", detail


def _health_protection_display(status):
    state = str(status.get("health_protection_state") or "idle").lower()
    reason = str(status.get("health_protection_reason") or status.get("health_hold_reason") or "").strip()
    active = bool(status.get("health_protection_active"))
    hold_error = str(status.get("hold_request_error") or "").strip()
    if state == "dead":
        label = "Character down — input stopped"
    elif state == "needs_attention":
        label = "Health needs attention"
    elif hold_error:
        return "Hold unavailable", hold_error
    elif active:
        startup_state = str(status.get("startup_return_state") or "").lower()
        label = ({"checking": "Preparing return — HP protection on",
                  "failed": "Return stopped — HP protection on",
                  "returning": "Return paused — HP protection on"}.get(startup_state)
                 or "Route stopped — HP protection on")
    elif state == "suspended" and status.get("full_pause") is False:
        label = "HP protection suspended"
    else:
        return None
    detail = reason or {
        "monitoring": "Watching the current HP reading while movement and combat stay stopped.",
        "healing": "Sending HP potion input while movement and combat stay stopped.",
        "awaiting_response": "Waiting for the HP reading to increase.",
        "needs_attention": "HP protection needs your attention. Check HP and potion availability.",
        "dead": "All input has stopped.",
        "suspended": "Waiting for a current game image before HP protection can continue.",
    }.get(state, "HP protection is active; movement and combat are stopped.")
    startup = _startup_return_display(status)
    if startup and startup[1] not in detail:
        detail = startup[1] + " " + detail
    diagnosis = str(status.get("health_protection_diagnosis_detail") or "").strip()
    if state == "needs_attention" and diagnosis and diagnosis not in detail:
        detail += " " + diagnosis
    taps = status.get("health_protection_taps")
    if isinstance(taps, int) and not isinstance(taps, bool) and taps > 0:
        detail += f" HP potion key presses: {taps}; a key press does not confirm HP restoration."
        responses = status.get("health_protection_responses")
        # One rise can answer several presses, so this counts rises, not presses.
        # A zero while healing only means the first response window is still open.
        if (isinstance(responses, int) and not isinstance(responses, bool) and 0 <= responses <= taps
                and (responses > 0 or state == "needs_attention")):
            rises = f"{responses} HP rise" + (" was" if responses == 1 else "s were")
            presses = f"{taps} sent press" + ("" if taps == 1 else "es")
            detail += f" {rises} observed after {presses}."
    return label, detail


def _pause_label(status, return_pending=False, hold_pending=False):
    if hold_pending:
        return "Pause all"
    if return_pending:
        return "Pause"
    if status.get("full_pause") is not True and (status.get("health_protection_active")
            or (status.get("paused") and status.get("full_pause") is False)):
        return "Pause all"
    return "Resume" if status.get("paused") or status.get("waiting_to_start") else "Pause"


def _can_resume_manual_hold(status):
    return bool(status.get("paused") and status.get("full_pause") is False
                and status.get("manual_health_hold") and not status.get("returning_to_start")
                and status.get("startup_return_state") not in {"checking", "returning"}
                and status.get("health_protection_state") != "dead"
                and not status.get("editor_test") and not status.get("potion_test"))


class SettingsWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Maplebot Settings")
        self.geometry("1440x900")
        self.minsize(1150, 820)
        self.configure(background="#eef3f9")
        self._configure_style()
        LOG.mkdir(exist_ok=True)
        self.values: dict[str, tk.Variable] = {}
        self.monsters: dict[str, tk.BooleanVar] = {}
        self.status: dict = {}
        self._health_alert_state = None
        self._health_alert_display = None
        self.launch_process = None
        self.launch_deadline = 0.0
        self.relay_process = None
        self.relay_output = None
        self.relay_image = None
        self.relay_frame = None
        self.relay_mtime = None
        self.relay_render_size = None
        self.relay_start_check = 0.0
        self.relay_state = tk.StringVar(value="Connecting to MapleStory…")
        self.relay_detail = tk.StringVar(value="Read-only view. Play using the MapleStory window.")
        self.analysis_state = tk.StringVar(value="No current analysis")
        self.route_text = tk.StringVar(value="No route — left/right sweep")
        self.analysis_detail = tk.StringVar(value="Start the bot to inspect detection. Pause its controls to inspect safely.")
        self.analysis_panels = {}
        self.map_editor = None
        self.editor_bridge = None
        self.message = tk.StringVar(value="Settings are saved locally. Closing this window leaves the bot running.")
        self.state_text = tk.StringVar(value="Stopped")
        self.vitals = tk.StringVar(value="HP  —     MP  —     Speed  —")
        self.activity = tk.StringVar(value="No active session")
        self.map_title = tk.StringVar(value="Character settings · No map profile selected")
        self._build()
        self._load()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self._ensure_relay()
        self.after(200, self._poll)
        self.after(250, self._poll_relay)

    def _configure_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10))
        style.configure("TFrame", background="#eef3f9")
        style.configure("Card.TFrame", background="white")
        style.configure("TLabel", background="#eef3f9", foreground="#24334b")
        style.configure("Card.TLabel", background="white", foreground="#24334b")
        style.configure("Muted.TLabel", foreground="#64748b")
        style.configure("Title.TLabel", font=("Segoe UI", 23, "bold"))
        style.configure("State.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1c62ba", background="white")
        style.configure("Alert.State.TLabel", font=("Segoe UI", 14, "bold"), foreground="#b42318", background="white")
        style.configure("TButton", padding=(14, 8), font=("Segoe UI", 10))
        style.configure("Accent.TButton", background="#2164bd", foreground="white")
        style.map("Accent.TButton", background=[("active", "#164b91"), ("disabled", "#a8bbd4")])
        style.configure("TNotebook", background="#eef3f9", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(9, 9), font=("Segoe UI", 10))
        style.map("TNotebook.Tab", background=[("selected", "white")], foreground=[("selected", "#1c62ba")])
        style.configure("TCheckbutton", background="#eef3f9", padding=(0, 5))
        style.configure("TEntry", padding=6)
        style.configure("TLabelframe", background="#eef3f9")
        style.configure("TLabelframe.Label", background="#eef3f9", font=("Segoe UI", 10, "bold"))

    def _build(self):
        shell = ttk.Frame(self, padding=24)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Maplebot", style="Title.TLabel").pack(anchor="w")
        ttk.Label(shell, textvariable=self.map_title, style="Muted.TLabel").pack(anchor="w", pady=(2, 16))
        card = ttk.Frame(shell, padding=16, style="Card.TFrame")
        card.pack(fill="x")
        top = ttk.Frame(card, style="Card.TFrame")
        top.pack(fill="x")
        self.state_label = ttk.Label(top, textvariable=self.state_text, style="State.TLabel")
        self.state_label.pack(side="left")
        ttk.Label(top, textvariable=self.vitals, style="Card.TLabel").pack(side="right")
        ttk.Label(card, textvariable=self.activity, style="Card.TLabel", wraplength=790).pack(anchor="w", pady=(8, 0))
        buttons = ttk.Frame(shell)
        buttons.pack(fill="x", pady=14)
        self.start_button = ttk.Button(buttons, text="Start active", style="Accent.TButton", command=self._start)
        self.start_button.pack(side="left")
        self.pause_button = ttk.Button(buttons, text="Pause", command=self._pause_resume)
        self.pause_button.pack(side="left", padx=8)
        self.hold_button = ttk.Button(buttons, text="Hold farming", command=self._hold_farming)
        self.hold_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(buttons, text="Stop", command=self._stop)
        self.stop_button.pack(side="left")
        self.return_button = ttk.Button(buttons, text="Return to point 1", command=self._return_to_start)
        self.return_button.pack(side="left", padx=8)
        ttk.Label(buttons, text="F1 pause/resume   ·   F12 stop", style="Muted.TLabel").pack(side="right")
        body = ttk.Panedwindow(shell, orient="horizontal")
        body.pack(fill="both", expand=True)
        settings = ttk.Frame(body)
        relay = ttk.Frame(body, padding=(18, 0, 0, 0))
        body.add(settings, weight=1)
        body.add(relay, weight=2)
        notebook = ttk.Notebook(settings)
        notebook.pack(fill="both", expand=True)
        tabs = {}
        for name in ("Controls", "Combat", "Monsters", "Potions", "Loot", "Circuit", "Detection"):
            tabs[name] = ttk.Frame(notebook, padding=(14, 18))
            tabs[name].columnconfigure(1, weight=1)
            notebook.add(tabs[name], text=name)
        controls = tabs["Controls"]
        for row, (label, path) in enumerate((("Attack skill", "keys.attack"), ("Normal attack (optional)", "keys.normal_attack"), ("Jump", "keys.jump"),
                                            ("HP potion", "keys.hp_potion"), ("MP potion", "keys.mp_potion"),
                                            ("Pick up items", "keys.pickup"))):
            self._field(controls, row, label, path)
        self.loot_enabled = tk.BooleanVar(value=True)
        self.post_combat_loot_enabled = tk.BooleanVar(value=False)
        loot = tabs["Loot"]
        ttk.Checkbutton(loot, text="Press pickup for nearby drops", variable=self.loot_enabled).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self._field(loot, 1, "Pick up every (seconds)", "keys.pickup_interval_s")
        ttk.Checkbutton(loot, text="Approach drops after combat", variable=self.post_combat_loot_enabled).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 5))
        self._field(loot, 3, "Wait over drops (seconds)", "loot.settle_s")
        self._field(loot, 4, "Maximum collection time (seconds)", "loot.max_approach_s")
        ttk.Label(loot, text="Uses the pickup key configured in Controls.\nAfter combat, briefly approaches the last target's position\nand waits for nearby drops before farming resumes.\nPickup key presses do not confirm that an item was collected.", style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(18, 0))
        combat = tabs["Combat"]
        for row, (label, path) in enumerate((("Attack reach (pixels)", "attack.range_x"),
                                            ("Attack height (pixels)", "attack.range_y"),
                                            ("Time between attacks (seconds)", "attack.cooldown_s"),
                                            ("Use normal attack below MP (%)", "attack.mp_fallback_percent"),
                                            ("Stop approaching within (pixels)", "attack.deadzone_px"),
                                            ("Turn around after (frames)", "search.flip_frames"))):
            self._field(combat, row, label, path)
        ttk.Label(combat, text="Current preset uses close-range Double Stab and left/right sweeping.\nLeave the normal-attack key empty to disable the low-MP fallback.", style="Muted.TLabel").grid(row=7, column=0, columnspan=2, sticky="w", pady=18)
        self.monster_tab = tabs["Monsters"]
        ttk.Label(self.monster_tab, text="Select the monsters to recognize on this map.").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        potions = tabs["Potions"]
        for row, (label, path) in enumerate((("Use HP potion below (%)", "health.hp_percent"),
                                            ("Use MP potion below (%)", "health.mp_percent"),
                                            ("Minimum HP for farming (%)", "health.minimum_hp_percent"),
                                            ("HP retry interval (seconds)", "health.protection_retry_s"),
                                            ("HP presses without improvement", "health.protection_max_taps"),
                                            ("MP retry interval (seconds)", "health.retry_s"),
                                            ("MP retries before waiting longer", "health.max_retries"))):
            self._field(potions, row, label, path)
        self.protect_on_route_failure = tk.BooleanVar(value=True)
        ttk.Checkbutton(potions, text="Keep HP protection on after a route failure", variable=self.protect_on_route_failure).grid(row=7, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Label(potions, text="HP refills pause farming until the recovery target is confirmed.\nHP and MP thresholds are independent. Keep potions on their assigned keys.\nPause all and Stop also stop potion use.", style="Muted.TLabel", wraplength=500).grid(row=8, column=0, columnspan=2, sticky="w", pady=14)
        circuit = tabs["Circuit"]
        self.circuit_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(circuit, text="Follow the current map's farming circuit", variable=self.circuit_enabled).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        map_buttons = ttk.Frame(circuit)
        map_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Button(map_buttons, text="Open map editor", style="Accent.TButton", command=self._open_map_editor).pack(side="left")
        ttk.Button(map_buttons, text="Calibrate minimap", command=self._calibrate_minimap).pack(side="left", padx=8)
        self.circuit_edit_hint = tk.StringVar(value="Advanced: X, Y, action, wait seconds per line.\nActions: walk, climb, jump, or drop.")
        ttk.Label(circuit, textvariable=self.circuit_edit_hint, style="Muted.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 6))
        self.circuit_points = tk.Text(circuit, height=4, width=46, wrap="none", font=("Consolas", 10), background="white", foreground="#24334b", relief="solid", borderwidth=1, padx=7, pady=6)
        self.circuit_points.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._field(circuit, 4, "Horizontal arrival tolerance", "route.circuit.tolerance_x")
        self._field(circuit, 5, "Vertical arrival tolerance", "route.circuit.tolerance_y")
        self._field(circuit, 6, "Waypoint timeout (seconds)", "route.circuit.timeout_s")
        self.return_on_block = tk.BooleanVar(value=False)
        ttk.Checkbutton(circuit, text="Return to point 1 if route blocks", variable=self.return_on_block).grid(row=7, column=0, sticky="w", pady=(6, 0))
        return_limit = ttk.Frame(circuit)
        return_limit.grid(row=7, column=1, sticky="w", pady=(6, 0))
        ttk.Label(return_limit, text="Max per lap:").pack(side="left", padx=(0, 5))
        self.values["route.circuit.max_returns_per_lap"] = tk.StringVar(value="1")
        ttk.Entry(return_limit, textvariable=self.values["route.circuit.max_returns_per_lap"], width=3).pack(side="left")
        ttk.Label(circuit, text="Returns use saved map connections and stop if no path is known.\nOpening the editor pauses movement; save a profile before resuming.", style="Muted.TLabel", wraplength=500).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))
        detection = tabs["Detection"]
        ttk.Label(detection, text="Calibrated for your current 1920 × 1080 game window.", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        self._field(detection, 1, "Character image", "player.nametag.path", width=28)
        self._field(detection, 2, "Character match tolerance", "player.nametag.diff_thres")
        self._field(detection, 3, "Monster match tolerance", "monsters.diff_thres")
        self._field(detection, 4, "Stop after missing character (frames)", "safety.max_missing_frames")
        ttk.Label(detection, text="Smaller tolerances require a closer image match.\nChanging the game's resolution or appearance may require recalibration.\nKeep the game in front for movement and attacks.", style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=16)
        footer = ttk.Frame(settings)
        footer.pack(fill="x", pady=(15, 0))
        ttk.Button(footer, text="Reload settings", command=self._load).pack(side="left")
        self.save_button = ttk.Button(footer, text="Save & apply", style="Accent.TButton", command=self._save)
        self.save_button.pack(side="right")
        ttk.Label(settings, textvariable=self.message, style="Muted.TLabel", wraplength=530).pack(anchor="w", pady=(12, 0))
        relay_heading = ttk.Frame(relay)
        relay_heading.pack(fill="x", pady=(2, 10))
        ttk.Label(relay_heading, text="Detection & game view", font=("Segoe UI", 13, "bold")).pack(side="left")
        view_tabs = ttk.Notebook(relay)
        view_tabs.pack(fill="both", expand=True)
        bot_view = ttk.Frame(view_tabs, padding=(0, 8, 0, 0))
        game_view = ttk.Frame(view_tabs, padding=(0, 8, 0, 0))
        view_tabs.add(bot_view, text="Bot view")
        view_tabs.add(game_view, text="Game view")
        bot_view.columnconfigure(0, weight=1)
        bot_view.rowconfigure(1, weight=3)
        bot_view.rowconfigure(2, weight=2)
        analysis_heading = ttk.Frame(bot_view)
        analysis_heading.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        ttk.Label(analysis_heading, textvariable=self.analysis_state, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(analysis_heading, textvariable=self.route_text, style="Muted.TLabel").pack(side="right")
        self._analysis_canvas(bot_view, "view", 1, 1).grid(row=1, column=0, sticky="nsew")
        detail_views = ttk.Frame(bot_view)
        detail_views.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        detail_views.rowconfigure(0, weight=1)
        detail_views.columnconfigure(0, weight=1)
        detail_views.columnconfigure(1, weight=2)
        mini = ttk.Frame(detail_views)
        mini.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        bars = ttk.Frame(detail_views)
        bars.grid(row=0, column=1, sticky="nsew")
        ttk.Label(mini, text="Minimap · click to edit", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(bars, text="HP / MP / EXP readings", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        minimap_canvas = self._analysis_canvas(mini, "minimap", 1, 1)
        minimap_canvas.pack(fill="both", expand=True)
        minimap_canvas.configure(cursor="hand2")
        minimap_canvas.bind("<Button-1>", lambda event: self._open_map_editor())
        self._analysis_canvas(bars, "bars", 1, 1).pack(fill="both", expand=True)
        analysis_footer = ttk.Label(bot_view, textvariable=self.analysis_detail, style="Muted.TLabel", wraplength=770)
        analysis_footer.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        bot_view.bind("<Configure>", lambda event: analysis_footer.configure(wraplength=max(200, event.width - 10)))
        game_view.columnconfigure(0, weight=1)
        game_view.rowconfigure(1, weight=1)
        ttk.Label(game_view, textvariable=self.relay_state, style="Muted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 7))
        self.relay_canvas = tk.Canvas(game_view, background="#142238", highlightthickness=0, width=1, height=1)
        self.relay_canvas.grid(row=1, column=0, sticky="nsew")
        self.relay_image_id = self.relay_canvas.create_image(0, 0, anchor="center")
        self.relay_placeholder = self.relay_canvas.create_text(0, 0, text="Waiting for MapleStory", fill="#dbe7fa", font=("Segoe UI", 13), width=480, justify="center")
        self.relay_canvas.bind("<Configure>", lambda event: self._render_relay())
        relay_footer = ttk.Label(game_view, textvariable=self.relay_detail, style="Muted.TLabel", wraplength=770)
        relay_footer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        game_view.bind("<Configure>", lambda event: relay_footer.configure(wraplength=max(200, event.width - 10)))
        self.after_idle(lambda: body.sashpos(0, 570))

    def _analysis_canvas(self, parent, key, width, height):
        canvas = tk.Canvas(parent, background="#142238", highlightthickness=0, width=width, height=height)
        image_id = canvas.create_image(0, 0, anchor="center")
        text_id = canvas.create_text(0, 0, text="No current analysis", fill="#dbe7fa", font=("Segoe UI", 11), justify="center")
        self.analysis_panels[key] = {"canvas": canvas, "image_id": image_id, "text_id": text_id,
                                     "frame": None, "photo": None, "modified": None}
        canvas.bind("<Configure>", lambda event, panel_key=key: self._render_analysis(panel_key))
        return canvas

    def _field(self, frame, row, label, path, width=16):
        var = tk.StringVar()
        self.values[path] = var
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 22), pady=5)
        ttk.Entry(frame, textvariable=var, width=width).grid(row=row, column=1, sticky="w", pady=5)

    def _load(self):
        try:
            cfg = load_config(str(USER_CONFIG)) if USER_CONFIG.exists() else load_config()
            self.map_title.set("Character settings · " + str(cfg.get("map_profile", {}).get("name") or "No map profile selected"))
            self._loaded_map_identity = (cfg.get("map_profile", {}).get("active"), tuple(cfg.get("minimap", {}).get("fixed_rect") or []))
            self._loaded_config_mtime = USER_CONFIG.stat().st_mtime_ns if USER_CONFIG.exists() else None
            for path, var in self.values.items():
                value = cfg
                for part in path.split("."):
                    value = value.get(part, {}) if isinstance(value, dict) else {}
                fallback = {"keys.normal_attack": "", "attack.mp_fallback_percent": "5",
                            "route.circuit.tolerance_x": "2", "route.circuit.tolerance_y": "2",
                            "route.circuit.timeout_s": "12", "loot.settle_s": "0.8",
                            "route.circuit.max_returns_per_lap": "1",
                            "health.minimum_hp_percent": "50", "health.hp_percent": "60",
                            "health.protection_retry_s": "1.0", "health.protection_max_taps": "3",
                            "loot.max_approach_s": "2.0"}.get(path, "3")
                var.set(str(value) if value != {} else fallback)
            pickup = str(cfg["keys"].get("pickup") or "")
            self._saved_pickup_key = pickup
            self.loot_enabled.set(bool(pickup))
            self.post_combat_loot_enabled.set(bool((cfg.get("loot") or {}).get("enabled", False)))
            self.protect_on_route_failure.set(bool((cfg.get("health") or {}).get("protect_on_route_failure", True)))
            self._saved_health_enabled = bool((cfg.get("health") or {}).get("enable", True))
            self._saved_hp_key = str(cfg.get("keys", {}).get("hp_potion") or "")
            self.values["keys.pickup"].set(pickup or "x")
            circuit = cfg.get("route", {}).get("circuit") or {}
            self._saved_circuit_enabled = bool(circuit.get("enabled", False))
            self.circuit_enabled.set(self._saved_circuit_enabled)
            self.return_on_block.set(bool(circuit.get("return_on_block", False)))
            points = []
            for point in circuit.get("waypoints") or []:
                if isinstance(point, dict):
                    points.append(f"{point.get('x', 0)}, {point.get('y', 0)}, {point.get('action', 'walk')}, {point.get('dwell_s', .5)}")
            self.circuit_points.configure(state="normal")
            self.circuit_points.delete("1.0", "end")
            self.circuit_points.insert("1.0", "\n".join(points))
            if circuit.get("editor_plan"):
                self.circuit_points.configure(state="disabled")
                self.circuit_edit_hint.set("Map editor controls this route.\nGenerated movement steps are shown below for reference.")
            else:
                self.circuit_edit_hint.set("Advanced: X, Y, action, wait seconds per line.\nActions: walk, climb, jump, or drop.")
            for widget in self.monster_tab.winfo_children()[1:]:
                widget.destroy()
            self.monsters.clear()
            root = ROOT / str(cfg.get("monsters", {}).get("dir", "monster"))
            names = sorted(path.name for path in root.iterdir() if path.is_dir() and any(path.glob("*.png"))) if root.exists() else []
            selected = cfg.get("monsters", {}).get("names") or []
            for i, name in enumerate(names):
                var = tk.BooleanVar(value=name in selected)
                self.monsters[name] = var
                ttk.Checkbutton(self.monster_tab, text=name.replace("_", " ").title(), variable=var).grid(row=1 + i // 2, column=i % 2, sticky="w", padx=(0, 40), pady=4)
            self.message.set("Settings loaded. Changes take effect when you choose Save & apply.")
        except Exception as exc:
            messagebox.showerror("Could not load settings", str(exc), parent=self)

    def _validated(self) -> dict:
        data = _read_user()
        special = {"alt", "alt_l", "alt_r", "ctrl", "ctrl_l", "ctrl_r", "shift", "shift_l", "shift_r", "space", "enter", "tab", "esc", "backspace", "delete", "insert", "home", "end", "page_up", "page_down", "left", "right", "up", "down"}
        special.update(f"f{i}" for i in range(1, 13))
        for path in ("keys.attack", "keys.normal_attack", "keys.jump", "keys.hp_potion", "keys.mp_potion", "keys.pickup"):
            value = self.values[path].get().strip().lower()
            if path == "keys.pickup" and not self.loot_enabled.get():
                value = ""
            elif path == "keys.normal_attack" and not value:
                pass
            elif (len(value) != 1 and value not in special) or value in {"f1", "f2", "f12"}:
                raise ValueError(f"Enter a valid key for {path.split('.')[-1].replace('_', ' ')}. F1, F2, and F12 are reserved for bot controls.")
            _set(data, path, value)
        bounds = {
            "keys.pickup_interval_s": (float, .1, 10), "attack.range_x": (int, 10, 600),
            "attack.range_y": (int, 10, 300), "attack.cooldown_s": (float, .1, 10),
            "attack.mp_fallback_percent": (float, 0, 100),
            "attack.deadzone_px": (int, 0, 200), "search.flip_frames": (int, 6, 1200),
            "health.hp_percent": (float, 1, 100), "health.mp_percent": (float, 1, 99),
            "health.minimum_hp_percent": (float, 0, 99),
            "health.protection_retry_s": (float, .5, 10), "health.protection_max_taps": (int, 1, 5),
            "health.retry_s": (float, .5, 60), "health.max_retries": (int, 0, 10),
            "player.nametag.diff_thres": (float, .01, .5), "monsters.diff_thres": (float, .01, .5),
            "safety.max_missing_frames": (int, 1, 30),
            "route.circuit.tolerance_x": (float, .1, 20), "route.circuit.tolerance_y": (float, .1, 20),
            "route.circuit.timeout_s": (float, 1, 120),
            "route.circuit.max_returns_per_lap": (int, 1, 3),
            "loot.settle_s": (float, 0, 3), "loot.max_approach_s": (float, .2, 5),
        }
        for path, (kind, low, high) in bounds.items():
            label = path.split(".")[-1].replace("_", " ")
            try:
                value = kind(self.values[path].get().strip())
            except ValueError:
                raise ValueError(f"Enter a number for {label}.") from None
            if not low <= value <= high:
                raise ValueError(f"{label.capitalize()} must be between {low} and {high}.")
            _set(data, path, value)
        if data["health"]["hp_percent"] <= data["health"]["minimum_hp_percent"]:
            raise ValueError("The HP potion threshold must be higher than the minimum HP for farming.")
        _set(data, "health.protect_on_route_failure", self.protect_on_route_failure.get())
        asset = self.values["player.nametag.path"].get().strip()
        if not asset or not (ROOT / asset).is_file():
            raise ValueError("The character image could not be found. Choose an existing image path.")
        _set(data, "player.nametag.path", asset)
        selected = [name for name, var in self.monsters.items() if var.get()]
        if not selected:
            raise ValueError("Select at least one monster to recognize.")
        _set(data, "monsters.names", selected)
        if self.post_combat_loot_enabled.get() and not self.loot_enabled.get():
            raise ValueError("Enable nearby pickup presses before enabling post-combat collection.")
        _set(data, "loot.enabled", self.post_combat_loot_enabled.get())
        current_circuit = (data.get("route") or {}).get("circuit") or {}
        previous = current_circuit.get("waypoints") or []
        points = []
        # A graphical plan owns its compiled waypoints, including jump takeoff
        # coordinates and connection IDs. Saving unrelated settings must not
        # reconstruct them from the intentionally abbreviated display.
        lines = [] if current_circuit.get("editor_plan") else self.circuit_points.get("1.0", "end").splitlines()
        if current_circuit.get("editor_plan"):
            points = previous
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) not in (2, 3, 4):
                raise ValueError(f"Circuit line {line_number}: use X, Y, action, wait seconds.")
            try:
                x, y = float(parts[0]), float(parts[1])
                dwell = float(parts[3]) if len(parts) == 4 else .5
            except ValueError:
                raise ValueError(f"Circuit line {line_number}: coordinates and wait time must be numbers.") from None
            action = parts[2].lower() if len(parts) >= 3 else "walk"
            if action not in {"walk", "climb", "jump", "drop", "ramp"}:
                raise ValueError(f"Circuit line {line_number}: action must be walk, climb, jump, drop, or ramp.")
            if not all(math.isfinite(value) for value in (x, y, dwell)) or x < 0 or y < 0 or not 0 <= dwell <= 10:
                raise ValueError(f"Circuit line {line_number}: use nonnegative coordinates and a wait of 0–10 seconds.")
            point = dict(previous[len(points)]) if len(points) < len(previous) and isinstance(previous[len(points)], dict) else {}
            point.update(x=x, y=y, action=action, dwell_s=dwell)
            points.append(point)
        if self.circuit_enabled.get() and len(points) < 2:
            raise ValueError("An enabled farming circuit needs at least two waypoints.")
        _set(data, "route.circuit.enabled", self.circuit_enabled.get())
        _set(data, "route.circuit.return_on_block", self.return_on_block.get())
        _set(data, "route.circuit.waypoints", points)
        return data

    def _save(self) -> bool:
        if self._editing_map():
            return False
        try:
            data = self._validated()
            from maplebot.editor_bridge import EditorBridge
            EditorBridge(ROOT).sync_active_settings(data)
            pending = USER_CONFIG.with_suffix(".yaml.tmp")
            pending.write_text("# Local Maplebot settings\n" + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
            pending.replace(USER_CONFIG)
            self._saved_pickup_key = data.get("keys", {}).get("pickup", "")
            self._saved_health_enabled = bool(data.get("health", {}).get("enable", True))
            self._saved_hp_key = str(data.get("keys", {}).get("hp_potion") or "")
            self._saved_circuit_enabled = bool(data.get("route", {}).get("circuit", {}).get("enabled"))
            if self._running():
                (LOG / "reload-active").touch()
                self.message.set("Saved. Applying these settings to the running bot…")
            else:
                self.message.set("Saved. Ready to start active mode.")
            return True
        except Exception as exc:
            messagebox.showerror("Check your settings", str(exc), parent=self)
            return False

    def _running(self) -> bool:
        return bool(self.status.get("running") and time.time() - self.status.get("updated", 0) < 6)

    def _editing_map(self) -> bool:
        return self.map_editor is not None and bool(self.map_editor.winfo_exists())

    def _open_map_editor(self):
        if self._editing_map():
            self.map_editor.lift()
            self.map_editor.focus_set()
            return
        try:
            from maplebot.editor_bridge import EditorBridge
            from maplebot.map_editor import MapEditor
            self.editor_bridge = EditorBridge(ROOT)
            self.editor_bridge.pause()
            plan, frame, player = self.editor_bridge.load_current()

            def save_plan(plan, image, apply, map_settings=None):
                result = self.editor_bridge.save_profile(plan, image, apply=apply, map_settings=map_settings)
                if apply:
                    self._load()
                self.message.set("Map profile saved. The bot stays paused until you resume it.")
                return result

            self.map_editor = MapEditor(self, plan, frame, save_plan,
                                        self.editor_bridge.test_connection,
                                        self.editor_bridge.stop_test,
                                        player_pos=player, bridge=self.editor_bridge)
            self.message.set("Map editor is open. Movement is paused; use its connection test to check a path.")
            self.start_button.configure(state="disabled")
            self.pause_button.configure(state="disabled")
        except Exception as exc:
            messagebox.showerror("Could not open map editor", str(exc), parent=self)

    def _calibrate_minimap(self):
        if self._editing_map():
            self.map_editor.lift()
            return
        try:
            from maplebot.editor_bridge import EditorBridge
            from maplebot.map_editor import MinimapCropDialog
            self.editor_bridge = EditorBridge(ROOT)
            self.editor_bridge.pause()
            frame, metadata = self.editor_bridge.full_snapshot()
            def selected(rect):
                self.editor_bridge.calibrate_crop(rect, frame, metadata)
                self._load()
                self.after_idle(self._open_map_editor)
            self.map_editor = MinimapCropDialog(self, frame, selected)
            self.message.set("Calibrating a new map. The previous route will remain disabled until a profile is applied.")
        except Exception as exc:
            messagebox.showerror("Could not calibrate minimap", str(exc), parent=self)

    def _start(self):
        if self._editing_map():
            return
        if self._running():
            if _can_resume_manual_hold(self.status):
                SettingsWindow._resume_farming(self)
            return
        if time.time() < self.launch_deadline:
            return
        if not self._save():
            return
        try:
            self.launch_process = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-File", str(ROOT / "scripts/start_active.ps1")],
                cwd=str(ROOT), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.launch_deadline = time.time() + 45
            self.message.set("Approve the Windows administrator prompt, then bring MapleStory to the front.")
            self.start_button.configure(state="disabled")
        except OSError as exc:
            self.launch_deadline = 0
            messagebox.showerror("Could not start Maplebot", str(exc), parent=self)

    def _resume_farming(self):
        for name in ("pause-active", "hold-farming-active", "return-to-start-active"):
            (LOG / name).unlink(missing_ok=True)
        (LOG / "resume-active").touch()
        (LOG / "arm-active").touch()
        if self.status.get("circuit_enabled", getattr(self, "_saved_circuit_enabled", False)):
            self.message.set("Resume requested. The bot checks HP and returns to point 1 before farming. Bring MapleStory to the front.")
        else:
            self.message.set("Resume requested. Bring MapleStory to the front for movement and combat.")

    def _pause_resume(self):
        if self._editing_map() or not self._running():
            return
        if (LOG / "hold-farming-active").exists():
            for name in ("hold-farming-active", "return-to-start-active", "resume-active"):
                (LOG / name).unlink(missing_ok=True)
            (LOG / "pause-active").touch()
            self.message.set("Hold cancelled. All input, including HP protection, stays paused.")
            return
        if (LOG / "return-to-start-active").exists():
            (LOG / "return-to-start-active").unlink(missing_ok=True)
            (LOG / "resume-active").unlink(missing_ok=True)
            (LOG / "pause-active").touch()
            self.message.set("Return cancelled. The controller stays paused.")
            return
        if _pause_label(self.status) == "Pause all":
            (LOG / "resume-active").unlink(missing_ok=True)
            (LOG / "pause-active").touch()
            self.message.set("Pausing all input, including HP protection…")
            return
        if self.status.get("paused") or self.status.get("waiting_to_start"):
            SettingsWindow._resume_farming(self)
        else:
            (LOG / "return-to-start-active").unlink(missing_ok=True)
            (LOG / "resume-active").unlink(missing_ok=True)
            (LOG / "pause-active").touch()
            self.message.set("Pausing the bot…")

    def _stop(self):
        (LOG / "return-to-start-active").unlink(missing_ok=True)
        (LOG / "hold-farming-active").unlink(missing_ok=True)
        (LOG / "stop-active").touch()
        self.message.set("Stopping the bot and releasing its keys…")

    def _hold_farming(self):
        if self._editing_map() or not self._running():
            return
        try:
            from maplebot.editor_bridge import EditorBridge
            self.message.set(EditorBridge(ROOT).hold_farming())
            self.hold_button.configure(state="disabled")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not hold farming", str(exc), parent=self)

    def _return_to_start(self):
        if self._editing_map() or not self._running():
            return
        try:
            from maplebot.editor_bridge import EditorBridge
            self.message.set(EditorBridge(ROOT).return_to_start())
            self.return_button.configure(state="disabled")
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not return to point 1", str(exc), parent=self)

    def _ensure_relay(self):
        now = time.time()
        if now - self.relay_start_check < 3:
            return
        self.relay_start_check = now
        try:
            status = json.loads((LOG / "relay-status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        if status.get("running") and now - status.get("updated", 0) < 4:
            (LOG / "stop-relay").unlink(missing_ok=True)
            return
        if self.relay_process is not None and self.relay_process.poll() is None:
            return
        try:
            (LOG / "stop-relay").unlink(missing_ok=True)
            if self.relay_output is not None:
                self.relay_output.close()
            self.relay_output = open(LOG / "relay-helper.log", "a", encoding="utf-8")
            self.relay_process = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts/relay_client.py")], cwd=str(ROOT),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                stdout=self.relay_output, stderr=self.relay_output,
            )
        except OSError as exc:
            self.relay_state.set("Preview unavailable")
            self.relay_detail.set(str(exc))

    def _render_relay(self):
        if not hasattr(self, "relay_canvas"):
            return
        width = self.relay_canvas.winfo_width()
        height = self.relay_canvas.winfo_height()
        self.relay_canvas.coords(self.relay_image_id, width // 2, height // 2)
        self.relay_canvas.coords(self.relay_placeholder, width // 2, height // 2)
        self.relay_canvas.itemconfigure(self.relay_placeholder, width=max(100, width - 48))
        if self.relay_frame is None or min(width, height) <= 24:
            return
        frame_h, frame_w = self.relay_frame.shape[:2]
        scale = min((width - 20) / frame_w, (height - 20) / frame_h, 1.0)
        target = (max(1, int(frame_w * scale)), max(1, int(frame_h * scale)))
        frame = cv2.resize(self.relay_frame, target, interpolation=cv2.INTER_AREA) if target != (frame_w, frame_h) else self.relay_frame
        ok, encoded = cv2.imencode(".png", frame)
        if not ok:
            return
        self.relay_image = tk.PhotoImage(data=encoded.tobytes())
        self.relay_canvas.itemconfigure(self.relay_image_id, image=self.relay_image)
        self.relay_render_size = (width, height)

    def _render_analysis(self, key):
        panel = self.analysis_panels[key]
        canvas = panel["canvas"]
        width, height = canvas.winfo_width(), canvas.winfo_height()
        canvas.coords(panel["image_id"], width // 2, height // 2)
        canvas.coords(panel["text_id"], width // 2, height // 2)
        canvas.itemconfigure(panel["text_id"], width=max(100, width - 24))
        frame = panel["frame"]
        if frame is None or min(width, height) <= 12:
            return
        frame_h, frame_w = frame.shape[:2]
        scale = min((width - 12) / frame_w, (height - 12) / frame_h, 1.0)
        target = (max(1, int(frame_w * scale)), max(1, int(frame_h * scale)))
        image = cv2.resize(frame, target, interpolation=cv2.INTER_AREA) if target != (frame_w, frame_h) else frame
        ok, encoded = cv2.imencode(".png", image)
        if ok:
            panel["photo"] = tk.PhotoImage(data=encoded.tobytes())
            canvas.itemconfigure(panel["image_id"], image=panel["photo"])

    def _poll_analysis(self, now):
        status = self.status
        circuit_enabled = status.get("circuit_enabled", getattr(self, "_saved_circuit_enabled", False))
        circuit_state = str(status.get("circuit_state") or "").lower()
        recovery = _recovery_display(status)
        protection = _health_protection_display(status)
        if recovery:
            index, count = status.get("recovery_index"), status.get("recovery_count")
            progress = f" · {index+1}/{count}" if isinstance(index, int) and isinstance(count, int) and count > 0 else ""
            self.route_text.set(recovery[0] + progress)
        elif circuit_enabled:
            index = status.get("circuit_index", status.get("circuit_idx"))
            count = status.get("circuit_count")
            point = str(index + 1) if isinstance(index, int) else "—"
            if isinstance(count, int) and count > 0:
                point += f"/{count}"
            self.route_text.set(f"Circuit blocked — point {point}" if circuit_state == "blocked"
                                else f"Circuit point {point}" + (f" · {circuit_state}" if circuit_state else ""))
        else:
            self.route_text.set("Recorded route loaded" if status.get("route_loaded") else "No route — left/right sweep")
        fresh = (status.get("running") and now - (status.get("updated") or 0) < 3
                 and now - (status.get("diagnostic_updated") or 0) < 3)
        for key, panel in self.analysis_panels.items():
            path = LOG / f"bot-{key}.png"
            try:
                modified = path.stat().st_mtime
            except OSError:
                modified = 0
            available = fresh and now - modified < 3
            if available and modified != panel["modified"]:
                frame = cv2.imread(str(path))
                if frame is not None:
                    panel["frame"] = frame
                    panel["modified"] = modified
                    self._render_analysis(key)
                else:
                    available = False
            canvas = panel["canvas"]
            if available and panel["photo"] is not None:
                canvas.itemconfigure(panel["image_id"], state="normal")
                canvas.itemconfigure(panel["text_id"], state="hidden")
            else:
                canvas.itemconfigure(panel["image_id"], state="hidden")
                text = "No current analysis"
                if key == "view":
                    text = "Start the bot to inspect its detection.\nYou can pause its controls while viewing." if not self._running() else "Waiting for a fresh engine analysis…"
                canvas.itemconfigure(panel["text_id"], state="normal", text=text)
        if fresh:
            self.analysis_state.set(protection[0] if protection else "Paused — current analysis" if status.get("paused") else "Live analysis")
            score = status.get("player_score")
            score_text = "—" if score is None else f"{score:.3f}"
            source = str(status.get("player_source") or "unknown").replace("_", " ")
            if status.get("vertical_minimap_active") or status.get("minimap_route_active"):
                source = "temporarily hidden; minimap navigation"
                score_text = "—"
            target = str(status.get("target") or "none")
            missing = status.get("missing_frames", 0)
            count = len(status.get("detections") or status.get("monsters") or [])
            loot_state = str(status.get("loot_state") or "idle")
            detail = f"Character: {source}   ·   Match: {score_text}   ·   Missing frames: {missing}   ·   Detected: {count}   ·   Target: {target}   ·   Loot: {loot_state}"
            if protection:
                detail += "\n" + protection[1]
            elif recovery:
                detail += "\n" + recovery[1]
            elif circuit_state == "blocked":
                detail += "\nCircuit blocked. Use Return to point 1, or edit its waypoints and Save & apply."
            self.analysis_detail.set(detail)
        else:
            self.analysis_state.set("No current analysis")
            self.analysis_detail.set("Bot view uses the engine's own detection and bar readings. Game view remains available while stopped.")

    def _poll_relay(self):
        self._ensure_relay()
        now = time.time()
        try:
            status = json.loads((LOG / "relay-status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            status = {}
        path = LOG / "relay-frame.png"
        try:
            modified = path.stat().st_mtime
        except OSError:
            modified = 0
        fresh = (status.get("running") and status.get("frame_available", True)
                 and now - status.get("updated", 0) < 3
                 and now - status.get("last_frame", modified) < 3
                 and now - modified < 3)
        if fresh and modified != self.relay_mtime:
            frame = cv2.imread(str(path))
            if frame is not None:
                self.relay_frame = frame
                self.relay_mtime = modified
                self._render_relay()
            else:
                fresh = False
        if fresh and self.relay_image is not None:
            self.relay_state.set("Live")
            self.relay_detail.set("Read-only preview. Use the MapleStory window for mouse and keyboard input.")
            self.relay_canvas.itemconfigure(self.relay_placeholder, state="hidden")
            self.relay_canvas.itemconfigure(self.relay_image_id, state="normal")
        else:
            error = str(status.get("error") or "")
            self.relay_state.set("Waiting for MapleStory")
            self.relay_detail.set(error or "Keep MapleStory open to see its live view here.")
            self.relay_canvas.itemconfigure(self.relay_image_id, state="hidden")
            self.relay_canvas.itemconfigure(self.relay_placeholder, state="normal", text=error or "Waiting for a fresh MapleStory frame…")
        self._poll_analysis(now)
        self.after(250, self._poll_relay)

    def _close(self):
        if self._editing_map():
            self.map_editor._close()
            if self._editing_map():
                return
        try:
            (LOG / "stop-relay").touch()
        except OSError:
            pass
        if self.relay_output is not None:
            self.relay_output.close()
        self.destroy()

    def _update_health_alert(self, running):
        if not running:
            return  # retain a visible last warning when the controller stops reporting
        state = str(self.status.get("health_protection_state") or "idle").lower()
        alert = state if state in {"needs_attention", "dead"} else None
        if alert and alert != self._health_alert_state:
            try:
                self.bell()
            except tk.TclError:
                pass
        self._health_alert_state = alert
        self._health_alert_display = _health_protection_display(self.status) if alert else None

    def _poll(self):
        try:
            modified = USER_CONFIG.stat().st_mtime_ns
            if modified != getattr(self, "_loaded_config_mtime", None) and not self._editing_map():
                current = load_config(str(USER_CONFIG))
                identity = (current.get("map_profile", {}).get("active"), tuple(current.get("minimap", {}).get("fixed_rect") or []))
                if identity != getattr(self, "_loaded_map_identity", None):
                    self._load()
                    self.message.set("The active map changed. Its settings are loaded; the bot remains paused until resumed.")
                self._loaded_config_mtime = modified
        except (OSError, ValueError):
            pass
        try:
            self.status = json.loads((LOG / "active-status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        running = self._running()
        self._update_health_alert(running)
        self.state_label.configure(style="Alert.State.TLabel" if self._health_alert_state else "State.TLabel")
        if running:
            self.launch_deadline = 0
            circuit_state = str(self.status.get("circuit_state") or "").lower()
            loot_state = str(self.status.get("loot_state") or "idle").lower()
            recovery = _recovery_display(self.status)
            protection = _health_protection_display(self.status)
            if protection:
                label = protection[0]
            elif recovery:
                label = recovery[0]
            elif self.status.get("paused") or self.status.get("waiting_to_start"):
                label = "Paused"
            elif circuit_state == "blocked":
                label = "Circuit blocked"
            elif not self.status.get("foreground"):
                label = "Waiting for MapleStory"
            elif self.status.get("minimap_route_active"):
                label = "Following route from minimap"
            elif self.status.get("vertical_minimap_active"):
                label = "Finishing climb from minimap"
            elif self.status.get("input_blocked"):
                label = "Finding your character"
            elif loot_state in {"settling", "approaching", "dwelling"}:
                label = "Collecting nearby drops"
            else:
                label = "Active"
            self.state_text.set(label)
            def percent(key):
                value = self.status.get(key)
                return "—" if value is None else f"{value:.0f}%"
            fps = self.status.get("fps")
            self.vitals.set(f"HP  {percent('hp')}     MP  {percent('mp')}     Speed  {fps:.1f} fps" if fps is not None else f"HP  {percent('hp')}     MP  {percent('mp')}")
            seen = sorted(set(self.status.get("monsters") or []))
            activity = "Nearby: " + ", ".join(name.replace("_", " ") for name in seen) if seen else "Watching for nearby monsters and drops"
            if protection:
                activity = protection[1]
            elif recovery:
                activity = recovery[1]
            elif circuit_state == "blocked":
                index = self.status.get("circuit_index", self.status.get("circuit_idx"))
                point = f"Point {index + 1}" if isinstance(index, int) else "The current point"
                activity = f"{point} could not be reached. Return to point 1, or edit Circuit and Save & apply."
            elif self.status.get("minimap_route_active"):
                activity = "Following the drawn route while the character is hidden; only movement is enabled"
            elif self.status.get("vertical_minimap_active"):
                activity = "Briefly following the minimap while the character is hidden; attacks are stopped"
            elif loot_state in {"settling", "approaching", "dwelling"}:
                activity = {"settling": "Waiting for nearby drops to settle", "approaching": "Approaching the last combat target's position", "dwelling": "Waiting over drops and pressing pickup"}[loot_state]
            pickup_key = getattr(self, "_saved_pickup_key", "")
            if pickup_key and not protection:
                pickup_count = (self.status.get("keys_sent") or {}).get(pickup_key, 0)
                activity += f"   ·   Pickup key presses: {pickup_count}"
            self.activity.set(activity)
        else:
            pending = time.time() < self.launch_deadline
            if self.launch_process is not None and self.launch_process.poll() not in (None, 0):
                self.launch_deadline = 0
                pending = False
                self.message.set("The launch did not complete. Try Start active and approve the Windows prompt.")
            self.state_text.set("Starting…" if pending else "Stopped")
            self.vitals.set("HP  —     MP  —     Speed  —")
            self.activity.set("Waiting for administrator approval or the game window" if pending else "Choose Start active when your character is ready")
            if self._health_alert_display:
                self.state_text.set(self._health_alert_display[0])
                self.activity.set("Last reported: " + self._health_alert_display[1] + " The controller is not reporting an active session.")
        editing = self._editing_map()
        resume_held = running and _can_resume_manual_hold(self.status)
        self.start_button.configure(text="Resume farming" if resume_held else "Start active",
            state="normal" if not editing and (resume_held or (not running and time.time() >= self.launch_deadline)) else "disabled")
        return_pending = (LOG / "return-to-start-active").exists()
        hold_pending = (LOG / "hold-farming-active").exists()
        self.pause_button.configure(state="normal" if running and not editing else "disabled", text=_pause_label(self.status, return_pending, hold_pending))
        self.stop_button.configure(state="normal" if running else "disabled")
        health_hold = self.status.get("health_protection_active") or (self.status.get("paused") and self.status.get("full_pause") is False)
        can_hold = (running and not editing and self.status.get("map_ready") and self.status.get("minimap_seen")
                    and getattr(self, "_saved_health_enabled", True) and getattr(self, "_saved_hp_key", "")
                    and not self.status.get("editor_test") and not self.status.get("potion_test")
                    and not health_hold and not hold_pending)
        self.hold_button.configure(state="normal" if can_hold else "disabled")
        can_return = (running and not editing and self.status.get("map_ready") and self.status.get("minimap_seen")
                      and self.status.get("circuit_enabled", getattr(self, "_saved_circuit_enabled", False))
                      and not self.status.get("editor_test") and not self.status.get("returning_to_start")
                      and not self.status.get("recovery_terminal")
                      and not hold_pending
                      and not (LOG / "return-to-start-active").exists())
        self.return_button.configure(state="normal" if can_return else "disabled")
        self.save_button.configure(state="disabled" if editing else "normal")
        self.after(700, self._poll)


def main():
    os.chdir(ROOT)
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    SettingsWindow().mainloop()


if __name__ == "__main__":
    main()
