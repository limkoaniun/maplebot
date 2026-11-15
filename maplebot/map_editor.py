"""Map geometry and route editor using original minimap pixel coordinates."""
from __future__ import annotations

from copy import deepcopy
import math
import tkinter as tk
from tkinter import messagebox, ttk
import uuid

import cv2

from maplebot.nav.editor import compile_plan, platform_vertices, platform_y, snap_to_platform, validate_plan
from maplebot.map_profiles import validate_map_settings


class MinimapCropDialog(tk.Toplevel):
    """Draw an exact crop on a frozen, full-resolution prepared game frame."""
    def __init__(self, parent, frame, on_use):
        super().__init__(parent)
        self.title("Calibrate this map's minimap")
        self.geometry("1100x740")
        self.minsize(800, 540)
        self.frame, self.on_use = frame.copy(), on_use
        self.rect = self.start = None
        self.photo = None
        shell = ttk.Frame(self, padding=16)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        ttk.Label(shell, text="Drag around the minimap's map content. Exclude its title, buttons, and border.\nThis creates a new map setup and disables the previous route.", wraplength=1050).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.canvas = tk.Canvas(shell, background="#101b2c", highlightthickness=0, width=1, height=1, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self._draw())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.readout = tk.StringVar(value="No crop selected")
        controls = ttk.Frame(shell)
        controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(controls, textvariable=self.readout).pack(side="left")
        ttk.Button(controls, text="Use this crop", style="Accent.TButton", command=self._use).pack(side="right")
        ttk.Button(controls, text="Cancel", command=self.destroy).pack(side="right", padx=8)

    def _close(self):
        self.destroy()

    def _point(self, event):
        h, w = self.frame.shape[:2]
        return (min(w, max(0, round((event.x-self.ox)/self.scale))),
                min(h, max(0, round((event.y-self.oy)/self.scale))))

    def _draw(self):
        h, w = self.frame.shape[:2]
        vw, vh = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min(vw/w, vh/h)
        self.ox, self.oy = (vw-w*self.scale)/2, (vh-h*self.scale)/2
        scaled = cv2.resize(self.frame, (max(1, round(w*self.scale)), max(1, round(h*self.scale))), interpolation=cv2.INTER_NEAREST)
        ok, encoded = cv2.imencode(".png", scaled)
        if ok:
            self.photo = tk.PhotoImage(data=encoded.tobytes())
        self.canvas.delete("all")
        self.canvas.create_image(self.ox, self.oy, image=self.photo, anchor="nw")
        if self.rect:
            x, y, width, height = self.rect
            self.canvas.create_rectangle(self.ox+x*self.scale, self.oy+y*self.scale,
                                          self.ox+(x+width)*self.scale, self.oy+(y+height)*self.scale,
                                          outline="#54e9c3", width=3)

    def _press(self, event):
        self.start = self._point(event)

    def _drag(self, event):
        if self.start is None:
            return
        x, y = self._point(event)
        sx, sy = self.start
        self.rect = min(x, sx), min(y, sy), abs(x-sx), abs(y-sy)
        self.readout.set("X %d · Y %d · Width %d · Height %d" % self.rect)
        self._draw()

    def _use(self):
        if self.rect is None or min(self.rect[2:]) < 8:
            messagebox.showerror("Select a minimap", "Draw a crop at least 8 pixels wide and high.", parent=self)
            return
        try:
            self.on_use(self.rect)
            self.destroy()
        except Exception as error:
            messagebox.showerror("Could not calibrate map", str(error), parent=self)


COLORS = {"platform": "#57d6af", "rope": "#ffd477", "jump": "#bba4ff",
          "slope": "#57d6af", "drop": "#ff9c85", "ramp": "#e9b9ed", "route": "#72baff", "selected": "#ffffff"}
HELP = {
    "select": "Click an object to select it. Drag its handles to adjust it. Drag empty space with the middle button to pan.",
    "platform": "Drag a floor line at the yellow dot's walking height, not the grass edge. Keep both ends inside the actual edges.",
    "slope": "Drag between the two ends of a sloping floor at the yellow dot's walking height. Use Jump to connect a gap onto it.",
    "rope": "Drag from one platform to another along a rope or ladder. The connection works in both directions.",
    "jump": "Drag from the takeoff platform to the landing platform. The arrow shows the allowed jump direction.",
    "drop": "Drag from an upper platform to the floor below. Drops are vertical and go down only.",
    "ramp": "Drag along stairs or a sloping walkway from its start to its end. This connection walks continuously instead of jumping.",
    "route": "Click a platform to add a numbered farming stop. Stops run in order; connectors join different floors.",
}


class MapEditor(tk.Toplevel):
    def __init__(self, parent, plan, image_bgr, on_save, on_test, on_stop,
                 player_pos=None, *, bridge=None):
        super().__init__(parent)
        self.title("Maplebot · Map editor")
        self.geometry("1180x820")
        self.minsize(1000, 720)
        self.configure(background="#eef3f9")
        self.plan = deepcopy(plan)
        self.image_bgr = image_bgr.copy()
        self.player_pos = player_pos
        self.on_save, self.on_test, self.on_stop = on_save, on_test, on_stop
        self.bridge = bridge
        self.map_settings = bridge.settings_for_plan(plan) if bridge is not None else None
        self.selected = None
        self.undo_stack, self.redo_stack = [], []
        self.saved_plan = deepcopy(plan)
        self.saved_map_settings = deepcopy(self.map_settings)
        self.zoom = 1.0
        self.pan = [0., 0.]
        self.scale, self.origin = 1.0, (0., 0.)
        self.drag = None
        self.photo = None
        self.render_key = None
        self.test_requested = False
        self.preview_points = None
        self.tool = tk.StringVar(value="select")
        self.name = tk.StringVar(value=plan.get("name", "Map route"))
        self.profile_name = tk.StringVar()
        self.loop = tk.BooleanVar(value=bool(plan.get("loop", True)))
        self.status = tk.StringVar(value="The bot is paused. Draw, preview, then test a connection before using this route.")
        self.help_text = tk.StringVar(value=HELP["select"])
        self.selection_text = tk.StringVar(value="No object selected")
        self.context_warning = tk.StringVar(value=str(plan.get("context_warning") or ""))
        self.coordinates = tk.StringVar(value="Original minimap coordinates")
        self.dwell = tk.StringVar(value="0.3")
        self._build()
        self._refresh_profiles()
        self._refresh()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Control-z>", lambda event: self._undo())
        self.bind("<Control-y>", lambda event: self._redo())
        self.bind("<Delete>", self._delete_key)
        self.bind("<Escape>", lambda event: self._stop_test())
        self.name.trace_add("write", lambda *args: self._update_title())
        self.after(400, self._poll_test)

    def _build(self):
        shell = ttk.Frame(self, padding=16)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(5, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Map editor", font=("Segoe UI", 18, "bold")).pack(side="left")
        ttk.Label(header, text="Editing pauses the bot", foreground="#94620e").pack(side="right")
        profiles = ttk.Frame(shell)
        profiles.grid(row=1, column=0, sticky="ew", pady=(12, 8))
        ttk.Label(profiles, text="Map profile").pack(side="left")
        ttk.Entry(profiles, textvariable=self.name, width=27).pack(side="left", padx=(8, 16))
        self.profiles = ttk.Combobox(profiles, textvariable=self.profile_name, width=27, state="readonly")
        self.profiles.pack(side="left")
        ttk.Button(profiles, text="Load", command=self._load_profile).pack(side="left", padx=6)
        ttk.Button(profiles, text="New map", command=self._new_map).pack(side="left")
        ttk.Button(profiles, text="Save draft", command=lambda: self._save(False)).pack(side="right")
        toolbar = ttk.Frame(shell)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        drawing_tools = ttk.Frame(toolbar)
        drawing_tools.pack(fill="x")
        for value, label in (("select", "Select / edit"), ("platform", "Platform / boundary"), ("slope", "Slope boundary"), ("rope", "Rope / ladder"),
                             ("jump", "Jump"), ("drop", "Drop"), ("ramp", "Ramp / stairs"), ("route", "Route stops")):
            ttk.Radiobutton(drawing_tools, text=label, value=value, variable=self.tool,
                            command=self._tool_changed).pack(side="left", padx=(0, 12))
        map_tools = ttk.Frame(toolbar)
        map_tools.pack(fill="x", pady=(5, 0))
        ttk.Button(map_tools, text="Minimap crop", command=self._calibrate).pack(side="left")
        ttk.Button(map_tools, text="Map settings", command=self._map_settings_dialog).pack(side="left", padx=5)
        ttk.Label(map_tools, text="Calibrate the crop and monsters for each map.", style="Muted.TLabel").pack(side="left", padx=8)
        info = ttk.Label(shell, textvariable=self.help_text, wraplength=1100, foreground="#52637a")
        info.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.context_banner = tk.Label(shell, textvariable=self.context_warning, wraplength=1080,
                                       anchor="w", justify="left", background="#fff0cf",
                                       foreground="#87450b", font=("Segoe UI", 10), padx=10, pady=7)
        if self.context_warning.get():
            self.context_banner.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        area = ttk.Frame(shell)
        self.drawing_area = area
        area.grid(row=5, column=0, sticky="nsew")
        area.columnconfigure(0, weight=1)
        area.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(area, background="#101b2c", width=1, height=1,
                                highlightthickness=1, highlightbackground="#b9c8da")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self._draw())
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind("<ButtonRelease-1>", self._release)
        self.canvas.bind("<Motion>", self._pointer)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<ButtonPress-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        self.sidebar_canvas = tk.Canvas(area, background="#eef3f9", width=270, height=1, highlightthickness=0)
        self.sidebar_canvas.grid(row=0, column=1, sticky="ns", padx=(14, 0))
        sidebar_scroll = ttk.Scrollbar(area, orient="vertical", command=self.sidebar_canvas.yview)
        sidebar_scroll.grid(row=0, column=2, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)
        sidebar = ttk.Frame(self.sidebar_canvas, padding=(0, 0, 8, 0))
        sidebar_window = self.sidebar_canvas.create_window(0, 0, anchor="nw", window=sidebar)
        sidebar.bind("<Configure>", lambda event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda event: self.sidebar_canvas.itemconfigure(sidebar_window, width=event.width))
        self.sidebar_canvas.bind("<MouseWheel>", self._scroll_sidebar)
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(1, weight=1)
        ttk.Label(sidebar, text="Map objects", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.objects = ttk.Treeview(sidebar, show="tree", selectmode="browse", height=6)
        self.objects.column("#0", width=215, minwidth=180, stretch=True)
        self.objects.grid(row=1, column=0, sticky="nsew")
        self.objects.bind("<<TreeviewSelect>>", self._list_selected)
        self.objects.bind("<Delete>", self._delete_key)
        self.tree_items = {}
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.objects.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        self.objects.configure(yscrollcommand=scroll.set)
        ttk.Label(sidebar, textvariable=self.selection_text, wraplength=220).grid(row=2, column=0, sticky="ew", pady=(10, 6))
        dwell = ttk.Frame(sidebar)
        self.dwell_frame = dwell
        dwell.grid(row=3, column=0, sticky="ew")
        ttk.Label(dwell, text="Stop wait (s)").pack(side="left")
        self.dwell_entry = ttk.Entry(dwell, textvariable=self.dwell, width=6)
        self.dwell_entry.pack(side="left", padx=6)
        self.dwell_button = ttk.Button(dwell, text="Set", command=self._set_dwell, width=4)
        self.dwell_button.pack(side="left")
        order = ttk.Frame(sidebar)
        self.order_frame = order
        order.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(order, text="Move earlier", command=lambda: self._reorder(-1)).pack(side="left")
        ttk.Button(order, text="Move later", command=lambda: self._reorder(1)).pack(side="left", padx=(5, 0))
        ttk.Checkbutton(sidebar, text="Repeat route", variable=self.loop, command=self._loop_changed).grid(row=5, column=0, sticky="w", pady=6)
        self.test_button = ttk.Button(sidebar, text="Test selected connection", command=self._test)
        self.test_button.grid(row=6, column=0, sticky="ew", pady=(4, 5))
        ttk.Button(sidebar, text="Stop test  ·  Esc", command=self._stop_test).grid(row=7, column=0, sticky="ew")
        ttk.Label(sidebar, text="Test with MapleStory in front.\nCombat and pickup stay off.", style="Muted.TLabel", wraplength=220).grid(row=8, column=0, sticky="w", pady=(8, 0))
        route_actions = ttk.Frame(sidebar)
        route_actions.grid(row=9, column=0, sticky="ew", pady=(8, 0))
        self.player_stop_button = ttk.Button(route_actions, text="Add at player", command=self._add_player_stop,
                   state="normal" if self.player_pos is not None else "disabled")
        self.player_stop_button.pack(side="left")
        ttk.Button(route_actions, text="Clear stops", command=self._clear_route).pack(side="left", padx=(5, 0))
        nav = ttk.Frame(shell)
        nav.grid(row=6, column=0, sticky="ew", pady=(8, 0))
        self.undo_button = ttk.Button(nav, text="Undo", command=self._undo)
        self.undo_button.pack(side="left")
        self.redo_button = ttk.Button(nav, text="Redo", command=self._redo)
        self.redo_button.pack(side="left", padx=5)
        ttk.Button(nav, text="Delete selected", command=self._delete).pack(side="left", padx=(0, 12))
        ttk.Button(nav, text="−", width=3, command=lambda: self._zoom(.8)).pack(side="left")
        ttk.Button(nav, text="Fit map", command=self._fit).pack(side="left", padx=4)
        ttk.Button(nav, text="+", width=3, command=lambda: self._zoom(1.25)).pack(side="left")
        ttk.Label(nav, textvariable=self.coordinates, style="Muted.TLabel").pack(side="right")
        legend = ttk.Frame(shell)
        legend.grid(row=7, column=0, sticky="ew", pady=(8, 0))
        for kind, label in (("platform", "Floor boundaries"), ("rope", "Rope / ladder"), ("jump", "Jump"),
                            ("drop", "Drop"), ("route", "Ordered route")):
            tk.Label(legend, text="━", foreground=COLORS[kind], background="#eef3f9", font=("Segoe UI", 14, "bold")).pack(side="left")
            ttk.Label(legend, text=label).pack(side="left", padx=(2, 14))
        status_label = ttk.Label(shell, textvariable=self.status, wraplength=1120, foreground="#334d70")
        status_label.grid(row=8, column=0, sticky="ew", pady=(9, 8))
        footer = ttk.Frame(shell)
        footer.grid(row=9, column=0, sticky="ew")
        ttk.Button(footer, text="Close editor", command=self._close).pack(side="left")
        ttk.Label(footer, text="The bot stays paused after saving or closing.", style="Muted.TLabel").pack(side="left", padx=12)
        ttk.Button(footer, text="Use route", style="Accent.TButton", command=lambda: self._save(True)).pack(side="right")
        ttk.Button(footer, text="Preview route", command=self._preview).pack(side="right", padx=8)
        shell.bind("<Configure>", lambda event: (
            info.configure(wraplength=max(300, event.width-32)),
            status_label.configure(wraplength=max(300, event.width-32)),
            self.context_banner.configure(wraplength=max(300, event.width-52))))
        self._bind_sidebar_scroll(sidebar)

    def _scroll_sidebar(self, event):
        self.sidebar_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        return "break"

    def _bind_sidebar_scroll(self, widget):
        # Keep the object's own scrollbar while allowing the inspector beneath
        # it to remain accessible at the smallest supported window size.
        if not isinstance(widget, (ttk.Treeview, ttk.Scrollbar)):
            widget.bind("<MouseWheel>", self._scroll_sidebar)
        for child in widget.winfo_children():
            self._bind_sidebar_scroll(child)

    def _current_plan(self):
        result = deepcopy(self.plan)
        result["name"] = self.name.get().strip()
        result["loop"] = self.loop.get()
        return result

    def _update_title(self):
        dirty = self._dirty()
        self.title(f"Maplebot · Map editor{' · unsaved' if dirty else ''}")

    def _dirty(self):
        return self._current_plan() != self.saved_plan or self.map_settings != self.saved_map_settings

    def _checkpoint(self):
        if self.test_requested:
            self._stop_test(quiet=True)
        self.preview_points = None
        self.undo_stack.append(self._current_plan())
        del self.undo_stack[:-80]
        self.redo_stack.clear()

    def _new_map(self):
        self._checkpoint()
        self.plan.update(name="New map", platforms=[], connections=[], route=[], loop=True)
        self.name.set("New map")
        self.loop.set(True)
        self.selected = None
        self._refresh()
        self.status.set("Blank drawing on this map snapshot. Name the profile, then mark platforms at the yellow dot's walking height. Undo restores the previous drawing.")

    def _clear_route(self):
        if not self.plan["route"]:
            return
        self._checkpoint()
        self.plan["route"] = []
        self.selected = None
        self._refresh()
        self.status.set("Route stops cleared. Platforms and connections are kept; use Route stops to draw a new circuit. Undo restores the stops.")

    def _restore(self, plan):
        if self.test_requested:
            self._stop_test(quiet=True)
        self.preview_points = None
        self.plan = deepcopy(plan)
        self.name.set(plan.get("name", "Map route"))
        self.loop.set(bool(plan.get("loop", True)))
        self.selected = None
        self._refresh()

    def _undo(self):
        if self.undo_stack:
            self.redo_stack.append(self._current_plan())
            self._restore(self.undo_stack.pop())

    def _redo(self):
        if self.redo_stack:
            self.undo_stack.append(self._current_plan())
            self._restore(self.redo_stack.pop())

    def _platform(self, identifier):
        return next((p for p in self.plan["platforms"] if p["id"] == identifier), None)

    def _connection(self, identifier):
        return next((p for p in self.plan["connections"] if p["id"] == identifier), None)

    def _point(self, key):
        kind, identifier = key
        if kind == "route" and 0 <= identifier < len(self.plan["route"]):
            point = self.plan["route"][identifier]
            platform = self._platform(point["platform_id"])
            return (point["x"], platform_y(platform, point["x"])) if platform else None
        return None

    def _transform(self):
        width, height = self.plan["image_size"]
        view_w, view_h = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        fit = min(max(1, view_w - 56) / width, max(1, view_h - 56) / height)
        self.scale = fit * self.zoom
        self.origin = ((view_w - width * self.scale) / 2 + self.pan[0],
                       (view_h - height * self.scale) / 2 + self.pan[1])

    def _screen(self, x, y):
        return self.origin[0] + x * self.scale, self.origin[1] + y * self.scale

    def _image(self, x, y):
        width, height = self.plan["image_size"]
        return (round(min(width - 1, max(0., (x - self.origin[0]) / self.scale)), 1),
                round(min(height - 1, max(0., (y - self.origin[1]) / self.scale)), 1))

    def _draw(self):
        if not self.winfo_exists():
            return
        self._transform()
        canvas = self.canvas
        canvas.delete("all")
        width, height = self.plan["image_size"]
        render_size = max(1, round(width * self.scale)), max(1, round(height * self.scale))
        if render_size != self.render_key:
            scaled = cv2.resize(self.image_bgr, render_size, interpolation=cv2.INTER_NEAREST)
            okay, png = cv2.imencode(".png", scaled)
            if okay:
                self.photo = tk.PhotoImage(data=png.tobytes())
                self.render_key = render_size
        if self.photo is not None:
            canvas.create_image(*self.origin, image=self.photo, anchor="nw")
        # Dark lines beneath the colors keep geometry readable over bright maps.
        for p in self.plan["platforms"]:
            vertices = [self._screen(x, y) for x, y in platform_vertices(p)]
            coordinates = [value for point in vertices for value in point]
            selected = self.selected == ("platform", p["id"])
            canvas.create_line(*coordinates, fill="#102c2a", width=9)
            canvas.create_line(*coordinates, fill=COLORS["platform"], width=4)
            for x, y in (vertices[0], vertices[-1]):
                canvas.create_line(x, y - 7, x, y + 7, fill=COLORS["platform"], width=3)
            if selected:
                for x, y in vertices:
                    self._handle(x, y)
        for connection in self.plan["connections"]:
            first, second = self._platform(connection["from_platform"]), self._platform(connection["to_platform"])
            if not first or not second:
                continue
            start = self._screen(connection["from_x"], platform_y(first, connection["from_x"]))
            end = self._screen(connection["to_x"], platform_y(second, connection["to_x"]))
            kind = connection["kind"]
            canvas.create_line(*start, *end, fill="#101b2c", width=8)
            canvas.create_line(*start, *end, fill=COLORS[kind], width=3,
                               arrow="both" if kind in {"rope", "ramp"} else "last", arrowshape=(11, 13, 5),
                               dash=(6, 4) if kind == "jump" else ())
            if self.selected == ("connection", connection["id"]):
                for point in (start, end):
                    self._handle(*point)
        points = [self._point(("route", i)) for i in range(len(self.plan["route"]))]
        path_points = self.preview_points if self.preview_points is not None else points
        path = list(zip(path_points, path_points[1:]))
        if self.preview_points is None and self.loop.get() and len(points) > 1:
            path.append((points[-1], points[0]))
        for start, end in path:
            if start and end and start != end:
                canvas.create_line(*self._screen(*start), *self._screen(*end), fill=COLORS["route"],
                                   width=3 if self.preview_points is not None else 2,
                                   dash=() if self.preview_points is not None else (4, 4),
                                   arrow="last", arrowshape=(9, 11, 4))
        for i, point in enumerate(points):
            if point:
                x, y = self._screen(*point)
                selected = self.selected == ("route", i)
                radius = 12 if selected else 10
                canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="#154f88",
                                   outline="white" if selected else COLORS["route"], width=2)
                canvas.create_text(x, y, text=str(i+1), fill="white", font=("Segoe UI", 9, "bold"))
        if self.player_pos is not None:
            x, y = self._screen(*self.player_pos)
            canvas.create_oval(x-5, y-5, x+5, y+5, fill="#fff36a", outline="#17202d", width=2)
            canvas.create_text(x+9, y-9, text="You · snapshot", anchor="sw", fill="#fff7ae", font=("Segoe UI", 9))
        if self.drag and self.drag.get("kind") == "draw":
            start, end = self.drag["start"], self.drag["end"]
            kind = self.tool.get()
            if kind == "platform":
                end = (end[0], start[1])
            elif kind in {"rope", "drop"}:
                end = (start[0], end[1])
            canvas.create_line(*self._screen(*start), *self._screen(*end), fill=COLORS.get(kind, "white"),
                               width=3, dash=(5, 3), arrow="last" if kind not in {"platform", "slope"} else "none")

    def _handle(self, x, y):
        self.canvas.create_rectangle(x-5, y-5, x+5, y+5, fill="white", outline="#16364e", width=2)

    @staticmethod
    def _segment_distance(point, start, end):
        dx, dy = end[0]-start[0], end[1]-start[1]
        length = dx*dx+dy*dy
        t = min(1, max(0, ((point[0]-start[0])*dx+(point[1]-start[1])*dy)/length)) if length else 0
        return math.hypot(point[0]-(start[0]+t*dx), point[1]-(start[1]+t*dy))

    def _hit(self, x, y):
        if self.selected:
            kind, identifier = self.selected
            handles = []
            if kind == "platform":
                p = self._platform(identifier)
                if p:
                    handles = [("left", self._screen(p["left"], p["y"])),
                               ("right", self._screen(p["right"], p.get("right_y", p["y"])))]
                    handles.extend((f"knot:{i}", self._screen(*point)) for i, point in enumerate(p.get("knots", [])))
            elif kind == "connection":
                c = self._connection(identifier)
                if c:
                    for endpoint in ("from", "to"):
                        p = self._platform(c[endpoint+"_platform"])
                        if p:
                            handles.append((endpoint, self._screen(c[endpoint+"_x"], platform_y(p, c[endpoint+"_x"]))))
            for handle, point in handles:
                if math.hypot(x-point[0], y-point[1]) <= 10:
                    return self.selected, handle
        candidates = []
        for i in reversed(range(len(self.plan["route"]))):
            point = self._point(("route", i))
            if point:
                screen = self._screen(*point)
                distance = math.hypot(x-screen[0], y-screen[1])
                if distance <= 13:
                    candidates.append((distance - 6, ("route", i), "point"))
        for c in self.plan["connections"]:
            first, second = self._platform(c["from_platform"]), self._platform(c["to_platform"])
            if first and second:
                a, b = self._screen(c["from_x"], platform_y(first, c["from_x"])), self._screen(c["to_x"], platform_y(second, c["to_x"]))
                distance = self._segment_distance((x, y), a, b)
                if distance <= 9:
                    handle = "from" if math.hypot(x-a[0], y-a[1]) <= 13 else "to" if math.hypot(x-b[0], y-b[1]) <= 13 else "body"
                    candidates.append((distance, ("connection", c["id"]), handle))
        for p in self.plan["platforms"]:
            vertices = [self._screen(px, py) for px, py in platform_vertices(p)]
            a, b = vertices[0], vertices[-1]
            distance = min(self._segment_distance((x, y), start, end) for start, end in zip(vertices, vertices[1:]))
            if distance <= 9:
                handle = "left" if math.hypot(x-a[0], y-a[1]) <= 13 else "right" if math.hypot(x-b[0], y-b[1]) <= 13 else "body"
                candidates.append((distance + 1, ("platform", p["id"]), handle))
        return min(candidates, key=lambda item: item[0])[1:] if candidates else (None, None)

    def _press(self, event):
        self.canvas.focus_set()
        point = self._image(event.x, event.y)
        tool = self.tool.get()
        if tool == "select":
            self.selected, handle = self._hit(event.x, event.y)
            self.drag = {"kind": "edit", "handle": handle, "start": point, "before": deepcopy(self.plan), "changed": False} if self.selected else None
            self._refresh()
        elif tool == "route":
            snapped = snap_to_platform(self.plan, *point, max_distance=8)
            if snapped is None:
                self.status.set("Add a platform first, then click on it to place a route stop.")
                return
            self._checkpoint()
            self.plan["route"].append({"platform_id": snapped[0], "x": round(snapped[1], 1), "dwell_s": .3})
            self.selected = ("route", len(self.plan["route"])-1)
            self._refresh()
        else:
            self.drag = {"kind": "draw", "start": point, "end": point}

    def _motion(self, event):
        if not self.drag:
            return
        point = self._image(event.x, event.y)
        if self.drag["kind"] == "draw":
            self.drag["end"] = point
        elif self.selected:
            if not self.drag["changed"]:
                self._checkpoint()
                self.drag["changed"] = True
            kind, identifier = self.selected
            handle = self.drag["handle"]
            if kind == "platform":
                platform = self._platform(identifier)
                original = next(p for p in self.drag["before"]["platforms"] if p["id"] == identifier)
                if handle in {"left", "right"}:
                    knots = platform.get("knots", [])
                    if knots and (point[0] >= knots[0][0] if handle == "left" else point[0] <= knots[-1][0]):
                        return  # endpoint handles cannot cross interior surface vertices
                    other = "right" if handle == "left" else "left"
                    if abs(point[0]-platform[other]) < .1:
                        return  # do not create a zero-width intermediate segment
                    if knots:
                        platform.setdefault("right_y", platform["y"])
                    platform[handle] = point[0]
                    platform["right_y" if "right_y" in platform and handle == "right" else "y"] = point[1]
                    if platform["left"] > platform["right"]:
                        platform["left"], platform["right"] = platform["right"], platform["left"]
                        if "right_y" in platform:
                            platform["y"], platform["right_y"] = platform["right_y"], platform["y"]
                        self.drag["handle"] = "right" if handle == "left" else "left"
                elif handle and handle.startswith("knot:"):
                    index = int(handle.split(":", 1)[1])
                    vertices = platform_vertices(platform)
                    if not vertices[index][0] < point[0] < vertices[index+2][0]:
                        return
                    platform["knots"] = [list(knot) for knot in platform["knots"]]
                    platform["knots"][index] = [point[0], point[1]]
                else:
                    delta_x = point[0]-self.drag["start"][0]
                    delta_x = min(self.plan["image_size"][0]-1-original["right"], max(-original["left"], delta_x))
                    original_end_y = original.get("right_y", original["y"])
                    heights = [y for _, y in platform_vertices(original)]
                    delta_y = point[1]-self.drag["start"][1]
                    delta_y = min(self.plan["image_size"][1]-1-max(heights), max(-min(heights), delta_y))
                    platform.update(left=original["left"]+delta_x, right=original["right"]+delta_x,
                                    y=original["y"]+delta_y)
                    if "right_y" in original:
                        platform["right_y"] = original_end_y+delta_y
                    if "knots" in original:
                        platform["knots"] = [[x+delta_x, y+delta_y] for x, y in original["knots"]]
            elif kind == "route":
                snapped = snap_to_platform(self.plan, *point, max_distance=8)
                if snapped:
                    self.plan["route"][identifier].update(platform_id=snapped[0], x=round(snapped[1], 1))
            elif kind == "connection":
                connection = self._connection(identifier)
                snapped = snap_to_platform(self.plan, *point, max_distance=8)
                if handle in {"from", "to"} and snapped:
                    connection[handle+"_platform"] = snapped[0]
                    connection[handle+"_x"] = round(snapped[1], 1)
                    if connection["kind"] in {"rope", "drop"}:
                        connection["from_x"] = connection["to_x"] = round(snapped[1], 1)
                elif handle == "body":
                    original = next(c for c in self.drag["before"]["connections"] if c["id"] == identifier)
                    delta = point[0]-self.drag["start"][0]
                    connection["from_x"] = round(original["from_x"]+delta, 1)
                    connection["to_x"] = round(original["to_x"]+delta, 1)
        self._draw()
        self._pointer(event)

    def _release(self, event):
        if not self.drag:
            return
        drag = self.drag
        self.drag = None
        if drag["kind"] == "draw":
            start, end = drag["start"], self._image(event.x, event.y)
            kind = self.tool.get()
            if kind in {"platform", "slope"}:
                left, right = sorted((start[0], end[0]))
                if right-left < 2:
                    self.status.set("Drag across at least two minimap pixels to make a platform.")
                else:
                    self._checkpoint()
                    identifier = "floor_" + uuid.uuid4().hex[:8]
                    platform = {"id": identifier, "left": left, "right": right, "y": start[1]}
                    if kind == "slope":
                        platform["y"], platform["right_y"] = (start[1], end[1]) if start[0] <= end[0] else (end[1], start[1])
                    self.plan["platforms"].append(platform)
                    self.selected = ("platform", identifier)
            else:
                if kind in {"rope", "drop"}:
                    end = start[0], end[1]
                first = snap_to_platform(self.plan, *start, max_distance=8)
                second = snap_to_platform(self.plan, *end, max_distance=8)
                if first is None or second is None or first[0] == second[0]:
                    self.status.set("Start and finish on two different platforms. Draw the platforms first.")
                elif kind == "drop" and second[2] <= first[2]:
                    self.status.set("Draw a drop from the upper platform down to the lower platform.")
                else:
                    self._checkpoint()
                    identifier = kind + "_" + uuid.uuid4().hex[:8]
                    self.plan["connections"].append({"id": identifier, "kind": kind, "from_platform": first[0],
                                                      "to_platform": second[0], "from_x": round(first[1], 1),
                                                      "to_x": round(first[1] if kind in {"rope", "drop"} else second[1], 1)})
                    self.selected = ("connection", identifier)
        self._refresh()

    def _pointer(self, event):
        x, y = self._image(event.x, event.y)
        self.coordinates.set(f"X {x:g}  ·  Y {y:g}  ·  {self.zoom:.1f}× fit")

    def _pan_start(self, event):
        self.pan_anchor = (event.x, event.y, *self.pan)

    def _pan_move(self, event):
        x, y, px, py = self.pan_anchor
        self.pan = [px+event.x-x, py+event.y-y]
        self._draw()

    def _zoom(self, factor):
        self.zoom = min(4, max(.5, self.zoom*factor))
        self._draw()

    def _wheel(self, event):
        self._zoom(1.12 if event.delta > 0 else 1/1.12)

    def _fit(self):
        self.zoom, self.pan = 1., [0., 0.]
        self._draw()

    def _tool_changed(self):
        self.drag = None
        self.help_text.set(HELP[self.tool.get()])
        self.canvas.configure(cursor="arrow" if self.tool.get() == "select" else "crosshair")
        self._draw()

    def _refresh(self):
        selected_item = None
        self.objects.delete(*self.objects.get_children())
        self.tree_items.clear()
        for kind, label in (("platform", "Platforms"), ("connection", "Connections"), ("route", "Route stops")):
            group = self.objects.insert("", "end", text=label, open=True)
            items = self.plan[{"platform": "platforms", "connection": "connections", "route": "route"}[kind]]
            for i, item in enumerate(items):
                identifier = i if kind == "route" else item["id"]
                if kind == "platform":
                    text = (f"Shaped floor {i+1} · {len(item['knots'])+1} segments" if item.get("knots") else
                            f"Slope {i+1} · Y {item['y']:g} → {item['right_y']:g}" if "right_y" in item
                            else f"Floor {i+1} · Y {item['y']:g}")
                elif kind == "connection":
                    text = f"{item['kind'].capitalize()} {i+1}"
                else:
                    text = f"Stop {i+1} · X {item['x']:g}"
                tree_id = self.objects.insert(group, "end", text=text)
                self.tree_items[tree_id] = (kind, identifier)
                if self.selected == (kind, identifier):
                    selected_item = tree_id
        if selected_item:
            self.objects.selection_set(selected_item)
            self.objects.see(selected_item)
        self.undo_button.configure(state="normal" if self.undo_stack else "disabled")
        self.redo_button.configure(state="normal" if self.redo_stack else "disabled")
        self._selection_changed()
        self._update_title()
        self._draw()

    def _list_selected(self, event=None):
        selection = self.objects.selection()
        if selection:
            key = self.tree_items.get(selection[0])
            if key is not None and key != self.selected:
                self.selected = key
                self._selection_changed()
                self._draw()

    def _selection_changed(self):
        selected = self.selected
        text = "No object selected"
        if selected:
            kind, identifier = selected
            if kind == "platform":
                p = self._platform(identifier)
                if p:
                    text = f"Platform · X {p['left']:g} to {p['right']:g}\nHeight Y {p['y']:g}" + (f" → {p['right_y']:g}" if "right_y" in p else "")
                    if p.get("knots"):
                        text += f"\n{len(p['knots'])} interior handles · drag to adjust the surface."
            elif kind == "connection":
                c = self._connection(identifier)
                if c:
                    text = f"{c['kind'].capitalize()} connection\nDrag either end to adjust its attachment."
            elif kind == "route" and identifier < len(self.plan["route"]):
                p = self.plan["route"][identifier]
                self.dwell.set(str(p.get("dwell_s", .3)))
                text = f"Route stop {identifier+1}\nDrag to move; use Earlier / Later to reorder."
        self.selection_text.set(text)
        self.dwell_entry.configure(state="normal" if selected and selected[0] == "route" else "disabled")
        self.dwell_button.configure(state="normal" if selected and selected[0] == "route" else "disabled")
        if selected and selected[0] == "route":
            self.dwell_frame.grid()
            self.order_frame.grid()
        else:
            self.dwell_frame.grid_remove()
            self.order_frame.grid_remove()
        if selected and selected[0] == "connection":
            self.test_button.grid()
        else:
            self.test_button.grid_remove()
        self.test_button.configure(state="normal" if selected and selected[0] == "connection" else "disabled")

    def _delete_key(self, event):
        if not isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text)):
            self._delete()
            return "break"

    def _delete(self):
        if not self.selected:
            return
        kind, identifier = self.selected
        self._checkpoint()
        if kind == "route":
            self.plan["route"].pop(identifier)
        elif kind == "connection":
            self.plan["connections"] = [c for c in self.plan["connections"] if c["id"] != identifier]
        else:
            self.plan["platforms"] = [p for p in self.plan["platforms"] if p["id"] != identifier]
            self.plan["connections"] = [c for c in self.plan["connections"] if identifier not in (c["from_platform"], c["to_platform"])]
            self.plan["route"] = [p for p in self.plan["route"] if p["platform_id"] != identifier]
            self.status.set("Removed the platform and its attached connections and route stops. Undo restores them.")
        self.selected = None
        self._refresh()

    def _set_dwell(self):
        if not self.selected or self.selected[0] != "route":
            return
        try:
            dwell = float(self.dwell.get())
            if not math.isfinite(dwell) or not 0 <= dwell <= 10:
                raise ValueError
        except ValueError:
            messagebox.showerror("Check the wait time", "Use a wait from 0 to 10 seconds.", parent=self)
            return
        self._checkpoint()
        self.plan["route"][self.selected[1]]["dwell_s"] = dwell
        self._refresh()

    def _reorder(self, delta):
        if not self.selected or self.selected[0] != "route":
            return
        old = self.selected[1]
        new = old+delta
        if 0 <= new < len(self.plan["route"]):
            self._checkpoint()
            self.plan["route"][old], self.plan["route"][new] = self.plan["route"][new], self.plan["route"][old]
            self.selected = "route", new
            self._refresh()

    def _loop_changed(self):
        if self.test_requested:
            self._stop_test(quiet=True)
        self.preview_points = None
        previous = deepcopy(self.plan)
        self.undo_stack.append(previous)
        self.redo_stack.clear()
        self.plan["loop"] = self.loop.get()
        self._refresh()

    def _preview(self):
        plan = self._current_plan()
        errors = validate_plan(plan)
        if errors:
            messagebox.showerror("Check your map", "\n".join(errors[:10]), parent=self)
            self.status.set("The map needs changes before it can be used. Your draft can still be saved.")
            return False
        try:
            compiled = compile_plan(plan)
        except ValueError as exc:
            messagebox.showerror("Route is not connected", str(exc), parent=self)
            self.status.set("Connect the platforms between route stops, then preview again.")
            return False
        self.preview_points = [(p["x"], p["y"]) for p in compiled.get("waypoints", [])]
        self._draw()
        self.status.set(f"Preview ready: {len(plan['route'])} stops, {len(compiled.get('waypoints', []))} movement steps. Test connections before using the route.")
        return True

    def _add_player_stop(self):
        if self.player_pos is None:
            return
        snapped = snap_to_platform(self.plan, *self.player_pos, max_distance=4)
        if snapped is None:
            self.status.set("The player snapshot is not on a marked platform. Draw or adjust that platform first.")
            return
        self._checkpoint()
        self.plan["route"].append({"platform_id": snapped[0], "x": round(snapped[1], 1), "dwell_s": .3})
        self.selected = "route", len(self.plan["route"])-1
        self._refresh()

    def _save(self, apply):
        if not self.name.get().strip():
            messagebox.showerror("Name your map", "Enter a name for this map profile.", parent=self)
            return
        if apply and not self._preview():
            return
        try:
            self._stop_test(quiet=True)
            plan = self._current_plan()
            plan.pop("context_warning", None)
            if apply:
                issues = validate_map_settings(self.map_settings, plan.get("capture"), plan["image_size"])
                if issues:
                    raise ValueError("\n".join(issues) + " Open Map settings to finish setup.")
            result = self.on_save(plan, self.image_bgr, apply, self.map_settings)
            self.plan = deepcopy(plan)
            self.saved_plan = deepcopy(plan)
            self.saved_map_settings = deepcopy(self.map_settings)
            self.context_warning.set("")
            self.context_banner.grid_remove()
            self._refresh_profiles()
            self._update_title()
            self.status.set(result or ("Route saved and applied. The bot stays paused." if apply else "Draft saved. The active route has not changed."))
        except Exception as exc:
            messagebox.showerror("Could not save map", str(exc), parent=self)

    def _refresh_profiles(self):
        if self.bridge is None:
            return
        try:
            profiles = self.bridge.list_profiles()
            self.profiles.configure(values=profiles)
            if self.name.get() in profiles:
                self.profile_name.set(self.name.get())
        except Exception:
            self.profiles.configure(values=[])

    def _load_profile(self):
        name = self.profile_name.get()
        if not name or self.bridge is None:
            self.status.set("Choose a saved map profile first.")
            return
        if self._dirty() and not messagebox.askyesno("Load another map?", "Discard this map's unsaved changes and load the selected profile?", parent=self):
            return
        try:
            self._stop_test(quiet=True)
            plan, image, player = self.bridge.load_profile(name)
            self.image_bgr, self.player_pos = image.copy(), player
            self.map_settings = self.bridge.settings_for_plan(plan)
            self.player_stop_button.configure(state="normal" if player is not None else "disabled")
            self.render_key = None
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.saved_plan = deepcopy(plan)
            self.saved_map_settings = deepcopy(self.map_settings)
            self.context_warning.set(str(plan.get("context_warning") or ""))
            if self.context_warning.get():
                self.context_banner.grid(row=4, column=0, sticky="ew", pady=(0, 8))
            else:
                self.context_banner.grid_remove()
            self._restore(plan)
            self._fit()
            self.status.set("Loaded the saved map snapshot. Preview and test it before using the route.")
        except Exception as exc:
            messagebox.showerror("Could not load map", str(exc), parent=self)

    def _test(self):
        if not self.selected or self.selected[0] != "connection":
            self.status.set("Select a rope, jump, or drop to test.")
            return
        try:
            result = self.on_test(self._current_plan(), self.selected[1], self.image_bgr)
            self.test_requested = True
            self.status.set(result or "Switch to MapleStory to test; return here for the result. F12 stops the bot; Escape here stops the test.")
        except Exception as exc:
            messagebox.showerror("Could not test connection", str(exc), parent=self)

    def _calibrate(self):
        if self.bridge is None:
            return
        try:
            self._stop_test(quiet=True)
            frame, metadata = self.bridge.full_snapshot()
            def selected(rect):
                plan, image, player = self.bridge.calibrate_crop(rect, frame, metadata)
                self.image_bgr, self.player_pos = image, player
                self.map_settings = self.bridge.settings_for_plan(plan)
                self.render_key = None
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.saved_plan = deepcopy(plan)
                self.saved_map_settings = deepcopy(self.map_settings)
                self._restore(plan)
                self.context_warning.set(plan.get("context_warning", ""))
                self.context_banner.grid(row=4, column=0, sticky="ew", pady=(0, 8))
                self._fit()
                self.status.set("Crop calibrated. Open Map settings to select monsters, then draw this map's route.")
            MinimapCropDialog(self, frame, selected)
        except Exception as error:
            messagebox.showerror("Could not calibrate minimap", str(error), parent=self)

    def _map_settings_dialog(self):
        if self.bridge is None:
            return
        window = tk.Toplevel(self)
        window.title("Settings for this map")
        window.geometry("560x520")
        window.minsize(470, 440)
        shell = ttk.Frame(window, padding=20)
        shell.pack(fill="both", expand=True)
        ttk.Label(shell, text="Monsters on this map", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(shell, text="These choices belong to this map profile.\nYour character keys, skills, and potion settings stay shared.", style="Muted.TLabel").pack(anchor="w", pady=(6, 12))
        settings = deepcopy(self.map_settings or self.bridge.settings_for_plan(self.plan))
        monsters = settings.setdefault("monsters", {})
        root = self.bridge.root / monsters.get("dir", "monster")
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob("*.png"))) if root.exists() else []
        choices = {}
        checks = ttk.Frame(shell)
        checks.pack(fill="x")
        for index, name in enumerate(names):
            variable = tk.BooleanVar(value=name in monsters.get("names", []))
            choices[name] = variable
            ttk.Checkbutton(checks, text=name.replace("_", " ").title(), variable=variable).grid(row=index//2, column=index%2, sticky="w", padx=(0, 25))
        calibration = ttk.Frame(shell)
        calibration.pack(fill="x", pady=(15, 8))
        ttk.Label(calibration, text="Game pixels per minimap pixel").pack(side="left")
        scale = tk.StringVar(value=str(settings.setdefault("circuit_settings", {}).get("pixels_per_minimap_x", 10)))
        ttk.Entry(calibration, textvariable=scale, width=9).pack(side="right")
        ttk.Label(shell, text="Used to estimate monster and drop positions on the minimap.\nKeep the calibrated value unless movement measurements change.", style="Muted.TLabel", wraplength=500).pack(anchor="w")
        rect = settings.get("minimap", {}).get("fixed_rect", [])
        ttk.Label(shell, text=f"Minimap crop: {rect}\nUse Minimap crop in the editor to change it.", style="Muted.TLabel").pack(anchor="w", pady=12)
        def save():
            try:
                selected = [name for name, variable in choices.items() if variable.get()]
                if not selected:
                    raise ValueError("Select at least one monster for this map.")
                value = float(scale.get())
                if not math.isfinite(value) or not .1 <= value <= 100:
                    raise ValueError("Use a minimap scale from 0.1 to 100.")
                settings["monsters"]["names"] = selected
                settings["circuit_settings"]["pixels_per_minimap_x"] = value
                self.map_settings = settings
                self._update_title()
                self.status.set("Map settings updated in this draft. Save draft or Use route to keep them.")
                window.destroy()
            except ValueError as error:
                messagebox.showerror("Check map settings", str(error), parent=window)
        ttk.Button(shell, text="Keep these map settings", style="Accent.TButton", command=save).pack(side="bottom", anchor="e", pady=(12, 0))

    def _stop_test(self, quiet=False):
        try:
            self.on_stop()
            self.test_requested = False
            if not quiet:
                self.status.set("Test stopped. The bot stays paused.")
        except Exception as exc:
            if not quiet:
                messagebox.showerror("Could not stop test", str(exc), parent=self)

    def _poll_test(self):
        if self.test_requested and self.bridge is not None and hasattr(self.bridge, "test_status"):
            try:
                result = self.bridge.test_status()
                if result and result.get("message"):
                    self.status.set(result["message"])
                if result and result.get("state") in {"completed", "blocked", "stopped", "failed", "cancelled", "error"}:
                    self.test_requested = False
            except Exception:
                pass
        if self.winfo_exists():
            self.after(400, self._poll_test)

    def _close(self):
        if self._dirty() and not messagebox.askyesno("Close map editor?", "Discard unsaved map changes? The bot will stay paused.", parent=self):
            return
        self._stop_test(quiet=True)
        if self.bridge is not None:
            self.bridge.pause()
        self.destroy()
