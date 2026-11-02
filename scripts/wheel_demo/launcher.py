"""Desktop launcher for the wheel demo: pick the town and settings, then drive.

    python scripts/wheel_demo/launcher.py

Starts the CARLA server when it is not running (or restarts it from CARLA_ROOT when the running one
lacks the chosen town), then runs run_demo.py with the chosen settings and comes back when it exits.
Install the desktop icon with scripts/wheel_demo/install_launcher.py.
"""

import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import tkinter as tk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DEMO = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "run_demo.py")
START_CARLA = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "start_carla.sh")
# run_demo.py reads these (relative to REPO_ROOT, where the launcher runs it) on every start
CALIBRATION_PATH = os.path.join(REPO_ROOT, "wheel_calibration.json")
BUTTON_MAP_PATH = os.path.join(REPO_ROOT, "wheel_buttons.json")
CALIBRATION_BACKUP_SUFFIX = ".previous.json"
BUTTON_SETTLE_SECONDS = 1.0
BUTTON_CAPTURE_SECONDS = 4.0
WIZARD_POLL_MS = 50
WIZARD_TEXT_WRAP_PX = 520
DEFAULT_CARLA_ROOT = os.path.expanduser("~/carla-0.9.16")
LOG_DIR = os.path.expanduser("~/.cache/pufferdrive_wheel_demo")
CARLA_HOST = "localhost"
CARLA_PORT = 2000
CARLA_START_TIMEOUT_S = 90.0
CARLA_POLL_INTERVAL_S = 0.5
POLICY_STEP_SECONDS = 0.3
LOG_TAIL_LINES = 12

TOWNS = (
    ("Town01", "Small town, river and bridges. Flat."),
    ("Town02", "Small residential and commercial town. Flat."),
    ("Town03", "Downtown: roundabout, tunnel, hills."),
    ("Town04", "Mountain town on a highway loop."),
    ("Town05", "Grid city, multi-lane, overpasses."),
    ("Town06", "Long multi-lane highways."),
    ("Town07", "Rural, narrow roads, farms."),
    ("Town10HD", "Dense city centre, high detail."),
)
DEFAULTS = {
    "town": "Town10HD",
    "camera": "chase",
    "night": "day",
    "level": "normal",
    "assist": "off",
    "assist_blend": 50,
    "full_steering": "car",
    "driving_style": "off",
    "ffb": True,
    "ffb_gain": 70,
    "gearbox": "automatic",
    "grip": "realistic",
    "input": "wheel",
    "minutes": 20,
    "window": "fullscreen",
}

COLOR_BG = "#14161b"
COLOR_CARD = "#1d2027"
COLOR_CARD_SELECTED = "#173a4d"
COLOR_BORDER = "#2b2f38"
COLOR_TEXT = "#eef0f3"
COLOR_MUTED = "#8b919c"
COLOR_DISABLED = "#4a4f59"
COLOR_ACCENT = "#5ac8ff"
COLOR_SEGMENT_SELECTED = "#2a7fa8"
COLOR_GO = "#3ccf7a"
COLOR_GO_ACTIVE = "#33b56a"
COLOR_ERROR = "#ff6a5a"
FONT_TITLE = ("Helvetica", 22, "bold")
FONT_SUBTITLE = ("Helvetica", 11)
FONT_SECTION = ("Helvetica", 11, "bold")
FONT_BODY = ("Helvetica", 11)
FONT_SMALL = ("Helvetica", 9)
FONT_BUTTON = ("Helvetica", 15, "bold")
TOWN_COLUMNS = 4
PAD = 14
SLIDER_LENGTH_PX = 260
SLIDER_HANDLE_PX = 18
SLIDER_THICKNESS_PX = 12


def carla_is_listening(host=CARLA_HOST, port=CARLA_PORT):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(CARLA_POLL_INTERVAL_S)
        return probe.connect_ex((host, port)) == 0


def carla_available_towns(host=CARLA_HOST, port=CARLA_PORT):
    """Towns the running server can load, or None when it cannot be asked."""
    try:
        import carla

        client = carla.Client(host, port)
        client.set_timeout(5.0)
        return {name.rsplit("/", 1)[-1] for name in client.get_available_maps()}
    except Exception:
        return None


def installed_towns(carla_root):
    maps_dir = os.path.join(carla_root, "CarlaUE4", "Content", "Carla", "Maps")
    if not os.path.isdir(maps_dir):
        return set()
    return {name[: -len(".umap")] for name in os.listdir(maps_dir) if name.endswith(".umap")}


def demo_command(settings):
    """run_demo.py arguments for the chosen settings."""
    command = [
        sys.executable,
        RUN_DEMO,
        "--town", settings["town"],
        "--render",
        "--camera", settings["camera"],
        "--level", settings["level"],
        "--assist", settings["assist"],
        "--assist-blend", f"{settings['assist_blend'] / 100.0:.2f}",
        "--full-steering", settings["full_steering"],
        "--driving-style", settings["driving_style"],
        "--gearbox", settings["gearbox"],
        "--grip", settings["grip"],
        "--input", settings["input"],
        "--steps", str(int(settings["minutes"] * 60 / POLICY_STEP_SECONDS)),
    ]
    if settings["night"] == "night":
        command.append("--night")
    if settings["window"] == "fullscreen":
        command.append("--fullscreen")
    if settings["ffb"] and settings["input"] == "wheel":
        command += ["--ffb", "--ffb-gain", f"{settings['ffb_gain'] / 100.0:.2f}"]
    return command


class Launcher:
    def __init__(self, root):
        self.root = root
        self.carla_root = os.environ.get("CARLA_ROOT", DEFAULT_CARLA_ROOT)
        self.vars = {
            key: (tk.BooleanVar(value=value) if isinstance(value, bool)
                  else tk.IntVar(value=value) if isinstance(value, int) else tk.StringVar(value=value))
            for key, value in DEFAULTS.items()
        }
        self.town_cards = {}
        root.title("PufferDrive wheel demo")
        root.configure(bg=COLOR_BG)
        root.resizable(False, False)
        self._build()
        self._refresh_town_availability()

    # ----- layout -----

    def _label(self, parent, text, font=FONT_BODY, color=COLOR_TEXT, **grid):
        label = tk.Label(parent, text=text, font=font, fg=color, bg=parent["bg"], anchor="w", justify="left")
        label.grid(**grid)
        return label

    def _segmented(self, parent, key, options, row, column):
        frame = tk.Frame(parent, bg=COLOR_BORDER, padx=1, pady=1)
        frame.grid(row=row, column=column, sticky="w", pady=4)
        for value, text in options:
            tk.Radiobutton(
                frame, text=text, value=value, variable=self.vars[key], indicatoron=False, font=FONT_BODY,
                fg=COLOR_TEXT, bg=COLOR_CARD, selectcolor=COLOR_SEGMENT_SELECTED, activebackground=COLOR_CARD_SELECTED,
                activeforeground=COLOR_TEXT, relief="flat", bd=0, padx=14, pady=6, highlightthickness=0,
                command=self._on_change,
            ).pack(side="left", padx=1)
        return frame

    def _slider(self, parent, key, low, high, suffix, row, column):
        frame = tk.Frame(parent, bg=parent["bg"])
        frame.grid(row=row, column=column, sticky="w", pady=4)
        value_label = tk.Label(frame, font=FONT_BODY, fg=COLOR_ACCENT, bg=parent["bg"], width=6, anchor="w")
        scale = tk.Scale(
            frame, from_=low, to=high, orient="horizontal", variable=self.vars[key], showvalue=False,
            length=SLIDER_LENGTH_PX, sliderlength=SLIDER_HANDLE_PX, width=SLIDER_THICKNESS_PX, bg=COLOR_ACCENT,
            fg=COLOR_TEXT, troughcolor=COLOR_CARD, activebackground=COLOR_TEXT, highlightthickness=0, bd=0,
            sliderrelief="flat",
            command=lambda _value: value_label.configure(text=f"{self.vars[key].get()}{suffix}"),
        )
        scale.pack(side="left")
        value_label.pack(side="left", padx=(8, 0))
        value_label.configure(text=f"{self.vars[key].get()}{suffix}")
        return scale

    def _build(self):
        outer = tk.Frame(self.root, bg=COLOR_BG, padx=2 * PAD, pady=2 * PAD)
        outer.pack(fill="both", expand=True)
        self._label(outer, "PufferDrive", FONT_TITLE, row=0, column=0, sticky="w")
        self._label(outer, "Steering-wheel demo  ·  your car among frozen-policy traffic, rendered by CARLA",
                    FONT_SUBTITLE, COLOR_MUTED, row=1, column=0, sticky="w", pady=(0, PAD))

        self._label(outer, "TOWN", FONT_SECTION, COLOR_MUTED, row=2, column=0, sticky="w")
        towns = tk.Frame(outer, bg=COLOR_BG)
        towns.grid(row=3, column=0, sticky="w", pady=(4, PAD))
        for index, (town, description) in enumerate(TOWNS):
            self.town_cards[town] = self._town_card(towns, town, description, index)

        settings = tk.Frame(outer, bg=COLOR_BG)
        settings.grid(row=4, column=0, sticky="w")
        rows = (
            ("Camera", lambda r: self._segmented(settings, "camera", (("chase", "Chase"), ("driver", "Driver"),
                                                                       ("agent", "Top-down")), r, 1)),
            ("Time of day", lambda r: self._segmented(settings, "night", (("day", "Day"), ("night", "Night")), r, 1)),
            ("Window", lambda r: self._segmented(settings, "window", (("fullscreen", "Full screen"),
                                                                       ("windowed", "Windowed")), r, 1)),
            ("Traffic", lambda r: self._segmented(settings, "level", (("calm", "Calm"), ("normal", "Normal"),
                                                                       ("aggressive", "Aggressive")), r, 1)),
            ("Steering assist", lambda r: self._segmented(settings, "assist", (("off", "Off"), ("display", "Display"),
                                                                               ("blend", "Blend"), ("nudge", "Nudge"),
                                                                               ("full", "Full Self Driving")),
                                                          r, 1)),
            ("Blend share", lambda r: self._slider(settings, "assist_blend", 0, 100, "%", r, 1)),
            ("Self-driving steering", lambda r: self._segmented(settings, "full_steering", (("car", "Car steers, wheel shows"),
                                                                                     ("wheel", "Wheel steers")), r, 1)),
            ("Your style", lambda r: self._segmented(settings, "driving_style", (("off", "Off"), ("calm", "Calm"),
                                                                                  ("normal", "Normal"),
                                                                                  ("aggressive", "Aggressive")), r, 1)),
            ("Force feedback", lambda r: self._segmented(settings, "ffb", ((True, "On"), (False, "Off")), r, 1)),
            ("Strength", lambda r: self._slider(settings, "ffb_gain", 5, 100, "%", r, 1)),
            ("Gearbox", lambda r: self._segmented(settings, "gearbox", (("automatic", "Automatic"),
                                                                         ("manual", "Manual"),
                                                                         ("none", "None (policy's model)")), r, 1)),
            ("Grip", lambda r: self._segmented(settings, "grip", (("realistic", "Realistic"),
                                                                   ("kinematic", "Kinematic"),
                                                                   ("skid", "Skid (can spin)")), r, 1)),
            ("Input", lambda r: self._segmented(settings, "input", (("wheel", "G920 wheel"),
                                                                     ("synthetic", "Scripted (no wheel)")), r, 1)),
            ("Duration", lambda r: self._slider(settings, "minutes", 1, 60, " min", r, 1)),
        )
        self.widgets = {}
        for row, (name, build) in enumerate(rows):
            self._label(settings, name, row=row, column=0, sticky="w", padx=(0, 2 * PAD))
            self.widgets[name] = build(row)

        footer = tk.Frame(outer, bg=COLOR_BG)
        footer.grid(row=5, column=0, sticky="ew", pady=(PAD, 0))
        self.status = self._label(footer, "", FONT_SMALL, COLOR_MUTED, row=0, column=0, sticky="w")
        footer.columnconfigure(0, weight=1)
        self.drive_button = tk.Button(
            footer, text="Drive  ▶", font=FONT_BUTTON, fg=COLOR_BG, bg=COLOR_GO, activebackground=COLOR_GO_ACTIVE,
            activeforeground=COLOR_BG, relief="flat", bd=0, padx=28, pady=10, cursor="hand2", command=self._drive,
        )
        self.drive_button.grid(row=0, column=2, sticky="e")
        self.calibrate_button = tk.Button(
            footer, text="Calibrate wheel", font=FONT_BODY, fg=COLOR_TEXT, bg=COLOR_CARD, activebackground=COLOR_CARD_SELECTED,
            activeforeground=COLOR_TEXT, relief="flat", bd=0, padx=16, pady=10, cursor="hand2", command=self._calibrate,
        )
        self.calibrate_button.grid(row=0, column=1, sticky="e", padx=(0, PAD))
        self.root.bind("<Return>", lambda _event: self._drive())
        self._on_change()

    def _town_card(self, parent, town, description, index):
        card = tk.Frame(parent, bg=COLOR_CARD, highlightthickness=2, highlightbackground=COLOR_BORDER,
                        padx=12, pady=10, cursor="hand2")
        card.grid(row=index // TOWN_COLUMNS, column=index % TOWN_COLUMNS, padx=(0, 10), pady=(0, 10), sticky="nsew")
        title = tk.Label(card, text=town, font=FONT_SECTION, fg=COLOR_TEXT, bg=COLOR_CARD, anchor="w")
        title.pack(fill="x")
        body = tk.Label(card, text=description, font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD, anchor="w",
                        justify="left", wraplength=170, width=26)
        body.pack(fill="x")
        for widget in (card, title, body):
            widget.bind("<Button-1>", lambda _event, name=town: self._select_town(name))
        return card, title, body

    # ----- state -----

    def _select_town(self, town):
        if town in self.unavailable_towns:
            return
        self.vars["town"].set(town)
        self._on_change()

    def _refresh_town_availability(self):
        on_disk = installed_towns(self.carla_root)
        running = carla_available_towns() if carla_is_listening() else None
        self.unavailable_towns = {town for town, _ in TOWNS if town not in (on_disk | (running or set()))}
        source = f"CARLA at {self.carla_root}" if on_disk else "the running CARLA server"
        self._set_status(f"{len(TOWNS) - len(self.unavailable_towns)} of {len(TOWNS)} towns available from {source}")
        self._on_change()

    def _on_change(self):
        selected = self.vars["town"].get()
        for town, (card, title, body) in self.town_cards.items():
            unavailable = town in getattr(self, "unavailable_towns", set())
            background = COLOR_CARD_SELECTED if town == selected else COLOR_CARD
            card.configure(bg=background, highlightbackground=COLOR_ACCENT if town == selected else COLOR_BORDER)
            title.configure(bg=background, fg=COLOR_DISABLED if unavailable else COLOR_TEXT)
            body.configure(bg=background, fg=COLOR_DISABLED if unavailable else COLOR_MUTED)
        if hasattr(self, "widgets"):
            self.widgets["Blend share"].configure(state="normal" if self.vars["assist"].get() == "blend" else "disabled")
            full_state = "normal" if self.vars["assist"].get() == "full" else "disabled"
            for name in ("Self-driving steering", "Your style"):
                for button in self.widgets[name].winfo_children():
                    button.configure(state=full_state)
            wheel = self.vars["input"].get() == "wheel"
            self.widgets["Strength"].configure(state="normal" if self.vars["ffb"].get() and wheel else "disabled")

    def _set_status(self, text, color=COLOR_MUTED):
        self.status.configure(text=text, fg=color)
        self.root.update_idletasks()

    # ----- calibration -----

    def _calibrate(self):
        self.drive_button.configure(state="disabled")
        self.calibrate_button.configure(state="disabled")
        CalibrationWizard(self.root, on_close=self._calibration_closed)

    def _calibration_closed(self, saved_paths):
        self.drive_button.configure(state="normal")
        self.calibrate_button.configure(state="normal")
        if saved_paths:
            self._set_status("Saved " + " and ".join(os.path.basename(path) for path in saved_paths)
                             + "; the next drive uses them.")

    # ----- running -----

    def _ensure_carla(self, town):
        if carla_is_listening():
            towns = carla_available_towns()
            if towns is not None and town in towns:
                return
            self._set_status(f"The running CARLA server has no {town}; restarting it from {self.carla_root} ...")
            subprocess.run(["pkill", "-f", "CarlaUE4-Linux-Shipping"], check=False)
            deadline = time.monotonic() + CARLA_START_TIMEOUT_S
            while carla_is_listening() and time.monotonic() < deadline:
                self.root.update()
                time.sleep(CARLA_POLL_INTERVAL_S)
        self._set_status("Starting the CARLA server on the GPU ...")
        with open(os.path.join(LOG_DIR, "carla_server.log"), "w") as server_log:
            subprocess.Popen([START_CARLA], env=dict(os.environ, CARLA_ROOT=self.carla_root), stdout=server_log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + CARLA_START_TIMEOUT_S
        while not carla_is_listening():
            if time.monotonic() > deadline:
                raise RuntimeError(f"CARLA did not start within {CARLA_START_TIMEOUT_S:.0f} s; see {LOG_DIR}/carla_server.log")
            self.root.update()
            time.sleep(CARLA_POLL_INTERVAL_S)

    def _drive(self):
        settings = {key: var.get() for key, var in self.vars.items()}
        self.drive_button.configure(state="disabled")
        log_path = os.path.join(LOG_DIR, "last_run.log")
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            self._ensure_carla(settings["town"])
            self._set_status(f"Loading {settings['town']} ...")
            with open(log_path, "w") as run_log:
                process = subprocess.Popen(
                    demo_command(settings), cwd=REPO_ROOT, stdout=run_log, stderr=subprocess.STDOUT
                )
        except Exception as error:
            self._set_status(f"Could not start the demo: {error}", COLOR_ERROR)
            self.drive_button.configure(state="normal")
            return
        self.root.withdraw()
        while process.poll() is None:
            self.root.update()
            time.sleep(CARLA_POLL_INTERVAL_S)
        self.root.deiconify()
        self.drive_button.configure(state="normal")
        if process.returncode == 0:
            self._set_status(f"Finished. Log: {log_path}")
            return
        with open(log_path) as run_log:
            tail = "".join(run_log.readlines()[-LOG_TAIL_LINES:]).strip()
        self._set_status(f"The demo stopped with an error (exit {process.returncode}):\n{tail}", COLOR_ERROR)


def save_with_backup(write, path):
    """Keep the file being replaced next to it, so a bad recalibration can be undone by hand."""
    if os.path.exists(path):
        shutil.copyfile(path, path[: -len(".json")] + CALIBRATION_BACKUP_SUFFIX)
    write(path)


class CalibrationCancelled(Exception):
    pass


class CalibrationWizard:
    """Recalibrates the wheel from the launcher: the same capture and checks as `run_demo.py --calibrate`
    and `probe_g920.py buttons`, one instruction at a time. The capture runs on a worker thread; the window
    only reacts to the messages it posts, because Tk must only be touched from its own thread."""

    def __init__(self, root, on_close):
        from pufferlib.wheel_demo import calibrate

        self.calibrate = calibrate
        self.on_close = on_close
        self.messages = queue.Queue()
        self.ready = threading.Event()
        self.cancelled = False
        self.closed = False
        self.map_buttons = True  # read from the checkbox on the window thread when the capture starts
        self.result = None
        self.window = tk.Toplevel(root, bg=COLOR_BG, padx=2 * PAD, pady=2 * PAD)
        self.window.title("Calibrate wheel")
        self.window.transient(root)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)
        self.include_buttons = tk.BooleanVar(value=True)
        self._build()
        self.window.grab_set()

    def _build(self):
        window = self.window
        tk.Label(window, text="Calibrate wheel", font=FONT_TITLE, fg=COLOR_TEXT, bg=COLOR_BG).grid(row=0, column=0, sticky="w")
        self.step_label = tk.Label(window, text="", font=FONT_SECTION, fg=COLOR_MUTED, bg=COLOR_BG, anchor="w")
        self.step_label.grid(row=1, column=0, sticky="w", pady=(PAD, 0))
        self.instruction = tk.Label(
            window, font=FONT_SUBTITLE, fg=COLOR_TEXT, bg=COLOR_BG, justify="left", anchor="w", wraplength=WIZARD_TEXT_WRAP_PX,
            text=("Records your wheel's steering range, pedal travel and button layout, and saves them for the demo "
                  f"({os.path.basename(CALIBRATION_PATH)}, {os.path.basename(BUTTON_MAP_PATH)}; the old files are kept "
                  f"as *{CALIBRATION_BACKUP_SUFFIX}). Close the demo first: only one program can use the wheel."),
        )
        self.instruction.grid(row=2, column=0, sticky="w", pady=(4, PAD))
        self.buttons_check = tk.Checkbutton(
            window, text="Also map the buttons, paddles and D-pad", variable=self.include_buttons, font=FONT_BODY,
            fg=COLOR_TEXT, bg=COLOR_BG, selectcolor=COLOR_CARD, activebackground=COLOR_BG, activeforeground=COLOR_TEXT,
        )
        self.buttons_check.grid(row=3, column=0, sticky="w")
        self.status = tk.Label(window, text="", font=FONT_BODY, fg=COLOR_ACCENT, bg=COLOR_BG, justify="left", anchor="w",
                               wraplength=WIZARD_TEXT_WRAP_PX)
        self.status.grid(row=4, column=0, sticky="w", pady=(PAD, 0))
        self.log = tk.Text(window, width=72, height=10, font=FONT_SMALL, fg=COLOR_MUTED, bg=COLOR_CARD, relief="flat",
                           state="disabled")
        self.log.grid(row=5, column=0, sticky="ew", pady=(PAD, PAD))
        actions = tk.Frame(window, bg=COLOR_BG)
        actions.grid(row=6, column=0, sticky="e")
        self.cancel_button = self._action(actions, "Cancel", self._cancel, COLOR_CARD, COLOR_TEXT)
        self.save_button = self._action(actions, "Save", self._save, COLOR_GO, COLOR_BG)
        self.start_button = self._action(actions, "Start", self._start, COLOR_ACCENT, COLOR_BG)
        self.save_button.pack_forget()

    def _action(self, parent, text, command, background, foreground):
        button = tk.Button(parent, text=text, font=FONT_BODY, fg=foreground, bg=background, activebackground=background,
                           activeforeground=foreground, relief="flat", bd=0, padx=18, pady=8, cursor="hand2",
                           command=command)
        button.pack(side="left", padx=(PAD, 0))
        return button

    # ----- window thread -----

    def _start(self):
        self.start_button.configure(state="disabled")
        if self.result is None and not hasattr(self, "worker"):
            self.buttons_check.configure(state="disabled")
            self.map_buttons = self.include_buttons.get()
            self.phase_seconds = self._phase_seconds()
            self.phase_idx = -1
            self.worker = threading.Thread(target=self._run, name="calibration", daemon=True)
            self.worker.start()
            self.window.after(WIZARD_POLL_MS, self._poll)
            return
        self._recording_started_s = time.monotonic()
        self.ready.set()

    def _phase_seconds(self):
        """How long each prompt's recording lasts, in the order the capture asks for them, for the countdown."""
        calibrate = self.calibrate
        seconds = [calibrate.SETTLE_SECONDS + 2 * calibrate.CAPTURE_SECONDS]
        seconds += [calibrate.SETTLE_SECONDS + calibrate.CAPTURE_SECONDS] * len(calibrate.PEDAL_BYTE_IDX)
        if self.map_buttons:
            button_prompts = 1 + len(calibrate.BUTTON_CONTROLS) + len(calibrate.DPAD_DIRECTION_NAMES)
            seconds += [BUTTON_SETTLE_SECONDS + BUTTON_CAPTURE_SECONDS] * button_prompts
        return seconds

    def _poll(self):
        if self.closed:
            return
        while not self.messages.empty():
            kind, payload = self.messages.get()
            if kind == "prompt":
                self.phase_idx += 1
                self.step_label.configure(text=f"STEP {self.phase_idx + 1} OF {len(self.phase_seconds)}")
                self.instruction.configure(text=payload)
                self.status.configure(text="Get ready, then press Start and do it.", fg=COLOR_ACCENT)
                self._recording_started_s = None
                self.start_button.configure(state="normal")
            elif kind == "log":
                self._append_log(payload)
            elif kind == "error":
                self._finish_with_error(payload)
                return
            elif kind == "done":
                self._show_result(payload)
                return
        started_s = getattr(self, "_recording_started_s", None)
        if started_s is not None and 0 <= self.phase_idx < len(self.phase_seconds):
            remaining_s = max(0.0, self.phase_seconds[self.phase_idx] - (time.monotonic() - started_s))
            self.status.configure(text=f"Recording ... {remaining_s:.0f} s", fg=COLOR_GO)
        self.window.after(WIZARD_POLL_MS, self._poll)

    def _append_log(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _finish_with_error(self, message):
        self.step_label.configure(text="CALIBRATION STOPPED")
        self.instruction.configure(text=message)
        self.status.configure(text="Nothing was saved.", fg=COLOR_ERROR)
        self.start_button.pack_forget()
        self.cancel_button.configure(text="Close")

    def _show_result(self, result):
        self.result = result
        calibration, problems, button_map = result
        self.step_label.configure(text="DONE")
        lines = [f"Steering {calibration.steering_raw_min} .. {calibration.steering_raw_max}, pedals recorded."]
        if button_map is not None:
            unmapped = [name for name, value in button_map.__dict__.items() if value is None]
            lines.append("Buttons: all mapped." if not unmapped else f"Buttons not mapped: {', '.join(unmapped)}.")
            if button_map.shift_up_bit is None or button_map.shift_down_bit is None:
                lines.append("The paddles are unmapped: gear and reverse selection will not work.")
        self.instruction.configure(text="\n".join(lines))
        if problems:
            self.status.configure(text="Check before saving:\n- " + "\n- ".join(problems), fg=COLOR_ERROR)
        else:
            self.status.configure(text="No problems found. Save to use it on the next drive.", fg=COLOR_GO)
        self.start_button.pack_forget()
        self.save_button.pack(side="left", padx=(PAD, 0))
        self.cancel_button.configure(text="Discard")

    def _save(self):
        calibration, _problems, button_map = self.result
        saved = []
        save_with_backup(calibration.to_json, CALIBRATION_PATH)
        saved.append(CALIBRATION_PATH)
        if button_map is not None:
            save_with_backup(button_map.to_json, BUTTON_MAP_PATH)
            saved.append(BUTTON_MAP_PATH)
        self._close(saved)

    def _cancel(self):
        self.cancelled = True
        self.ready.set()
        self._close([])

    def _close(self, saved_paths):
        self.closed = True
        self.window.grab_release()
        self.window.destroy()
        self.on_close(saved_paths)

    # ----- worker thread -----

    def _prompt(self, instruction):
        self.messages.put(("prompt", instruction))
        self.ready.wait()
        self.ready.clear()
        if self.cancelled:
            raise CalibrationCancelled()

    def _button_prompt(self, instruction):
        self._prompt(instruction)
        time.sleep(BUTTON_SETTLE_SECONDS)

    def _log(self, line):
        self.messages.put(("log", line))

    def _run(self):
        calibrate = self.calibrate
        try:
            device = calibrate.open_wheel()
        except Exception as error:
            self.messages.put(("error", f"Could not open the wheel: {error}"))
            return
        try:
            calibration = calibrate.capture_calibration(device, prompt=self._prompt, log=self._log)
            problems = calibrate.validate_calibration(calibration)
            button_map = None
            if self.map_buttons:
                button_map = calibrate.capture_button_map(
                    device, self._button_prompt, lambda reader: calibrate.collect(reader, BUTTON_CAPTURE_SECONDS),
                    log=self._log,
                )
            self.messages.put(("done", (calibration, problems, button_map)))
        except CalibrationCancelled:
            return
        except Exception as error:
            self.messages.put(("error", str(error)))
        finally:
            device.close()


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
