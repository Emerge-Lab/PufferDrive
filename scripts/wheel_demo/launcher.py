"""Desktop launcher for the wheel demo: pick the town and settings, then drive.

    python scripts/wheel_demo/launcher.py

Starts the CARLA server when it is not running (or restarts it from CARLA_ROOT when the running one
lacks the chosen town), then runs run_demo.py with the chosen settings and comes back when it exits.
Install the desktop icon with scripts/wheel_demo/install_launcher.py.
"""

import os
import socket
import subprocess
import sys
import time
import tkinter as tk

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DEMO = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "run_demo.py")
START_CARLA = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "start_carla.sh")
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
    "town": "Town03",
    "camera": "chase",
    "night": "day",
    "level": "normal",
    "assist": "off",
    "assist_blend": 50,
    "ffb": True,
    "ffb_gain": 70,
    "gearbox": "automatic",
    "grip": "realistic",
    "input": "wheel",
    "minutes": 20,
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
        "--gearbox", settings["gearbox"],
        "--grip", settings["grip"],
        "--input", settings["input"],
        "--steps", str(int(settings["minutes"] * 60 / POLICY_STEP_SECONDS)),
    ]
    if settings["night"] == "night":
        command.append("--night")
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
            ("Traffic", lambda r: self._segmented(settings, "level", (("calm", "Calm"), ("normal", "Normal"),
                                                                       ("aggressive", "Aggressive")), r, 1)),
            ("Steering assist", lambda r: self._segmented(settings, "assist", (("off", "Off"), ("display", "Display"),
                                                                               ("blend", "Blend"), ("nudge", "Nudge"),
                                                                               ("full", "Full")),
                                                          r, 1)),
            ("Blend share", lambda r: self._slider(settings, "assist_blend", 0, 100, "%", r, 1)),
            ("Force feedback", lambda r: self._segmented(settings, "ffb", ((True, "On"), (False, "Off")), r, 1)),
            ("Strength", lambda r: self._slider(settings, "ffb_gain", 5, 100, "%", r, 1)),
            ("Gearbox", lambda r: self._segmented(settings, "gearbox", (("automatic", "Automatic"),
                                                                         ("manual", "Manual")), r, 1)),
            ("Grip", lambda r: self._segmented(settings, "grip", (("realistic", "Realistic"),
                                                                   ("kinematic", "Kinematic")), r, 1)),
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
        self.drive_button.grid(row=0, column=1, sticky="e")
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
            wheel = self.vars["input"].get() == "wheel"
            self.widgets["Strength"].configure(state="normal" if self.vars["ffb"].get() and wheel else "disabled")

    def _set_status(self, text, color=COLOR_MUTED):
        self.status.configure(text=text, fg=color)
        self.root.update_idletasks()

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


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
