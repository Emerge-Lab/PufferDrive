"""Put a PufferDrive wheel-demo icon on the desktop and in the application menu.

    .venv312/bin/python scripts/wheel_demo/install_launcher.py

The icon runs scripts/wheel_demo/launcher.py with this interpreter, so run it from the venv the demo
uses. Paths are written for this checkout; re-run after moving the repo.
"""

import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

APP_ID = "pufferdrive-wheel-demo"
ICON_SIZE_PX = 256
ICON_WHEEL_RADIUS_PX = 100
ICON_CORNER_RADIUS_PX = 48
ICON_BACKGROUND = (20, 22, 27)
ICON_DIR = os.path.expanduser("~/.local/share/icons")
APPLICATIONS_DIR = os.path.expanduser("~/.local/share/applications")
TOWN_MAP = os.path.join(REPO_ROOT, "pufferlib", "resources", "drive", "binaries", "carla", "opendrive__Town01.bin")


def render_icon(path):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from pufferlib.wheel_demo.hud import Hud

    pygame.init()
    pygame.display.set_mode((1, 1))
    icon = pygame.Surface((ICON_SIZE_PX, ICON_SIZE_PX), pygame.SRCALPHA)
    pygame.draw.rect(icon, ICON_BACKGROUND, icon.get_rect(), border_radius=ICON_CORNER_RADIUS_PX)
    wheel = Hud(pygame, TOWN_MAP)._wheel_sprite(ICON_WHEEL_RADIUS_PX)
    icon.blit(wheel, wheel.get_rect(center=icon.get_rect().center))
    pygame.image.save(icon, path)
    pygame.quit()


def desktop_entry(icon_path):
    launcher = os.path.join(REPO_ROOT, "scripts", "wheel_demo", "launcher.py")
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=PufferDrive Wheel Demo\n"
        "Comment=Drive among frozen-policy traffic with the G920, rendered by CARLA\n"
        f'Exec="{sys.executable}" "{launcher}"\n'
        f"Path={REPO_ROOT}\n"
        f"Icon={icon_path}\n"
        "Terminal=false\n"
        "Categories=Game;Simulation;\n"
    )


def desktop_dir():
    try:
        return subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return os.path.expanduser("~/Desktop")


def main():
    os.makedirs(ICON_DIR, exist_ok=True)
    os.makedirs(APPLICATIONS_DIR, exist_ok=True)
    icon_path = os.path.join(ICON_DIR, f"{APP_ID}.png")
    render_icon(icon_path)
    entry = desktop_entry(icon_path)
    targets = [os.path.join(APPLICATIONS_DIR, f"{APP_ID}.desktop")]
    desktop = desktop_dir()
    if os.path.isdir(desktop):
        targets.append(os.path.join(desktop, f"{APP_ID}.desktop"))
    for target in targets:
        with open(target, "w") as handle:
            handle.write(entry)
        os.chmod(target, 0o755)
        # GNOME will not launch a desktop file until it is marked trusted
        subprocess.run(["gio", "set", target, "metadata::trusted", "true"], check=False)
        print(f"installed {target}")
    print(f"icon {icon_path}")


if __name__ == "__main__":
    main()
