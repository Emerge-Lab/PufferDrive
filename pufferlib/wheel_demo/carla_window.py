"""The demo window: CARLA's camera image with the HUD drawn over it, paced to real time.

Needs a running CARLA server (scripts/wheel_demo/start_carla.sh) and the CARLA Python client, which
ships wheels for Python 3.10-3.12 only.
"""

import os
import time

from pufferlib.ocean.cosim import carla_bridge
from pufferlib.wheel_demo.carla_scene import CAMERA_MODE_AGENT, CAMERA_MODE_CHASE, CAMERA_MODE_DRIVER, CarlaScene
from pufferlib.wheel_demo.hud import Hud, HudInfo


CAMERA_MODE_NAMES = ("chase", "driver", "agent")
DEFAULT_WINDOW_SIZE_PX = (1280, 720)
WINDOW_TITLE = "PufferDrive wheel demo"
REAR_MIRROR_TOP_PX = 10
SIDE_MIRROR_TOP_FRACTION = 0.38
SIDE_MIRROR_EDGE_PX = 14
# (x0, y0, x1, y1) fractions of the flipped rear image; flipped, the car's left is on the left.
MIRROR_CROPS = {"left": (0.0, 0.2, 0.22, 1.0), "rear": (0.33, 0.3, 0.67, 0.75), "right": (0.78, 0.2, 1.0, 1.0)}
MIRROR_DISPLAY_SIZES_PX = {"left": (300, 190), "rear": (640, 150), "right": (300, 190)}
# Behind real time by more than this many frames, a frame is skipped rather than slowing the sim.
MAX_FRAMES_BEHIND = 2
# The wheel's own buttons drive the demo; the keyboard (1 2 3 N L M TAB R Esc) still works too.
HELP_TEXT = (
    "A camera   B night   X traffic   Y next car   VIEW restart engine   MENU assist   "
    "D-pad up/down force   stopped: L paddle reverse, R paddle drive"
)


class CarlaDemoWindow:
    def __init__(
        self,
        env,
        town,
        human_agent_idx,
        substep_dt_s,
        camera_mode=CAMERA_MODE_CHASE,
        night_mode=False,
        host="localhost",
        port=2000,
        size_px=DEFAULT_WINDOW_SIZE_PX,
    ):
        import carla
        import pygame

        self.pygame = pygame
        self.size_px = size_px
        self.substep_dt_s = substep_dt_s
        self.scene = CarlaScene(
            carla, town, env, human_agent_idx, substep_dt_s, host=host, port=port, image_size_px=size_px
        )
        try:
            pygame.init()
            self.screen = pygame.display.set_mode(size_px)
            pygame.display.set_caption(WINDOW_TITLE)
            self.hud = Hud(pygame, carla_bridge.bin_path_for_town(town))
            self.night_mode = bool(night_mode)
            self.scene.set_night(self.night_mode)
            self.set_camera_mode(camera_mode)
        except Exception:
            self.scene.close()
            raise
        self._keys = {}
        self._mirror_cache = None
        self._closed = False
        self._next_frame_s = None
        self.frames_drawn = 0
        self.frames_skipped = 0

    def set_camera_mode(self, camera_mode):
        self.scene.set_camera_mode(int(camera_mode))
        self.camera_mode = int(camera_mode)

    def set_human_agent(self, agent_idx):
        self.scene.set_human_agent(agent_idx)

    def toggle_night(self):
        self.night_mode = not self.night_mode
        self.scene.set_night(self.night_mode)

    def _surface(self, image, mirrored=False):
        surface = self.pygame.image.frombuffer(image.raw_data, (image.width, image.height), "BGRA")
        return self.pygame.transform.flip(surface, True, False) if mirrored else surface

    def _mirror_surfaces(self, image):
        """Rear-view, left and right mirrors cut from one flipped rear image, rebuilt only on a new frame."""
        if self._mirror_cache is not None and self._mirror_cache[0] == image.frame:
            return self._mirror_cache[1]
        flipped = self._surface(image, mirrored=True)
        surfaces = {}
        for name, (x0, y0, x1, y1) in MIRROR_CROPS.items():
            crop = self.pygame.Rect(
                int(x0 * image.width),
                int(y0 * image.height),
                int((x1 - x0) * image.width),
                int((y1 - y0) * image.height),
            )
            surfaces[name] = self.pygame.transform.smoothscale(flipped.subsurface(crop), MIRROR_DISPLAY_SIZES_PX[name])
        self._mirror_cache = (image.frame, surfaces)
        return surfaces

    def _mirror_layout(self, images):
        if "rear" not in images:
            return []
        pygame = self.pygame
        width, height = self.size_px
        surfaces = self._mirror_surfaces(images["rear"])
        rear_rect = pygame.Rect((0, REAR_MIRROR_TOP_PX), MIRROR_DISPLAY_SIZES_PX["rear"])
        rear_rect.centerx = width // 2
        side_top = int(SIDE_MIRROR_TOP_FRACTION * height)
        side_width, side_height = MIRROR_DISPLAY_SIZES_PX["left"]
        return [
            (surfaces["rear"], rear_rect),
            (surfaces["left"], pygame.Rect(SIDE_MIRROR_EDGE_PX, side_top, side_width, side_height)),
            (
                surfaces["right"],
                pygame.Rect(width - SIDE_MIRROR_EDGE_PX - side_width, side_top, side_width, side_height),
            ),
        ]

    def draw(self, human, wheel_state, elapsed_seconds, assist=None, level_name="normal", ffb_gain=None):
        """One real-time frame. Returns False when the frame was skipped to catch up."""
        now_s = time.perf_counter()
        if self._next_frame_s is None:
            self._next_frame_s = now_s
        self._next_frame_s += self.substep_dt_s
        if now_s - self._next_frame_s > MAX_FRAMES_BEHIND * self.substep_dt_s:
            self.frames_skipped += 1
            self._pump_events()
            return False
        images = self.scene.update(human, wheel_state, self.substep_dt_s)
        self.screen.blit(self._surface(images["main"]), (0, 0))
        goal_marker = self.scene.goal_marker(human, images["main"])
        self.hud.draw(
            self.screen,
            human,
            wheel_state,
            HudInfo(camera_mode=self.camera_mode, level_name=level_name, ffb_gain=ffb_gain, help_text=HELP_TEXT),
            elapsed_seconds,
            mirrors=self._mirror_layout(images),
            assist=assist,
            goal_marker=goal_marker,
        )
        self.pygame.display.flip()
        self.frames_drawn += 1
        self._pump_events()
        remaining_s = self._next_frame_s - time.perf_counter()
        if remaining_s > 0.0:
            time.sleep(remaining_s)
        return True

    def _pump_events(self):
        pygame = self.pygame
        names = {
            pygame.K_1: "camera_chase",
            pygame.K_2: "camera_driver",
            pygame.K_3: "camera_agent",
            pygame.K_n: "toggle_night",
            pygame.K_l: "cycle_level",
            pygame.K_TAB: "next_agent",
            pygame.K_r: "restart_engine",
            pygame.K_m: "cycle_assist",
        }
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self._closed = True
            elif event.type == pygame.KEYDOWN and event.key in names:
                self._keys[names[event.key]] = True

    def poll(self):
        keys, self._keys = self._keys, {}
        return keys

    def should_close(self):
        return self._closed

    def screenshot(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.pygame.image.save(self.screen, path)

    def close(self):
        self.scene.close()
        self.pygame.quit()


CAMERA_MODES_BY_NAME = {"chase": CAMERA_MODE_CHASE, "driver": CAMERA_MODE_DRIVER, "agent": CAMERA_MODE_AGENT}
