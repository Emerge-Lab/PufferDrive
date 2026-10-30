"""Window for the wheel demo.

The renderer is a separate extension (pufferlib/ocean/drive/render) so the training build never
links raylib. Build it with `make -C pufferlib/ocean/drive/render`; importing without that build
raises with the command to run rather than a bare ImportError.
"""

import os
import sys

RENDER_MODULE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ocean", "drive", "render"
)
CAR_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "drive")

CAMERA_MODE_CHASE = 0
CAMERA_MODE_DRIVER = 1
CAMERA_MODE_AGENT = 2
CAMERA_MODE_NAMES = ("chase", "driver", "agent")

DEFAULT_WINDOW_WIDTH = 1280
DEFAULT_WINDOW_HEIGHT = 720


def _import_render_module():
    if RENDER_MODULE_DIR not in sys.path:
        sys.path.insert(0, RENDER_MODULE_DIR)
    try:
        import drive_render
    except ImportError as error:
        raise ImportError(
            "the wheel demo renderer is not built. Run:\n"
            "    make -C pufferlib/ocean/drive/render\n"
            f"(looked in {RENDER_MODULE_DIR})"
        ) from error
    return drive_render


class DemoWindow:
    """Owns the window. One per process; raylib has a single global context."""

    def __init__(
        self,
        camera_mode=CAMERA_MODE_CHASE,
        night_mode=False,
        width=DEFAULT_WINDOW_WIDTH,
        height=DEFAULT_WINDOW_HEIGHT,
        resource_dir=CAR_MODEL_DIR,
    ):
        self._render = _import_render_module()
        self._render.init(resource_dir, width, height, camera_mode, int(bool(night_mode)))
        self.camera_mode = camera_mode
        self.night_mode = bool(night_mode)

    def draw(self, env, elapsed_seconds):
        self._render.frame(env.c_envs, float(elapsed_seconds))

    def set_camera_mode(self, camera_mode):
        self._render.set_camera_mode(int(camera_mode))
        self.camera_mode = int(camera_mode)

    def toggle_night(self):
        self.night_mode = not self.night_mode
        self._render.set_night_mode(int(self.night_mode))

    def poll(self):
        return self._render.poll_keys()

    def should_close(self):
        return self._render.should_close()

    def close(self):
        self._render.close()
