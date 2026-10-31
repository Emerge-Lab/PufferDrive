"""Heads-up display drawn over the CARLA camera image.

Everything the driver reads is in DRIVER terms: a goal to the right points right. The sim is
positive-left, so every bearing taken from sim headings is negated here before it is drawn.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np



MPS_TO_KMH = 3.6
GEAR_REVERSE = -1
GEAR_NEUTRAL = 0
GEARBOX_MODE_AUTOMATIC = 1

# Lane polylines in the bin are road types 0..9; edges and markings are drawn separately.
BIN_LANE_TYPE_MAX = 9

PANEL_MARGIN_PX = 18
SPEED_FONT_PX = 64
LABEL_FONT_PX = 22
SMALL_FONT_PX = 17
PEDAL_BAR_WIDTH_PX = 22
PEDAL_BAR_HEIGHT_PX = 120
PEDAL_BAR_GAP_PX = 14
GOAL_ARROW_RADIUS_PX = 46
GOAL_ARROW_HEAD_PX = 30
SPEED_SIGN_RADIUS_PX = 44
SPEED_SIGN_RING_PX = 8
MINIMAP_SIZE_PX = 250
MINIMAP_MARGIN_M = 20.0
MINIMAP_EGO_RADIUS_PX = 6
MINIMAP_GOAL_RADIUS_PX = 6
MINIMAP_LANE_WIDTH_PX = 2
OVERSPEED_FLASH_HZ = 2.0

COLOR_TEXT = (240, 242, 245)
COLOR_SHADOW = (10, 12, 16)
COLOR_PANEL = (14, 16, 22, 165)
COLOR_THROTTLE = (70, 200, 110)
COLOR_BRAKE = (230, 70, 60)
COLOR_BAR_TRACK = (60, 64, 72)
COLOR_WARN = (255, 80, 64)
COLOR_SIGN_FACE = (250, 250, 250)
COLOR_SIGN_RING = (210, 30, 30)
COLOR_SIGN_TEXT = (20, 20, 20)
COLOR_ARROW = (90, 200, 255)
COLOR_MINIMAP_BG = (22, 26, 32, 200)
COLOR_MINIMAP_LANE = (120, 126, 136)
COLOR_MINIMAP_EGO = (90, 200, 255)
COLOR_MINIMAP_GOAL = (255, 200, 60)
COLOR_MIRROR_FRAME = (20, 20, 24)
COLOR_ASSIST_YOU = (240, 242, 245)
COLOR_ASSIST_POLICY = (90, 200, 255)
ASSIST_MODE_NAMES = {0: "off", 1: "display", 2: "blend", 3: "nudge", 4: "full"}
CAMERA_MODE_DRIVER = 1
DRIVER_WHEEL_RADIUS_PX = 110
# Left-hand drive: in the driver's view the wheel sits left of centre, in front of the driver.
DRIVER_WHEEL_CENTER_X_FRACTION = 0.36
DIAL_WHEEL_RADIUS_PX = 46
WHEEL_BOTTOM_GAP_PX = 26
WHEEL_RIM_WIDTH_FRACTION = 0.13
# the hand grips at nine and three o'clock are thicker than the rest of the rim
WHEEL_GRIP_WIDTH_FRACTION = 0.17
WHEEL_GRIP_HALF_SPAN_DEG = 28.0
WHEEL_HUB_WIDTH_FRACTION = 0.62
WHEEL_HUB_HEIGHT_FRACTION = 0.44
WHEEL_BADGE_FRACTION = 0.1
WHEEL_STRIPE_HALF_SPAN_DEG = 4.0
WHEEL_SPOKE_WIDTH_FRACTION = 0.22
# spokes at nine, three and six o'clock, measured clockwise from twelve with the wheel centred
WHEEL_SPOKE_ANGLES_DEG = (100.0, 260.0, 180.0)
WHEEL_SPRITE_PAD_PX = 4
ASSIST_ARROW_HEAD_FRACTION = 0.3
GOAL_BEACON_WIDTH_PX = 4
GOAL_DIAMOND_PX = 11
INFO_TOP_PX = 2 * PANEL_MARGIN_PX + 2 * GOAL_ARROW_RADIUS_PX
INFO_LINE_PX = 24
COLOR_WHEEL_LEATHER_DARK = (16, 17, 19)
COLOR_WHEEL_LEATHER_LIGHT = (52, 54, 58)
COLOR_WHEEL_SPOKE = (78, 81, 88)
COLOR_WHEEL_SPOKE_EDGE = (120, 124, 132)
COLOR_WHEEL_HUB = (30, 31, 35)
COLOR_WHEEL_BADGE = (176, 180, 188)
COLOR_WHEEL_STRIPE = (214, 40, 36)
COLOR_GOAL_BEACON = (255, 200, 60)
PEDAL_MARKER_OVERHANG_PX = 4
PEDAL_MARKER_THICKNESS_PX = 3
COLOR_LEVEL = {"calm": (110, 210, 120), "normal": (240, 242, 245), "aggressive": (255, 90, 70)}


def speed_excess_mps(speed_mps, speed_limit_mps, overspeed_tolerance_mps):
    """How far over the limit, by the sim's overspeed metric; positive only when over."""
    # the sim sums limit and tolerance in float32; doing the same keeps the threshold bit-identical
    threshold = float(np.float32(speed_limit_mps) + np.float32(overspeed_tolerance_mps))
    return abs(speed_mps) - threshold


def wrap_angle_rad(angle_rad):
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def goal_bearing_driver_rad(x, y, heading_rad, goal_x, goal_y):
    """Bearing to the goal relative to the car's nose, POSITIVE = to the driver's RIGHT."""
    sim_bearing = math.atan2(goal_y - y, goal_x - x) - heading_rad
    return -wrap_angle_rad(sim_bearing)


def gear_label(gear, gearbox_mode):
    if gear == GEAR_REVERSE:
        return "R"
    if gear == GEAR_NEUTRAL:
        return "N"
    return f"D{gear}" if gearbox_mode == GEARBOX_MODE_AUTOMATIC else str(gear)


@dataclass(frozen=True)
class SpeedLimitReadout:
    limit_kmh: int
    over_limit: bool


def speed_limit_readout(speed_mps, speed_limit_mps, overspeed_tolerance_mps):
    """The sign shows the posted limit; the warning fires on the sim's own overspeed test."""
    over = speed_excess_mps(speed_mps, speed_limit_mps, overspeed_tolerance_mps) > 0.0
    return SpeedLimitReadout(limit_kmh=round(speed_limit_mps * MPS_TO_KMH), over_limit=over)


@dataclass(frozen=True)
class AssistReadout:
    """Your steering and the policy's, in DRIVER terms (positive = right), and how far apart."""

    your_steering_rad: float
    policy_steering_rad: float
    steer_delta_wheel_deg: float
    accel_delta_mps2: float


def assist_readout(assist, human_accel_long_mps2, steering_ratio):
    """Delta = policy - you: a positive steering delta means the policy would turn further RIGHT."""
    your_rad = -assist["human_steering_rad"]
    policy_rad = -assist["reference_steering_rad"]
    return AssistReadout(
        your_steering_rad=your_rad,
        policy_steering_rad=policy_rad,
        steer_delta_wheel_deg=math.degrees(policy_rad - your_rad) * steering_ratio,
        accel_delta_mps2=assist["reference_accel_mps2"] - human_accel_long_mps2,
    )


def wheel_display_deg(sim_steering_rad, steering_ratio, steering_limit_rad):
    """Hand-wheel angle in DRIVER terms (positive = clockwise = right), clamped to the wheel's lock."""
    road_driver_rad = max(-steering_limit_rad, min(steering_limit_rad, -sim_steering_rad))
    return math.degrees(road_driver_rad) * steering_ratio


def clockwise_point(center, radius_px, angle_deg):
    """Screen point at angle_deg clockwise from twelve o'clock (screen y grows downward)."""
    angle = math.radians(angle_deg)
    return (center[0] + radius_px * math.sin(angle), center[1] - radius_px * math.cos(angle))


def ring_segment(center, outer_radius_px, inner_radius_px, start_deg, stop_deg, step_deg=2.0):
    """Polygon of an annulus between two clockwise-from-twelve angles: smooth where a thick arc is ragged."""
    count = max(2, int(abs(stop_deg - start_deg) / step_deg) + 1)
    angles = [start_deg + (stop_deg - start_deg) * i / (count - 1) for i in range(count)]
    outer = [clockwise_point(center, outer_radius_px, angle) for angle in angles]
    inner = [clockwise_point(center, inner_radius_px, angle) for angle in reversed(angles)]
    return outer + inner


def policy_pedal_targets(reference_accel_mps2, accel_min_mps2, accel_max_mps2):
    """Where the policy would have the pedals, as (throttle, brake) fractions of the car's full accel/decel."""
    if reference_accel_mps2 >= 0.0:
        return min(1.0, reference_accel_mps2 / accel_max_mps2), 0.0
    return 0.0, min(1.0, reference_accel_mps2 / accel_min_mps2)


def policy_drives_speed(assist, wheel_state, gear, full_mode, pedal_released_fraction):
    """FULL, a forward gear and both feet off the pedals: the policy is driving the car's speed. Compared
    in float32, as the sim compares them, so the HUD and the car agree at the threshold."""
    if assist is None or assist["mode"] != full_mode or not assist["reference_valid"] or gear < 1:
        return False
    threshold = np.float32(pedal_released_fraction)
    return np.float32(wheel_state.throttle_norm) < threshold and np.float32(wheel_state.brake_norm) < threshold


@dataclass(frozen=True)
class HudInfo:
    """What the window knows that the sim does not: view, traffic level, force-feedback strength."""

    camera_mode: int = 0
    level_name: str = "normal"
    ffb_gain: float = None
    help_text: str = ""


def pedal_fraction(value):
    return min(1.0, max(0.0, float(value)))


class MinimapProjection:
    """Bin-frame metres to minimap pixels, north-up (bin +y is up), whole town in view."""

    def __init__(self, lane_points_xy, size_px=MINIMAP_SIZE_PX, margin_m=MINIMAP_MARGIN_M):
        if len(lane_points_xy) == 0:
            raise ValueError("minimap needs at least one lane point")
        low = lane_points_xy.min(axis=0) - margin_m
        high = lane_points_xy.max(axis=0) + margin_m
        span_m = float(max(high[0] - low[0], high[1] - low[1]))
        self.size_px = size_px
        self.scale_px_per_m = size_px / span_m
        self.center = (low + high) / 2.0

    def to_px(self, x, y):
        px = self.size_px / 2.0 + (x - self.center[0]) * self.scale_px_per_m
        py = self.size_px / 2.0 - (y - self.center[1]) * self.scale_px_per_m
        return px, py


def bin_lane_polylines(bin_path):
    from data_utils import mirror_map_bin

    data = mirror_map_bin.read_bin(Path(bin_path))
    return [
        np.column_stack([road["x"], road["y"]])
        for road in data["roads"]
        if 0 <= road["type"] <= BIN_LANE_TYPE_MAX and len(road["x"]) >= 2
    ]


class Hud:
    """Draws onto a pygame surface. Owns fonts and the pre-rendered minimap."""

    def __init__(self, pygame, bin_path):
        self.pygame = pygame
        self.font_speed = pygame.font.Font(None, SPEED_FONT_PX)
        self.font_label = pygame.font.Font(None, LABEL_FONT_PX)
        self.font_small = pygame.font.Font(None, SMALL_FONT_PX)
        self.font_sign = pygame.font.Font(None, int(SPEED_SIGN_RADIUS_PX * 1.05))
        polylines = bin_lane_polylines(bin_path)
        self.minimap = MinimapProjection(np.vstack(polylines))
        self.minimap_surface = self._render_minimap_base(polylines)
        self._wheel_sprites = {}

    def _render_minimap_base(self, polylines):
        pygame = self.pygame
        surface = pygame.Surface((MINIMAP_SIZE_PX, MINIMAP_SIZE_PX), pygame.SRCALPHA)
        surface.fill(COLOR_MINIMAP_BG)
        for line in polylines:
            points = [self.minimap.to_px(x, y) for x, y in line]
            pygame.draw.lines(surface, COLOR_MINIMAP_LANE, False, points, MINIMAP_LANE_WIDTH_PX)
        return surface

    def _text(self, surface, font, text, position, color=COLOR_TEXT, anchor="topleft"):
        rendered = font.render(text, True, color)
        shadow = font.render(text, True, COLOR_SHADOW)
        rect = rendered.get_rect(**{anchor: position})
        surface.blit(shadow, rect.move(2, 2))
        surface.blit(rendered, rect)
        return rect

    def draw(self, surface, human, wheel_state, info, elapsed_s, mirrors=(), assist=None, goal_marker=None):
        width, height = surface.get_size()
        if goal_marker is not None:
            self._draw_goal_beacon(surface, human, goal_marker)
        for image, rect in mirrors:
            self._draw_mirror(surface, image, rect)
        assist_active = assist is not None and assist["mode"] != 0 and assist["reference_valid"]
        if info.camera_mode == CAMERA_MODE_DRIVER or assist_active:
            self._draw_wheel(surface, human, assist if assist_active else None, info, width, height, assist)
        self._draw_goal_arrow(surface, human)
        self._draw_info(surface, info, assist)
        self._draw_speed_limit(surface, human, width, elapsed_s)
        self._draw_minimap(surface, human, width, height)
        self._draw_panel(surface, human, wheel_state, height, assist if assist_active else None)
        self._text(surface, self.font_small, info.help_text, (width // 2, height - 8), anchor="midbottom")

    def _draw_info(self, surface, info, assist):
        level_color = COLOR_LEVEL.get(info.level_name, COLOR_TEXT)
        self._text(surface, self.font_label, f"Traffic: {info.level_name.upper()}", (PANEL_MARGIN_PX, INFO_TOP_PX),
                   color=level_color)
        ffb = "off" if info.ffb_gain is None else f"{round(100 * info.ffb_gain)}%"
        self._text(surface, self.font_label, f"Force feedback: {ffb}", (PANEL_MARGIN_PX, INFO_TOP_PX + INFO_LINE_PX))
        if assist is not None:
            self._text(surface, self.font_label, f"Assist: {ASSIST_MODE_NAMES[assist['mode']]}",
                       (PANEL_MARGIN_PX, INFO_TOP_PX + 2 * INFO_LINE_PX))

    def _draw_goal_beacon(self, surface, human, goal_marker):
        pygame = self.pygame
        base, top = goal_marker
        pygame.draw.line(surface, COLOR_GOAL_BEACON, base, top, GOAL_BEACON_WIDTH_PX)
        diamond = ((top[0], top[1] - GOAL_DIAMOND_PX), (top[0] + GOAL_DIAMOND_PX, top[1]),
                   (top[0], top[1] + GOAL_DIAMOND_PX), (top[0] - GOAL_DIAMOND_PX, top[1]))
        pygame.draw.polygon(surface, COLOR_GOAL_BEACON, diamond)
        distance_m = math.hypot(human["goal_x"] - human["sim_x"], human["goal_y"] - human["sim_y"])
        self._text(surface, self.font_label, f"GOAL  {distance_m:.0f} m", (top[0], top[1] - GOAL_DIAMOND_PX - 4),
                   color=COLOR_GOAL_BEACON, anchor="midbottom")

    def _wheel_sprite(self, radius):
        """The wheel drawn once at rest; each frame only rotates it."""
        cached = self._wheel_sprites.get(radius)
        if cached is not None:
            return cached
        pygame = self.pygame
        size = 2 * radius + 2 * WHEEL_SPRITE_PAD_PX
        sprite = pygame.Surface((size, size), pygame.SRCALPHA)
        center = (size / 2.0, size / 2.0)
        spoke_width = max(3, int(WHEEL_SPOKE_WIDTH_FRACTION * radius))
        for spoke_deg in WHEEL_SPOKE_ANGLES_DEG:
            end = clockwise_point(center, radius, spoke_deg)
            pygame.draw.line(sprite, COLOR_WHEEL_SPOKE_EDGE, center, end, spoke_width + 2)
            pygame.draw.line(sprite, COLOR_WHEEL_SPOKE, center, end, spoke_width)
        rim = max(3, int(WHEEL_RIM_WIDTH_FRACTION * radius))
        # leather shading: dark at both edges of the rim, lighter along its middle
        for band in range(rim):
            shade = 1.0 - abs(2.0 * band / max(rim - 1, 1) - 1.0)
            color = tuple(int(d + shade * (l - d)) for d, l in zip(COLOR_WHEEL_LEATHER_DARK, COLOR_WHEEL_LEATHER_LIGHT))
            pygame.draw.circle(sprite, color, center, radius - band, 2)
        grip = max(rim + 2, int(WHEEL_GRIP_WIDTH_FRACTION * radius))
        for grip_center_deg in (90.0, 270.0):
            pygame.draw.polygon(sprite, COLOR_WHEEL_LEATHER_DARK, ring_segment(
                center, radius + (grip - rim) / 2.0, radius - rim - (grip - rim) / 2.0,
                grip_center_deg - WHEEL_GRIP_HALF_SPAN_DEG, grip_center_deg + WHEEL_GRIP_HALF_SPAN_DEG))
        pygame.draw.polygon(sprite, COLOR_WHEEL_STRIPE, ring_segment(
            center, radius, radius - rim, -WHEEL_STRIPE_HALF_SPAN_DEG, WHEEL_STRIPE_HALF_SPAN_DEG))
        hub = pygame.Rect(0, 0, int(WHEEL_HUB_WIDTH_FRACTION * radius), int(WHEEL_HUB_HEIGHT_FRACTION * radius))
        hub.center = center
        pygame.draw.rect(sprite, COLOR_WHEEL_SPOKE_EDGE, hub.inflate(2, 2), border_radius=hub.height // 2)
        pygame.draw.rect(sprite, COLOR_WHEEL_HUB, hub, border_radius=hub.height // 2)
        pygame.draw.circle(sprite, COLOR_WHEEL_BADGE, center, max(2, int(WHEEL_BADGE_FRACTION * radius)), 2)
        self._wheel_sprites[radius] = sprite
        return sprite

    def _draw_wheel(self, surface, human, assist, info, width, height, assist_state=None):
        """Driver view: the steering wheel at your actual angle. Any view with assist on: a dial with the
        policy's angle as an arrow, plus your and the policy's angle and acceleration side by side."""
        from pufferlib.ocean.drive import binding

        pygame = self.pygame
        ratio, limit = binding.HUMAN_STEERING_RATIO, binding.HUMAN_STEERING_LIMIT_RAD
        is_driver_view = info.camera_mode == CAMERA_MODE_DRIVER
        radius = DRIVER_WHEEL_RADIUS_PX if is_driver_view else DIAL_WHEEL_RADIUS_PX
        center_x = int(DRIVER_WHEEL_CENTER_X_FRACTION * width) if is_driver_view else width // 2
        center = (center_x, height - WHEEL_BOTTOM_GAP_PX - radius)
        # the wheel in your hands: under BLEND the road wheels follow the blend, your hands do not
        hands_rad = assist_state["human_steering_rad"] if assist_state is not None else human["steering_angle_rad"]
        your_deg = wheel_display_deg(hands_rad, ratio, limit)
        rim = max(3, int(WHEEL_RIM_WIDTH_FRACTION * radius))
        # pygame rotates counter-clockwise for a positive angle; driver terms turn clockwise for positive
        rotated = pygame.transform.rotozoom(self._wheel_sprite(radius), -your_deg, 1.0)
        surface.blit(rotated, rotated.get_rect(center=center))
        if assist is None:
            return
        policy_deg = wheel_display_deg(assist["reference_steering_rad"], ratio, limit)
        head = ASSIST_ARROW_HEAD_FRACTION * radius
        tip = clockwise_point(center, radius + head, policy_deg)
        pygame.draw.line(surface, COLOR_ASSIST_POLICY, center, tip, max(2, rim // 2))
        pygame.draw.polygon(surface, COLOR_ASSIST_POLICY, (
            tip,
            clockwise_point(center, radius, policy_deg - math.degrees(head / radius)),
            clockwise_point(center, radius, policy_deg + math.degrees(head / radius)),
        ))
        readout = assist_readout(assist, human["accel_long_mps2"], ratio)
        top = center[1] - radius - head - 4
        self._text(surface, self.font_label, f"accel  you {human['accel_long_mps2']:+.1f}   policy "
                   f"{assist['reference_accel_mps2']:+.1f} m/s2", (center[0], top), color=COLOR_ASSIST_POLICY,
                   anchor="midbottom")
        self._text(surface, self.font_label, f"wheel  you {your_deg:+.0f} deg   policy {policy_deg:+.0f} deg"
                   f"   ({readout.steer_delta_wheel_deg:+.0f})", (center[0], top - LABEL_FONT_PX),
                   color=COLOR_ASSIST_POLICY, anchor="midbottom")

    def _draw_mirror(self, surface, image, rect):
        pygame = self.pygame
        pygame.draw.rect(surface, COLOR_MIRROR_FRAME, rect.inflate(8, 8), border_radius=10)
        surface.blit(image if image.get_size() == rect.size else pygame.transform.smoothscale(image, rect.size), rect)

    def _draw_goal_arrow(self, surface, human):
        pygame = self.pygame
        center = (PANEL_MARGIN_PX + GOAL_ARROW_RADIUS_PX, PANEL_MARGIN_PX + GOAL_ARROW_RADIUS_PX)
        bearing = goal_bearing_driver_rad(
            human["sim_x"], human["sim_y"], human["sim_heading"], human["goal_x"], human["goal_y"]
        )
        disc = pygame.Surface((2 * GOAL_ARROW_RADIUS_PX, 2 * GOAL_ARROW_RADIUS_PX), pygame.SRCALPHA)
        pygame.draw.circle(disc, COLOR_PANEL, (GOAL_ARROW_RADIUS_PX, GOAL_ARROW_RADIUS_PX), GOAL_ARROW_RADIUS_PX)
        surface.blit(disc, (center[0] - GOAL_ARROW_RADIUS_PX, center[1] - GOAL_ARROW_RADIUS_PX))
        # screen y grows downward, so "ahead" is -y and a right bearing moves the tip to +x
        tip = (center[0] + GOAL_ARROW_HEAD_PX * math.sin(bearing), center[1] - GOAL_ARROW_HEAD_PX * math.cos(bearing))
        left = (
            center[0] + 0.55 * GOAL_ARROW_HEAD_PX * math.sin(bearing - 2.5),
            center[1] - 0.55 * GOAL_ARROW_HEAD_PX * math.cos(bearing - 2.5),
        )
        right = (
            center[0] + 0.55 * GOAL_ARROW_HEAD_PX * math.sin(bearing + 2.5),
            center[1] - 0.55 * GOAL_ARROW_HEAD_PX * math.cos(bearing + 2.5),
        )
        pygame.draw.polygon(surface, COLOR_ARROW, (tip, left, center, right))
        distance_m = math.hypot(human["goal_x"] - human["sim_x"], human["goal_y"] - human["sim_y"])
        self._text(
            surface,
            self.font_label,
            f"goal {human['goal_idx'] + 1}  {distance_m:.0f} m",
            (center[0] + GOAL_ARROW_RADIUS_PX + 10, center[1]),
            anchor="midleft",
        )

    def _draw_speed_limit(self, surface, human, width, elapsed_s):
        pygame = self.pygame
        readout = speed_limit_readout(
            human["speed_mps"], human["lane_speed_limit_mps"], human["overspeed_tolerance_mps"]
        )
        center = (width - PANEL_MARGIN_PX - SPEED_SIGN_RADIUS_PX, PANEL_MARGIN_PX + SPEED_SIGN_RADIUS_PX)
        pygame.draw.circle(surface, COLOR_SIGN_RING, center, SPEED_SIGN_RADIUS_PX)
        pygame.draw.circle(surface, COLOR_SIGN_FACE, center, SPEED_SIGN_RADIUS_PX - SPEED_SIGN_RING_PX)
        label = self.font_sign.render(str(readout.limit_kmh), True, COLOR_SIGN_TEXT)
        surface.blit(label, label.get_rect(center=center))
        if readout.over_limit and int(elapsed_s * 2.0 * OVERSPEED_FLASH_HZ) % 2 == 0:
            self._text(
                surface,
                self.font_label,
                "OVER SPEED LIMIT",
                (center[0], center[1] + SPEED_SIGN_RADIUS_PX + 8),
                color=COLOR_WARN,
                anchor="midtop",
            )

    def _draw_minimap(self, surface, human, width, height):
        pygame = self.pygame
        origin = (width - PANEL_MARGIN_PX - MINIMAP_SIZE_PX, height - PANEL_MARGIN_PX - MINIMAP_SIZE_PX)
        surface.blit(self.minimap_surface, origin)
        goal_px = self.minimap.to_px(human["goal_x"], human["goal_y"])
        pygame.draw.circle(
            surface, COLOR_MINIMAP_GOAL, (origin[0] + goal_px[0], origin[1] + goal_px[1]), MINIMAP_GOAL_RADIUS_PX
        )
        ego_px = self.minimap.to_px(human["sim_x"], human["sim_y"])
        ego = (origin[0] + ego_px[0], origin[1] + ego_px[1])
        heading = human["sim_heading"]
        # bin +y is screen up, so the sim's CCW-positive heading maps to screen with y negated
        nose = (
            ego[0] + 2.2 * MINIMAP_EGO_RADIUS_PX * math.cos(heading),
            ego[1] - 2.2 * MINIMAP_EGO_RADIUS_PX * math.sin(heading),
        )
        pygame.draw.circle(surface, COLOR_MINIMAP_EGO, ego, MINIMAP_EGO_RADIUS_PX)
        pygame.draw.line(surface, COLOR_MINIMAP_EGO, ego, nose, 3)

    def _draw_panel(self, surface, human, wheel_state, height, assist=None):
        from pufferlib.ocean.drive import binding

        pygame = self.pygame
        targets = None
        if assist is not None:
            targets = policy_pedal_targets(
                assist["reference_accel_mps2"], binding.HUMAN_ACCEL_MIN_MPS2, binding.HUMAN_ACCEL_MAX_MPS2
            )
        panel_height = PEDAL_BAR_HEIGHT_PX + 2 * PANEL_MARGIN_PX
        panel_width = 330
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill(COLOR_PANEL)
        top = height - PANEL_MARGIN_PX - panel_height
        surface.blit(panel, (PANEL_MARGIN_PX, top))
        left = 2 * PANEL_MARGIN_PX
        speed_kmh = abs(human["speed_mps"]) * MPS_TO_KMH
        readout = speed_limit_readout(
            human["speed_mps"], human["lane_speed_limit_mps"], human["overspeed_tolerance_mps"]
        )
        speed_color = COLOR_WARN if readout.over_limit else COLOR_TEXT
        self._text(surface, self.font_speed, f"{speed_kmh:3.0f}", (left, top + PANEL_MARGIN_PX), color=speed_color)
        self._text(surface, self.font_label, "km/h", (left + 2, top + PANEL_MARGIN_PX + SPEED_FONT_PX - 8))
        gear = gear_label(human["gear"], human["gearbox_mode"])
        self._text(surface, self.font_speed, gear, (left + 120, top + PANEL_MARGIN_PX))
        self._text(surface, self.font_label, "gear", (left + 122, top + PANEL_MARGIN_PX + SPEED_FONT_PX - 8))
        flags = [name for name, key in (("COLLISION", "collision"), ("OFF ROAD", "offroad")) if human[key] > 0.0]
        if human["engine_stalled"]:
            flags.append("STALLED - press R")
        if policy_drives_speed(
            assist, wheel_state, human["gear"], binding.ASSIST_MODE_FULL, binding.ASSIST_PEDAL_RELEASED_FRACTION
        ):
            flags.append("POLICY DRIVING SPEED")
        if flags:
            self._text(
                surface,
                self.font_label,
                "  ".join(flags),
                (left, top + panel_height - PANEL_MARGIN_PX - 6),
                color=COLOR_WARN,
                anchor="bottomleft",
            )
        bars_left = PANEL_MARGIN_PX + panel_width - 2 * (PEDAL_BAR_WIDTH_PX + PEDAL_BAR_GAP_PX) - 4
        for idx, (label, value, color) in enumerate(
            (("THR", wheel_state.throttle_norm, COLOR_THROTTLE), ("BRK", wheel_state.brake_norm, COLOR_BRAKE))
        ):
            x = bars_left + idx * (PEDAL_BAR_WIDTH_PX + PEDAL_BAR_GAP_PX)
            bar_top = top + PANEL_MARGIN_PX
            pygame.draw.rect(surface, COLOR_BAR_TRACK, (x, bar_top, PEDAL_BAR_WIDTH_PX, PEDAL_BAR_HEIGHT_PX - 18))
            fill = int(pedal_fraction(value) * (PEDAL_BAR_HEIGHT_PX - 18))
            pygame.draw.rect(surface, color, (x, bar_top + PEDAL_BAR_HEIGHT_PX - 18 - fill, PEDAL_BAR_WIDTH_PX, fill))
            if targets is not None:
                marker_y = bar_top + PEDAL_BAR_HEIGHT_PX - 18 - int(targets[idx] * (PEDAL_BAR_HEIGHT_PX - 18))
                pygame.draw.line(surface, COLOR_ASSIST_POLICY, (x - PEDAL_MARKER_OVERHANG_PX, marker_y),
                                 (x + PEDAL_BAR_WIDTH_PX + PEDAL_MARKER_OVERHANG_PX, marker_y),
                                 PEDAL_MARKER_THICKNESS_PX)
            percent = f"{round(100 * pedal_fraction(value))}"
            self._text(
                surface, self.font_small, percent, (x + PEDAL_BAR_WIDTH_PX // 2, bar_top - 2), anchor="midbottom"
            )
            self._text(
                surface,
                self.font_small,
                label,
                (x + PEDAL_BAR_WIDTH_PX // 2, bar_top + PEDAL_BAR_HEIGHT_PX - 14),
                anchor="midtop",
            )
