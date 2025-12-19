"""Pure Python/NumPy implementation of CLASSIC and JERK dynamics models.

These implementations follow the C++ code in pufferlib/ocean/drive/drive.h
to ensure consistency with the simulation environment.
"""

import numpy as np
from .trajectory_utils import wrap_angle


class ClassicDynamics:
    """
    Kinematic bicycle model with direct acceleration and steering control.

    Action space: 91 discrete actions (7 acceleration × 13 steering)
    State: {x, y, heading, vx, vy, length}

    Based on drive.h lines 1558-1618
    """

    # Action space constants (from drive.h lines 99-100)
    ACCELERATION_VALUES = np.array([-4.0, -2.667, -1.333, 0.0, 1.333, 2.667, 4.0])
    STEERING_VALUES = np.array([-1.0, -0.833, -0.667, -0.5, -0.333, -0.167, 0.0,
                                 0.167, 0.333, 0.5, 0.667, 0.833, 1.0])

    def __init__(self, dt=0.1):
        """
        Initialize CLASSIC dynamics model.

        Args:
            dt: Time step in seconds (default 0.1s = 10 Hz)
        """
        self.dt = dt
        self.num_accel = len(self.ACCELERATION_VALUES)
        self.num_steer = len(self.STEERING_VALUES)
        self.num_actions = self.num_accel * self.num_steer  # 7 * 13 = 91

    def step(self, state, action_idx):
        """
        Step the dynamics forward by one timestep.

        Args:
            state: dict with {x, y, heading, vx, vy, length}
            action_idx: int in [0, 90] (discrete action index)
        Returns:
            next_state: dict with updated {x, y, heading, vx, vy, length}
        """
        # Decode action (drive.h lines 1576-1579)
        acceleration_index = action_idx // self.num_steer
        steering_index = action_idx % self.num_steer
        acceleration = self.ACCELERATION_VALUES[acceleration_index]
        steering = self.STEERING_VALUES[steering_index]

        # Current state (drive.h lines 1583-1587)
        x = state['x']
        y = state['y']
        heading = state['heading']
        vx = state['vx']
        vy = state['vy']
        length = state['length']

        # Calculate current speed (drive.h line 1590)
        speed = np.sqrt(vx**2 + vy**2)

        # Update speed with acceleration (drive.h lines 1593-1594)
        speed_new = speed + acceleration * self.dt
        speed_new = np.clip(speed_new, -100.0, 100.0)  # clipSpeed

        # Compute beta and yaw rate (drive.h lines 1597-1600)
        beta = np.tanh(0.5 * np.tan(steering))
        yaw_rate = (speed * np.cos(beta) * np.tan(steering)) / length

        # New velocity components (drive.h lines 1603-1604)
        vx_new = speed_new * np.cos(heading + beta)
        vy_new = speed_new * np.sin(heading + beta)

        # Update position and heading (drive.h lines 1607-1609)
        x_new = x + vx_new * self.dt
        y_new = y + vy_new * self.dt
        heading_new = heading + yaw_rate * self.dt

        # Normalize heading to [-pi, pi]
        heading_new = wrap_angle(heading_new)

        return {
            'x': x_new,
            'y': y_new,
            'heading': heading_new,
            'vx': vx_new,
            'vy': vy_new,
            'length': length,
        }


class JerkDynamics:
    """
    Stateful jerk bicycle model with persistent acceleration state.

    Action space: Configurable discrete actions (jerk_long × jerk_lat)
    State: {x, y, heading, vx, vy, a_long, a_lat, steering_angle, wheelbase}

    Based on drive.h lines 1620-1714
    """

    # Default action space constants (from drive.h lines 95-96)
    DEFAULT_JERK_LONG = np.array([-15.0, -4.0, 0.0, 4.0])
    DEFAULT_JERK_LAT = np.array([-4.0, 0.0, 4.0])

    def __init__(self, dt=0.1, jerk_long=None, jerk_lat=None):
        """
        Initialize JERK dynamics model.

        Args:
            dt: Time step in seconds (default 0.1s = 10 Hz)
            jerk_long: Longitudinal jerk discretization values (default: [-15, -4, 0, 4])
            jerk_lat: Lateral jerk discretization values (default: [-4, 0, 4])
        """
        self.dt = dt
        self.JERK_LONG = np.array(jerk_long) if jerk_long is not None else self.DEFAULT_JERK_LONG
        self.JERK_LAT = np.array(jerk_lat) if jerk_lat is not None else self.DEFAULT_JERK_LAT
        self.num_jerk_long = len(self.JERK_LONG)
        self.num_jerk_lat = len(self.JERK_LAT)
        self.num_actions = self.num_jerk_long * self.num_jerk_lat

    def step(self, state, action_idx):
        """
        Step the dynamics forward by one timestep.

        Args:
            state: dict with {x, y, heading, vx, vy, a_long, a_lat,
                             steering_angle, wheelbase}
            action_idx: int in [0, 11] (discrete action index)
        Returns:
            next_state: dict with updated state including accelerations
        """
        # Decode action (drive.h lines 1638-1642)
        # Note: In the C++ code, actions are stored as [long_idx, lat_idx] pairs
        # But for discrete actions encoded as single int:
        jerk_long_idx = action_idx // self.num_jerk_lat
        jerk_lat_idx = action_idx % self.num_jerk_lat
        a_long_jerk = self.JERK_LONG[jerk_long_idx]
        a_lat_jerk = self.JERK_LAT[jerk_lat_idx]

        # Current state
        x = state['x']
        y = state['y']
        heading = state['heading']
        vx = state['vx']
        vy = state['vy']
        a_long = state['a_long']
        a_lat = state['a_lat']
        steering_angle = state['steering_angle']
        wheelbase = state['wheelbase']

        # Heading unit vector
        heading_x = np.cos(heading)
        heading_y = np.sin(heading)

        # Calculate new acceleration (drive.h lines 1646-1660)
        a_long_new = a_long + a_long_jerk * self.dt
        a_lat_new = a_lat + a_lat_jerk * self.dt

        # Make it easy to stop with 0 accel (sign-change detection)
        if a_long * a_long_new < 0:
            a_long_new = 0.0
        else:
            a_long_new = np.clip(a_long_new, -5.0, 2.5)

        if a_lat * a_lat_new < 0:
            a_lat_new = 0.0
        else:
            a_lat_new = np.clip(a_lat_new, -4.0, 4.0)

        # Calculate new velocity (drive.h lines 1663-1672)
        v_dot_heading = vx * heading_x + vy * heading_y
        speed = np.sqrt(vx**2 + vy**2)
        signed_v = np.copysign(speed, v_dot_heading)

        # Trapezoidal integration
        v_new = signed_v + 0.5 * (a_long_new + a_long) * self.dt

        # Make it easy to stop with 0 vel (sign-change detection)
        if signed_v * v_new < 0:
            v_new = 0.0
        else:
            v_new = np.clip(v_new, -2.0, 20.0)

        # Calculate new steering angle (drive.h lines 1675-1683)
        signed_curvature = a_lat_new / max(v_new * v_new, 1e-5)
        # Ensure minimum curvature magnitude (drive.h line 1676)
        signed_curvature = np.copysign(max(abs(signed_curvature), 1e-5), signed_curvature)

        steering_angle_target = np.arctan(signed_curvature * wheelbase)

        # Rate-limited steering (max 0.6 rad/s)
        delta_steer = np.clip(
            steering_angle_target - steering_angle,
            -0.6 * self.dt,
            0.6 * self.dt
        )
        steering_angle_new = np.clip(
            steering_angle + delta_steer,
            -0.55,
            0.55
        )

        # Update curvature and accel to account for limited steering
        signed_curvature = np.tan(steering_angle_new) / wheelbase
        a_lat_new = v_new * v_new * signed_curvature

        # Calculate resulting movement using bicycle dynamics (drive.h lines 1686-1700)
        d = 0.5 * (v_new + signed_v) * self.dt  # Average displacement
        theta = d * signed_curvature  # Rotation angle

        if abs(signed_curvature) < 1e-5 or abs(theta) < 1e-5:
            # Straight line motion
            dx_local = d
            dy_local = 0.0
        else:
            # Curved motion
            dx_local = np.sin(theta) / signed_curvature
            dy_local = (1.0 - np.cos(theta)) / signed_curvature

        # Transform to global coordinates
        dx = dx_local * heading_x - dy_local * heading_y
        dy = dx_local * heading_y + dy_local * heading_x

        # Update everything (drive.h lines 1702-1713)
        x_new = x + dx
        y_new = y + dy
        heading_new = wrap_angle(heading + theta)

        # Update velocity components
        vx_new = v_new * np.cos(heading_new)
        vy_new = v_new * np.sin(heading_new)

        # Compute jerk (for diagnostics)
        jerk_long = (a_long_new - a_long) / self.dt
        jerk_lat = (a_lat_new - a_lat) / self.dt

        return {
            'x': x_new,
            'y': y_new,
            'heading': heading_new,
            'vx': vx_new,
            'vy': vy_new,
            'a_long': a_long_new,
            'a_lat': a_lat_new,
            'steering_angle': steering_angle_new,
            'wheelbase': wheelbase,
            'jerk_long': jerk_long,
            'jerk_lat': jerk_lat,
        }
