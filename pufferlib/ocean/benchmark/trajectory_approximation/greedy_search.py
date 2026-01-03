"""Greedy action selection for trajectory approximation."""

import numpy as np
from .trajectory_utils import wrap_angle
from .dynamics import ClassicDynamics, JerkDynamics


class GreedyActionSelector:
    """
    Greedy action selection for trajectory approximation.

    At each timestep, evaluates all possible actions and selects the one
    that minimizes the weighted error to the next ground truth state.
    """

    def __init__(self, dynamics_model, weights=None):
        """
        Initialize greedy action selector.

        Args:
            dynamics_model: ClassicDynamics or JerkDynamics instance
            weights: Dict with weights for pos, vel, heading errors
                    Default: {'pos': 1.0, 'vel': 1.0, 'heading': 10.0}
        """
        self.dynamics = dynamics_model
        self.num_actions = dynamics_model.num_actions

        if weights is None:
            # Default weights - heading gets higher weight because it's in radians
            self.weights = {'pos': 1.0, 'vel': 1.0, 'heading': 10.0}
        else:
            self.weights = weights

    def compute_error(self, predicted_state, target_state):
        """
        Compute weighted error between predicted and target states.

        Args:
            predicted_state: Dict with {x, y, heading, vx, vy, ...}
            target_state: Dict with {x, y, heading, vx, vy, ...}
        Returns:
            float: Weighted total error
        """
        # Position error (Euclidean distance)
        pos_error = np.sqrt(
            (predicted_state['x'] - target_state['x'])**2 +
            (predicted_state['y'] - target_state['y'])**2
        )

        # Velocity error (Euclidean distance in velocity space)
        vel_error = np.sqrt(
            (predicted_state['vx'] - target_state['vx'])**2 +
            (predicted_state['vy'] - target_state['vy'])**2
        )

        # Heading error (angular distance, wrapped to [-pi, pi])
        heading_error = abs(wrap_angle(
            predicted_state['heading'] - target_state['heading']
        ))

        # Weighted sum
        total_error = (
            self.weights['pos'] * pos_error +
            self.weights['vel'] * vel_error +
            self.weights['heading'] * heading_error
        )

        return total_error

    def select_best_action(self, current_state, target_state):
        """
        Test all possible actions and return the one with minimum error.

        Args:
            current_state: Current state dict
            target_state: Desired next state from ground truth
        Returns:
            tuple: (best_action_idx, best_state, min_error)
        """
        min_error = float('inf')
        best_action_idx = 0
        best_state = None

        for action_idx in range(self.num_actions):
            # Simulate this action
            predicted_state = self.dynamics.step(current_state, action_idx)

            # Compute error
            error = self.compute_error(predicted_state, target_state)

            # Update best if this is better
            if error < min_error:
                min_error = error
                best_action_idx = action_idx
                best_state = predicted_state

        return best_action_idx, best_state, min_error


def approximate_trajectory(dynamics_model, ground_truth, initial_state, weights=None):
    """
    Approximate a trajectory using greedy search.

    Args:
        dynamics_model: ClassicDynamics or JerkDynamics instance
        ground_truth: Dict with arrays {x, y, heading, vx, vy, valid, ...}
        initial_state: Initial state dict
        weights: Optional weights for error computation
    Returns:
        tuple: (approximated_trajectory, action_sequence, per_step_errors)
    """
    selector = GreedyActionSelector(dynamics_model, weights=weights)
    num_steps = len(ground_truth['x'])

    # Initialize storage
    traj = {
        'x': np.zeros(num_steps, dtype=np.float32),
        'y': np.zeros(num_steps, dtype=np.float32),
        'heading': np.zeros(num_steps, dtype=np.float32),
        'vx': np.zeros(num_steps, dtype=np.float32),
        'vy': np.zeros(num_steps, dtype=np.float32),
    }

    # Additional fields for JERK model
    if isinstance(dynamics_model, JerkDynamics):
        traj['a_long'] = np.zeros(num_steps, dtype=np.float32)
        traj['a_lat'] = np.zeros(num_steps, dtype=np.float32)
        traj['steering_angle'] = np.zeros(num_steps, dtype=np.float32)

    actions = np.zeros(num_steps - 1, dtype=np.int32)
    errors = np.zeros(num_steps - 1, dtype=np.float32)

    # Set initial state
    current_state = initial_state.copy()
    traj['x'][0] = current_state['x']
    traj['y'][0] = current_state['y']
    traj['heading'][0] = current_state['heading']
    traj['vx'][0] = current_state['vx']
    traj['vy'][0] = current_state['vy']

    if isinstance(dynamics_model, JerkDynamics):
        traj['a_long'][0] = current_state['a_long']
        traj['a_lat'][0] = current_state['a_lat']
        traj['steering_angle'][0] = current_state['steering_angle']

    # Greedy search at each timestep
    for t in range(num_steps - 1):
        # Skip if next timestep is invalid
        if not ground_truth['valid'][t + 1]:
            break

        # Target state from ground truth
        target_state = {
            'x': ground_truth['x'][t + 1],
            'y': ground_truth['y'][t + 1],
            'heading': ground_truth['heading'][t + 1],
            'vx': ground_truth['vx'][t + 1],
            'vy': ground_truth['vy'][t + 1],
        }

        # Select best action
        action_idx, next_state, error = selector.select_best_action(
            current_state, target_state
        )

        # Record action and error
        actions[t] = action_idx
        errors[t] = error

        # Update trajectory
        traj['x'][t + 1] = next_state['x']
        traj['y'][t + 1] = next_state['y']
        traj['heading'][t + 1] = next_state['heading']
        traj['vx'][t + 1] = next_state['vx']
        traj['vy'][t + 1] = next_state['vy']

        if isinstance(dynamics_model, JerkDynamics):
            traj['a_long'][t + 1] = next_state['a_long']
            traj['a_lat'][t + 1] = next_state['a_lat']
            traj['steering_angle'][t + 1] = next_state['steering_angle']

        # Update current state
        current_state = next_state

    return traj, actions, errors


def decode_action(dynamics_model, action_idx):
    """
    Decode action index into human-readable components.

    Args:
        dynamics_model: ClassicDynamics or JerkDynamics instance
        action_idx: Action index to decode
    Returns:
        Dict with action components
    """
    if isinstance(dynamics_model, ClassicDynamics):
        accel_idx = action_idx // dynamics_model.num_steer
        steer_idx = action_idx % dynamics_model.num_steer
        return {
            'type': 'classic',
            'acceleration': dynamics_model.ACCELERATION_VALUES[accel_idx],
            'steering': dynamics_model.STEERING_VALUES[steer_idx],
        }
    elif isinstance(dynamics_model, JerkDynamics):
        jerk_long_idx = action_idx // dynamics_model.num_jerk_lat
        jerk_lat_idx = action_idx % dynamics_model.num_jerk_lat
        return {
            'type': 'jerk',
            'jerk_long': dynamics_model.JERK_LONG[jerk_long_idx],
            'jerk_lat': dynamics_model.JERK_LAT[jerk_lat_idx],
        }
    else:
        return {'type': 'unknown'}
