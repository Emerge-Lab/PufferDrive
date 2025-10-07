import numpy as np
import gymnasium
from gymnasium import spaces
import pufferlib
from pufferlib.ocean.diplomacy import binding


class Diplomacy(pufferlib.PufferEnv):
    """
    Welfare Diplomacy environment for PufferLib.

    A multi-agent general-sum variant of the board game Diplomacy where:
    - Powers can voluntarily disband units
    - Welfare points = cumulative (centers - units) after each adjustment phase
    - Game ends after fixed number of years (no victory condition)
    - Final utility = total welfare points accumulated

    This is a C/C++ implementation for high-performance RL training.
    """

    def __init__(
        self,
        max_years=10,
        welfare_mode=True,
        num_players=7,
        render_mode=None,
        buf=None,
        seed=1,
    ):
        self.max_years = max_years
        self.welfare_mode = welfare_mode
        self.num_players = num_players
        self.render_mode = render_mode
        self.seed = seed

        # Define single observation space (required by PufferEnv)
        # TODO: Expand this to proper structured observation later
        # For now, simple flat vector: 76 board + 76 units + 7 centers + 7 units_count + 7 welfare + 1 phase + 1 year
        # Total: 175 features
        self.single_observation_space = spaces.Box(
            low=0, high=2100, shape=(175,), dtype=np.float32
        )

        # Define action space
        # For now, simplified discrete action space
        # Will expand to full order space later
        self.single_action_space = spaces.Discrete(1000)

        # Number of agents (required by PufferEnv)
        self.num_agents = num_players

        # For vectorized environments
        if buf is not None:
            self.envs = buf.envs
        else:
            self.envs = 1

        # Now call parent init (sets up buffers)
        super().__init__(buf)

        # Initialize C environment handle
        self.env_handle = binding.env_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            seed
        )

        # Configure runtime settings in C
        try:
            binding.env_configure(self.env_handle, int(self.welfare_mode), int(self.max_years))
        except AttributeError:
            # Backward compatibility if binding not yet exposes configure
            pass

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed = seed

        # Call C reset function
        binding.env_reset(self.env_handle, self.seed)

        # Observations are updated in self.observations by C code
        # Return empty info list (one per agent)
        return self.observations, []

    def step(self, actions):
        """Execute one step of the environment."""
        # Actions are already in self.actions (set by PufferLib)
        # Call C step function which will process them
        binding.env_step(self.env_handle)

        # All results are updated in place in the buffers by C code
        # Return empty info list (one per agent)
        return self.observations, self.rewards, self.terminals, self.truncations, []

    def render(self):
        """Render the current game state."""
        if self.render_mode == "human":
            binding.env_render(self.env_handle)

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'env_handle'):
            binding.env_close(self.env_handle)
