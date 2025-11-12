from pufferlib.ocean.drive import Drive
import pufferlib


class AdaptiveDrivingAgent(Drive):
    def __init__(self, **kwargs):
        self.env_name = "adaptive_drive"
        self.k_scenarios = kwargs["k_scenarios"]
        self.scenario_length = kwargs["scenario_length"]
        self.dynamics_model = kwargs["dynamics_model"]

        kwargs["ini_file"] = "pufferlib/config/ocean/adaptive.ini"
        kwargs["adaptive_driving_agent"] = True

        kwargs["resample_frequency"] = self.k_scenarios * self.scenario_length
        self.episode_length = kwargs["resample_frequency"]
        # print(f"resample frequency is ", kwargs["resample_frequency"], flush=True)
        super().__init__(**kwargs)
