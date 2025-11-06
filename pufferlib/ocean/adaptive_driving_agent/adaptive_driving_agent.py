from pufferlib.ocean.drive import Drive
import pufferlib


class AdaptiveDrivingAgent(Drive):
    def __init__(self, **kwargs):
        self.env_name = "adaptive_driving_agent"
        self.k_scenarios = kwargs["k_scenarios"]
        self.scenario_length = kwargs["scenario_length"]
        kwargs["ini_file"] = "pufferlib/config/ocean/adaptive_driving_agent.ini"
        kwargs["adaptive_driving_agent"] = True
        self.dynamics_model = kwargs["dynamics_model"]
        resample_frequency = self.k_scenarios * self.scenario_length
        kwargs["resample_frequency"] = resample_frequency
        self.episode_length = resample_frequency
        print(f"resample frequency is ", kwargs["resample_frequency"], flush=True)
        super().__init__(**kwargs)
