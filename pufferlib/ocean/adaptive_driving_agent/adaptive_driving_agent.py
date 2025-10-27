from pufferlib.ocean.drive import Drive
import pufferlib

class AdaptiveDrivingAgent(Drive):
    def __init__(self, **kwargs):
        self.env_name = "adaptive_driving_agent"
        self.k_scenarios = kwargs["k_scenarios"]
        self.scenario_length = kwargs["scenario_length"]
        self.episode_length= kwargs["resample_frequency"] =self.k_scenarios * self.scenario_length
        print(f"DEBUG: kwargs are {kwargs}", flush= True)
        super().__init__(**kwargs)

