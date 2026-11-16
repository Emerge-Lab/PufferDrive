"""Remaining-yellow timing for co-sim shadow envs (obs_traffic_light_yellow_remaining)."""


class YellowCountdown:
    """Remaining yellow seconds for lights whose simulator exposes no usable phase timer: the shortest
    trained yellow (traffic_light_yellow_min_s) counted down from the first tick the light was seen
    yellow, never below one env step (a yellow light always has the current step left)."""

    def __init__(self, assumed_total_s, min_remaining_s):
        if not assumed_total_s > 0.0 or not min_remaining_s > 0.0:
            raise ValueError(f"YellowCountdown needs positive durations. Got: {assumed_total_s}, {min_remaining_s}")
        self.assumed_total_s = float(assumed_total_s)
        self.min_remaining_s = float(min_remaining_s)
        self._onset_s = {}

    def tick(self, yellow_keys, now_s):
        """-> {key: remaining_s} for the lights yellow at now_s; a light absent from a tick restarts its
        countdown when it is next seen yellow."""
        now_s = float(now_s)
        self._onset_s = {key: self._onset_s.get(key, now_s) for key in yellow_keys}
        return {
            key: max(self.assumed_total_s - (now_s - onset_s), self.min_remaining_s)
            for key, onset_s in self._onset_s.items()
        }
