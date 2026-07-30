"""Unified evaluator framework for PufferDrive.

Each Evaluator subclass owns one rollout pattern. The EvalManager (parent
package) discovers evaluators from `eval.<name>` sections in puffer_drive.yaml
and dispatches them inline (during training) or as subprocesses.

See docs/eval_unification.md for the full design rationale.
"""

from pufferlib.ocean.benchmark.evaluators.base import EvalResult, Evaluator
from pufferlib.ocean.benchmark.evaluators.behavior_class import BehaviorClassEvaluator
from pufferlib.ocean.benchmark.evaluators.human_replay import HumanReplayEvaluator
from pufferlib.ocean.benchmark.evaluators.multi_scenario import MultiScenarioEvaluator
from pufferlib.ocean.benchmark.evaluators.wosac import WOSACEvaluator

# Type registry for [eval.<name>].type → class lookup. Manager uses this
# to instantiate the right subclass per config section.
EVALUATOR_REGISTRY = {
    "multi_scenario": MultiScenarioEvaluator,
    "behavior_class": BehaviorClassEvaluator,
    "human_replay": HumanReplayEvaluator,
    "wosac": WOSACEvaluator,
}

__all__ = [
    "EVALUATOR_REGISTRY",
    "EvalResult",
    "Evaluator",
    "MultiScenarioEvaluator",
    "BehaviorClassEvaluator",
    "HumanReplayEvaluator",
    "WOSACEvaluator",
]
