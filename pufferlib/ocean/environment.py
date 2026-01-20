import importlib
import pufferlib

MAKE_FUNCTIONS = {"drive": "Drive"}


def env_creator(name="squared", *args, **kwargs):
    if "puffer_" not in name:
        raise pufferlib.APIUsageError(f"Invalid environment name: {name}")

    # TODO: Robust sanity / ocean imports
    name = name.replace("puffer_", "")
    try:
        module = importlib.import_module(f"pufferlib.ocean.{name}.{name}")
        return getattr(module, MAKE_FUNCTIONS[name])
    except ModuleNotFoundError:
        return MAKE_FUNCTIONS[name]
