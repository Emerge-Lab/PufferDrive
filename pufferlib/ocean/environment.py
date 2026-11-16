import importlib
import pufferlib

OCEAN_MODULE_BY_ENV_NAME = {"opengigaflow": "drive"}
MAKE_FUNCTIONS = {"drive": "Drive"}


def env_creator(name="squared", *args, **kwargs):
    if name not in OCEAN_MODULE_BY_ENV_NAME:
        raise pufferlib.APIUsageError(f"Invalid environment name: {name}")

    # TODO: Robust sanity / ocean imports
    name = OCEAN_MODULE_BY_ENV_NAME[name]
    try:
        module = importlib.import_module(f"pufferlib.ocean.{name}.{name}")
        return getattr(module, MAKE_FUNCTIONS[name])
    except ModuleNotFoundError:
        return MAKE_FUNCTIONS[name]
