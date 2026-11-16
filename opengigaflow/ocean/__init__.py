import importlib.util

if importlib.util.find_spec("torch") is not None:
    from . import torch  # noqa: F401
