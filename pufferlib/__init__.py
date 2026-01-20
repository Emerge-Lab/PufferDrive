__version__ = 3.0

import os

# Silence noisy dependencies
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Silence noisy packages
import sys

original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")
try:
    import gymnasium
    import pygame
except ImportError:
    pass
sys.stdout.close()
sys.stderr.close()
sys.stdout = original_stdout
sys.stderr = original_stderr

from pufferlib.pufferlib import *
