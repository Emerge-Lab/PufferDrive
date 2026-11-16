__version__ = 3.0

# Silence noisy dependencies
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from opengigaflow.core import *
