__version__ = "1.0.0"

# Silence noisy dependencies
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from opengigaflow.core import *
