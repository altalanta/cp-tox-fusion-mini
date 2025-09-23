"""cp-tox-mini: Cell Painting × toxicity fusion workflow.

A lightweight, reproducible pipeline for fusing morphological profiles 
with chemical descriptors for toxicity prediction.
"""

__version__ = "0.1.0"

# Set deterministic behavior
import os
import random
import numpy as np

# Set environment variable for Python hash randomization
os.environ['PYTHONHASHSEED'] = '42'

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
except ImportError:
    pass  # torch not available