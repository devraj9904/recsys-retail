"""Real-time two-stage recommender on RetailRocket."""

import os
import sys

if sys.platform == "darwin":
    # PyTorch, faiss-cpu and LightGBM can each load their own OpenMP runtime on
    # macOS; allow the duplicates instead of aborting. Set here so it applies
    # before any of those libraries is imported through this package.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
