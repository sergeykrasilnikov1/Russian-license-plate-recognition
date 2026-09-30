"""LPR: recognition of Russian registration plates of types 1, 1A (square) and 1B (yellow).

Importing this package puts Ultralytics into a strictly offline mode: no version checks, no analytics,
no automatic downloads or pip installs. The task requires the solution to run without internet access,
so this is enforced in code rather than left to the environment.
"""
import os

os.environ.setdefault("YOLO_OFFLINE", "1")        # is_online() -> False: disables sync/analytics/downloads
os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install anything at runtime
os.environ.setdefault("YOLO_VERBOSE", "false")

__version__ = "1.0.0"
