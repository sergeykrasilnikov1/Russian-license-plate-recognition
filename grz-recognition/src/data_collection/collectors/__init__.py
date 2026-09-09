"""Per-source collectors. Each one is optional and degrades gracefully.

A missing SDK, absent credentials or an unreachable host must never abort the
run: the collector reports itself unavailable and the orchestrator records the
reason in the Stage 2 report.
"""

from .base import Collector, CollectorUnavailable, RawItem, YoloBox
from .commons import WikimediaCommonsCollector
from .huggingface import HuggingFaceCollector
from .kaggle_ds import KaggleCollector
from .openverse import OpenverseCollector
from .platesmania import PlatesManiaCollector
from .roboflow_universe import RoboflowCollector

COLLECTOR_REGISTRY: dict[str, type[Collector]] = {
    "huggingface": HuggingFaceCollector,
    "commons": WikimediaCommonsCollector,
    "openverse": OpenverseCollector,
    "roboflow": RoboflowCollector,
    "kaggle": KaggleCollector,
    "platesmania": PlatesManiaCollector,
}

__all__ = [
    "Collector",
    "CollectorUnavailable",
    "RawItem",
    "YoloBox",
    "COLLECTOR_REGISTRY",
    "HuggingFaceCollector",
    "WikimediaCommonsCollector",
    "OpenverseCollector",
    "RoboflowCollector",
    "KaggleCollector",
    "PlatesManiaCollector",
]
