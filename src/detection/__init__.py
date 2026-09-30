"""Plate detection (YOLOv11n → ONNX). Classes encode plate type via PLATE_TYPES."""

from .split import (
    ImageRecord,
    load_image_records,
    stratified_split,
    summarize_split,
    write_data_yaml,
    write_split_lists,
)
from .vehicle import VehicleFilterConfig, is_on_vehicle, vehicle_context_score

__all__ = [
    "ImageRecord",
    "VehicleFilterConfig",
    "is_on_vehicle",
    "load_image_records",
    "stratified_split",
    "summarize_split",
    "vehicle_context_score",
    "write_data_yaml",
    "write_split_lists",
]
