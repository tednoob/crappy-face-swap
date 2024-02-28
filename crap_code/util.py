import sys
import logging
import onnxruntime
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Frame = np.ndarray[Any, Any]
PROVIDERS = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
HAS_CUDA = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
HAS_METAL = "CoreMLExecutionProvider" in onnxruntime.get_available_providers()
HAS_GPU = HAS_CUDA or HAS_METAL


def normalize_path(path: str) -> str:
    """Converts a Windows path to a WSL path if necessary."""
    if ":" in path:
        parts = path.split(":", 1)
        drive = parts[0].lower()
        drive_path = parts[1].replace("\\", "/")
        path = f"/mnt/{drive}{drive_path}"
    return path


def is_image(file: str) -> bool:
    """Checks if a file is an image."""
    return file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"))


def is_video(file: str) -> bool:
    """Checks if a file is a video."""
    return file.lower().endswith(
        (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm")
    )
