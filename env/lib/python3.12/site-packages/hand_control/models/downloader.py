"""MediaPipe model downloader utility."""

import urllib.request
from pathlib import Path

from hand_control.config import config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def download_model(model_path: Path | None = None) -> Path:
    """Download the hand landmarker model if not present.

    Args:
        model_path: Path to save the model. Defaults to config value.

    Returns:
        Path to the model file.
    """
    path = model_path or config.model_path

    if not path.exists():
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Model downloaded!")

    return path
