"""Type definitions for the hand control system."""

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

import numpy as np
import numpy.typing as npt

# Type aliases
FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]
ServoAngle = int  # 0-128

# Finger indices for landmark lookup
FINGER_NAMES: tuple[FingerName, ...] = ("thumb", "index", "middle", "ring", "pinky")


class FingerAngles(TypedDict):
    """Dictionary mapping finger names to servo angles."""

    thumb: ServoAngle
    index: ServoAngle
    middle: ServoAngle
    ring: ServoAngle
    pinky: ServoAngle


class FingerRatios(TypedDict):
    """Dictionary mapping finger names to distance ratios."""

    thumb: float
    index: float
    middle: float
    ring: float
    pinky: float


class LandmarkPoint(Protocol):
    """Protocol for MediaPipe landmark points."""

    @property
    def x(self) -> float:
        """X coordinate (normalized 0-1)."""
        ...

    @property
    def y(self) -> float:
        """Y coordinate (normalized 0-1)."""
        ...

    @property
    def z(self) -> float:
        """Z coordinate (depth)."""
        ...


@dataclass
class MutableLandmark:
    """Mutable landmark point satisfying LandmarkPoint protocol.

    Used to inject real depth values from the RealSense camera into
    MediaPipe landmarks, replacing the estimated z coordinate with
    actual metric depth in meters.
    """

    x: float
    y: float
    z: float


# NumPy array type aliases
Vector3D = npt.NDArray[np.float64]
Vector2D = npt.NDArray[np.float64]
ImageArray = npt.NDArray[np.uint8]
