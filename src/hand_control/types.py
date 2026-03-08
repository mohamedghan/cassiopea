"""Type definitions for the hand control system."""

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


# NumPy array type aliases
Vector3D = npt.NDArray[np.floating[np.float64]]
Vector2D = npt.NDArray[np.floating[np.float64]]
ImageArray = npt.NDArray[np.uint8]
