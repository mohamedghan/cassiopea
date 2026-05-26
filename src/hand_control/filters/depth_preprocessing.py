"""Depth map preprocessing for hand tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from hand_control.filters.low_pass import LowPassFilter
from hand_control.types import LandmarkPoint, MutableLandmark

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyrealsense2 as rs

    from hand_control.camera.realsense import RealSenseCamera


class DepthEMA:
    """Temporal Exponential Moving Average filter for per-landmark z-coordinates.

    Maintains 21 independent LowPassFilter instances — one per hand landmark —
    so each joint's depth is temporally smoothed independently.
    """

    NUM_LANDMARKS = 21

    def __init__(self, alpha: float = 0.3) -> None:
        """Initialize the depth EMA filter.

        Args:
            alpha: EMA smoothing factor in (0, 1). Higher = less smoothing,
                faster response to depth changes.
        """
        self._alpha = alpha
        self._filters: list[LowPassFilter] = [
            LowPassFilter(alpha=alpha) for _ in range(self.NUM_LANDMARKS)
        ]

    def update(self, landmarks: list[MutableLandmark]) -> list[MutableLandmark]:
        """Apply EMA smoothing to the z-coordinate of each landmark.

        Args:
            landmarks: List of 21 MutableLandmarks with real metric z.

        Returns:
            New list of MutableLandmarks with smoothed z values.
        """
        smoothed: list[MutableLandmark] = []
        for i, lm in enumerate(landmarks):
            z = self._filters[i].update(lm.z)
            smoothed.append(MutableLandmark(x=lm.x, y=lm.y, z=z))
        return smoothed

    def reset(self) -> None:
        """Reset all 21 filters — call when tracking is lost."""
        for f in self._filters:
            f.reset()


def _get_depth_at(
    landmarks: Sequence[LandmarkPoint],
    depth_frame: rs.depth_frame,
    realsense: RealSenseCamera,
    frame_width: int,
    frame_height: int,
) -> list[float]:
    """Look up depth at each landmark from the pre-filtered spatial depth frame.

    Args:
        landmarks: 21 hand landmarks (normalized 0-1, flipped space).
        depth_frame: Aligned RealSense depth frame (spatial filter already applied).
        realsense: RealSenseCamera instance.
        frame_width: Original frame width.
        frame_height: Original frame height.

    Returns:
        List of 21 depth values in meters (0 where invalid).
    """
    depths: list[float] = []
    for lm in landmarks:
        depth_px_x = int((1.0 - lm.x) * frame_width)
        depth_px_y = int(lm.y * frame_height)
        d = realsense.get_depth_at(depth_frame, depth_px_x, depth_px_y)
        depths.append(d)
    return depths


def _extract_hand_roi_depth(
    landmarks: Sequence[LandmarkPoint],
    depth_frame: rs.depth_frame,
    realsense: RealSenseCamera,
    frame_width: int,
    frame_height: int,
) -> tuple[list[float], int, int, int, int]:
    """Extract depth values and bounds for the hand ROI.

    Args:
        landmarks: 21 hand landmarks (normalized 0-1, flipped space).
        depth_frame: Aligned RealSense depth frame.
        realsense: RealSenseCamera instance.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Tuple of (depths_list, px_min, py_min, roi_w, roi_h). The depths_list
        contains depth in meters for each landmark (0 where invalid).
    """
    pts_norm = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)

    margin = 0.08
    x_min = max(0.0, pts_norm[:, 0].min() - margin)
    x_max = min(1.0, pts_norm[:, 0].max() + margin)
    y_min = max(0.0, pts_norm[:, 1].min() - margin)
    y_max = min(1.0, pts_norm[:, 1].max() + margin)

    px_min = int(x_min * frame_width)
    px_max = int(x_max * frame_width)
    py_min = int(y_min * frame_height)
    py_max = int(y_max * frame_height)

    roi_w = max(1, px_max - px_min)
    roi_h = max(1, py_max - py_min)

    depths = _get_depth_at(landmarks, depth_frame, realsense, frame_width, frame_height)

    return depths, px_min, py_min, roi_w, roi_h
