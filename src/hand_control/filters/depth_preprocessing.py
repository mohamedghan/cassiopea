"""Depth map spatial and temporal preprocessing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
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


def bilateral_filter_depth_roi(
    depth_roi: np.ndarray,
    d: int = 7,
    sigma_color: float = 20.0,
    sigma_space: float = 20.0,
) -> np.ndarray:
    """Apply bilateral filtering to a depth ROI array.

    Bilateral filtering smooths depth while preserving edges, preventing
    background depth from bleeding into the hand region.

    Args:
        depth_roi: (H, W) float32 depth array in meters; may contain NaN.
        d: Pixel neighborhood diameter; must be odd and >= 5.
        sigma_color: Depth差异 threshold — larger values mean more blur.
        sigma_space: Spatial distance threshold.

    Returns:
        (H, W) float32 filtered depth array with NaN preserved where input was NaN.
    """
    mask_valid = ~np.isnan(depth_roi)
    if not np.any(mask_valid):
        return depth_roi

    depth_filled = depth_roi.copy()
    valid_vals = depth_roi[mask_valid]
    if valid_vals.size > 0:
        fill_value = float(np.nanmean(valid_vals))
        depth_filled[~mask_valid] = fill_value
    else:
        return depth_roi

    h, w = depth_roi.shape
    if h < 5 or w < 5:
        return depth_roi

    d_actual = d if (d % 2 == 1) else d + 1

    filtered = cv2.bilateralFilter(
        depth_filled.astype(np.float32),
        d_actual,
        sigma_color,
        sigma_space,
    )

    result = filtered.copy()
    result[~mask_valid] = np.nan
    return result


def _get_filtered_depth_at(
    depth_roi: np.ndarray,
    landmarks: Sequence[LandmarkPoint],
    frame_width: int,
    frame_height: int,
    px_min: int,
    py_min: int,
) -> list[float]:
    """Look up filtered depth for each landmark from the pre-filtered ROI array.

    Args:
        depth_roi: Filtered (H, W) float32 depth array.
        landmarks: 21 hand landmarks (normalized 0-1, flipped space).
        frame_width: Original frame width.
        frame_height: Original frame height.
        px_min: ROI left boundary in original frame pixel coords.
        py_min: ROI top boundary in original frame pixel coords.

    Returns:
        List of 21 depth values in meters (0 where invalid/NaN).
    """
    depths: list[float] = []
    for lm in landmarks:
        col = int((1.0 - lm.x) * frame_width) - px_min
        row = int(lm.y * frame_height) - py_min
        if 0 <= row < depth_roi.shape[0] and 0 <= col < depth_roi.shape[1]:
            val = float(depth_roi[row, col])
            depths.append(val if not np.isnan(val) else 0.0)
        else:
            depths.append(0.0)
    return depths


def _extract_hand_roi_depth(
    landmarks: Sequence[LandmarkPoint],
    depth_frame: rs.depth_frame,
    realsense: RealSenseCamera,
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Extract and return the hand ROI depth array and its bounds.

    Args:
        landmarks: 21 hand landmarks (normalized 0-1, flipped space).
        depth_frame: Aligned RealSense depth frame.
        realsense: RealSenseCamera instance.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        Tuple of (depth_roi, px_min, py_min, roi_w, roi_h). The depth_roi
        is a (roi_h, roi_w) float32 array in meters (0 where invalid).
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

    depth_roi = np.zeros((roi_h, roi_w), dtype=np.float32)

    for row in range(py_min, py_max):
        for col in range(px_min, px_max):
            depth_px_x = int((1.0 - col / frame_width) * frame_width)
            depth_px_y = row
            d = realsense.get_depth_at(depth_frame, depth_px_x, depth_px_y)
            depth_roi[row - py_min, col - px_min] = d if d > 0 else np.nan

    return depth_roi, px_min, py_min, roi_w, roi_h
