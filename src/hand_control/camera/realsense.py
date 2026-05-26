"""Intel RealSense D435i camera integration."""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

from hand_control.types import ImageArray


class RealSenseCamera:
    """Manages Intel RealSense D435i synchronized color and depth streams.

    Streams both color (BGR) and depth (Z16) at the same resolution so that
    each color pixel maps 1-to-1 with its corresponding depth value via
    RealSense's built-in frame alignment.  An optional spatial filter is
    applied to the raw depth to reduce noise while preserving edges.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        spatial_filter_enabled: bool = True,
        spatial_filter_magnitude: int = 2,
        spatial_filter_smooth_alpha: float = 0.5,
        spatial_filter_smooth_delta: float = 20.0,
    ) -> None:
        """Initialize the RealSense camera.

        Args:
            width: Stream width in pixels.
            height: Stream height in pixels.
            fps: Frames per second.
            spatial_filter_enabled: Whether to apply RealSense spatial filter.
            spatial_filter_magnitude: Spatial filter neighborhood diameter (1-5).
            spatial_filter_smooth_alpha: Edge-preserving smoothing strength (0-1).
            spatial_filter_smooth_delta: Depth difference threshold for averaging.
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._spatial_enabled = spatial_filter_enabled
        self._spatial_magnitude = spatial_filter_magnitude
        self._spatial_alpha = spatial_filter_smooth_alpha
        self._spatial_delta = spatial_filter_smooth_delta
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None
        self._spatial_filter: rs.spatial_filter | None = None

    @property
    def width(self) -> int:
        """Stream width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Stream height in pixels."""
        return self._height

    def open(self) -> bool:
        """Start the RealSense pipeline with color and depth streams.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
            )
            cfg.enable_stream(
                rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
            )
            pipeline.start(cfg)
            self._pipeline = pipeline
            self._align = rs.align(rs.stream.color)

            if self._spatial_enabled:
                self._spatial_filter = rs.spatial_filter()
                self._spatial_filter.set_option(rs.option.filter_magnitude, self._spatial_magnitude)
                self._spatial_filter.set_option(rs.option.filter_smooth_alpha, self._spatial_alpha)
                self._spatial_filter.set_option(rs.option.filter_smooth_delta, self._spatial_delta)

            return True
        except Exception:
            self._pipeline = None
            self._align = None
            self._spatial_filter = None
            return False

    def close(self) -> None:
        """Stop the RealSense pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._align = None
        self._spatial_filter = None

    def get_frames(self) -> tuple[ImageArray | None, rs.depth_frame | None]:
        """Wait for and return synchronized color and depth frames.

        The depth frame is aligned to the color frame, so pixel (x, y) in
        the color image corresponds to the same pixel in the depth frame.
        The RealSense spatial filter is applied to the depth frame to reduce
        noise while preserving edges.

        Returns:
            Tuple of (color_image_bgr, depth_frame).
            Both values are None if a frame could not be retrieved.
        """
        if self._pipeline is None or self._align is None:
            return None, None
        try:
            frameset = self._pipeline.wait_for_frames()
            aligned = self._align.process(frameset)

            color_frame = aligned.get_color_frame()
            depth_frame: rs.depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            if self._spatial_filter is not None:
                spatial_result = self._spatial_filter.process(depth_frame)
                depth_frame = spatial_result.as_depth_frame()

            color_image: ImageArray = np.asanyarray(color_frame.get_data())
            return color_image, depth_frame
        except Exception:
            return None, None

    def get_depth_at(self, depth_frame: rs.depth_frame, px: int, py: int) -> float:
        """Return real depth in meters at the given pixel coordinate.

        Args:
            depth_frame: Aligned RealSense depth frame.
            px: Pixel column (x), in the unflipped original frame space.
            py: Pixel row (y).

        Returns:
            Depth in meters, or 0.0 if pixel is out of bounds or depth
            is invalid (RealSense returns 0 for invalid/out-of-range pixels).
        """
        if px < 0 or px >= self._width or py < 0 or py >= self._height:
            return 0.0
        return float(depth_frame.get_distance(px, py))
