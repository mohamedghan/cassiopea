"""Intel RealSense D435i camera integration."""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

from hand_control.types import ImageArray


class RealSenseCamera:
    """Manages Intel RealSense D435i synchronized color and depth streams.

    Streams both color (BGR) and depth (Z16) at the same resolution so that
    each color pixel maps 1-to-1 with its corresponding depth value via
    RealSense's built-in frame alignment.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Initialize the RealSense camera.

        Args:
            width: Stream width in pixels.
            height: Stream height in pixels.
            fps: Frames per second.
        """
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline: rs.pipeline | None = None
        self._align: rs.align | None = None

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
            self._pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(
                rs.stream.color, self._width, self._height, rs.format.bgr8, self._fps
            )
            cfg.enable_stream(
                rs.stream.depth, self._width, self._height, rs.format.z16, self._fps
            )
            self._pipeline.start(cfg)
            # Align depth frame to color frame so pixel coords match 1-to-1
            self._align = rs.align(rs.stream.color)
            return True
        except Exception:
            self._pipeline = None
            self._align = None
            return False

    def close(self) -> None:
        """Stop the RealSense pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        self._align = None

    def get_frames(self) -> tuple[ImageArray | None, rs.depth_frame | None]:
        """Wait for and return synchronized color and depth frames.

        The depth frame is aligned to the color frame, so pixel (x, y) in
        the color image corresponds to the same pixel in the depth frame.

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
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

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
