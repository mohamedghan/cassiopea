"""Camera capture and video streaming module."""

import logging
import time
from collections.abc import Generator, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.interpolate import griddata

from hand_control.arduino import ArduinoController
from hand_control.camera.realsense import RealSenseCamera
from hand_control.config import config
from hand_control.filters import DepthEMA
from hand_control.filters.depth_preprocessing import _extract_hand_roi_depth
from hand_control.tracking import (
    HandTracker,
    draw_landmarks_on_image,
    get_all_finger_angles,
    get_all_finger_angles_distance,
    get_all_finger_ratios,
)
from hand_control.types import (
    FingerAngles,
    FingerRatios,
    ImageArray,
    LandmarkPoint,
    MutableLandmark,
)

_RS_DEPTH_MIN = 0.0
_RS_DEPTH_MAX = 3.0
_INTERP_GRID = 100

logger = logging.getLogger(__name__)


def _inject_real_depth(
    landmarks: Sequence[LandmarkPoint],
    realsense: RealSenseCamera,
    depth_frame: object,
    frame_width: int,
    frame_height: int,
) -> list[MutableLandmark]:
    """Replace MediaPipe estimated z with real depth from RealSense.

    MediaPipe processes the horizontally flipped color frame, so its landmark
    x coordinates are in flipped space.  The RealSense depth frame is aligned
    to the original (unflipped) color frame, so the x coordinate must be
    mirrored back before the depth lookup: ``depth_x = (1 - lm.x) * width``.

    Args:
        landmarks: 21 MediaPipe hand landmarks (normalized, flipped space).
        realsense: RealSenseCamera instance used for depth queries.
        depth_frame: Aligned RealSense depth frame (spatially filtered).
        frame_width: Width of the color frame in pixels.
        frame_height: Height of the color frame in pixels.

    Returns:
        List of MutableLandmark with z set to real depth in meters.
        Landmarks where the sensor returns 0 (invalid/out-of-range) keep
        MediaPipe's estimated z value.
    """
    depths, _, _, _, _ = _extract_hand_roi_depth(
        landmarks, depth_frame, realsense, frame_width, frame_height
    )

    enriched: list[MutableLandmark] = []
    for lm, z in zip(landmarks, depths, strict=True):
        z = z if z > 0.0 else lm.z
        enriched.append(MutableLandmark(x=lm.x, y=lm.y, z=z))
    return enriched


def _build_mediapipe_depth_map(
    landmarks: Sequence[LandmarkPoint],
) -> ImageArray | None:
    """Build an interpolated Jet heatmap from MediaPipe landmark z-values.

    The 21 sparse landmark z-values (normalized, estimated depth) are
    interpolated onto a dense grid within the hand bounding box, then
    colormapped with Jet.

    Args:
        landmarks: 21 MediaPipe landmarks (normalized 0-1, flipped space).
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        BGR Jet heatmap (uint8) sized to the hand ROI, or None if interpolation
        fails or fewer than 3 landmarks are available.
    """
    if landmarks is None or len(landmarks) < 3:
        return None

    pts_norm = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)
    z_vals = np.array([lm.z for lm in landmarks], dtype=np.float64)

    margin = 0.08
    x_min = max(0.0, pts_norm[:, 0].min() - margin)
    x_max = min(1.0, pts_norm[:, 0].max() + margin)
    y_min = max(0.0, pts_norm[:, 1].min() - margin)
    y_max = min(1.0, pts_norm[:, 1].max() + margin)

    grid_x = np.linspace(x_min, x_max, _INTERP_GRID)
    grid_y = np.linspace(y_min, y_max, _INTERP_GRID)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

    try:
        interp_z = griddata(pts_norm, z_vals, (grid_xx, grid_yy), method="linear")
    except Exception:
        return None

    if interp_z is None or np.all(np.isnan(interp_z)):
        return None

    interp_z = np.nan_to_num(interp_z, nan=float(np.mean(z_vals)))
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    z_range = z_max - z_min
    if z_range < 0.01:
        z_min -= 0.05
        z_max += 0.05
        z_range = z_max - z_min
    z_norm = (interp_z - z_min) / z_range
    z_uint8 = np.clip(z_norm * 255, 0, 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(z_uint8, cv2.COLORMAP_JET)
    return heatmap  # type: ignore[no-any-return]


def _build_realsense_depth_map(
    landmarks: Sequence[LandmarkPoint],
    realsense: RealSenseCamera,
    depth_frame: object,
    frame_w: int,
    frame_h: int,
) -> ImageArray | None:
    """Build a Jet heatmap of the RealSense depth sensor values over the hand ROI.

    Args:
        landmarks: 21 hand landmarks (normalized 0-1, flipped space).
        realsense: RealSenseCamera instance for depth queries.
        depth_frame: Aligned RealSense depth frame.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        BGR Jet heatmap (uint8) sized to the hand ROI, or None on failure.
    """
    if landmarks is None or len(landmarks) < 3:
        return None

    pts_norm = np.array([(lm.x, lm.y) for lm in landmarks], dtype=np.float64)

    margin = 0.08
    x_min = max(0.0, pts_norm[:, 0].min() - margin)
    x_max = min(1.0, pts_norm[:, 0].max() + margin)
    y_min = max(0.0, pts_norm[:, 1].min() - margin)
    y_max = min(1.0, pts_norm[:, 1].max() + margin)

    px_min = int(x_min * frame_w)
    px_max = int(x_max * frame_w)
    py_min = int(y_min * frame_h)
    py_max = int(y_max * frame_h)

    roi_w = max(1, px_max - px_min)
    roi_h = max(1, py_max - py_min)

    depth_roi = np.zeros((roi_h, roi_w), dtype=np.float32)
    for row in range(py_min, py_max):
        for col in range(px_min, px_max):
            depth_px_x = int((1.0 - col / frame_w) * frame_w)
            depth_px_y = row
            d = realsense.get_depth_at(depth_frame, depth_px_x, depth_px_y)
            depth_roi[row - py_min, col - px_min] = d if d > 0 else np.nan

    valid_mask = ~np.isnan(depth_roi)
    if not np.any(valid_mask):
        return None

    valid_vals = depth_roi[valid_mask]
    d_min = float(np.fmax(_RS_DEPTH_MIN, np.nanmin(valid_vals)))
    d_max = float(np.fmin(_RS_DEPTH_MAX, np.nanmax(valid_vals)))

    if d_max <= d_min:
        return None

    depth_norm = np.zeros_like(depth_roi)
    depth_norm[valid_mask] = (depth_roi[valid_mask] - d_min) / (d_max - d_min)
    depth_norm[~valid_mask] = 0
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    return heatmap  # type: ignore[return-value]


class CameraStream:
    """Manages RealSense camera capture and hand tracking video stream."""

    def __init__(
        self,
        arduino: ArduinoController | None = None,
    ) -> None:
        """Initialize the camera stream.

        Args:
            arduino: Optional Arduino controller for sending angles.
        """
        self._arduino = arduino
        self._realsense = RealSenseCamera(
            width=config.realsense_width,
            height=config.realsense_height,
            fps=config.realsense_fps,
            spatial_filter_enabled=config.spatial_filter_enabled,
            spatial_filter_magnitude=config.spatial_filter_magnitude,
            spatial_filter_smooth_alpha=config.spatial_filter_smooth_alpha,
            spatial_filter_smooth_delta=config.spatial_filter_smooth_delta,
        )
        self._finger_angles: FingerAngles = FingerAngles(
            thumb=config.max_servo_angle,
            index=config.max_servo_angle,
            middle=config.max_servo_angle,
            ring=config.max_servo_angle,
            pinky=config.max_servo_angle,
        )
        self._finger_ratios: FingerRatios = {
            "thumb": 0.0,
            "index": 0.0,
            "middle": 0.0,
            "ring": 0.0,
            "pinky": 0.0,
        }
        self._strategy: str = "distance"  # "distance" or "joint_angle"
        self._distance_ratio_min: float = config.distance_ratio_min
        self._distance_ratio_max: float = config.distance_ratio_max
        self._depth_ema = DepthEMA(alpha=config.depth_ema_alpha)

    @property
    def finger_angles(self) -> FingerAngles:
        """Current finger angles from hand tracking."""
        return self._finger_angles

    @property
    def finger_ratios(self) -> FingerRatios:
        """Current finger distance ratios."""
        return self._finger_ratios

    @property
    def strategy(self) -> str:
        """Current angle calculation strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, value: str) -> None:
        """Set angle calculation strategy."""
        if value in ("distance", "joint_angle"):
            self._strategy = value

    @property
    def distance_ratio_min(self) -> float:
        """Minimum distance ratio for distance strategy."""
        return self._distance_ratio_min

    @distance_ratio_min.setter
    def distance_ratio_min(self, value: float) -> None:
        """Set minimum distance ratio."""
        self._distance_ratio_min = value

    @property
    def distance_ratio_max(self) -> float:
        """Maximum distance ratio for distance strategy."""
        return self._distance_ratio_max

    @distance_ratio_max.setter
    def distance_ratio_max(self, value: float) -> None:
        """Set maximum distance ratio."""
        self._distance_ratio_max = value

    def open(self) -> bool:
        """Open the RealSense camera.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        return self._realsense.open()

    def close(self) -> None:
        """Close the RealSense camera."""
        self._realsense.close()

    def generate_frames(self) -> Generator[bytes, None, None]:
        """Generate JPEG frames with hand tracking and real depth overlay.

        The RealSense D435i provides synchronized color (BGR) and depth
        (Z16) streams.  Depth is aligned to the color frame so every color
        pixel has a corresponding metric depth value.

        For each detected hand landmark the MediaPipe estimated z coordinate
        is replaced with the real depth in meters queried from the depth frame.
        The wrist landmark depth is also rendered on-screen as the hand
        distance from the camera.

        Yields:
            JPEG-encoded frames with MJPEG boundary markers.
        """
        if not self._realsense.open():
            return

        # Initialize MediaPipe hand landmarker
        base_options = python.BaseOptions(model_asset_path=str(config.model_path))
        running_mode = (
            vision.RunningMode.VIDEO if config.running_mode == "VIDEO" else vision.RunningMode.IMAGE
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
            min_hand_presence_confidence = 0.5,
            running_mode=running_mode,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        hand_tracker = HandTracker()

        frame_w = self._realsense.width
        frame_h = self._realsense.height

        try:
            while True:
                color_bgr, depth_frame = self._realsense.get_frames()
                if color_bgr is None or depth_frame is None:
                    continue

                # Flip horizontally for mirror effect (depth frame stays unflipped)
                frame = cv2.flip(color_bgr, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detect hands (VIDEO mode for temporal smoothing)
                timestamp_ms = int(time.time() * 1000)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)

                # Default angles (all open)
                current_angles = FingerAngles(
                    thumb=config.max_servo_angle,
                    index=config.max_servo_angle,
                    middle=config.max_servo_angle,
                    ring=config.max_servo_angle,
                    pinky=config.max_servo_angle,
                )
                current_ratios: FingerRatios = FingerRatios(
                    thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0
                )

                # Find and track hand
                tracked_hand = hand_tracker.find_tracked_hand(detection_result.hand_landmarks)

                hand_distance_m: float = 0.0

                if tracked_hand is not None:
                    # Replace estimated z with real metric depth from RealSense
                    # (bilateral-filtered and looked up per-landmark)
                    enriched_landmarks = _inject_real_depth(
                        tracked_hand,
                        self._realsense,
                        depth_frame,
                        frame_w,
                        frame_h,
                    )

                    # Apply temporal EMA smoothing to z-coordinates
                    enriched_landmarks = self._depth_ema.update(enriched_landmarks)

                    # Wrist landmark (index 0) gives hand distance from camera
                    hand_distance_m = enriched_landmarks[0].z

                    if logger.isEnabledFor(logging.DEBUG):
                        lm_str = "  ".join(
                            f"[{i:02d}] x={lm.x:.3f} y={lm.y:.3f} z={lm.z:.3f}"
                            for i, lm in enumerate(enriched_landmarks)
                        )
                        logger.debug("Landmarks: %s", lm_str)

                    if self._strategy == "distance":
                        current_angles = get_all_finger_angles_distance(
                            enriched_landmarks,
                            ratio_min=self._distance_ratio_min,
                            ratio_max=self._distance_ratio_max,
                        )
                        current_ratios = get_all_finger_ratios(enriched_landmarks)
                    else:
                        current_angles = get_all_finger_angles(enriched_landmarks)
                        current_ratios = FingerRatios(
                            thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0
                        )
                    self._finger_angles = current_angles
                    self._finger_ratios = current_ratios

                    # Send to Arduino
                    if self._arduino is not None:
                        angle_list = [
                            current_angles["thumb"],
                            current_angles["index"],
                            current_angles["middle"],
                            current_angles["ring"],
                            current_angles["pinky"],
                        ]
                        self._arduino.send_all_angles(angle_list)

                # Draw landmarks on frame
                rgb_frame = draw_landmarks_on_image(
                    rgb_frame,
                    tracked_hand,
                    current_angles,
                    hand_tracker.tracked_hand_center,
                    current_ratios,
                )

                # Overlay: number of hands detected
                num_hands = (
                    len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
                )
                cv2.putText(
                    rgb_frame,
                    f"Hands detected: {num_hands}",
                    (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Overlay: real hand distance from RealSense depth sensor
                if hand_distance_m > 0.0:
                    cv2.putText(
                        rgb_frame,
                        f"Distance: {hand_distance_m:.3f} m",
                        (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 128),
                        2,
                    )

                # Convert back to BGR for JPEG encoding
                bgr_out = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                ret, buffer = cv2.imencode(".jpg", bgr_out)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        finally:
            detector.close()
            self._realsense.close()

    def generate_composite_frames(self) -> Generator[bytes, None, None]:
        """Generate a 3-pane composite MJPEG frame for event presentation.

        Pane 1 (left):  Color camera feed with MediaPipe hand landmarks.
        Pane 2 (center): Interpolated Jet heatmap of MediaPipe estimated z-values.
        Pane 3 (right):  Jet heatmap of RealSense depth sensor values.

        Yields:
            MJPEG multipart frames (JPEG-encoded 3-pane composite).
        """
        if not self._realsense.open():
            return

        base_options = python.BaseOptions(model_asset_path=str(config.model_path))
        running_mode = (
            vision.RunningMode.VIDEO if config.running_mode == "VIDEO" else vision.RunningMode.IMAGE
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
            min_hand_presence_confidence = 0.5,
            running_mode=running_mode,
        )
        detector = vision.HandLandmarker.create_from_options(options)
        hand_tracker = HandTracker()

        frame_w = self._realsense.width
        frame_h = self._realsense.height

        try:
            while True:
                color_bgr, depth_frame = self._realsense.get_frames()
                if color_bgr is None or depth_frame is None:
                    continue

                frame = cv2.flip(color_bgr, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int(time.time() * 1000)
                detection_result = detector.detect_for_video(mp_image, timestamp_ms)
                tracked_hand = hand_tracker.find_tracked_hand(detection_result.hand_landmarks)

                current_angles = FingerAngles(
                    thumb=config.max_servo_angle,
                    index=config.max_servo_angle,
                    middle=config.max_servo_angle,
                    ring=config.max_servo_angle,
                    pinky=config.max_servo_angle,
                )
                current_ratios: FingerRatios = FingerRatios(
                    thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0
                )

                if tracked_hand is not None:
                    enriched = _inject_real_depth(
                        tracked_hand,
                        self._realsense,
                        depth_frame,
                        frame_w,
                        frame_h,
                    )
                    self._finger_angles = get_all_finger_angles_distance(
                        enriched,
                        ratio_min=self._distance_ratio_min,
                        ratio_max=self._distance_ratio_max,
                    )
                    self._finger_ratios = get_all_finger_ratios(enriched)
                    current_angles = self._finger_angles
                    current_ratios = self._finger_ratios

                    if self._arduino is not None:
                        self._arduino.send_all_angles(
                            [
                                current_angles["thumb"],
                                current_angles["index"],
                                current_angles["middle"],
                                current_angles["ring"],
                                current_angles["pinky"],
                            ]
                        )

                rgb_overlay = draw_landmarks_on_image(
                    rgb_frame,
                    tracked_hand,
                    current_angles,
                    hand_tracker.tracked_hand_center,
                    current_ratios,
                )
                pane_left = cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR)

                pane_center: ImageArray | None = None
                pane_right: ImageArray | None = None

                if tracked_hand is not None:
                    pane_center = _build_mediapipe_depth_map(tracked_hand)
                    pane_right = _build_realsense_depth_map(
                        tracked_hand,
                        self._realsense,
                        depth_frame,
                        frame_w,
                        frame_h,
                    )

                pane_left_resized = cv2.resize(pane_left, (frame_w, frame_h))
                if pane_center is not None:
                    pane_center_resized = cv2.resize(pane_center, (frame_w, frame_h))
                else:
                    pane_center_resized = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    cv2.putText(
                        pane_center_resized,
                        "No hand detected",
                        (frame_w // 2 - 100, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                if pane_right is not None:
                    pane_right_resized = cv2.resize(pane_right, (frame_w, frame_h))
                else:
                    pane_right_resized = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
                    cv2.putText(
                        pane_right_resized,
                        "No hand detected",
                        (frame_w // 2 - 100, frame_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                composite = np.hstack([pane_left_resized, pane_center_resized, pane_right_resized])

                ret, buffer = cv2.imencode(".jpg", composite)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        finally:
            detector.close()
            self._realsense.close()
