"""Camera capture and video streaming module."""

import logging
from collections.abc import Generator, Sequence

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_control.arduino import ArduinoController
from hand_control.camera.realsense import RealSenseCamera
from hand_control.config import config
from hand_control.tracking import (
    HandTracker,
    draw_landmarks_on_image,
    get_all_finger_angles,
    get_all_finger_angles_distance,
    get_all_finger_ratios,
)
from hand_control.types import FingerAngles, FingerRatios, ImageArray, MutableLandmark, LandmarkPoint

logger = logging.getLogger(__name__)


def _inject_real_depth(
    landmarks: Sequence[LandmarkPoint],
    realsense: RealSenseCamera,
    depth_frame: object,
    frame_width: int,
    frame_height: int,
) -> list[MutableLandmark]:
    """Replace MediaPipe estimated z with real depth from the RealSense sensor.

    MediaPipe processes the horizontally flipped color frame, so its landmark
    x coordinates are in flipped space.  The RealSense depth frame is aligned
    to the original (unflipped) color frame, so the x coordinate must be
    mirrored back before the depth lookup: ``depth_x = (1 - lm.x) * width``.

    Args:
        landmarks: 21 MediaPipe hand landmarks (normalized, flipped space).
        realsense: RealSenseCamera instance used for depth queries.
        depth_frame: Aligned RealSense depth frame.
        frame_width: Width of the color frame in pixels.
        frame_height: Height of the color frame in pixels.

    Returns:
        List of MutableLandmark with z set to real depth in meters.
        Landmarks where the sensor returns 0 (invalid/out-of-range) keep
        MediaPipe's estimated z value.
    """
    enriched: list[MutableLandmark] = []
    for lm in landmarks:
        # Un-flip x to map back into the original (unflipped) depth frame space
        depth_px_x = int((1.0 - lm.x) * frame_width)
        depth_px_y = int(lm.y * frame_height)

        real_depth = realsense.get_depth_at(depth_frame, depth_px_x, depth_px_y)

        # Use real depth when valid; fall back to MediaPipe estimate otherwise
        z = real_depth if real_depth > 0.0 else lm.z
        enriched.append(MutableLandmark(x=lm.x, y=lm.y, z=z))
    return enriched


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
        )
        self._finger_angles: FingerAngles = FingerAngles(
            thumb=config.max_servo_angle,
            index=config.max_servo_angle,
            middle=config.max_servo_angle,
            ring=config.max_servo_angle,
            pinky=config.max_servo_angle,
        )
        self._finger_ratios: FingerRatios = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
        self._strategy: str = "distance"  # "distance" or "joint_angle"
        self._distance_ratio_min: float = config.distance_ratio_min
        self._distance_ratio_max: float = config.distance_ratio_max

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
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config.max_num_hands,
            min_hand_detection_confidence=config.hand_detection_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
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
                frame: ImageArray = cv2.flip(color_bgr, 1)
                rgb_frame: ImageArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                # Detect hands
                detection_result = detector.detect(mp_image)

                # Default angles (all open)
                current_angles = FingerAngles(
                    thumb=config.max_servo_angle,
                    index=config.max_servo_angle,
                    middle=config.max_servo_angle,
                    ring=config.max_servo_angle,
                    pinky=config.max_servo_angle,
                )

                # Find and track hand
                tracked_hand = hand_tracker.find_tracked_hand(
                    detection_result.hand_landmarks
                )

                hand_distance_m: float = 0.0

                if tracked_hand is not None:
                    # Replace estimated z with real metric depth from RealSense
                    enriched_landmarks = _inject_real_depth(
                        tracked_hand,
                        self._realsense,
                        depth_frame,
                        frame_w,
                        frame_h,
                    )

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
                        current_ratios = {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}
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
                    len(detection_result.hand_landmarks)
                    if detection_result.hand_landmarks
                    else 0
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
                bgr_out: ImageArray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                ret, buffer = cv2.imencode(".jpg", bgr_out)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        finally:
            detector.close()
            self._realsense.close()
