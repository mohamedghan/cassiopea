"""Camera capture and video streaming module."""

from collections.abc import Generator

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from hand_control.arduino import ArduinoController
from hand_control.config import config
from hand_control.tracking import HandTracker, draw_landmarks_on_image, get_all_finger_angles
from hand_control.types import FingerAngles, ImageArray


class CameraStream:
    """Manages camera capture and hand tracking video stream."""

    def __init__(
        self,
        camera_index: int | None = None,
        arduino: ArduinoController | None = None,
    ) -> None:
        """Initialize the camera stream.

        Args:
            camera_index: Camera device index. Defaults to config value.
            arduino: Optional Arduino controller for sending angles.
        """
        self._camera_index = camera_index if camera_index is not None else config.camera_index
        self._arduino = arduino
        self._camera: cv2.VideoCapture | None = None
        self._finger_angles: FingerAngles = FingerAngles(
            thumb=config.max_servo_angle,
            index=config.max_servo_angle,
            middle=config.max_servo_angle,
            ring=config.max_servo_angle,
            pinky=config.max_servo_angle,
        )

    @property
    def finger_angles(self) -> FingerAngles:
        """Current finger angles from hand tracking."""
        return self._finger_angles

    def open(self) -> bool:
        """Open the camera.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        self._camera = cv2.VideoCapture(self._camera_index)
        return self._camera.isOpened()

    def close(self) -> None:
        """Close the camera."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None

    def generate_frames(self) -> Generator[bytes, None, None]:
        """Generate JPEG frames with hand tracking overlay.

        Yields:
            JPEG-encoded frames with MJPEG boundary markers.
        """
        if self._camera is None:
            self.open()

        if self._camera is None or not self._camera.isOpened():
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

        try:
            while True:
                success, frame = self._camera.read()
                if not success:
                    break

                # Flip horizontally for mirror effect
                frame = cv2.flip(frame, 1)
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
                tracked_hand = hand_tracker.find_tracked_hand(detection_result.hand_landmarks)

                if tracked_hand is not None:
                    raw_angles = get_all_finger_angles(tracked_hand)
                    current_angles = hand_tracker.filter_angles(raw_angles)
                    self._finger_angles = current_angles

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
                )

                # Show number of hands detected
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

                # Convert back to BGR for encoding
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                # Encode as JPEG
                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )

        finally:
            detector.close()
