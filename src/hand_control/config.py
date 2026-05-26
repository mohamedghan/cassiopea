"""Configuration management using environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Application configuration with environment-based overrides."""

    # Serial/Arduino configuration
    serial_port: str = "/dev/ttyACM0"
    baud_rate: int = 115200

    # RealSense D435i configuration
    realsense_width: int = 640
    realsense_height: int = 480
    realsense_fps: int = 30

    # Servo configuration
    max_servo_angle: int = 180

    # Per-finger servo angle ranges [min, max]
    thumb_min_angle: int = 52
    thumb_max_angle: int = 180
    index_min_angle: int = 36
    index_max_angle: int = 120
    middle_min_angle: int = 36
    middle_max_angle: int = 120
    ring_min_angle: int = 40
    ring_max_angle: int = 100
    pinky_min_angle: int = 20
    pinky_max_angle: int = 120

    # Distance strategy calibration
    distance_ratio_min: float = 0.5  # Normalized ratio when fist is closed
    distance_ratio_max: float = 2.0  # Normalized ratio when hand is open

    # Model configuration
    model_path: Path = Path("hand_landmarker.task")

    # Flask configuration
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    flask_debug: bool = False

    # Hand detection configuration
    hand_detection_confidence: float = 0.8
    hand_tracking_confidence: float = 0.5
    max_num_hands: int = 2

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            serial_port=os.getenv("SERIAL_PORT", cls.serial_port),
            baud_rate=int(os.getenv("BAUD_RATE", str(cls.baud_rate))),
            realsense_width=int(os.getenv("REALSENSE_WIDTH", str(cls.realsense_width))),
            realsense_height=int(os.getenv("REALSENSE_HEIGHT", str(cls.realsense_height))),
            realsense_fps=int(os.getenv("REALSENSE_FPS", str(cls.realsense_fps))),
            max_servo_angle=int(os.getenv("MAX_SERVO_ANGLE", str(cls.max_servo_angle))),
            thumb_min_angle=int(os.getenv("THUMB_MIN_ANGLE", str(cls.thumb_min_angle))),
            thumb_max_angle=int(os.getenv("THUMB_MAX_ANGLE", str(cls.thumb_max_angle))),
            index_min_angle=int(os.getenv("INDEX_MIN_ANGLE", str(cls.index_min_angle))),
            index_max_angle=int(os.getenv("INDEX_MAX_ANGLE", str(cls.index_max_angle))),
            middle_min_angle=int(os.getenv("MIDDLE_MIN_ANGLE", str(cls.middle_min_angle))),
            middle_max_angle=int(os.getenv("MIDDLE_MAX_ANGLE", str(cls.middle_max_angle))),
            ring_min_angle=int(os.getenv("RING_MIN_ANGLE", str(cls.ring_min_angle))),
            ring_max_angle=int(os.getenv("RING_MAX_ANGLE", str(cls.ring_max_angle))),
            pinky_min_angle=int(os.getenv("PINKY_MIN_ANGLE", str(cls.pinky_min_angle))),
            pinky_max_angle=int(os.getenv("PINKY_MAX_ANGLE", str(cls.pinky_max_angle))),
            distance_ratio_min=float(
                os.getenv("DISTANCE_RATIO_MIN", str(cls.distance_ratio_min))
            ),
            distance_ratio_max=float(
                os.getenv("DISTANCE_RATIO_MAX", str(cls.distance_ratio_max))
            ),
            model_path=Path(os.getenv("MODEL_PATH", str(cls.model_path))),
            flask_host=os.getenv("FLASK_HOST", cls.flask_host),
            flask_port=int(os.getenv("FLASK_PORT", str(cls.flask_port))),
            flask_debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
            hand_detection_confidence=float(
                os.getenv("HAND_DETECTION_CONFIDENCE", str(cls.hand_detection_confidence))
            ),
            hand_tracking_confidence=float(
                os.getenv("HAND_TRACKING_CONFIDENCE", str(cls.hand_tracking_confidence))
            ),
            max_num_hands=int(os.getenv("MAX_NUM_HANDS", str(cls.max_num_hands))),
        )


# Default configuration instance
config = Config.from_env()
