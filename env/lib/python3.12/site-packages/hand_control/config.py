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

    # Camera configuration
    camera_index: int = 0

    # Servo configuration
    max_servo_angle: int = 120

    # Model configuration
    model_path: Path = Path("hand_landmarker.task")

    # Flask configuration
    flask_host: str = "0.0.0.0"
    flask_port: int = 5000
    flask_debug: bool = False

    # Hand detection configuration
    hand_detection_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    max_num_hands: int = 2

    # Filter configuration
    filter_min_cutoff: float = 0.5
    filter_beta: float = 0.01

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            serial_port=os.getenv("SERIAL_PORT", cls.serial_port),
            baud_rate=int(os.getenv("BAUD_RATE", str(cls.baud_rate))),
            camera_index=int(os.getenv("CAMERA_INDEX", str(cls.camera_index))),
            max_servo_angle=int(os.getenv("MAX_SERVO_ANGLE", str(cls.max_servo_angle))),
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
            filter_min_cutoff=float(
                os.getenv("FILTER_MIN_CUTOFF", str(cls.filter_min_cutoff))
            ),
            filter_beta=float(os.getenv("FILTER_BETA", str(cls.filter_beta))),
        )


# Default configuration instance
config = Config.from_env()
