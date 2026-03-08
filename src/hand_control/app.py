"""Flask application factory."""

from flask import Flask

from hand_control.arduino import ArduinoController
from hand_control.camera import CameraStream
from hand_control.config import Config
from hand_control.config import config as default_config
from hand_control.web import create_blueprint


def create_app(app_config: Config | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        app_config: Optional configuration. Defaults to environment-based config.

    Returns:
        Configured Flask application instance.
    """
    cfg = app_config or default_config
    app = Flask(__name__)

    # Store configuration
    app.config["hand_control_config"] = cfg

    # Initialize Arduino controller
    arduino = ArduinoController(port=cfg.serial_port, baud_rate=cfg.baud_rate)
    app.config["arduino"] = arduino

    # Initialize camera stream with Arduino
    camera_stream = CameraStream(camera_index=cfg.camera_index, arduino=arduino)
    app.config["camera_stream"] = camera_stream

    # Register blueprint
    bp = create_blueprint()
    app.register_blueprint(bp)

    return app
