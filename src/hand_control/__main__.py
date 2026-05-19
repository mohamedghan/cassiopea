"""Entry point for running the hand control application."""

import logging

from hand_control.app import create_app
from hand_control.config import config
from hand_control.models import download_model


def main() -> None:
    """Run the hand control application."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Download model if needed
    download_model()

    # Create Flask app
    app = create_app()

    # Connect to Arduino
    arduino = app.config.get("arduino")
    if arduino is not None:
        arduino.connect()

    # Open camera
    camera_stream = app.config.get("camera_stream")
    if camera_stream is not None:
        camera_stream.open()

    print(f"Starting server at http://{config.flask_host}:{config.flask_port}")

    try:
        app.run(
            host=config.flask_host,
            port=config.flask_port,
            threaded=True,
            debug=config.flask_debug,
        )
    finally:
        # Cleanup
        if camera_stream is not None:
            camera_stream.close()
        if arduino is not None:
            arduino.disconnect()


if __name__ == "__main__":
    main()
