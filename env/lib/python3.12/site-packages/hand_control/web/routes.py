"""Flask routes for the hand control web interface."""

from flask import Blueprint, Response, current_app, jsonify, render_template

from hand_control.config import config


def create_blueprint() -> Blueprint:
    """Create the Flask blueprint for hand control routes.

    Returns:
        Flask Blueprint with all routes registered.
    """
    bp = Blueprint("hand_control", __name__, template_folder="templates")

    @bp.route("/")
    def index() -> str:
        """Render the main control interface."""
        return render_template("index.html", max_angle=config.max_servo_angle)

    @bp.route("/video_feed")
    def video_feed() -> Response:
        """Stream video with hand tracking overlay."""
        camera_stream = current_app.config.get("camera_stream")
        if camera_stream is None:
            return Response("Camera not available", status=503)
        return Response(
            camera_stream.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @bp.route("/finger/<int:finger>/<int:angle>")
    def set_finger(finger: int, angle: int) -> Response:
        """Set angle for a single finger.

        Args:
            finger: Finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky).
            angle: Servo angle (0-128).

        Returns:
            JSON response with success status.
        """
        angle = max(0, min(config.max_servo_angle, angle))
        arduino = current_app.config.get("arduino")
        if arduino is not None:
            arduino.send_finger_angle(finger, angle)
        return jsonify({"success": True, "finger": finger, "angle": angle})

    @bp.route("/all/<angles>")
    def set_all(angles: str) -> Response:
        """Set all finger angles at once.

        Args:
            angles: Comma-separated list of 5 angles.

        Returns:
            JSON response with success status.
        """
        try:
            angle_list = [
                min(config.max_servo_angle, max(0, int(a))) for a in angles.split(",")
            ]
            if len(angle_list) == 5:
                arduino = current_app.config.get("arduino")
                if arduino is not None:
                    arduino.send_all_angles(angle_list)
                return jsonify({"success": True, "angles": angle_list})
        except ValueError:
            pass
        return jsonify({"success": False})

    @bp.route("/status")
    def status() -> Response:
        """Get current finger angles and Arduino connection status.

        Returns:
            JSON response with angles and connection status.
        """
        camera_stream = current_app.config.get("camera_stream")
        arduino = current_app.config.get("arduino")

        angles = (
            camera_stream.finger_angles
            if camera_stream is not None
            else {
                "thumb": config.max_servo_angle,
                "index": config.max_servo_angle,
                "middle": config.max_servo_angle,
                "ring": config.max_servo_angle,
                "pinky": config.max_servo_angle,
            }
        )

        return jsonify(
            {
                "angles": angles,
                "arduino_connected": arduino is not None and arduino.is_connected,
            }
        )

    return bp
