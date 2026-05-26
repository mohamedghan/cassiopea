"""Flask routes for the hand control web interface."""

from flask import Blueprint, Response, current_app, jsonify, render_template

from hand_control.config import config


def create_blueprint() -> Blueprint:
    """Create the Flask blueprint for hand control routes.

    Returns:
        Flask Blueprint with all routes registered.
    """
    bp = Blueprint("hand_control", __name__, template_folder="templates")

    # Per-finger (thumb=0 … pinky=4) servo range: (min, max)
    _finger_ranges = [
        (config.thumb_min_angle, config.thumb_max_angle),
        (config.index_min_angle, config.index_max_angle),
        (config.middle_min_angle, config.middle_max_angle),
        (config.ring_min_angle, config.ring_max_angle),
        (config.pinky_min_angle, config.pinky_max_angle),
    ]

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

    @bp.route("/presentation")
    def presentation() -> str:
        """Render the event presentation view with 3-pane composite."""
        return render_template("presentation.html")

    @bp.route("/presentation_feed")
    def presentation_feed() -> Response:
        """Stream the 3-pane composite video feed for event presentation."""
        camera_stream = current_app.config.get("camera_stream")
        if camera_stream is None:
            return Response("Camera not available", status=503)
        return Response(
            camera_stream.generate_composite_frames(),
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
        if 0 <= finger < len(_finger_ranges):
            lo, hi = _finger_ranges[finger]
        else:
            lo, hi = 0, config.max_servo_angle
        angle = max(lo, min(hi, angle))
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
                max(lo, min(hi, int(a)))
                for (lo, hi), a in zip(_finger_ranges, angles.split(","), strict=True)
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
                "thumb": config.thumb_max_angle,
                "index": config.index_max_angle,
                "middle": config.middle_max_angle,
                "ring": config.ring_max_angle,
                "pinky": config.pinky_max_angle,
            }
        )

        return jsonify(
            {
                "angles": angles,
                "arduino_connected": arduino is not None and arduino.is_connected,
            }
        )

    @bp.route("/strategy")
    def get_strategy() -> Response:
        """Get current angle calculation strategy and ratio settings.

        Returns:
            JSON response with strategy name and distance ratio bounds.
        """
        camera_stream = current_app.config.get("camera_stream")
        if camera_stream is None:
            return jsonify({
                "strategy": "distance",
                "ratio_min": config.distance_ratio_min,
                "ratio_max": config.distance_ratio_max,
            })
        return jsonify({
            "strategy": camera_stream.strategy,
            "ratio_min": camera_stream.distance_ratio_min,
            "ratio_max": camera_stream.distance_ratio_max,
        })

    @bp.route("/strategy/<name>")
    def set_strategy(name: str) -> Response:
        """Set angle calculation strategy.

        Args:
            name: Strategy name ("distance" or "joint_angle").

        Returns:
            JSON response with success status.
        """
        if name not in ("distance", "joint_angle"):
            return jsonify({"success": False, "error": "Invalid strategy"})

        camera_stream = current_app.config.get("camera_stream")
        if camera_stream is not None:
            camera_stream.strategy = name
        return jsonify({"success": True, "strategy": name})

    @bp.route("/ratio/<float:ratio_min>/<float:ratio_max>")
    def set_ratio(ratio_min: float, ratio_max: float) -> Response:
        """Set distance ratio bounds for distance-based strategy.

        Args:
            ratio_min: Minimum ratio (closed fist), typically ~0.5.
            ratio_max: Maximum ratio (open hand), typically ~2.0.

        Returns:
            JSON response with success status.
        """
        camera_stream = current_app.config.get("camera_stream")
        if camera_stream is not None:
            camera_stream.distance_ratio_min = ratio_min
            camera_stream.distance_ratio_max = ratio_max
        return jsonify({
            "success": True,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
        })

    return bp
