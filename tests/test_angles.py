"""Tests for angle calculation functions."""

import math

from hand_control.tracking.angles import (
    calculate_angle,
    calculate_finger_curl,
    calculate_thumb_curl,
    get_all_finger_angles,
)
from tests.conftest import MockLandmark


class TestCalculateAngle:
    """Tests for calculate_angle function."""

    def test_straight_angle(self):
        """Three points in a line should give 180 degrees."""
        p1 = MockLandmark(0.0, 0.0, 0.0)
        p2 = MockLandmark(0.5, 0.0, 0.0)
        p3 = MockLandmark(1.0, 0.0, 0.0)
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 180.0) < 1.0

    def test_right_angle(self):
        """Perpendicular vectors should give 90 degrees."""
        p1 = MockLandmark(0.0, 1.0, 0.0)
        p2 = MockLandmark(0.0, 0.0, 0.0)
        p3 = MockLandmark(1.0, 0.0, 0.0)
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 90.0) < 1.0

    def test_acute_angle(self):
        """Test 60 degree angle."""
        # Create 60 degree angle
        p1 = MockLandmark(1.0, 0.0, 0.0)
        p2 = MockLandmark(0.0, 0.0, 0.0)
        p3 = MockLandmark(0.5, math.sqrt(3) / 2, 0.0)
        angle = calculate_angle(p1, p2, p3)
        assert abs(angle - 60.0) < 1.0


class TestCalculateFingerCurl:
    """Tests for calculate_finger_curl function."""

    def test_straight_finger_high_angle(self, mock_straight_finger_landmarks):
        """Straight finger should have high servo angle (extended)."""
        # Create full hand landmarks with straight index
        landmarks = [MockLandmark(0.0, 0.0, 0.0)] * 21
        # Index finger landmarks at indices 5, 6, 7, 8
        for i, landmark in enumerate(mock_straight_finger_landmarks):
            landmarks[5 + i] = landmark

        angle = calculate_finger_curl(landmarks, [5, 6, 7, 8])
        # Straight finger should give high angle (closer to 128)
        assert angle > 64  # At least half open

    def test_curled_finger_low_angle(self, mock_curled_finger_landmarks):
        """Curled finger should have low servo angle (closed)."""
        landmarks = [MockLandmark(0.0, 0.0, 0.0)] * 21
        for i, landmark in enumerate(mock_curled_finger_landmarks):
            landmarks[5 + i] = landmark

        angle = calculate_finger_curl(landmarks, [5, 6, 7, 8])
        # Curled finger should give lower angle
        assert angle < 100

    def test_angle_bounds(self, mock_straight_finger_landmarks):
        """Output should be within valid servo range."""
        landmarks = [MockLandmark(0.0, 0.0, 0.0)] * 21
        for i, landmark in enumerate(mock_straight_finger_landmarks):
            landmarks[5 + i] = landmark

        angle = calculate_finger_curl(landmarks, [5, 6, 7, 8], min_angle=36, max_angle=120)
        assert 36 <= angle <= 120

    def test_angle_bounds_with_defaults(self, mock_straight_finger_landmarks):
        """Output with default args should be within [0, max_servo_angle]."""
        from hand_control.config import config

        landmarks = [MockLandmark(0.0, 0.0, 0.0)] * 21
        for i, landmark in enumerate(mock_straight_finger_landmarks):
            landmarks[5 + i] = landmark

        angle = calculate_finger_curl(landmarks, [5, 6, 7, 8])
        assert 0 <= angle <= config.max_servo_angle


class TestCalculateThumbCurl:
    """Tests for calculate_thumb_curl function."""

    def test_thumb_curl_bounds(self, mock_hand_landmarks):
        """Thumb curl should be within valid servo range."""
        angle = calculate_thumb_curl(mock_hand_landmarks, min_angle=52, max_angle=180)
        assert 52 <= angle <= 180

    def test_thumb_curl_bounds_with_defaults(self, mock_hand_landmarks):
        """Thumb curl with defaults should be within [0, max_servo_angle]."""
        from hand_control.config import config

        angle = calculate_thumb_curl(mock_hand_landmarks)
        assert 0 <= angle <= config.max_servo_angle

    def test_straight_thumb_gives_high_angle(self, mock_straight_thumb_landmarks):
        """Fully extended (collinear) thumb should produce maximum servo angle."""
        angle = calculate_thumb_curl(
            mock_straight_thumb_landmarks, min_angle=52, max_angle=180
        )
        # All joint angles ≈ 180° → weighted avg clips to max end of range
        assert angle == 180

    def test_curled_thumb_gives_low_angle(self, mock_curled_thumb_landmarks):
        """Strongly curled thumb should produce a servo angle well below maximum."""
        angle = calculate_thumb_curl(
            mock_curled_thumb_landmarks, min_angle=52, max_angle=180
        )
        # Weighted avg ≈ 79° → significantly lower than max
        assert angle < 130

    def test_straight_thumb_higher_than_curled(
        self, mock_straight_thumb_landmarks, mock_curled_thumb_landmarks
    ):
        """Straight thumb servo angle must exceed curled thumb servo angle."""
        straight_angle = calculate_thumb_curl(
            mock_straight_thumb_landmarks, min_angle=52, max_angle=180
        )
        curled_angle = calculate_thumb_curl(
            mock_curled_thumb_landmarks, min_angle=52, max_angle=180
        )
        assert straight_angle > curled_angle

    def test_min_joint_drives_curl_detection(self):
        """The more-bent joint must drive the output regardless of which one it is.

        Build two hands where one has a bent MCP and straight IP, and the
        other has a straight MCP and bent IP.  Both should produce a lower
        servo angle than a fully straight thumb.
        """
        from tests.conftest import MockLandmark

        # Fully straight reference
        straight = [MockLandmark(0.0, 0.0, 0.0)] * 21
        straight[1] = MockLandmark(0.4, 0.7, 0.0)  # CMC
        straight[2] = MockLandmark(0.3, 0.6, 0.0)  # MCP collinear
        straight[3] = MockLandmark(0.2, 0.5, 0.0)  # IP  collinear
        straight[4] = MockLandmark(0.1, 0.4, 0.0)  # TIP collinear
        straight_angle = calculate_thumb_curl(straight, min_angle=0, max_angle=180)

        # Bent MCP, straight IP
        bent_mcp = [MockLandmark(0.0, 0.0, 0.0)] * 21
        bent_mcp[1] = MockLandmark(0.4, 0.7, 0.0)   # CMC
        bent_mcp[2] = MockLandmark(0.3, 0.6, 0.0)   # MCP
        bent_mcp[3] = MockLandmark(0.4, 0.5, 0.0)   # IP  (bent at MCP)
        bent_mcp[4] = MockLandmark(0.5, 0.4, 0.0)   # TIP (IP stays straight)
        bent_mcp_angle = calculate_thumb_curl(bent_mcp, min_angle=0, max_angle=180)

        # Bent IP, straight MCP
        bent_ip = [MockLandmark(0.0, 0.0, 0.0)] * 21
        bent_ip[1] = MockLandmark(0.4, 0.7, 0.0)   # CMC
        bent_ip[2] = MockLandmark(0.3, 0.6, 0.0)   # MCP
        bent_ip[3] = MockLandmark(0.2, 0.5, 0.0)   # IP  collinear with MCP
        bent_ip[4] = MockLandmark(0.3, 0.4, 0.0)   # TIP (bent at IP)
        bent_ip_angle = calculate_thumb_curl(bent_ip, min_angle=0, max_angle=180)

        assert bent_mcp_angle < straight_angle, (
            "Bent MCP should give lower angle than fully straight thumb."
        )
        assert bent_ip_angle < straight_angle, (
            "Bent IP should give lower angle than fully straight thumb."
        )


class TestGetAllFingerAngles:
    """Tests for get_all_finger_angles function."""

    def test_returns_all_fingers(self, mock_hand_landmarks):
        """Should return angles for all five fingers."""
        angles = get_all_finger_angles(mock_hand_landmarks)
        assert "thumb" in angles
        assert "index" in angles
        assert "middle" in angles
        assert "ring" in angles
        assert "pinky" in angles

    def test_all_angles_in_range(self, mock_hand_landmarks):
        """All angles should be within their per-finger servo range."""
        from hand_control.config import config

        finger_ranges = {
            "thumb": (config.thumb_min_angle, config.thumb_max_angle),
            "index": (config.index_min_angle, config.index_max_angle),
            "middle": (config.middle_min_angle, config.middle_max_angle),
            "ring": (config.ring_min_angle, config.ring_max_angle),
            "pinky": (config.pinky_min_angle, config.pinky_max_angle),
        }
        angles = get_all_finger_angles(mock_hand_landmarks)
        for finger, angle in angles.items():
            lo, hi = finger_ranges[finger]
            assert lo <= angle <= hi, f"{finger} angle {angle} outside [{lo}, {hi}]"
