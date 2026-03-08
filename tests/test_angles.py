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

        angle = calculate_finger_curl(landmarks, [5, 6, 7, 8])
        assert 0 <= angle <= 128


class TestCalculateThumbCurl:
    """Tests for calculate_thumb_curl function."""

    def test_thumb_curl_bounds(self, mock_hand_landmarks):
        """Thumb curl should be within valid servo range."""
        angle = calculate_thumb_curl(mock_hand_landmarks)
        assert 0 <= angle <= 128


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
        """All angles should be within valid servo range."""
        angles = get_all_finger_angles(mock_hand_landmarks)
        for finger, angle in angles.items():
            assert 0 <= angle <= 128, f"{finger} angle {angle} out of range"
