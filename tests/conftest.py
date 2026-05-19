"""Pytest configuration and fixtures."""

from dataclasses import dataclass

import pytest


@dataclass
class MockLandmark:
    """Mock landmark point for testing."""

    x: float
    y: float
    z: float


@pytest.fixture
def mock_straight_finger_landmarks():
    """Create landmarks for a straight finger (fully extended).

    Returns landmarks where the angle between joints is ~180 degrees.
    """
    # Simulate a straight index finger (landmarks 5, 6, 7, 8)
    return [
        MockLandmark(0.0, 0.0, 0.0),  # MCP
        MockLandmark(0.0, -0.1, 0.0),  # PIP
        MockLandmark(0.0, -0.2, 0.0),  # DIP
        MockLandmark(0.0, -0.3, 0.0),  # TIP
    ]


@pytest.fixture
def mock_curled_finger_landmarks():
    """Create landmarks for a curled finger.

    Returns landmarks where the angle between joints is ~60 degrees.
    """
    # Simulate a curled finger
    return [
        MockLandmark(0.0, 0.0, 0.0),  # MCP
        MockLandmark(0.0, -0.1, 0.0),  # PIP
        MockLandmark(0.05, -0.05, 0.0),  # DIP (angled)
        MockLandmark(0.1, 0.0, 0.0),  # TIP (curled back)
    ]


@pytest.fixture
def mock_hand_landmarks():
    """Create a full set of 21 hand landmarks for testing.

    Returns landmarks simulating an open hand with straight fingers.
    """
    landmarks = []

    # Wrist (0)
    landmarks.append(MockLandmark(0.5, 0.8, 0.0))

    # Thumb (1-4): CMC, MCP, IP, TIP
    landmarks.append(MockLandmark(0.4, 0.7, 0.0))
    landmarks.append(MockLandmark(0.35, 0.6, 0.0))
    landmarks.append(MockLandmark(0.3, 0.5, 0.0))
    landmarks.append(MockLandmark(0.25, 0.4, 0.0))

    # Index (5-8): MCP, PIP, DIP, TIP
    landmarks.append(MockLandmark(0.45, 0.6, 0.0))
    landmarks.append(MockLandmark(0.45, 0.5, 0.0))
    landmarks.append(MockLandmark(0.45, 0.4, 0.0))
    landmarks.append(MockLandmark(0.45, 0.3, 0.0))

    # Middle (9-12): MCP, PIP, DIP, TIP
    landmarks.append(MockLandmark(0.5, 0.55, 0.0))
    landmarks.append(MockLandmark(0.5, 0.45, 0.0))
    landmarks.append(MockLandmark(0.5, 0.35, 0.0))
    landmarks.append(MockLandmark(0.5, 0.25, 0.0))

    # Ring (13-16): MCP, PIP, DIP, TIP
    landmarks.append(MockLandmark(0.55, 0.6, 0.0))
    landmarks.append(MockLandmark(0.55, 0.5, 0.0))
    landmarks.append(MockLandmark(0.55, 0.4, 0.0))
    landmarks.append(MockLandmark(0.55, 0.3, 0.0))

    # Pinky (17-20): MCP, PIP, DIP, TIP
    landmarks.append(MockLandmark(0.6, 0.65, 0.0))
    landmarks.append(MockLandmark(0.6, 0.55, 0.0))
    landmarks.append(MockLandmark(0.6, 0.45, 0.0))
    landmarks.append(MockLandmark(0.6, 0.35, 0.0))

    return landmarks


@pytest.fixture
def mock_straight_thumb_landmarks():
    """Full 21-landmark hand with a fully extended (straight) thumb.

    Landmarks 0-4 are collinear so every joint angle is 180°.
    The remaining landmarks are neutral zeros.
    """
    landmarks = [MockLandmark(0.0, 0.0, 0.0)] * 21
    # Straight line: wrist → CMC → MCP → IP → TIP
    landmarks[0] = MockLandmark(0.5, 0.8, 0.0)   # WRIST
    landmarks[1] = MockLandmark(0.4, 0.7, 0.0)   # THUMB_CMC
    landmarks[2] = MockLandmark(0.3, 0.6, 0.0)   # THUMB_MCP  (collinear)
    landmarks[3] = MockLandmark(0.2, 0.5, 0.0)   # THUMB_IP   (collinear)
    landmarks[4] = MockLandmark(0.1, 0.4, 0.0)   # THUMB_TIP  (collinear)
    return landmarks


@pytest.fixture
def mock_curled_thumb_landmarks():
    """Full 21-landmark hand with a strongly curled thumb.

    The CMC angle (wrist→CMC→MCP) is ≈45°, MCP angle ≈135°, IP angle ≈90°,
    giving a weighted average well below the straight-thumb value.
    """
    landmarks = [MockLandmark(0.0, 0.0, 0.0)] * 21
    landmarks[0] = MockLandmark(0.5, 0.8, 0.0)    # WRIST
    landmarks[1] = MockLandmark(0.4, 0.7, 0.0)    # THUMB_CMC
    landmarks[2] = MockLandmark(0.5, 0.7, 0.0)    # THUMB_MCP  (bent back, CMC ≈ 45°)
    landmarks[3] = MockLandmark(0.55, 0.65, 0.0)  # THUMB_IP   (MCP ≈ 135°)
    landmarks[4] = MockLandmark(0.5, 0.6, 0.0)    # THUMB_TIP  (IP ≈ 90°)
    return landmarks
