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
