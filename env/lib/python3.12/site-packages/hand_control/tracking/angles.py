"""Angle calculation functions for finger curl detection."""

from collections.abc import Sequence

import numpy as np

from hand_control.config import config
from hand_control.types import FingerAngles, LandmarkPoint

# Finger landmark indices (MCP, PIP, DIP, TIP)
FINGER_INDICES: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def calculate_angle(p1: LandmarkPoint, p2: LandmarkPoint, p3: LandmarkPoint) -> float:
    """Calculate angle between three points in degrees.

    The angle is measured at p2, between vectors p2->p1 and p2->p3.

    Args:
        p1: First point (start of first vector).
        p2: Middle point (vertex of angle).
        p3: Third point (end of second vector).

    Returns:
        Angle in degrees between the two vectors.
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle)
    return float(np.degrees(angle))


def calculate_finger_curl(
    landmarks: Sequence[LandmarkPoint],
    finger_indices: Sequence[int],
) -> int:
    """Calculate finger curl based on joint angles.

    Args:
        landmarks: List of all hand landmarks.
        finger_indices: Indices for MCP, PIP, DIP, TIP of the finger.

    Returns:
        Servo angle 0-128 where 0 = fully curled, 128 = fully extended.
    """
    mcp, pip, dip, tip = [landmarks[i] for i in finger_indices]

    pip_angle = calculate_angle(mcp, pip, dip)
    dip_angle = calculate_angle(pip, dip, tip)

    avg_angle = (pip_angle + dip_angle) / 2

    # Map: 60 degrees (curled) -> 0, 180 degrees (straight) -> max_servo_angle
    servo_angle = int(np.interp(avg_angle, [60, 180], [0, config.max_servo_angle]))
    return int(np.clip(servo_angle, 0, config.max_servo_angle))


def calculate_thumb_curl(landmarks: Sequence[LandmarkPoint]) -> int:
    """Calculate thumb curl based on joint angles.

    Args:
        landmarks: List of all hand landmarks.

    Returns:
        Servo angle 0-128 where 0 = fully curled, 128 = fully extended.
    """
    cmc = landmarks[1]
    mcp = landmarks[2]
    ip = landmarks[3]
    tip = landmarks[4]

    mcp_angle = calculate_angle(cmc, mcp, ip)
    ip_angle = calculate_angle(mcp, ip, tip)

    avg_angle = (mcp_angle + ip_angle) / 2
    servo_angle = int(np.interp(avg_angle, [60, 180], [0, config.max_servo_angle]))
    return int(np.clip(servo_angle, 0, config.max_servo_angle))


def get_all_finger_angles(landmarks: Sequence[LandmarkPoint]) -> FingerAngles:
    """Get curl angles for all fingers.

    Args:
        landmarks: List of all hand landmarks (21 points).

    Returns:
        Dictionary mapping finger names to servo angles.
    """
    return FingerAngles(
        thumb=calculate_thumb_curl(landmarks),
        index=calculate_finger_curl(landmarks, FINGER_INDICES["index"]),
        middle=calculate_finger_curl(landmarks, FINGER_INDICES["middle"]),
        ring=calculate_finger_curl(landmarks, FINGER_INDICES["ring"]),
        pinky=calculate_finger_curl(landmarks, FINGER_INDICES["pinky"]),
    )
