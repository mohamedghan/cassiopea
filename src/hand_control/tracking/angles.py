"""Angle calculation functions for finger curl detection."""

from collections.abc import Sequence

import numpy as np

from hand_control.config import config
from hand_control.types import FingerAngles, FingerRatios, LandmarkPoint

# Finger landmark indices (MCP, PIP, DIP, TIP)
FINGER_INDICES: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Fingertip landmark indices for distance-based strategy
FINGERTIP_INDICES: dict[str, int] = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}


def calculate_distance(p1: LandmarkPoint, p2: LandmarkPoint) -> float:
    """Calculate 3D Euclidean distance between two landmarks.

    Args:
        p1: First landmark point.
        p2: Second landmark point.

    Returns:
        Euclidean distance between the two points.
    """
    return float(
        np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)
    )


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
    min_angle: int = 0,
    max_angle: int | None = None,
) -> int:
    """Calculate finger curl based on joint angles.

    Args:
        landmarks: List of all hand landmarks.
        finger_indices: Indices for MCP, PIP, DIP, TIP of the finger.
        min_angle: Minimum servo angle (fully curled position).
        max_angle: Maximum servo angle (fully extended position).
            Defaults to config.max_servo_angle when not provided.

    Returns:
        Servo angle clamped to [min_angle, max_angle].
    """
    if max_angle is None:
        max_angle = config.max_servo_angle

    mcp, pip, dip, tip = [landmarks[i] for i in finger_indices]

    pip_angle = calculate_angle(mcp, pip, dip)
    dip_angle = calculate_angle(pip, dip, tip)

    avg_angle = (pip_angle + dip_angle) / 2

    # Map: 60 degrees (curled) -> min_angle, 180 degrees (straight) -> max_angle
    servo_angle = int(np.interp(avg_angle, [60, 180], [min_angle, max_angle]))
    return int(np.clip(servo_angle, min_angle, max_angle))


def calculate_thumb_curl(
    landmarks: Sequence[LandmarkPoint],
    min_angle: int = 0,
    max_angle: int | None = None,
) -> int:
    """Calculate thumb curl based on joint angles.

    Computes the angle at each of the two flexion joints and maps the
    *minimum* (most-bent) of the pair to the servo range.  Using the
    minimum rather than an average means that even partial curl of a
    single joint is immediately reflected in the output, and a clearly
    curled joint cannot be masked by a straight one.

    Joint angle definitions (measured at the *middle* landmark):
        mcp_angle : CMC → MCP → IP    (angle at landmark 2)
        ip_angle  : MCP → IP  → TIP   (angle at landmark 3)

    The minimum of the two is interpolated from the thumb physiological
    range [30°, 160°] to [min_angle, max_angle].

    Args:
        landmarks: List of all hand landmarks (21 points; index 1–4 used).
        min_angle: Minimum servo angle (fully curled position).
        max_angle: Maximum servo angle (fully extended position).
            Defaults to config.max_servo_angle when not provided.

    Returns:
        Servo angle clamped to [min_angle, max_angle].
    """
    if max_angle is None:
        max_angle = config.max_servo_angle

    cmc = landmarks[1]
    mcp = landmarks[2]
    ip = landmarks[3]
    tip = landmarks[4]

    # Angle at landmark 2 (MCP): CMC → MCP → IP
    mcp_angle = calculate_angle(cmc, mcp, ip)
    # Angle at landmark 3 (IP): MCP → IP → TIP
    ip_angle = calculate_angle(mcp, ip, tip)

    # Use the most-bent joint: whichever angle is smaller drives the output.
    # This avoids the average masking a clearly curled joint and is more
    # sensitive to partial curl.
    curl_angle = min(mcp_angle, ip_angle)

    # Thumb physiological range is tighter than finger range [60, 180].
    # Fully curled  ≈ 30°  → min_angle
    # Fully extended ≈ 160° → max_angle
    servo_angle = int(np.interp(curl_angle, [30, 160], [min_angle, max_angle]))
    return int(np.clip(servo_angle, min_angle, max_angle))


def get_all_finger_angles(landmarks: Sequence[LandmarkPoint]) -> FingerAngles:
    """Get curl angles for all fingers.

    Args:
        landmarks: List of all hand landmarks (21 points).

    Returns:
        Dictionary mapping finger names to servo angles.
    """
    return FingerAngles(
        thumb=calculate_thumb_curl(
            landmarks,
            min_angle=config.thumb_min_angle,
            max_angle=config.thumb_max_angle,
        ),
        index=calculate_finger_curl(
            landmarks,
            FINGER_INDICES["index"],
            min_angle=config.index_min_angle,
            max_angle=config.index_max_angle,
        ),
        middle=calculate_finger_curl(
            landmarks,
            FINGER_INDICES["middle"],
            min_angle=config.middle_min_angle,
            max_angle=config.middle_max_angle,
        ),
        ring=calculate_finger_curl(
            landmarks,
            FINGER_INDICES["ring"],
            min_angle=config.ring_min_angle,
            max_angle=config.ring_max_angle,
        ),
        pinky=calculate_finger_curl(
            landmarks,
            FINGER_INDICES["pinky"],
            min_angle=config.pinky_min_angle,
            max_angle=config.pinky_max_angle,
        ),
    )


def get_all_finger_angles_distance(
    landmarks: Sequence[LandmarkPoint],
    ratio_min: float | None = None,
    ratio_max: float | None = None,
) -> FingerAngles:
    """Get servo angles using normalized distance-based strategy.

    Computes the ratio of fingertip-to-wrist distance divided by palm scale
    (wrist to middle MCP), then maps this ratio to servo angles.

    This approach is computationally efficient (O(1), no trig functions),
    depth-invariant, and immune to gimbal lock.

    Args:
        landmarks: List of all hand landmarks (21 points).
        ratio_min: Minimum expected ratio (closed fist). Defaults to config value.
        ratio_max: Maximum expected ratio (open hand). Defaults to config value.

    Returns:
        Dictionary mapping finger names to servo angles.
    """
    if ratio_min is None:
        ratio_min = config.distance_ratio_min
    if ratio_max is None:
        ratio_max = config.distance_ratio_max

    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # Palm scale: distance between wrist and middle finger base
    palm_scale = calculate_distance(wrist, middle_mcp)
    if palm_scale < 1e-6:
        # Avoid division by zero; return default open position
        return FingerAngles(
            thumb=config.thumb_max_angle,
            index=config.index_max_angle,
            middle=config.middle_max_angle,
            ring=config.ring_max_angle,
            pinky=config.pinky_max_angle,
        )

    # Per-finger servo ranges
    finger_ranges = {
        "thumb": (config.thumb_min_angle, config.thumb_max_angle),
        "index": (config.index_min_angle, config.index_max_angle),
        "middle": (config.middle_min_angle, config.middle_max_angle),
        "ring": (config.ring_min_angle, config.ring_max_angle),
        "pinky": (config.pinky_min_angle, config.pinky_max_angle),
    }

    angles: dict[str, int] = {}
    for finger_name, tip_idx in FINGERTIP_INDICES.items():
        fingertip = landmarks[tip_idx]
        distance = calculate_distance(wrist, fingertip)
        ratio = distance / palm_scale

        min_angle, max_angle = finger_ranges[finger_name]
        # Map ratio to servo angle: ratio_min -> min_angle, ratio_max -> max_angle
        servo_angle = int(np.interp(ratio, [ratio_min, ratio_max], [min_angle, max_angle]))
        angles[finger_name] = int(np.clip(servo_angle, min_angle, max_angle))

    return FingerAngles(
        thumb=angles["thumb"],
        index=angles["index"],
        middle=angles["middle"],
        ring=angles["ring"],
        pinky=angles["pinky"],
    )


def get_all_finger_ratios(landmarks: Sequence[LandmarkPoint]) -> FingerRatios:
    """Get normalized distance ratios for all fingers.

    Computes the ratio of fingertip-to-wrist distance divided by palm scale
    (wrist to middle MCP). This is the raw ratio before mapping to servo angles.

    Args:
        landmarks: List of all hand landmarks (21 points).

    Returns:
        Dictionary mapping finger names to distance ratios.
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    palm_scale = calculate_distance(wrist, middle_mcp)

    if palm_scale < 1e-6:
        return FingerRatios(thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0)

    ratios = FingerRatios(thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0)
    for finger_name, tip_idx in FINGERTIP_INDICES.items():
        fingertip = landmarks[tip_idx]
        distance = calculate_distance(wrist, fingertip)
        ratios[finger_name] = float(np.clip(distance / palm_scale, 0.0, 10.0))

    return ratios
