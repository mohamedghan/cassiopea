"""Visualization functions for drawing hand landmarks."""

from collections.abc import Sequence

import cv2

from hand_control.config import config
from hand_control.types import FingerAngles, FingerRatios, ImageArray, LandmarkPoint, Vector2D

# Colors for each finger (BGR format for OpenCV)
FINGER_COLORS: list[tuple[int, int, int]] = [
    (255, 0, 255),  # Thumb - Magenta
    (0, 255, 255),  # Index - Cyan
    (0, 255, 0),  # Middle - Green
    (255, 255, 0),  # Ring - Yellow (actually Cyan in BGR)
    (255, 0, 0),  # Pinky - Blue
]

# Finger bone connections (start_landmark, end_landmark)
FINGER_CONNECTIONS: list[list[tuple[int, int]]] = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],  # Thumb
    [(0, 5), (5, 6), (6, 7), (7, 8)],  # Index
    [(0, 9), (9, 10), (10, 11), (11, 12)],  # Middle
    [(0, 13), (13, 14), (14, 15), (15, 16)],  # Ring
    [(0, 17), (17, 18), (18, 19), (19, 20)],  # Pinky
]

# Palm connections
PALM_CONNECTIONS: list[tuple[int, int]] = [(5, 9), (9, 13), (13, 17)]

# Finger names for display
FINGER_DISPLAY_NAMES: list[str] = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_KEYS: list[str] = ["thumb", "index", "middle", "ring", "pinky"]


def draw_landmarks_on_image(
    rgb_image: ImageArray,
    hand_landmarks: Sequence[LandmarkPoint] | None,
    angles: FingerAngles,
    tracked_center: Vector2D | None = None,
    ratios: FingerRatios | None = None,
) -> ImageArray:
    """Draw hand landmarks and angle info on the image.

    Args:
        rgb_image: RGB image array to draw on.
        hand_landmarks: List of 21 hand landmark points, or None if no hand detected.
        angles: Dictionary of finger angles for display.
        tracked_center: Optional center point of tracked hand for indicator.

    Returns:
        Annotated image with landmarks and info drawn.
    """
    if hand_landmarks is None:
        return rgb_image

    annotated_image: ImageArray = rgb_image.copy()
    h, w = annotated_image.shape[:2]

    # Convert landmarks to pixel coordinates
    landmarks_px: list[tuple[int, int]] = []
    for landmark in hand_landmarks:
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        landmarks_px.append((x_px, y_px))

    # Draw finger bones
    for finger_idx, connections in enumerate(FINGER_CONNECTIONS):
        color = FINGER_COLORS[finger_idx]
        for connection in connections:
            start = landmarks_px[connection[0]]
            end = landmarks_px[connection[1]]
            cv2.line(annotated_image, start, end, color, 3)

    # Draw palm connections
    for connection in PALM_CONNECTIONS:
        start = landmarks_px[connection[0]]
        end = landmarks_px[connection[1]]
        cv2.line(annotated_image, start, end, (200, 200, 200), 2)

    # Draw landmark points
    for px in landmarks_px:
        cv2.circle(annotated_image, px, 5, (255, 255, 255), -1)

    # Draw tracking indicator
    if tracked_center is not None:
        center_px = (int(tracked_center[0] * w), int(tracked_center[1] * h))
        cv2.circle(annotated_image, center_px, 15, (0, 255, 0), 2)
        cv2.putText(
            annotated_image,
            "TRACKED",
            (center_px[0] - 40, center_px[1] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Draw angle bars
    y_offset = 30
    for i, (name, key) in enumerate(zip(FINGER_DISPLAY_NAMES, FINGER_KEYS, strict=True)):
        angle = angles.get(key, 0)  # type: ignore[arg-type]
        bar_width = int(angle * 100 / config.max_servo_angle)
        color = FINGER_COLORS[i]

        # Draw filled bar
        cv2.rectangle(
            annotated_image,
            (10, y_offset - 15),
            (10 + bar_width, y_offset + 5),
            color,
            -1,
        )
        # Draw bar outline
        cv2.rectangle(
            annotated_image,
            (10, y_offset - 15),
            (110, y_offset + 5),
            color,
            2,
        )
        # Draw label with ratio
        if ratios is not None and key in ratios:
            ratio_val = ratios[key]
            label = f"{name}: {angle} R:{ratio_val:.2f}"
        else:
            label = f"{name}: {angle}"
        cv2.putText(
            annotated_image,
            label,
            (120, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
        y_offset += 35

    return annotated_image
