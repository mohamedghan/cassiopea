"""Arduino controller for servo communication."""

import contextlib
import logging
import time
from collections.abc import Sequence

import serial

from hand_control.config import config

logger = logging.getLogger(__name__)


class ArduinoController:
    """Controller for communicating with Arduino via serial.

    Sends finger angle commands to control servos on a robotic hand.
    """

    def __init__(
        self,
        port: str | None = None,
        baud_rate: int | None = None,
    ) -> None:
        """Initialize the Arduino controller.

        Args:
            port: Serial port path. Defaults to config value.
            baud_rate: Serial baud rate. Defaults to config value.
        """
        self._port = port or config.serial_port
        self._baud_rate = baud_rate or config.baud_rate
        self._serial: serial.Serial | None = None

    @property
    def is_connected(self) -> bool:
        """Check if Arduino is connected and port is open."""
        return self._serial is not None and self._serial.is_open

    def connect(self) -> bool:
        """Connect to the Arduino.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._serial = serial.Serial(self._port, self._baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to reset
            print(f"Connected to Arduino on {self._port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            self._serial = None
            return False

    def disconnect(self) -> None:
        """Disconnect from the Arduino."""
        if self._serial is not None:
            with contextlib.suppress(Exception):
                self._serial.close()
            self._serial = None

    def send_all_angles(self, angles: Sequence[int]) -> bool:
        """Send all finger angles at once.

        Args:
            angles: List of 5 angles [thumb, index, middle, ring, pinky].

        Returns:
            True if command sent successfully, False otherwise.
        """
        if not self.is_connected or self._serial is None:
            return False

        try:
            cmd = f"A:{angles[0]},{angles[1]},{angles[2]},{angles[3]},{angles[4]}\n"
            logger.debug("Serial TX: %s", cmd.strip())
            self._serial.write(cmd.encode())
            return True
        except Exception:
            logger.error("Failed to send all-angles command", exc_info=True)
            return False

    def send_finger_angle(self, finger_idx: int, angle: int) -> bool:
        """Send angle for a single finger.

        Args:
            finger_idx: Finger index (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky).
            angle: Servo angle (0-128).

        Returns:
            True if command sent successfully, False otherwise.
        """
        if not self.is_connected or self._serial is None:
            return False

        try:
            cmd = f"F:{finger_idx},{angle}\n"
            logger.debug("Serial TX: %s", cmd.strip())
            self._serial.write(cmd.encode())
            return True
        except Exception:
            logger.error("Failed to send finger-angle command", exc_info=True)
            return False
