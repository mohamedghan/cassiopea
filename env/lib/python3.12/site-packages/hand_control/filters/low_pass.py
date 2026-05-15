"""Low-pass filter for signal smoothing."""


class LowPassFilter:
    """Simple low-pass filter for smoothing values.

    Uses exponential smoothing with a configurable alpha parameter.
    Lower alpha values result in more smoothing but slower response.
    """

    def __init__(self, alpha: float = 0.15) -> None:
        """Initialize the low-pass filter.

        Args:
            alpha: Smoothing factor between 0 and 1. Lower values = more smoothing.
        """
        self.alpha = alpha
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """Current filtered value."""
        return self._value

    def update(self, new_value: float) -> float:
        """Update the filter with a new value.

        Args:
            new_value: The new input value to filter.

        Returns:
            The filtered output value.
        """
        if self._value is None:
            self._value = new_value
        else:
            self._value = self.alpha * new_value + (1 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        """Reset the filter state."""
        self._value = None
