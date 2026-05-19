"""Tests for signal filtering modules."""

from hand_control.filters import LowPassFilter


class TestLowPassFilter:
    """Tests for LowPassFilter."""

    def test_initial_value(self):
        """First update should return the input value."""
        f = LowPassFilter(alpha=0.5)
        assert f.update(100.0) == 100.0

    def test_smoothing(self):
        """Filter should smooth values over time."""
        f = LowPassFilter(alpha=0.5)
        f.update(0.0)
        result = f.update(100.0)
        # With alpha=0.5: 0.5 * 100 + 0.5 * 0 = 50
        assert result == 50.0

    def test_low_alpha_more_smoothing(self):
        """Lower alpha should result in more smoothing (slower response)."""
        f = LowPassFilter(alpha=0.1)
        f.update(0.0)
        result = f.update(100.0)
        # With alpha=0.1: 0.1 * 100 + 0.9 * 0 = 10
        assert result == 10.0

    def test_high_alpha_less_smoothing(self):
        """Higher alpha should result in less smoothing (faster response)."""
        f = LowPassFilter(alpha=0.9)
        f.update(0.0)
        result = f.update(100.0)
        # With alpha=0.9: 0.9 * 100 + 0.1 * 0 = 90
        assert result == 90.0

    def test_reset(self):
        """Reset should clear the filter state."""
        f = LowPassFilter(alpha=0.5)
        f.update(100.0)
        f.reset()
        assert f.value is None
        # After reset, should behave like initial call
        assert f.update(50.0) == 50.0

