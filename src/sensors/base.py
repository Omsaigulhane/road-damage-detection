"""
SensorProvider interface (Section 4).

Phase 1 has no physical sensors. This interface exists purely so that
future hardware (MPU6050, GPS) can be plugged into the severity engine
and database layer WITHOUT rewriting the application.

No implementation in this file fabricates sensor data. Phase-1 code
paths must treat sensor providers as absent/None.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class VibrationReading:
    magnitude: float          # e.g. m/s^2 or normalized units — defined when hardware exists
    timestamp: float
    is_simulated: bool = False


@dataclass
class GPSReading:
    latitude: float
    longitude: float
    timestamp: float
    is_simulated: bool = False


class SensorProvider(ABC):
    """Base interface all sensor providers must implement."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True only if real hardware is connected and initialized."""
        raise NotImplementedError


class VibrationSensorProvider(SensorProvider):
    @abstractmethod
    def read(self) -> Optional[VibrationReading]:
        """Return a real vibration reading, or None if unavailable.
        Implementations must NEVER return fabricated values."""
        raise NotImplementedError


class GPSSensorProvider(SensorProvider):
    @abstractmethod
    def read(self) -> Optional[GPSReading]:
        """Return a real GPS fix, or None if unavailable.
        Implementations must NEVER return fabricated coordinates."""
        raise NotImplementedError


class NullVibrationProvider(VibrationSensorProvider):
    """Phase-1 default: no vibration hardware present."""

    def is_available(self) -> bool:
        return False

    def read(self) -> Optional[VibrationReading]:
        return None


class NullGPSProvider(GPSSensorProvider):
    """Phase-1 default: no GPS hardware present."""

    def is_available(self) -> bool:
        return False

    def read(self) -> Optional[GPSReading]:
        return None
