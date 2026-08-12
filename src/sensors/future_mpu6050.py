"""
Future Phase-2 hardware providers.

These are intentionally UNIMPLEMENTED in Phase 1 (Section 3, 43). They exist
only to document the integration point so Phase 2 does not require an
application rewrite. Do not instantiate these in Phase-1 code paths.
"""
from __future__ import annotations

from typing import Optional

from src.sensors.base import (
    VibrationSensorProvider,
    GPSSensorProvider,
    VibrationReading,
    GPSReading,
)


class MPU6050Provider(VibrationSensorProvider):
    """
    Phase-2 target implementation.

    Will read real accelerometer/gyroscope data from an MPU6050 over I2C
    (typically via a Raspberry Pi) and compute a vibration magnitude to feed
    into the severity engine's future `vibration` input (Section 21).

    NOT IMPLEMENTED in Phase 1 — no MPU6050 hardware is available.
    """

    def __init__(self, i2c_address: int = 0x68):
        self.i2c_address = i2c_address
        raise NotImplementedError(
            "MPU6050Provider is a Phase-2 placeholder. No hardware is available in Phase 1."
        )

    def is_available(self) -> bool:
        return False

    def read(self) -> Optional[VibrationReading]:
        return None


class GPSNeo6MProvider(GPSSensorProvider):
    """
    Phase-2 target implementation.

    Will read NMEA sentences from a Neo-6M GPS module (typically via UART
    on a Raspberry Pi) and provide latitude/longitude/timestamp for the
    database's future geolocation fields (Section 27, 43).

    NOT IMPLEMENTED in Phase 1 — no GPS hardware is available.
    """

    def __init__(self, serial_port: str = "/dev/ttyS0", baud_rate: int = 9600):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        raise NotImplementedError(
            "GPSNeo6MProvider is a Phase-2 placeholder. No hardware is available in Phase 1."
        )

    def is_available(self) -> bool:
        return False

    def read(self) -> Optional[GPSReading]:
        return None
