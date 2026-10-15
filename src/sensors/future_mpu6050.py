"""
Real HTTP-based sensor providers for the two-board ESP32 setup:
- Sensor ESP32 serves /sensors as JSON: {accel_g, gyro_dps, gps}

This replaces the earlier NotImplementedError stubs now that hardware
is bench-verified and serving real data.
"""
from __future__ import annotations

import time
from typing import Optional

import requests

from src.sensors.base import (
    VibrationSensorProvider,
    GPSSensorProvider,
    VibrationReading,
    GPSReading,
)


class ESP32SensorBoard:
    """Shared HTTP client for the sensor ESP32's /sensors endpoint."""

    def __init__(self, base_url: str, timeout: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(self) -> dict:
        resp = requests.get(f"{self.base_url}/sensors", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()


class MPU6050Provider(VibrationSensorProvider):
    """Reads live accelerometer data from the sensor ESP32 over HTTP."""

    def __init__(self, board: ESP32SensorBoard):
        self.board = board
        self._last_ok = True

    def is_available(self) -> bool:
        return self._last_ok

    def read(self) -> Optional[VibrationReading]:
        try:
            data = self.board.fetch()
            accel = data["accel_g"]
            magnitude = (accel["x"] ** 2 + accel["y"] ** 2 + accel["z"] ** 2) ** 0.5
            self._last_ok = True
            return VibrationReading(magnitude=magnitude, timestamp=time.time(), is_simulated=False)
        except Exception:
            self._last_ok = False
            return None


class GPSNeo6MProvider(GPSSensorProvider):
    """Reads live GPS data from the sensor ESP32 over HTTP."""

    def __init__(self, board: ESP32SensorBoard):
        self.board = board
        self._last_ok = True

    def is_available(self) -> bool:
        return self._last_ok

    def read(self) -> Optional[GPSReading]:
        try:
            data = self.board.fetch()
            gps = data["gps"]
            self._last_ok = True
            if not gps.get("fix"):
                return None
            return GPSReading(
                latitude=gps["lat"],
                longitude=gps["lng"],
                timestamp=time.time(),
                is_simulated=False,
            )
        except Exception:
            self._last_ok = False
            return None