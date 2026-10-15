#!/usr/bin/env python3
"""Poll the sensor ESP32 and write a timestamped CSV (laptop clock)."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import get_config  # noqa: E402

HEADER = ["laptop_epoch", "board_ms", "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps",
          "accel_mag_g", "peak_mag_g", "range_g", "window_ms", "window_samples", "clipped",
          "bad_reads", "bad_reads_total", "gps_fix", "lat", "lng", "satellites",
          "gps_age_ms", "speed_kmph"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="Sensor board base URL")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    url = (args.url or get_config().get("hardware", {}).get("sensor_board_url") or "").rstrip("/")
    if url.endswith("/sensors"):
        url = url[: -len("/sensors")]
    if not url:
        sys.exit("No sensor board URL. Pass --url or set hardware.sensor_board_url in config.yaml")

    out = Path(args.out) if args.out else REPO_ROOT / "outputs" / "drive_logs" / f"sensors_{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    period = 1.0 / args.hz
    ok = fail = 0
    start = time.time()
    next_t = start
    print(f"Logging {url}/sensors at ~{args.hz} Hz -> {out}\nCtrl+C to stop.")

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        try:
            while True:
                t0 = time.time()
                try:
                    r = requests.get(f"{url}/sensors", timeout=1.0)
                    r.raise_for_status()
                    d = r.json()
                    a, g, gps = d["accel_g"], d["gyro_dps"], d["gps"]
                    mag = (a["x"] ** 2 + a["y"] ** 2 + a["z"] ** 2) ** 0.5
                    t_mid = (t0 + time.time()) / 2
                    w.writerow([f"{t_mid:.3f}", d.get("timestamp_ms"), a["x"], a["y"], a["z"],
                                g["x"], g["y"], g["z"], f"{mag:.4f}",
                                d.get("peak_mag_g"), d.get("accel_range_g"), d.get("window_ms"),
                                d.get("window_samples"), d.get("clipped"), d.get("bad_reads"),
                                d.get("bad_reads_total"), int(bool(gps.get("fix"))), gps.get("lat"),
                                gps.get("lng"), gps.get("satellites"), gps.get("age_ms"),
                                gps.get("speed_kmph")])
                    f.flush()
                    ok += 1
                except Exception as e:
                    fail += 1
                    print(f"read failed ({fail}): {e}")
                if (ok + fail) % 50 == 0:
                    print(f"{ok} ok, {fail} failed, {ok / max(time.time() - start, 1e-9):.1f} Hz achieved")
                next_t += period
                time.sleep(max(0.0, next_t - time.time()))
        except KeyboardInterrupt:
            pass

    print(f"Done: {ok} samples, {fail} failures, {ok / max(time.time() - start, 1e-9):.1f} Hz achieved. File: {out}")


if __name__ == "__main__":
    main()