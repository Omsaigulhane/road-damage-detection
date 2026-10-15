#!/usr/bin/env python3
"""Summarize a sensor log. Usage: python scripts\\summarize_peaks.py taps.csv [--baseline still.csv]"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--baseline", default=None)
ap.add_argument("--gap", type=float, default=0.5, help="seconds that separate two events")
args = ap.parse_args()


def load(path):
    df = pd.read_csv(path)
    return df.iloc[1:].reset_index(drop=True)  # first window covers everything since boot


df = load(args.log)
t0 = df["laptop_epoch"].iloc[0]
span = df["laptop_epoch"].iloc[-1] - t0
rate = df["window_samples"].sum() / (df["window_ms"].sum() / 1000.0)
print(f"rows: {len(df)}  duration: {span:.1f} s  range: +-{int(df['range_g'].iloc[0])} g")
print(f"board sampling rate: {rate:.0f} samples/s")
print(f"windows with clipping: {int((df['clipped'] == True).sum())}")
print(f"bad reads (last total): {int(df['bad_reads_total'].iloc[-1])}")
p = df["peak_mag_g"]
print(f"peak_mag_g: median {p.median():.3f}  std {p.std():.3f}  min {p.min():.3f}  max {p.max():.3f}")

if args.baseline:
    b = load(args.baseline)["peak_mag_g"]
    thr = max(b.median() + 5 * b.std(), b.max() + 0.1)
    print(f"baseline (still): median {b.median():.3f}  std {b.std():.3f}  max {b.max():.3f}")
else:
    thr = p.median() + 0.5
print(f"event threshold: {thr:.3f} g")

hot = df[p > thr]
events, cur = [], []
for _, r in hot.iterrows():
    if cur and r["laptop_epoch"] - cur[-1]["laptop_epoch"] > args.gap:
        events.append(cur)
        cur = []
    cur.append(r)
if cur:
    events.append(cur)

print(f"events found: {len(events)}")
for i, ev in enumerate(events, 1):
    peak = max(ev, key=lambda r: r["peak_mag_g"])
    clip = any(r["clipped"] == True for r in ev)
    print(f"  {i:2d}  t={ev[0]['laptop_epoch'] - t0:6.1f} s  peak {peak['peak_mag_g']:.2f} g  clipped={clip}")
print(f"GPS fix in {int(df['gps_fix'].sum())} of {len(df)} rows")