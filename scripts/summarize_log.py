#!/usr/bin/env python3
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
m = df["accel_mag_g"]
base = m.median()
print(f"samples: {len(df)}  duration: {df['laptop_epoch'].iloc[-1] - df['laptop_epoch'].iloc[0]:.1f} s")
print(f"accel magnitude (g): median {base:.3f}  std {m.std():.3f}  min {m.min():.3f}  max {m.max():.3f}")
thr = base + 0.25
above = df[m > thr]
print(f"samples more than 0.25 g above median: {len(above)}")
t0 = df["laptop_epoch"].iloc[0]
print("top 8 peaks (seconds from start, magnitude g):")
for _, r in df.nlargest(8, "accel_mag_g").sort_values("laptop_epoch").iterrows():
    print(f"  {r['laptop_epoch'] - t0:7.2f} s   {r['accel_mag_g']:.3f}")
print(f"GPS fix in {int(df['gps_fix'].sum())} of {len(df)} samples")