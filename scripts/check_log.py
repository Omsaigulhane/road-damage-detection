import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
t0 = df["laptop_epoch"].iloc[0]
low = df[df["accel_mag_g"] < 0.2]
print(f"rows with magnitude < 0.2 g: {len(low)}")
for _, r in low.iterrows():
    print(f"  {r['laptop_epoch'] - t0:7.2f} s  mag {r['accel_mag_g']:.3f}  ax {r['ax_g']} ay {r['ay_g']} az {r['az_g']}")
b = df["board_ms"]
print(f"unique board readings: {b.nunique()} of {len(df)}")
d = b.diff().dropna()
d = d[d > 0]
print(f"board update interval (ms): median {d.median():.0f}  min {d.min():.0f}  max {d.max():.0f}")
g = df["laptop_epoch"].diff().dropna()
print(f"largest gap between our samples: {g.max():.3f} s")