#!/usr/bin/env python3
"""Time-stretch the real SDWPF turbine CSVs by k (each data row repeated k times).
Original COMPRESSED playback: 1 row (=10 real min) per sim-second. With k=10 a 6h real
calm becomes a 360 sim-s trough -> deep synchronized troughs collide with tight job
slacks -> the 'will the feast arrive within my slack?' decision becomes frequent and
consequential = forecast load-bearing. Writes Turbine_{7000+id}_2021.csv siblings.
Layout keeps env skip semantics: out = rows[:12] + repeat(rows[12:], k), so sim t=0
still sees original row 12 and sim t maps to original row 12 + t//k."""
import sys
from pathlib import Path
K = 10
IDS = [12, 36, 95, 91, 96]
SRC = Path("../cloudsimplus-gateway/src/main/resources/windProduction/simplified")
for tid in IDS:
    src = SRC / f"Turbine_{tid}_2021.csv"
    lines = src.read_text().strip().split("\n")
    hdr, rows = lines[0], lines[1:]
    out = rows[:12] + [r for row in rows[12:12+1500] for r in [row]*K]  # 1500 orig rows = 15000 sim-s cover
    dst = SRC / f"Turbine_{7000+tid}_2021.csv"
    dst.write_text(hdr + "\n" + "\n".join(out) + "\n")
    print(f"{src.name} -> {dst.name}: {len(out)} rows")
