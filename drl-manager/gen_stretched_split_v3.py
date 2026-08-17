#!/usr/bin/env python3
"""Stretch the TimeCAP feature CSVs (windProduction/split/) for the V3 9xxx
turbines, mirroring EXACTLY the simplified/ recipe verified row-by-row on
2026-08-17: out = rows[:12] + repeat(rows[12:12+2500], 6) -> 15012 data rows.
Keeping split/ and simplified/ isomorphic keeps the TimeCAP provider's
sim-step->row mapping aligned with the green series the simulator replays
(the npy-desync bug class). TimeCAP sees 10-min data replayed at 6x speed -
quality degradation is expected and acceptable: timecap IS the imperfect
forecaster arm."""
from pathlib import Path

K = 6
IDS = [12, 36, 95, 91, 96, 101, 103]
SRC = Path("../cloudsimplus-gateway/src/main/resources/windProduction/split")
for tid in IDS:
    for year in (2020, 2021, 2022):
        src = SRC / f"Turbine_{tid}_{year}.csv"
        if not src.exists():
            continue
        lines = src.read_text().strip().split("\n")
        hdr, rows = lines[0], lines[1:]
        out = rows[:12] + [r for row in rows[12:12 + 2500] for r in [row] * K]
        dst = SRC / f"Turbine_{9000 + tid}_{year}.csv"
        dst.write_text(hdr + "\n" + "\n".join(out) + "\n")
        print(f"{src.name} -> {dst.name}: {len(out)} rows")
