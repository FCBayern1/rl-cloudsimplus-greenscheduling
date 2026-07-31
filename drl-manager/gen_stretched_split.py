#!/usr/bin/env python3
"""Stretch the TimeCAP input CSVs (windProduction/split/, full SDWPF feature files) by k=10,
same recipe as gen_stretched_wind.py (rows[:12] + repeat(rows[12:1512], 10)) so the TimeCAP
provider's sim-step->row mapping stays aligned with the stretched simplified/ green files.
NOTE: TimeCAP now sees 10-min-resolution data replayed at 1-min steps (slower dynamics than
it was trained on) -> some forecast-quality degradation is expected and acceptable: timecap
IS the imperfect-forecaster arm; godeye carries the perfect upper bound."""
from pathlib import Path
K=10
SRC=Path("../cloudsimplus-gateway/src/main/resources/windProduction/split")
for tid in [12,36,95,91,96]:
    for year in [2020,2021,2022]:
        src=SRC/f"Turbine_{tid}_{year}.csv"
        if not src.exists(): continue
        lines=src.read_text().strip().split("\n")
        hdr,rows=lines[0],lines[1:]
        out=rows[:12]+[r for row in rows[12:12+1500] for r in [row]*K]
        (SRC/f"Turbine_{7000+tid}_{year}.csv").write_text(hdr+"\n"+"\n".join(out)+"\n")
        print(f"Turbine_{7000+tid}_{year}.csv: {len(out)} rows")
