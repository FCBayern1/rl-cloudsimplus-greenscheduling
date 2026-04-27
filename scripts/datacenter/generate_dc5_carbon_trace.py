#!/usr/bin/env python3
"""
Generate a bursty workload trace for the 5-DC carbon-aware routing experiment.

Target environment:
  - 5 DCs, total 1920 PEs, VM MIPS = 40000
  - global_routing_batch_size = 20, max_episode_length = 3000
  - Routing capacity: 20 cloudlets/step * 3000 steps = 60,000 slots
  - Target ~40,000 cloudlets (avg arrival ~20/sec over 2000s)
  - Natural workload mix (mean ~800K MI, mean ~2.4 PEs)

Utilisation estimates:
  concurrent_PEs = arrival_rate * mean_pes * mean_runtime
  mean_runtime = 800K / 40K = 20s
  avg (20/s): 20 * 2.4 * 20 = 960 PEs -> 50%
  burst (28/s): 28 * 2.4 * 20 = 1344 -> 70%
  valley (12/s): 12 * 2.4 * 20 = 576 -> 30%

Output: traces/dc5_carbon_bursty.csv
"""

import csv
import random
import os

SEED = 42
random.seed(SEED)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cloudsimplus-gateway", "src", "main", "resources", "traces"
)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dc5_carbon_bursty.csv")

ARRIVAL_SPAN = 2000
PHASE_DURATION = 100

HIGH_RATE = 28
LOW_RATE = 12

MI_MIN = 200_000
MI_MAX = 2_000_000
MI_MEAN = 800_000
MI_STD = 400_000

PE_WEIGHTS = {1: 30, 2: 35, 3: 15, 4: 10, 5: 5, 6: 3, 7: 1, 8: 1}

def sample_pes():
    pop = list(PE_WEIGHTS.keys())
    weights = list(PE_WEIGHTS.values())
    return random.choices(pop, weights=weights, k=1)[0]

def sample_mi():
    mi = int(random.gauss(MI_MEAN, MI_STD))
    return max(MI_MIN, min(MI_MAX, mi))

def get_rate(t):
    phase_idx = int(t / PHASE_DURATION)
    if phase_idx % 2 == 0:
        return HIGH_RATE
    else:
        return LOW_RATE

def main():
    cloudlets = []
    cid = 0
    t = 0.0

    while t < ARRIVAL_SPAN:
        rate = get_rate(t)
        inter_arrival = random.expovariate(rate)
        t += inter_arrival
        if t >= ARRIVAL_SPAN:
            break

        arrival_sec = int(t)
        mi = sample_mi()
        pes = sample_pes()
        file_size = mi // 1000
        output_size = file_size // 2

        cloudlets.append((cid, arrival_sec, mi, pes, file_size, output_size))
        cid += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["cloudlet_id", "arrival_time", "length",
                         "pes_required", "file_size", "output_size"])
        for row in cloudlets:
            writer.writerow(row)

    total = len(cloudlets)
    mean_mi = sum(c[2] for c in cloudlets) / total
    mean_pes = sum(c[3] for c in cloudlets) / total
    mean_runtime = mean_mi / 40000
    peak_concurrent = HIGH_RATE * mean_pes * mean_runtime
    valley_concurrent = LOW_RATE * mean_pes * mean_runtime
    avg_rate = (HIGH_RATE + LOW_RATE) / 2
    avg_concurrent = avg_rate * mean_pes * mean_runtime
    routing_cap = 20 * 3000

    print(f"Generated {total} cloudlets -> {OUTPUT_FILE}")
    print(f"Arrival span: 0-{ARRIVAL_SPAN}s")
    print(f"Mean MI: {mean_mi:.0f}, Mean PEs: {mean_pes:.1f}")
    print(f"Mean runtime per cloudlet: {mean_runtime:.1f}s")
    print(f"Est. concurrent PEs -- peak: {peak_concurrent:.0f}, "
          f"valley: {valley_concurrent:.0f}, avg: {avg_concurrent:.0f}")
    print(f"System capacity: 1920 PEs")
    print(f"Est. utilisation -- peak: {peak_concurrent/1920*100:.0f}%, "
          f"valley: {valley_concurrent/1920*100:.0f}%, "
          f"avg: {avg_concurrent/1920*100:.0f}%")
    print(f"Routing capacity: 20/step * 3000 steps = {routing_cap} slots "
          f"(headroom: {routing_cap - total})")

if __name__ == "__main__":
    main()
