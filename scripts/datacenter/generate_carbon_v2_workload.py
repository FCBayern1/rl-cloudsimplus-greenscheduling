"""
Generate diurnal carbon-v2 workload trace.

Design intent:
- 6000s of arrivals with 2 diurnal cycles (period 3000s)
- Peak/trough arrival rate ratio ~3x (simulated afternoon vs night)
- Cluster capacity 520 PEs × 40 GMIPS × 6000s ≈ 125 B PE-MI
- Total workload ≈ 67 B PE-MI → ~54% utilization (baseline completion ~85%)
- Length distribution matches existing dc10_20000 (200k-2M MI)
- PES skewed small (most tasks 1-4 PEs)
"""
import argparse
import csv
import math
import random

parser = argparse.ArgumentParser(description="Generate diurnal carbon-v2 workload trace.")
parser.add_argument("--out", type=str,
    default="/home/joshua/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/src/main/resources/traces/dc5_carbon_v2_diurnal.csv",
    help="Output CSV path (default: dc5_carbon_v2_diurnal.csv)")
parser.add_argument("--base-rate", type=float, default=3.0,
    help="Cloudlets/sec at diurnal trough; expected total ≈ base_rate × (1+peak_ratio)/2 × sim_duration")
parser.add_argument("--peak-ratio", type=float, default=3.0,
    help="Peak/trough arrival rate ratio")
parser.add_argument("--sim-duration", type=int, default=6000,
    help="Seconds of arrivals")
parser.add_argument("--diurnal-period", type=int, default=3000,
    help="One 'day' length in sim seconds")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
args = parser.parse_args()

random.seed(args.seed)

OUT = args.out
SIM_DURATION = args.sim_duration
DIURNAL_PERIOD = args.diurnal_period
BASE_RATE = args.base_rate
PEAK_RATIO = args.peak_ratio
N_TARGET = int(BASE_RATE * (1 + PEAK_RATIO) / 2 * SIM_DURATION)  # informational

# Work out arrival rate lambda(t): base * (1 + (peak-1)*sin^2(pi*t/period))
def rate(t):
    phase = math.sin(math.pi * t / DIURNAL_PERIOD) ** 2
    return BASE_RATE * (1 + (PEAK_RATIO - 1) * phase)

# Generate arrivals via thinning (non-homogeneous Poisson)
max_rate = BASE_RATE * PEAK_RATIO
arrivals = []
t = 0.0
while t < SIM_DURATION:
    u = random.random()
    t += -math.log(u) / max_rate
    if t >= SIM_DURATION:
        break
    if random.random() < rate(t) / max_rate:
        arrivals.append(int(t))

# Length distribution: lognormal-ish, clamp to [200k, 2M]
def sample_length():
    mu, sigma = math.log(650000), 0.6
    v = random.lognormvariate(mu, sigma)
    return int(max(200000, min(2000000, v)))

# PES distribution: weighted toward small
PES_CHOICES = [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8]

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["cloudlet_id", "arrival_time", "length", "pes_required", "file_size", "output_size"])
    for cid, at in enumerate(arrivals):
        length = sample_length()
        pes = random.choice(PES_CHOICES)
        fs = max(100, length // 1000)
        os_ = max(50, fs // 2)
        w.writerow([cid, at, length, pes, fs, os_])

# Summary
total_mi = 0
total_pe_mi = 0
per_hour = [0] * (SIM_DURATION // 600 + 1)
with open(OUT) as f:
    r = csv.DictReader(f)
    for row in r:
        l = int(row['length'])
        p = int(row['pes_required'])
        total_mi += l
        total_pe_mi += l * p
        per_hour[int(row['arrival_time']) // 600] += 1

print(f"Written: {OUT}")
print(f"Cloudlets: {len(arrivals)}")
print(f"Total MI: {total_mi/1e9:.2f} B")
print(f"Total PE-MI: {total_pe_mi/1e9:.2f} B")
print(f"Cluster capacity (520 PEs × 40k MIPS × 6000s): {520*40000*6000/1e9:.1f} B PE-MI")
print(f"Utilization: {total_pe_mi / (520*40000*6000) * 100:.1f}%")
print()
print("arrivals per 10-min bucket (diurnal check):")
for i, c in enumerate(per_hour):
    bar = "#" * int(c / 50)
    print(f"  bucket {i} ({i*600}-{(i+1)*600}s): {c:5d} {bar}")
