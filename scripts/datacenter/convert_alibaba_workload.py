"""
Convert the Alibaba cluster-trace-v2018 ``batch_task`` table into a CloudSim Plus
CSV workload trace (``cloudlet_id,arrival_time,length,pes_required,file_size,output_size``)
that drops straight into the gateway's CSV ``workload_mode`` (see
``WorkloadFileReader`` — it reads the first 4 columns).

Why this trace
--------------
``batch_task`` is the task-level summary of Alibaba's production batch workload
(~14.3M tasks over 8.83 days). Each row gives the three fields a cloudlet needs:
arrival (``start_time``), runtime (``end_time - start_time``) and CPU demand
(``plan_cpu``, where 100 == 1 core). This replaces the synthetic Poisson traces
with a real, bursty cloud workload while keeping the existing multi-DC routing
(one trace -> the RL agent routes cloudlets across DCs) unchanged.

batch_task schema (no header in the raw CSV), columns in order:
    task_name, instance_num, job_name, task_type, status,
    start_time, end_time, plan_cpu (100=1core), plan_mem

Time handling (the crucial part)
--------------------------------
Arrival time AND runtime are compressed by the SAME factor so the per-(sim-)second
load intensity is preserved:

    compress  = raw_window / sim_duration            (>= 1)
    arrival   = (start_time - raw_start) / compress
    length_MI = (runtime / compress) * ref_mips * pes

- ``compress == 1`` (raw_window == sim_duration): an honest 1:1 real slice.
- ``raw_window == 86400`` -> a full real day squeezed into the episode, which
  gives a diurnal arrival shape inside ``sim_duration`` seconds.

Subsampling to ``--target-count`` keeps the arrival *shape* but scales the
*intensity* down; the report prints the resulting cluster utilization so you can
calibrate count / window / cluster size to a target completion rate.

Example
-------
    python3 scripts/datacenter/convert_alibaba_workload.py \
        --input data/alibaba_v2018/batch_task.csv \
        --out   cloudsimplus-gateway/src/main/resources/traces/alibaba_5dc_6000s_50k.csv \
        --raw-start 86400 --raw-window 6000 --sim-duration 6000 \
        --target-count 50000 --seed 42
"""
import argparse
import csv
import math
import random
import sys

# batch_task column indices
C_STATUS = 4
C_START = 5
C_END = 6
C_PLAN_CPU = 7


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Convert Alibaba batch_task -> CloudSim CSV workload.")
    p.add_argument("--input", required=True, help="Path to batch_task.csv (raw, no header).")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--raw-start", type=int, default=86400,
                   help="First raw start_time (s) to include; default 86400 skips the warmup day.")
    p.add_argument("--raw-window", type=int, default=6000,
                   help="Width (s) of the raw-time window to draw from. Use 86400 for a diurnal day.")
    p.add_argument("--sim-duration", type=int, default=6000,
                   help="Episode seconds the window is compressed into.")
    p.add_argument("--target-count", type=int, default=50000,
                   help="Subsample to this many cloudlets (0 = keep all tasks in window).")
    p.add_argument("--ref-mips", type=int, default=50000,
                   help="MIPS per core used to turn runtime seconds into MI (match workload_reader_mips).")
    p.add_argument("--max-pes", type=int, default=0,
                   help="Cap pes_required (0 = no cap; let split_large_cloudlets handle big tasks).")
    p.add_argument("--min-length", type=int, default=1, help="Floor for cloudlet length (MI).")
    p.add_argument("--max-length", type=int, default=0, help="Cap for cloudlet length (MI); 0 = no cap.")
    p.add_argument("--seed", type=int, default=42)
    # Calibration-report-only knobs (do not affect the trace):
    p.add_argument("--cluster-pes", type=int, default=520, help="Total cluster PEs (for utilization report).")
    p.add_argument("--pe-mips", type=int, default=40000, help="MIPS per VM PE (for utilization report).")
    return p.parse_args(argv)


def collect_tasks(input_path, raw_start, raw_window):
    """Stream batch_task.csv, return list of (start, runtime, plan_cpu) within the window.

    Filters: status == 'Terminated', start > 0, end > start, plan_cpu > 0.
    """
    raw_end = raw_start + raw_window
    tasks = []
    skipped = {"short": 0, "fields": 0, "status": 0, "cpu": 0, "window": 0}
    with open(input_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 9:
                skipped["fields"] += 1
                continue
            if row[C_STATUS] != "Terminated":
                skipped["status"] += 1
                continue
            try:
                start = int(row[C_START]); end = int(row[C_END]); cpu = float(row[C_PLAN_CPU])
            except ValueError:
                skipped["fields"] += 1
                continue
            if start <= 0 or end <= start:
                skipped["short"] += 1
                continue
            if cpu <= 0:
                skipped["cpu"] += 1
                continue
            if start < raw_start or start >= raw_end:
                skipped["window"] += 1
                continue
            tasks.append((start, end - start, cpu))
    return tasks, skipped


def build_rows(tasks, args):
    """Map (start, runtime, plan_cpu) tasks to CSV rows. Returns (rows, stats)."""
    compress = args.raw_window / args.sim_duration
    rows = []
    for start, runtime, cpu in tasks:
        arrival = int((start - args.raw_start) / compress)
        pes = max(1, math.ceil(cpu / 100.0))
        if args.max_pes > 0:
            pes = min(pes, args.max_pes)
        length = int((runtime / compress) * args.ref_mips * pes)
        length = max(args.min_length, length)
        if args.max_length > 0:
            length = min(length, args.max_length)
        rows.append((arrival, length, pes))
    rows.sort(key=lambda r: r[0])  # by arrival
    out_rows = []
    for cid, (arrival, length, pes) in enumerate(rows):
        fs = max(100, length // 1000)
        os_ = max(50, fs // 2)
        out_rows.append([cid, arrival, length, pes, fs, os_])
    return out_rows


def write_csv(out_path, out_rows):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cloudlet_id", "arrival_time", "length", "pes_required", "file_size", "output_size"])
        w.writerows(out_rows)


def _pct(sorted_a, p):
    if not sorted_a:
        return None
    return sorted_a[min(len(sorted_a) - 1, int(p / 100.0 * len(sorted_a)))]


def report(out_rows, args, n_window, skipped):
    arrivals = [r[1] for r in out_rows]
    lengths = sorted(r[2] for r in out_rows)
    pes_list = [r[3] for r in out_rows]
    total_pe_mi = sum(r[2] * r[3] for r in out_rows)
    cap = args.cluster_pes * args.pe_mips * args.sim_duration
    print(f"Written: {args.out}")
    print(f"Tasks in raw window [{args.raw_start}, {args.raw_start + args.raw_window}): {n_window}")
    print(f"Skipped (status/short/cpu/fields/out-of-window): "
          f"{skipped['status']}/{skipped['short']}/{skipped['cpu']}/{skipped['fields']}/{skipped['window']}")
    print(f"Cloudlets written: {len(out_rows)}  (compress factor {args.raw_window / args.sim_duration:.2f}x)")
    if not out_rows:
        return
    print(f"Arrival span: {min(arrivals)}..{max(arrivals)} s  (sim_duration={args.sim_duration})")
    print(f"Length MI  p10/p50/p90/p99/max: {[_pct(lengths, p) for p in (10, 50, 90, 99, 100)]}")
    pes_dist = {p: pes_list.count(p) for p in sorted(set(pes_list))}
    print(f"PEs distribution: {pes_dist}")
    print(f"Total PE-MI: {total_pe_mi / 1e9:.2f} B")
    print(f"Cluster capacity ({args.cluster_pes} PEs x {args.pe_mips} MIPS x {args.sim_duration}s): {cap / 1e9:.1f} B PE-MI")
    print(f"Utilization: {total_pe_mi / cap * 100:.1f}%")
    nb = args.sim_duration // 600 + 1
    buckets = [0] * nb
    for a in arrivals:
        buckets[min(nb - 1, a // 600)] += 1
    print("arrivals per 10-min bucket (diurnal/burst check):")
    for i, c in enumerate(buckets):
        print(f"  bucket {i:2d} ({i * 600:5d}-{(i + 1) * 600:5d}s): {c:6d} {'#' * int(c / max(1, max(buckets)) * 50)}")


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    tasks, skipped = collect_tasks(args.input, args.raw_start, args.raw_window)
    n_window = len(tasks)
    if args.target_count > 0 and len(tasks) > args.target_count:
        tasks = random.sample(tasks, args.target_count)
    out_rows = build_rows(tasks, args)
    write_csv(args.out, out_rows)
    report(out_rows, args, n_window, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
