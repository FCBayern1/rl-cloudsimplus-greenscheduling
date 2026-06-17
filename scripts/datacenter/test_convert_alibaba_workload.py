"""Unit tests for convert_alibaba_workload.py (Alibaba batch_task -> CloudSim CSV)."""
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import convert_alibaba_workload as conv  # noqa: E402


# task_name,instance_num,job_name,task_type,status,start_time,end_time,plan_cpu,plan_mem
SAMPLE_ROWS = [
    # in-window, valid: start=1000, runtime=100, plan_cpu=100 -> 1 core
    "M1,1,j_1,1,Terminated,1000,1100,100,0.2",
    # in-window, valid: start=1500, runtime=50, plan_cpu=400 -> 4 cores
    "R2_1,1,j_2,1,Terminated,1500,1550,400,0.5",
    # filtered: not Terminated
    "M2,1,j_3,1,Running,1200,1300,100,0.2",
    # filtered: runtime 0 (end <= start)
    "M3,1,j_4,1,Terminated,1200,1200,100,0.2",
    # filtered: plan_cpu <= 0
    "M4,1,j_5,1,Terminated,1200,1300,0,0.2",
    # filtered: out of window (start >= raw_start+raw_window=2000)
    "M5,1,j_6,1,Terminated,5000,5100,100,0.2",
    # filtered: out of window (start < raw_start=1000)
    "M6,1,j_7,1,Terminated,500,600,100,0.2",
    # filtered: too few fields
    "broken,row",
]


def _write_sample(path):
    with open(path, "w", newline="") as f:
        f.write("\n".join(SAMPLE_ROWS) + "\n")


class TestConvert(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.inp = os.path.join(self.tmp, "batch_task.csv")
        self.out = os.path.join(self.tmp, "out.csv")
        _write_sample(self.inp)

    def _run(self, **over):
        argv = ["--input", self.inp, "--out", self.out,
                "--raw-start", "1000", "--raw-window", "1000", "--sim-duration", "1000",
                "--target-count", "0", "--ref-mips", "50000", "--seed", "1"]
        for k, v in over.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        conv.main(argv)
        with open(self.out) as f:
            r = csv.reader(f)
            header = next(r)
            rows = [row for row in r]
        return header, rows

    def test_filters_keep_only_two_valid_rows(self):
        _, rows = self._run()
        self.assertEqual(len(rows), 2, "only the two valid in-window Terminated rows survive")

    def test_header_and_six_columns(self):
        header, rows = self._run()
        self.assertEqual(header, ["cloudlet_id", "arrival_time", "length", "pes_required", "file_size", "output_size"])
        self.assertTrue(all(len(row) == 6 for row in rows))

    def test_field_mapping_compress_one(self):
        # compress = raw_window/sim_duration = 1.0 here.
        # Row M1: start=1000 -> arrival 0; runtime 100; cpu 100 -> 1 pe; length = 100*50000*1 = 5_000_000
        # Row R2_1: start=1500 -> arrival 500; runtime 50; cpu 400 -> 4 pe; length = 50*50000*4 = 10_000_000
        _, rows = self._run()
        by_arrival = {int(r[1]): r for r in rows}
        self.assertIn(0, by_arrival)
        self.assertIn(500, by_arrival)
        r0 = by_arrival[0]
        self.assertEqual(int(r0[2]), 5_000_000)  # length
        self.assertEqual(int(r0[3]), 1)          # pes
        r1 = by_arrival[500]
        self.assertEqual(int(r1[2]), 10_000_000)
        self.assertEqual(int(r1[3]), 4)

    def test_sequential_ids_sorted_by_arrival(self):
        _, rows = self._run()
        arrivals = [int(r[1]) for r in rows]
        ids = [int(r[0]) for r in rows]
        self.assertEqual(ids, list(range(len(rows))), "ids reassigned sequentially")
        self.assertEqual(arrivals, sorted(arrivals), "rows sorted by arrival")

    def test_compress_scales_arrival_and_length_together(self):
        # raw_window=1000 compressed into sim_duration=500 -> compress=2.
        # M1: arrival = (1000-1000)/2 = 0; length = (100/2)*50000*1 = 2_500_000
        # R2_1: arrival = (1500-1000)/2 = 250; length = (50/2)*50000*4 = 5_000_000
        _, rows = self._run(sim_duration=500)
        by_arrival = {int(r[1]): r for r in rows}
        self.assertIn(250, by_arrival)
        self.assertEqual(int(by_arrival[0][2]), 2_500_000)
        self.assertEqual(int(by_arrival[250][2]), 5_000_000)

    def test_max_pes_cap(self):
        _, rows = self._run(max_pes=2)
        pes = sorted(int(r[3]) for r in rows)
        self.assertEqual(pes, [1, 2], "4-core task capped to 2 pes")

    def test_target_count_subsamples(self):
        _, rows = self._run(target_count=1)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
