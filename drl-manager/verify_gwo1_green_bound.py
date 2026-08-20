#!/usr/bin/env python3
"""独立复算绿窗侧的可决策量与碳硬上界（5080 报 10.6% / 6.09%）。

出数的人不做唯一核对人 —— 这份用 gate_flags 里那条 clairvoyant 判据的
定义,从 schedule + trace 直接重算,不经过仿真器,也不读 5080 的产物。

  可决策 MI   : rem_green < runtime  且  rem_green + next_trough <= budget
  落棕电的 MI : 上一集合中 (runtime - rem_green)/runtime 那一段
                (前半段本来就在绿窗里跑,不该记进差值)
"""
import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_sqt2 import build_schedule                      # noqa: E402
from oracle_slack_planner import WARMUP_ROWS             # noqa: E402
from sqt2_prescreen import HORIZON_S, MARGIN_S, MIPS, TroughIndex  # noqa: E402
from teacher_reward_audit import effective_budget, episode_offset  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parent.parent
ANCHORS10 = (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)


def analyse(schedule_art, trace_csv, anchors, offset_range):
    art = json.loads((pathlib.Path(__file__).resolve().parent
                      / schedule_art).read_text())
    build_schedule(art["rows"], art["seed"])          # 口径自证:同一构造函数
    ti = TroughIndex(art["troughs"], horizon=art["rows"])
    rows = list(csv.DictReader(open(trace_csv)))

    tot = act = brown = 0.0
    n_green = n_act = 0
    for k in anchors:
        off = episode_offset(k, offset_range)
        for r in rows:
            arrival = float(r["arrival_time"])
            if arrival >= HORIZON_S:
                continue
            mi = float(r["length"])
            pes = max(1, int(r["pes_required"]))
            runtime = max(1.0, mi / (pes * MIPS))
            in_trough, _, _, rem_green, _ = ti.query(int(WARMUP_ROWS + off + arrival))
            if in_trough:
                continue                              # 绿窗侧决策集
            n_green += 1
            tot += mi
            budget = effective_budget(float(r["deadline"]) - arrival, runtime,
                                      MARGIN_S, HORIZON_S - arrival)
            if budget <= 0:
                continue
            nxt = ti.next_trough_dur(int(WARMUP_ROWS + off + arrival))
            if rem_green < runtime and rem_green + nxt <= budget:
                n_act += 1
                act += mi
                brown += mi * (runtime - rem_green) / runtime
    return {"green_decision_mi": tot, "n_green": n_green, "n_act": n_act,
            "act_share": act / tot if tot else 0.0,
            "brown_share": brown / tot if tot else 0.0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default="calib/gwo1_schedule.json")
    ap.add_argument("--trace", default=str(
        _REPO / "cloudsimplus-gateway/src/main/resources/traces/gwo1_n1200_x130.csv"))
    ap.add_argument("--offset-range", type=int, default=180000)
    a = ap.parse_args()
    for label, anch in (("10 锚(预注册全集)", ANCHORS10),
                        ("3 锚(0/79/158,本轮跑的)", (0, 79, 158))):
        d = analyse(a.schedule, a.trace, anch, a.offset_range)
        print(f"{label}:")
        print(f"  绿窗决策集 MI      {d['green_decision_mi']:.4e}"
              f"  ({d['n_green']} 次到达)")
        print(f"  clairvoyant 动手   {100*d['act_share']:.2f}%"
              f"  ({d['n_act']} 个作业)")
        print(f"  其中落棕电         {100*d['brown_share']:.2f}%   <- 碳差硬上界")
