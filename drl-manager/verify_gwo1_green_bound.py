#!/usr/bin/env python3
"""独立复算绿窗侧的可决策量与碳硬上界（5080 报 10.6% / 6.09%）。

出数的人不做唯一核对人 —— 这份用 gate_flags 里那条 clairvoyant 判据的
定义,从 schedule + trace 直接重算,不经过仿真器,也不读 5080 的产物。

  可决策 MI   : rem_green < runtime  且  rem_green + next_trough <= budget
  落棕电的 MI : 上一集合中 (runtime - rem_green)/runtime 那一段
                (前半段本来就在绿窗里跑,不该记进差值)

碳口径硬上界(④ 的验收线)
------------------------
碳 ~ 棕电 MI x 棕因子(绿因子只有棕的 1/55,忽略它让上界略偏大 = 保守),所以

    上界 = 宽门可搬走的棕电 MI / nowait 下的棕电 MI = 12.95%

分母是 nowait 的【棕电】MI,不是全部 MI。我一度按全部 MI 算成 4.14%,
那是把可搬走的量摊到了 3 倍大的分母上 —— 会过紧,可能把真实机制效应误判
成 bug。方向要记住:碳口径的线比 MI 口径【宽】。
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
    tot_mi = nowait_brown = 0.0        # 碳口径分母
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
            base = int(WARMUP_ROWS + off + arrival)
            tot_mi += mi
            # nowait 立即释放:[arrival, arrival+runtime) 落在槽内的那部分是棕电
            brown_s = sum(1 for d in range(int(runtime)) if ti.query(base + d)[0])
            nowait_brown += mi * brown_s / max(1.0, int(runtime))
            in_trough, _, _, rem_green, _ = ti.query(base)
            if in_trough:
                continue                              # 绿窗侧决策集
            n_green += 1
            tot += mi
            budget = effective_budget(float(r["deadline"]) - arrival, runtime,
                                      MARGIN_S, HORIZON_S - arrival)
            if budget <= 0:
                continue
            nxt = ti.next_trough_dur(base)
            if rem_green < runtime and rem_green + nxt <= budget:
                n_act += 1
                act += mi
                brown += mi * (runtime - rem_green) / runtime
    return {"green_decision_mi": tot, "n_green": n_green, "n_act": n_act,
            "act_share": act / tot if tot else 0.0,
            "brown_share": brown / tot if tot else 0.0,
            "total_mi": tot_mi, "nowait_brown_mi": nowait_brown,
            "movable_brown_mi": brown,
            # ④ 的验收线:分母是 nowait 的棕电 MI,不是全部 MI
            "carbon_bound": brown / nowait_brown if nowait_brown else 0.0}


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
        print(f"  其中落棕电         {100*d['brown_share']:.2f}%  (MI 口径,非验收线)")
        print(f"  nowait 棕电 MI     {d['nowait_brown_mi']:.4e}"
              f"  ({100*d['nowait_brown_mi']/d['total_mi']:.1f}% of {d['total_mi']:.4e})")
        print(f"  ★ 碳口径硬上界     {100*d['carbon_bound']:.2f}%   <- ④ 的验收线")
