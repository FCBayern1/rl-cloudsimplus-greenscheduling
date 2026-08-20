#!/usr/bin/env python3
"""gwo1 (第十考场) trace 生成器 —— runtime ×1.30,deadline 一致重算。

与 SQT2 的差别只在 runtime 缩放:主机制是"释放时刻落在绿窗中后段 → 剩余绿电
不够跑完 → 溢出到棕电",所以 runtime 相对 ON 窗长的比例就是机制本身。
5080 标定的暴露带宽 [0.20, 0.50],×1.30 落在中部(23.2%)且保留 cashability 余量:
    runtime_max = 1300 < ON_min = 1500   ✅

deadline 必须按新 runtime 一致重算(deadline = arrival + runtime + slack),
否则 slack 语义漂移、preflight 的 B 类检查连带失真。
"""
import argparse, csv, json, random
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "cloudsimplus-gateway/src/main/resources/traces"
CALIB = Path(__file__).resolve().parent / "calib"
MIPS = 40000.0


def runtime_of(row, scale=1.0):
    return max(1.0, scale * float(row["length"]) / (max(1, int(row["pes_required"])) * MIPS))


def generate(src, scale, prefix, seed):
    rows = list(csv.DictReader(open(src)))
    old_rt = [runtime_of(r, 1.0) for r in rows]
    new_rt = [runtime_of(r, scale) for r in rows]
    out = []
    for i, r in enumerate(rows):
        r = dict(r)
        # slack 是原 trace 里蕴含的量,逐作业保留
        slack = float(r["deadline"]) - float(r["arrival_time"]) - old_rt[i]
        r["length"] = str(int(round(float(r["length"]) * scale)))
        r["deadline"] = str(int(round(float(r["arrival_time"]) + new_rt[i] + slack)))
        out.append(r)
    name = f"{prefix}_n{len(rows)}_x{int(scale*100)}.csv"
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(out)
    art = {"scenario": prefix, "source": Path(src).name, "runtime_scale": scale,
           "seed": seed, "n": len(rows),
           "runtime_max": max(new_rt), "runtime_median": sorted(new_rt)[len(new_rt)//2],
           "slack_median": sorted(float(o["deadline"]) - float(o["arrival_time"]) - new_rt[i]
                                  for i, o in enumerate(out))[len(out)//2]}
    CALIB.mkdir(exist_ok=True)
    (CALIB / f"{prefix}_trace_x{int(scale*100)}.json").write_text(json.dumps(art, indent=1))
    return name, out, new_rt, art


ANCHORS = (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)   # 预注册锚点(铺满序列)


def exposure_anchored(trace_rows, runtimes, on_flags, anchors=ANCHORS):
    """在全部注册锚点上量暴露门,返回逐锚值。

    只在 offset=0 处量会严重偏斜:序列开头恰是一段 ON,早到达的作业几乎全落在绿窗里。
    实际评测用 (1009*k) mod range 的偏移铺满全域,暴露门必须按同一口径标定。
    """
    T = len(on_flags)
    out = []
    for k in anchors:
        off = (1009 * k) % T
        rolled = on_flags[off:] + on_flags[:off]
        out.append(exposure(trace_rows, runtimes, rolled))
    return out


def exposure(trace_rows, runtimes, on_flags):
    """暴露门:到达时处于绿窗、且【立刻释放就会溢出到棕电】的 MI 占比。

    这是 SQT2 那个 worthy/not-worthy 暴露门在新机制下的对应物 —— 它度量的是
    "此刻这个决策有没有意义"。逐作业只看到达时刻的状态,不扫描未来。
    """
    T = len(on_flags)
    mi = [float(r["length"]) for r in trace_rows]
    tot = sum(mi)
    in_green = spill = 0.0
    for i, r in enumerate(trace_rows):
        a = int(float(r["arrival_time"]))
        if a >= T or not on_flags[a]:
            continue
        in_green += mi[i]
        end = min(a + int(runtimes[i]), T)
        if not all(on_flags[a:end]):        # 立刻跑会溢出到棕电
            spill += mi[i]
    return in_green / tot, spill / tot, (spill / in_green if in_green else 0.0)


def main():
    ap = argparse.ArgumentParser()
    # 源 trace 必须是 t60(与 config 的 SQT2 块一致);t50 会静默产出
    # 一个同名但底座不同的 trace,是 2026-08-19 抓到的地雷。
    ap.add_argument("--src", default=str(OUT / "sqt2_n1200_t60.csv"))
    ap.add_argument("--scales", default="1.00,1.15,1.30,1.45")
    ap.add_argument("--prefix", default="gwo1")
    ap.add_argument("--schedule", default="calib/gwo1_schedule.json")
    ap.add_argument("--seed", type=int, default=20260820)
    a = ap.parse_args()

    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gen_sqt2 as gs
    art = json.load(open(Path(__file__).resolve().parent / a.schedule))
    on, _ = gs.build_schedule(art["rows"], art["seed"])

    print(f"源 trace: {Path(a.src).name}   绿电序列: {a.schedule} (green={art['green_ratio']})")
    print(f"{'scale':>7}{'runtime_max':>13}{'cashability':>13}{'绿窗到达MI':>12}{'暴露MI':>10}{'占绿窗':>9}")
    ON_MIN = gs.ON_LO
    for sc in [float(x) for x in a.scales.split(",")]:
        name, rows, rts, meta = generate(a.src, sc, a.prefix, a.seed)
        per = exposure_anchored(rows, rts, on)
        import statistics as st
        g = st.median(p[0] for p in per); s = st.median(p[1] for p in per)
        share = st.median(p[2] for p in per)
        lo_, hi_ = min(p[1] for p in per), max(p[1] for p in per)
        cash = "OK" if meta["runtime_max"] <= ON_MIN else f"违反(>{ON_MIN})"
        print(f"{sc:>7.2f}{meta['runtime_max']:>13.0f}{cash:>13}"
              f"{100*g:>11.1f}%{100*s:>9.1f}%{100*share:>8.1f}%"
              f"   [锚点范围 {100*lo_:.1f}-{100*hi_:.1f}%]")


if __name__ == "__main__":
    main()
