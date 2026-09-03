"""Simulator-free green-scheduling model: where does a forecast earn carbon?

Physics kept: real wind rows (one row = one step), several DCs each with its own green
power G_d(t), jobs of fixed power p and runtime R that must start within slack S of
arrival (forced start at a+S), and DC-level competition: brown power at a DC and row is
max(0, load_d(t) - G_d(t)). Physics dropped on purpose: host floor, VM placement, MIPS.

Policies on the same truth:
  run_now  : start at arrival on the DC with the largest present headroom.
  myopic   : each row, start waiting jobs (FIFO) on any DC whose present headroom covers
             p; forced start at the latest start. Reads only the present.
  planner  : on arrival, commit the (DC, start) in [a, a+S] with the least predicted
             brown energy given predicted headroom (view minus committed load).
             view = truth | shrink (regression to mean, c) | anti | shuffle (lead-0 exact).

Reported per policy: brown Wh, green Wh, and the forecast-only lever
  (brown_myopic - brown_planner_truth) / total job energy.
"""
from __future__ import annotations

import numpy as np

# Power of one 32-PE job as the simulator actually runs it on a zero-floor RS500A_DYN
# host: the 32-PE VM has vm_pe_mips 40000 against 50000 MIPS host cores, so the host
# utilisation is 32*40000 / (64*50000) = 0.4 and the draw is 1 W floor + 0.4 * 161.6 W.
# Pinned by ZeroFloorSentinelTest (real simulation, 1e-9 W). The idealised 81.3 W
# (half the 162.6 W span) is what a job would draw at equal VM and host MIPS; the
# 192-config structural sweep of 2026-09-03 used that value, the per-cell pilot
# predictions and the simulator comparison used 65.64 W.
P_DYN_W = 1.0 + 0.4 * (214.0 - 51.4 - 1.0)  # = 65.64 W


def brown_of(load: np.ndarray, G: np.ndarray) -> float:
    return float(np.maximum(load - G, 0.0).sum())


class Sim:
    def __init__(self, G: np.ndarray, arrivals, runtime: int, slack: int, p: float = P_DYN_W):
        self.G, self.arr, self.R, self.S, self.p = G, list(arrivals), runtime, slack, p
        self.n_dc, self.T = G.shape

    def _energy(self, starts, dcs):
        load = np.zeros_like(self.G)
        for s, d in zip(starts, dcs):
            load[d, s:s + self.R] += self.p
        brown = brown_of(load, self.G)
        total = self.p * self.R * len(starts)
        return {"brown_w_rows": brown, "green_w_rows": total - brown, "total_w_rows": total,
                "brown_share": brown / total, "mean_wait": float(np.mean(np.array(starts) - np.array(self.arr)))}

    def run_now(self):
        load = np.zeros_like(self.G)
        starts, dcs = [], []
        for a in self.arr:
            d = int(np.argmax(self.G[:, a] - load[:, a]))
            load[d, a:a + self.R] += self.p
            starts.append(a); dcs.append(d)
        return self._energy(starts, dcs)

    def myopic(self):
        load = np.zeros_like(self.G)
        starts, dcs = [None] * len(self.arr), [None] * len(self.arr)
        waiting = []
        order = sorted(range(len(self.arr)), key=lambda j: self.arr[j])
        nxt = 0
        for t in range(self.T):
            while nxt < len(order) and self.arr[order[nxt]] <= t:
                waiting.append(order[nxt]); nxt += 1
            for j in list(waiting):
                head = self.G[:, t] - load[:, t]
                d = int(np.argmax(head))
                if head[d] >= self.p or t >= self.arr[j] + self.S:
                    load[d, t:t + self.R] += self.p
                    starts[j], dcs[j] = t, d
                    waiting.remove(j)
            if nxt >= len(order) and not waiting:
                break
        return self._energy(starts, dcs)

    def planner(self, view: np.ndarray):
        """view: (n_dc, T) predicted green; lead-0 column is assumed exact by the caller."""
        committed = np.zeros_like(self.G)
        starts, dcs = [], []
        for a in self.arr:
            best = None
            for s in range(a, min(a + self.S, self.T - self.R) + 1):
                for d in range(self.n_dc):
                    seg = committed[d, s:s + self.R] + self.p - view[d, s:s + self.R]
                    b = float(np.maximum(seg, 0.0).sum())
                    if best is None or b < best[0] - 1e-9:
                        best = (b, s, d)
            _, s, d = best
            committed[d, s:s + self.R] += self.p
            starts.append(s); dcs.append(d)
        return self._energy(starts, dcs)


def view_shrink(G: np.ndarray, c: float, now_rows=None) -> np.ndarray:
    """Regression to the mean: mu + c (G - mu) per DC. Lead-0 exactness is handled by the
    planner reading the view at the decision row itself, which we keep exact by mixing."""
    mu = G.mean(axis=1, keepdims=True)
    return mu + c * (G - mu)


def view_anti(G: np.ndarray) -> np.ndarray:
    mu = G.mean(axis=1, keepdims=True)
    return np.clip(2 * mu - G, 0.0, None)


def view_shuffle(G: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = G.copy()
    for d in range(G.shape[0]):
        out[d] = G[d, rng.permutation(G.shape[1])]
    return out


def lead0_exact(sim: Sim, view: np.ndarray) -> np.ndarray:
    """Planner decisions happen at arrival rows; make those columns exact (lead-0 truth)."""
    v = view.copy()
    for a in sim.arr:
        v[:, a] = sim.G[:, a]
    return v


def evaluate(G, arrivals, runtime, slack, p=P_DYN_W, shrink_c=0.5):
    sim = Sim(G, arrivals, runtime, slack, p)
    res = {"run_now": sim.run_now(), "myopic": sim.myopic(), "truth": sim.planner(G)}
    for name, v in (("shrink", view_shrink(G, shrink_c)), ("anti", view_anti(G)), ("shuffle", view_shuffle(G))):
        res[name] = sim.planner(lead0_exact(sim, v))
    tot = res["truth"]["total_w_rows"]
    blind = min(res["run_now"]["brown_w_rows"], res["myopic"]["brown_w_rows"])
    res["lever_forecast_only_pp"] = 100 * (res["myopic"]["brown_w_rows"] - res["truth"]["brown_w_rows"]) / tot
    res["lever_vs_best_blind_pp"] = 100 * (blind - res["truth"]["brown_w_rows"]) / tot
    gain = blind - res["truth"]["brown_w_rows"]
    for name in ("shrink", "anti", "shuffle"):
        res[f"retention_{name}"] = (blind - res[name]["brown_w_rows"]) / gain if abs(gain) > 1e-9 else float("nan")
    return res
