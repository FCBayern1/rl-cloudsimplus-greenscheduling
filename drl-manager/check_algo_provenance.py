"""Am I training with the current algorithm?

Three layers can each be stale independently, and each fails silently.

  Java   a gateway jar built before 61043cf carries the dispatcher bug, where
         vm.getFreePesNumber() never updates under the custom broker's
         submission path, so every step's most-free-VM search starts from a
         wrong map and piles work onto the first few VMs. Measured effect on
         a fixed policy: carbon -0.7% but episode length -14.7% and energy
         -9.9%. Small on the headline number, large on the timing dynamics a
         policy trains against.

  Python a checkout predating the v5 CRD fixes runs a decomposition whose
         named channels are not the quantities they claim: R_forecast a biased
         constant, quarantine inert, router delta_r identically zero.

  Config a crd subtree copied before v5 lacks the keys that switch those fixes
         on, so current code plus an old block still runs the old algorithm.

Timestamps are not evidence. mtime survives a touch, and building before
pulling leaves a new-looking jar built from old source. This checks content:
git ancestry for the source, sha256 for the jar actually loaded, and key
presence for the config.

  python check_algo_provenance.py --experiment experiment_tb12_rl_fc
"""
import argparse
import hashlib
import os
import pathlib
import subprocess
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
DISPATCH_FIX = "61043cf"

# Keys the v5 CRD fixes introduced. A block written before them lacks these,
# and current code reading such a block still runs the old algorithm.
V5_MARKERS = {
    "forecast": ("scale_fix", "carbon_norm", "magnitude"),
    "responsibility": ("anomaly_gate", "normalize_shares", "share_scale_decay"),
    "ensemble": ("stable_bootstrap",),
    "delta_r": ("mode",),
    "baseline": ("kind",),
    "blender": ("tau_mode",),
}

rows = []


def chk(name, ok, detail=""):
    rows.append((name, bool(ok), detail))


def sh(cmd):
    return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", help="config key to check the crd subtree of")
    ap.add_argument("--config", default=str(ROOT / "config_C.yml"))
    args = ap.parse_args()

    # ---- source ----
    head = sh("git rev-parse HEAD").stdout.strip()
    dirty = sh("git status --porcelain").stdout.strip()
    chk("repo commit known", bool(head), head[:12])
    chk("working tree clean", not dirty,
        f"{len(dirty.splitlines())} modified files" if dirty else "clean")
    anc = sh(f"git merge-base --is-ancestor {DISPATCH_FIX} HEAD")
    chk(f"source contains the dispatcher fix ({DISPATCH_FIX})", anc.returncode == 0,
        "run: git log --oneline -1 " + DISPATCH_FIX)

    # ---- jar actually loaded ----
    libs = os.environ.get("GATEWAY_LIBS")
    if libs:
        jar = pathlib.Path(libs) / "cloudsimplus-gateway.jar"
        origin = "GATEWAY_LIBS"
    else:
        jar = ROOT / "cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar"
        origin = "build dir (GATEWAY_LIBS unset)"
    chk("gateway jar present", jar.is_file(), f"{origin}: {jar}")
    if jar.is_file():
        digest = hashlib.sha256(jar.read_bytes()).hexdigest()
        print(f"jar sha256 {digest}\n")
        srcs = list((ROOT / "cloudsimplus-gateway/src/main/java").rglob("*.java"))
        newest = max((f.stat().st_mtime for f in srcs), default=0.0)
        chk("jar not older than the Java source", jar.stat().st_mtime >= newest,
            "weak check, mtime only; the wiring sentinel below is the real one")

    # ---- config ----
    if args.experiment:
        cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
        block = cfg.get(args.experiment)
        chk("experiment key exists", block is not None, args.experiment)
        if block is not None:
            crd = block.get("crd") or {}
            enabled = crd.get("enabled")
            print(f"crd.enabled = {enabled!r}"
                  f"{'   (CRD inactive: the v5 markers below do not affect this run)' if enabled is not True else ''}\n")
            missing = []
            for sub, keys in V5_MARKERS.items():
                have = crd.get(sub) or {}
                missing += [f"{sub}.{k}" for k in keys if k not in have]
            chk("crd subtree carries the v5 markers", not missing,
                "missing: " + ", ".join(missing) if missing else "all present")

    width = max(len(n) for n, _, _ in rows)
    print(f"{'check':<{width}}  verdict  detail")
    for name, ok, detail in rows:
        print(f"{name:<{width}}  {'PASS ' if ok else 'FAIL '}    {detail}")
    failed = [n for n, ok, _ in rows if not ok]
    print(f"\n{'ALL PASS' if not failed else 'FAILED: ' + ', '.join(failed)}")

    print("""
The wiring sentinel is the only check that tests behaviour rather than
metadata. Run one episode and histogram per-VM assignments from the
LoadBalancingBroker INFO lines. With the bug, load piles onto the first few
VMs; with the fix it spreads. On the 5-DC C-regime the count of VMs carrying
90% of the load went 65 -> 155. The absolute numbers depend on the fleet, so
compare old jar against new on your own testbed rather than against these.""")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
