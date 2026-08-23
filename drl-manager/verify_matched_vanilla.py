#!/usr/bin/env python3
"""P0-B verification: the matched Vanilla arm differs from knSV3b by EXACTLY
three keys and nothing else (Codex ruling 2026-08-23).

The previous Vanilla arm was an invalid comparator because it differed from
knSV3b in eight objective-level parameters (carbon_penalty_mode, carbon
normalisation, completion coefficients, gamma, entropy, per-action weights).
This check exists so that cannot recur silently.

Codex tightening, 2026-08-23:
  - the admissible difference set is an EXACT set, not a prefix allowlist: a
    prefix such as "crd." would let the whole CRD subtree diverge unnoticed,
    and "wandb." would hide identity drift;
  - no startswith() matching;
  - no str() coercion - a missing key, None, the string "1" and the integer 1
    must all compare unequal;
  - the observed difference set must EQUAL the admissible set, not merely be
    contained in it, so an accidentally identical crd.enabled also fails.

Run-manifest fields (gateway_log_dir, output_dir, WandB identity) are recorded
separately in the freeze manifest and are deliberately NOT semantic config, so
they must not appear in either arm's resolved config.

Exit 0 = matched. Exit 1 = anything else.
"""
import sys
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
CFG = REPO / "config_C.yml"
EU = "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b"
VAN = "experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan"

# EXACT set. Not prefixes.
ALLOWED_DIFFS = {
    "crd.enabled",
    "experiment_name",
    "simulation_name",
}

# Must not be baked into semantic config; they belong to the run manifest.
RUN_MANIFEST_KEYS = {"gateway_log_dir", "output_dir"}

_MISSING = object()


def flat(x, prefix=""):
    """Flatten to dotted keys, preserving value types. Lists become tuples so
    they hash and compare by value without stringification."""
    out = {}
    for k, v in (x or {}).items():
        key = prefix + k
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        elif isinstance(v, list):
            out[key] = tuple(v)
        else:
            out[key] = v
    return out


def typed_ne(a, b):
    """True when the two values differ, distinguishing missing / None / type.
    bool is checked before int because bool is an int subclass in Python."""
    if a is _MISSING or b is _MISSING:
        return True
    if (a is None) != (b is None):
        return True
    if isinstance(a, bool) != isinstance(b, bool):
        return True
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
        and not isinstance(a, bool) and not isinstance(b, bool)
    ):
        return True
    return a != b


def diff_keys(eu, van):
    return {k for k in set(eu) | set(van)
            if typed_ne(eu.get(k, _MISSING), van.get(k, _MISSING))}


def main():
    cfg = yaml.safe_load(open(CFG))
    for k in (EU, VAN):
        if k not in cfg:
            print(f"FAIL: config key missing: {k}")
            return 1
    eu, van = flat(cfg[EU]), flat(cfg[VAN])
    observed = diff_keys(eu, van)

    print(f"knSV3b keys={len(eu)}  matchedvan keys={len(van)}  differing={len(observed)}")
    for k in sorted(observed):
        print(f"  {k:44s} eucrd={eu.get(k, _MISSING)!r:28s} van={van.get(k, _MISSING)!r}")

    ok = True

    if observed != ALLOWED_DIFFS:
        extra = observed - ALLOWED_DIFFS
        missing = ALLOWED_DIFFS - observed
        print("\nFAIL: the observed difference set is not exactly the admissible set.")
        if extra:
            print(f"  unexpected differences: {sorted(extra)}")
        if missing:
            print(f"  expected differences that are absent: {sorted(missing)}")
        ok = False

    # the CRD switch must be a real boolean flip, not a truthy string
    if eu.get("crd.enabled", _MISSING) is not True:
        print("FAIL: knSV3b crd.enabled is not boolean True")
        ok = False
    if van.get("crd.enabled", _MISSING) is not False:
        print("FAIL: matchedvan crd.enabled is not boolean False")
        ok = False

    # the Vanilla arm keeps the full CRD subtree, inert
    eu_crd = {k for k in eu if k.startswith("crd.")}
    van_crd = {k for k in van if k.startswith("crd.")}
    if eu_crd != van_crd:
        print(f"FAIL: CRD subtree shape differs: only-eucrd={sorted(eu_crd - van_crd)} "
              f"only-van={sorted(van_crd - eu_crd)}")
        ok = False

    for k in sorted(RUN_MANIFEST_KEYS):
        for name, arm in (("knSV3b", eu), ("matchedvan", van)):
            if k in arm:
                print(f"FAIL: {name} bakes run-manifest key '{k}' into semantic config")
                ok = False

    print("\nPASS: the two arms differ by exactly the CRD switch and the two identity fields."
          if ok else "\nVERIFICATION FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
