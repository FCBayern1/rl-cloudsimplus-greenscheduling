"""Pre-submission preflight for the paper.

The red [UNRESOLVED] tags were turned off on 2026-08-24 so the draft reads
clean. They were the only thing standing between an ungraded claim and the
submitted version, so the guard moved here. A visible marker gets ignored once
a reader is used to it; a preflight has to be run on purpose and prints a
verdict.

Run before every submission:

    python drl-manager/check_submission_ready.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TEX = ROOT / "paper_latest/iclr2027_conference.tex"

rows = []


def chk(name, ok, detail=""):
    rows.append((name, bool(ok), detail))


def switch(src, name):
    """True if \\<name>true is the active setting."""
    m = re.findall(rf"\\{name}(true|false)", src)
    return m[-1] == "true" if m else None


def main():
    src = TEX.read_text()
    body = "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in src.split("\n"))

    # --- App F: an ungraded fragility claim must not render ---
    renders = switch(src, "FragileNone") is False and switch(src, "FragilePattern") is True
    graded = (ROOT / "reports/APPF_GRADING.md").is_file()
    chk("App F fragility claim is graded before it renders",
        graded or not renders,
        "renders but reports/APPF_GRADING.md is missing; grade per reports/APPF_PREREG.md"
        if renders and not graded else
        ("graded" if graded else "claim does not render"))

    # --- result macros still carry pre-campaign values ---
    warned = "PLACEHOLDERS" in src
    results = (ROOT / "reports/G1_RESULTS.md").is_file()
    chk("headline macros come from the G1 campaign", results,
        "reports/G1_RESULTS.md is missing, so \\ResContain and friends are still "
        "the pre-campaign (v4 build, pre-61043cf) placeholders" if not results else "recorded")
    chk("placeholder warning still present in the preamble", warned or results,
        "present" if warned else ("superseded by G1 results" if results
        else "removed without producing G1 results"))

    # --- sampling appendix ---
    stoch_placeholder = "Do NOT submit until\n% these are regenerated" in src or \
                        "scheduled for regeneration" in src
    stoch_done = (ROOT / "reports/G1_SAMPLING.md").is_file()
    chk("sampling appendix regenerated", stoch_done,
        "Appendix E still carries pre-campaign numbers" if not stoch_done else "recorded")

    # --- risk baselines under sampling ---
    risk_on = switch(src, "RiskSampling")
    risk_done = (ROOT / "reports/G1_RISK_SAMPLING.md").is_file()
    chk("RiskSampling only on when the risk baselines were re-evaluated",
        (not risk_on) or risk_done,
        "\\RiskSamplingtrue but no re-evaluation record" if risk_on and not risk_done else
        ("off" if not risk_on else "recorded"))

    # --- figure and equation consistency ---
    uses_drawio = "new_EUCRD_drawio_nature.png" in body
    fig_note = (ROOT / "reports/FIGURE1_FIXED.md").is_file()
    chk("Figure 1 agrees with Eq. 2",
        (not uses_drawio) or fig_note,
        "the drawio figure shows R_forecast without the C_max normalisation and "
        "without the absolute values that Eq. 2 now carries, and it has an empty "
        "'Key note' box" if uses_drawio and not fig_note else "ok")

    width = max(len(n) for n, _, _ in rows)
    print(f"{'gate':<{width}}  verdict  detail")
    for name, ok, detail in rows:
        print(f"{name:<{width}}  {'PASS ' if ok else 'BLOCK'}    {detail}")
    blocked = [n for n, ok, _ in rows if not ok]
    print()
    if blocked:
        print(f"NOT READY: {len(blocked)} gate(s) blocking")
        for n in blocked:
            print(f"  - {n}")
    else:
        print("READY")
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
