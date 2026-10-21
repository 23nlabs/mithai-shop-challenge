"""Addon to final_analysis.py: paired regret differences for the BIAS-corrected variant in the
calendar-dependence (R1-R3) and rolling families (the main run computed 'none' and 'bias+cal' there).
Same hierarchical bootstrap (shops + Yug-1 seeds). Pairs restricted to those involving Yug-1 + calendar
or the censoring-aware calendar regression."""
import json
import numpy as np
import final_analysis as fa
import mithai_core as mc
import robust_core as rc
import run_v2 as r2
import evaluate as ev
import eval_v2 as e2

PAIRS = [(a, b) for a in ('Yug-1 + calendar', 'Calendar regression (censoring-aware)') for b in fa.CAL_B if b != a]


def main():
    cal = mc.load_calendar(fa.STUDY / 'data/festivals.json')
    jv, Fv = fa.load_family('v2', cal)
    out = {}
    ridx = [i for i, j in enumerate(jv) if j['scenario'] in rc.ROBUST]
    jr = [jv[i] for i in ridx]
    sub = {m: [q[ridx] for q in qs] for m, qs in Fv.items() if m in fa.CAL_B}
    obs = fa.observed_lookup(cal, {fa.history_family(j) for j in jr})
    R = {}
    for m, qs in sub.items():
        cols = []
        for si, q in enumerate(qs):
            bk = fa.calibration_factors(jr, jr, q, obs, fa.tau_of(jr), m)
            cols.append(fa.committed_regret(fa.targets(m, q, jr, 'bias', bk), jr)[0])
        R[m] = np.stack(cols, axis=1)
    for s, mask in r2.strata_v2(jr).items():
        if not s.endswith('festival season'):
            continue
        g = [np.searchsorted(np.where(mask)[0], gi) for gi in fa.groups_of(jr, mask)]
        out[s] = {f'{a} − {b}': fa.hboot_diff(R[a][mask], R[b][mask], g) for a, b in PAIRS}
    print('R done', flush=True)
    (fa.V3 / 'results/final-v3-addon.json').write_text(json.dumps(dict(bias_pairwise_calendar_dependence=out), indent=1))


if __name__ == '__main__':
    main()
