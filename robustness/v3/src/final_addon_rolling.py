"""Addon: bias-variant paired regret differences under weekly rolling replanning (same method as final_analysis)."""
import json
import numpy as np
import final_analysis as fa
import final_addon as fad
import mithai_core as mc
import robust_core as rc
import evaluate as ev
import eval_v2 as e2

H = mc.HORIZON


def main():
    cal = mc.load_calendar(fa.STUDY / 'data/festivals.json')
    jo, Fo = fa.load_family('orig', cal)
    obs = fa.observed_lookup(cal, {fa.history_family(j) for j in jo})
    bi = [i for i, j in enumerate(jo) if j['scenario'] == 'base']
    fam_jobs = [jo[i] for i in bi]
    fam_Fs = {m: [q[bi] for q in qs] for m, qs in Fo.items()}
    jv, Fv = fa.load_family('v2', cal)
    bm = {(j['seed'], j['year'], j['cutoff'], j['product'], j['offset']): i for i, j in enumerate(jv) if j['scenario'] == 'base'}
    j0 = [jv[bm[(s, y, c, p, 0)]] for s in rc.ROLL_SEEDS for y in mc.YEARS for c in mc.FESTIVAL_CUTOFFS for p in mc.PRODUCTS]
    D = np.stack([j['truth'] for j in j0]); cap = np.stack([j['cap'] for j in j0]) * ev.CAP_FACTOR
    pr, co, sh, ish, tau = e2.econ_arrays(j0)
    ref = e2.rolling_inventory(np.repeat(D[None], len(rc.ROLL_OFFSETS), axis=0), D, cap, sh, ish, pr, co)
    R = {}
    for m in fa.CAL_B:
        cols = []
        for si in range(len(Fv[m])):
            bk = fa.calibration_factors(j0, fam_jobs, fam_Fs[m][si], obs, tau, m)
            Tk = np.zeros((len(rc.ROLL_OFFSETS), len(j0), H))
            for k_, off in enumerate(rc.ROLL_OFFSETS):
                ik = [bm[(j['seed'], j['year'], j['cutoff'], j['product'], off)] for j in j0]
                pt = fa.base_point(m, Fv[m][si][ik], tau, 'bias') * bk[0][:, None]
                plan = np.zeros_like(D); plan[:, off:] = pt[:, :H - off]
                Tk[k_] = np.maximum(plan, np.stack([jv[i]['confirmed'] for i in ik]))
            cols.append(ref['contribution'] - e2.rolling_inventory(Tk, D, cap, sh, ish, pr, co)['contribution'])
        R[m] = np.stack(cols, axis=1)
    g = fa.groups_of(j0, np.ones(len(j0), bool))
    out = {f'{a} − {b}': fa.hboot_diff(R[a], R[b], g) for a, b in fad.PAIRS}
    p = fa.V3 / 'results/final-v3-addon.json'
    d = json.loads(p.read_text()); d['bias_pairwise_rolling'] = out
    p.write_text(json.dumps(d, indent=1))
    for k, v in out.items():
        print(f"{k:80s} {v['point']:>8,.0f} [{v['lo']:,.0f}, {v['hi']:,.0f}] {v['verdict']}")


if __name__ == '__main__':
    main()
